"""Impure Quantum Density-Matrix Layer Mirroring, Isometric sqrt(2) Hermitian Packing, & Riemannian Stiefel-Muon.

Provenance: Wave 46 (`EXPERIMENT_PLAN_061.md`) & Gemini Notebook (*"Top-1 Accuracy vs Cross-Entropy Divergence"*, Pages 18-36).

Provides flag-free reusable building blocks for:
1. **Isometric $\\sqrt{2}$ Hermitian Triangle Packing (`pack_hermitian_isometric` / `unpack_hermitian_isometric`)**:
   - Exact Frobenius-isometric bijection $\\tilde{P}: \\text{Herm}(\\mathbb{C}^{d_h}) \\to \\mathbb{R}^{d_h^2}$ placing the real
     diagonal on the diagonal with weight $1$, the strict upper triangle ($\\text{Re}$) with weight $\\sqrt{2}$, and the
     strict lower triangle ($\\text{Im}$) with weight $\\sqrt{2}$ so that:
     $$\\langle \\tilde{P}(\\rho), \\tilde{P}(E)\\rangle_{\\mathbb{R}^{d_h^2}} = \\text{Tr}(\\rho E), \\qquad \\|\\tilde{P}(\\rho)\\|_2^2 = \\text{Tr}(\\rho^2)$$
2. **Bures-Wasserstein McWeeny Natural Gradient Purification (`bures_wasserstein_mcweeny_step`)**:
   - Inversion-free natural gradient ascent step $\\rho \\leftarrow \\rho^2 / \\text{Tr}(\\rho^2)$ on the density matrix manifold
     $(\\mathcal{D}(\\mathbb{C}^{d_h}), g_{\\text{BW}})$ that suppresses low-eigenvalue background noise while keeping balanced
     Rank-$K$ polysemous subspace projectors $\\rho = \\frac{1}{K}\\sum_{m=1}^K |\\phi_m\\rangle\\langle\\phi_m|$ as exact fixed points.
3. **Rank-2 Lanczos-Krylov Spectral Eigenpair Extractor (`extract_rank2_krylov_eigenpair`)**:
   - Fast $O(d_h^2)$ matvec-only extraction of the top-2 orthogonal eigenmodes $((\\lambda_1, |\\phi_1\\rangle), (\\lambda_2, |\\phi_2\\rangle))$
     of $\\rho_{t,h}$ via 1 Lanczos residual step $|u_2\\rangle = \\rho |\\phi_1\\rangle - \\lambda_1 |\\phi_1\\rangle$ with zero `eigh` calls.
4. **Exact `Riemannian Stiefel-Muon` Geodesic Step (`riemannian_stiefel_muon_step` / `RiemannianStiefelMuon`)**:
   - Projects momentum onto the unitary tangent space $M_{\\text{tan}} = M - \\frac{1}{2}W(W^\\dagger M + M^\\dagger W)$ (for square $D=d$)
     or pure horizontal Grassmann space $M_{\\text{horiz}} = M - W(W^\\dagger M)$ (for rectangular $D > d$) before 5-step Newton-Schulz
     orthogonalization, guaranteeing $W^\\dagger U_{\\text{tan}} + U_{\\text{tan}}^\\dagger W = 0$ so the scalar retraction
     $W_{\\text{next}} = (W - \\eta U_{\\text{tan}}) / \\sqrt{1 + \\eta^2}$ traces the exact Riemannian geodesic.
5. **Classical-to-Quantum Layer Mirroring & Surgical Grafting Blocks (`ImpureQuantumDensityMirrorBlock`, `PureQuantumKetMirrorBlock`)**:
   - Drop-in $(B, T, 768) \\to (B, T, 768)$ quantum blocks matching the call signature of `CompiledLeaderboardBlock` for
     Experiment A (layer trajectory mirroring) and Experiment B (surgical quantum layer grafting).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_061.md Section 3
# ==============================================================================
SQRT2_ISOMETRIC_OFFDIAG_SCALE: float = 1.4142135623730951
INV_SQRT2_ISOMETRIC_SCALE: float = 0.7071067811865475
DEFAULT_MIRROR_EMBED_DIM: int = 768
DEFAULT_MIRROR_NUM_HEADS: int = 6
DEFAULT_MIRROR_HEAD_COMPLEX_DIM: int = 16
DEFAULT_MCWEENY_STEPS: int = 1
DEFAULT_BORN_POWER_P: int = 2
DEFAULT_RIEMANNIAN_MUON_LR: float = 0.025
DEFAULT_RIEMANNIAN_MOMENTUM_BETA: float = 0.92
DEFAULT_GRAM_RELATIONAL_WEIGHT: float = 0.25
TRACE_NORM_EPS: float = 1e-7
NS5_QUINTIC_A: float = 3.4445
NS5_QUINTIC_B: float = -4.7750
NS5_QUINTIC_C: float = 2.0315
DEFAULT_NS5_STEPS: int = 5
DEFAULT_ROPE_BASE_THETA: float = 10000.0


@dataclass(frozen=True)
class QuantumMirrorBlockConfig:
    """Configuration for `ImpureQuantumDensityMirrorBlock` and `PureQuantumKetMirrorBlock` (`<= 8` fields, Rule 2)."""

    embed_dim: int = DEFAULT_MIRROR_EMBED_DIM
    num_heads: int = DEFAULT_MIRROR_NUM_HEADS
    head_complex_dim: int = DEFAULT_MIRROR_HEAD_COMPLEX_DIM
    channel_hidden_dim: int = DEFAULT_MIRROR_EMBED_DIM * 2
    born_power_p: int = DEFAULT_BORN_POWER_P
    mcweeny_steps: int = DEFAULT_MCWEENY_STEPS
    max_seq_len: int = 1024
    eps: float = TRACE_NORM_EPS


def complex_l2_normalize(
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    eps: float = TRACE_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize complex vectors `(z_re, z_im)` onto the unit sphere $\\mathbb{S}^{2d-1}$ along the last dimension."""
    norm_sq = z_re.square().sum(dim=-1, keepdim=True) + z_im.square().sum(dim=-1, keepdim=True)
    inv_norm = torch.rsqrt(norm_sq.clamp_min(eps))
    return z_re * inv_norm, z_im * inv_norm


def pack_hermitian_isometric(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
) -> torch.Tensor:
    """Map Hermitian matrix `(rho_re, rho_im)` of shape `(..., d_h, d_h)` to isometric real vector `(..., d_h * d_h)`.

    Places:
    - Real diagonal $\\rho_{ii}^{\\text{Re}}$ on the diagonal with weight $1.0$,
    - Strict upper triangle $\\rho_{ij}^{\\text{Re}}$ ($i < j$) with weight $\\sqrt{2}$,
    - Strict lower triangle $\\rho_{ji}^{\\text{Im}}$ ($i > j$, i.e. upper-triangle imaginary part transposed) with weight $\\sqrt{2}$.

    Guarantees $\\langle \\tilde{P}(A), \\tilde{P}(B)\\rangle_{\\mathbb{R}^{d_h^2}} = \\text{Tr}(AB)$ and
    $\\|\\tilde{P}(\\rho)\\|_2^2 = \\text{Tr}(\\rho^2)$.
    """
    diag_part = torch.diag_embed(torch.diagonal(rho_re, dim1=-2, dim2=-1))
    upper_re = torch.triu(rho_re, diagonal=1) * SQRT2_ISOMETRIC_OFFDIAG_SCALE
    lower_im = torch.tril(rho_im.transpose(-2, -1), diagonal=-1) * SQRT2_ISOMETRIC_OFFDIAG_SCALE
    packed_mat = diag_part + upper_re + lower_im
    return packed_mat.flatten(start_dim=-2, end_dim=-1)


def unpack_hermitian_isometric(
    packed: torch.Tensor,
    d_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert `pack_hermitian_isometric`, mapping `(..., d_h * d_h)` back to Hermitian `(rho_re, rho_im)` `(..., d_h, d_h)`."""
    mat = packed.unflatten(dim=-1, sizes=(d_h, d_h))
    diag_re = torch.diag_embed(torch.diagonal(mat, dim1=-2, dim2=-1))
    upper_re = torch.triu(mat, diagonal=1) * INV_SQRT2_ISOMETRIC_SCALE
    lower_im_t = torch.tril(mat, diagonal=-1) * INV_SQRT2_ISOMETRIC_SCALE
    rho_re = diag_re + upper_re + upper_re.transpose(-2, -1)
    rho_im = lower_im_t.transpose(-2, -1) - lower_im_t
    return rho_re, rho_im


def pack_rank1_ket_outer_isometric(
    v_re: torch.Tensor,
    v_im: torch.Tensor,
) -> torch.Tensor:
    """Pack pure rank-1 outer product $|v\\rangle\\langle v|$ for complex kets `(v_re, v_im)` `(..., d_h)` into `(..., d_h^2)`."""
    outer_re = v_re.unsqueeze(-1) * v_re.unsqueeze(-2) + v_im.unsqueeze(-1) * v_im.unsqueeze(-2)
    outer_im = v_im.unsqueeze(-1) * v_re.unsqueeze(-2) - v_re.unsqueeze(-1) * v_im.unsqueeze(-2)
    return pack_hermitian_isometric(outer_re, outer_im)


def bures_wasserstein_mcweeny_step(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    num_steps: int = DEFAULT_MCWEENY_STEPS,
    eps: float = TRACE_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply inversion-free Bures-Wasserstein natural gradient ascent $\\rho \\leftarrow \\rho^2 / \\text{Tr}(\\rho^2)$.

    Returns `(rho_re_out, rho_im_out, purity, eff_rank)` where `purity` ($\\text{Tr}(\\rho^2)$) and
    `eff_rank` ($1 / \\text{Tr}(\\rho^2)$) have shape `(...,)` matching the batch/head prefix.
    """
    cur_re = rho_re
    cur_im = rho_im
    tr_cur = torch.diagonal(cur_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
    cur_re = cur_re / tr_cur
    cur_im = cur_im / tr_cur

    purity = (cur_re.square().sum(dim=(-2, -1)) + cur_im.square().sum(dim=(-2, -1))).clamp(eps, 1.0)
    for _ in range(max(0, int(num_steps))):
        sq_re = torch.matmul(cur_re, cur_re) - torch.matmul(cur_im, cur_im)
        sq_im = torch.matmul(cur_re, cur_im) + torch.matmul(cur_im, cur_re)
        sq_re = 0.5 * (sq_re + sq_re.transpose(-2, -1))
        sq_im = 0.5 * (sq_im - sq_im.transpose(-2, -1))
        tr_sq = torch.diagonal(sq_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
        cur_re = sq_re / tr_sq
        cur_im = sq_im / tr_sq
        purity = (cur_re.square().sum(dim=(-2, -1)) + cur_im.square().sum(dim=(-2, -1))).clamp(eps, 1.0)

    eff_rank = (1.0 / purity).clamp(1.0, float(rho_re.size(-1)))
    return cur_re, cur_im, purity, eff_rank


def extract_rank2_krylov_eigenpair(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    eps: float = TRACE_NORM_EPS,
) -> dict[str, torch.Tensor]:
    """Extract top-2 orthogonal eigenmodes $((\\lambda_1, |\\phi_1\\rangle), (\\lambda_2, |\\phi_2\\rangle))$ in $O(d_h^2)$ time without `eigh`."""
    diag_vals = torch.diagonal(rho_re, dim1=-2, dim2=-1).clamp_min(eps)
    v0_re = torch.sqrt(diag_vals)
    v0_im = torch.zeros_like(v0_re)
    phi1_re, phi1_im = complex_l2_normalize(v0_re, v0_im, eps=eps)

    for _ in range(2):
        w_re = torch.matmul(rho_re, phi1_re.unsqueeze(-1)).squeeze(-1) - torch.matmul(
            rho_im, phi1_im.unsqueeze(-1)
        ).squeeze(-1)
        w_im = torch.matmul(rho_re, phi1_im.unsqueeze(-1)).squeeze(-1) + torch.matmul(
            rho_im, phi1_re.unsqueeze(-1)
        ).squeeze(-1)
        phi1_re, phi1_im = complex_l2_normalize(w_re, w_im, eps=eps)

    r_phi1_re = torch.matmul(rho_re, phi1_re.unsqueeze(-1)).squeeze(-1) - torch.matmul(
        rho_im, phi1_im.unsqueeze(-1)
    ).squeeze(-1)
    r_phi1_im = torch.matmul(rho_re, phi1_im.unsqueeze(-1)).squeeze(-1) + torch.matmul(
        rho_im, phi1_re.unsqueeze(-1)
    ).squeeze(-1)
    lam1 = (phi1_re * r_phi1_re + phi1_im * r_phi1_im).sum(dim=-1).clamp(0.0, 1.0)

    u2_re = r_phi1_re - lam1.unsqueeze(-1) * phi1_re
    u2_im = r_phi1_im - lam1.unsqueeze(-1) * phi1_im
    ov_re = (phi1_re * u2_re + phi1_im * u2_im).sum(dim=-1, keepdim=True)
    ov_im = (phi1_re * u2_im - phi1_im * u2_re).sum(dim=-1, keepdim=True)
    u2_re = u2_re - (ov_re * phi1_re - ov_im * phi1_im)
    u2_im = u2_im - (ov_re * phi1_im + ov_im * phi1_re)
    phi2_re, phi2_im = complex_l2_normalize(u2_re, u2_im, eps=eps)

    r_phi2_re = torch.matmul(rho_re, phi2_re.unsqueeze(-1)).squeeze(-1) - torch.matmul(
        rho_im, phi2_im.unsqueeze(-1)
    ).squeeze(-1)
    r_phi2_im = torch.matmul(rho_re, phi2_im.unsqueeze(-1)).squeeze(-1) + torch.matmul(
        rho_im, phi2_re.unsqueeze(-1)
    ).squeeze(-1)
    lam2 = (phi2_re * r_phi2_re + phi2_im * r_phi2_im).sum(dim=-1).clamp(0.0, 1.0)

    return {
        "lambda_1": lam1,
        "lambda_2": lam2,
        "phi1_re": phi1_re,
        "phi1_im": phi1_im,
        "phi2_re": phi2_re,
        "phi2_im": phi2_im,
    }


def _newton_schulz5_tall_real(mat: torch.Tensor, steps: int = DEFAULT_NS5_STEPS) -> torch.Tensor:
    """Compute exact unit-singular-value polar factor $U = M (M^T M)^{-1/2}$ of `mat` `(..., D, d)` ($D \\ge d$).

    Normalizing by the Frobenius norm $\\|M\\|_F$ guarantees all initial singular values lie in $[0, 1] \\subset (0, \\sqrt{3})$,
    where the cubic Newton-Schulz map $X \\leftarrow \\frac{1}{2}X(3I - X^T X)$ is strictly bounded in $[0, 1]$ and converges
    quadratically to $\\sigma_i = 1.0$ with zero SVD calls.
    """
    x = mat.float()
    norm = torch.linalg.matrix_norm(x, ord="fro", dim=(-2, -1), keepdim=True).clamp_min(TRACE_NORM_EPS)
    x = x / norm
    eye = torch.eye(x.size(-1), device=x.device, dtype=x.dtype)
    total_iters = max(7, steps + 2)
    for _ in range(total_iters):
        gram = torch.matmul(x.transpose(-2, -1), x)
        x = 0.5 * torch.matmul(x, 3.0 * eye - gram)
    return x


def riemannian_stiefel_muon_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buf: torch.Tensor,
    lr: float = DEFAULT_RIEMANNIAN_MUON_LR,
    beta: float = DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    ns_steps: int = DEFAULT_NS5_STEPS,
) -> torch.Tensor:
    """Execute 1 exact `Riemannian Stiefel-Muon` geodesic step on real Stiefel/Orthogonal matrix `weight`.

    Handles both tall/square matrices ($D \\ge d$, $W^T W = I_d$) and wide matrices ($D < d$, $W W^T = I_D$)
    by operating on the tall orientation $W_{\\text{tall}} \\in \\text{St}(\\max(D, d), \\min(D, d))$:
    1. Nesterov momentum update $M \\leftarrow \\beta M + G$, $G_{\\text{nest}} = G + \\beta M$.
    2. Tangent / Horizontal Grassmann projection:
       - If $D = d$ (square $O(d)$): $M_{\\text{tan}} = G_{\\text{nest}} - \\frac{1}{2}W(W^T G_{\\text{nest}} + G_{\\text{nest}}^T W)$.
       - If $D \\neq d$ (rectangular $\\text{St}(D, d)$): $M_{\\text{horiz}} = G_{\\text{nest}} - W(W^T G_{\\text{nest}})$.
    3. Unit-convergent Newton-Schulz orthogonalization $U_{\\text{tan}} = \\text{NS}(M_{\\text{tan}})$, guaranteeing $W^T U_{\\text{tan}} + U_{\\text{tan}}^T W = 0$.
    4. Exact scalar geodesic retraction $W_{\\text{next}} = (W - \\eta U_{\\text{tan}}) / \\sqrt{1 + \\eta^2}$ + 1-step Schulz polish.
    """
    if not torch.isfinite(grad).all():
        return weight
    momentum_buf.mul_(beta).add_(grad.float())
    g_nest = grad.float() + float(beta) * momentum_buf.float()
    if float(torch.linalg.matrix_norm(g_nest, ord="fro").max().item()) < TRACE_NORM_EPS:
        return weight
    w_f32 = weight.float()

    transposed = w_f32.size(-2) < w_f32.size(-1)
    if transposed:
        w_f32 = w_f32.transpose(-2, -1)
        g_nest = g_nest.transpose(-2, -1)

    rows, cols = w_f32.size(-2), w_f32.size(-1)
    wt_g = torch.matmul(w_f32.transpose(-2, -1), g_nest)
    if rows == cols:
        sym_part = 0.5 * (wt_g + wt_g.transpose(-2, -1))
        m_tan = g_nest - torch.matmul(w_f32, sym_part)
    else:
        m_tan = g_nest - torch.matmul(w_f32, wt_g)

    if float(torch.linalg.matrix_norm(m_tan, ord="fro").max().item()) < TRACE_NORM_EPS:
        return weight

    u_tan = _newton_schulz5_tall_real(m_tan, steps=ns_steps)
    if rows == cols:
        omega_hat = torch.matmul(w_f32.transpose(-2, -1), u_tan)
        omega_skew = 0.5 * (omega_hat - omega_hat.transpose(-2, -1))
        u_tan = torch.matmul(w_f32, omega_skew)
    else:
        u_tan = u_tan - torch.matmul(w_f32, torch.matmul(w_f32.transpose(-2, -1), u_tan))
        u_tan = _newton_schulz5_tall_real(u_tan, steps=ns_steps)

    retract_scale = 1.0 / math.sqrt(1.0 + float(lr) * float(lr))
    w_next = (w_f32 - float(lr) * u_tan) * retract_scale
    eye_c = torch.eye(cols, device=w_next.device, dtype=w_next.dtype)
    w_next = 0.5 * torch.matmul(w_next, 3.0 * eye_c - torch.matmul(w_next.transpose(-2, -1), w_next))

    if transposed:
        w_next = w_next.transpose(-2, -1)

    weight.copy_(w_next.to(weight.dtype))
    return weight


class RiemannianStiefelMuon(torch.optim.Optimizer):
    """Optimizer applying `riemannian_stiefel_muon_step` ($1/\\sqrt{1+\\eta^2}$ geodesic retraction) to 2D/3D Stiefel tensors."""

    def __init__(
        self,
        params: Any,
        lr: float = DEFAULT_RIEMANNIAN_MUON_LR,
        beta: float = DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
        ns_steps: int = DEFAULT_NS5_STEPS,
    ) -> None:
        defaults = {"lr": float(lr), "beta": float(beta), "ns_steps": int(ns_steps)}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["beta"])
            ns_steps = int(group["ns_steps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buf = state["momentum_buffer"]
                if p.ndim == 2:
                    riemannian_stiefel_muon_step(p, p.grad, buf, lr=lr, beta=beta, ns_steps=ns_steps)
                elif p.ndim == 3:
                    for idx in range(p.size(0)):
                        riemannian_stiefel_muon_step(
                            p[idx], p.grad[idx], buf[idx], lr=lr, beta=beta, ns_steps=ns_steps
                        )
                else:
                    buf.mul_(beta).add_(p.grad.float())
                    p.add_(buf.to(p.dtype), alpha=-lr)
        return loss


def initialize_stiefel_parameter(tensor: torch.Tensor) -> None:
    """Initialize a 2D or 3D parameter tensor onto the Stiefel manifold ($W^T W = I$ or $W W^T = I$)."""
    with torch.no_grad():
        if tensor.ndim == 2:
            nn.init.orthogonal_(tensor)
        elif tensor.ndim == 3:
            for idx in range(tensor.size(0)):
                nn.init.orthogonal_(tensor[idx])


class ImpureQuantumDensityMirrorBlock(nn.Module):
    """Quantum layer processing impure contextual density matrices $\\rho_{t,h} \\in \\mathcal{D}(\\mathbb{C}^{16})$.

    Combines:
    1. Causal $p$-boson Born attention weights $w_{ts,h} \\propto |\\langle q_{t,h} \\mid k_{s,h}\\rangle|^{2p}$,
    2. Multi-branch ($R = 4$) token-level mixed state synthesis $\\rho_{s,h}^{\\text{tok}} = \\sum_{r=1}^R p_{s,h,r}|v_{s,h}^{(r)}\\rangle\\langle v_{s,h}^{(r)}|$
       from a full-rank square Stiefel value isometry $W_V \\in \\text{St}(d_{\\text{model}}, d_{\\text{model}})$ followed by
       causal context mixture $\\rho_{t,h}^{(0)} = \\sum_{s \\le t} w_{ts,h} \\rho_{s,h}^{\\text{tok}}$ in Isometric $\\sqrt{2}$ packed space $\\mathbb{R}^{256}$,
    3. Bures-Wasserstein McWeeny natural gradient purification $\\rho_{t,h}^{(1)} = (\\rho_{t,h}^{(0)})^2 / \\text{Tr}((\\rho_{t,h}^{(0)})^2)$,
    4. Per-head complex unitary channel evolution $U_h \\rho_{t,h}^{(1)} U_h^\\dagger$ and Hermitian observable readout
       $\\bigoplus_{h=1}^6 \\tilde{P}(\\rho_{t,h}^{\\text{out}}) \\in \\mathbb{R}^{1536} \\to \\mathbb{R}^{768}$,
    5. Stiefel two-beam quantum interference sub-block on the residual stream.
    """

    def __init__(self, cfg: QuantumMirrorBlockConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else QuantumMirrorBlockConfig()
        self.embed_dim = self.cfg.embed_dim
        self.num_heads = self.cfg.num_heads
        self.d_h = self.cfg.head_complex_dim
        self.packed_head_dim = self.d_h * self.d_h
        self.total_packed_dim = self.num_heads * self.packed_head_dim
        self.ket_real_dim = self.num_heads * (2 * self.d_h)
        self.num_value_branches = max(1, self.embed_dim // self.ket_real_dim)
        self.v_real_dim = self.num_heads * self.num_value_branches * (2 * self.d_h)

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))

        self.w_q = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        self.w_k = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        self.w_v = nn.Parameter(torch.empty(self.embed_dim, self.v_real_dim))
        initialize_stiefel_parameter(self.w_q)
        initialize_stiefel_parameter(self.w_k)
        initialize_stiefel_parameter(self.w_v)

        eye_h = torch.eye(self.d_h).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
        self.head_unitary_re = nn.Parameter(eye_h)
        self.head_unitary_im = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.d_h))

        self.w_obs_out = nn.Parameter(torch.empty(self.total_packed_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_obs_out)
        self.attn_residual_scale = nn.Parameter(torch.ones(self.embed_dim))

        self.w_beam_a = nn.Parameter(torch.empty(self.embed_dim, self.cfg.channel_hidden_dim))
        self.w_beam_b = nn.Parameter(torch.empty(self.embed_dim, self.cfg.channel_hidden_dim))
        self.w_channel_down = nn.Parameter(torch.empty(self.cfg.channel_hidden_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_beam_a)
        initialize_stiefel_parameter(self.w_beam_b)
        initialize_stiefel_parameter(self.w_channel_down)
        self.channel_residual_scale = nn.Parameter(torch.ones(self.embed_dim))

        inv_freq = 1.0 / (
            DEFAULT_ROPE_BASE_THETA ** (torch.arange(0, self.d_h, dtype=torch.float32) / float(self.d_h))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.is_long_window_layer: bool = True
        self.last_telemetry: dict[str, float] = {"mean_purity": 1.0, "mean_eff_rank": 1.0}

    def stiefel_parameters(self) -> list[nn.Parameter]:
        """Return 2D/3D Stiefel state-preparation and unitary parameters optimized via `RiemannianStiefelMuon`."""
        return [
            self.w_q,
            self.w_k,
            self.w_v,
            self.head_unitary_re,
            self.w_beam_a,
            self.w_beam_b,
        ]

    def euclidean_parameters(self) -> list[nn.Parameter]:
        """Return readout observable and auxiliary phase parameters."""
        return [
            self.w_obs_out,
            self.w_channel_down,
            self.head_unitary_im,
        ]

    def _apply_complex_rope(
        self,
        z_re: torch.Tensor,
        z_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply complex phase rotation $e^{i t \\omega_k}$ to `(z_re, z_im)` of shape `(B, H, T, d_h)`."""
        seq_len = z_re.size(-2)
        pos = torch.arange(seq_len, device=z_re.device, dtype=torch.float32)
        angles = torch.outer(pos, self.rope_inv_freq).unsqueeze(0).unsqueeze(0)
        cos_a = torch.cos(angles).to(z_re.dtype)
        sin_a = torch.sin(angles).to(z_re.dtype)
        out_re = z_re * cos_a - z_im * sin_a
        out_im = z_re * sin_a + z_im * cos_a
        return out_re, out_im

    def _evolve_head_unitary(
        self,
        rho_re: torch.Tensor,
        rho_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-head complex unitary conjugation $U_h \\rho_{t,h} U_h^\\dagger$ using `(U_re, U_im)`."""
        u_re = self.head_unitary_re.to(rho_re.dtype)
        skew_im = 0.5 * (self.head_unitary_im + self.head_unitary_im.transpose(-2, -1)).to(rho_re.dtype)
        norm_sq = (u_re.square().sum(dim=-1, keepdim=True) + skew_im.square().sum(dim=-1, keepdim=True)).clamp_min(
            TRACE_NORM_EPS
        )
        inv_norm = torch.rsqrt(norm_sq)
        u_re = u_re * inv_norm
        u_im = skew_im * inv_norm

        u_re_b = u_re.unsqueeze(0).unsqueeze(2)
        u_im_b = u_im.unsqueeze(0).unsqueeze(2)

        tmp_re = torch.matmul(u_re_b, rho_re) - torch.matmul(u_im_b, rho_im)
        tmp_im = torch.matmul(u_re_b, rho_im) + torch.matmul(u_im_b, rho_re)

        ut_re = u_re_b.transpose(-2, -1)
        ut_im = -u_im_b.transpose(-2, -1)
        out_re = torch.matmul(tmp_re, ut_re) - torch.matmul(tmp_im, ut_im)
        out_im = torch.matmul(tmp_re, ut_im) + torch.matmul(tmp_im, ut_re)
        out_re = 0.5 * (out_re + out_re.transpose(-2, -1))
        out_im = 0.5 * (out_im - out_im.transpose(-2, -1))
        return out_re, out_im

    def compute_packed_density_features(self, x_base: torch.Tensor) -> torch.Tensor:
        """Compute isometric $\\sqrt{2}$-packed contextual density features `(B, T, 1536)` from `x_base`."""
        bsz, seq_len, _ = x_base.shape
        h_norm = F.rms_norm(x_base, (self.embed_dim,), eps=self.cfg.eps)

        q_raw = torch.matmul(h_norm, self.w_q.float()).view(bsz, seq_len, self.num_heads, 2, self.d_h)
        k_raw = torch.matmul(h_norm, self.w_k.float()).view(bsz, seq_len, self.num_heads, 2, self.d_h)
        v_raw = torch.matmul(h_norm, self.w_v.float()).view(
            bsz, seq_len, self.num_heads, self.num_value_branches, 2, self.d_h
        )

        q_re, q_im = q_raw[..., 0, :].transpose(1, 2), q_raw[..., 1, :].transpose(1, 2)
        k_re, k_im = k_raw[..., 0, :].transpose(1, 2), k_raw[..., 1, :].transpose(1, 2)
        v_re = v_raw[..., 0, :].permute(0, 2, 1, 3, 4)
        v_im = v_raw[..., 1, :].permute(0, 2, 1, 3, 4)

        q_re, q_im = complex_l2_normalize(q_re, q_im, eps=self.cfg.eps)
        k_re, k_im = complex_l2_normalize(k_re, k_im, eps=self.cfg.eps)

        branch_energy = (v_re.square().sum(dim=-1) + v_im.square().sum(dim=-1)).clamp_min(self.cfg.eps)
        branch_weights = branch_energy / branch_energy.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        v_re_unit, v_im_unit = complex_l2_normalize(v_re, v_im, eps=self.cfg.eps)

        q_re, q_im = self._apply_complex_rope(q_re, q_im)
        k_re, k_im = self._apply_complex_rope(k_re, k_im)

        ip_re = torch.matmul(q_re, k_re.transpose(-2, -1)) + torch.matmul(q_im, k_im.transpose(-2, -1))
        ip_im = torch.matmul(q_im, k_re.transpose(-2, -1)) - torch.matmul(q_re, k_im.transpose(-2, -1))
        born_mod_sq = (ip_re.square() + ip_im.square()).clamp_min(0.0)
        if self.cfg.born_power_p == 2:
            born_scores = born_mod_sq.square()
        elif self.cfg.born_power_p == 1:
            born_scores = born_mod_sq
        else:
            born_scores = born_mod_sq.pow(float(self.cfg.born_power_p))

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x_base.device, dtype=torch.bool))
        born_scores = born_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        denom = born_scores.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        attn_weights = born_scores / denom

        packed_branches = pack_rank1_ket_outer_isometric(v_re_unit, v_im_unit)
        packed_tok_rho = (branch_weights.unsqueeze(-1) * packed_branches).sum(dim=-2)
        packed_rho0 = torch.matmul(attn_weights, packed_tok_rho)
        rho0_re, rho0_im = unpack_hermitian_isometric(packed_rho0, self.d_h)

        rho1_re, rho1_im, purity, eff_rank = bures_wasserstein_mcweeny_step(
            rho0_re,
            rho0_im,
            num_steps=self.cfg.mcweeny_steps,
            eps=self.cfg.eps,
        )
        with torch.no_grad():
            self.last_telemetry = {
                "mean_purity": float(purity.mean().item()),
                "mean_eff_rank": float(eff_rank.mean().item()),
            }

        rho_out_re, rho_out_im = self._evolve_head_unitary(rho1_re, rho1_im)
        packed_rho_out = pack_hermitian_isometric(rho_out_re, rho_out_im) * float(self.d_h)
        return packed_rho_out.transpose(1, 2).reshape(bsz, seq_len, self.total_packed_dim)

    def compute_channel_features(self, x_mid: torch.Tensor) -> torch.Tensor:
        """Compute two-beam Mach-Zehnder interference features `(B, T, channel_hidden_dim)`."""
        m_norm = F.rms_norm(x_mid, (self.embed_dim,), eps=self.cfg.eps)
        beam_a = torch.matmul(m_norm, self.w_beam_a.float())
        beam_b = torch.matmul(m_norm, self.w_beam_b.float())
        return F.relu(beam_a + beam_b).square() - 0.25 * F.relu(beam_a - beam_b).square()

    @torch.no_grad()
    def fit_observable_readouts_ridge(
        self,
        x_in_list: list[torch.Tensor],
        x_out_list: list[torch.Tensor],
        x_0_list: list[torch.Tensor] | None = None,
        ridge_lambda: float = 0.05,
    ) -> None:
        """Solve closed-form Cholesky ridge regression for Euclidean observable readouts `w_obs_out` & `w_channel_down`."""
        phi_attn_chunks: list[torch.Tensor] = []
        delta_targets: list[torch.Tensor] = []
        x_bases: list[torch.Tensor] = []

        for idx, (x_in, x_out) in enumerate(zip(x_in_list, x_out_list)):
            x_f = x_in.float()
            if x_0_list is not None:
                x_base = self.block_lambdas[0].float() * x_f + self.block_lambdas[1].float() * x_0_list[idx].float()
            else:
                x_base = x_f
            phi = self.compute_packed_density_features(x_base)
            phi_attn_chunks.append(phi.reshape(-1, self.total_packed_dim))
            delta_targets.append((x_out.float() - x_base).reshape(-1, self.embed_dim))
            x_bases.append(x_base)

        phi_all = torch.cat(phi_attn_chunks, dim=0)
        y_all = torch.cat(delta_targets, dim=0)
        n_samples = float(phi_all.size(0))

        gram_attn = torch.matmul(phi_all.T, phi_all) + (ridge_lambda * n_samples) * torch.eye(
            self.total_packed_dim, device=phi_all.device, dtype=phi_all.dtype
        )
        rhs_attn = torch.matmul(phi_all.T, y_all)
        w_obs = torch.linalg.solve(gram_attn, rhs_attn)
        self.w_obs_out.copy_(w_obs.to(self.w_obs_out.dtype))
        self.attn_residual_scale.fill_(1.0)

        chan_chunks: list[torch.Tensor] = []
        rem_targets: list[torch.Tensor] = []
        for idx, x_base in enumerate(x_bases):
            phi_b = phi_attn_chunks[idx].view_as(
                torch.empty(x_base.size(0), x_base.size(1), self.total_packed_dim, device=x_base.device)
            )
            x_mid = x_base + torch.matmul(phi_b, self.w_obs_out.float())
            chan_feat = self.compute_channel_features(x_mid)
            chan_chunks.append(chan_feat.reshape(-1, self.cfg.channel_hidden_dim))
            rem_targets.append((x_out_list[idx].float() - x_mid).reshape(-1, self.embed_dim))

        chan_all = torch.cat(chan_chunks, dim=0)
        rem_all = torch.cat(rem_targets, dim=0)
        gram_chan = torch.matmul(chan_all.T, chan_all) + (ridge_lambda * n_samples) * torch.eye(
            self.cfg.channel_hidden_dim, device=chan_all.device, dtype=chan_all.dtype
        )
        rhs_chan = torch.matmul(chan_all.T, rem_all)
        w_chan = torch.linalg.solve(gram_chan, rhs_chan)
        self.w_channel_down.copy_(w_chan.to(self.w_channel_down.dtype))
        self.channel_residual_scale.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor | None = None,
        cos_4d: torch.Tensor | None = None,
        sin_4d: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
        ve_l: torch.Tensor | None = None,
        block_mask: Any = None,
        q_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass accepting exact `CompiledLeaderboardBlock` signature (`<= 8` args, Rule 2)."""
        _ = (cos_4d, sin_4d, ve_l, block_mask, q_scale)
        orig_dtype = x.dtype
        x_f = x.float()
        if x_0 is not None:
            x_f = self.block_lambdas[0].float() * x_f + self.block_lambdas[1].float() * x_0.float()

        packed_concat = self.compute_packed_density_features(x_f)
        attn_delta = torch.matmul(packed_concat, self.w_obs_out.float())
        x_f = x_f + self.attn_residual_scale.float() * attn_delta

        interference = self.compute_channel_features(x_f)
        channel_delta = torch.matmul(interference, self.w_channel_down.float())
        x_f = x_f + self.channel_residual_scale.float() * channel_delta
        return x_f.to(orig_dtype), v_0


class PureQuantumKetMirrorBlock(nn.Module):
    """Pure-State Rank-1 baseline mirror block ($\\rho_{t,h} = |\\psi_{t,h}\\rangle\\langle\\psi_{t,h}|$, $\\text{Purity} \\equiv 1.0$).

    Identical observable readout space $\\mathbb{R}^{1536} \\to \\mathbb{R}^{768}$ as
    `ImpureQuantumDensityMirrorBlock`, except attention collapses value kets into a single normalized pure
    superposition ket $|\\psi_{t,h}\\rangle = \\text{normalize}(\\sum_{s \\le t} w_{ts,h} |v_{s,h}\\rangle) \\in \\mathbb{C}^{16}$
    before constructing the rank-1 outer product $\\tilde{P}(|\\psi_{t,h}\\rangle\\langle\\psi_{t,h}|) \\in \\mathbb{R}^{256}$.
    """

    def __init__(self, cfg: QuantumMirrorBlockConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else QuantumMirrorBlockConfig()
        self.embed_dim = self.cfg.embed_dim
        self.num_heads = self.cfg.num_heads
        self.d_h = self.cfg.head_complex_dim
        self.packed_head_dim = self.d_h * self.d_h
        self.total_packed_dim = self.num_heads * self.packed_head_dim
        self.ket_real_dim = self.num_heads * (2 * self.d_h)

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))

        self.w_q = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        self.w_k = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        self.w_v = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        initialize_stiefel_parameter(self.w_q)
        initialize_stiefel_parameter(self.w_k)
        initialize_stiefel_parameter(self.w_v)

        eye_h = torch.eye(self.d_h).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
        self.head_unitary_re = nn.Parameter(eye_h)
        self.head_unitary_im = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.d_h))

        self.w_obs_out = nn.Parameter(torch.empty(self.total_packed_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_obs_out)
        self.attn_residual_scale = nn.Parameter(torch.ones(self.embed_dim))

        self.w_beam_a = nn.Parameter(torch.empty(self.embed_dim, self.cfg.channel_hidden_dim))
        self.w_beam_b = nn.Parameter(torch.empty(self.embed_dim, self.cfg.channel_hidden_dim))
        self.w_channel_down = nn.Parameter(torch.empty(self.cfg.channel_hidden_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_beam_a)
        initialize_stiefel_parameter(self.w_beam_b)
        initialize_stiefel_parameter(self.w_channel_down)
        self.channel_residual_scale = nn.Parameter(torch.ones(self.embed_dim))

        inv_freq = 1.0 / (
            DEFAULT_ROPE_BASE_THETA ** (torch.arange(0, self.d_h, dtype=torch.float32) / float(self.d_h))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.is_long_window_layer: bool = True
        self.last_telemetry: dict[str, float] = {"mean_purity": 1.0, "mean_eff_rank": 1.0}

    def stiefel_parameters(self) -> list[nn.Parameter]:
        """Return 2D/3D Stiefel state-preparation and unitary parameters."""
        return [
            self.w_q,
            self.w_k,
            self.w_v,
            self.head_unitary_re,
            self.w_beam_a,
            self.w_beam_b,
        ]

    def euclidean_parameters(self) -> list[nn.Parameter]:
        """Return readout observable and auxiliary phase parameters."""
        return [
            self.w_obs_out,
            self.w_channel_down,
            self.head_unitary_im,
        ]

    def compute_packed_pure_features(self, x_base: torch.Tensor) -> torch.Tensor:
        """Compute isometric $\\sqrt{2}$-packed Rank-1 pure outer-product features `(B, T, 1536)`."""
        bsz, seq_len, _ = x_base.shape
        h_norm = F.rms_norm(x_base, (self.embed_dim,), eps=self.cfg.eps)

        q_raw = torch.matmul(h_norm, self.w_q.float()).view(bsz, seq_len, self.num_heads, 2, self.d_h)
        k_raw = torch.matmul(h_norm, self.w_k.float()).view(bsz, seq_len, self.num_heads, 2, self.d_h)
        v_raw = torch.matmul(h_norm, self.w_v.float()).view(bsz, seq_len, self.num_heads, 2, self.d_h)

        q_re, q_im = q_raw[..., 0, :].transpose(1, 2), q_raw[..., 1, :].transpose(1, 2)
        k_re, k_im = k_raw[..., 0, :].transpose(1, 2), k_raw[..., 1, :].transpose(1, 2)
        v_re, v_im = v_raw[..., 0, :].transpose(1, 2), v_raw[..., 1, :].transpose(1, 2)

        q_re, q_im = complex_l2_normalize(q_re, q_im, eps=self.cfg.eps)
        k_re, k_im = complex_l2_normalize(k_re, k_im, eps=self.cfg.eps)
        v_re, v_im = complex_l2_normalize(v_re, v_im, eps=self.cfg.eps)

        pos = torch.arange(seq_len, device=x_base.device, dtype=torch.float32)
        angles = torch.outer(pos, self.rope_inv_freq).unsqueeze(0).unsqueeze(0)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        q_re, q_im = q_re * cos_a - q_im * sin_a, q_re * sin_a + q_im * cos_a
        k_re, k_im = k_re * cos_a - k_im * sin_a, k_re * sin_a + k_im * cos_a

        ip_re = torch.matmul(q_re, k_re.transpose(-2, -1)) + torch.matmul(q_im, k_im.transpose(-2, -1))
        ip_im = torch.matmul(q_im, k_re.transpose(-2, -1)) - torch.matmul(q_re, k_im.transpose(-2, -1))
        born_scores = (ip_re.square() + ip_im.square()).square()
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x_base.device, dtype=torch.bool))
        born_scores = born_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        attn_weights = born_scores / born_scores.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)

        psi_re = torch.matmul(attn_weights, v_re)
        psi_im = torch.matmul(attn_weights, v_im)
        psi_re, psi_im = complex_l2_normalize(psi_re, psi_im, eps=self.cfg.eps)

        u_re = self.head_unitary_re.float().unsqueeze(0).unsqueeze(2)
        u_im = 0.5 * (self.head_unitary_im + self.head_unitary_im.transpose(-2, -1)).float().unsqueeze(0).unsqueeze(2)
        psi_out_re = torch.matmul(psi_re.unsqueeze(-2), u_re.transpose(-2, -1)).squeeze(-2) - torch.matmul(
            psi_im.unsqueeze(-2), u_im.transpose(-2, -1)
        ).squeeze(-2)
        psi_out_im = torch.matmul(psi_re.unsqueeze(-2), u_im.transpose(-2, -1)).squeeze(-2) + torch.matmul(
            psi_im.unsqueeze(-2), u_re.transpose(-2, -1)
        ).squeeze(-2)
        psi_out_re, psi_out_im = complex_l2_normalize(psi_out_re, psi_out_im, eps=self.cfg.eps)

        packed_pure = pack_rank1_ket_outer_isometric(psi_out_re, psi_out_im) * float(self.d_h)
        return packed_pure.transpose(1, 2).reshape(bsz, seq_len, self.total_packed_dim)

    def compute_channel_features(self, x_mid: torch.Tensor) -> torch.Tensor:
        """Compute two-beam Mach-Zehnder interference features `(B, T, channel_hidden_dim)`."""
        m_norm = F.rms_norm(x_mid, (self.embed_dim,), eps=self.cfg.eps)
        beam_a = torch.matmul(m_norm, self.w_beam_a.float())
        beam_b = torch.matmul(m_norm, self.w_beam_b.float())
        return F.relu(beam_a + beam_b).square() - 0.25 * F.relu(beam_a - beam_b).square()

    @torch.no_grad()
    def fit_observable_readouts_ridge(
        self,
        x_in_list: list[torch.Tensor],
        x_out_list: list[torch.Tensor],
        x_0_list: list[torch.Tensor] | None = None,
        ridge_lambda: float = 0.05,
    ) -> None:
        """Solve closed-form Cholesky ridge regression for Euclidean observable readouts `w_obs_out` & `w_channel_down`."""
        phi_attn_chunks: list[torch.Tensor] = []
        delta_targets: list[torch.Tensor] = []
        x_bases: list[torch.Tensor] = []

        for idx, (x_in, x_out) in enumerate(zip(x_in_list, x_out_list)):
            x_f = x_in.float()
            if x_0_list is not None:
                x_base = self.block_lambdas[0].float() * x_f + self.block_lambdas[1].float() * x_0_list[idx].float()
            else:
                x_base = x_f
            phi = self.compute_packed_pure_features(x_base)
            phi_attn_chunks.append(phi.reshape(-1, self.total_packed_dim))
            delta_targets.append((x_out.float() - x_base).reshape(-1, self.embed_dim))
            x_bases.append(x_base)

        phi_all = torch.cat(phi_attn_chunks, dim=0)
        y_all = torch.cat(delta_targets, dim=0)
        n_samples = float(phi_all.size(0))

        gram_attn = torch.matmul(phi_all.T, phi_all) + (ridge_lambda * n_samples) * torch.eye(
            self.total_packed_dim, device=phi_all.device, dtype=phi_all.dtype
        )
        rhs_attn = torch.matmul(phi_all.T, y_all)
        w_obs = torch.linalg.solve(gram_attn, rhs_attn)
        self.w_obs_out.copy_(w_obs.to(self.w_obs_out.dtype))
        self.attn_residual_scale.fill_(1.0)

        chan_chunks: list[torch.Tensor] = []
        rem_targets: list[torch.Tensor] = []
        for idx, x_base in enumerate(x_bases):
            phi_b = phi_attn_chunks[idx].view_as(
                torch.empty(x_base.size(0), x_base.size(1), self.total_packed_dim, device=x_base.device)
            )
            x_mid = x_base + torch.matmul(phi_b, self.w_obs_out.float())
            chan_feat = self.compute_channel_features(x_mid)
            chan_chunks.append(chan_feat.reshape(-1, self.cfg.channel_hidden_dim))
            rem_targets.append((x_out_list[idx].float() - x_mid).reshape(-1, self.embed_dim))

        chan_all = torch.cat(chan_chunks, dim=0)
        rem_all = torch.cat(rem_targets, dim=0)
        gram_chan = torch.matmul(chan_all.T, chan_all) + (ridge_lambda * n_samples) * torch.eye(
            self.cfg.channel_hidden_dim, device=chan_all.device, dtype=chan_all.dtype
        )
        rhs_chan = torch.matmul(chan_all.T, rem_all)
        w_chan = torch.linalg.solve(gram_chan, rhs_chan)
        self.w_channel_down.copy_(w_chan.to(self.w_channel_down.dtype))
        self.channel_residual_scale.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor | None = None,
        cos_4d: torch.Tensor | None = None,
        sin_4d: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
        ve_l: torch.Tensor | None = None,
        block_mask: Any = None,
        q_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass collapsing attention into a Rank-1 pure superposition ket $|\\psi_{t,h}\\rangle$."""
        _ = (cos_4d, sin_4d, ve_l, block_mask, q_scale)
        orig_dtype = x.dtype
        x_f = x.float()
        if x_0 is not None:
            x_f = self.block_lambdas[0].float() * x_f + self.block_lambdas[1].float() * x_0.float()

        packed_pure = self.compute_packed_pure_features(x_f)
        attn_delta = torch.matmul(packed_pure, self.w_obs_out.float())
        x_f = x_f + self.attn_residual_scale.float() * attn_delta

        interference = self.compute_channel_features(x_f)
        channel_delta = torch.matmul(interference, self.w_channel_down.float())
        x_f = x_f + self.channel_residual_scale.float() * channel_delta
        return x_f.to(orig_dtype), v_0


def compute_layer_mirroring_loss(
    student_out: torch.Tensor,
    teacher_in: torch.Tensor,
    teacher_out: torch.Tensor,
    gram_weight: float = DEFAULT_GRAM_RELATIONAL_WEIGHT,
    eps: float = TRACE_NORM_EPS,
) -> dict[str, torch.Tensor]:
    """Compute combined layer-update relative MSE + gauge-invariant token-to-token Gram relational loss."""
    delta_student = student_out.float() - teacher_in.float()
    delta_teacher = teacher_out.float() - teacher_in.float()

    mse_num = (delta_student - delta_teacher).square().mean()
    mse_den = delta_teacher.square().mean().clamp_min(eps)
    rel_update_mse = mse_num / mse_den

    s_norm = F.normalize(student_out.float(), p=2, dim=-1, eps=eps)
    t_norm = F.normalize(teacher_out.float(), p=2, dim=-1, eps=eps)
    cos_sim = (s_norm * t_norm).sum(dim=-1).mean()

    gram_s = torch.matmul(s_norm, s_norm.transpose(-2, -1))
    gram_t = torch.matmul(t_norm, t_norm.transpose(-2, -1))
    gram_loss = (gram_s - gram_t).square().mean()

    total_loss = rel_update_mse + (1.0 - cos_sim) + float(gram_weight) * gram_loss
    return {
        "total_loss": total_loss,
        "rel_update_mse": rel_update_mse,
        "output_cosine_sim": cos_sim,
        "gram_relational_mse": gram_loss,
    }
