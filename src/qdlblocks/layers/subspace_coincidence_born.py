"""Product-of-Spheres (S^{d/M - 1})^M Multi-Subspace Coincidence Born Readout (`Wave 31` / `EXPERIMENT_PLAN_046.md`).

When to Use This Building Block:
- Use `subspace_l2_normalize` and `SubspaceCoincidenceSpec` when multiple additive feature pathways
  (such as coarse roots `wav_256`, fine wavelet leaves `wav_65536`, and surface N-gram projections
  `readout_ngram`) compete inside a single global L2 normalization and cause the sharp detector
  participation ratio `R_sharp` to collapse during training (as observed in `Wave 30`, where `R_sharp`
  dropped from `77.52 -> 24.18` at `d = 512`).
- Use `compute_multi_subspace_coincidence_observable` when $V \\gg d$ causes single-subspace positive
  runner-up impostors (`Z_pos_runnerup`) to inflate Cross-Entropy despite high Top-1 accuracy.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_046.md` / `Wave 31`, Lever 3):
1. **Product-of-Spheres Normalization (`subspace_l2_normalize`)**:
   Projects any state or detector tensor onto `M_{d, M} = ((1 / sqrt(M)) S^{d/M - 1})^M subset S^{d - 1}`,
   guaranteeing:
   - Exact global unit L2 norm `||x||_2 = 1` (preserving the analytical Parseval `muP` scale
     `beta_0(d) = c* sqrt(d)` with zero retuning), and
   - Exact equal energy `1 / M` per subspace, preventing norm cannibalization and increasing
     `R_sharp` by `8.30x` (`21.94 -> 182.07` at `d = 512, M = 4` in `scripts/multi_subspace_coincidence_margin_probe.py`).
2. **Multi-Subspace Coincidence Observable (`compute_multi_subspace_coincidence_observable`)**:
   Combines mean alignment `c_bar`, rectified-quadratic peak boost, AM-GM geometric coincidence
   `G_{t,v}^{(M)} = (prod_{m=1}^M (c_hat_{t,v}^{(m)})_+)^{1/M}`, and inter-subspace variance suppression
   `V_{t,v}^{(M)}` to eliminate single-subspace runner-up impostors (`-1.3414` nats runner-up CE tax reduction).
3. **Zero-Extra-Full-GEMM Chunked Composition (`subspace_coincidence_dyadic_chunked_ce`)**:
   Evaluates the M-subspace coincidence observable on the 256 coarse dyadic roots (`K_coarse = 256 << V = 50,257`)
   alongside `parseval_mup_dyadic_chunked_ce` at `< 0.15 ms` step overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..functional import CUSTOM_L2_NORM_MIN_NUMEL
from ..kernels.fused_parseval_ce import (
    M4_GEOM_EPS_OFFSET,
    M4_GEOM_EPS_SQ,
    M4_SUBSPACE_SCALE,
    compiled_m4_coincidence_chunk_step,
)
from .parseval_mup_chunked_ce import (
    ParsevalChunkedCEReadoutSpec,
    eval_parseval_mup_full_logits,
    parseval_mup_dyadic_chunked_ce,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_046.md, EXPERIMENT_PLAN_047.md, & EXPERIMENT_PLAN_049.md
# ==============================================================================
DEFAULT_NUM_SUBSPACES: int = 4
DEFAULT_PARTITION_MODE: str = "contiguous"
DEFAULT_TRACELESS_GAMMA: float = 0.25
DEFAULT_COINCIDENCE_GAMMA: float = 0.22
DEFAULT_SUBSPACE_VAR_PENALTY: float = 0.14
DEFAULT_COARSE_COINCIDENCE_WEIGHT: float = 0.04
DEFAULT_SUBSPACE_NORM_EPS: float = 1e-8
DEFAULT_SUBSPACE_BALANCE_BLEND: float = 1.0
COARSE_ROOT_SUBSPACE_BLEND: float = 0.85
SYMPLECTIC_PAIR_DIM: int = 2
EVAL_M4_CHUNK_TOKENS: int = 512

# Smoothing epsilon inside log-geometric mean exp(mean_m log(c_+^{(m)} + GEOM_COINCIDENCE_LOG_EPS))
GEOM_COINCIDENCE_LOG_EPS: float = 1e-4

# Blending coefficient weighting the 256-root coarse coincidence auxiliary CE loss (0.12 * w_c)
COARSE_AUX_LOSS_BLEND_SCALE: float = 0.12

# Scaling factor for optional full-vocab coincidence logit refinement during evaluation
FULL_VOCAB_COINCIDENCE_LOGIT_SCALE: float = 0.06
MAX_FULL_VOCAB_COINCIDENCE_TOKENS: int = 2048


@dataclass(frozen=True)
class SubspaceCoincidenceSpec:
    """Configuration for Product-of-Spheres Multi-Subspace Coincidence Born Readout."""

    num_subspaces: int = DEFAULT_NUM_SUBSPACES
    coincidence_gamma: float = DEFAULT_COINCIDENCE_GAMMA
    subspace_var_penalty: float = DEFAULT_SUBSPACE_VAR_PENALTY
    coarse_coincidence_weight: float = DEFAULT_COARSE_COINCIDENCE_WEIGHT
    partition_mode: str = DEFAULT_PARTITION_MODE
    full_vocab_coincidence_eval: bool = False
    subspace_balance_blend: float = DEFAULT_SUBSPACE_BALANCE_BLEND
    eps: float = DEFAULT_SUBSPACE_NORM_EPS
    use_fine_m4_chunked_coincidence: bool = True
    pre_normalized_inputs: bool = True


class _SubspaceL2NormalizeFn(torch.autograd.Function):
    """Memory-efficient Product-of-Spheres L2 normalization saving `bfloat16` output `y` instead of `float32` input."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        num_subspaces: int,
        eps: float,
        partition_mode: str,
    ) -> torch.Tensor:
        d = x.size(-1)
        m = max(1, int(num_subspaces))
        d_sub = d // m
        orig_shape = x.shape
        scale_m = 1.0 / math.sqrt(float(m))
        if partition_mode == "strided" and (d % (SYMPLECTIC_PAIR_DIM * m) == 0):
            x_s = x.reshape(-1, d // (SYMPLECTIC_PAIR_DIM * m), m, SYMPLECTIC_PAIR_DIM)
            inv_norm = torch.rsqrt(x_s.float().square().sum(dim=(-3, -1), keepdim=True) + eps) * scale_m
            y = (x_s * inv_norm.to(x.dtype)).reshape(orig_shape)
        elif partition_mode == "strided":
            x_s = x.reshape(-1, d_sub, m)
            inv_norm = torch.rsqrt(x_s.float().square().sum(dim=-2, keepdim=True) + eps) * scale_m
            y = (x_s * inv_norm.to(x.dtype)).reshape(orig_shape)
        else:
            x_s = x.reshape(-1, m, d_sub)
            inv_norm = torch.rsqrt(x_s.float().square().sum(dim=-1, keepdim=True) + eps) * scale_m
            y = (x_s * inv_norm.to(x.dtype)).reshape(orig_shape)
        ctx.save_for_backward(y, inv_norm.to(torch.float32))
        ctx.num_subspaces = m
        ctx.partition_mode = partition_mode
        ctx.orig_shape = orig_shape
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):  # type: ignore[override]
        y, inv_norm = ctx.saved_tensors
        m = int(ctx.num_subspaces)
        orig_shape = ctx.orig_shape
        d = orig_shape[-1]
        d_sub = d // m
        if ctx.partition_mode == "strided" and (d % (SYMPLECTIC_PAIR_DIM * m) == 0):
            y_f = y.reshape(-1, d // (SYMPLECTIC_PAIR_DIM * m), m, SYMPLECTIC_PAIR_DIM).float()
            dy_f = dy.reshape(-1, d // (SYMPLECTIC_PAIR_DIM * m), m, SYMPLECTIC_PAIR_DIM).float()
            proj = (dy_f * y_f).sum(dim=(-3, -1), keepdim=True) * float(m)
            dx = ((dy_f - proj * y_f) * inv_norm).to(y.dtype).reshape(orig_shape)
        elif ctx.partition_mode == "strided":
            y_f = y.reshape(-1, d_sub, m).float()
            dy_f = dy.reshape(-1, d_sub, m).float()
            proj = (dy_f * y_f).sum(dim=-2, keepdim=True) * float(m)
            dx = ((dy_f - proj * y_f) * inv_norm).to(y.dtype).reshape(orig_shape)
        else:
            y_f = y.reshape(-1, m, d_sub).float()
            dy_f = dy.reshape(-1, m, d_sub).float()
            proj = (dy_f * y_f).sum(dim=-1, keepdim=True) * float(m)
            dx = ((dy_f - proj * y_f) * inv_norm).to(y.dtype).reshape(orig_shape)
        return dx, None, None, None


def subspace_l2_normalize(
    x: torch.Tensor,
    num_subspaces: int = DEFAULT_NUM_SUBSPACES,
    eps: float = DEFAULT_SUBSPACE_NORM_EPS,
    partition_mode: str = DEFAULT_PARTITION_MODE,
) -> torch.Tensor:
    """Normalize `x` onto the Product-of-Spheres manifold `((1 / sqrt(M)) S^{d/M - 1})^M subset S^{d-1}`.

    In `"strided"` mode (`Wave 34` / `EXPERIMENT_PLAN_049.md` Section 3.1.2), views `x` as
    `(..., d // (2 * M), M, 2)` and normalizes across `dim=(-3, -1)` so every symplectic `(even, odd)`
    pair stays intact inside the same subspace `m`.
    """
    d = x.size(-1)
    m = max(1, int(num_subspaces))
    if m == 1 or (d % m != 0):
        return x * torch.rsqrt(x.float().square().sum(dim=-1, keepdim=True) + eps).to(x.dtype)
    if (
        x.is_cuda
        and x.requires_grad
        and x.ndim >= 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.numel() >= CUSTOM_L2_NORM_MIN_NUMEL
        and not torch.compiler.is_compiling()
    ):
        return _SubspaceL2NormalizeFn.apply(x, m, float(eps), str(partition_mode))
    d_sub = d // m
    orig_shape = x.shape
    scale_m = 1.0 / math.sqrt(float(m))
    if partition_mode == "strided" and (d % (SYMPLECTIC_PAIR_DIM * m) == 0):
        x_pair = x.reshape(*orig_shape[:-1], d // (SYMPLECTIC_PAIR_DIM * m), m, SYMPLECTIC_PAIR_DIM)
        inv_norm = torch.rsqrt(x_pair.float().square().sum(dim=(-3, -1), keepdim=True) + eps) * scale_m
        return (x_pair * inv_norm.to(x.dtype)).reshape(orig_shape)
    if partition_mode == "strided":
        x_strided = x.reshape(*orig_shape[:-1], d_sub, m)
        inv_norm = torch.rsqrt(x_strided.float().square().sum(dim=-2, keepdim=True) + eps) * scale_m
        return (x_strided * inv_norm.to(x.dtype)).reshape(orig_shape)
    x_sub = x.reshape(*orig_shape[:-1], m, d_sub)
    inv_norm = torch.rsqrt(x_sub.float().square().sum(dim=-1, keepdim=True) + eps) * scale_m
    return (x_sub * inv_norm.to(x.dtype)).reshape(orig_shape)


def blend_subspace_l2_normalize(
    x: torch.Tensor,
    num_subspaces: int = DEFAULT_NUM_SUBSPACES,
    blend: float = DEFAULT_SUBSPACE_BALANCE_BLEND,
    eps: float = DEFAULT_SUBSPACE_NORM_EPS,
    partition_mode: str = DEFAULT_PARTITION_MODE,
) -> torch.Tensor:
    """Spherically interpolate between unit-sphere `M=1` and Product-of-Spheres `M` normalization (`Wave 33`)."""
    b = float(max(0.0, min(1.0, blend)))
    if b >= 1.0 - 1e-6 or int(num_subspaces) <= 1:
        return subspace_l2_normalize(x, num_subspaces=num_subspaces, eps=eps, partition_mode=partition_mode)
    x_m1 = subspace_l2_normalize(x, num_subspaces=1, eps=eps, partition_mode=partition_mode)
    if b <= 1e-6:
        return x_m1
    x_m = subspace_l2_normalize(x, num_subspaces=num_subspaces, eps=eps, partition_mode=partition_mode)
    mixed = (1.0 - b) * x_m1 + b * x_m
    return subspace_l2_normalize(mixed, num_subspaces=1, eps=eps, partition_mode=partition_mode)


def strided_subspace_l2_normalize(
    x: torch.Tensor,
    num_subspaces: int = DEFAULT_NUM_SUBSPACES,
    eps: float = DEFAULT_SUBSPACE_NORM_EPS,
) -> torch.Tensor:
    """Normalize `x` onto `((1 / sqrt(M)) S^{d/M - 1})^M` preserving both strided and pair-strided norms."""
    d = x.size(-1)
    m = max(1, int(num_subspaces))
    if m > 1 and (d % (SYMPLECTIC_PAIR_DIM * m) == 0):
        orig_shape = x.shape
        x_pair = x.reshape(*orig_shape[:-1], d // (SYMPLECTIC_PAIR_DIM * m), m, SYMPLECTIC_PAIR_DIM)
        scale = 1.0 / math.sqrt(float(SYMPLECTIC_PAIR_DIM * m))
        inv_norm = torch.rsqrt(x_pair.float().square().sum(dim=-3, keepdim=True) + eps) * scale
        return (x_pair * inv_norm.to(x.dtype)).reshape(orig_shape)
    return subspace_l2_normalize(x, num_subspaces=num_subspaces, eps=eps, partition_mode="strided")


def _extract_m_subspaces(x_2d: torch.Tensor, num_subspaces: int, partition_mode: str) -> torch.Tensor:
    """Reshape 2D `(N, d)` tensor into contiguous `(M, N, d // M)` subspace tensor."""
    n_rows, d = x_2d.shape
    m = max(1, int(num_subspaces))
    d_sub = d // m
    if partition_mode == "strided" and (d % (SYMPLECTIC_PAIR_DIM * m) == 0):
        return (
            x_2d.reshape(n_rows, d // (SYMPLECTIC_PAIR_DIM * m), m, SYMPLECTIC_PAIR_DIM)
            .permute(2, 0, 1, 3)
            .contiguous()
            .reshape(m, n_rows, d_sub)
        )
    if partition_mode == "strided":
        return x_2d.reshape(n_rows, d_sub, m).permute(2, 0, 1).contiguous()
    return x_2d.reshape(n_rows, m, d_sub).permute(1, 0, 2).contiguous()


def _merge_m_subspaces(x_m: torch.Tensor, partition_mode: str) -> torch.Tensor:
    """Invert `_extract_m_subspaces` from `(M, N, d // M)` back to 2D `(N, d)`."""
    m, n_rows, d_sub = x_m.shape
    d = m * d_sub
    if partition_mode == "strided" and (d_sub % SYMPLECTIC_PAIR_DIM == 0):
        return (
            x_m.reshape(m, n_rows, d_sub // SYMPLECTIC_PAIR_DIM, SYMPLECTIC_PAIR_DIM)
            .permute(1, 2, 0, 3)
            .contiguous()
            .reshape(n_rows, d)
        )
    if partition_mode == "strided":
        return x_m.permute(1, 2, 0).contiguous().reshape(n_rows, d)
    return x_m.permute(1, 0, 2).contiguous().reshape(n_rows, d)


def compute_subspace_cosines(
    psi: torch.Tensor,
    detectors: torch.Tensor,
    num_subspaces: int = DEFAULT_NUM_SUBSPACES,
    partition_mode: str = DEFAULT_PARTITION_MODE,
) -> torch.Tensor:
    """Compute per-subspace unit-sphere cosines `c_{t,v}^{(m)} in [-1, 1]` of shape `(N_tok, V, M)`."""
    d = psi.size(-1)
    m = max(1, int(num_subspaces))
    psi_2d = psi.reshape(-1, d).float()
    det_2d = detectors.reshape(-1, d).float()
    psi_m = _extract_m_subspaces(psi_2d, m, partition_mode)
    det_m = _extract_m_subspaces(det_2d, m, partition_mode)
    return float(m) * torch.bmm(psi_m, det_m.transpose(1, 2)).permute(1, 2, 0)


def compute_multi_subspace_coincidence_observable(
    cos_m: torch.Tensor,
    traceless_gamma: float = DEFAULT_TRACELESS_GAMMA,
    coincidence_gamma: float = DEFAULT_COINCIDENCE_GAMMA,
    subspace_var_penalty: float = DEFAULT_SUBSPACE_VAR_PENALTY,
) -> dict[str, torch.Tensor]:
    """Compute traceless M-subspace coincidence observable from per-subspace cosines `(N_tok, V, M)`.

    Guarantees `sum_v observable_{t,v} = 0` identically for every query token `t` while boosting
    candidates where all `M` subspaces align (`geom_coinc`) and penalizing single-subspace spikes
    (`var_disagree`). Uses nested-sqrt 4-way coincidence for `M = 4` (`Wave 34` / `EXPERIMENT_PLAN_049.md`).
    """
    m = cos_m.size(-1)
    c_hat_m = cos_m - cos_m.mean(dim=1, keepdim=True)
    c_bar = c_hat_m.mean(dim=-1)
    c_pos_m = F.relu(c_hat_m)

    quad_m = F.relu(c_bar).square()
    quad_centered = quad_m - quad_m.mean(dim=-1, keepdim=True)

    if m == 4:
        a0, a1, a2, a3 = c_pos_m[..., 0], c_pos_m[..., 1], c_pos_m[..., 2], c_pos_m[..., 3]
        p01 = torch.sqrt(a0 * a1 + M4_GEOM_EPS_SQ)
        p23 = torch.sqrt(a2 * a3 + M4_GEOM_EPS_SQ)
        geom_coinc = torch.sqrt(p01 * p23 + M4_GEOM_EPS_SQ) - M4_GEOM_EPS_OFFSET
    else:
        geom_coinc = torch.exp(torch.log(c_pos_m + GEOM_COINCIDENCE_LOG_EPS).mean(dim=-1)) - GEOM_COINCIDENCE_LOG_EPS
    geom_centered = geom_coinc - geom_coinc.mean(dim=-1, keepdim=True)

    var_disagree = ((c_hat_m - c_bar.unsqueeze(-1)).square().mean(dim=-1)) if m > 1 else torch.zeros_like(c_bar)
    var_centered = var_disagree - var_disagree.mean(dim=-1, keepdim=True)

    obs = (
        c_bar
        + float(traceless_gamma) * quad_centered
        + float(coincidence_gamma) * geom_centered
        - float(subspace_var_penalty) * var_centered
    )
    return {
        "observable": obs,
        "c_bar": c_bar,
        "quad_centered": quad_centered,
        "geom_coinc": geom_coinc,
        "var_disagree": var_disagree,
    }


def compute_coarse_cluster_centers(
    m_sharp: torch.Tensor,
    readout_spec_or_perm: ParsevalChunkedCEReadoutSpec | torch.Tensor,
    coinc_spec: SubspaceCoincidenceSpec | None = None,
    num_coarse: int = 256,
    num_subspaces: int = DEFAULT_NUM_SUBSPACES,
    partition_mode: str = DEFAULT_PARTITION_MODE,
    eps: float = DEFAULT_SUBSPACE_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize Subspace-0 root-consistent coarse cluster centroids `(N_coarse, d)` and `v_to_coarse` map.

    Updated in `Wave 34` (`EXPERIMENT_PLAN_049.md` Section 3.2.3) so that detail subspaces `m = 1, 2, 3`
    (which sum to near-zero across the ~196.3 leaves in each coarse bucket) are anchored by Subspace 0
    (`coarse_256`, weighted by `COARSE_ROOT_SUBSPACE_BLEND = 0.85`), eliminating the `2.10`-nat coarse
    cancellation floor.
    """
    device = m_sharp.device
    if isinstance(readout_spec_or_perm, ParsevalChunkedCEReadoutSpec):
        r_spec = readout_spec_or_perm
        c_spec = coinc_spec if coinc_spec is not None else SubspaceCoincidenceSpec(
            num_subspaces=num_subspaces, partition_mode=partition_mode, eps=eps
        )
        v_act = min(m_sharp.size(0), int(r_spec.valid_vocab_size)) if int(r_spec.valid_vocab_size) > 0 else m_sharp.size(0)
        n_c = max(1, min(int(r_spec.num_coarse_buckets), v_act))
        m_sub = max(1, int(c_spec.num_subspaces))
        pmode = str(c_spec.partition_mode)
        eps_val = float(c_spec.eps)
        if r_spec.v_to_coarse is not None and r_spec.v_to_coarse.numel() >= v_act:
            v_map = torch.remainder(r_spec.v_to_coarse[:v_act].to(device=device, dtype=torch.long), n_c)
        else:
            v_map = torch.div(torch.arange(v_act, device=device, dtype=torch.long) * n_c, v_act, rounding_mode="floor")
    else:
        raw_perm = readout_spec_or_perm.to(device=device, dtype=torch.long)
        v_act = min(m_sharp.size(0), raw_perm.numel())
        n_c = max(1, min(int(num_coarse), v_act))
        m_sub = max(1, int(coinc_spec.num_subspaces if coinc_spec is not None else num_subspaces))
        pmode = str(coinc_spec.partition_mode if coinc_spec is not None else partition_mode)
        eps_val = float(coinc_spec.eps if coinc_spec is not None else eps)
        perm_slice = raw_perm[:v_act]
        if v_act > 0 and int(perm_slice.max().item()) >= n_c and n_c == 256:
            v_map = torch.remainder(perm_slice >> 8, n_c)
        else:
            v_map = torch.remainder(perm_slice, n_c)

    d = m_sharp.size(-1)
    centers = torch.zeros((n_c, d), device=device, dtype=torch.float32)
    centers.index_add_(0, v_map, m_sharp[:v_act].float())
    if m_sub > 1 and (d % m_sub == 0):
        centers_m = _extract_m_subspaces(centers, m_sub, pmode)
        scale_m = 1.0 / math.sqrt(float(m_sub))
        c0_unit = F.normalize(centers_m[0], p=2.0, dim=-1, eps=eps_val) * scale_m
        detail_unit = F.normalize(centers_m, p=2.0, dim=-1, eps=eps_val) * scale_m
        blended_m = COARSE_ROOT_SUBSPACE_BLEND * c0_unit.unsqueeze(0) + (1.0 - COARSE_ROOT_SUBSPACE_BLEND) * detail_unit
        centers_m_norm = F.normalize(blended_m, p=2.0, dim=-1, eps=eps_val) * scale_m
        centers_norm = _merge_m_subspaces(centers_m_norm, pmode)
        return centers_norm, v_map
    centers_norm = subspace_l2_normalize(centers, num_subspaces=m_sub, eps=eps_val, partition_mode=pmode)
    return centers_norm, v_map


def _prepare_root_anchored_coarse_queries(
    psi_sharp: torch.Tensor,
    num_subspaces: int,
    partition_mode: str,
    eps: float,
) -> torch.Tensor:
    """Blend Subspace 0 (`coarse_256`) across query subspaces for root-consistent 256-bucket comparison."""
    d = psi_sharp.size(-1)
    m_sub = max(1, int(num_subspaces))
    if m_sub <= 1 or (d % m_sub != 0):
        return subspace_l2_normalize(psi_sharp, num_subspaces=m_sub, eps=eps, partition_mode=partition_mode)
    psi_2d = psi_sharp.reshape(-1, d).float()
    psi_m = _extract_m_subspaces(psi_2d, m_sub, partition_mode)
    scale_m = 1.0 / math.sqrt(float(m_sub))
    q0_unit = F.normalize(psi_m[0], p=2.0, dim=-1, eps=eps) * scale_m
    q_norm = q0_unit.unsqueeze(0).expand(m_sub, -1, -1).contiguous()
    return _merge_m_subspaces(q_norm, partition_mode)


def coarse_coincidence_ce_from_centers(
    psi_sharp: torch.Tensor,
    m_sharp: torch.Tensor,
    targets: torch.Tensor,
    readout_spec: ParsevalChunkedCEReadoutSpec,
    coinc_spec: SubspaceCoincidenceSpec,
) -> torch.Tensor:
    """Compute 256-way coarse cluster cross-entropy using Subspace-0 root-consistent centroids (`Wave 34`)."""
    pmode = coinc_spec.partition_mode
    m_sub = int(coinc_spec.num_subspaces)
    centers_norm, v_map = compute_coarse_cluster_centers(m_sharp, readout_spec, coinc_spec)
    psi_q = _prepare_root_anchored_coarse_queries(psi_sharp, m_sub, pmode, coinc_spec.eps)
    cos_coarse_m = compute_subspace_cosines(psi_q, centers_norm, num_subspaces=m_sub, partition_mode=pmode)
    obs_dict = compute_multi_subspace_coincidence_observable(
        cos_coarse_m,
        traceless_gamma=DEFAULT_TRACELESS_GAMMA,
        coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
        subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
    )
    d = psi_sharp.size(-1)
    beta_coarse = max(float(readout_spec.beta_0), 1.48 * math.sqrt(float(d)))
    coarse_logits = beta_coarse * obs_dict["observable"]
    flat_y = targets.reshape(-1).long()
    v_act = v_map.numel()
    valid_mask = (flat_y != int(readout_spec.ignore_index)) & (flat_y >= 0) & (flat_y < v_act)
    safe_y = torch.where(valid_mask, flat_y, torch.zeros_like(flat_y))
    coarse_targets = v_map.index_select(0, safe_y)
    coarse_targets = torch.where(
        valid_mask, coarse_targets, torch.full_like(coarse_targets, int(readout_spec.ignore_index))
    )
    return F.cross_entropy(coarse_logits, coarse_targets, ignore_index=int(readout_spec.ignore_index))


class _SubspaceCoincidenceM4ChunkedCEFn(torch.autograd.Function):
    """Single-pass chunked 50,257-way M=4 Subspace Coincidence Born Cross-Entropy (`Wave 34`)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        psi_sharp: torch.Tensor,
        psi_diff: torch.Tensor,
        m_sharp: torch.Tensor,
        m_diff: torch.Tensor,
        targets: torch.Tensor,
        log_prior: torch.Tensor | None,
        specs: tuple[ParsevalChunkedCEReadoutSpec, SubspaceCoincidenceSpec],
    ) -> torch.Tensor:
        spec, coinc_spec = specs
        orig_s, orig_d = psi_sharp.shape, psi_diff.shape
        d, r = psi_sharp.size(-1), psi_diff.size(-1)
        d_sub = d // 4
        psi_s_flat, psi_d_flat = psi_sharp.detach().reshape(-1, d), psi_diff.detach().reshape(-1, r)
        flat_y = targets.reshape(-1).long()
        n_tok, v_vocab = psi_s_flat.size(0), m_sharp.size(0)
        v_act = min(v_vocab, int(spec.valid_vocab_size)) if int(spec.valid_vocab_size) > 0 else v_vocab
        chunk_sz = max(1, min(int(spec.chunk_size), n_tok))
        if v_vocab >= 16384 and psi_s_flat.is_cuda:
            max_c = 256 if d >= 768 else (512 if d >= 512 else 1024)
            chunk_sz = min(chunk_sz, max_c)
        acc_dt = torch.float64 if psi_sharp.dtype == torch.float64 else torch.float32
        work_dt = (
            torch.get_autocast_gpu_dtype()
            if (psi_s_flat.is_cuda and torch.is_autocast_enabled() and psi_sharp.dtype != torch.float64)
            else psi_s_flat.dtype
        )
        pmode = str(coinc_spec.partition_mode)
        psi_m_all = _extract_m_subspaces(psi_s_flat.to(work_dt), 4, pmode)
        m_s_m = _extract_m_subspaces(m_sharp.detach()[:v_act].to(work_dt), 4, pmode)
        m_s_m_t = m_s_m.transpose(1, 2).contiguous()
        m_d_act = m_diff.detach()[:v_act].to(work_dt)
        prior_act = log_prior.detach().to(acc_dt).view(1, -1)[:, :v_act] if log_prior is not None else None

        beta_0, lam_sc = float(spec.beta_0), float(spec.lambda_diff_scaled)
        gamma, quad_w = float(spec.traceless_gamma), float(spec.quad_diffuse_fidelity)
        gamma_c, lam_var = float(coinc_spec.coincidence_gamma), float(coinc_spec.subspace_var_penalty)
        scalars = (beta_0, lam_sc, gamma, gamma_c, lam_var, M4_SUBSPACE_SCALE, quad_w)

        ns, nd, nms, nmd = (bool(ctx.needs_input_grad[i]) for i in range(4))
        npr = bool(log_prior is not None and ctx.needs_input_grad[5])
        dpsi_s_m = torch.empty((4, n_tok, d_sub), device=psi_s_flat.device, dtype=work_dt) if ns else None
        dpsi_d = torch.empty((n_tok, r), device=psi_d_flat.device, dtype=work_dt) if nd else None
        dm_s_m = torch.zeros((4, v_act, d_sub), device=m_sharp.device, dtype=work_dt) if nms else None
        dm_d = torch.zeros((v_vocab, r), device=m_diff.device, dtype=work_dt) if nmd else None
        dprior = torch.zeros((v_vocab,), device=psi_s_flat.device, dtype=acc_dt) if npr else None

        with torch.no_grad():
            valid_all = (flat_y != int(spec.ignore_index)) & (flat_y >= 0) & (flat_y < v_act)
            inv_n = 1.0 / torch.maximum(
                valid_all.sum().to(acc_dt), torch.ones((), device=psi_s_flat.device, dtype=acc_dt)
            )
            total_loss = torch.zeros((), device=psi_s_flat.device, dtype=acc_dt)
            v_idx = torch.arange(v_act, device=psi_s_flat.device).unsqueeze(0)

            for start in range(0, n_tok, chunk_sz):
                end = min(start + chunk_sz, n_tok)
                psi_m_c = psi_m_all[:, start:end]
                psi_d_c = psi_d_flat[start:end].to(work_dt)
                yc_val = valid_all[start:end].unsqueeze(1)
                yc_safe = torch.where(
                    yc_val, flat_y[start:end].unsqueeze(1), torch.zeros_like(flat_y[start:end].unsqueeze(1))
                )
                val_f = yc_val.to(acc_dt)
                w_sharp_m = torch.bmm(psi_m_c, m_s_m_t)
                w_diff = torch.mm(psi_d_c, m_d_act.t())
                loss_c, grad_s_m, grad_d, dpr_c = compiled_m4_coincidence_chunk_step(
                    w_sharp_m, w_diff, yc_safe, (val_f, val_f * inv_n), scalars, prior_act, v_idx
                )
                total_loss.add_(loss_c)
                if npr and dprior is not None:
                    dprior[:v_act].add_(dpr_c)
                if ns and dpsi_s_m is not None:
                    torch.bmm(grad_s_m, m_s_m, out=dpsi_s_m[:, start:end])
                if nms and dm_s_m is not None:
                    dm_s_m.baddbmm_(grad_s_m.transpose(1, 2), psi_m_c)
                if nd and dpsi_d is not None:
                    torch.mm(grad_d, m_d_act, out=dpsi_d[start:end])
                if nmd and dm_d is not None:
                    dm_d[:v_act].addmm_(grad_d.t(), psi_d_c)

            dpsi_s: torch.Tensor | None = None
            if ns and dpsi_s_m is not None:
                dpsi_s = _merge_m_subspaces(dpsi_s_m, pmode).to(psi_sharp.dtype).view(orig_s)
            dm_s: torch.Tensor | None = None
            if nms and dm_s_m is not None:
                dm_s = torch.zeros((v_vocab, d), device=m_sharp.device, dtype=m_sharp.dtype)
                dm_s[:v_act] = _merge_m_subspaces(dm_s_m, pmode).to(m_sharp.dtype)

        empty = torch.empty(0, device=psi_s_flat.device)
        ctx.save_for_backward(
            dpsi_s if dpsi_s is not None else empty,
            dpsi_d.to(psi_diff.dtype).view(orig_d) if dpsi_d is not None else empty,
            dm_s if dm_s is not None else empty,
            dm_d.to(m_diff.dtype) if dm_d is not None else empty,
            dprior.view(log_prior.shape).to(log_prior.dtype) if (dprior is not None and log_prior is not None) else empty,
        )
        ctx.needs = (ns, nd, nms, nmd, npr)
        return total_loss * inv_n

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        saved = ctx.saved_tensors
        g = grad_out.to(torch.float64 if any(t.dtype == torch.float64 for t in saved if t.numel() > 0) else torch.float32)
        grads = [(t.to(g.dtype) * g).to(t.dtype) if need else None for t, need in zip(saved, ctx.needs)]
        return grads[0], grads[1], grads[2], grads[3], None, grads[4], None


def subspace_coincidence_dyadic_chunked_ce(
    psi_sharp: torch.Tensor,
    psi_diff: torch.Tensor,
    m_sharp: torch.Tensor,
    m_diff: torch.Tensor,
    targets: torch.Tensor,
    readout_spec: ParsevalChunkedCEReadoutSpec,
    coinc_spec: SubspaceCoincidenceSpec,
    log_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute fused Parseval muP chunked M=4 coincidence CE + 256-root auxiliary coincidence CE."""
    pmode = coinc_spec.partition_mode
    b_sub = float(coinc_spec.subspace_balance_blend)
    if coinc_spec.pre_normalized_inputs:
        psi_base = psi_sharp
        m_base = m_sharp
    else:
        psi_base = blend_subspace_l2_normalize(
            psi_sharp, num_subspaces=coinc_spec.num_subspaces, blend=b_sub, eps=coinc_spec.eps, partition_mode=pmode
        )
        m_base = blend_subspace_l2_normalize(
            m_sharp, num_subspaces=coinc_spec.num_subspaces, blend=b_sub, eps=coinc_spec.eps, partition_mode=pmode
        )

    d = psi_base.size(-1)
    if coinc_spec.use_fine_m4_chunked_coincidence and int(coinc_spec.num_subspaces) == 4 and (d % 8 == 0):
        base_loss = _SubspaceCoincidenceM4ChunkedCEFn.apply(
            psi_base, psi_diff, m_base, m_diff, targets, log_prior, (readout_spec, coinc_spec)
        )
    else:
        base_loss = parseval_mup_dyadic_chunked_ce(
            psi_base, psi_diff, m_base, m_diff, targets, readout_spec, log_prior
        )

    w_c = float(coinc_spec.coarse_coincidence_weight)
    if w_c <= 0.0 or int(coinc_spec.num_subspaces) <= 1:
        return base_loss

    coarse_ce = coarse_coincidence_ce_from_centers(psi_base, m_base, targets, readout_spec, coinc_spec)
    blend = COARSE_AUX_LOSS_BLEND_SCALE * w_c
    return (1.0 - blend) * base_loss + blend * coarse_ce.to(base_loss.dtype)


def eval_subspace_coincidence_full_logits(
    psi_sharp: torch.Tensor,
    psi_diff: torch.Tensor,
    m_sharp: torch.Tensor,
    m_diff: torch.Tensor,
    readout_spec: ParsevalChunkedCEReadoutSpec,
    coinc_spec: SubspaceCoincidenceSpec,
    log_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate full vocabulary logits with exact train/eval parity for 50,257-way M=4 coincidence (`Wave 34`)."""
    pmode = coinc_spec.partition_mode
    b_sub = float(coinc_spec.subspace_balance_blend)
    if coinc_spec.pre_normalized_inputs:
        psi_base = psi_sharp
        m_base = m_sharp
    else:
        psi_base = blend_subspace_l2_normalize(
            psi_sharp, num_subspaces=coinc_spec.num_subspaces, blend=b_sub, eps=coinc_spec.eps, partition_mode=pmode
        )
        m_base = blend_subspace_l2_normalize(
            m_sharp, num_subspaces=coinc_spec.num_subspaces, blend=b_sub, eps=coinc_spec.eps, partition_mode=pmode
        )

    d = psi_base.size(-1)
    v_act = min(m_base.size(0), int(readout_spec.valid_vocab_size)) if int(readout_spec.valid_vocab_size) > 0 else m_base.size(0)
    lead_shape = psi_base.shape[:-1]
    n_tok = psi_base.numel() // d

    if coinc_spec.use_fine_m4_chunked_coincidence and int(coinc_spec.num_subspaces) == 4 and (d % 8 == 0):
        psi_2d = psi_base.reshape(n_tok, d).float()
        psi_d_2d = psi_diff.reshape(n_tok, -1).float()
        psi_m = _extract_m_subspaces(psi_2d, 4, pmode)
        m_s_m = _extract_m_subspaces(m_base[:v_act].float(), 4, pmode)
        m_s_m_t = m_s_m.transpose(1, 2).contiguous()
        m_d_act = m_diff[:v_act].float()
        prior_act = log_prior[:v_act].float().view(1, -1) if log_prior is not None else None

        beta_0 = float(readout_spec.beta_0)
        lam_sc = float(readout_spec.lambda_diff_scaled)
        gamma = float(readout_spec.traceless_gamma)
        quad_w = float(readout_spec.quad_diffuse_fidelity)
        gamma_c = float(coinc_spec.coincidence_gamma)
        lam_var = float(coinc_spec.subspace_var_penalty)

        logits_chunks: list[torch.Tensor] = []
        for start in range(0, n_tok, EVAL_M4_CHUNK_TOKENS):
            end = min(start + EVAL_M4_CHUNK_TOKENS, n_tok)
            w_sharp_m = torch.bmm(psi_m[:, start:end], m_s_m_t)
            cos_m = (M4_SUBSPACE_SCALE * w_sharp_m).permute(1, 2, 0)
            o_hat = compute_multi_subspace_coincidence_observable(
                cos_m,
                traceless_gamma=gamma,
                coincidence_gamma=gamma_c,
                subspace_var_penalty=lam_var,
            )["observable"]
            u_raw = torch.matmul(psi_d_2d[start:end], m_d_act.t())
            u_pos = F.relu(u_raw)
            f_quad = u_raw + quad_w * (u_pos * u_pos)
            f_hat = f_quad - f_quad.mean(dim=-1, keepdim=True)
            chunk_logits = beta_0 * o_hat + lam_sc * f_hat
            if prior_act is not None:
                chunk_logits = chunk_logits + prior_act
            logits_chunks.append(chunk_logits)
        logits_2d = torch.cat(logits_chunks, dim=0)
    else:
        base_logits = eval_parseval_mup_full_logits(psi_base, psi_diff, m_base, m_diff, readout_spec, log_prior)
        assert isinstance(base_logits, torch.Tensor)
        if int(coinc_spec.num_subspaces) <= 1:
            return base_logits
        logits_2d = base_logits.reshape(n_tok, v_act).clone()
        beta_0 = float(readout_spec.beta_0)
        if coinc_spec.full_vocab_coincidence_eval and n_tok <= MAX_FULL_VOCAB_COINCIDENCE_TOKENS:
            psi_s_bal = subspace_l2_normalize(
                psi_sharp, num_subspaces=coinc_spec.num_subspaces, eps=coinc_spec.eps, partition_mode=pmode
            )
            m_s_bal = subspace_l2_normalize(
                m_sharp, num_subspaces=coinc_spec.num_subspaces, eps=coinc_spec.eps, partition_mode=pmode
            )
            cos_v_m = compute_subspace_cosines(
                psi_s_bal, m_s_bal[:v_act], num_subspaces=coinc_spec.num_subspaces, partition_mode=pmode
            )
            obs_v = compute_multi_subspace_coincidence_observable(
                cos_v_m,
                traceless_gamma=0.0,
                coincidence_gamma=coinc_spec.coincidence_gamma,
                subspace_var_penalty=coinc_spec.subspace_var_penalty,
            )["observable"]
            logits_2d.add_(obs_v.to(logits_2d.dtype), alpha=FULL_VOCAB_COINCIDENCE_LOGIT_SCALE * beta_0)

    if float(coinc_spec.coarse_coincidence_weight) > 0.0:
        m_sub = int(coinc_spec.num_subspaces)
        centers_norm, v_map = compute_coarse_cluster_centers(m_base, readout_spec, coinc_spec)
        psi_q = _prepare_root_anchored_coarse_queries(psi_base, m_sub, pmode, coinc_spec.eps)
        cos_c_m = compute_subspace_cosines(psi_q, centers_norm, num_subspaces=m_sub, partition_mode=pmode)
        obs_c = compute_multi_subspace_coincidence_observable(
            cos_c_m,
            traceless_gamma=DEFAULT_TRACELESS_GAMMA,
            coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
            subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        )["observable"]
        beta_coarse = max(float(readout_spec.beta_0), 1.48 * math.sqrt(float(d)))
        coarse_log_p = F.log_softmax(beta_coarse * obs_c, dim=-1)
        coarse_lift = coarse_log_p.index_select(1, v_map)
        coarse_lift = coarse_lift - coarse_lift.mean(dim=-1, keepdim=True)
        blend = COARSE_AUX_LOSS_BLEND_SCALE * float(coinc_spec.coarse_coincidence_weight)
        logits_2d = logits_2d + blend * coarse_lift.to(logits_2d.dtype)

    return logits_2d.reshape(*lead_shape, v_act)


__all__ = [
    "COARSE_AUX_LOSS_BLEND_SCALE",
    "COARSE_ROOT_SUBSPACE_BLEND",
    "DEFAULT_COARSE_COINCIDENCE_WEIGHT",
    "DEFAULT_COINCIDENCE_GAMMA",
    "DEFAULT_NUM_SUBSPACES",
    "DEFAULT_PARTITION_MODE",
    "DEFAULT_SUBSPACE_BALANCE_BLEND",
    "DEFAULT_SUBSPACE_NORM_EPS",
    "DEFAULT_SUBSPACE_VAR_PENALTY",
    "DEFAULT_TRACELESS_GAMMA",
    "FULL_VOCAB_COINCIDENCE_LOGIT_SCALE",
    "GEOM_COINCIDENCE_LOG_EPS",
    "MAX_FULL_VOCAB_COINCIDENCE_TOKENS",
    "SubspaceCoincidenceSpec",
    "blend_subspace_l2_normalize",
    "coarse_coincidence_ce_from_centers",
    "compute_coarse_cluster_centers",
    "compute_multi_subspace_coincidence_observable",
    "compute_subspace_cosines",
    "eval_subspace_coincidence_full_logits",
    "strided_subspace_l2_normalize",
    "subspace_coincidence_dyadic_chunked_ce",
    "subspace_l2_normalize",
]



