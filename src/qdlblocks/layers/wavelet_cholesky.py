"""Multi-level Wavelet-Cholesky amplitude downsampling/upsampling and tied 2^16 basis operators.

Implements Experiment Plan 034:
1. `BlockCholeskyLinear`: Block-banded lower-triangular Cholesky operator `C = L @ L^T` with
   strictly positive diagonal `diag(L) = softplus(d + inv_softplus(1)) > 0` (identity at Step 0).
2. `WaveletCholeskyMultiscaleBranch`: 3-level Sweldens lifting wavelet analysis (downsampling
   `d -> d/2 -> d/4 -> d/8`), scale-specific Cholesky transformations `L_s @ L_s^T`, and exact
   inverse lifting wavelet synthesis (upsampling `d/8 -> d/4 -> d/2 -> d`).
3. `apply_register_cholesky_coupling`: Applies per-register lower-triangular Cholesky factors
   `L_q` (`Q x A x A`) and dyadic `2^16` (`65,536`) wavelet-packet mother amplitudes to a tied
   qubit register basis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "WaveletCholeskyConfig",
    "BlockCholeskyLinear",
    "WaveletCholeskyMultiscaleBranch",
    "build_lower_triangular_cholesky",
    "apply_register_cholesky_coupling",
]

_INV_SOFTPLUS_ONE = math.log(math.e - 1.0)


@dataclass(frozen=True)
class WaveletCholeskyConfig:
    """Configuration for multi-level Wavelet-Cholesky amplitude downsampling and upsampling."""

    embed_dim: int = 256
    num_levels: int = 3
    max_block_size: int = 32
    init_nonlin_gate: float = 0.0
    init_branch_scale: float = 0.25
    fast_spd_bmm: bool = False


def build_lower_triangular_cholesky(
    raw_tril: torch.Tensor,
    raw_diag: torch.Tensor,
    min_diag: float = 1e-3,
) -> torch.Tensor:
    """Construct a lower-triangular Cholesky factor `L` with strictly positive diagonal.

    At `raw_tril == 0` and `raw_diag == 0`, returns the exact identity matrix `I`.
    """
    strict_lower = torch.tril(raw_tril, diagonal=-1)
    pos_diag = F.softplus(raw_diag + _INV_SOFTPLUS_ONE).clamp_min(min_diag)
    return strict_lower + torch.diag_embed(pos_diag)


class BlockCholeskyLinear(nn.Module):
    """Symmetric positive-definite block-Cholesky linear transformation `y = L (L^T x)`."""

    def __init__(self, dim: int, max_block_size: int = 32, fast_spd_bmm: bool = False) -> None:
        super().__init__()
        self.dim = int(dim)
        self.block_size = math.gcd(self.dim, min(int(max_block_size), self.dim))
        self.num_blocks = self.dim // self.block_size
        self.fast_spd_bmm = bool(fast_spd_bmm)
        self.raw_tril = nn.Parameter(torch.zeros(self.num_blocks, self.block_size, self.block_size))
        self.raw_diag = nn.Parameter(torch.zeros(self.num_blocks, self.block_size))

    def cholesky_factor(self) -> torch.Tensor:
        """Return the `(num_blocks, block_size, block_size)` lower-triangular Cholesky tensor `L`."""
        return build_lower_triangular_cholesky(self.raw_tril.float(), self.raw_diag.float())

    def spd_matrix(self) -> torch.Tensor:
        """Return the `(num_blocks, block_size, block_size)` SPD block matrix `C = L @ L^T`."""
        l_mat = self.cholesky_factor()
        return torch.bmm(l_mat, l_mat.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if self.fast_spd_bmm:
            if x.is_cuda and torch.is_autocast_enabled():
                work_dtype = torch.get_autocast_gpu_dtype()
                out_dtype = work_dtype
            elif x.is_cuda and x.dtype in (torch.float32, torch.bfloat16):
                work_dtype = torch.bfloat16
                out_dtype = x.dtype
            else:
                work_dtype = x.dtype
                out_dtype = x.dtype
            c_mat = self.spd_matrix().to(work_dtype)
            if self.num_blocks == 1:
                return torch.matmul(x.to(work_dtype), c_mat[0]).to(out_dtype)
            x_3d = x.to(work_dtype).reshape(-1, self.num_blocks, self.block_size).transpose(0, 1)
            y_3d = torch.bmm(x_3d, c_mat).transpose(0, 1)
            return y_3d.reshape(orig_shape).to(out_dtype)
        x_blk = x.float().reshape(*orig_shape[:-1], self.num_blocks, self.block_size)
        l_mat = self.cholesky_factor()
        # Apply L^T followed by L: C = L @ L^T
        z_blk = torch.einsum("...gb,gbi->...gi", x_blk, l_mat)
        y_blk = torch.einsum("...gi,gbi->...gb", z_blk, l_mat)
        return y_blk.reshape(orig_shape).to(x.dtype)

    def condition_metrics(self) -> dict[str, float]:
        """Compute minimum diagonal, maximum singular value, and condition number of `L @ L^T`."""
        with torch.no_grad():
            l_mat = self.cholesky_factor()
            s_vals = torch.linalg.svdvals(l_mat)
            s_min = float(s_vals.min().clamp_min(1e-8).item())
            s_max = float(s_vals.max().item())
            diag_min = float(torch.diagonal(l_mat, dim1=-2, dim2=-1).min().item())
            return {
                "min_diag": diag_min,
                "max_chol_sv": s_max,
                "min_chol_sv": s_min,
                "spd_cond_number": float((s_max / s_min) ** 2),
            }


class WaveletCholeskyMultiscaleBranch(nn.Module):
    """Multi-level Sweldens lifting wavelet downsample -> Cholesky scale coupling -> upsample V-cycle."""

    def __init__(self, cfg: WaveletCholeskyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = int(cfg.embed_dim)
        self.num_levels = int(cfg.num_levels)
        if self.embed_dim % (2**self.num_levels) != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by 2**num_levels={2**self.num_levels}"
            )

        self.lift_predict = nn.ParameterList()
        self.lift_update = nn.ParameterList()
        self.detail_cholesky = nn.ModuleList()
        self.detail_nonlin_gate = nn.ParameterList()

        cur_dim = self.embed_dim
        for _ in range(self.num_levels):
            half_dim = cur_dim // 2
            self.lift_predict.append(nn.Parameter(torch.zeros(half_dim)))
            self.lift_update.append(nn.Parameter(torch.zeros(half_dim)))
            self.detail_cholesky.append(
                BlockCholeskyLinear(half_dim, max_block_size=cfg.max_block_size, fast_spd_bmm=cfg.fast_spd_bmm)
            )
            self.detail_nonlin_gate.append(
                nn.Parameter(torch.tensor(float(cfg.init_nonlin_gate), dtype=torch.float32))
            )
            cur_dim = half_dim

        self.coarse_dim = cur_dim
        self.coarse_cholesky = BlockCholeskyLinear(
            self.coarse_dim, max_block_size=min(self.coarse_dim, cfg.max_block_size), fast_spd_bmm=cfg.fast_spd_bmm
        )
        self.coarse_nonlin_gate = nn.Parameter(
            torch.tensor(float(cfg.init_nonlin_gate), dtype=torch.float32)
        )
        self.branch_scale = nn.Parameter(
            torch.tensor(float(cfg.init_branch_scale), dtype=torch.float32)
        )

    def _work_dtype(self, x: torch.Tensor) -> torch.dtype:
        if x.is_cuda:
            if torch.is_autocast_enabled():
                return torch.get_autocast_gpu_dtype()
            if x.dtype in (torch.bfloat16, torch.float16) or self.cfg.fast_spd_bmm:
                return x.dtype
        return torch.float32

    def downsample_amplitudes(
        self, psi: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decompose `psi` across `num_levels` dyadic wavelet scales into `(coarse, details)`."""
        w_dtype = self._work_dtype(psi)
        approx = psi.to(w_dtype)
        details: list[torch.Tensor] = []
        inv_sqrt2 = math.sqrt(0.5)
        for s in range(self.num_levels):
            x_e = approx[..., 0::2]
            x_o = approx[..., 1::2]
            p_s = self.lift_predict[s].to(w_dtype)
            u_s = self.lift_update[s].to(w_dtype)
            r_o = x_o - p_s * x_e
            r_e = x_e + u_s * r_o
            approx = inv_sqrt2 * (r_e + r_o)
            detail = inv_sqrt2 * (r_o - r_e)
            details.append(detail)
        return approx, details

    def apply_cholesky_scale_operators(
        self, coarse: torch.Tensor, details: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply scale-specific Cholesky operators `L_s @ L_s^T` and gated wavelet shrinkage."""
        w_dtype = self._work_dtype(coarse)
        g_c = (0.5 * torch.tanh(self.coarse_nonlin_gate.float())).to(w_dtype)
        c_in = coarse.to(w_dtype) + g_c * F.silu(coarse.to(w_dtype))
        coarse_out = self.coarse_cholesky(c_in)

        details_out: list[torch.Tensor] = []
        for s, det in enumerate(details):
            g_s = (0.5 * torch.tanh(self.detail_nonlin_gate[s].float())).to(w_dtype)
            d_t = det.to(w_dtype)
            d_in = d_t + g_s * F.silu(d_t)
            details_out.append(self.detail_cholesky[s](d_in))
        return coarse_out, details_out

    def upsample_amplitudes(
        self, coarse: torch.Tensor, details: list[torch.Tensor]
    ) -> torch.Tensor:
        """Synthesize full-resolution amplitude in `R^d` via exact inverse lifting wavelet steps."""
        w_dtype = self._work_dtype(coarse)
        approx = coarse.to(w_dtype)
        inv_sqrt2 = math.sqrt(0.5)
        for s in range(self.num_levels - 1, -1, -1):
            det = details[s].to(w_dtype)
            p_s = self.lift_predict[s].to(w_dtype)
            u_s = self.lift_update[s].to(w_dtype)
            r_e = inv_sqrt2 * (approx - det)
            r_o = inv_sqrt2 * (approx + det)
            x_e = r_e - u_s * r_o
            x_o = r_o + p_s * x_e
            interleaved = torch.stack((x_e, x_o), dim=-1).reshape(*approx.shape[:-1], -1)
            approx = interleaved
        return approx

    def _vcycle_impl(self, psi: torch.Tensor) -> torch.Tensor:
        coarse, details = self.downsample_amplitudes(psi)
        coarse_t, details_t = self.apply_cholesky_scale_operators(coarse, details)
        recon = self.upsample_amplitudes(coarse_t, details_t)
        return recon.to(self._work_dtype(psi))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        if (
            psi.requires_grad
            and psi.numel() >= 4_194_304
            and not torch.compiler.is_compiling()
        ):
            return torch.utils.checkpoint.checkpoint(
                self._vcycle_impl, psi, use_reentrant=False
            )
        return self._vcycle_impl(psi)


def apply_register_cholesky_coupling(
    base_qubits: torch.Tensor,
    raw_chol_tril: torch.Tensor,
    raw_chol_diag: torch.Tensor,
    wavelet_packets: torch.Tensor | None = None,
    packet_indices: torch.Tensor | None = None,
    wavelet_mix: float = 0.15,
) -> torch.Tensor:
    """Couple `(V, Q, A)` qubit register amplitudes via `2^16` wavelet packets and `L_q @ L_q^T`.

    Args:
        base_qubits: Unigram qubit tensor of shape `(V, Q, A)`.
        raw_chol_tril: Lower-triangular Cholesky parameters of shape `(Q, A, A)`.
        raw_chol_diag: Diagonal Cholesky parameters of shape `(Q, A)`.
        wavelet_packets: Optional dyadic wavelet-packet mother states of shape `(P, Q, A)`.
        packet_indices: Optional mapping from vocabulary token `v` to wavelet packet `p in [0, P)`.
        wavelet_mix: Mixing weight for the coarse wavelet-packet mother state.
    """
    q_f = base_qubits.float()
    if wavelet_packets is not None and packet_indices is not None and wavelet_mix > 0.0:
        idx = packet_indices[: q_f.shape[0]].to(device=q_f.device, dtype=torch.long)
        q_f = q_f + float(wavelet_mix) * wavelet_packets.float().index_select(0, idx)
    l_reg = build_lower_triangular_cholesky(raw_chol_tril.float(), raw_chol_diag.float())
    return torch.einsum("vqa,qab->vqb", q_f, l_reg).to(base_qubits.dtype)


def apply_soft_powerlaw_spectral_floor(weights: torch.Tensor, min_eff_rank: float = 6.5) -> torch.Tensor:
    """Ensure 1D Kraus weights satisfy inverse participation ratio `>= min_eff_rank`."""
    w = weights / weights.sum().clamp_min(1e-12)
    u = torch.arange(1, w.numel() + 1, device=w.device, dtype=w.dtype) ** -0.25
    u, out = u / u.sum().clamp_min(1e-12), w
    for _ in range(6):
        eff_r = 1.0 / (out * out).sum().clamp_min(1e-12)
        blended = (0.70 * out + 0.30 * u) / (0.70 * out + 0.30 * u).sum().clamp_min(1e-12)
        out = torch.where(eff_r >= float(min_eff_rank), out, blended)
    return out

