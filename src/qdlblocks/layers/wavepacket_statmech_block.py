r"""Native `(B, T, 48, 16)` Wavepacket Quantum Statistical-Mechanics Transformer Block.

Maintains the hidden state natively as a pure complex wavefunction tensor $\Psi^{(l)} \in \mathbb{C}^{B \times T \times 24 \times 16}$
(represented in real coordinates as $X^{(l)} \in \mathbb{R}^{B \times T \times 48 \times 16}$, i.e., $H=6$ heads $\times R=4$
complex modes $\times 2$ Re/Im $\times d_h=16$ Hilbert dim) across all $L$ layers with zero Hermitian matrix-to-vector packing:

1. **Factored `(M_in, M_out, 16)` Stiefel Operators (`apply_factored_wavepacket_stiefel`)**:
   - Per-packet $16 \times 16$ Stiefel rotation $W_{\text{pkt}} \in \mathrm{St}_{\text{col}}(16, 16)^{M}$ (shape `(M, 16, 16)`).
   - Cross-packet $(16, M_{\text{in}}, M_{\text{out}})$ Stiefel mode mixer $M_{\text{mode}} \in \mathrm{St}_{\text{col}}(M_{\text{in}}, M_{\text{out}})^{16}$.
2. **Outer-Product Wavefunction Causal Self-Attention on `(B, T, 48, 16)`**:
   - Applies multi-scale complex rotary phase $e^{i t \omega_{r,p}}$ across $(R=4, d_h=16)$ per head $h \in \{1, \dots, 6\}$.
   - Forms the $16 \times 16$ Hilbert-space outer product $\rho_{s,h}^{\text{tok}} = \sum_{r=1}^4 |\tilde{v}_{s,h,r}\rangle\langle\tilde{v}_{s,h,r}| \in \mathbb{C}^{16 \times 16}$
     of the $V$-space wavefunctions at each token $s$, and uses fused FlashAttention SDPA (`D=128`) to causally aggregate
     both $\rho_{t,h}^{(0)} = \sum_{s \le t} A_{t,s}^{(h)} \rho_{s,h}^{\text{tok}} \in \mathbb{C}^{16 \times 16}$ and $|\bar{v}_{t,h,r}\rangle = \sum_{s \le t} A_{t,s}^{(h)} |\tilde{v}_{s,h,r}\rangle \in \mathbb{C}^{16}$
     with zero packing masks, zero $\sqrt{2}$ scaling, and zero `triu`/`tril` unpacking.
3. **$16 \times 16$ Density Purification, Unitary/Kraus Evolution & Direct Operator Action on $V$-Space Wavefunctions**:
   - Purifies $\rho_{t,h}^{(0)} \mapsto \rho_{\text{pure}, t,h}$ via McWeeny/Fermi-Dirac cooling, evolves $\rho_{\text{pure}, t,h} \mapsto \rho_{\text{evol}, t,h}$
     via $U_h \in U(16)$ or CPTP Kraus channel, and applies the $16 \times 16$ density operator $\rho_{\text{evol}, t,h}$ directly
     to the $V$-space wavefunctions in $\mathbb{C}^{16}$, followed by native `(B, T, 48, 16)` two-beam Mach-Zehnder interference.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .impure_density_mirror_block import (
    DEFAULT_ROPE_BASE_THETA,
    DEFAULT_TIKHONOV_GRAM_LAMBDA,
    TRACE_NORM_EPS,
    initialize_stiefel_parameter,
)
from .isometric import apply_partial_key_offset
from .quantum_statmech_channels import (
    DEFAULT_KRAUS_CHANNEL_RANK,
    DEFAULT_TSALLIS_PURITY_CENTER,
    EntropyGateMode,
    PurificationMode,
    QuantumChannelMode,
    StiefelMomentumMode,
    evolve_head_cptp_kraus_functional,
    evolve_head_unitary_traceless_functional,
    initialize_complex_kraus_stiefel_pair,
    purify_density_statmech,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_WAVEPACKET_CHANNELS: int = 48
DEFAULT_WAVEPACKET_HILBERT_DIM: int = 16
DEFAULT_WAVEPACKET_COMPLEX_HEADS: int = 24
DEFAULT_WAVEPACKET_COMPLEX_REAL_DIM: int = 32
DEFAULT_WAVEPACKET_SDPA_HEAD_DIM: int = 128
DEFAULT_WAVEPACKET_SDPA_SCALE: float = 0.22
DEFAULT_WAVEPACKET_RESIDUAL_GAIN: float = 0.35
DEFAULT_WAVEPACKET_V0_LAMBDA: float = 0.15
DEFAULT_PPMI_ORTHO_SEED: int = 2026064
DEFAULT_HYBRID_STIEFEL_LR: float = 0.04
DEFAULT_HYBRID_SCALAR_LR: float = 0.04
DEFAULT_DM_RES_MIX_INIT: float = 0.25
DEFAULT_DM_LAG_MIX_INIT: float = 0.20
DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO: float = 0.25
_DEFAULT_PPMI_CACHE_PATH: Path = Path(__file__).resolve().parents[3] / ".cache" / "ppmi_svd_50304_768_shard1.pt"


def initialize_packet_stiefel_3d(weight_3d: torch.Tensor) -> None:
    """Initialize a 3D `(M, d, d)` per-packet Stiefel tensor as independent orthogonal matrices."""
    with torch.no_grad():
        for m_idx in range(weight_3d.size(0)):
            initialize_stiefel_parameter(weight_3d[m_idx])


def initialize_wavepacket_ppmi_sphere_embeddings_(
    wte_3d: torch.Tensor,
    ppmi_embeddings: torch.Tensor | None = None,
) -> bool:
    r"""Initialize `(V, M, d_h)` wavepacket embeddings via Spectral-Interleaved + Ortho-$M$ Balanced PPMI-SVD on $(\sqrt{d_h}\,\mathbb{S}^{d_h - 1})^M$."""
    from ..optim.vocab_manifold_muon import project_rows_to_rms_sphere_

    vocab_size, num_pkts, d_h = wte_3d.shape
    embed_dim = num_pkts * d_h
    if ppmi_embeddings is None and vocab_size == 50304 and embed_dim == 768 and _DEFAULT_PPMI_CACHE_PATH.exists():
        raw_payload = torch.load(_DEFAULT_PPMI_CACHE_PATH, map_location="cpu", weights_only=True)
        ppmi_embeddings = raw_payload["embeddings"] if isinstance(raw_payload, dict) else raw_payload
    with torch.no_grad():
        if ppmi_embeddings is not None and tuple(ppmi_embeddings.shape) == (vocab_size, embed_dim):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(DEFAULT_PPMI_ORTHO_SEED)
            q_m, r_m = torch.linalg.qr(torch.randn(num_pkts, num_pkts, generator=gen, dtype=torch.float32))
            diag_sign = torch.sign(torch.diagonal(r_m))
            diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
            q_m = q_m * diag_sign.unsqueeze(0)
            ppmi_f = ppmi_embeddings.detach().to(device="cpu", dtype=torch.float32)
            balanced_3d = torch.matmul(ppmi_f.view(vocab_size, d_h, num_pkts), q_m).transpose(1, 2).contiguous()
            project_rows_to_rms_sphere_(balanced_3d)
            wte_3d.copy_(balanced_3d.to(device=wte_3d.device, dtype=wte_3d.dtype))
            return True
        project_rows_to_rms_sphere_(wte_3d)
        return False


def _kets_to_hermitian_outer(vn_re: torch.Tensor, vn_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `(rho_re, rho_im)` `(B, T, H, d_h, d_h)` from normalized kets `(B, T, H, R, d_h)`."""
    rho_re = torch.matmul(vn_re.transpose(-2, -1), vn_re) + torch.matmul(vn_im.transpose(-2, -1), vn_im)
    rho_im = torch.matmul(vn_im.transpose(-2, -1), vn_re) - torch.matmul(vn_re.transpose(-2, -1), vn_im)
    return rho_re, rho_im


def _normalize_mode_kets_btmd(
    z_btmd: torch.Tensor, num_heads: int, v_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize `(B, T, M_pkt, d_h)` into `(zr, zi)` of shape `(B, T, H, R, d_h)` with unit per-head norm."""
    bsz, seq_len, _, d_h = z_btmd.shape
    zc = z_btmd.view(bsz, seq_len, num_heads, v_rank, 2, d_h)
    zr, zi = zc[..., 0, :], zc[..., 1, :]
    nsq = (zr.square().sum(dim=-1, keepdim=True) + zi.square().sum(dim=-1, keepdim=True)).clamp_min(TRACE_NORM_EPS)
    inv_n = torch.rsqrt(nsq * float(v_rank))
    return zr * inv_n, zi * inv_n


def form_wavepacket_outer_product_and_v(
    v_btmd: torch.Tensor,
    num_heads: int = 6,
    v_rank: int = 4,
) -> torch.Tensor:
    r"""Form $d_h \times d_h$ outer products $\rho_{s,h}^{\text{tok}} = \sum_{r=1}^R |\tilde{v}_{s,h,r}\rangle\langle\tilde{v}_{s,h,r}|$ (`k_rho x 128`D) + $|\tilde{v}_{s,h,r}\rangle$ (`1x128`D) with zero packing."""
    bsz, seq_len, _, d_h = v_btmd.shape
    vn_re, vn_im = _normalize_mode_kets_btmd(v_btmd, num_heads, v_rank)
    rho_tok_re, rho_tok_im = _kets_to_hermitian_outer(vn_re, vn_im)
    k_rho = max(1, d_h // v_rank)
    row_chunk = v_rank
    sdpa_dim = 2 * row_chunk * d_h
    sdpa_rho = torch.stack(
        [
            rho_tok_re.view(bsz, seq_len, num_heads, k_rho, row_chunk, d_h),
            rho_tok_im.view(bsz, seq_len, num_heads, k_rho, row_chunk, d_h),
        ],
        dim=4,
    ).reshape(bsz, seq_len, num_heads, k_rho, sdpa_dim)
    sdpa_ket = torch.stack([vn_re, vn_im], dim=4).reshape(bsz, seq_len, num_heads, 1, sdpa_dim)
    return torch.cat([sdpa_rho, sdpa_ket], dim=3).reshape(bsz, seq_len, num_heads * (k_rho + 1), sdpa_dim)


def build_configurable_rho_and_ket_sdpa_chunks(
    v_wp: torch.Tensor,
    h_norm: torch.Tensor,
    num_heads: int,
    v_rank: int,
    dm_res_mix: torch.Tensor | None = None,
    dm_lag_mix: torch.Tensor | None = None,
    use_causal_ensemble: bool = False,
    use_h_norm_for_rho: bool = False,
) -> torch.Tensor:
    r"""Build `(B, H * (k_rho + 1), T, 128)` SDPA value chunks for $d_h \in \{8, 16, 32\}$ and `CAUSAL_ENSEMBLE_DM_IO`."""
    bsz, seq_len, _, d_h = v_wp.shape
    k_rho = max(1, d_h // v_rank)
    row_chunk = v_rank
    sdpa_dim = 2 * row_chunk * d_h

    vn_re, vn_im = _normalize_mode_kets_btmd(v_wp, num_heads, v_rank)
    if use_causal_ensemble:
        hn_re, hn_im = _normalize_mode_kets_btmd(h_norm, num_heads, v_rank)
        rv_re, rv_im = _kets_to_hermitian_outer(vn_re, vn_im)
        r0_re, r0_im = _kets_to_hermitian_outer(hn_re, hn_im)
        w0 = dm_res_mix.abs().to(v_wp.dtype) if dm_res_mix is not None else torch.tensor(0.25, dtype=v_wp.dtype, device=v_wp.device)
        w_lag = dm_lag_mix.abs().to(v_wp.dtype) if dm_lag_mix is not None else torch.tensor(0.20, dtype=v_wp.dtype, device=v_wp.device)
        if seq_len > 1:
            lag_re = torch.cat([torch.zeros_like(vn_re[:, :1]), vn_re[:, :-1]], dim=1)
            lag_im = torch.cat([torch.zeros_like(vn_im[:, :1]), vn_im[:, :-1]], dim=1)
            rl_re, rl_im = _kets_to_hermitian_outer(lag_re, lag_im)
            mask_t = torch.ones((1, seq_len, 1, 1, 1), dtype=v_wp.dtype, device=v_wp.device)
            mask_t[:, 0] = 0.0
            z_den = 1.0 + w0 + mask_t * w_lag
            rho_tok_re = (rv_re + w0 * r0_re + (mask_t * w_lag) * rl_re) / z_den
            rho_tok_im = (rv_im + w0 * r0_im + (mask_t * w_lag) * rl_im) / z_den
        else:
            z_den = 1.0 + w0
            rho_tok_re, rho_tok_im = (rv_re + w0 * r0_re) / z_den, (rv_im + w0 * r0_im) / z_den
    elif use_h_norm_for_rho:
        hn_re, hn_im = _normalize_mode_kets_btmd(h_norm, num_heads, v_rank)
        rho_tok_re, rho_tok_im = _kets_to_hermitian_outer(hn_re, hn_im)
    else:
        rho_tok_re, rho_tok_im = _kets_to_hermitian_outer(vn_re, vn_im)

    sdpa_rho = torch.stack(
        [
            rho_tok_re.view(bsz, seq_len, num_heads, k_rho, row_chunk, d_h),
            rho_tok_im.view(bsz, seq_len, num_heads, k_rho, row_chunk, d_h),
        ],
        dim=4,
    ).reshape(bsz, seq_len, num_heads, k_rho, sdpa_dim)
    sdpa_ket = torch.stack([vn_re, vn_im], dim=4).reshape(bsz, seq_len, num_heads, 1, sdpa_dim)
    return (
        torch.cat([sdpa_rho, sdpa_ket], dim=3)
        .reshape(bsz, seq_len, num_heads * (k_rho + 1), sdpa_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def apply_factored_wavepacket_stiefel(
    x_btmd: torch.Tensor,
    w_pkt_mdd: torch.Tensor,
    m_mode: torch.Tensor,
) -> torch.Tensor:
    r"""Apply per-packet $16 \times 16$ Stiefel rotation followed by per-coordinate `(16, M_in, M_out)` cross-packet Stiefel mixing."""
    bsz, seq_len, num_pkts_in, d_h = x_btmd.shape
    bt = bsz * seq_len
    z_mbtd = torch.bmm(x_btmd.permute(2, 0, 1, 3).reshape(num_pkts_in, bt, d_h), w_pkt_mdd)
    if m_mode.ndim == 3:
        y_dbtn = torch.bmm(z_mbtd.permute(2, 1, 0).reshape(d_h, bt, num_pkts_in), m_mode)
        return y_dbtn.view(d_h, bsz, seq_len, -1).permute(1, 2, 3, 0).contiguous()
    z_btdm = z_mbtd.view(num_pkts_in, bsz, seq_len, d_h).permute(1, 2, 3, 0)
    return torch.matmul(z_btdm, m_mode).transpose(-2, -1).contiguous()


def apply_causal_wavepacket_dispersion(
    raw_wp: torch.Tensor,
    wp_causal_re: torch.Tensor,
    wp_causal_im: torch.Tensor,
) -> torch.Tensor:
    """Apply multi-lag causal complex wavepacket dispersion across $(H, R, 2, d_h)$ modes."""
    bsz, seq_len, num_pkts, d_h = raw_wp.shape
    wp_complex_modes = max(1, num_pkts // 2)
    cur_wp = raw_wp.view(bsz, seq_len, wp_complex_modes, 2, d_h)
    if wp_complex_modes % 4 == 0:
        n_groups = wp_complex_modes // 4
        wp_c = raw_wp.view(bsz, seq_len, n_groups, 4, 2, d_h)
        ret_01 = F.pad(wp_c[:, :-1, :, :2], (0, 0, 0, 0, 0, 0, 0, 0, 1, 0))
        ret_2 = (
            F.pad(wp_c[:, :-2, :, 2:3], (0, 0, 0, 0, 0, 0, 0, 0, 2, 0))
            if seq_len > 2
            else torch.zeros_like(wp_c[:, :, :, 2:3])
        )
        ret_3 = (
            F.pad(wp_c[:, :-3, :, 3:4], (0, 0, 0, 0, 0, 0, 0, 0, 3, 0))
            if seq_len > 3
            else torch.zeros_like(wp_c[:, :, :, 3:4])
        )
        ret_wp = torch.cat([ret_01, ret_2, ret_3], dim=3).view(bsz, seq_len, wp_complex_modes, 2, d_h)
    else:
        lag1 = F.pad(cur_wp[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        lag2 = F.pad(cur_wp[:, :-2], (0, 0, 0, 0, 0, 0, 2, 0)) if seq_len > 2 else torch.zeros_like(cur_wp)
        lag3 = F.pad(cur_wp[:, :-3], (0, 0, 0, 0, 0, 0, 3, 0)) if seq_len > 3 else torch.zeros_like(cur_wp)
        m_mod = (torch.arange(wp_complex_modes, device=raw_wp.device) % 4).view(1, 1, wp_complex_modes, 1, 1)
        ret_wp = torch.where(m_mod < 2, lag1, torch.where(m_mod == 2, lag2, lag3))
    g_re = torch.tanh(wp_causal_re).to(raw_wp.dtype).view(1, 1, wp_complex_modes, 1)
    g_im = torch.tanh(wp_causal_im).to(raw_wp.dtype).view(1, 1, wp_complex_modes, 1)
    out_re = cur_wp[..., 0, :] + g_re * ret_wp[..., 0, :] - g_im * ret_wp[..., 1, :]
    out_im = cur_wp[..., 1, :] + g_re * ret_wp[..., 1, :] + g_im * ret_wp[..., 0, :]
    return torch.stack([out_re, out_im], dim=-2).reshape(bsz, seq_len, num_pkts, d_h)


def wavepacket_statmech_purify_evolve_impl(
    a_all: torch.Tensor,
    op_re: torch.Tensor,
    op_im: torch.Tensor,
    purify_logits_h3: torch.Tensor,
    entropy_gate_params: torch.Tensor,
    modes: tuple[str, str, str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Purify $16 \times 16$ outer-product density matrices and multiply $V$-space wavefunctions directly (`<= 8` args)."""
    work_dtype = a_all.dtype
    num_heads = op_re.size(0)
    d_h = op_re.size(-1)
    purify_mode = PurificationMode(modes[0])
    channel_mode = QuantumChannelMode(modes[1])
    gate_mode = EntropyGateMode(modes[2])

    if a_all.ndim == 5:
        bsz, _, chunks, seq_len, _ = a_all.shape
        k_rho = max(1, chunks - 1)
        row_chunk = max(1, d_h // k_rho)
        v_rank = row_chunk
        num_pkts = 2 * num_heads * v_rank
        a_rho = a_all[:, :, :k_rho].permute(0, 1, 3, 2, 4).reshape(bsz, num_heads, seq_len, k_rho, 2, row_chunk, d_h)
        rho0_re = a_rho[:, :, :, :, 0].reshape(bsz, num_heads, seq_len, d_h, d_h)
        rho0_im = a_rho[:, :, :, :, 1].reshape(bsz, num_heads, seq_len, d_h, d_h)
        a_v = a_all[:, :, k_rho].view(bsz, num_heads, seq_len, v_rank, 2, d_h)
        v_bar_re, v_bar_im = a_v[..., 0, :], a_v[..., 1, :]
        v_norm_sq = (v_bar_re.square().sum(dim=-1, keepdim=True) + v_bar_im.square().sum(dim=-1, keepdim=True)).clamp_min(
            TRACE_NORM_EPS
        )
        inv_vn = torch.rsqrt(v_norm_sq * float(v_rank))
        p_re, p_im = v_bar_re * inv_vn, v_bar_im * inv_vn
    else:
        bsz, seq_len, num_pkts, _ = a_all.shape
        v_rank = max(1, num_pkts // (2 * num_heads))
        psi = a_all.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
        psi_re, psi_im = psi[..., 0, :], psi[..., 1, :]
        v_norm_sq = (psi_re.square().sum(dim=-1, keepdim=True) + psi_im.square().sum(dim=-1, keepdim=True)).clamp_min(
            TRACE_NORM_EPS
        )
        inv_vn = torch.rsqrt(v_norm_sq * float(v_rank))
        p_re, p_im = psi_re * inv_vn, psi_im * inv_vn
        rho0_re = torch.matmul(p_re.transpose(-2, -1), p_re) + torch.matmul(p_im.transpose(-2, -1), p_im)
        rho0_im = torch.matmul(p_im.transpose(-2, -1), p_re) - torch.matmul(p_re.transpose(-2, -1), p_im)

    pure_re, pure_im, purity, eff_rank = purify_density_statmech(
        rho0_re, rho0_im, mode=purify_mode, purify_logits_h3=purify_logits_h3, eps=TRACE_NORM_EPS
    )

    if channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K and op_re.size(-2) > d_h:
        evol_re, evol_im = evolve_head_cptp_kraus_functional(pure_re, pure_im, op_re, op_im)
        u0_re = op_re[:, :d_h, :].to(work_dtype).view(1, num_heads, 1, d_h, d_h)
        u0_im = op_im[:, :d_h, :].to(work_dtype).view(1, num_heads, 1, d_h, d_h)
    else:
        evol_re, evol_im = evolve_head_unitary_traceless_functional(pure_re, pure_im, op_re, op_im)
        u0_re = op_re.to(work_dtype).view(1, num_heads, 1, d_h, d_h)
        u0_im = op_im.to(work_dtype).view(1, num_heads, 1, d_h, d_h)

    vu_re = torch.matmul(p_re, u0_re.transpose(-2, -1)) - torch.matmul(p_im, u0_im.transpose(-2, -1))
    vu_im = torch.matmul(p_re, u0_im.transpose(-2, -1)) + torch.matmul(p_im, u0_re.transpose(-2, -1))
    wp_u_re = vu_re + 0.5 * (torch.matmul(vu_re, evol_re) + torch.matmul(vu_im, evol_im))
    wp_u_im = vu_im + 0.5 * (torch.matmul(vu_im, evol_re) - torch.matmul(vu_re, evol_im))
    y_wp_48 = F.rms_norm(
        torch.stack([wp_u_re, wp_u_im], dim=-2).permute(0, 2, 1, 3, 4, 5).reshape(bsz, seq_len, num_pkts, d_h),
        (d_h,),
        eps=TRACE_NORM_EPS,
    )

    if gate_mode in (
        EntropyGateMode.DETACHED_TSALLIS_PURITY_GATE,
        EntropyGateMode.ISOMETRIC_NORMALIZED_PLUS_TSALLIS,
    ):
        p_det = purity.detach().to(work_dtype).transpose(1, 2).unsqueeze(-1)
        s2_det = (1.0 - p_det) - DEFAULT_TSALLIS_PURITY_CENTER
        gp = entropy_gate_params.to(work_dtype).view(1, 1, num_heads, 3)
        g_wp = 2.0 * torch.sigmoid(
            gp[..., 0:1] * (p_det - DEFAULT_TSALLIS_PURITY_CENTER) + gp[..., 1:2] * s2_det + gp[..., 2:3]
        )
        y_wp_48 = (y_wp_48.view(bsz, seq_len, num_heads, -1, d_h) * g_wp.unsqueeze(-1)).reshape(
            bsz, seq_len, num_pkts, d_h
        )

    diag_evol = torch.diagonal(evol_re, dim1=-2, dim2=-1)
    lam1 = diag_evol.max(dim=-1).values.clamp(0.0, 1.0)
    lam2 = (1.0 - lam1).clamp(0.0, 1.0)
    return y_wp_48, purity, eff_rank, lam1, lam2


_COMPILED_WP_STATMECH_HELPERS: dict[tuple[str, str, str], Any] = {}


def compiled_wavepacket_statmech_step(
    a_all: torch.Tensor,
    op_re: torch.Tensor,
    op_im: torch.Tensor,
    purify_logits_h3: torch.Tensor,
    entropy_gate_params: torch.Tensor,
    modes: tuple[str, str, str, bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute `wavepacket_statmech_purify_evolve_impl` with mode-specialized `torch.compile` caching (`<= 8` args)."""
    mode_key = (str(modes[0]), str(modes[1]), str(modes[2]))
    use_compile = bool(modes[3])
    if use_compile and a_all.is_cuda:
        if mode_key not in _COMPILED_WP_STATMECH_HELPERS:
            _COMPILED_WP_STATMECH_HELPERS[mode_key] = torch.compile(
                wavepacket_statmech_purify_evolve_impl, dynamic=False
            )
        return _COMPILED_WP_STATMECH_HELPERS[mode_key](
            a_all, op_re, op_im, purify_logits_h3, entropy_gate_params, mode_key
        )
    return wavepacket_statmech_purify_evolve_impl(
        a_all, op_re, op_im, purify_logits_h3, entropy_gate_params, mode_key
    )


class NativeWavepacketStatMechBlock(nn.Module):
    """Native `(B, T, 48, 16)` Quantum Wavepacket Self-Attention, Density Evolution & Interference Layer."""

    def __init__(
        self,
        cfg: Any,
        layer_idx: int = 0,
        kraus_rank: int = DEFAULT_KRAUS_CHANNEL_RANK,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.embed_dim = int(cfg.n_embd)
        self.num_heads = int(cfg.n_head)
        self.d_h = DEFAULT_WAVEPACKET_HILBERT_DIM
        self.num_pkts = max(1, self.embed_dim // self.d_h)
        self.complex_heads = max(1, self.num_pkts // 2)
        self.v_rank = max(1, self.complex_heads // self.num_heads)
        self.beam_pkts = 2 * self.num_pkts
        self.kraus_rank = max(1, int(kraus_rank))
        self.is_long_window_layer = (self.layer_idx % 4 == 3) or (self.layer_idx == cfg.n_layer - 1)
        self.use_compiled_krylov: bool = False

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))
        v0_init = DEFAULT_WAVEPACKET_V0_LAMBDA if self.layer_idx > 0 else 0.0
        self.v_lambdas = nn.Parameter(torch.tensor([1.0, v0_init], dtype=torch.float32))

        self.w_q_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_h, self.d_h))
        self.w_k_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_h, self.d_h))
        self.w_v_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_h, self.d_h))
        self.w_obs_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_h, self.d_h))
        self.w_beam_a_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_h, self.d_h))
        self.w_beam_b_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_h, self.d_h))
        self.w_chan_pkt = nn.Parameter(torch.empty(self.beam_pkts, self.d_h, self.d_h))
        for w_3d in (
            self.w_q_pkt,
            self.w_k_pkt,
            self.w_v_pkt,
            self.w_obs_pkt,
            self.w_beam_a_pkt,
            self.w_beam_b_pkt,
            self.w_chan_pkt,
        ):
            initialize_packet_stiefel_3d(w_3d)

        self.m_q_mode = nn.Parameter(torch.empty(self.d_h, self.num_pkts, self.num_pkts))
        self.m_k_mode = nn.Parameter(torch.empty(self.d_h, self.num_pkts, self.num_pkts))
        self.m_v_mode = nn.Parameter(torch.empty(self.d_h, self.num_pkts, self.num_pkts))
        self.m_obs_mode = nn.Parameter(torch.empty(self.d_h, self.num_pkts, self.num_pkts))
        self.m_beam_a_mode = nn.Parameter(torch.empty(self.d_h, self.num_pkts, self.beam_pkts))
        self.m_beam_b_mode = nn.Parameter(torch.empty(self.d_h, self.num_pkts, self.beam_pkts))
        self.m_chan_mode = nn.Parameter(torch.empty(self.d_h, self.beam_pkts, self.num_pkts))
        for m_3d in (
            self.m_q_mode,
            self.m_k_mode,
            self.m_v_mode,
            self.m_obs_mode,
            self.m_beam_a_mode,
            self.m_beam_b_mode,
            self.m_chan_mode,
        ):
            initialize_packet_stiefel_3d(m_3d)

        if cfg.channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K:
            kr_re, kr_im = initialize_complex_kraus_stiefel_pair(self.num_heads, self.d_h, kraus_rank=self.kraus_rank)
            self.head_unitary_re = nn.Parameter(kr_re)
            self.head_unitary_im = nn.Parameter(kr_im)
        else:
            phase_angles = torch.linspace(-0.25, 0.25, self.d_h, dtype=torch.float32)
            diag_re = torch.diag(torch.cos(phase_angles)).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
            diag_im = torch.diag(torch.sin(phase_angles)).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
            self.head_unitary_re = nn.Parameter(diag_re)
            self.head_unitary_im = nn.Parameter(diag_im)

        init_gain = DEFAULT_WAVEPACKET_RESIDUAL_GAIN / math.sqrt(float(max(1, cfg.n_layer)))
        self.obs_scale = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))
        self.chan_scale = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))

        init_purify = torch.tensor([-2.0, 2.0, 0.0], dtype=torch.float32).unsqueeze(0).expand(self.num_heads, -1).clone()
        self.purify_logits = nn.Parameter(
            init_purify, requires_grad=(cfg.purification_mode == PurificationMode.LEARNABLE_TEMP_MCWEENY)
        )
        self.entropy_gate_params = nn.Parameter(
            torch.zeros(self.num_heads, 3, dtype=torch.float32),
            requires_grad=(
                cfg.entropy_gate_mode
                in (EntropyGateMode.DETACHED_TSALLIS_PURITY_GATE, EntropyGateMode.ISOMETRIC_NORMALIZED_PLUS_TSALLIS)
            ),
        )

        qk_complex_dim = self.v_rank * self.d_h
        inv_freq = 1.0 / (
            DEFAULT_ROPE_BASE_THETA ** (torch.arange(0, qk_complex_dim, dtype=torch.float32) / float(qk_complex_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq.view(1, 1, 1, self.v_rank, self.d_h), persistent=False)
        self._last_telemetry_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def w_obs(self) -> nn.Parameter:
        """Expose final observable mode mixer for Level B unnormalized compatibility."""
        return self.m_obs_mode

    @property
    def w_chan(self) -> nn.Parameter:
        """Expose final channel mode mixer for Level B unnormalized compatibility."""
        return self.m_chan_mode

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all 14 factored per-packet 3D and cross-packet 2D real Stiefel parameters in this layer."""
        return [
            self.w_q_pkt,
            self.w_k_pkt,
            self.w_v_pkt,
            self.w_obs_pkt,
            self.w_beam_a_pkt,
            self.w_beam_b_pkt,
            self.w_chan_pkt,
            self.m_q_mode,
            self.m_k_mode,
            self.m_v_mode,
            self.m_obs_mode,
            self.m_beam_a_mode,
            self.m_beam_b_mode,
            self.m_chan_mode,
        ]

    def complex_unitary_pair(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return the 3D complex head operator pair `(head_unitary_re, head_unitary_im)`."""
        return self.head_unitary_re, self.head_unitary_im

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return 1D scalar residual gains, highway weights, and optional purification/gate parameters."""
        params = [self.obs_scale, self.chan_scale, self.block_lambdas, self.v_lambdas]
        if self.purify_logits.requires_grad:
            params.append(self.purify_logits)
        if self.entropy_gate_params.requires_grad:
            params.append(self.entropy_gate_params)
        return params

    def get_last_telemetry_dict(self) -> dict[str, float]:
        """Convert detached GPU telemetry tensors to Python floats only when requested at logging steps."""
        if self._last_telemetry_tensors is None:
            return {"mean_purity": 1.0, "mean_eff_rank": 1.0, "mean_lambda1": 0.5, "mean_lambda2": 0.5}
        p_t, er_t, l1_t, l2_t = self._last_telemetry_tensors
        return {
            "mean_purity": float(p_t.item()),
            "mean_eff_rank": float(er_t.item()),
            "mean_lambda1": float(l1_t.item()),
            "mean_lambda2": float(l2_t.item()),
        }

    def _apply_wavepacket_complex_rope(self, z_btmd: torch.Tensor) -> torch.Tensor:
        r"""Apply multi-scale complex rotary phase $e^{i t \omega_{r,p}}$ to `(B, T, 48, 16)` viewed as `(B, T, 6, 4, 2, 16)`."""
        bsz, seq_len, _, _ = z_btmd.shape
        z_c = z_btmd.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h)
        z_re, z_im = z_c[..., 0, :], z_c[..., 1, :]
        pos = torch.arange(seq_len, device=z_btmd.device, dtype=torch.float32).view(1, seq_len, 1, 1, 1)
        angles = pos * self.rope_inv_freq
        cos_a = torch.cos(angles).to(z_btmd.dtype)
        sin_a = torch.sin(angles).to(z_btmd.dtype)
        rot_re = z_re * cos_a - z_im * sin_a
        rot_im = z_re * sin_a + z_im * cos_a
        return torch.stack([rot_re, rot_im], dim=-2).reshape(bsz, seq_len, self.num_pkts, self.d_h)

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass updating `x` `(B, T, 48, 16) -> x_out (B, T, 48, 16)` and returning `(x_out, v_0_next, krylov_weights_bt2)`."""
        bsz, seq_len, _, _ = x.shape
        work_dtype = x.dtype
        if x_0 is not None:
            x_cur = self.block_lambdas[0].to(work_dtype) * x + self.block_lambdas[1].to(work_dtype) * x_0.to(work_dtype)
        else:
            x_cur = x

        h_norm = F.rms_norm(x_cur, (self.d_h,), eps=TRACE_NORM_EPS)
        q_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_q_pkt.to(work_dtype), self.m_q_mode.to(work_dtype))
        k_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_k_pkt.to(work_dtype), self.m_k_mode.to(work_dtype))
        v_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_v_pkt.to(work_dtype), self.m_v_mode.to(work_dtype))

        if v_0 is None:
            v_0_next = v_wp
            v_mixed = v_wp
        else:
            v_0_next = v_0
            v_mixed = self.v_lambdas[0].to(work_dtype) * v_wp + self.v_lambdas[1].to(work_dtype) * v_0

        q_rot = self._apply_wavepacket_complex_rope(q_wp)
        k_rot = self._apply_wavepacket_complex_rope(k_wp)

        sdpa_dim = 2 * self.v_rank * self.d_h
        q_h = F.rms_norm(q_rot.view(bsz, seq_len, self.num_heads, sdpa_dim), (sdpa_dim,), eps=TRACE_NORM_EPS)
        k_h = F.rms_norm(k_rot.view(bsz, seq_len, self.num_heads, sdpa_dim), (sdpa_dim,), eps=TRACE_NORM_EPS)
        if self.is_long_window_layer and seq_len > 1:
            k_h = apply_partial_key_offset(k_h)

        chunks = self.v_rank + 1
        sdpa_q = (
            q_h.permute(0, 2, 1, 3)
            .unsqueeze(2)
            .expand(bsz, self.num_heads, chunks, seq_len, sdpa_dim)
            .reshape(bsz, self.num_heads * chunks, seq_len, sdpa_dim)
        )
        sdpa_k = (
            k_h.permute(0, 2, 1, 3)
            .unsqueeze(2)
            .expand(bsz, self.num_heads, chunks, seq_len, sdpa_dim)
            .reshape(bsz, self.num_heads * chunks, seq_len, sdpa_dim)
        )
        sdpa_v = (
            form_wavepacket_outer_product_and_v(v_mixed, num_heads=self.num_heads, v_rank=self.v_rank)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        a_all = F.scaled_dot_product_attention(
            sdpa_q,
            sdpa_k,
            sdpa_v,
            is_causal=True,
            scale=DEFAULT_WAVEPACKET_SDPA_SCALE,
        ).view(bsz, self.num_heads, chunks, seq_len, sdpa_dim)

        y_wp_48, purity, eff_rank, lam1, lam2 = compiled_wavepacket_statmech_step(
            a_all,
            self.head_unitary_re,
            self.head_unitary_im,
            self.purify_logits,
            self.entropy_gate_params,
            (
                self.cfg.purification_mode.value,
                self.cfg.channel_mode.value,
                self.cfg.entropy_gate_mode.value,
                self.use_compiled_krylov,
            ),
        )

        with torch.no_grad():
            self._last_telemetry_tensors = (
                purity.detach().mean(),
                eff_rank.detach().mean(),
                lam1.detach().mean(),
                lam2.detach().mean(),
            )

        lam1_bar, lam2_bar = lam1.mean(dim=1), lam2.mean(dim=1)
        lam_sum = (lam1_bar + lam2_bar).clamp_min(1e-6)
        krylov_weights_bt2 = torch.stack([lam1_bar / lam_sum, lam2_bar / lam_sum], dim=-1)

        d_obs = apply_factored_wavepacket_stiefel(y_wp_48, self.w_obs_pkt.to(work_dtype), self.m_obs_mode.to(work_dtype))
        x_mid = x_cur + self.obs_scale.to(work_dtype) * d_obs

        m_norm = F.rms_norm(x_mid, (self.d_h,), eps=TRACE_NORM_EPS)
        beam_a = apply_factored_wavepacket_stiefel(
            m_norm, self.w_beam_a_pkt.to(work_dtype), self.m_beam_a_mode.to(work_dtype)
        )
        beam_b = apply_factored_wavepacket_stiefel(
            m_norm, self.w_beam_b_pkt.to(work_dtype), self.m_beam_b_mode.to(work_dtype)
        )
        mz_wp = F.relu(beam_a + beam_b).square() - 0.25 * F.relu(beam_a - beam_b).square()
        if self.cfg.entropy_gate_mode in (
            EntropyGateMode.ISOMETRIC_NORMALIZED_BRANCHES,
            EntropyGateMode.ISOMETRIC_NORMALIZED_PLUS_TSALLIS,
        ):
            mz_wp = F.rms_norm(mz_wp, (self.d_h,), eps=TRACE_NORM_EPS)
        d_chan = apply_factored_wavepacket_stiefel(mz_wp, self.w_chan_pkt.to(work_dtype), self.m_chan_mode.to(work_dtype))
        x_out = x_mid + self.chan_scale.to(work_dtype) * d_chan
        return x_out, v_0_next, krylov_weights_bt2


class ClassicalQuantumFrontEndMode(str, Enum):
    """Strategy enum selecting pure classical vs. quantum embeddings and/or Quantum Layer 0 (Rule 3)."""

    PURE_CLASSICAL = "pure_classical"
    QUANTUM_WAVEPACKET_EMB_ONLY = "quantum_wavepacket_emb_only"
    NATIVE_WAVEPACKET_EMB_AND_L0 = "native_wavepacket_emb_and_l0"
    FLAT_STATMECH_EMB_AND_L0 = "flat_statmech_emb_and_l0"


class HybridQuantumFrontEndClassicalGPT(nn.Module):
    """Classical Modded-NanoGPT benchmark with replaceable Quantum Wavepacket Embeddings & Quantum Layer 0."""

    def __init__(
        self,
        classical_cfg: Any,
        frontend_mode: ClassicalQuantumFrontEndMode = ClassicalQuantumFrontEndMode.PURE_CLASSICAL,
    ) -> None:
        super().__init__()
        from ..models.compiled_leaderboard_classical_gpt import CompiledLeaderboardClassicalGPT
        from ..models.modular_quantum_statmech_gpt import ModularQuantumStatMechBlock, ModularQuantumStatMechConfig
        from ..optim.vocab_manifold_muon import project_rows_to_rms_sphere_

        self.frontend_mode = ClassicalQuantumFrontEndMode(frontend_mode)
        self.classical = CompiledLeaderboardClassicalGPT(classical_cfg)
        self.cfg = self.classical.cfg
        self.vocab_size = self.classical.vocab_size
        self.embed_dim = self.classical.embed_dim
        self.num_heads = self.classical.num_heads
        self.num_layers = self.classical.num_layers
        self.d_h = DEFAULT_WAVEPACKET_HILBERT_DIM
        self.wp_channels = max(1, self.embed_dim // self.d_h)
        self.lm_head = self.classical.lm_head

        self.q_wte: nn.Parameter | None = None
        self.w_in_pkt: nn.Parameter | None = None
        self.w_in_mode: nn.Parameter | None = None
        self.w_in_stiefel: nn.Parameter | None = None
        self.in_scale: nn.Parameter | None = None
        self.wp_causal_re: nn.Parameter | None = None
        self.wp_causal_im: nn.Parameter | None = None
        self.quantum_block0: nn.Module | None = None
        self._compiled_tail_fn: Any = None

        if self.frontend_mode != ClassicalQuantumFrontEndMode.PURE_CLASSICAL:
            self.classical.wte.weight.requires_grad_(False)
            if self.classical.smear_gate is not None:
                self.classical.smear_gate.weight.requires_grad_(False)
            if self.classical.smear_lambda is not None:
                self.classical.smear_lambda.requires_grad_(False)

            q_wte = torch.randn(self.vocab_size, self.wp_channels, self.d_h, dtype=torch.float32)
            initialize_wavepacket_ppmi_sphere_embeddings_(q_wte)
            self.q_wte = nn.Parameter(q_wte)
            self.in_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            wp_complex_modes = max(1, self.wp_channels // 2)
            n_groups = max(1, wp_complex_modes // 4)
            init_wp_re = torch.tensor([0.15, 0.10, 0.08, 0.05], dtype=torch.float32).repeat(n_groups)[:wp_complex_modes]
            init_wp_im = torch.tensor([0.00, 0.10, -0.08, 0.05], dtype=torch.float32).repeat(n_groups)[:wp_complex_modes]
            self.wp_causal_re = nn.Parameter(init_wp_re)
            self.wp_causal_im = nn.Parameter(init_wp_im)

            if self.frontend_mode in (
                ClassicalQuantumFrontEndMode.QUANTUM_WAVEPACKET_EMB_ONLY,
                ClassicalQuantumFrontEndMode.NATIVE_WAVEPACKET_EMB_AND_L0,
            ):
                eye_pkt = torch.eye(self.d_h, dtype=torch.float32).unsqueeze(0).expand(self.wp_channels, -1, -1).clone()
                eye_mode = torch.eye(self.wp_channels, dtype=torch.float32).unsqueeze(0).expand(self.d_h, -1, -1).clone()
                self.w_in_pkt = nn.Parameter(eye_pkt)
                self.w_in_mode = nn.Parameter(eye_mode)
            else:
                self.w_in_stiefel = nn.Parameter(torch.eye(self.embed_dim, dtype=torch.float32))

            q_cfg = ModularQuantumStatMechConfig(
                vocab_size=self.vocab_size,
                n_layer=self.num_layers,
                n_head=self.num_heads,
                n_embd=self.embed_dim,
            )
            if self.frontend_mode == ClassicalQuantumFrontEndMode.NATIVE_WAVEPACKET_EMB_AND_L0:
                self.quantum_block0 = NativeWavepacketStatMechBlock(q_cfg, layer_idx=0)
                self.quantum_block0.use_compiled_krylov = True
                for p in self.classical.blocks[0].parameters():
                    p.requires_grad_(False)
            elif self.frontend_mode == ClassicalQuantumFrontEndMode.FLAT_STATMECH_EMB_AND_L0:
                self.quantum_block0 = ModularQuantumStatMechBlock(q_cfg, layer_idx=0)
                self.quantum_block0.use_compiled_krylov = True
                for p in self.classical.blocks[0].parameters():
                    p.requires_grad_(False)

    def enable_compiled_backbone(self, dynamic: bool = False) -> None:
        """Enable `torch.compile` on the classical transformer layers."""
        if self.frontend_mode == ClassicalQuantumFrontEndMode.PURE_CLASSICAL:
            self.classical.enable_compiled_backbone(dynamic=dynamic)
        else:
            self._compiled_tail_fn = torch.compile(self._run_classical_tail, dynamic=dynamic)

    def quantum_real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Collect all real Stiefel parameters in the quantum front-end."""
        params: list[nn.Parameter] = []
        if self.w_in_pkt is not None and self.w_in_mode is not None:
            params.extend([self.w_in_pkt, self.w_in_mode])
        if self.w_in_stiefel is not None:
            params.append(self.w_in_stiefel)
        if self.quantum_block0 is not None:
            params.extend(self.quantum_block0.real_stiefel_parameters())  # type: ignore[operator]
        return params

    def quantum_complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Collect complex head Unitary/Kraus pairs from Quantum Layer 0 if active."""
        if self.quantum_block0 is not None:
            return [self.quantum_block0.complex_unitary_pair()]  # type: ignore[operator]
        return []

    def quantum_scalar_parameters(self) -> list[nn.Parameter]:
        """Collect 1D scalar parameters from the quantum front-end."""
        params: list[nn.Parameter] = []
        for p in (self.in_scale, self.wp_causal_re, self.wp_causal_im):
            if p is not None and p.requires_grad:
                params.append(p)
        if self.quantum_block0 is not None:
            params.extend(self.quantum_block0.scalar_parameters())  # type: ignore[operator]
        return params

    @torch.no_grad()
    def restrict_quantum_scalars_(self) -> None:
        """Clamp quantum front-end scalar parameters to their bounded physical intervals."""
        if self.in_scale is not None:
            self.in_scale.clamp_(0.50, 2.00)
        if self.wp_causal_re is not None:
            self.wp_causal_re.clamp_(-1.50, 1.50)
        if self.wp_causal_im is not None:
            self.wp_causal_im.clamp_(-1.50, 1.50)
        if self.quantum_block0 is not None:
            blk: Any = self.quantum_block0
            blk.obs_scale.clamp_(0.02, 0.50)
            blk.chan_scale.clamp_(0.02, 0.50)
            blk.block_lambdas[0].clamp_(0.50, 1.50)
            blk.block_lambdas[1].clamp_(-0.50, 0.50)
            blk.v_lambdas[0].clamp_(0.50, 1.50)
            blk.v_lambdas[1].clamp_(-0.50, 0.50)

    def _run_classical_tail(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor,
        v_0: torch.Tensor | None,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
        q_scale: torch.Tensor,
        start_layer: int,
    ) -> torch.Tensor:
        """Execute classical U-Net layers `start_layer .. num_layers - 1` (`<= 8` args)."""
        n_half = self.num_layers // 2
        encoder_states: list[torch.Tensor] = [x] if start_layer == 1 else []
        for layer_idx in range(start_layer, self.num_layers):
            block = self.classical.blocks[layer_idx]
            if layer_idx >= n_half and encoder_states:
                dec_i = layer_idx - n_half
                if self.classical.cfg.use_sigmoid_skip:
                    skip_g = torch.sigmoid(self.classical.skip_weights[dec_i]).to(x.dtype)
                else:
                    skip_g = self.classical.skip_weights[dec_i].to(x.dtype)
                x = x + skip_g * encoder_states.pop()
            x, v_0 = block(x, x_0, cos_4d, sin_4d, v_0=v_0, ve_l=None, block_mask=None, q_scale=q_scale)
            if layer_idx < n_half:
                encoder_states.append(x)
        return F.rms_norm(x, (self.embed_dim,), eps=self.classical.cfg.eps)

    def forward_features(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode token IDs `(B, T)` through the Quantum Front-End + Classical Trunk."""
        if self.frontend_mode == ClassicalQuantumFrontEndMode.PURE_CLASSICAL:
            return self.classical.forward_features(idx)

        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if idx.is_cuda else torch.float32
        assert self.q_wte is not None and self.in_scale is not None
        assert self.wp_causal_re is not None and self.wp_causal_im is not None

        raw_wp = (
            F.embedding(idx, self.q_wte.view(self.vocab_size, self.embed_dim))
            .view(bsz, seq_len, self.wp_channels, self.d_h)
            .to(work_dtype)
        )
        if seq_len > 1:
            raw_wp = apply_causal_wavepacket_dispersion(raw_wp, self.wp_causal_re, self.wp_causal_im)
        tok_wp = F.rms_norm(raw_wp, (self.d_h,), eps=TRACE_NORM_EPS)

        if self.frontend_mode == ClassicalQuantumFrontEndMode.QUANTUM_WAVEPACKET_EMB_ONLY:
            assert self.w_in_pkt is not None and self.w_in_mode is not None
            x_0_wp = (
                apply_factored_wavepacket_stiefel(tok_wp, self.w_in_pkt.to(work_dtype), self.w_in_mode.to(work_dtype))
                * self.in_scale.to(work_dtype)
            )
            x_0 = F.rms_norm(x_0_wp.reshape(bsz, seq_len, self.embed_dim), (self.embed_dim,), eps=self.classical.cfg.eps)
            x = x_0
            v_0: torch.Tensor | None = None
            start_layer = 0
        elif self.frontend_mode == ClassicalQuantumFrontEndMode.NATIVE_WAVEPACKET_EMB_AND_L0:
            assert self.w_in_pkt is not None and self.w_in_mode is not None and self.quantum_block0 is not None
            x_0_wp = (
                apply_factored_wavepacket_stiefel(tok_wp, self.w_in_pkt.to(work_dtype), self.w_in_mode.to(work_dtype))
                * self.in_scale.to(work_dtype)
            )
            x_1_wp, v_0_wp, _ = self.quantum_block0(x_0_wp, x_0=x_0_wp, v_0=None)
            x_0 = F.rms_norm(x_0_wp.reshape(bsz, seq_len, self.embed_dim), (self.embed_dim,), eps=self.classical.cfg.eps)
            x = x_1_wp.reshape(bsz, seq_len, self.embed_dim)
            v_0 = v_0_wp.reshape(bsz, seq_len, self.num_heads, self.embed_dim // self.num_heads)
            start_layer = 1
        else:
            assert self.w_in_stiefel is not None and self.quantum_block0 is not None
            tok_emb = tok_wp.reshape(bsz, seq_len, self.embed_dim)
            x_0 = torch.matmul(tok_emb, self.w_in_stiefel.to(work_dtype)) * self.in_scale.to(work_dtype)
            x_1, v_0_flat, _, _ = self.quantum_block0(x_0, x_0=x_0, v_0=None)
            x_0 = F.rms_norm(x_0, (self.embed_dim,), eps=self.classical.cfg.eps)
            x = x_1
            v_0 = v_0_flat.reshape(bsz, seq_len, self.num_heads, self.embed_dim // self.num_heads)
            start_layer = 1

        cos_4d, sin_4d = self.classical.rope.get_cos_sin(seq_len, dtype=x.dtype)
        q_scale = self.classical.rope.yarn_q_scale
        if self._compiled_tail_fn is not None:
            return self._compiled_tail_fn(x, x_0, v_0, cos_4d, sin_4d, q_scale, start_layer)
        return self._run_classical_tail(x, x_0, v_0, cos_4d, sin_4d, q_scale, start_layer)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass returning `(logits_or_none, loss_or_none)` using single-pass chunked softcap CE."""
        if self.frontend_mode == ClassicalQuantumFrontEndMode.PURE_CLASSICAL:
            return self.classical(idx, targets=targets, return_logits=return_logits)

        from ..models.compiled_leaderboard_classical_gpt import _CompiledChunkedSoftcapLinearCEFn

        x_final = self.forward_features(idx)
        if targets is not None and not return_logits:
            loss = _CompiledChunkedSoftcapLinearCEFn.apply(
                x_final,
                self.classical.lm_head.weight,
                targets,
                self.classical.cfg.logit_softcap,
                self.classical.cfg.ce_chunk_tokens,
                True,
            )
            return None, loss

        logits = self.classical.compute_logits(x_final)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1).long(),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def compute_quantum_frontend_telemetry(self) -> dict[str, float]:
        """Return Stiefel/Unitary manifold error and density purity/rank telemetry for the quantum front-end."""
        if self.frontend_mode == ClassicalQuantumFrontEndMode.PURE_CLASSICAL:
            return {}
        st_errs: list[float] = []
        for w in self.quantum_real_stiefel_parameters():
            wf = w.float()
            if wf.size(-2) < wf.size(-1):
                wf = wf.transpose(-2, -1)
            wf_unit = wf / wf.norm(dim=-2, keepdim=True).clamp_min(1e-8)
            d_cols = wf.size(-1)
            eye = torch.eye(d_cols, device=wf.device, dtype=wf.dtype)
            gram = torch.matmul(wf_unit.transpose(-2, -1), wf_unit)
            st_errs.append(float(((gram - eye).norm(dim=(-2, -1)) / math.sqrt(float(d_cols))).max().item()))
        wte_err = 0.0
        if self.q_wte is not None:
            wte_rms = self.q_wte.float().pow(2).mean(dim=-1).sqrt()
            wte_err = float((wte_rms - 1.0).abs().max().item())
        out: dict[str, float] = {
            "q_max_stiefel_err": max(st_errs) if st_errs else 0.0,
            "q_wte_sphere_err": wte_err,
        }
        if self.quantum_block0 is not None:
            tel = self.quantum_block0.get_last_telemetry_dict()  # type: ignore[operator]
            out["q_l0_purity"] = tel["mean_purity"]
            out["q_l0_eff_rank"] = tel["mean_eff_rank"]
        return out


class HybridQuantumClassicalOptimizerBundle:
    """Unified optimizer bundle stepping Classical `StiefelMuon`+`Adam` and Quantum Front-End optimizers."""

    def __init__(
        self,
        model: HybridQuantumFrontEndClassicalGPT,
        muon_opt: Any,
        adam_opt: torch.optim.Optimizer,
        q_trunk_opt: Any = None,
        q_wte_opt: Any = None,
        q_scalar_opt: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.muon_opt = muon_opt
        self.adam_opt = adam_opt
        self.q_trunk_opt = q_trunk_opt
        self.q_wte_opt = q_wte_opt
        self.q_scalar_opt = q_scalar_opt

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients across both classical and quantum optimizer groups."""
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adam_opt.zero_grad(set_to_none=set_to_none)
        if self.q_trunk_opt is not None:
            self.q_trunk_opt.zero_grad(set_to_none=set_to_none)
        if self.q_wte_opt is not None:
            self.q_wte_opt.zero_grad(set_to_none=set_to_none)
        if self.q_scalar_opt is not None:
            self.q_scalar_opt.zero_grad(set_to_none=set_to_none)

    def step(self, idx: torch.Tensor, lr_mult: float) -> None:
        """Step both the classical trunk/readout optimizers and the quantum front-end optimizers."""
        from ..models.compiled_leaderboard_classical_gpt import (
            DEFAULT_LEADERBOARD_HEAD_LR,
            DEFAULT_LEADERBOARD_MUON_LR,
            DEFAULT_LEADERBOARD_SCALAR_LR,
            DEFAULT_LEADERBOARD_WTE_LR,
        )
        from ..models.standalone_quantum_born_gpt import DEFAULT_SCALAR_ADAM_LR

        for grp in self.muon_opt.param_groups:
            grp["lr"] = DEFAULT_LEADERBOARD_MUON_LR * lr_mult
        for grp in self.adam_opt.param_groups:
            g_name = grp.get("group_name", "")
            base_lr = (
                DEFAULT_LEADERBOARD_HEAD_LR
                if g_name == "lm_head"
                else (DEFAULT_LEADERBOARD_WTE_LR if g_name == "wte_and_ve" else DEFAULT_LEADERBOARD_SCALAR_LR)
            )
            grp["lr"] = base_lr * lr_mult
        self.muon_opt.step()
        self.adam_opt.step()

        if self.q_trunk_opt is not None:
            self.q_trunk_opt.set_lr_scale(lr_mult)
            self.q_trunk_opt.step()
        if self.q_wte_opt is not None:
            idx_in = torch.unique(idx.reshape(-1))
            self.q_wte_opt.step_active(idx_in, lr_scale=lr_mult)
        if self.q_scalar_opt is not None:
            for grp in self.q_scalar_opt.param_groups:
                grp["lr"] = DEFAULT_HYBRID_SCALAR_LR * lr_mult
            self.q_scalar_opt.step()
            self.model.restrict_quantum_scalars_()


def build_hybrid_quantum_classical_optimizers(
    model: HybridQuantumFrontEndClassicalGPT,
) -> HybridQuantumClassicalOptimizerBundle:
    """Build matched Classical `StiefelMuon`+`Adam` + Quantum Front-End Riemannian optimizers (`<= 8` args)."""
    from ..models.compiled_leaderboard_classical_gpt import (
        CompiledLeaderboardOptimizerConfig,
        DEFAULT_LEADERBOARD_ADAM_BETA1,
        DEFAULT_LEADERBOARD_ADAM_BETA2,
        DEFAULT_LEADERBOARD_ADAM_EPS,
        DEFAULT_LEADERBOARD_MUON_MOMENTUM,
        DEFAULT_LEADERBOARD_MUON_MOMENTUM_INIT,
    )
    from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
    from ..models.standalone_quantum_born_gpt import DEFAULT_UNITARY_GEODESIC_LR
    from ..optim.muon import StiefelMuon, StiefelMuonConfig
    from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon, DEFAULT_BATCH_GRAM_MUON_LR

    cfg = CompiledLeaderboardOptimizerConfig()
    c_mod = model.classical
    muon_params: list[nn.Parameter] = []
    wte_params: list[nn.Parameter] = [p for p in [c_mod.wte.weight] if p.requires_grad]
    head_params: list[nn.Parameter] = [c_mod.lm_head.weight]
    scalar_params: list[nn.Parameter] = [c_mod.skip_weights]

    if c_mod.smear_lambda is not None and c_mod.smear_lambda.requires_grad:
        scalar_params.append(c_mod.smear_lambda)
    if c_mod.smear_gate is not None and c_mod.smear_gate.weight.requires_grad:
        scalar_params.append(c_mod.smear_gate.weight)

    for blk in c_mod.blocks:
        if not any(p.requires_grad for p in blk.parameters()):
            continue
        scalar_params.append(blk.block_lambdas)
        if blk.sa_lambdas is not None:
            scalar_params.append(blk.sa_lambdas)
        if blk.xsa_alpha is not None:
            scalar_params.append(blk.xsa_alpha)
        if blk.attn_gate is not None:
            scalar_params.append(blk.attn_gate.weight)
        if blk.ve_gate is not None:
            scalar_params.append(blk.ve_gate.weight)
        if blk.qkv_proj is not None:
            muon_params.append(blk.qkv_proj.weight)
        if blk.out_proj is not None:
            muon_params.append(blk.out_proj.weight)
        muon_params.append(blk.fc1.weight)
        muon_params.append(blk.fc2.weight)

    muon_opt = StiefelMuon(
        muon_params,
        lr=cfg.muon_lr,
        momentum=DEFAULT_LEADERBOARD_MUON_MOMENTUM,
        weight_decay=cfg.weight_decay,
        decay_mode="cautious" if cfg.weight_decay > 0.0 else "none",
        aspect_scale=True,
        config=StiefelMuonConfig(
            nesterov=True,
            ns_steps=5,
            polar_method=cfg.polar_method,
            stiefel_penalty=0.0,
            use_normuon=cfg.use_normuon,
            use_twin_rail=cfg.use_twin_rail,
        ),
        momentum_warmup_steps=cfg.muon_warmup_steps,
        momentum_init=DEFAULT_LEADERBOARD_MUON_MOMENTUM_INIT,
        ema_momentum=True,
        batched_ns=True,
    )

    is_cuda = next(model.parameters()).is_cuda
    adam_groups: list[dict[str, Any]] = [
        {"params": head_params, "lr": cfg.head_lr, "group_name": "lm_head"},
        {"params": scalar_params, "lr": cfg.scalar_lr, "group_name": "scalars"},
    ]
    if wte_params:
        adam_groups.insert(1, {"params": wte_params, "lr": cfg.wte_lr, "group_name": "wte_and_ve"})
    adam_opt = torch.optim.Adam(
        adam_groups,
        betas=(DEFAULT_LEADERBOARD_ADAM_BETA1, DEFAULT_LEADERBOARD_ADAM_BETA2),
        eps=DEFAULT_LEADERBOARD_ADAM_EPS,
        fused=is_cuda,
    )

    if model.frontend_mode == ClassicalQuantumFrontEndMode.PURE_CLASSICAL:
        return HybridQuantumClassicalOptimizerBundle(model, muon_opt, adam_opt)

    q_stiefel_params = model.quantum_real_stiefel_parameters()
    q_trunk_opt = (
        ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=q_stiefel_params,
            complex_operator_pairs=model.quantum_complex_unitary_pairs(),
            stiefel_lr=DEFAULT_HYBRID_STIEFEL_LR,
            unitary_lr=DEFAULT_UNITARY_GEODESIC_LR,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
            momentum_mode=StiefelMomentumMode.UNNORMALIZED_FINAL_READOUT_HEAD,
        )
        if q_stiefel_params
        else None
    )
    assert model.q_wte is not None
    q_wte_opt = BatchSubspaceTikhonovGramMuon(
        [model.q_wte], lr=DEFAULT_BATCH_GRAM_MUON_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA
    )
    q_scalar_opt = torch.optim.AdamW(
        [{"params": model.quantum_scalar_parameters(), "lr": DEFAULT_HYBRID_SCALAR_LR, "weight_decay": 0.0}],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return HybridQuantumClassicalOptimizerBundle(
        model, muon_opt, adam_opt, q_trunk_opt=q_trunk_opt, q_wte_opt=q_wte_opt, q_scalar_opt=q_scalar_opt
    )


__all__ = [
    "ClassicalQuantumFrontEndMode",
    "DEFAULT_HYBRID_SCALAR_LR",
    "DEFAULT_HYBRID_STIEFEL_LR",
    "DEFAULT_PPMI_ORTHO_SEED",
    "DEFAULT_WAVEPACKET_CHANNELS",
    "DEFAULT_WAVEPACKET_COMPLEX_HEADS",
    "DEFAULT_WAVEPACKET_HILBERT_DIM",
    "DEFAULT_WAVEPACKET_RESIDUAL_GAIN",
    "DEFAULT_WAVEPACKET_SDPA_HEAD_DIM",
    "DEFAULT_WAVEPACKET_SDPA_SCALE",
    "HybridQuantumClassicalOptimizerBundle",
    "HybridQuantumFrontEndClassicalGPT",
    "NativeWavepacketStatMechBlock",
    "apply_causal_wavepacket_dispersion",
    "apply_factored_wavepacket_stiefel",
    "build_hybrid_quantum_classical_optimizers",
    "compiled_wavepacket_statmech_step",
    "form_wavepacket_outer_product_and_v",
    "initialize_packet_stiefel_3d",
    "initialize_wavepacket_ppmi_sphere_embeddings_",
    "wavepacket_statmech_purify_evolve_impl",
]
