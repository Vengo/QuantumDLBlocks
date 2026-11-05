r"""4D Complex Wavefunction Fiber-Bundle (`Meaning + Identity`) Initialization & Dynamic Packet Readout (`Wave 57` / `EXPERIMENT_PLAN_078.md`).

Implements:
1. **Joint `(2, d_h)` Complex Wavefunction Sphere Geometry (`project_complex_wavepackets_to_rms_sphere_`)**:
   - Normalizes each complex wavepacket $\psi_{v,m} = \operatorname{Re}\psi_{v,m} + i\operatorname{Im}\psi_{v,m} \in \mathbb{C}^{d_h}$
     onto the joint complex unit-RMS sphere $\|\operatorname{Re}\psi_{v,m}\|_2^2 + \|\operatorname{Im}\psi_{v,m}\|_2^2 = 2 d_h$,
     preserving $U(1)$ Berry-phase invariance and allowing tokens to trade energy between forward (`Re`) and reverse (`Im`) contexts.
2. **Support-Preserving Directed Multi-Lag ($k \in \{1, 2, 4, 8\}$) Complex PPMI CountSketch (`V = 50,304`)**:
   - Streams FineWeb training shards (`fineweb_train_000001.bin`, strictly excluding validation Shard 0) to accumulate
     multi-lag outgoing (`Forward -> Re`) and incoming (`Reverse -> Im`) Shifted-PPMI sketches.
   - Applies **Active-Only Centering** (`active_mask = row_norm > 0`) so unseen/rare tail tokens remain at zero before
     orthogonal blending, eliminating the $\max\cos = 1.0000$ rare-token collision collapse.
3. **Fiber-Bundle Split (`M_sem` Semantic + `M_id` Identity) & Complex Hermitian Gram-Schmidt (`_hermitian_gram_schmidt_complex_blend`)**:
   - Balances semantic packets via a Haar-random complex unitary $U_M \in U(M_{\text{sem}})$, projects out both real and
     imaginary overlaps $\langle s_{v,m} \mid z_{v,m}^\perp \rangle_{\mathbb{C}} = 0$, blends semantic packets via Empirical-Bayes
     shrinkage $\alpha_{v,\text{sem}} = \alpha_{\max}\sqrt{c_v / (c_v + k_0)}$, and reserves $M_{\text{id}} = M_c - M_{\text{sem}}$
     orthogonal identity complex packets ($\alpha_{v,\text{id}} = 0$).
4. **Dynamic Context-Gated Complex Packet Readout (`DynamicComplexPacketReadoutModulator`) & Step-0 Readout Calibration**:
   - Modulates pre-logit hidden states across the $M_c$ complex packets using a phase-preserving gate shared across $(\operatorname{Re}, \operatorname{Im})$.
"""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_078.md Section 2.5
# ==============================================================================
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_FINEWEB_TRAIN_SHARD1_PATH: Path = _REPO_ROOT / ".cache" / "fineweb_100M" / "fineweb_train_000001.bin"
DEFAULT_DIRECTED_PPMI_CACHE_PATH: Path = (
    _REPO_ROOT / ".cache" / "fineweb_directed_complex_ppmi_v50304_d768.pt"
)

DEFAULT_COMPLEX_PACKET_COUNT: int = 12
DEFAULT_COMPLEX_PACKET_HILBERT_DIM: int = 32
DEFAULT_FIBER_SEMANTIC_PACKETS_11: int = 11
DEFAULT_FIBER_SEMANTIC_PACKETS_10: int = 10
DEFAULT_EB_ALPHA_MAX: float = 0.96
DEFAULT_EB_PRIOR_COUNT_K0: float = 20.0
DEFAULT_IDENTITY_PACKET_LR_SCALE: float = 0.25
DEFAULT_SEMANTIC_GATE_BIAS_INIT: float = 0.85
DEFAULT_IDENTITY_GATE_BIAS_INIT: float = -0.85
DEFAULT_FIBER_BUNDLE_SEED: int = 20261102
DEFAULT_PPMI_SHIFT_KAPPA: float = 2.0
DEFAULT_PPMI_CDS_POWER: float = 0.75
DEFAULT_SKETCH_TOP_HUBS: int = 3072
DEFAULT_SKETCH_TAIL_BUCKETS: int = 1024
DEFAULT_MULTI_LAGS: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_MULTI_LAG_WEIGHTS: tuple[float, ...] = (1.0, 0.70, 0.45, 0.25)
FIBER_EPS: float = 1e-7


class QuantumPacketReadoutMode(str, Enum):
    """Strategy enum selecting standard linear readout vs. dynamic complex-packet gated readout (Rule 3)."""

    STANDARD_LINEAR_READOUT = "standard_linear_readout"
    DYNAMIC_COMPLEX_PACKET_GATED_READOUT = "dynamic_complex_packet_gated_readout"


def project_complex_wavepackets_to_rms_sphere_(
    weight: torch.Tensor,
    eps: float = FIBER_EPS,
) -> torch.Tensor:
    r"""Normalize each complex wavepacket in `weight` (`(..., M_c, 2, d_h)` or `(..., M_c, 2*d_h)`) onto $\sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1}$."""
    with torch.no_grad():
        if weight.ndim >= 4 and weight.size(-2) == 2:
            two_dh = float(2 * weight.size(-1))
            target_radius = math.sqrt(two_dh)
            flat_norm = weight.float().flatten(start_dim=-2).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
            scale = (target_radius / flat_norm).to(weight.dtype)
            weight.mul_(scale)
        else:
            d_float = float(weight.size(-1))
            target_radius = math.sqrt(d_float)
            row_norm = weight.float().norm(dim=-1, keepdim=True).clamp_min(eps)
            scale = (target_radius / row_norm).to(weight.dtype)
            weight.mul_(scale)
    return weight


def compute_hermitian_orthogonal_residual(
    s_re: torch.Tensor,
    s_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    eps: float = FIBER_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Project out both real and imaginary Hermitian inner products $\langle s_{v,m} \mid z_{v,m} \rangle_{\mathbb{C}}$ and normalize onto $\sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1}$."""
    two_dh = float(2 * s_re.size(-1))
    target_radius = math.sqrt(two_dh)
    sr_f, si_f = s_re.float(), s_im.float()
    zr_f, zi_f = z_re.float(), z_im.float()

    s_nsq = sr_f.square().sum(dim=-1, keepdim=True) + si_f.square().sum(dim=-1, keepdim=True)
    active_mask = s_nsq > (eps * eps)
    s_scale = torch.where(active_mask, target_radius * torch.rsqrt(s_nsq.clamp_min(eps * eps)), torch.zeros_like(s_nsq))
    sr_unit = sr_f * s_scale
    si_unit = si_f * s_scale

    c_re = (sr_unit * zr_f + si_unit * zi_f).sum(dim=-1, keepdim=True) / two_dh
    c_im = (sr_unit * zi_f - si_unit * zr_f).sum(dim=-1, keepdim=True) / two_dh
    zp_re = zr_f - (c_re * sr_unit - c_im * si_unit)
    zp_im = zi_f - (c_re * si_unit + c_im * sr_unit)

    zp_norm = (zp_re.square().sum(dim=-1, keepdim=True) + zp_im.square().sum(dim=-1, keepdim=True)).clamp_min(eps * eps).sqrt()
    zp_scale = target_radius / zp_norm
    return sr_unit, si_unit, zp_re * zp_scale, zp_im * zp_scale, active_mask


def _hermitian_gram_schmidt_complex_blend(
    s_re: torch.Tensor,
    s_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    alpha_v: torch.Tensor,
    eps: float = FIBER_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Blend directed complex semantic packet $(s_{\text{re}}, s_{\text{im}})$ with Hermitian-orthogonal $(z_{\text{re}}^\perp, z_{\text{im}}^\perp)$ on $\sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1}$."""
    sr_unit, si_unit, zp_re, zp_im, active_mask = compute_hermitian_orthogonal_residual(
        s_re, s_im, z_re, z_im, eps=eps
    )
    a_view = alpha_v.float().view(-1, 1, 1) if alpha_v.ndim == 1 else alpha_v.float()
    a_eff = torch.where(active_mask, a_view.clamp(0.0, 0.9999), torch.zeros_like(sr_unit[..., :1]))
    b_eff = (1.0 - a_eff.square()).clamp_min(0.0).sqrt()
    out_re = a_eff * sr_unit + b_eff * zp_re
    out_im = a_eff * si_unit + b_eff * zp_im
    return out_re, out_im


def sample_haar_complex_unitary(
    dim: int,
    seed: int = DEFAULT_FIBER_BUNDLE_SEED,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a deterministic Haar-random complex unitary matrix $U = U_{\\text{re}} + i U_{\\text{im}} \\in U(\\text{dim})$."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    a_re = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    a_im = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    z_c = torch.complex(a_re, a_im)
    q_c, r_c = torch.linalg.qr(z_c)
    diag_r = torch.diagonal(r_c)
    phase = torch.where(diag_r.abs() > FIBER_EPS, diag_r / diag_r.abs().clamp_min(FIBER_EPS), torch.ones_like(diag_r))
    q_haar = q_c * phase.unsqueeze(0)
    return q_haar.real.to(device).contiguous(), q_haar.imag.to(device).contiguous()


def apply_complex_unitary_mode_balance(
    s_re: torch.Tensor,
    s_im: torch.Tensor,
    u_re: torch.Tensor,
    u_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix `(V, M_sem, d_h)` complex packets across the `M_sem` packet axis via $U_M \\in U(M_{\\text{sem}})$."""
    ur = u_re.to(device=s_re.device, dtype=s_re.dtype)
    ui = u_im.to(device=s_re.device, dtype=s_re.dtype)
    bal_re = torch.einsum("mn,vnd->vmd", ur, s_re) - torch.einsum("mn,vnd->vmd", ui, s_im)
    bal_im = torch.einsum("mn,vnd->vmd", ur, s_im) + torch.einsum("mn,vnd->vmd", ui, s_re)
    return bal_re.contiguous(), bal_im.contiguous()


def active_only_center_rows(
    mat: torch.Tensor,
    eps: float = FIBER_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract the column mean strictly over active rows (`row_norm > eps`), keeping zero-energy unseen rows at exact `0.0`."""
    row_norms = mat.float().norm(dim=-1)
    active_mask = row_norms > eps
    if int(active_mask.sum().item()) <= 1:
        return mat.clone(), active_mask
    mu_active = mat[active_mask].float().mean(dim=0, keepdim=True).to(mat.dtype)
    out = mat.clone()
    out[active_mask] = out[active_mask] - mu_active
    out[~active_mask] = 0.0
    return out, active_mask


def _load_fineweb_tokens_from_shard(
    shard_path: Path,
    max_tokens: int,
) -> torch.Tensor:
    """Load up to `max_tokens` uint16 tokens from a FineWeb `.bin` shard (strictly rejecting Shard 0 validation files)."""
    if "val" in shard_path.name.lower() or "000000" in shard_path.name:
        raise ValueError(f"Refusing to read validation shard for initialization: {shard_path}")
    header = np.fromfile(shard_path, dtype=np.int32, count=256)
    n_avail = int(header[2]) if header.size >= 3 and header[0] == 20240520 else max(0, (shard_path.stat().st_size - 1024) // 2)
    n_read = min(int(max_tokens), n_avail)
    raw = np.fromfile(shard_path, dtype=np.uint16, count=n_read, offset=256 * 4)
    return torch.from_numpy(raw.astype(np.int64))


def _compute_shifted_ppmi_active_centered(
    counts_mat: torch.Tensor,
    shift_kappa: float = DEFAULT_PPMI_SHIFT_KAPPA,
    cds_power: float = DEFAULT_PPMI_CDS_POWER,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform non-negative co-occurrence matrix `(V, D_sketch)` into active-only centered log-Shifted-PPMI."""
    c_f = counts_mat.float().clamp_min(0.0)
    row_sum = c_f.sum(dim=-1, keepdim=True)
    col_sum = c_f.sum(dim=0, keepdim=True).pow(cds_power)
    tot_mass = col_sum.sum().clamp_min(1.0)
    denom = (row_sum * col_sum).clamp_min(FIBER_EPS)
    pmi = torch.log((c_f * tot_mass / denom).clamp_min(FIBER_EPS)) - math.log(max(shift_kappa, 1.0))
    ppmi = torch.log1p(F.relu(pmi))
    row_l2 = ppmi.norm(dim=-1, keepdim=True)
    active = (row_l2.squeeze(-1) > FIBER_EPS)
    ppmi = torch.where(row_l2 > FIBER_EPS, ppmi / row_l2.clamp_min(FIBER_EPS), torch.zeros_like(ppmi))
    centered, _ = active_only_center_rows(ppmi)
    return centered, active


def compute_directed_complex_ppmi_from_tokens(
    tokens_1d: torch.Tensor,
    vocab_size: int = 50304,
    m_c: int = DEFAULT_COMPLEX_PACKET_COUNT,
    d_h: int = DEFAULT_COMPLEX_PACKET_HILBERT_DIM,
    seed: int = DEFAULT_FIBER_BUNDLE_SEED,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Compute directed multi-lag ($k \\in \\{1, 2, 4, 8\\}$) Forward (`Re`) and Reverse (`Im`) active-centered PPMI embeddings (`<= 8` args)."""
    work_dev = torch.device(device) if device is not None else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    tok_cpu = tokens_1d.long().clamp(0, vocab_size - 1)
    token_counts = torch.bincount(tok_cpu, minlength=vocab_size).float()

    d_target = int(m_c * d_h)
    k_top = min(DEFAULT_SKETCH_TOP_HUBS, vocab_size // 2)
    b_tail = max(d_target, min(DEFAULT_SKETCH_TAIL_BUCKETS, max(64, vocab_size - k_top)))
    d_sketch = k_top + b_tail

    top_vals, top_idx = torch.topk(token_counts, k=k_top, largest=True, sorted=True)
    col_map = torch.full((vocab_size,), -1, dtype=torch.long)
    col_map[top_idx] = torch.arange(k_top, dtype=torch.long)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 17)
    tail_buckets = torch.randint(0, b_tail, (vocab_size,), generator=gen, dtype=torch.long) + k_top
    col_map = torch.where(col_map >= 0, col_map, tail_buckets).to(work_dev)

    c_fwd = torch.zeros(vocab_size * d_sketch, device=work_dev, dtype=torch.float32)
    c_rev = torch.zeros(vocab_size * d_sketch, device=work_dev, dtype=torch.float32)

    chunk_len = 5_000_000
    n_tok = int(tok_cpu.numel())
    for start in range(0, n_tok, chunk_len):
        end = min(n_tok, start + chunk_len)
        t_chunk = tok_cpu[start:end].to(work_dev)
        if t_chunk.numel() <= max(DEFAULT_MULTI_LAGS):
            continue
        for lag, w_lag in zip(DEFAULT_MULTI_LAGS, DEFAULT_MULTI_LAG_WEIGHTS):
            u_src = t_chunk[:-lag]
            v_dst = t_chunk[lag:]
            idx_f = u_src * d_sketch + col_map[v_dst]
            idx_r = v_dst * d_sketch + col_map[u_src]
            w_vec = torch.full((idx_f.numel(),), float(w_lag), device=work_dev, dtype=torch.float32)
            c_fwd.scatter_add_(0, idx_f, w_vec)
            c_rev.scatter_add_(0, idx_r, w_vec)

    c_fwd_2d = c_fwd.view(vocab_size, d_sketch)
    c_rev_2d = c_rev.view(vocab_size, d_sketch)
    ppmi_fwd, act_fwd = _compute_shifted_ppmi_active_centered(c_fwd_2d)
    ppmi_rev, act_rev = _compute_shifted_ppmi_active_centered(c_rev_2d)
    if act_fwd.any():
        af_idx = torch.nonzero(act_fwd, as_tuple=False).squeeze(-1)
        ppmi_fwd[af_idx, col_map[af_idx]] += ppmi_fwd[af_idx].amax(dim=-1).clamp_min(0.25)
        ppmi_fwd, _ = active_only_center_rows(ppmi_fwd, act_fwd)
    if act_rev.any():
        ar_idx = torch.nonzero(act_rev, as_tuple=False).squeeze(-1)
        ppmi_rev[ar_idx, col_map[ar_idx]] += ppmi_rev[ar_idx].amax(dim=-1).clamp_min(0.25)
        ppmi_rev, _ = active_only_center_rows(ppmi_rev, act_rev)

    d_target = int(m_c * d_h)
    if vocab_size >= d_target and d_sketch >= d_target:
        q_svd = min(vocab_size, d_sketch, d_target + 32)
        with torch.random.fork_rng(devices=[work_dev] if work_dev.type == "cuda" else []):
            torch.manual_seed(int(seed) + 101)
            if work_dev.type == "cuda":
                torch.cuda.manual_seed_all(int(seed) + 101)
            u_f, s_f, _ = torch.svd_lowrank(ppmi_fwd, q=q_svd, niter=3)
            u_r, s_r, _ = torch.svd_lowrank(ppmi_rev, q=q_svd, niter=3)
        wf = u_f[:, :d_target] * s_f[:d_target].clamp_min(FIBER_EPS).sqrt().unsqueeze(0)
        wr = u_r[:, :d_target] * s_r[:d_target].clamp_min(FIBER_EPS).sqrt().unsqueeze(0)
        s_fwd_raw = wf.view(vocab_size, d_h, m_c).transpose(1, 2).contiguous().cpu()
        s_rev_raw = wr.view(vocab_size, d_h, m_c).transpose(1, 2).contiguous().cpu()
    else:
        gen_proj = torch.Generator(device="cpu")
        gen_proj.manual_seed(int(seed) + 101)
        q_fwd, _ = torch.linalg.qr(torch.randn(d_sketch, d_target, generator=gen_proj, dtype=torch.float32).to(work_dev))
        q_rev, _ = torch.linalg.qr(torch.randn(d_sketch, d_target, generator=gen_proj, dtype=torch.float32).to(work_dev))
        s_fwd_raw = torch.matmul(ppmi_fwd, q_fwd).view(vocab_size, m_c, d_h).cpu()
        s_rev_raw = torch.matmul(ppmi_rev, q_rev).view(vocab_size, m_c, d_h).cpu()
    active_union = (act_fwd | act_rev).cpu()
    s_fwd_raw[~active_union] = 0.0
    s_rev_raw[~active_union] = 0.0

    return {
        "s_fwd_raw": s_fwd_raw.contiguous(),
        "s_rev_raw": s_rev_raw.contiguous(),
        "token_counts": token_counts.cpu().contiguous(),
        "active_mask": active_union.contiguous(),
    }


def build_or_load_directed_complex_ppmi_sketch(
    vocab_size: int = 50304,
    m_c: int = DEFAULT_COMPLEX_PACKET_COUNT,
    d_h: int = DEFAULT_COMPLEX_PACKET_HILBERT_DIM,
    max_tokens: int = 25_000_000,
    shard_path: Path = DEFAULT_FINEWEB_TRAIN_SHARD1_PATH,
    cache_path: Path | None = DEFAULT_DIRECTED_PPMI_CACHE_PATH,
) -> dict[str, torch.Tensor]:
    """Load cached directed multi-lag complex PPMI sketch or stream `shard_path` (`fineweb_train_000001.bin`) to build it (`<= 8` args)."""
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            isinstance(payload, dict)
            and "s_fwd_raw" in payload
            and tuple(payload["s_fwd_raw"].shape) == (vocab_size, m_c, d_h)
        ):
            return payload

    if shard_path.exists():
        tokens_1d = _load_fineweb_tokens_from_shard(shard_path, max_tokens=max_tokens)
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(DEFAULT_FIBER_BUNDLE_SEED)
        tokens_1d = torch.randint(0, max(16, (vocab_size * 4) // 5), (min(max_tokens, 200_000),), generator=gen)

    sketch = compute_directed_complex_ppmi_from_tokens(
        tokens_1d, vocab_size=vocab_size, m_c=m_c, d_h=d_h
    )
    if cache_path is not None and shard_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sketch, cache_path)
    return sketch


def initialize_directed_complex_ppmi_fiber_bundle_embeddings_(
    wte_tensor: torch.Tensor,
    m_sem: int = DEFAULT_FIBER_SEMANTIC_PACKETS_11,
    seed: int = DEFAULT_FIBER_BUNDLE_SEED,
    sketch_dict: dict[str, torch.Tensor] | None = None,
    k0: float = DEFAULT_EB_PRIOR_COUNT_K0,
    alpha_max: float = DEFAULT_EB_ALPHA_MAX,
    id_lr_scale: float = DEFAULT_IDENTITY_PACKET_LR_SCALE,
) -> dict[str, float]:
    r"""Initialize `wte_tensor` (`(V, M_c, 2, d_h)` or `(V, M_c, 2*d_h)`) with `M_sem` Semantic + `M_id` Identity Complex Wavepackets (`<= 8` args)."""
    with torch.no_grad():
        if wte_tensor.ndim == 4:
            vocab_size, m_c, _, d_h = wte_tensor.shape
        elif wte_tensor.ndim == 3 and wte_tensor.size(-1) % 2 == 0:
            vocab_size, m_c, two_dh = wte_tensor.shape
            d_h = two_dh // 2
        else:
            raise ValueError(f"Expected 4D (V, M_c, 2, d_h) or 3D (V, M_c, 2*d_h), got {tuple(wte_tensor.shape)}")

        m_sem_eff = max(1, min(int(m_sem), m_c))
        if sketch_dict is None:
            sketch_dict = build_or_load_directed_complex_ppmi_sketch(vocab_size=vocab_size, m_c=m_c, d_h=d_h)

        s_fwd_raw = sketch_dict["s_fwd_raw"][:, :m_sem_eff, :].float()
        s_rev_raw = sketch_dict["s_rev_raw"][:, :m_sem_eff, :].float()
        counts = sketch_dict["token_counts"].float()

        u_re, u_im = sample_haar_complex_unitary(m_sem_eff, seed=int(seed))
        s_re_bal, s_im_bal = apply_complex_unitary_mode_balance(s_fwd_raw, s_rev_raw, u_re, u_im)

        gen_z = torch.Generator(device="cpu")
        gen_z.manual_seed(int(seed) + 777)
        z_4d = torch.randn(vocab_size, m_c, 2, d_h, generator=gen_z, dtype=torch.float32)
        project_complex_wavepackets_to_rms_sphere_(z_4d)

        alpha_v = float(alpha_max) * torch.sqrt(counts / (counts + float(k0)).clamp_min(FIBER_EPS))
        alpha_v = torch.where(counts > 0.0, alpha_v, torch.zeros_like(alpha_v))

        sem_re, sem_im = _hermitian_gram_schmidt_complex_blend(
            s_re_bal, s_im_bal, z_4d[:, :m_sem_eff, 0, :], z_4d[:, :m_sem_eff, 1, :], alpha_v
        )
        out_4d = z_4d.clone()
        out_4d[:, :m_sem_eff, 0, :] = sem_re
        out_4d[:, :m_sem_eff, 1, :] = sem_im
        project_complex_wavepackets_to_rms_sphere_(out_4d)

        if wte_tensor.ndim == 4:
            wte_tensor.copy_(out_4d.to(device=wte_tensor.device, dtype=wte_tensor.dtype))
        else:
            wte_tensor.copy_(out_4d.reshape(vocab_size, m_c, 2 * d_h).to(device=wte_tensor.device, dtype=wte_tensor.dtype))

        slice_mult = torch.ones(m_c, device=wte_tensor.device, dtype=torch.float32)
        if m_sem_eff < m_c:
            slice_mult[m_sem_eff:] = float(id_lr_scale)
        setattr(wte_tensor, "_slice_lr_mult", slice_mult)

        rare_mask = counts <= 1.0
        rare_idx = torch.nonzero(rare_mask, as_tuple=False).squeeze(-1)
        if rare_idx.numel() > 256:
            rare_idx = rare_idx[:256]
        if rare_idx.numel() >= 2:
            rare_flat = F.normalize(out_4d.index_select(0, rare_idx).reshape(rare_idx.numel(), -1), dim=-1)
            cos_mat = torch.matmul(rare_flat, rare_flat.t())
            cos_mat.fill_diagonal_(-1.0)
            rare_max_cos = float(cos_mat.max().item())
        else:
            rare_max_cos = 0.0

        return {
            "m_sem": float(m_sem_eff),
            "m_id": float(m_c - m_sem_eff),
            "active_tokens": float((counts > 0).sum().item()),
            "rare_max_cos": rare_max_cos,
            "mean_alpha_sem": float(alpha_v.mean().item()),
        }


class DynamicComplexPacketReadoutModulator(nn.Module):
    r"""Phase-preserving dynamic context-gated complex wavepacket modulator $g_m(x_{\text{final}}) = \operatorname{softplus}(g_{0,m} + W_{\text{gate}} x_{\text{final}})$."""

    def __init__(
        self,
        embed_dim: int = 768,
        m_c: int = DEFAULT_COMPLEX_PACKET_COUNT,
        m_sem: int = DEFAULT_FIBER_SEMANTIC_PACKETS_11,
        gate_in_ch: int = 32,
        init_biased: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.m_c = int(m_c)
        self.m_sem = max(1, min(int(m_sem), self.m_c))
        self.d_h = self.embed_dim // (2 * self.m_c)
        self.gate_in_ch = min(int(gate_in_ch), self.embed_dim)

        g0 = torch.zeros(self.m_c, dtype=torch.float32)
        if init_biased:
            g0[: self.m_sem] = DEFAULT_SEMANTIC_GATE_BIAS_INIT
            if self.m_sem < self.m_c:
                g0[self.m_sem :] = DEFAULT_IDENTITY_GATE_BIAS_INIT
        self.gate_bias = nn.Parameter(g0)
        self.gate_proj = nn.Linear(self.gate_in_ch, self.m_c, bias=False)
        nn.init.zeros_(self.gate_proj.weight)

    def compute_packet_gates(self, x_final: torch.Tensor) -> torch.Tensor:
        """Return mean-normalized positive per-complex-packet gates `(..., M_c)` shared across `(Re, Im)`."""
        work_dtype = x_final.dtype
        g_in = x_final[..., : self.gate_in_ch].float()
        raw_logits = self.gate_bias.float() + F.linear(g_in, self.gate_proj.weight.float())
        pos_g = F.softplus(raw_logits)
        norm_g = pos_g * (float(self.m_c) / pos_g.sum(dim=-1, keepdim=True).clamp_min(FIBER_EPS))
        return norm_g.sqrt().to(work_dtype)

    def forward(self, x_final: torch.Tensor) -> torch.Tensor:
        """Modulate `x_final` `(..., 768)` across `(..., M_c, 2, d_h)` without shearing complex packet phase."""
        orig_shape = x_final.shape
        g_mc = self.compute_packet_gates(x_final).unsqueeze(-1).unsqueeze(-1)
        x_4d = x_final.view(*orig_shape[:-1], self.m_c, 2, self.d_h)
        return (x_4d * g_mc).reshape(orig_shape)

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return `[gate_bias, gate_proj.weight]` for scalar optimizer registration."""
        return [self.gate_bias, self.gate_proj.weight]

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp packet gate bias and projection weights to bounded physical intervals."""
        self.gate_bias.clamp_(-3.0, 3.0)
        self.gate_proj.weight.clamp_(-2.0, 2.0)


@torch.no_grad()
def calibrate_cold_start_readout_from_shard_(
    model: Any,
    shard_path: Path = DEFAULT_FINEWEB_TRAIN_SHARD1_PATH,
    calibration_tokens: int = 262_144,
    ridge_lambda: float = 0.25,
    logit_scale: float = 0.85,
    tokens_override: torch.Tensor | None = None,
) -> dict[str, float]:
    """Calibrate `lm_head.weight` at Step 0 from training tokens (strictly excluding Shard 0) via empirical-Bayes transition expectations (`<= 8` args)."""
    if tokens_override is not None:
        tok_1d = tokens_override.long().cpu()
    else:
        tok_1d = _load_fineweb_tokens_from_shard(shard_path, max_tokens=calibration_tokens + 1)

    dev = model.q_wte.device
    vocab_size = int(model.vocab_size)
    d_model = int(model.embed_dim)
    seq_len = 1024
    n_usable = ((tok_1d.numel() - 1) // seq_len) * seq_len
    if n_usable <= 0:
        seq_len = max(1, tok_1d.numel() - 1)
        n_usable = seq_len

    x_seq = tok_1d[:n_usable].view(-1, seq_len)
    y_seq = tok_1d[1 : n_usable + 1].view(-1, seq_len)

    sum_h_by_target = torch.zeros(vocab_size, d_model, device=dev, dtype=torch.float32)
    target_counts = torch.zeros(vocab_size, device=dev, dtype=torch.float32)
    cov_hh = torch.zeros(d_model, d_model, device=dev, dtype=torch.float32)
    mean_h = torch.zeros(d_model, device=dev, dtype=torch.float32)
    total_tok = 0

    was_training = model.training
    model.eval()
    batch_rows = 8
    for b_start in range(0, x_seq.size(0), batch_rows):
        xb = x_seq[b_start : b_start + batch_rows].to(dev)
        yb = y_seq[b_start : b_start + batch_rows].to(dev).reshape(-1)
        h_feat = model.forward_features(xb).float().reshape(-1, d_model)
        sum_h_by_target.index_add_(0, yb, h_feat)
        target_counts.scatter_add_(0, yb, torch.ones_like(yb, dtype=torch.float32))
        cov_hh.add_(torch.matmul(h_feat.t(), h_feat))
        mean_h.add_(h_feat.sum(dim=0))
        total_tok += int(yb.numel())

    if was_training:
        model.train()

    n_f = float(max(1, total_tok))
    cov_hh.div_(n_f)
    mean_h.div_(n_f)
    eye_d = torch.eye(d_model, device=dev, dtype=torch.float32)
    cov_inv = torch.linalg.solve(cov_hh + float(ridge_lambda) * eye_d, eye_d)

    prior_k = 10.0
    mu_prev = sum_h_by_target / (target_counts.unsqueeze(-1) + prior_k)
    w_trans = torch.matmul(mu_prev, cov_inv)
    w_dir = F.normalize(w_trans, dim=-1, eps=FIBER_EPS)
    eb_weight = float(logit_scale) * torch.sqrt(target_counts / (target_counts + 50.0)).unsqueeze(-1)
    w_head_new = (w_dir * eb_weight).to(model.classical.lm_head.weight.dtype)
    model.classical.lm_head.weight.copy_(w_head_new)
    return {
        "calibrated_tokens": float(total_tok),
        "active_target_tokens": float((target_counts > 0).sum().item()),
        "lm_head_rms": float(w_head_new.float().pow(2).mean().sqrt().item()),
    }


__all__ = [
    "DEFAULT_COMPLEX_PACKET_COUNT",
    "DEFAULT_COMPLEX_PACKET_HILBERT_DIM",
    "DEFAULT_DIRECTED_PPMI_CACHE_PATH",
    "DEFAULT_EB_ALPHA_MAX",
    "DEFAULT_EB_PRIOR_COUNT_K0",
    "DEFAULT_FIBER_BUNDLE_SEED",
    "DEFAULT_FIBER_SEMANTIC_PACKETS_10",
    "DEFAULT_FIBER_SEMANTIC_PACKETS_11",
    "DEFAULT_FINEWEB_TRAIN_SHARD1_PATH",
    "DEFAULT_IDENTITY_GATE_BIAS_INIT",
    "DEFAULT_IDENTITY_PACKET_LR_SCALE",
    "DEFAULT_MULTI_LAGS",
    "DEFAULT_MULTI_LAG_WEIGHTS",
    "DEFAULT_PPMI_CDS_POWER",
    "DEFAULT_PPMI_SHIFT_KAPPA",
    "DEFAULT_SEMANTIC_GATE_BIAS_INIT",
    "DynamicComplexPacketReadoutModulator",
    "QuantumPacketReadoutMode",
    "_hermitian_gram_schmidt_complex_blend",
    "active_only_center_rows",
    "apply_complex_unitary_mode_balance",
    "build_or_load_directed_complex_ppmi_sketch",
    "calibrate_cold_start_readout_from_shard_",
    "compute_directed_complex_ppmi_from_tokens",
    "compute_hermitian_orthogonal_residual",
    "initialize_directed_complex_ppmi_fiber_bundle_embeddings_",
    "project_complex_wavepackets_to_rms_sphere_",
    "sample_haar_complex_unitary",
]
