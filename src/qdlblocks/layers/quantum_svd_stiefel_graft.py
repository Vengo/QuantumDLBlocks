r"""Biorthogonal SVD-Stiefel Wavepacket Projections & Fubini-Study Self-State Orthogonalization (`EXPERIMENT_PLAN_068.md`).

Implements **Wave 52D** (`Axes K, L, M`) for Surgical Quantum Front-End Grafting onto a Frozen Classical GPT:
1. **Axis K — Biorthogonal SVD-Stiefel Operator Geometry (`QuantumStiefelProjMode`)**:
   - Parameterizes Layer-0 quantum wavepacket boundary projections $\alpha \in \{q, k, v, \text{obs}\}$ on the
     Biorthogonal SVD-Stiefel manifold $W_\alpha(x) = ((x V_\alpha) \odot \vec{\sigma}_\alpha) U_\alpha^\top$ with
     $U_\alpha \in \mathrm{St}(D_{\text{out}}, r_{\text{eff}})$, $V_\alpha \in \mathrm{St}(D_{\text{in}}, r_{\text{eff}})$,
     and $\vec{\sigma}_\alpha \in \mathbb{R}^{r_{\text{eff}}}$ across ranks $r \in \{128, 256, 512, 768\}$.
2. **Axis L — Fubini-Study Projective Self-State Orthogonalization (`QuantumSelfOrthoMode`)**:
   - Projects out the trivial self-echo component parallel to $|\hat{\psi}_{v, h}\rangle$ in Hilbert space via
     $|\psi_{\text{ortho}, h}\rangle = (I - \tanh(\alpha_h) |\hat{\psi}_{v, h}\rangle\langle\hat{\psi}_{v, h}|)|y_{\text{scaled}, h}\rangle$
     and calibrates Layer-0 residual highway scales (`block_lambdas`, `sa_lambdas`, `ve_gate`, `attn_gate`, `obs_scale`).
"""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.compiled_leaderboard_classical_gpt import (
    DEFAULT_BOS_TOKEN_ID,
    apply_native_leaderboard_rotary,
)
from .impure_density_mirror_block import TRACE_NORM_EPS, initialize_stiefel_parameter
from .quantum_statmech_channels import QuantumChannelMode
from .wavepacket_statmech_block import DEFAULT_PPMI_ORTHO_SEED

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_DEFAULT_PPMI_CACHE_PATH: Path = _REPO_ROOT / ".cache" / "ppmi_svd_50304_768_shard1.pt"

DEFAULT_SVD_STIEFEL_OBS_SCALE_INIT: float = 2.0 / 3.0
DEFAULT_SVD_STIEFEL_SIGMA_MAX: float = 50.0
DEFAULT_SVD_STIEFEL_OBS_SCALE_MIN: float = 0.02
DEFAULT_SVD_STIEFEL_OBS_SCALE_MAX: float = 5.00
DEFAULT_CALIBRATED_LAMBDA_BOUND: float = 30.0
DEFAULT_PARTIAL_KEY_SHIFT_HALF: int = 64
DEFAULT_WSD_COOLDOWN_FRAC: float = 0.35
DEFAULT_WSD_MIN_LR_RATIO: float = 0.08
DEFAULT_COLD_START_OBS_SCALE_INIT: float = 0.25


class QuantumStiefelProjMode(str, Enum):
    """Axis K & N enum selecting Kronecker-factored vs. Biorthogonal SVD-Stiefel projections (Rule 3)."""

    KRONEKER_FACTORED_STIEFEL = "kroneker_factored_stiefel"
    BIORTHOGONAL_SVD_STIEFEL_R64 = "biorthogonal_svd_stiefel_r64"
    BIORTHOGONAL_SVD_STIEFEL_R128 = "biorthogonal_svd_stiefel_r128"
    BIORTHOGONAL_SVD_STIEFEL_R192 = "biorthogonal_svd_stiefel_r192"
    BIORTHOGONAL_SVD_STIEFEL_R256 = "biorthogonal_svd_stiefel_r256"
    BIORTHOGONAL_SVD_STIEFEL_R384 = "biorthogonal_svd_stiefel_r384"
    BIORTHOGONAL_SVD_STIEFEL_R512 = "biorthogonal_svd_stiefel_r512"
    BIORTHOGONAL_SVD_STIEFEL_R768 = "biorthogonal_svd_stiefel_r768"


class QuantumSelfOrthoMode(str, Enum):
    """Axis L enum selecting Fubini-Study projective self-state orthogonalization (Rule 3)."""

    NONE = "none"
    FUBINI_STUDY_SELF_ORTHO = "fubini_study_self_ortho"


class QuantumQKPhaseMode(str, Enum):
    """Axis R enum selecting real dot-product vs. U(1) Berry-phase complex Hermitian QK coupling (Rule 3)."""

    STANDARD_REAL_DOT = "standard_real_dot"
    BERRY_PHASE_COMPLEX_QK = "berry_phase_complex_qk"


class QuantumDocSWCMode(str, Enum):
    """Axis U enum selecting full-1024 causal vs. Document-Boundary Sliding-Window Causal mask (Rule 3)."""

    FULL_CAUSAL_1024 = "full_causal_1024"
    TEACHER_DOC_SWC_MASK = "teacher_doc_swc_mask"


class QuantumDepthLRMode(str, Enum):
    """Axis V enum selecting uniform flat-then-cosine vs. depth-scaled/unscaled warmup-cosine or WSD cooldown schedule (Rule 3)."""

    UNIFORM_FLAT_THEN_COSINE = "uniform_flat_then_cosine"
    DEPTH_SQRT_WARMUP_COSINE = "depth_sqrt_warmup_cosine"
    WARMUP_COSINE_UNSCALED = "warmup_cosine_unscaled"
    DEPTH_SQRT_WARMUP_WSD_COOLDOWN = "depth_sqrt_warmup_wsd_cooldown"


def build_doc_swc_sdpa_masks(
    idx: torch.Tensor,
    long_win: int,
    short_win: int,
    final_win: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct `(mask_long, mask_short, mask_final)` of shape `(B, 1, T, T)` for Document-Boundary SWC SDPA (`<= 8` args)."""
    _, seq_len = idx.shape
    docs = (idx == DEFAULT_BOS_TOKEN_ID).cumsum(dim=1)
    q_pos = torch.arange(seq_len, device=idx.device).view(1, seq_len, 1)
    k_pos = torch.arange(seq_len, device=idx.device).view(1, 1, seq_len)
    delta = q_pos - k_pos
    causal_same_doc = (delta >= 0) & (docs.unsqueeze(2) == docs.unsqueeze(1))
    mask_long = (causal_same_doc & (delta < int(long_win))).unsqueeze(1)
    mask_short = (causal_same_doc & (delta < int(short_win))).unsqueeze(1)
    mask_final = (causal_same_doc & (delta < int(final_win))).unsqueeze(1)
    return mask_long, mask_short, mask_final


def compute_depth_lr_scale(n_sublayers: int) -> float:
    r"""Compute depth geodesic trust-region factor $\gamma_{\text{depth}}(N_{\text{sub}}) = 1 / \sqrt{\max(1, N_{\text{sub}} / 2)}$."""
    return 1.0 / math.sqrt(max(1.0, float(n_sublayers) / 2.0))


def compute_quantum_backbone_lr_multiplier(
    step: int,
    total_steps: int,
    depth_lr_mode: QuantumDepthLRMode = QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE,
) -> float:
    """Compute step learning-rate multiplier for `UNIFORM_FLAT_THEN_COSINE`, `DEPTH_SQRT_WARMUP_COSINE`, or `DEPTH_SQRT_WARMUP_WSD_COOLDOWN`."""
    t_tot = max(1, int(total_steps))
    s_cur = max(1, min(int(step), t_tot))
    mode = QuantumDepthLRMode(depth_lr_mode)
    if mode == QuantumDepthLRMode.UNIFORM_FLAT_THEN_COSINE:
        progress = float(s_cur - 1) / float(t_tot)
        if progress < 0.70:
            return 1.0
        return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * (progress - 0.70) / 0.30))
    t_warm = max(1, int(0.10 * float(t_tot)))
    if s_cur <= t_warm:
        if t_warm == 1:
            return 1.0
        return 0.10 + 0.90 * (float(s_cur - 1) / float(t_warm - 1))
    if mode == QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN:
        t_stable = max(t_warm, int(round((1.0 - DEFAULT_WSD_COOLDOWN_FRAC) * float(t_tot))))
        if s_cur <= t_stable:
            return 1.0
        cool_prog = float(s_cur - t_stable) / float(max(1, t_tot - t_stable))
        return 1.0 - (1.0 - DEFAULT_WSD_MIN_LR_RATIO) * min(1.0, max(0.0, cool_prog))
    rem = float(s_cur - t_warm) / float(max(1, t_tot - t_warm))
    return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * rem))


def apply_berry_qk_phase(q_rot: torch.Tensor, berry_phase_hrd: torch.Tensor) -> torch.Tensor:
    """Rotate complex query kets `(B, T, H, 2*R*d_h)` by learnable U(1) Berry phase `berry_phase_hrd` `(H, R, d_h)`."""
    bsz, seq_len, n_h, head_dim = q_rot.shape
    v_rank, d_h = berry_phase_hrd.size(1), berry_phase_hrd.size(2)
    qc = q_rot.view(bsz, seq_len, n_h, v_rank, 2, d_h)
    qr, qi = qc[..., 0, :], qc[..., 1, :]
    phi = berry_phase_hrd.to(q_rot.dtype).view(1, 1, n_h, v_rank, d_h)
    c_phi, s_phi = torch.cos(phi), torch.sin(phi)
    qr_out = c_phi * qr - s_phi * qi
    qi_out = s_phi * qr + c_phi * qi
    return torch.stack([qr_out, qi_out], dim=-2).reshape(bsz, seq_len, n_h, head_dim)


def initialize_arbitrary_shape_ppmi_sphere_embeddings_(
    wte_3d: torch.Tensor, ppmi_embeddings: torch.Tensor | None = None
) -> bool:
    r"""Initialize 3D `(V, M_slices, d_sphere)` embeddings from `(V, 768)` PPMI-SVD for $D_{\text{emb}} \in \{512, 768, 1024\}$."""
    from ..optim.vocab_manifold_muon import project_rows_to_rms_sphere_

    vocab_size, num_slices, d_sphere = wte_3d.shape
    d_emb = num_slices * d_sphere
    if ppmi_embeddings is None and vocab_size == 50304 and _DEFAULT_PPMI_CACHE_PATH.exists():
        raw_payload = torch.load(_DEFAULT_PPMI_CACHE_PATH, map_location="cpu", weights_only=True)
        ppmi_embeddings = raw_payload["embeddings"] if isinstance(raw_payload, dict) else raw_payload
    with torch.no_grad():
        if ppmi_embeddings is not None and ppmi_embeddings.size(0) == vocab_size:
            ppmi_f = ppmi_embeddings.detach().to(device="cpu", dtype=torch.float32)
            src_dim = ppmi_f.size(1)
            adapted = (
                ppmi_f[:, :d_emb].contiguous()
                if d_emb <= src_dim
                else torch.cat([ppmi_f, -ppmi_f[:, : (d_emb - src_dim)]], dim=1).contiguous()
            )
            gen = torch.Generator(device="cpu")
            gen.manual_seed(DEFAULT_PPMI_ORTHO_SEED + num_slices)
            q_m, r_m = torch.linalg.qr(torch.randn(num_slices, num_slices, generator=gen, dtype=torch.float32))
            diag_sign = torch.sign(torch.diagonal(r_m))
            q_m = q_m * torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign).unsqueeze(0)
            balanced_3d = torch.matmul(adapted.view(vocab_size, d_sphere, num_slices), q_m).transpose(1, 2).contiguous()
            project_rows_to_rms_sphere_(balanced_3d)
            wte_3d.copy_(balanced_3d.to(device=wte_3d.device, dtype=wte_3d.dtype))
            return True
        project_rows_to_rms_sphere_(wte_3d)
        return False


def initialize_teacher_procrustes_sphere_embeddings_(
    wte_3d: torch.Tensor,
    teacher_wte_weight: torch.Tensor,
    w_in_bridge: torch.Tensor | None = None,
    w_out_bridge: torch.Tensor | None = None,
) -> None:
    r"""Initialize `q_wte` and boundary Stiefel bridges from RMS-normalized `teacher_wte_weight` (`Axis E`)."""
    from ..optim.vocab_manifold_muon import project_rows_to_rms_sphere_

    vocab_size, num_slices, d_sphere = wte_3d.shape
    d_emb = num_slices * d_sphere
    d_teacher = teacher_wte_weight.size(1)
    with torch.no_grad():
        e_t = F.rms_norm(teacher_wte_weight.detach().to(device="cpu", dtype=torch.float32), (d_teacher,), eps=1e-6)
        if d_emb == d_teacher:
            z_emb = e_t
        elif d_emb < d_teacher:
            _, _, vh = torch.linalg.svd(e_t, full_matrices=(vocab_size < d_emb))
            q_mix = torch.empty(d_emb, d_emb, dtype=torch.float32)
            initialize_stiefel_parameter(q_mix)
            w_star = torch.matmul(vh[:d_emb, :].t(), q_mix).contiguous()
            if w_in_bridge is not None:
                w_in_bridge.copy_(w_star.to(device=w_in_bridge.device, dtype=w_in_bridge.dtype))
            if w_out_bridge is not None:
                w_out_bridge.copy_(w_star.to(device=w_out_bridge.device, dtype=w_out_bridge.dtype))
            z_emb = torch.matmul(e_t, w_star)
        else:
            w_star = torch.empty(d_emb, d_teacher, dtype=torch.float32)
            initialize_stiefel_parameter(w_star)
            if w_in_bridge is not None:
                w_in_bridge.copy_(w_star.to(device=w_in_bridge.device, dtype=w_in_bridge.dtype))
            if w_out_bridge is not None:
                w_out_bridge.copy_(w_star.to(device=w_out_bridge.device, dtype=w_out_bridge.dtype))
            z_emb = torch.matmul(e_t, w_star.t())
        z_3d = z_emb.view(vocab_size, num_slices, d_sphere).contiguous()
        project_rows_to_rms_sphere_(z_3d)
        wte_3d.copy_(z_3d.to(device=wte_3d.device, dtype=wte_3d.dtype))


def apply_boundary_stiefel_bridge(z_bt_demb: torch.Tensor, w_bridge: torch.Tensor | None) -> torch.Tensor:
    r"""Project `(B, T, D_emb)` to `(B, T, 768)` via tall Stiefel matrix $W_{\text{bridge}} \in \mathrm{St}_{\text{col}}(\max, \min)$."""
    if w_bridge is None:
        return z_bt_demb
    w_work = w_bridge.to(z_bt_demb.dtype)
    return torch.matmul(z_bt_demb, w_work.t() if z_bt_demb.size(-1) == w_work.size(1) else w_work)


def map_teacher_768_to_quantum_packets(
    z_bt_768: torch.Tensor, w_in_bridge: torch.Tensor | None, pkt_count: int, pkt_dim: int
) -> torch.Tensor:
    """Map a `(B, T, 768)` teacher tensor into `(B, T, pkt_count, pkt_dim)` quantum packet coordinates."""
    bsz, seq_len, _ = z_bt_768.shape
    if w_in_bridge is None:
        return z_bt_768.view(bsz, seq_len, pkt_count, pkt_dim)
    w_work = w_in_bridge.to(z_bt_768.dtype)
    proj = torch.matmul(z_bt_768, w_work if z_bt_768.size(-1) == w_work.size(0) else w_work.t())
    return proj.view(bsz, seq_len, pkt_count, pkt_dim)


def resolve_svd_stiefel_rank(proj_mode: QuantumStiefelProjMode, d_in: int, d_out: int) -> int:
    """Resolve effective SVD-Stiefel rank $r_{\\text{eff}} = \\min(r, D_{\\text{in}}, D_{\\text{out}})$."""
    mode = QuantumStiefelProjMode(proj_mode)
    max_r = min(int(d_in), int(d_out))
    if mode == QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64:
        return min(64, max_r)
    if mode == QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128:
        return min(128, max_r)
    if mode == QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R192:
        return min(192, max_r)
    if mode == QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R256:
        return min(256, max_r)
    if mode == QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384:
        return min(384, max_r)
    if mode == QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R512:
        return min(512, max_r)
    return max_r


def compute_liouville_commutator(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute exact closed-form Hermitian traceless commutator $C = -i[A, B] = C_R + i C_I$ via one complex GEMM."""
    p_re = torch.matmul(a_re, b_re) - torch.matmul(a_im, b_im)
    p_im = torch.matmul(a_re, b_im) + torch.matmul(a_im, b_re)
    c_re = p_im + p_im.transpose(-2, -1)
    d_h = c_re.size(-1)
    tr_mean = torch.diagonal(c_re, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    c_re = c_re - tr_mean * torch.eye(d_h, device=c_re.device, dtype=c_re.dtype)
    c_im = p_re.transpose(-2, -1) - p_re
    return c_re, c_im


def apply_interior_rho_v_coupling(
    vu_re: torch.Tensor,
    vu_im: torch.Tensor,
    lu_re: torch.Tensor,
    lu_im: torch.Tensor,
    evol_re: torch.Tensor,
    evol_im: torch.Tensor,
    interior_mode_val: str,
    head_param: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine rotated value kets `(vu_re, vu_im)` / `(lu_re, lu_im)` with evolved context density `(evol_re, evol_im)`."""
    if interior_mode_val == "attended_rho_on_local_v":
        wp_u_re = vu_re + 0.5 * (torch.matmul(lu_re, evol_re) + torch.matmul(lu_im, evol_im))
        wp_u_im = vu_im + 0.5 * (torch.matmul(lu_im, evol_re) - torch.matmul(lu_re, evol_im))
        return wp_u_re, wp_u_im
    if interior_mode_val == "symmetrized_local_context_rho_v":
        a_re = torch.matmul(lu_re.transpose(-2, -1), lu_re) + torch.matmul(lu_im.transpose(-2, -1), lu_im)
        a_im = torch.matmul(lu_im.transpose(-2, -1), lu_re) - torch.matmul(lu_re.transpose(-2, -1), lu_im)
        alpha = (
            torch.sigmoid(head_param).to(evol_re.dtype).view(1, -1, 1, 1, 1)
            if head_param is not None
            else 0.25
        )
        sym_re = (1.0 - alpha) * evol_re + alpha * a_re
        sym_im = (1.0 - alpha) * evol_im + alpha * a_im
        wp_u_re = vu_re + 0.5 * (torch.matmul(vu_re, sym_re) + torch.matmul(vu_im, sym_im))
        wp_u_im = vu_im + 0.5 * (torch.matmul(vu_im, sym_re) - torch.matmul(vu_re, sym_im))
        return wp_u_re, wp_u_im
    if interior_mode_val == "liouville_commutator_rho_v":
        a_re = torch.matmul(lu_re.transpose(-2, -1), lu_re) + torch.matmul(lu_im.transpose(-2, -1), lu_im)
        a_im = torch.matmul(lu_im.transpose(-2, -1), lu_re) - torch.matmul(lu_re.transpose(-2, -1), lu_im)
        c_re, c_im = compute_liouville_commutator(a_re, a_im, evol_re, evol_im)
        beta = (
            torch.tanh(head_param).to(evol_re.dtype).view(1, -1, 1, 1, 1)
            if head_param is not None
            else 0.25
        )
        eff_re = evol_re + beta * c_re
        eff_im = evol_im + beta * c_im
        wp_u_re = vu_re + 0.5 * (torch.matmul(vu_re, eff_re) + torch.matmul(vu_im, eff_im))
        wp_u_im = vu_im + 0.5 * (torch.matmul(vu_im, eff_re) - torch.matmul(vu_re, eff_im))
        return wp_u_re, wp_u_im
    wp_u_re = vu_re + 0.5 * (torch.matmul(vu_re, evol_re) + torch.matmul(vu_im, evol_im))
    wp_u_im = vu_im + 0.5 * (torch.matmul(vu_im, evol_re) - torch.matmul(vu_re, evol_im))
    return wp_u_re, wp_u_im


def interpolate_head_vector(src_vec: torch.Tensor, target_heads: int) -> torch.Tensor:
    """Resample a 1D or 2D `(H_src, ...)` per-head parameter tensor to `target_heads` in `float32`."""
    src_f = src_vec.detach().float()
    h_src = src_f.size(0)
    if h_src == target_heads:
        return src_f.clone()
    if src_f.ndim == 1:
        inp = src_f.view(1, 1, h_src)
        out = F.interpolate(inp, size=target_heads, mode="linear", align_corners=True)
        return out.view(target_heads).contiguous()
    cols = src_f.size(1)
    inp2 = src_f.t().unsqueeze(0)
    out2 = F.interpolate(inp2, size=target_heads, mode="linear", align_corners=True)
    return out2.squeeze(0).t().reshape(target_heads, cols).contiguous()


def apply_fubini_study_self_ortho(
    y_scaled: torch.Tensor,
    v_mixed: torch.Tensor,
    xsa_alpha: torch.Tensor,
) -> torch.Tensor:
    r"""Apply rank-1 Fubini-Study projective self-state orthogonalizer $(I - \tanh(\alpha_h)|\hat{\psi}_v\rangle\langle\hat{\psi}_v|)$."""
    work_dtype = y_scaled.dtype
    v_unit = F.normalize(v_mixed.float(), dim=-1, eps=1e-5).to(work_dtype)
    overlap = (y_scaled * v_unit).sum(dim=-1, keepdim=True)
    alpha_h = torch.tanh(xsa_alpha).view(1, 1, -1, 1).to(work_dtype)
    return y_scaled - alpha_h * (overlap * v_unit)


class BiorthogonalSVDStiefelProjection(nn.Module):
    r"""Biorthogonal SVD-Stiefel Operator $W(x) = ((x V) \odot \vec{\sigma}) U^\top$ on $\mathrm{St}(D_{\text{out}}, r) \times \mathrm{St}(D_{\text{in}}, r)$."""

    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.rank = min(int(rank), self.d_in, self.d_out)

        u_init = torch.empty(self.d_out, self.rank, dtype=torch.float32)
        v_init = torch.empty(self.d_in, self.rank, dtype=torch.float32)
        initialize_stiefel_parameter(u_init)
        initialize_stiefel_parameter(v_init)
        self.u_stiefel = nn.Parameter(u_init)
        self.v_stiefel = nn.Parameter(v_init)
        self.sigma = nn.Parameter(torch.ones(self.rank, dtype=torch.float32))

    @torch.no_grad()
    def initialize_from_teacher_matrix_(
        self,
        w_teacher: torch.Tensor,
        w_bridge: torch.Tensor | None = None,
        bridge_on_output: bool = True,
    ) -> None:
        """Initialize `(u_stiefel, sigma, v_stiefel)` via thin SVD of teacher matrix mapped into wavepacket space."""
        wt = w_teacher.detach().to(device="cpu", dtype=torch.float32)
        if w_bridge is not None:
            b_cpu = w_bridge.detach().to(device="cpu", dtype=torch.float32)
            if bridge_on_output:
                wt = torch.matmul(b_cpu if b_cpu.size(0) == self.d_out else b_cpu.t(), wt)
            else:
                wt = torch.matmul(wt, b_cpu.t() if b_cpu.size(0) == self.d_in else b_cpu)
        u_svd, s_svd, vh_svd = torch.linalg.svd(wt.float(), full_matrices=False)
        r = self.rank
        self.u_stiefel.copy_(u_svd[:, :r].contiguous().to(device=self.u_stiefel.device, dtype=self.u_stiefel.dtype))
        self.sigma.copy_(s_svd[:r].contiguous().to(device=self.sigma.device, dtype=self.sigma.dtype))
        self.v_stiefel.copy_(
            vh_svd[:r, :].t().contiguous().to(device=self.v_stiefel.device, dtype=self.v_stiefel.dtype)
        )

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return tall column-orthonormal Stiefel factors `[u_stiefel, v_stiefel]`."""
        return [self.u_stiefel, self.v_stiefel]

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return singular value spectrum parameter `[sigma]`."""
        return [self.sigma]

    @torch.no_grad()
    def clamp_sigma_(self) -> None:
        """Clamp singular values to physical non-negative bounded interval `[0.0, DEFAULT_SVD_STIEFEL_SIGMA_MAX]`."""
        self.sigma.clamp_(0.0, DEFAULT_SVD_STIEFEL_SIGMA_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply `((x @ v_stiefel) * sigma) @ u_stiefel.t()` in `x.dtype`."""
        dt = x.dtype
        xv = torch.matmul(x, self.v_stiefel.to(dt))
        xvs = xv * self.sigma.to(dt)
        return torch.matmul(xvs, self.u_stiefel.to(dt).t())


class SVDStiefelWavepacketStatMechBlock(nn.Module):
    r"""Quantum Wavepacket Outer-Product StatMech Layer 0 with Biorthogonal SVD-Stiefel Projections & Fubini-Study Self-Ortho."""

    def __init__(
        self,
        core_block: Any,
        d_model: int,
        d_emb: int,
        proj_mode: QuantumStiefelProjMode,
        self_ortho_mode: QuantumSelfOrthoMode,
        gate_in_ch: int = 16,
        qk_phase_mode: QuantumQKPhaseMode = QuantumQKPhaseMode.STANDARD_REAL_DOT,
    ) -> None:
        super().__init__()
        self.core_block = core_block
        self.core_block.disable_mach_zehnder_()
        if getattr(self.core_block.state_io_mode, "value", "") == "dm_in_dm_out":
            self.core_block.state_io_mode = type(self.core_block.state_io_mode)("causal_ensemble_dm_io")
        for p in self.core_block.real_stiefel_parameters():
            p.requires_grad_(False)

        self.d_model = int(d_model)
        self.d_emb = int(d_emb)
        self.num_pkts = int(core_block.num_pkts)
        self.d_h = int(core_block.d_h)
        self.num_heads = int(core_block.num_heads)
        self.v_rank = int(core_block.v_rank)
        self.head_dim = 2 * self.v_rank * self.d_h
        self.proj_mode = QuantumStiefelProjMode(proj_mode)
        self.self_ortho_mode = QuantumSelfOrthoMode(self_ortho_mode)
        self.qk_phase_mode = QuantumQKPhaseMode(qk_phase_mode)
        self.gate_in_ch = min(int(gate_in_ch), self.d_model)
        self.layer_idx: int = 0
        self.is_long_window_layer: bool = True
        self.ket_amplitude_mode: str = "entropy_preserving_coherent_ket"
        self.attn_kernel_mode: str = "expanded_sdpa_chunks"
        self.use_partial_key_offset: bool = True
        self.sdpa_scale: float = 0.22
        self.ve_part_ch: int = 6
        self.q_ve_gate: nn.Linear | None = None
        self.rho_coupling_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32), requires_grad=False)
        self.qk_born_beta = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32), requires_grad=False)

        r_qkv = resolve_svd_stiefel_rank(self.proj_mode, self.d_model, self.d_emb)
        r_obs = resolve_svd_stiefel_rank(self.proj_mode, self.d_emb, self.d_model)
        self.q_proj = BiorthogonalSVDStiefelProjection(self.d_model, self.d_emb, r_qkv)
        self.k_proj = BiorthogonalSVDStiefelProjection(self.d_model, self.d_emb, r_qkv)
        self.v_proj = BiorthogonalSVDStiefelProjection(self.d_model, self.d_emb, r_qkv)
        self.obs_proj = BiorthogonalSVDStiefelProjection(self.d_emb, self.d_model, r_obs)

        self.block_lambdas = self.core_block.block_lambdas
        self.obs_scale = self.core_block.obs_scale
        self.sa_lambdas = nn.Parameter(torch.tensor([1.0, 0.5], dtype=torch.float32))
        self.q_xsa_alpha = nn.Parameter(
            torch.zeros(self.num_heads, dtype=torch.float32),
            requires_grad=(self.self_ortho_mode == QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO),
        )
        self.berry_qk_phase = nn.Parameter(
            torch.zeros(self.num_heads, self.v_rank, self.d_h, dtype=torch.float32),
            requires_grad=(self.qk_phase_mode == QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK),
        )
        self.q_attn_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
        with torch.no_grad():
            self.q_attn_gate.weight.zero_()
            self.obs_scale.fill_(DEFAULT_SVD_STIEFEL_OBS_SCALE_INIT)
            chan_mode = getattr(self.core_block, "channel_mode", QuantumChannelMode.UNITARY_U16)
            if chan_mode != QuantumChannelMode.CPTP_KRAUS_RANK_K:
                u_count, u_dim = self.core_block.head_unitary_re.size(0), self.core_block.head_unitary_re.size(-1)
                eye_h = (
                    torch.eye(u_dim, device=self.obs_scale.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(u_count, -1, -1)
                    .clone()
                )
                self.core_block.head_unitary_re.copy_(eye_h)
                self.core_block.head_unitary_im.zero_()

    @torch.no_grad()
    def calibrate_from_teacher_block0_(
        self,
        teacher_blk0: Any,
        w_in_bridge: torch.Tensor | None = None,
    ) -> None:
        """Calibrate SVD-Stiefel projections, `block_lambdas`, `sa_lambdas`, `q_xsa_alpha`, and `q_attn_gate` from `teacher_blk0`."""
        self.layer_idx = int(getattr(teacher_blk0, "layer_idx", 0))
        self.is_long_window_layer = bool(getattr(teacher_blk0, "is_long_window_layer", True))
        self.use_partial_key_offset = bool(getattr(teacher_blk0, "use_partial_key_offset", True))
        self.sdpa_scale = float(getattr(teacher_blk0, "attn_scale", 0.12))
        if teacher_blk0.qkv_proj is not None and teacher_blk0.out_proj is not None:
            wq, wk, wv = teacher_blk0.qkv_proj.weight.detach().float().chunk(3, dim=0)
            wout = teacher_blk0.out_proj.weight.detach().float()
            self.q_proj.initialize_from_teacher_matrix_(wq, w_in_bridge, bridge_on_output=True)
            self.k_proj.initialize_from_teacher_matrix_(wk, w_in_bridge, bridge_on_output=True)
            self.v_proj.initialize_from_teacher_matrix_(wv, w_in_bridge, bridge_on_output=True)
            self.obs_proj.initialize_from_teacher_matrix_(wout, w_in_bridge, bridge_on_output=False)
        self.block_lambdas.copy_(teacher_blk0.block_lambdas.detach().to(self.block_lambdas.dtype))
        if teacher_blk0.sa_lambdas is not None:
            self.sa_lambdas.copy_(teacher_blk0.sa_lambdas.detach().to(self.sa_lambdas.dtype))
        if teacher_blk0.xsa_alpha is not None:
            alpha_rs = interpolate_head_vector(teacher_blk0.xsa_alpha, self.num_heads)
            self.q_xsa_alpha.copy_(alpha_rs.to(device=self.q_xsa_alpha.device, dtype=self.q_xsa_alpha.dtype))
        if teacher_blk0.attn_gate is not None:
            gw_rs = interpolate_head_vector(teacher_blk0.attn_gate.weight[:, : self.gate_in_ch], self.num_heads)
            self.q_attn_gate.weight.copy_(
                gw_rs.to(device=self.q_attn_gate.weight.device, dtype=self.q_attn_gate.weight.dtype)
            )
        if getattr(teacher_blk0, "ve_gate", None) is not None:
            self.ve_part_ch = int(getattr(teacher_blk0, "ve_part_ch", 6))
            self.q_ve_gate = nn.Linear(2 * self.ve_part_ch, self.num_heads, bias=False).to(
                device=self.q_attn_gate.weight.device, dtype=self.q_attn_gate.weight.dtype
            )
            vw_rs = interpolate_head_vector(teacher_blk0.ve_gate.weight, self.num_heads)
            self.q_ve_gate.weight.copy_(vw_rs.to(device=self.q_ve_gate.weight.device, dtype=self.q_ve_gate.weight.dtype))
        else:
            self.q_ve_gate = None

    @torch.no_grad()
    def calibrate_cold_start_block_(self, structural_blk: Any) -> None:
        """Configure structural metadata (`layer_idx`, window flags, zero gates, unit `sigma`) for cold-start from-scratch training."""
        self.layer_idx = int(getattr(structural_blk, "layer_idx", 0))
        self.is_long_window_layer = bool(getattr(structural_blk, "is_long_window_layer", True))
        self.use_partial_key_offset = bool(getattr(structural_blk, "use_partial_key_offset", True))
        self.sdpa_scale = float(getattr(structural_blk, "attn_scale", 0.12))
        self.block_lambdas.copy_(structural_blk.block_lambdas.detach().to(self.block_lambdas.dtype))
        if getattr(structural_blk, "sa_lambdas", None) is not None:
            self.sa_lambdas.copy_(structural_blk.sa_lambdas.detach().to(self.sa_lambdas.dtype))
        self.q_xsa_alpha.zero_()
        self.q_attn_gate.weight.zero_()
        self.obs_scale.fill_(DEFAULT_COLD_START_OBS_SCALE_INIT)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            proj.sigma.fill_(1.0)
        if getattr(structural_blk, "ve_gate", None) is not None:
            self.ve_part_ch = int(getattr(structural_blk, "ve_part_ch", 6))
            self.q_ve_gate = nn.Linear(2 * self.ve_part_ch, self.num_heads, bias=False).to(
                device=self.q_attn_gate.weight.device, dtype=self.q_attn_gate.weight.dtype
            )
            self.q_ve_gate.weight.zero_()
        else:
            self.q_ve_gate = None

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all 8 tall column-orthonormal Stiefel matrices from `(q_proj, k_proj, v_proj, obs_proj)`."""
        params: list[nn.Parameter] = []
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            params.extend(proj.real_stiefel_parameters())
        return params

    def complex_unitary_pair(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return complex head unitary/Kraus pair from `core_block`."""
        return self.core_block.complex_unitary_pair()

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return active 1D scalar and gate parameters for `q_scalar_opt`."""
        params: list[nn.Parameter] = []
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            params.extend(proj.scalar_parameters())
        params.extend(self.core_block.scalar_parameters())
        params.append(self.sa_lambdas)
        if self.rho_coupling_scale.requires_grad:
            params.append(self.rho_coupling_scale)
        if self.q_xsa_alpha.requires_grad:
            params.append(self.q_xsa_alpha)
        if self.berry_qk_phase.requires_grad:
            params.append(self.berry_qk_phase)
        if self.qk_born_beta.requires_grad:
            params.append(self.qk_born_beta)
        for p in self.q_attn_gate.parameters():
            if p.requires_grad:
                params.append(p)
        if self.q_ve_gate is not None:
            for p in self.q_ve_gate.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp singular values and scalar gains to bounded physical intervals."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            proj.clamp_sigma_()
        self.obs_scale.clamp_(DEFAULT_SVD_STIEFEL_OBS_SCALE_MIN, DEFAULT_SVD_STIEFEL_OBS_SCALE_MAX)
        self.rho_coupling_scale.clamp_(0.002, 1.00)
        self.block_lambdas.clamp_(-DEFAULT_CALIBRATED_LAMBDA_BOUND, DEFAULT_CALIBRATED_LAMBDA_BOUND)
        self.sa_lambdas.clamp_(-5.0, 5.0)
        self.q_xsa_alpha.clamp_(-4.0, 4.0)
        self.berry_qk_phase.clamp_(-3.1416, 3.1416)
        self.qk_born_beta.clamp_(-1.50, 1.50)
        if hasattr(self.core_block, "dm_res_mix"):
            self.core_block.dm_res_mix.clamp_(0.01, 1.00)
        if hasattr(self.core_block, "dm_lag_mix"):
            self.core_block.dm_lag_mix.clamp_(0.01, 1.00)
        if hasattr(self.core_block, "local_rho_mix_logit"):
            self.core_block.local_rho_mix_logit.clamp_(-4.0, 4.0)
        if hasattr(self.core_block, "liouville_beta"):
            self.core_block.liouville_beta.clamp_(-2.0, 2.0)

    def get_last_telemetry_dict(self) -> dict[str, float]:
        """Delegate density-matrix purity and effective rank telemetry to `core_block`."""
        return self.core_block.get_last_telemetry_dict()

    def forward(
        self,
        x_0_768: torch.Tensor,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
        q_scale: torch.Tensor,
        ve_wp_gated: torch.Tensor | None = None,
        x_anchor_768: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute SVD-Stiefel Quantum Wavepacket Outer-Product Layer returning `(x_mid_768, d_attn_768, v_wp)` (`<= 8` args)."""
        bsz, seq_len, _ = x_0_768.shape
        work_dtype = x_0_768.dtype
        l0, l1 = self.block_lambdas[0].to(work_dtype), self.block_lambdas[1].to(work_dtype)
        x_in_768 = l0 * x_0_768 + l1 * x_anchor_768.to(work_dtype) if x_anchor_768 is not None else (l0 + l1) * x_0_768
        h_768 = F.rms_norm(x_in_768, (self.d_model,), eps=TRACE_NORM_EPS)

        q_wp = self.q_proj(h_768).view(bsz, seq_len, self.num_pkts, self.d_h)
        k_wp = self.k_proj(h_768).view(bsz, seq_len, self.num_pkts, self.d_h)
        v_wp = self.v_proj(h_768).view(bsz, seq_len, self.num_pkts, self.d_h)

        s0, s1 = self.sa_lambdas[0].to(work_dtype), self.sa_lambdas[1].to(work_dtype)
        if ve_wp_gated is not None:
            v_mixed = s0 * v_wp + s1 * ve_wp_gated.to(work_dtype)
        else:
            v_mixed = s0 * v_wp

        q_heads = F.rms_norm(q_wp.view(bsz, seq_len, self.num_heads, self.head_dim), (self.head_dim,), eps=TRACE_NORM_EPS)
        k_heads = F.rms_norm(k_wp.view(bsz, seq_len, self.num_heads, self.head_dim), (self.head_dim,), eps=TRACE_NORM_EPS)
        q_rot = apply_native_leaderboard_rotary(q_heads, cos_4d, sin_4d) * q_scale.to(work_dtype)
        if self.qk_phase_mode == QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK:
            q_rot = apply_berry_qk_phase(q_rot, self.berry_qk_phase)
        k_rot = apply_native_leaderboard_rotary(k_heads, cos_4d, sin_4d)
        if self.use_partial_key_offset and seq_len > 1 and self.head_dim == 2 * DEFAULT_PARTIAL_KEY_SHIFT_HALF:
            d_half = DEFAULT_PARTIAL_KEY_SHIFT_HALF
            k_shifted = torch.cat([k_rot[:, :1, :, d_half:], k_rot[:, :-1, :, d_half:]], dim=1)
            k_rot = torch.cat([k_rot[..., :d_half], k_shifted], dim=-1)

        h_pkt = F.rms_norm(v_wp, (self.d_h,), eps=TRACE_NORM_EPS)
        y_wp = self.core_block._run_attention_and_evolution(
            h_pkt, q_rot.view(bsz, seq_len, self.num_pkts, self.d_h), q_rot, k_rot, v_mixed
        )
        v_mixed_h = v_mixed.view(bsz, seq_len, self.num_heads, self.head_dim)
        v_head_rms = v_mixed_h.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS).sqrt().to(work_dtype)
        y_scaled = y_wp.view(bsz, seq_len, self.num_heads, self.head_dim) * v_head_rms * self.obs_scale.to(work_dtype)

        if self.self_ortho_mode == QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO:
            y_ortho = apply_fubini_study_self_ortho(y_scaled, v_mixed_h, self.q_xsa_alpha)
        else:
            y_ortho = y_scaled

        g_attn = (
            2.0 * torch.sigmoid(F.linear(h_768[..., : self.gate_in_ch], self.q_attn_gate.weight.to(work_dtype)))
        ).unsqueeze(-1)
        y_gated = (y_ortho * g_attn).reshape(bsz, seq_len, self.d_emb)
        d_attn_768 = self.obs_proj(y_gated)
        x_mid_768 = x_in_768 + d_attn_768
        return x_mid_768, d_attn_768, v_wp


def encode_svd_stiefel_layer0(
    gpt_model: Any,
    idx: torch.Tensor,
    raw_wp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute calibrated SVD-Stiefel Quantum Layer 0 (`quantum_block0` + optional frozen `MLP0`) returning `(x_0, x_1, v_0)`."""
    bsz, seq_len = idx.shape
    work_dtype = raw_wp.dtype
    raw_emb = raw_wp.reshape(bsz, seq_len, gpt_model.d_emb)
    raw_768 = apply_boundary_stiefel_bridge(raw_emb, gpt_model.w_in_bridge)
    if gpt_model.classical.smear_gate is not None and gpt_model.classical.smear_lambda is not None and seq_len > 1:
        s_gate = gpt_model.classical.smear_lambda.to(work_dtype) * torch.sigmoid(
            gpt_model.classical.smear_gate(raw_768[:, 1:, : gpt_model.classical.smear_in_ch])
        )
        raw_768 = torch.cat([raw_768[:, :1], raw_768[:, 1:] + s_gate * raw_768[:, :-1]], dim=1)
    x_0 = F.rms_norm(raw_768, (gpt_model.embed_dim,), eps=gpt_model.cfg.eps) * gpt_model.in_scale.to(work_dtype)
    cos_4d, sin_4d = gpt_model.classical.rope.get_cos_sin(seq_len, dtype=work_dtype)
    q_scale = gpt_model.classical.rope.yarn_q_scale
    ve_wp_gated: torch.Tensor | None = None
    blk0 = gpt_model.classical.blocks[0]
    if (
        gpt_model.graft_cfg.ve_coupling_mode.value == "frozen_teacher_ve0_coupled"
        and gpt_model.classical.value_embeds is not None
        and len(gpt_model.classical.value_embeds) > 0
    ):
        with torch.no_grad():
            ve0_768 = gpt_model.classical.value_embeds[0](idx).to(work_dtype)
            if blk0.ve_gate is not None:
                h_0_c = F.rms_norm(
                    ((blk0.block_lambdas[0] + blk0.block_lambdas[1]).to(work_dtype)) * x_0,
                    (gpt_model.embed_dim,),
                    eps=gpt_model.cfg.eps,
                )
                c_ve = blk0.ve_part_ch
                ve_in = torch.cat([h_0_c[..., :c_ve], ve0_768[..., :c_ve]], dim=-1)
                g_ve = (2.0 * torch.sigmoid(blk0.ve_gate(ve_in))).unsqueeze(-1)
                ve0_768 = (ve0_768.view(bsz, seq_len, gpt_model.num_heads, -1) * g_ve).reshape(
                    bsz, seq_len, gpt_model.embed_dim
                )
        ve_wp_gated = map_teacher_768_to_quantum_packets(
            ve0_768, gpt_model.w_in_bridge, gpt_model.pkt_count, gpt_model.pkt_dim
        )
    x_mid_768, _, v_wp = gpt_model.quantum_block0(x_0, cos_4d, sin_4d, q_scale, ve_wp_gated=ve_wp_gated)
    v_0_proj = apply_boundary_stiefel_bridge(v_wp.reshape(bsz, seq_len, gpt_model.d_emb), gpt_model.w_in_bridge)
    v_0 = v_0_proj.view(bsz, seq_len, gpt_model.num_heads, gpt_model.embed_dim // gpt_model.num_heads)
    if gpt_model.graft_cfg.replacement_scope.value in ("attn0_quantum_keep_frozen_mlp0", "emb_plus_l0_l1"):
        h_mlp = F.rms_norm(x_mid_768, (gpt_model.embed_dim,), eps=gpt_model.cfg.eps)
        x_1 = x_mid_768 + blk0.fc2(F.relu(blk0.fc1(h_mlp)).square())
    else:
        x_1 = x_mid_768
    return x_0, x_1, v_0


def encode_svd_stiefel_layer1(
    gpt_model: Any,
    idx: torch.Tensor,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    v_0: torch.Tensor,
) -> torch.Tensor:
    """Execute calibrated SVD-Stiefel Quantum Layer 1 (`quantum_block1` + frozen `blocks[1].mlp`) returning `x_2`."""
    bsz, seq_len = idx.shape
    work_dtype = x_1.dtype
    cos_4d, sin_4d = gpt_model.classical.rope.get_cos_sin(seq_len, dtype=work_dtype)
    q_scale = gpt_model.classical.rope.yarn_q_scale
    blk1 = gpt_model.classical.blocks[1]
    ve1_wp_gated: torch.Tensor | None = None
    bank_idx = gpt_model.classical._ve_bank_index(1)
    if (
        gpt_model.graft_cfg.ve_coupling_mode.value == "frozen_teacher_ve0_coupled"
        and gpt_model.classical.value_embeds is not None
        and bank_idx is not None
        and bank_idx < len(gpt_model.classical.value_embeds)
    ):
        with torch.no_grad():
            ve1_768 = gpt_model.classical.value_embeds[bank_idx](idx).to(work_dtype)
            if blk1.ve_gate is not None:
                x_in_1 = blk1.block_lambdas[0].to(work_dtype) * x_1 + blk1.block_lambdas[1].to(work_dtype) * x_0
                h_1_c = F.rms_norm(x_in_1, (gpt_model.embed_dim,), eps=gpt_model.cfg.eps)
                c_ve = blk1.ve_part_ch
                ve_in = torch.cat([h_1_c[..., :c_ve], ve1_768[..., :c_ve]], dim=-1)
                g_ve = (2.0 * torch.sigmoid(blk1.ve_gate(ve_in))).unsqueeze(-1)
                ve1_768 = (ve1_768.view(bsz, seq_len, gpt_model.num_heads, -1) * g_ve).reshape(
                    bsz, seq_len, gpt_model.embed_dim
                )
        ve1_wp_gated = map_teacher_768_to_quantum_packets(
            ve1_768, gpt_model.w_in_bridge, gpt_model.pkt_count, gpt_model.pkt_dim
        )
    elif gpt_model.cfg.interior_value_mode == "v0_residual":
        ve1_wp_gated = map_teacher_768_to_quantum_packets(
            v_0.reshape(bsz, seq_len, gpt_model.embed_dim),
            gpt_model.w_in_bridge,
            gpt_model.pkt_count,
            gpt_model.pkt_dim,
        )
    x_mid_1, _, _ = gpt_model.quantum_block1(
        x_1, cos_4d, sin_4d, q_scale, ve_wp_gated=ve1_wp_gated, x_anchor_768=x_0
    )
    h_mlp1 = F.rms_norm(x_mid_1, (gpt_model.embed_dim,), eps=gpt_model.cfg.eps)
    return x_mid_1 + blk1.fc2(F.relu(blk1.fc1(h_mlp1)).square())


@torch.no_grad()
def compute_teacher_layer1_target(
    gpt_model: Any,
    idx: torch.Tensor,
    x_0_c: torch.Tensor,
    x_1_c: torch.Tensor,
    v_0_c: torch.Tensor,
) -> torch.Tensor:
    """Compute detached classical teacher Layer-1 post-MLP target `x_2_c` for multi-layer interface warmup."""
    _, seq_len = idx.shape
    cos_4d, sin_4d = gpt_model.classical.rope.get_cos_sin(seq_len, dtype=x_1_c.dtype)
    q_scale = gpt_model.classical.rope.yarn_q_scale
    bm_long, bm_short = gpt_model.classical._build_flex_block_masks(idx)
    bank_idx = gpt_model.classical._ve_bank_index(1)
    ve_1 = (
        gpt_model.classical.value_embeds[bank_idx](idx)
        if (
            gpt_model.classical.value_embeds is not None
            and bank_idx is not None
            and bank_idx < len(gpt_model.classical.value_embeds)
        )
        else None
    )
    blk1 = gpt_model.classical.blocks[1]
    layer_bm = bm_long if blk1.is_long_window_layer else bm_short
    x_2_c, _ = blk1(x_1_c, x_0_c, cos_4d, sin_4d, v_0=v_0_c, ve_l=ve_1, block_mask=layer_bm, q_scale=q_scale)
    return x_2_c.detach()


__all__ = [
    "BiorthogonalSVDStiefelProjection",
    "DEFAULT_COLD_START_OBS_SCALE_INIT",
    "DEFAULT_SVD_STIEFEL_OBS_SCALE_INIT",
    "DEFAULT_WSD_COOLDOWN_FRAC",
    "DEFAULT_WSD_MIN_LR_RATIO",
    "QuantumDepthLRMode",
    "QuantumDocSWCMode",
    "QuantumQKPhaseMode",
    "QuantumSelfOrthoMode",
    "QuantumStiefelProjMode",
    "SVDStiefelWavepacketStatMechBlock",
    "apply_berry_qk_phase",
    "apply_boundary_stiefel_bridge",
    "apply_fubini_study_self_ortho",
    "apply_interior_rho_v_coupling",
    "build_doc_swc_sdpa_masks",
    "compute_depth_lr_scale",
    "compute_liouville_commutator",
    "compute_quantum_backbone_lr_multiplier",
    "compute_teacher_layer1_target",
    "encode_svd_stiefel_layer0",
    "encode_svd_stiefel_layer1",
    "initialize_arbitrary_shape_ppmi_sphere_embeddings_",
    "initialize_teacher_procrustes_sphere_embeddings_",
    "interpolate_head_vector",
    "map_teacher_768_to_quantum_packets",
    "resolve_svd_stiefel_rank",
]
