"""Krylov-Probed Packed Complex Register Density Attention (`Wave 37` / `EXPERIMENT_PLAN_052.md`).

When to Use This Building Block:
- Use `PackedComplexRegisterDensityAttention`, `ComplexRoPE`, `ComplexLinear`,
  `pack_hermitian_outer_product`, `unpack_hermitian_matrix`, and `krylov_purified_density_probe`
  when you want:
  1. Native complex rotary position embeddings (`ComplexRoPE`) applied strictly to queries and keys
     ($q_t^{\text{rot}} = q_t e^{i t \boldsymbol{\omega}}$, $k_s^{\text{rot}} = k_s e^{i s \boldsymbol{\omega}}$)
     for shift-invariant Hermitian inner products $c_{ts} = \langle q_t^{\text{rot}} \mid k_s^{\text{rot}} \rangle_{\mathbb{C}}$,
     while keeping value kets $|v_s\rangle \in \mathbb{C}^{d_h}$ un-rotated in position-invariant content $V$-space
     (**Refinement 1**).
  2. Softmax-free $p$-boson Born probability weights $w_{ts} = \frac{|c_{ts}|^{2p}\,\mathbb{I}(s \le t)}{\sum_{j \le t} |c_{tj}|^{2p} + \varepsilon}$
     ($p = 2$, zero `exp`, and zero $\gamma_{\text{coh}}$ coherent-ket branch).
  3. Fused single-allocation Hermitian Triangle Packing & Unpacking (`pack_hermitian_outer_product_and_ket`
     and `unpack_hermitian_matrix`) that packs the symmetric real upper triangle $\operatorname{triu}(\rho_{\text{re}}, 0)$,
     the skew-symmetric imaginary strict lower triangle $\operatorname{tril}(\rho_{\text{im}}, -1)$, and the
     first-moment value ket $[v_{\text{re}}, v_{\text{im}}]$ into a single real Tensor-Core GEMM of width
     $d_h^2 + 2 d_h$ (**Refinements 3 & 4**).
  4. $V$-Space Krylov Power Iteration probing (`krylov_purified_density_probe`) that probes the McWeeny-purified
     density operator $\rho_t^{\text{purified}} = \frac{\rho_t^2}{\operatorname{Tr}(\rho_t^2)}$ with the normalized
     first-moment $V$-space anchor ket $|\bar{v}_t\rangle = \operatorname{normalize}_{\mathbb{C}}(\sum_{s \le t} w_{ts} |v_s\rangle)$
     instead of $|q_t\rangle$, eliminating random cross-subspace $Q \leftrightarrow V$ phase/norm scrambling at Step 0
     and executing two associative Krylov steps $\frac{\rho_t(\rho_t |\bar{v}_t\rangle)}{\|\rho_{\text{re}, t}\|_F^2 + \|\rho_{\text{im}, t}\|_F^2 + \varepsilon}$
     in $O(d_h^2)$ time (**Refinement 4**).

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_052.md` / `Wave 37`):
- Translates `GeminiBornComplexPackedWithRoPE.py` into a production building block with Refinements 1, 3, and 4
  (and zero $\gamma_{\text{coh}}$) compatible with both standalone complex `(psi_re, psi_im)` register states
  and our symplectic real-wavefunction `LevelSeparatedWaveletConvGPT` backbone (`Option 2` pure-neural setup).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..functional import l2_normalize

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_052.md (Section 6.1)
# ==============================================================================
DEFAULT_COMPLEX_ROPE_BASE: float = 10000.0
DEFAULT_COMPLEX_REGISTER_DIM: int = 32
DEFAULT_P_BOSON_POWER: int = 2
DEFAULT_COMPLEX_NORM_EPS: float = 1e-7
INIT_V0_ANCHOR_MIX_LOGIT: float = -1.1

ComplexProbeMode = Literal["v_krylov", "q_probe"]


def complex_l2_normalize(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    dim: int = -1,
    eps: float = DEFAULT_COMPLEX_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a complex tensor pair `(x_re, x_im)` to unit complex L2 norm along `dim`."""
    norm_sq = x_re.float().square().sum(dim=dim, keepdim=True) + x_im.float().square().sum(dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(norm_sq + eps).to(dtype=x_re.dtype)
    return x_re * inv_norm, x_im * inv_norm


class ComplexRoPE(nn.Module):
    """Native Complex Rotary Position Embeddings for complex registers `z = z_re + i * z_im`.

    Each complex coordinate `a in {0, ..., d_h - 1}` has frequency `omega_a = base^(-a / d_h)`
    and rotates in the complex plane by `e^(i * t * omega_a) = cos(t * omega_a) + i * sin(t * omega_a)`.
    Applied strictly to `Q` and `K` (**Refinement 1**) so `<q_t^rot | k_s^rot>_C` depends only on `t - s`.
    """

    def __init__(
        self,
        d_register: int = DEFAULT_COMPLEX_REGISTER_DIM,
        base: float = DEFAULT_COMPLEX_ROPE_BASE,
    ) -> None:
        super().__init__()
        self.d_register = int(d_register)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d_register, dtype=torch.float32) / float(self.d_register))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, z_re: torch.Tensor, z_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate `(z_re, z_im)` of shape `(B, H, T, d_h)` by `e^(i * t * omega_a)`."""
        seq_t = z_re.size(2)
        positions = torch.arange(seq_t, device=z_re.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        cos_theta = torch.cos(angles).unsqueeze(0).unsqueeze(0).to(dtype=z_re.dtype)
        sin_theta = torch.sin(angles).unsqueeze(0).unsqueeze(0).to(dtype=z_re.dtype)
        rot_re = z_re * cos_theta - z_im * sin_theta
        rot_im = z_re * sin_theta + z_im * cos_theta
        return rot_re, rot_im


class ComplexLinear(nn.Module):
    """Complex-linear transformation `W = W_re + i * W_im` initialized on the complex unitary group `U(d)` via QR."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_re = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_im = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize `W_re + i * W_im` as a random Haar-distributed complex unitary/semi-unitary matrix."""
        rows = max(self.out_features, self.in_features)
        cols = min(self.out_features, self.in_features)
        z = torch.complex(
            torch.randn(rows, cols, dtype=torch.float32),
            torch.randn(rows, cols, dtype=torch.float32),
        )
        q, r = torch.linalg.qr(z)
        diag_r = torch.diagonal(r, 0)
        phase = torch.sgn(diag_r)
        phase = torch.where(phase.abs() > 0, phase, torch.ones_like(phase))
        q = q * phase.unsqueeze(0)
        if self.out_features < self.in_features:
            q = q.t().contiguous()
        with torch.no_grad():
            self.weight_re.copy_(q.real)
            self.weight_im.copy_(q.imag)

    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply `(W_re + i * W_im)(x_re + i * x_im)`."""
        w_re = self.weight_re.to(dtype=x_re.dtype)
        w_im = self.weight_im.to(dtype=x_re.dtype)
        out_re = F.linear(x_re, w_re) - F.linear(x_im, w_im)
        out_im = F.linear(x_re, w_im) + F.linear(x_im, w_re)
        return out_re, out_im


def pack_hermitian_outer_product(v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
    """Pack rank-1 Hermitian density matrices `|v_s><v_s| = rho_re + i * rho_im` into `(B, H, T, d_h * d_h)`.

    Uses a single `torch.where` selection (`Refinement 3`):
    - Upper triangle + diagonal (`i <= j`): stores symmetric `rho_re[i, j] = v_re[i]*v_re[j] + v_im[i]*v_im[j]`.
    - Strict lower triangle (`i > j`): stores skew-symmetric `rho_im[i, j] = v_im[i]*v_re[j] - v_re[i]*v_im[j]`.
    """
    b_sz, h_num, seq_t, dh = v_re.shape
    rho_re = v_re.unsqueeze(-1) * v_re.unsqueeze(-2) + v_im.unsqueeze(-1) * v_im.unsqueeze(-2)
    rho_im = v_im.unsqueeze(-1) * v_re.unsqueeze(-2) - v_re.unsqueeze(-1) * v_im.unsqueeze(-2)
    idx = torch.arange(dh, device=v_re.device)
    upper_mask = (idx[:, None] <= idx[None, :]).view(1, 1, 1, dh, dh)
    packed = torch.where(upper_mask, rho_re, rho_im)
    return packed.reshape(b_sz, h_num, seq_t, dh * dh)


def pack_hermitian_outer_product_and_ket(v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
    """Pack `|v_s><v_s|` (`d_h^2`) and `[v_re, v_im]` (`2 * d_h`) into `(B, H, T, d_h^2 + 2 * d_h)` (`Refinements 3 & 4`)."""
    packed_flat = pack_hermitian_outer_product(v_re, v_im)
    return torch.cat([packed_flat, v_re, v_im], dim=-1)


def unpack_hermitian_matrix(packed_flat: torch.Tensor, dh: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack `(B, H, T, d_h * d_h)` back into symmetric `rho_re` and skew-symmetric `rho_im` (`Refinement 3`).

    Reconstructs `rho_re` without ever doubling the diagonal (`i == j`) and sets the diagonal of
    skew-symmetric `rho_im` to exact zero via `torch.where`.
    """
    b_sz, h_num, seq_t, _ = packed_flat.shape
    p_mat = packed_flat.view(b_sz, h_num, seq_t, dh, dh)
    p_t = p_mat.transpose(-2, -1)
    idx = torch.arange(dh, device=packed_flat.device)
    upper_mask = (idx[:, None] <= idx[None, :]).view(1, 1, 1, dh, dh)
    strict_lower_mask = (idx[:, None] > idx[None, :]).view(1, 1, 1, dh, dh)
    strict_upper_mask = (idx[:, None] < idx[None, :]).view(1, 1, 1, dh, dh)

    rho_re = torch.where(upper_mask, p_mat, p_t)
    zero = torch.zeros(1, device=packed_flat.device, dtype=packed_flat.dtype)
    rho_im = torch.where(strict_lower_mask, p_mat, torch.where(strict_upper_mask, -p_t, zero))
    return rho_re, rho_im


def explicit_mcweeny_density_probe(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    probe_re: torch.Tensor,
    probe_im: torch.Tensor,
    eps: float = DEFAULT_COMPLEX_NORM_EPS,
    normalize_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference explicit matrix-squared McWeeny purification `((rho @ rho) / Tr(rho^2)) |probe>` (`O(d_h^3)`)."""
    r_re = rho_re.float()
    r_im = rho_im.float()
    p_re = probe_re.float().unsqueeze(-1)
    p_im = probe_im.float().unsqueeze(-1)

    sq_re = torch.matmul(r_re, r_re) - torch.matmul(r_im, r_im)
    sq_im = torch.matmul(r_re, r_im) + torch.matmul(r_im, r_re)
    tr_sq = sq_re.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    pur_re = sq_re / (tr_sq + eps)
    pur_im = sq_im / (tr_sq + eps)

    u_re = (torch.matmul(pur_re, p_re) - torch.matmul(pur_im, p_im)).squeeze(-1).to(dtype=probe_re.dtype)
    u_im = (torch.matmul(pur_re, p_im) + torch.matmul(pur_im, p_re)).squeeze(-1).to(dtype=probe_im.dtype)
    if normalize_output:
        return complex_l2_normalize(u_re, u_im, dim=-1, eps=eps)
    return u_re, u_im


def krylov_purified_density_probe(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    probe_re: torch.Tensor,
    probe_im: torch.Tensor,
    eps: float = DEFAULT_COMPLEX_NORM_EPS,
    normalize_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associative two-step Krylov evaluation of `rho_t^purified |probe_t> = rho_t (rho_t |probe_t>) / Tr(rho_t^2)`.

    Mathematical Identity (`Refinement 4`):
    1. Because `rho_t = rho_re + i * rho_im` is Hermitian (`rho_re^T = rho_re`, `rho_im^T = -rho_im`),
       `Tr(rho_t^2) = Tr(rho_t^dagger rho_t) = ||rho_re||_F^2 + ||rho_im||_F^2`.
    2. By associativity, `(rho_t^2) |probe_t> == rho_t (rho_t |probe_t>)`, replacing four `(d_h, d_h) @ (d_h, d_h)`
       matrix-matrix multiplications (`O(d_h^3)`) with two `(d_h, d_h) @ (d_h, 1)` matrix-vector steps (`O(d_h^2)`).
    """
    r_re = rho_re.float()
    r_im = rho_im.float()
    p_re = probe_re.float().unsqueeze(-1)
    p_im = probe_im.float().unsqueeze(-1)

    # Krylov Step 1: |w_t> = rho_t |probe_t>
    w_re = torch.matmul(r_re, p_re) - torch.matmul(r_im, p_im)
    w_im = torch.matmul(r_re, p_im) + torch.matmul(r_im, p_re)

    # Krylov Step 2: |z_t> = rho_t |w_t> = rho_t^2 |probe_t>
    z_re = (torch.matmul(r_re, w_re) - torch.matmul(r_im, w_im)).squeeze(-1)
    z_im = (torch.matmul(r_re, w_im) + torch.matmul(r_im, w_re)).squeeze(-1)

    # Exact Hermitian trace identity: Tr(rho_t^2) = ||rho_re||_F^2 + ||rho_im||_F^2
    tr_sq = (r_re.square().sum(dim=(-2, -1)) + r_im.square().sum(dim=(-2, -1))).unsqueeze(-1)
    u_re = (z_re / (tr_sq + eps)).to(dtype=probe_re.dtype)
    u_im = (z_im / (tr_sq + eps)).to(dtype=probe_im.dtype)
    if normalize_output:
        return complex_l2_normalize(u_re, u_im, dim=-1, eps=eps)
    return u_re, u_im


def _mix_and_krylov_probe_step(
    w_born: torch.Tensor,
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    q_rot_re: torch.Tensor,
    q_rot_im: torch.Tensor,
    dh: int,
    use_vkrylov: bool = True,
    eps: float = DEFAULT_COMPLEX_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-level checkpointable 1-GEMM Hermitian pack + mix + unpack + Krylov probe (`<= 8` args)."""
    if use_vkrylov:
        v_pk = pack_hermitian_outer_product_and_ket(v_re, v_im)
        m_all = torch.matmul(w_born.to(dtype=v_pk.dtype), v_pk)
        d_sq = dh * dh
        r_pk = m_all[..., :d_sq]
        vb_r = m_all[..., d_sq : d_sq + dh]
        vb_i = m_all[..., d_sq + dh :]
        pr_r, pr_i = complex_l2_normalize(vb_r, vb_i, dim=-1, eps=eps)
    else:
        v_fl = pack_hermitian_outer_product(v_re, v_im)
        r_pk = torch.matmul(w_born.to(dtype=v_fl.dtype), v_fl)
        pr_r, pr_i = q_rot_re, q_rot_im
    r_re, r_im = unpack_hermitian_matrix(r_pk, dh=dh)
    ur_re, ur_im = krylov_purified_density_probe(
        r_re, r_im, pr_r, pr_i, eps=eps, normalize_output=False
    )
    tr_s = r_re.float().square().sum(dim=(-2, -1)) + r_im.float().square().sum(dim=(-2, -1))
    u_nm = torch.sqrt(ur_re.float().square().sum(dim=-1) + ur_im.float().square().sum(dim=-1) + eps)
    return ur_re, ur_im, tr_s.mean(), u_nm.mean()


class PackedComplexRegisterDensityAttention(nn.Module):
    """Krylov-Probed Packed Complex Register Density Attention (`Refinements 1, 3, & 4`, Zero `gamma_coh`).

    Supports both:
    - Complex register pairs `(psi_re, psi_im)` of shape `(B, T, d_complex)` where `d_complex = n_registers * d_register`,
    - Symplectic real wavefunctions `psi` of shape `(B, T, 2 * d_complex)` (`psi_re = psi[..., 0::2]`, `psi_im = psi[..., 1::2]`).
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_registers: int = 32,
        d_register: int = DEFAULT_COMPLEX_REGISTER_DIM,
        p_boson: int = DEFAULT_P_BOSON_POWER,
        probe_mode: ComplexProbeMode = "v_krylov",
        normalize_krylov_ket: bool = True,
        value_residual_mixing: bool = True,
    ) -> None:
        super().__init__()
        self.d_complex = int(d_model)
        self.embed_dim = 2 * self.d_complex
        self.H = int(n_registers)
        self.dh = int(d_register)
        if self.H * self.dh != self.d_complex:
            raise ValueError(
                f"n_registers ({self.H}) * d_register ({self.dh}) must equal d_model ({self.d_complex})"
            )
        self.p_boson = int(p_boson)
        self.probe_mode: ComplexProbeMode = probe_mode
        self.normalize_krylov_ket = bool(normalize_krylov_ket)
        self.value_residual_mixing = bool(value_residual_mixing)

        self.rope = ComplexRoPE(d_register=self.dh)
        self.W_q = ComplexLinear(self.d_complex, self.d_complex)
        self.W_k = ComplexLinear(self.d_complex, self.d_complex)
        self.W_v = ComplexLinear(self.d_complex, self.d_complex)
        self.W_out = ComplexLinear(self.d_complex, self.d_complex)
        if self.value_residual_mixing:
            self.v_mix_logit: nn.Parameter | None = nn.Parameter(torch.tensor(INIT_V0_ANCHOR_MIX_LOGIT))
        else:
            self.register_parameter("v_mix_logit", None)

        self.last_purity_mean: float = 1.0
        self.last_probe_norm_mean: float = 1.0

    def _compute_born_weights(
        self,
        q_rot_re: torch.Tensor,
        q_rot_im: torch.Tensor,
        k_rot_re: torch.Tensor,
        k_rot_im: torch.Tensor,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
        eps: float = DEFAULT_COMPLEX_NORM_EPS,
    ) -> torch.Tensor:
        """Compute causal softmax-free p-boson Born weights `w_ts = |<q_t^rot | k_s^rot>_C|^(2p) / Z_t`."""
        seq_t = q_rot_re.size(2)
        c_re = torch.matmul(q_rot_re, k_rot_re.transpose(-2, -1)) + torch.matmul(
            q_rot_im, k_rot_im.transpose(-2, -1)
        )
        c_im = torch.matmul(q_rot_re, k_rot_im.transpose(-2, -1)) - torch.matmul(
            q_rot_im, k_rot_re.transpose(-2, -1)
        )
        mag_sq = c_re.float().square() + c_im.float().square()

        idx_t = torch.arange(seq_t, device=q_rot_re.device)
        diff = idx_t[:, None] - idx_t[None, :]
        mask_bool = diff >= 0
        if window_size > 0:
            mask_bool = mask_bool & (diff < int(window_size))
        mask_4d = mask_bool.view(1, 1, seq_t, seq_t)
        if doc_ids is not None:
            mask_4d = mask_4d & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])

        intensity = mag_sq.pow(self.p_boson).masked_fill(~mask_4d, 0.0)
        return intensity / (intensity.sum(dim=-1, keepdim=True) + eps)

    def forward_complex(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
        v0_re: torch.Tensor | None = None,
        v0_im: torch.Tensor | None = None,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
        eps: float = DEFAULT_COMPLEX_NORM_EPS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute complex register density attention returning `(out_re, out_im, v_raw_re, v_raw_im)`."""
        b_sz, seq_t, d_c = psi_re.shape
        h_num, dh = self.H, self.dh

        q_re, q_im = self.W_q(psi_re, psi_im)
        k_re, k_im = self.W_k(psi_re, psi_im)
        v_re, v_im = self.W_v(psi_re, psi_im)

        q_re = q_re.view(b_sz, seq_t, h_num, dh).transpose(1, 2).contiguous()
        q_im = q_im.view(b_sz, seq_t, h_num, dh).transpose(1, 2).contiguous()
        k_re = k_re.view(b_sz, seq_t, h_num, dh).transpose(1, 2).contiguous()
        k_im = k_im.view(b_sz, seq_t, h_num, dh).transpose(1, 2).contiguous()
        v_re = v_re.view(b_sz, seq_t, h_num, dh).transpose(1, 2).contiguous()
        v_im = v_im.view(b_sz, seq_t, h_num, dh).transpose(1, 2).contiguous()

        v_raw_re, v_raw_im = v_re, v_im
        if self.value_residual_mixing and self.v_mix_logit is not None and v0_re is not None and v0_im is not None:
            alpha = torch.sigmoid(self.v_mix_logit).to(dtype=v_re.dtype)
            v_re = (1.0 - alpha) * v_re + alpha * v0_re.to(dtype=v_re.dtype)
            v_im = (1.0 - alpha) * v_im + alpha * v0_im.to(dtype=v_im.dtype)

        q_re, q_im = complex_l2_normalize(q_re, q_im, dim=-1, eps=eps)
        k_re, k_im = complex_l2_normalize(k_re, k_im, dim=-1, eps=eps)
        v_re, v_im = complex_l2_normalize(v_re, v_im, dim=-1, eps=eps)

        # Refinement 1: Apply ComplexRoPE strictly to Q and K (V stays un-rotated in content space)
        q_rot_re, q_rot_im = self.rope(q_re, q_im)
        k_rot_re, k_rot_im = self.rope(k_re, k_im)

        # Softmax-free p-Boson Born weights w_ts >= 0, sum_{s<=t} w_ts = 1 (zero gamma_coh)
        w_born = self._compute_born_weights(
            q_rot_re, q_rot_im, k_rot_re, k_rot_im, window_size=window_size, doc_ids=doc_ids, eps=eps
        )
        use_vkrylov = self.probe_mode == "v_krylov"

        if self.training and torch.is_grad_enabled() and not torch.compiler.is_compiling():
            u_raw_re, u_raw_im, tr_mean, unorm_mean = checkpoint(
                _mix_and_krylov_probe_step,
                w_born,
                v_re,
                v_im,
                q_rot_re,
                q_rot_im,
                dh,
                use_vkrylov,
                eps,
                use_reentrant=False,
            )
        else:
            u_raw_re, u_raw_im, tr_mean, unorm_mean = _mix_and_krylov_probe_step(
                w_born, v_re, v_im, q_rot_re, q_rot_im, dh, use_vkrylov, eps
            )

        if not torch.compiler.is_compiling():
            self.last_purity_mean = float(tr_mean.detach().item())
            self.last_probe_norm_mean = float(unorm_mean.detach().item())

        if self.normalize_krylov_ket:
            u_re, u_im = complex_l2_normalize(u_raw_re, u_raw_im, dim=-1, eps=eps)
        else:
            u_re, u_im = u_raw_re, u_raw_im

        out_re, out_im = self.W_out(
            u_re.transpose(1, 2).contiguous().view(b_sz, seq_t, d_c),
            u_im.transpose(1, 2).contiguous().view(b_sz, seq_t, d_c),
        )
        return out_re, out_im, v_raw_re, v_raw_im

    def forward(
        self,
        psi: torch.Tensor,
        psi_im: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
        return_v: bool = False,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run attention on either a complex pair `(psi_re, psi_im)` or a symplectic real wavefunction `psi`."""
        del cos, sin  # Native ComplexRoPE handles Q/K rotation internally
        if psi_im is not None:
            out_re, out_im, _, _ = self.forward_complex(
                psi, psi_im, window_size=window_size, doc_ids=doc_ids
            )
            return complex_l2_normalize(psi + out_re, psi_im + out_im, dim=-1)

        psi_n = l2_normalize(psi, dim=-1)
        p_re = psi_n[..., 0::2].contiguous()
        p_im = psi_n[..., 1::2].contiguous()
        v0_re: torch.Tensor | None = None
        v0_im: torch.Tensor | None = None
        if v_0 is not None:
            v0_re = v_0[..., 0::2].contiguous()
            v0_im = v_0[..., 1::2].contiguous()

        out_re, out_im, v_raw_re, v_raw_im = self.forward_complex(
            p_re, p_im, v0_re=v0_re, v0_im=v0_im, window_size=window_size, doc_ids=doc_ids
        )
        out_symplectic = torch.stack((out_re, out_im), dim=-1).reshape_as(psi)
        out_symplectic = l2_normalize(out_symplectic, dim=-1)
        if return_v:
            v_curr = torch.stack((v_raw_re, v_raw_im), dim=-1).reshape(
                psi.size(0), self.H, psi.size(1), 2 * self.dh
            )
            return out_symplectic, v_curr
        return out_symplectic


__all__ = [
    "ComplexLinear",
    "ComplexProbeMode",
    "ComplexRoPE",
    "DEFAULT_COMPLEX_NORM_EPS",
    "DEFAULT_COMPLEX_REGISTER_DIM",
    "DEFAULT_COMPLEX_ROPE_BASE",
    "DEFAULT_P_BOSON_POWER",
    "INIT_V0_ANCHOR_MIX_LOGIT",
    "PackedComplexRegisterDensityAttention",
    "complex_l2_normalize",
    "explicit_mcweeny_density_probe",
    "krylov_purified_density_probe",
    "pack_hermitian_outer_product",
    "pack_hermitian_outer_product_and_ket",
    "unpack_hermitian_matrix",
]
