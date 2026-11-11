r"""Wave 77 Quantum State-Routing, Calibrated Associative Copy, & Causal MTP MPS Primitives (`EXPERIMENT_PLAN_097.md`).

Provenance & Mathematical Rationale
-----------------------------------
Built for `Wave 77` (`EXPERIMENT_PLAN_097.md`) following Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`
to resolve the failure modes isolated in `Wave 76` (`EXPERIMENT_PLAN_096.md`) and test Carbon's
two quantum state-routing upgrades without $O(d^2)$ matrix materialization or causal target leakage:

1. **`<BOS>` Anchor Sink & Document Masking (`build_bos_anchor_sink_doc_masks`, `apply_bos_anchor_sink_row_norm`)**:
   - Ensures zero cross-document (`<|endoftext|>`) attention leakage while routing intra-document
     positions `pos_in_doc < k_win` (whose exterior `s <= t - k_win` keys all lie in the prior document)
     to the active document's `<BOS>` anchor `doc_start_pos[b, t]` so every attention row sums to `1.0`.

2. **Vacuum-Normalized Confidence-Gated $p=3$ Associative Copy (`CalibratedPBosonAssociativeCopyGate`)**:
   - Fixes `Wave 76`'s `MQAR K=4` calibration gap (`99.90%` Top-1 yet `6.43` CE under fixed `1.65` scale)
     by normalizing $p=3$ Born overlaps $|\langle q_t \mid k_{s-1}\rangle|^6$ against a learnable no-copy
     vacuum floor $\\varepsilon_{\\text{vac}} = \\operatorname{softplus}(\\nu_{\\text{vac}})$ and gating
     log-odds across `[0, 12.5]` logits. Single confident matches add `~10+` logits (`MQAR CE < 0.25`),
     repeated function words divide conditionally as $1/m$, and non-matches vanish (`< 0.01` logits).

3. **Zero-`(d, d)` Subsystem-Blocked ($d_h = 16$) Complementary Schur Gate (`ComplementarySchurSubsystemGate`)**:
   - Implements Carbon's trace-preserving two-outcome `AND` / `Disagreement` quantum instrument across
     $G = d // 16$ subsystem heads of $\\mathbb{C}^{16}$ ($2/d_h = 12.5\\%$ contrast) using the exact
     $O(R d_h)$ probe identity:
     $$\\tilde{\\rho}_{A\\setminus B}|\\psi\\rangle = (1 - b)\\odot(\\rho_A|\\psi\\rangle) - \\rho_A(b\\odot|\\psi\\rangle) + (a\\odot b)\\odot|\\psi\\rangle$$
     where $a = \\operatorname{diag}(\\rho_A), b = \\operatorname{diag}(\\rho_B)$ and $p_{\\text{and}} = \\langle a, b \\rangle$,
     with smooth $\\varepsilon$-safe fallback to $\\frac{1}{16}I_{16}$ as $p_{\\text{and}} \\to 0$ or $1$.

4. **Strictly Causal $K=4$ Virtual MTP Backward-then-Forward MPS Sweep (`CausalVirtualMTPBackwardForwardMPSSweep`)**:
   - Predicts $K = 4$ virtual future site states solely from causal state $h_t$ ($x_{\\le t}$), runs a
     $\\chi = 8$ 1D MPS backward ($K \\to 1$) to form $\\rho_{\\text{bwd}}[t, k]$, and sweeps forward
     ($1 \\to K$), intersecting $\\rho_{\\text{fwd}} \\odot \\rho_{\\text{bwd}}$ before advancing $k \\to k+1$.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .fused_bf16_four_compartment_baton_lab import DEFAULT_TRACE_EPS, normalize_complex_ket

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_097.md Section 3)
# ==============================================================================
FINEWEB_BOS_TOKEN_ID: int = 50256
DEFAULT_COPY_BOSON_POWER: float = 3.0
DEFAULT_COPY_VACUUM_INIT_LOGIT: float = -1.80
DEFAULT_COPY_MAX_LOGIT_GAIN: float = 12.5
DEFAULT_COPY_GATE_INIT_BIAS: float = 1.40

COMPLEX_HEAD_DIM: int = 16
COMP_SCHUR_BEAM_RANK: int = 2
COMP_SCHUR_TRACE_EPS: float = 1.0e-4

CAUSAL_MTP_HORIZON_K: int = 4
CAUSAL_MTP_MPS_BOND_CHI: int = 8
CAUSAL_MTP_AUX_LOSS_WEIGHT: float = 0.05


@dataclass(frozen=True)
class BosDocumentMaskPack:
    """Precomputed `<BOS>` intra-document mask and anchor-sink indices (`<= 8` fields, Rule 2)."""

    doc_ids: Tensor
    doc_same: Tensor
    doc_start_pos: Tensor
    pos_in_doc: Tensor


def build_bos_anchor_sink_doc_masks(
    idx: Tensor,
    bos_token_id: int = FINEWEB_BOS_TOKEN_ID,
) -> BosDocumentMaskPack:
    """Build `<BOS>` document equality mask `(B, T, T)` and active `<BOS>` anchor `doc_start_pos` `(B, T)`."""
    bsz, seq_len = idx.shape
    is_bos = idx == int(bos_token_id)
    doc_ids = is_bos.to(torch.long).cumsum(dim=1)
    doc_same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)

    t_idx = torch.arange(seq_len, device=idx.device, dtype=torch.long).unsqueeze(0).expand(bsz, seq_len)
    bos_positions = torch.where(is_bos, t_idx, torch.zeros_like(t_idx))
    last_bos_pos = torch.cummax(bos_positions, dim=1).values
    doc_start_pos = torch.where(doc_ids > 0, last_bos_pos, torch.zeros_like(t_idx))
    pos_in_doc = t_idx - doc_start_pos
    return BosDocumentMaskPack(
        doc_ids=doc_ids,
        doc_same=doc_same,
        doc_start_pos=doc_start_pos,
        pos_in_doc=pos_in_doc,
    )


def apply_bos_anchor_sink_row_norm(
    raw_scores: Tensor,
    valid_mask: Tensor,
    doc_start_pos: Tensor,
    eps: float = DEFAULT_TRACE_EPS,
) -> Tensor:
    """Normalize non-negative attention scores while routing empty rows to `doc_start_pos` (`<= 8` args).

    Supports both 3D `(B, T, T)` and 4D `(B, H, T, T)` score tensors and guarantees every row
    sums to `1.0` with `0.0%` cross-`<BOS>` leakage.
    """
    masked = (raw_scores * valid_mask.to(raw_scores.dtype)).clamp_min(0.0)
    row_sum = masked.sum(dim=-1, keepdim=True)
    has_mass = row_sum > float(eps)

    sink_one_hot = torch.zeros_like(masked)
    if masked.ndim == 4:
        anchor_idx = doc_start_pos.unsqueeze(1).unsqueeze(-1).expand(
            masked.size(0), masked.size(1), masked.size(2), 1
        )
    else:
        anchor_idx = doc_start_pos.unsqueeze(-1)
    sink_one_hot.scatter_(dim=-1, index=anchor_idx, value=1.0)

    safe_probs = masked / row_sum.clamp_min(float(eps))
    return torch.where(has_mass, safe_probs, sink_one_hot)


# ==============================================================================
# Upgrade 1 (Rung 1+): Vacuum-Normalized Confidence-Gated p-Boson Associative Copy
# ==============================================================================
class CalibratedPBosonAssociativeCopyGate(nn.Module):
    """Hardware-wired `<BOS>`-masked $p$-boson associative copy gate with calibrated `[0..12.5]` log-odds."""

    def __init__(
        self,
        copy_ket_dim: int,
        ctx_dim: int,
        boson_power: float = DEFAULT_COPY_BOSON_POWER,
        max_logit_gain: float = DEFAULT_COPY_MAX_LOGIT_GAIN,
    ) -> None:
        super().__init__()
        self.copy_ket_dim = int(copy_ket_dim)
        self.ctx_dim = int(ctx_dim)
        self.boson_power = float(boson_power)
        self.max_logit_gain = float(max_logit_gain)

        real_copy_dim = 2 * self.copy_ket_dim
        self.q_copy_proj = nn.Linear(real_copy_dim, real_copy_dim, bias=False)
        self.k_copy_proj = nn.Linear(real_copy_dim, real_copy_dim, bias=False)
        nn.init.eye_(self.q_copy_proj.weight)
        nn.init.eye_(self.k_copy_proj.weight)

        self.ctx_gate_proj = nn.Linear(self.ctx_dim, 1, bias=True)
        nn.init.zeros_(self.ctx_gate_proj.weight)
        nn.init.constant_(self.ctx_gate_proj.bias, DEFAULT_COPY_GATE_INIT_BIAS)

        self.vacuum_logit = nn.Parameter(
            torch.tensor(DEFAULT_COPY_VACUUM_INIT_LOGIT, dtype=torch.float32)
        )

    def forward(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        ctx_feat: Tensor,
        doc_same: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return `(copy_log_odds, copy_conf)` of shapes `(B, T, T)` and `(B, T)` (`<= 8` args).

        `copy_log_odds[b, t, s]` is scattered onto token `idx[b, s]` for strictly past in-document
        positions `1 <= s < t` where `ket[t]` matches `ket[s - 1]`.
        """
        bsz, seq_len, d_c = ket_re.shape
        packed = torch.cat([ket_re, ket_im], dim=-1)
        q_raw = self.q_copy_proj(packed)
        k_raw = self.k_copy_proj(packed)

        q_re, q_im = normalize_complex_ket(q_raw[..., :d_c], q_raw[..., d_c:])
        k_re, k_im = normalize_complex_ket(k_raw[..., :d_c], k_raw[..., d_c:])

        prev_k_re = F.pad(k_re[:, :-1, :], (0, 0, 1, 0))
        prev_k_im = F.pad(k_im[:, :-1, :], (0, 0, 1, 0))

        ov_re = torch.matmul(q_re, prev_k_re.transpose(-2, -1)) + torch.matmul(
            q_im, prev_k_im.transpose(-2, -1)
        )
        ov_im = torch.matmul(q_im, prev_k_re.transpose(-2, -1)) - torch.matmul(
            q_re, prev_k_im.transpose(-2, -1)
        )
        ov_sq = (ov_re.float().square() + ov_im.float().square()).clamp(min=0.0, max=1.0)
        sharp_ov = ov_sq.pow(self.boson_power) if abs(self.boson_power - 1.0) > 1e-5 else ov_sq

        t_idx = torch.arange(seq_len, device=ket_re.device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=ket_re.device).unsqueeze(0)
        strict_past = ((t_idx > s_idx) & (s_idx >= 1)).unsqueeze(0)

        doc_prev = F.pad(doc_same[:, :, :-1], (1, 0), value=False)
        valid_pair = (strict_past & doc_same & doc_prev).to(torch.float32)

        masked_scores = sharp_ov * valid_pair
        eps_vac = F.softplus(self.vacuum_logit.float()).clamp_min(1.0e-4)
        denom = masked_scores.sum(dim=-1, keepdim=True) + eps_vac
        w_copy = masked_scores / denom
        copy_conf = w_copy.sum(dim=-1)

        gain = self.max_logit_gain * torch.sigmoid(self.ctx_gate_proj(ctx_feat).float())
        copy_log_odds = (gain * w_copy).to(ctx_feat.dtype)
        return copy_log_odds, copy_conf.to(ctx_feat.dtype)


# ==============================================================================
# Carbon Upgrade 2 (Rung 5A/5B): Complementary Schur Gate (Zero-(d,d) + Reference)
# ==============================================================================
def explicit_complementary_schur_matrix_pair(
    rho_a: Tensor,
    rho_b: Tensor,
    eps: float = COMP_SCHUR_TRACE_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Explicit $d_h \\times d_h$ complex reference for Carbon's Complementary Schur Gate (`<= 8` args).

    Used in unit tests to verify `< 1e-5` exact agreement with our $O(R d_h)$ zero-matrix probe action,
    exact trace conservation $p_{\\text{and}} + (1 - p_{\\text{and}}) = 1$, and PSD eigenvalues.
    """
    d_h = rho_a.size(-1)
    schur_ab = rho_a * rho_b
    diag_a = torch.diagonal(rho_a, dim1=-2, dim2=-1).real
    diag_b = torch.diagonal(rho_b, dim1=-2, dim2=-1).real
    p_and = (diag_a * diag_b).sum(dim=-1).clamp(0.0, 1.0)

    eye = torch.eye(d_h, device=rho_a.device, dtype=rho_a.dtype)
    uni = eye / float(d_h)

    w_and = (p_and.square() / (p_and.square() + eps * eps)).unsqueeze(-1).unsqueeze(-1)
    rho_and = w_and * (schur_ab / p_and.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)) + (1.0 - w_and) * uni

    off_diag_mask = (1.0 - torch.eye(d_h, device=rho_a.device, dtype=diag_a.dtype))
    factor_ab = 1.0 - diag_b.unsqueeze(-1) - off_diag_mask * diag_b.unsqueeze(-2)
    factor_ba = 1.0 - diag_a.unsqueeze(-1) - off_diag_mask * diag_a.unsqueeze(-2)
    tilde_ab = rho_a * factor_ab.to(rho_a.dtype)
    tilde_ba = rho_b * factor_ba.to(rho_b.dtype)

    one_minus_p = (1.0 - p_and).clamp(0.0, 1.0)
    w_comp = (one_minus_p.square() / (one_minus_p.square() + eps * eps)).unsqueeze(-1).unsqueeze(-1)
    raw_comp = 0.5 * (tilde_ab + tilde_ba) / one_minus_p.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    rho_comp = w_comp * raw_comp + (1.0 - w_comp) * uni
    return rho_and, rho_comp, p_and


def _apply_rank_r_density_to_ket(
    u_re: Tensor,
    u_im: Tensor,
    psi_re: Tensor,
    psi_im: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute $\\rho|\\psi\\rangle = \\sum_{r=1}^R |u_r\\rangle\\langle u_r \\mid \\psi\\rangle$ in $O(R d_h)$ (`<= 8` args)."""
    dot_re = (u_re * psi_re.unsqueeze(-2) + u_im * psi_im.unsqueeze(-2)).sum(dim=-1, keepdim=True)
    dot_im = (u_re * psi_im.unsqueeze(-2) - u_im * psi_re.unsqueeze(-2)).sum(dim=-1, keepdim=True)
    out_re = (u_re * dot_re - u_im * dot_im).sum(dim=-2)
    out_im = (u_re * dot_im + u_im * dot_re).sum(dim=-2)
    return out_re, out_im


def zero_matrix_complementary_schur_probe_action(
    u_a_re: Tensor,
    u_a_im: Tensor,
    u_b_re: Tensor,
    u_b_im: Tensor,
    psi_re: Tensor,
    psi_im: Tensor,
    eps: float = COMP_SCHUR_TRACE_EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Exact $O(R d_h)$ zero-`(d_h, d_h)` probe action for Carbon's Complementary Schur Gate (`<= 8` args).

    Args:
        u_a_re, u_a_im: `(..., R, d_h)` trace-normalized rank-$R$ packet factors for $\\rho_A$.
        u_b_re, u_b_im: `(..., R, d_h)` trace-normalized rank-$R$ packet factors for $\\rho_B$.
        psi_re, psi_im: `(..., d_h)` complex probe ket $|\\psi\\rangle$.
        eps: Safe trace threshold $\\varepsilon$ for smooth `I / d_h` fallback.

    Returns:
        `(and_re, and_im, comp_re, comp_im, p_and)` where `(and_re, and_im) = \\rho_{\\text{AND}}|\\psi\\rangle`
        and `(comp_re, comp_im) = \\rho_{\\text{comp}}|\\psi\\rangle`.
    """
    d_h = psi_re.size(-1)
    diag_a = (u_a_re.square() + u_a_im.square()).sum(dim=-2)
    diag_b = (u_b_re.square() + u_b_im.square()).sum(dim=-2)
    p_and = (diag_a * diag_b).sum(dim=-1).clamp(0.0, 1.0)

    # 1. Consensus rho_AND |psi> via rank-R^2 Hadamard beams (no d_h x d_h matrix!)
    pair_re = (
        u_a_re.unsqueeze(-2) * u_b_re.unsqueeze(-3)
        - u_a_im.unsqueeze(-2) * u_b_im.unsqueeze(-3)
    ).flatten(-3, -2)
    pair_im = (
        u_a_re.unsqueeze(-2) * u_b_im.unsqueeze(-3)
        + u_a_im.unsqueeze(-2) * u_b_re.unsqueeze(-3)
    ).flatten(-3, -2)
    schur_psi_re, schur_psi_im = _apply_rank_r_density_to_ket(pair_re, pair_im, psi_re, psi_im)

    inv_d = 1.0 / float(d_h)
    w_and = (p_and.square() / (p_and.square() + eps * eps)).unsqueeze(-1)
    p_safe = p_and.clamp_min(eps).unsqueeze(-1)
    and_re = w_and * (schur_psi_re / p_safe) + (1.0 - w_and) * (inv_d * psi_re)
    and_im = w_and * (schur_psi_im / p_safe) + (1.0 - w_and) * (inv_d * psi_im)

    # 2. Disagreement tilde_rho_{A\B}|psi> = (1 - b) * (rho_A|psi>) - rho_A(b * |psi>) + (a*b) * |psi>
    rho_a_psi_re, rho_a_psi_im = _apply_rank_r_density_to_ket(u_a_re, u_a_im, psi_re, psi_im)
    rho_a_bpsi_re, rho_a_bpsi_im = _apply_rank_r_density_to_ket(
        u_a_re, u_a_im, diag_b * psi_re, diag_b * psi_im
    )
    ab_diag = diag_a * diag_b
    tilde_ab_re = (1.0 - diag_b) * rho_a_psi_re - rho_a_bpsi_re + ab_diag * psi_re
    tilde_ab_im = (1.0 - diag_b) * rho_a_psi_im - rho_a_bpsi_im + ab_diag * psi_im

    rho_b_psi_re, rho_b_psi_im = _apply_rank_r_density_to_ket(u_b_re, u_b_im, psi_re, psi_im)
    rho_b_apsi_re, rho_b_apsi_im = _apply_rank_r_density_to_ket(
        u_b_re, u_b_im, diag_a * psi_re, diag_a * psi_im
    )
    tilde_ba_re = (1.0 - diag_a) * rho_b_psi_re - rho_b_apsi_re + ab_diag * psi_re
    tilde_ba_im = (1.0 - diag_a) * rho_b_psi_im - rho_b_apsi_im + ab_diag * psi_im

    one_minus_p = (1.0 - p_and).clamp(0.0, 1.0)
    w_comp = (one_minus_p.square() / (one_minus_p.square() + eps * eps)).unsqueeze(-1)
    comp_denom = (2.0 * one_minus_p.clamp_min(eps)).unsqueeze(-1)
    comp_re = w_comp * ((tilde_ab_re + tilde_ba_re) / comp_denom) + (1.0 - w_comp) * (inv_d * psi_re)
    comp_im = w_comp * ((tilde_ab_im + tilde_ba_im) / comp_denom) + (1.0 - w_comp) * (inv_d * psi_im)
    return and_re, and_im, comp_re, comp_im, p_and


class ComplementarySchurSubsystemGate(nn.Module):
    """Subsystem-blocked ($d_h = 16$) Schur router supporting both post-selected and Complementary split."""

    def __init__(
        self,
        real_dim: int,
        use_complementary_split: bool = True,
        beam_rank: int = COMP_SCHUR_BEAM_RANK,
    ) -> None:
        super().__init__()
        if real_dim % (2 * COMPLEX_HEAD_DIM) != 0:
            raise ValueError(f"real_dim ({real_dim}) must be divisible by {2 * COMPLEX_HEAD_DIM}")
        self.real_dim = int(real_dim)
        self.num_groups = self.real_dim // (2 * COMPLEX_HEAD_DIM)
        self.use_complementary_split = bool(use_complementary_split)
        self.beam_rank = int(beam_rank)

        self.beam_a_proj = nn.Linear(self.real_dim, self.beam_rank * self.real_dim, bias=False)
        self.beam_b_proj = nn.Linear(self.real_dim, self.beam_rank * self.real_dim, bias=False)
        self.probe_proj = nn.Linear(self.real_dim, self.real_dim, bias=False)
        self.even_odd_out_proj = nn.Linear(2 * self.real_dim, self.real_dim, bias=False)

        std = 0.5 / math.sqrt(float(self.real_dim))
        nn.init.normal_(self.beam_a_proj.weight, std=std)
        nn.init.normal_(self.beam_b_proj.weight, std=std)
        nn.init.eye_(self.probe_proj.weight)
        nn.init.zeros_(self.even_odd_out_proj.weight)

        self.last_mean_p_and: float = 0.0
        self.last_even_purity: float = 0.0
        self.last_odd_purity: float = 0.0

    def _pack_trace_unit_beams(self, raw: Tensor, bsz: int, seq_len: int) -> tuple[Tensor, Tensor]:
        """Reshape `(B, T, R * 2d)` into trace-1 `(B, T, G, R, 16)` `(u_re, u_im)` (`<= 8` args)."""
        g = self.num_groups
        r = self.beam_rank
        d_h = COMPLEX_HEAD_DIM
        shaped = raw.view(bsz, seq_len, g, r, 2 * d_h).float()
        u_re = shaped[..., :d_h]
        u_im = shaped[..., d_h:]
        tr = (u_re.square() + u_im.square()).sum(dim=(-2, -1), keepdim=True).clamp_min(DEFAULT_TRACE_EPS)
        inv_s = torch.rsqrt(tr)
        return u_re * inv_s, u_im * inv_s

    @torch.no_grad()
    def _record_detached_purity_telemetry(
        self,
        u_a_re: Tensor,
        u_a_im: Tensor,
        u_b_re: Tensor,
        u_b_im: Tensor,
        p_and: Tensor,
    ) -> None:
        """Record exact detached `mean_p_and` and `Tr(rho^2)` on a lightweight sample slice (`<= 8` args)."""
        self.last_mean_p_and = float(p_and.detach().mean().item())
        ua = torch.complex(u_a_re[:2, :16].detach(), u_a_im[:2, :16].detach())
        ub = torch.complex(u_b_re[:2, :16].detach(), u_b_im[:2, :16].detach())
        rho_a = torch.matmul(ua.transpose(-2, -1), ua.conj())
        rho_b = torch.matmul(ub.transpose(-2, -1), ub.conj())
        rho_and, rho_comp, _ = explicit_complementary_schur_matrix_pair(rho_a, rho_b)
        pur_even = (rho_and.real.square() + rho_and.imag.square()).sum(dim=(-2, -1)).mean()
        pur_odd = (rho_comp.real.square() + rho_comp.imag.square()).sum(dim=(-2, -1)).mean()
        self.last_even_purity = float(pur_even.item())
        self.last_odd_purity = float(pur_odd.item())

    def forward(self, state_a: Tensor, state_b: Tensor) -> Tensor:
        """Intersect `state_a` and `state_b` `(B, T, D)` via post-selected or Complementary Schur (`<= 8` args)."""
        bsz, seq_len, _ = state_a.shape
        g = self.num_groups
        d_h = COMPLEX_HEAD_DIM

        u_a_re, u_a_im = self._pack_trace_unit_beams(self.beam_a_proj(state_a), bsz, seq_len)
        u_b_re, u_b_im = self._pack_trace_unit_beams(self.beam_b_proj(state_b), bsz, seq_len)

        pr_raw = self.probe_proj(0.5 * (state_a + state_b)).view(bsz, seq_len, g, 2 * d_h).float()
        psi_re, psi_im = normalize_complex_ket(pr_raw[..., :d_h], pr_raw[..., d_h:])

        and_re, and_im, comp_re, comp_im, p_and = zero_matrix_complementary_schur_probe_action(
            u_a_re, u_a_im, u_b_re, u_b_im, psi_re, psi_im
        )
        self._record_detached_purity_telemetry(u_a_re, u_a_im, u_b_re, u_b_im, p_and)

        p_w = p_and.unsqueeze(-1)
        even_pkt = (p_w * torch.cat([and_re, and_im], dim=-1)).reshape(bsz, seq_len, self.real_dim)
        if self.use_complementary_split:
            odd_pkt = ((1.0 - p_w) * torch.cat([comp_re, comp_im], dim=-1)).reshape(
                bsz, seq_len, self.real_dim
            )
        else:
            odd_pkt = torch.zeros_like(even_pkt)

        joined = torch.cat([even_pkt, odd_pkt], dim=-1).to(state_a.dtype)
        return self.even_odd_out_proj(joined)


# ==============================================================================
# Carbon Upgrade 1 (Rung 5B): Strictly Causal K=4 Virtual MTP Bwd-Fwd MPS Sweep
# ==============================================================================
class CausalVirtualMTPBackwardForwardMPSSweep(nn.Module):
    """Strictly causal $K=4$ virtual rollout smoothed by a $\\chi=8$ backward-then-forward 1D MPS sweep."""

    def __init__(
        self,
        real_dim: int,
        horizon_k: int = CAUSAL_MTP_HORIZON_K,
        bond_chi: int = CAUSAL_MTP_MPS_BOND_CHI,
    ) -> None:
        super().__init__()
        if real_dim % (2 * COMPLEX_HEAD_DIM) != 0:
            raise ValueError(f"real_dim ({real_dim}) must be divisible by {2 * COMPLEX_HEAD_DIM}")
        self.real_dim = int(real_dim)
        self.num_groups = self.real_dim // (2 * COMPLEX_HEAD_DIM)
        self.horizon_k = int(horizon_k)
        self.bond_chi = int(bond_chi)

        site_real = 2 * COMPLEX_HEAD_DIM
        self.virtual_rollout_proj = nn.Linear(self.real_dim, self.horizon_k * self.real_dim, bias=False)
        self.fwd_bond_init = nn.Linear(site_real, self.bond_chi, bias=False)
        self.bwd_site_to_bond = nn.Parameter(
            torch.randn(self.horizon_k, site_real, self.bond_chi) / math.sqrt(float(site_real))
        )
        self.bwd_bond_transfer = nn.Parameter(
            torch.eye(self.bond_chi).unsqueeze(0).repeat(self.horizon_k, 1, 1)
        )
        self.bwd_bond_to_diag = nn.Parameter(
            torch.randn(self.horizon_k, self.bond_chi, COMPLEX_HEAD_DIM) / math.sqrt(float(self.bond_chi))
        )
        self.fwd_bond_to_site = nn.Parameter(
            torch.randn(self.horizon_k, self.bond_chi, site_real) / math.sqrt(float(self.bond_chi))
        )
        self.fwd_site_to_bond = nn.Parameter(
            torch.randn(self.horizon_k, site_real, self.bond_chi) / math.sqrt(float(site_real))
        )
        self.mtp_out_proj = nn.Linear(self.real_dim, self.real_dim, bias=False)

        std = 0.5 / math.sqrt(float(self.real_dim))
        nn.init.normal_(self.virtual_rollout_proj.weight, std=std)
        nn.init.normal_(self.fwd_bond_init.weight, std=1.0 / math.sqrt(float(site_real)))
        nn.init.zeros_(self.mtp_out_proj.weight)
        self.last_mean_mps_p_and: float = 0.0

    def forward(
        self,
        x_causal: Tensor,
        future_target_embeds: Tensor | None = None,
        doc_same: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run backward ($K \\to 1$) then Schur-intersected forward ($1 \\to K$) causal virtual MPS sweep."""
        bsz, seq_len, _ = x_causal.shape
        g = self.num_groups
        d_h = COMPLEX_HEAD_DIM
        site_real = 2 * d_h

        virt_raw = self.virtual_rollout_proj(x_causal).view(
            bsz, seq_len, self.horizon_k, g, site_real
        )
        virt_f = F.normalize(virt_raw.float(), dim=-1, eps=1.0e-6)

        # 1. Backward sweep k = K-1 .. 0 over virtual future sites (100% causal in x_{<=t}!)
        r_bwd = torch.full(
            (bsz, seq_len, g, self.bond_chi),
            1.0 / math.sqrt(float(self.bond_chi)),
            device=x_causal.device,
            dtype=torch.float32,
        )
        rho_bwd_diags: list[Tensor] = [torch.empty(0, device=x_causal.device)] * self.horizon_k
        for k in range(self.horizon_k - 1, -1, -1):
            site_drive = torch.matmul(virt_f[:, :, k], self.bwd_site_to_bond[k].float())
            bond_prop = torch.matmul(r_bwd, self.bwd_bond_transfer[k].float())
            r_bwd = F.normalize(bond_prop + site_drive, dim=-1, eps=1.0e-6)
            diag_raw = torch.matmul(r_bwd, self.bwd_bond_to_diag[k].float()).square() + (
                1.0 / float(d_h)
            ) * 0.10
            rho_bwd_diags[k] = diag_raw / diag_raw.sum(dim=-1, keepdim=True).clamp_min(DEFAULT_TRACE_EPS)

        # 2. Forward sweep k = 0 .. K-1 intersecting rho_fwd * rho_bwd[k] BEFORE bond advance!
        x_heads = x_causal.view(bsz, seq_len, g, site_real).float()
        l_fwd = F.normalize(self.fwd_bond_init(x_heads.to(x_causal.dtype)).float(), dim=-1, eps=1.0e-6)
        smoothed_k0: Tensor | None = None
        p_mps_accum = 0.0

        for k in range(self.horizon_k):
            phi_raw = virt_f[:, :, k] + torch.matmul(l_fwd, self.fwd_bond_to_site[k].float())
            phi_re, phi_im = normalize_complex_ket(phi_raw[..., :d_h], phi_raw[..., d_h:])
            rho_fwd_diag = phi_re.square() + phi_im.square()
            bwd_diag_k = rho_bwd_diags[k]

            p_mps = (rho_fwd_diag * bwd_diag_k).sum(dim=-1, keepdim=True).clamp_min(COMP_SCHUR_TRACE_EPS)
            p_mps_accum += float(p_mps.detach().mean().item())
            bwd_amp = torch.sqrt(bwd_diag_k.clamp_min(DEFAULT_TRACE_EPS))
            inv_norm = torch.rsqrt(p_mps)
            smooth_re = phi_re * bwd_amp * inv_norm
            smooth_im = phi_im * bwd_amp * inv_norm
            smooth_site = torch.cat([smooth_re, smooth_im], dim=-1)
            if k == 0:
                smoothed_k0 = smooth_site
            l_fwd = F.normalize(
                l_fwd + torch.matmul(smooth_site, self.fwd_site_to_bond[k].float()),
                dim=-1,
                eps=1.0e-6,
            )

        self.last_mean_mps_p_and = p_mps_accum / float(max(1, self.horizon_k))
        assert smoothed_k0 is not None
        delta_x = self.mtp_out_proj(smoothed_k0.reshape(bsz, seq_len, self.real_dim).to(x_causal.dtype))

        # 3. Optional training-only causal MTP cosine alignment against stop-grad future targets
        aux_loss: Tensor | None = None
        if future_target_embeds is not None:
            tgt_norm = F.normalize(future_target_embeds.detach().float(), dim=-1, eps=1.0e-6)
            pred_flat = F.normalize(
                virt_raw.float().reshape(bsz, seq_len, self.horizon_k, self.real_dim),
                dim=-1,
                eps=1.0e-6,
            )
            step_losses: list[Tensor] = []
            for k in range(self.horizon_k):
                if seq_len <= k:
                    break
                p_slice = pred_flat[:, : seq_len - k, k, :]
                t_slice = tgt_norm[:, k:, :]
                cos_err = 1.0 - (p_slice * t_slice).sum(dim=-1)
                if doc_same is not None and k > 0:
                    t_ar = torch.arange(seq_len - k, device=x_causal.device)
                    same_k = doc_same[:, t_ar, t_ar + k].to(torch.float32)
                    denom = same_k.sum().clamp_min(1.0)
                    step_losses.append((cos_err * same_k).sum() / denom)
                else:
                    step_losses.append(cos_err.mean())
            if step_losses:
                aux_loss = CAUSAL_MTP_AUX_LOSS_WEIGHT * (torch.stack(step_losses).mean())
        return delta_x, aux_loss
