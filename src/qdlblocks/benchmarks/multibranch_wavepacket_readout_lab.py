"""Multi-Branch Wavepacket Readout Laboratory (`(50304, 8, 2, 32)` Stiefel Identity Frame & Coupled Mixture).

Investigates overcoming the Johnson-Lindenstrauss single-vector superposition ceiling on the `64D` Identity
wavepacket sphere:
    K_max(d_id = 64, V = 50304) = d_id / (2 * ln V) = 64 / 21.652 ≈ 2.95 tokens

by replacing the single-point readout with a rank-R Quantum Mixed-State / Mixture-of-Wavepackets Readout
(R in {1, 4, 8}) where:
1. The R identity branches {h_{t,r,id} in R^64}_{r=0}^{R-1} are orthogonalized via Modified Gram-Schmidt /
   Fubini-Study projection into a strict orthogonal R-frame on the 64D identity sphere (multiplying the
   identity superposition capacity from K_max ≈ 2.95 to R * K_max ≈ 11.8 at R=4 and 23.6 at R=8).
2. Each branch r pairs its orthogonal 64D identity wavepacket with a branch-specific combination of two
   orthogonal 448D semantic wavepackets (Calibrated Trunk+NGram Anchor h_{sem,0} and Deep Contextual
   Residual h_{sem,1}), solving the Semantic-Identity binding problem without repeating 448D GEMMs per branch.
3. A single-pass custom autograd kernel (`_FactoredMultiBranchWavepacketCEFn`, Rule 9 compliant) computes the
   locally-normalized log-probability mixture:
       log P_t(v) = LogSumExp_{r=0}^{R-1} ( log pi_{t,r} + log_softmax_v(z_{t,r}(v)) )
   and backpropagates via exact E-M posterior branch responsibilities gamma_{t,r} = P(branch r | x_t, y_t).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.impure_density_mirror_block import DEFAULT_TIKHONOV_GRAM_LAMBDA
from ..layers.quantum_statmech_channels import (
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
)
from ..layers.quantum_svd_stiefel_graft import BiorthogonalSVDStiefelProjection
from ..models.compiled_leaderboard_classical_gpt import _CompiledChunkedSoftcapLinearCEFn
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon
from .shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_CARRIER_BIAS_DIM,
    DEFAULT_CE_CHUNK_TOKENS,
    DEFAULT_SHALLOW_BATCH_SIZE,
    DEFAULT_SHALLOW_EMBED_DIM,
    DEFAULT_SHALLOW_MLP_HIDDEN,
    DEFAULT_SHALLOW_MUON_RETRACT_EVERY,
    DEFAULT_SHALLOW_NGRAM_LR,
    DEFAULT_SHALLOW_NUM_HEADS,
    DEFAULT_SHALLOW_NUM_LAYERS,
    DEFAULT_SHALLOW_PKT_COUNT,
    DEFAULT_SHALLOW_PKT_DIM,
    DEFAULT_SHALLOW_READOUT_LR,
    DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SHALLOW_SEQ_LEN,
    DEFAULT_SHALLOW_STIEFEL_LR,
    DEFAULT_SHALLOW_UNITARY_LR,
    DEFAULT_SHALLOW_VOCAB_SIZE,
    DEFAULT_SHALLOW_WTE_LR,
    DEFAULT_SOFTCAP_LOGIT,
    LongSeqFineWebStream,
    ShallowArchMode,
    ShallowLongSeqSpec,
    ShallowLongSeqWavepacketGPT,
    _get_or_build_val_kn_log_priors,
    initialize_and_calibrate_shallow_model_,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_MULTIBRANCH_TOKEN_CHUNK: int = 32
DEFAULT_ANCHOR_BRANCH_LOGIT_BIAS: float = 1.65  # Initializes pi_0 ≈ 0.50..0.60 with residual-calibrated explorers
DEFAULT_EXPLORER_ID_TEMP_SCALE: float = 1.05
DEFAULT_CONE_TANGENT_SIN: float = 0.50  # Geodesic Stiefel cone angle sin(theta)=0.50, cos(theta)=0.8660 on S^{63}
DEFAULT_LOAD_BALANCE_WEIGHT: float = 0.0
DEFAULT_GRAM_SCHMIDT_EPS: float = 1e-6

DEFAULT_PROBE_VOCAB_SAMPLES: int = 128
DEFAULT_PROBE_OPT_STEPS: int = 120
DEFAULT_PROBE_OPT_LR: float = 0.12


class MultiBranchMode(str, Enum):
    """Strategy enum selecting the Multi-Branch Wavepacket Readout variant (Rule 3)."""

    M0_SINGLE_BRANCH_REF = "m0_single_branch_ref"
    M1_FIBER_STIEFEL_FRAME_R4 = "m1_fiber_stiefel_frame_r4"
    M2_COUPLED_SEM_ID_WAVEPACKETS_R4 = "m2_coupled_sem_id_wavepackets_r4"
    M3_COUPLED_SEM_ID_WAVEPACKETS_R8 = "m3_coupled_sem_id_wavepackets_r8"
    M4_COUPLED_R8_CALIBRATED_SYNERGY = "m4_coupled_r8_calibrated_synergy"


@dataclass(frozen=True)
class MultiBranchReadoutSpec:
    """Configuration for Multi-Branch Wavepacket Readout experiments (Rule 2)."""

    mode: MultiBranchMode = MultiBranchMode.M3_COUPLED_SEM_ID_WAVEPACKETS_R8
    vocab_size: int = DEFAULT_SHALLOW_VOCAB_SIZE
    embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM
    num_heads: int = DEFAULT_SHALLOW_NUM_HEADS
    num_layers: int = DEFAULT_SHALLOW_NUM_LAYERS
    mlp_hidden: int = DEFAULT_SHALLOW_MLP_HIDDEN
    seq_len: int = DEFAULT_SHALLOW_SEQ_LEN
    batch_size: int = DEFAULT_SHALLOW_BATCH_SIZE
    num_init_shards: int = 9
    seed: int = 1337

    @property
    def num_branches(self) -> int:
        """Return the number of wavepacket readout branches R."""
        if self.mode == MultiBranchMode.M0_SINGLE_BRANCH_REF:
            return 1
        if self.mode in (
            MultiBranchMode.M1_FIBER_STIEFEL_FRAME_R4,
            MultiBranchMode.M2_COUPLED_SEM_ID_WAVEPACKETS_R4,
        ):
            return 4
        return 8

    @property
    def use_coupled_semantic(self) -> bool:
        """Return True if branches couple with the second orthogonal 448D semantic wavepacket basis."""
        return self.mode in (
            MultiBranchMode.M2_COUPLED_SEM_ID_WAVEPACKETS_R4,
            MultiBranchMode.M3_COUPLED_SEM_ID_WAVEPACKETS_R8,
            MultiBranchMode.M4_COUPLED_R8_CALIBRATED_SYNERGY,
        )


def orthogonalize_identity_stiefel_frame(
    id_branches: torch.Tensor,
    target_rms_norm: float,
) -> torch.Tensor:
    r"""Project `id_branches` `(..., R, 64)` onto a strict orthogonal R-frame on the 64D identity sphere.

    Uses differentiable Modified Gram-Schmidt along the branch axis `R` so that:
    - Branch `0` (`id_branches[..., 0, :]`) is preserved in its exact calibrated direction.
    - Every subsequent branch `r in {1, ..., R-1}` has its parallel components along `{0, ..., r-1}`
      removed before normalizing onto `\text{target\_rms\_norm} \cdot \mathbb{S}^{63}`.
    """
    num_branches = id_branches.size(-2)
    d_id = id_branches.size(-1)
    ortho_list: list[torch.Tensor] = []
    scale = float(target_rms_norm) / math.sqrt(float(d_id))

    for r_idx in range(num_branches):
        v_r = id_branches[..., r_idx, :].float()
        for prev_u in ortho_list:
            # prev_u has unit L2 norm (||prev_u||_2 == 1.0)
            proj_coeff = (v_r * prev_u).sum(dim=-1, keepdim=True)
            v_r = v_r - proj_coeff * prev_u
        u_r = F.normalize(v_r, p=2.0, dim=-1, eps=DEFAULT_GRAM_SCHMIDT_EPS)
        ortho_list.append(u_r)

    stacked = torch.stack(ortho_list, dim=-2) * float(target_rms_norm)
    return (stacked * scale / float(target_rms_norm) * float(target_rms_norm)).to(id_branches.dtype)


class _FactoredMultiBranchWavepacketCEFn(torch.autograd.Function):
    r"""Rule-9 Single-Pass Chunked Autograd Function for Factored Multi-Branch Wavepacket Mixture CE.

    Computes the locally-normalized rank-R Mixture-of-Wavepackets cross-entropy:
        z_{i,r}(v) = C_cap * tanh( (z_base_i(v) + beta_{i,r} * z_sem1_i(v) + tau_{i,r} * z_id_{i,r}(v)) / C_cap )
        log P_{i,r}(v) = log_softmax_v(z_{i,r}(v))
        log P_i(v) = LogSumExp_{r=0}^{R-1} ( log pi_{i,r} + log P_{i,r}(v) )

    In the single-pass token-chunked backward loop, computes exact E-M posterior branch responsibilities:
        gamma_{i,r} = softmax_r( log pi_{i,r} + log P_{i,r}(y_i) )
    and accumulates all input gradients (`dh_base`, `dh_sem1`, `dh_id`, `dbeta`, `dtau`, `dalpha`, `dw_aug`)
    in a single pass over the chunks (`bwd_flops / fwd_flops <= 2.2 <= 3.5`).
    """

    @staticmethod
    def forward(
        ctx: Any,
        h_base_464: torch.Tensor,
        h_sem1_448: torch.Tensor,
        h_id_branches: torch.Tensor,
        branch_controls: torch.Tensor,
        w_aug_528: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chunked forward pass and save compact tensors for single-pass backward."""
        bsz, seq_len, _ = h_base_464.shape
        n_total = bsz * seq_len
        num_branches = h_id_branches.size(-2)
        v_sz = w_aug_528.size(0)
        sem_dim = h_sem1_448.size(-1)  # 448
        id_dim = h_id_branches.size(-1)  # 64
        car_dim = h_base_464.size(-1) - sem_dim  # 16

        hb_flat = h_base_464.reshape(n_total, sem_dim + car_dim)
        hs1_flat = h_sem1_448.reshape(n_total, sem_dim)
        hid_flat = h_id_branches.reshape(n_total, num_branches, id_dim)
        ctrl_flat = branch_controls.reshape(n_total, num_branches, 3).float()
        beta_flat = ctrl_flat[:, :, 0]
        tau_flat = ctrl_flat[:, :, 1]
        alpha_flat = ctrl_flat[:, :, 2]
        y_flat = targets.reshape(n_total)

        w_sem = w_aug_528[:, :sem_dim]
        w_id = w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous()
        w_sem_t = w_sem.t().contiguous()
        w_id_t = w_id.t().contiguous()

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        chunk_sz = DEFAULT_MULTIBRANCH_TOKEN_CHUNK

        total_nll = torch.zeros((), device=h_base_464.device, dtype=torch.float32)
        valid_mask = y_flat >= 0
        n_valid = int(valid_mask.sum().item())
        has_sem1 = bool(beta_flat.abs().max().item() > 1e-8)

        for start in range(0, n_total, chunk_sz):
            end = min(start + chunk_sz, n_total)
            c_len = end - start
            yc = y_flat[start:end]
            if not bool((yc >= 0).any().item()):
                continue

            z_base = torch.matmul(hb_flat[start:end], w_base_t).float()
            raw_r = torch.matmul(hid_flat[start:end].reshape(c_len * num_branches, id_dim), w_id_t).float().view(
                c_len, num_branches, v_sz
            )
            t_c = tau_flat[start:end].unsqueeze(-1)
            raw_r.mul_(t_c).add_(z_base.unsqueeze(1))
            if has_sem1:
                z_sem1 = torch.matmul(hs1_flat[start:end], w_sem_t).float()
                raw_r.addcmul_(beta_flat[start:end].unsqueeze(-1), z_sem1.unsqueeze(1))
            raw_r.mul_(inv_cap).tanh_().mul_(cap)

            log_pi = F.log_softmax(alpha_flat[start:end], dim=-1)
            yc_idx = yc.clamp_min(0).view(c_len, 1, 1).expand(c_len, num_branches, 1)
            target_log_p_r = raw_r.gather(dim=-1, index=yc_idx).squeeze(-1) - torch.logsumexp(raw_r, dim=-1)
            target_log_mix = torch.logsumexp(log_pi + target_log_p_r, dim=-1)
            total_nll = total_nll - target_log_mix[yc >= 0].sum()
            del z_base, raw_r

        ctx.save_for_backward(
            h_base_464,
            h_sem1_448,
            h_id_branches,
            branch_controls,
            w_aug_528,
            targets,
        )
        ctx.n_valid = max(1, n_valid)
        ctx.has_sem1 = has_sem1
        return total_nll / float(max(1, n_valid))

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
    ]:
        """Single-pass chunked backward computing exact E-M responsibilities and factored GEMM gradients."""
        (
            h_base_464,
            h_sem1_448,
            h_id_branches,
            branch_controls,
            w_aug_528,
            targets,
        ) = ctx.saved_tensors

        bsz, seq_len, _ = h_base_464.shape
        n_total = bsz * seq_len
        num_branches = h_id_branches.size(-2)
        v_sz = w_aug_528.size(0)
        sem_dim = h_sem1_448.size(-1)
        id_dim = h_id_branches.size(-1)
        car_dim = h_base_464.size(-1) - sem_dim
        work_dtype = h_base_464.dtype
        has_sem1 = bool(getattr(ctx, "has_sem1", True))

        hb_flat = h_base_464.reshape(n_total, sem_dim + car_dim)
        hs1_flat = h_sem1_448.reshape(n_total, sem_dim)
        hid_flat = h_id_branches.reshape(n_total, num_branches, id_dim)
        ctrl_flat = branch_controls.reshape(n_total, num_branches, 3).float()
        beta_flat = ctrl_flat[:, :, 0]
        tau_flat = ctrl_flat[:, :, 1]
        alpha_flat = ctrl_flat[:, :, 2]
        y_flat = targets.reshape(n_total)

        w_sem = w_aug_528[:, :sem_dim]
        w_id = w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base = torch.cat([w_sem, w_car], dim=-1).contiguous()
        w_base_t = w_base.t().contiguous()
        w_sem_t = w_sem.t().contiguous()
        w_id_t = w_id.t().contiguous()

        grad_hb = torch.zeros_like(hb_flat)
        grad_hs1 = torch.zeros_like(hs1_flat)
        grad_hid = torch.zeros_like(hid_flat)
        grad_ctrl = torch.zeros_like(ctrl_flat, dtype=branch_controls.dtype)

        grad_w_base = torch.zeros_like(w_base)
        grad_w_sem1 = torch.zeros_like(w_sem)
        grad_w_id = torch.zeros_like(w_id)

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        scale = float(grad_output.item()) / float(ctx.n_valid)
        chunk_sz = DEFAULT_MULTIBRANCH_TOKEN_CHUNK

        for start in range(0, n_total, chunk_sz):
            end = min(start + chunk_sz, n_total)
            c_len = end - start
            yc = y_flat[start:end]
            valid_c = yc >= 0
            if not bool(valid_c.any().item()):
                continue

            hb_c = hb_flat[start:end]
            hs1_c = hs1_flat[start:end]
            hid_c = hid_flat[start:end]
            b_c = beta_flat[start:end]
            t_c = tau_flat[start:end]
            a_c = alpha_flat[start:end]

            z_base = torch.matmul(hb_c, w_base_t).float()
            z_id = torch.matmul(hid_c.reshape(c_len * num_branches, id_dim), w_id_t).float().view(
                c_len, num_branches, v_sz
            )
            tanh_u = z_base.unsqueeze(1) + t_c.unsqueeze(-1) * z_id
            z_sem1: torch.Tensor | None = None
            if has_sem1:
                z_sem1 = torch.matmul(hs1_c, w_sem_t).float()
                tanh_u.addcmul_(b_c.unsqueeze(-1), z_sem1.unsqueeze(1))
            tanh_u.mul_(inv_cap).tanh_()
            log_p_r = F.log_softmax(cap * tanh_u, dim=-1)

            log_pi = F.log_softmax(a_c, dim=-1)
            pi_c = log_pi.exp()
            yc_idx = yc.clamp_min(0).view(c_len, 1, 1).expand(c_len, num_branches, 1)
            target_log_p_r = log_p_r.gather(dim=-1, index=yc_idx).squeeze(-1)

            gamma_c = F.softmax(log_pi + target_log_p_r, dim=-1)
            valid_f = valid_c.float().unsqueeze(-1)
            grad_ctrl[start:end, :, 2] = (scale * valid_f * (pi_c - gamma_c)).to(branch_controls.dtype)

            p_r = log_p_r.exp_()
            p_r.scatter_add_(
                dim=-1,
                index=yc_idx,
                src=torch.full((c_len, num_branches, 1), -1.0, device=p_r.device, dtype=p_r.dtype),
            )
            weight_r = (scale * valid_f * gamma_c).unsqueeze(-1)
            tanh_u.square_().neg_().add_(1.0)
            g_raw = p_r.mul_(weight_r).mul_(tanh_u)

            grad_ctrl[start:end, :, 1] = (g_raw * z_id).sum(dim=-1).to(branch_controls.dtype)
            g_base_2d = g_raw.sum(dim=1).to(work_dtype)
            g_id_2d = (g_raw * t_c.unsqueeze(-1)).to(work_dtype).reshape(c_len * num_branches, v_sz)

            grad_hb[start:end] = torch.matmul(g_base_2d, w_base)
            grad_hid[start:end] = torch.matmul(g_id_2d, w_id).view(c_len, num_branches, id_dim)
            grad_w_base.add_(torch.matmul(g_base_2d.t(), hb_c))
            grad_w_id.add_(torch.matmul(g_id_2d.t(), hid_c.reshape(c_len * num_branches, id_dim)))

            if has_sem1 and z_sem1 is not None:
                grad_ctrl[start:end, :, 0] = (g_raw * z_sem1.unsqueeze(1)).sum(dim=-1).to(branch_controls.dtype)
                g_sem1_2d = (g_raw * b_c.unsqueeze(-1)).sum(dim=1).to(work_dtype)
                grad_hs1[start:end] = torch.matmul(g_sem1_2d, w_sem)
                grad_w_sem1.add_(torch.matmul(g_sem1_2d.t(), hs1_c))
            del z_base, z_id, tanh_u, g_raw, g_base_2d, g_id_2d

        grad_w_sem_total = grad_w_base[:, :sem_dim] + grad_w_sem1
        grad_w_car_total = grad_w_base[:, sem_dim:]
        grad_w_aug = torch.cat([grad_w_sem_total, grad_w_id, grad_w_car_total], dim=-1)

        return (
            grad_hb.view_as(h_base_464),
            grad_hs1.view_as(h_sem1_448),
            grad_hid.view_as(h_id_branches),
            grad_ctrl.view_as(branch_controls),
            grad_w_aug,
            None,
        )


class MultiBranchWavepacketReadoutHead(nn.Module):
    r"""Multi-Branch Stiefel Identity Frame & Coupled Semantic Wavepacket Readout Head (`R in {4, 8}`).

    Constructs `R` distinct wavepacket hypotheses per token:
    - **Branch 0**: Exact calibrated `S3` `(Semantic + Identity)` wavepacket anchor (`h_sem_0`, `h_id_0`).
    - **Branch 1**: Pure Deep Contextual Trunk Identity (`h_id_tr`) + Stiefel rotation of final layer state.
    - **Branch 2**: Pure Local N-Gram Identity (`readout_id_64`) + Stiefel rotation of mid-layer state.
    - **Branches 3..R-1**: Multi-scale Stiefel Identity Explorer projections (`U_{id,r} in V_{64}(R^{512})`)
      initialized as mutually orthogonal Stiefel blocks.
    All `R` identity vectors are orthogonalized via Modified Gram-Schmidt (`orthogonalize_identity_stiefel_frame`)
    to form an exact `R`-frame on the `64D` identity sphere, and coupled with an orthogonal second `448D`
    semantic basis `h_sem_1` via per-branch gates `\beta_{t,r}`.
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM,
        sem_dim: int = 448,
        num_branches: int = 8,
        use_coupled_semantic: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.sem_dim = int(sem_dim)
        self.id_dim = self.embed_dim - self.sem_dim  # 64
        self.num_branches = int(num_branches)
        self.use_coupled_semantic = bool(use_coupled_semantic)

        # Orthogonal second 448D semantic wavepacket projector (operates on mid + final trunk state)
        self.sem1_bridge = BiorthogonalSVDStiefelProjection(self.embed_dim, self.sem_dim, self.sem_dim)

        # Stiefel identity frame generators: (R - 1, 64, 512) initialized as mutually orthogonal 64D slices
        q_init, _ = torch.linalg.qr(torch.randn(self.embed_dim, self.embed_dim, dtype=torch.float32))
        n_extra = max(1, self.num_branches - 1)
        slices: list[torch.Tensor] = []
        for r_idx in range(n_extra):
            c0 = (r_idx * self.id_dim) % (self.embed_dim - self.id_dim + 1)
            slices.append(q_init[:, c0 : c0 + self.id_dim].t().contiguous())
        self.id_branch_stiefel = nn.Parameter(torch.stack(slices, dim=0))  # (R-1, 64, 512)

        # Context-dependent router producing gate_logits (R), sem1_gates (R), and id_scales (R)
        self.router_in_dim = 128
        self.branch_gate_proj = nn.Linear(self.router_in_dim, self.num_branches, bias=True)
        self.sem1_gate_proj = nn.Linear(self.router_in_dim, self.num_branches, bias=True)
        self.id_scale_proj = nn.Linear(self.router_in_dim, self.num_branches, bias=True)

        nn.init.normal_(self.branch_gate_proj.weight, std=0.02)
        nn.init.zeros_(self.branch_gate_proj.bias)
        with torch.no_grad():
            self.branch_gate_proj.bias[0] = DEFAULT_ANCHOR_BRANCH_LOGIT_BIAS
            # Stagger initial semantic-1 coupling across branches (-0.35 .. +0.35) so branches diversify immediately
            if self.num_branches > 1:
                lin_b = torch.linspace(-0.35, 0.35, steps=self.num_branches - 1)
                self.sem1_gate_proj.bias[0] = 0.0
                self.sem1_gate_proj.bias[1:] = lin_b
        nn.init.normal_(self.sem1_gate_proj.weight, std=0.02)
        nn.init.zeros_(self.id_scale_proj.weight)
        nn.init.zeros_(self.id_scale_proj.bias)

    @torch.no_grad()
    def retract_stiefel_branches_(self) -> None:
        """Retract each `(64, 512)` identity branch generator onto the Stiefel manifold `U U^T = I_{64}`."""
        w = self.id_branch_stiefel.data.float()
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        self.id_branch_stiefel.data.copy_(torch.bmm(u, vh).to(self.id_branch_stiefel.dtype))
        self.sem1_bridge.clamp_sigma_()

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return 2D Stiefel matrices from `sem1_bridge`."""
        return self.sem1_bridge.real_stiefel_parameters()

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return router parameters and singular values."""
        params: list[nn.Parameter] = list(self.sem1_bridge.scalar_parameters())
        params.extend(
            [
                self.branch_gate_proj.weight,
                self.branch_gate_proj.bias,
                self.sem1_gate_proj.weight,
                self.sem1_gate_proj.bias,
                self.id_scale_proj.weight,
                self.id_scale_proj.bias,
            ]
        )
        return params


class MultiBranchWavepacketGPT(ShallowLongSeqWavepacketGPT):
    r"""Shallow Long-Sequence (`L = 4, d = 512, T = 512`) Quantum Wavepacket GPT with Multi-Branch Readout."""

    def __init__(self, spec: MultiBranchReadoutSpec) -> None:
        base_arch = (
            ShallowArchMode.S4_QUANTUM_WAVEPACKET_4L_CALIBRATED_SYNERGY
            if spec.mode == MultiBranchMode.M4_COUPLED_R8_CALIBRATED_SYNERGY
            else ShallowArchMode.S3_QUANTUM_WAVEPACKET_4L_SEMANTIC_NGRAM
        )
        base_spec = ShallowLongSeqSpec(
            arch_mode=base_arch,
            vocab_size=spec.vocab_size,
            embed_dim=spec.embed_dim,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            mlp_hidden=spec.mlp_hidden,
            seq_len=spec.seq_len,
            batch_size=spec.batch_size,
            num_init_shards=spec.num_init_shards,
            seed=spec.seed,
        )
        super().__init__(base_spec)
        self.mb_spec = spec
        self.mb_mode = MultiBranchMode(spec.mode)
        self.num_branches = spec.num_branches

        self.mb_head: MultiBranchWavepacketReadoutHead | None = None
        if self.num_branches > 1:
            self.mb_head = MultiBranchWavepacketReadoutHead(
                embed_dim=self.embed_dim,
                sem_dim=self.sem_real_dim,
                num_branches=self.num_branches,
                use_coupled_semantic=spec.use_coupled_semantic,
            )

    @torch.no_grad()
    def clamp_identity_dispersion_and_scalars_(self) -> None:
        """Clamp base physical scalars and retract multi-branch Stiefel generators."""
        super().clamp_identity_dispersion_and_scalars_()
        if self.mb_head is not None:
            self.mb_head.retract_stiefel_branches_()

    def _encode_multiscale_trunk(
        self,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode `idx` `(B, T)` into `(h_pre_final, h_mid_layer, readout_sem_448, readout_id_64)`."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32

        tok_wp = F.embedding(idx, self.wte_3d.view(self.vocab_size, self.embed_dim)).view(
            bsz, seq_len, self.pkt_count, self.pkt_dim
        )
        raw_wp = tok_wp.to(work_dtype)
        if seq_len > 1:
            raw_wp = self._apply_dispersion(raw_wp)

        raw_flat = raw_wp.reshape(bsz, seq_len, self.embed_dim)
        if seq_len > 1:
            s_gate = self.smear_lambda.to(work_dtype) * torch.sigmoid(
                F.linear(raw_flat[:, 1:, : self.smear_in_ch], self.smear_gate.weight.to(work_dtype))
            )
            sem_curr = raw_flat[:, 1:, : self.sem_real_dim] + s_gate * raw_flat[:, :-1, : self.sem_real_dim]
            sem_full = torch.cat([raw_flat[:, :1, : self.sem_real_dim], sem_curr], dim=1)
            raw_flat = torch.cat([sem_full, raw_flat[:, :, self.sem_real_dim :]], dim=-1)

        readout_sem_448: torch.Tensor | None = None
        readout_id_64: torch.Tensor | None = None
        if self.sem_ngram is not None:
            in_sem_delta, readout_sem_448, readout_id_64 = self.sem_ngram(idx, work_dtype)
            raw_flat = raw_flat + in_sem_delta

        x_0 = F.rms_norm(raw_flat, (self.embed_dim,), eps=1e-6)
        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)

        mid_idx = max(0, (len(self.quantum_blocks) // 2) - 1)
        h_mid = x_0
        for b_idx, q_blk in enumerate(self.quantum_blocks):
            x = q_blk(x, x_0, cos_4d, sin_4d)
            if b_idx == mid_idx:
                h_mid = F.rms_norm(x, (self.embed_dim,), eps=1e-6)

        h_pre = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        return h_pre, h_mid, readout_sem_448, readout_id_64

    def _apply_dispersion(self, raw_wp: torch.Tensor) -> torch.Tensor:
        """Apply causal wavepacket dispersion on Semantic Packets `0..6`."""
        from ..layers.wavepacket_statmech_block import apply_causal_wavepacket_dispersion

        return apply_causal_wavepacket_dispersion(raw_wp, self.wp_causal_re, self.wp_causal_im)

    def build_multibranch_readout_tensors(
        self,
        idx: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Construct factored multi-branch readout tensors `(h_base_464, h_sem1_448, h_id_R, branch_controls, w_aug)`."""
        h_pre, h_mid, readout_sem_448, readout_id_64 = self._encode_multiscale_trunk(idx)
        bsz, seq_len, _ = h_pre.shape
        work_dtype = h_pre.dtype
        h_bridge = self.exit_bridge(h_pre)

        h_sem_tr = F.rms_norm(h_bridge[:, :, : self.sem_real_dim], (self.sem_real_dim,), eps=1e-6)
        h_id_tr = F.rms_norm(h_bridge[:, :, self.sem_real_dim :], (self.id_real_dim,), eps=1e-6)
        if readout_sem_448 is not None and readout_id_64 is not None:
            g_sem, g_id = self.trunk_sem_gate.to(work_dtype), self.trunk_id_gate.to(work_dtype)
            h_sem_0 = F.rms_norm(g_sem * h_sem_tr + readout_sem_448, (self.sem_real_dim,), eps=1e-6)
            h_id_0 = F.rms_norm(g_id * h_id_tr + readout_id_64, (self.id_real_dim,), eps=1e-6)
        else:
            h_sem_0, h_id_0 = h_sem_tr, h_id_tr

        sem_scale = (self.temp_sem_scale / float(self.sem_real_dim)).to(work_dtype)
        id_scale_scalar = (self.temp_id_scale / float(self.id_real_dim)).to(work_dtype)

        h_sem_0_scaled = h_sem_0 * sem_scale
        c0_batch = self.carrier_c0.to(work_dtype).view(1, 1, DEFAULT_CARRIER_BIAS_DIM).expand(
            bsz, seq_len, DEFAULT_CARRIER_BIAS_DIM
        )
        h_base_464 = torch.cat([h_sem_0_scaled, c0_batch], dim=-1).contiguous()

        w_dir = (self.w_det_512 * self.radial_scale.unsqueeze(1)).to(work_dtype)
        b_col = (self.prior_gate * self.unigram_bias).to(work_dtype).unsqueeze(1) * self.carrier_c0.to(
            work_dtype
        ).unsqueeze(0)
        w_aug = torch.cat([w_dir, b_col], dim=-1).contiguous()

        if self.mb_head is None:
            h_sem_1_zero = torch.zeros_like(h_sem_0_scaled)
            h_id_single = (h_id_0 * id_scale_scalar).unsqueeze(-2)
            z_one = torch.zeros((bsz, seq_len, 1), device=idx.device, dtype=work_dtype)
            o_one = torch.ones((bsz, seq_len, 1), device=idx.device, dtype=work_dtype)
            ctrl_one = torch.stack([z_one, o_one, z_one], dim=-1)
            return h_base_464, h_sem_1_zero, h_id_single, ctrl_one, w_aug

        # Second orthogonal 448D semantic wavepacket basis (Fubini-Study orthogonalized against h_sem_0)
        if self.mb_head.use_coupled_semantic:
            sem_mix_in = F.rms_norm(0.70 * h_pre + 0.30 * h_mid, (self.embed_dim,), eps=1e-6)
            u_sem1 = self.mb_head.sem1_bridge(sem_mix_in)
            u_f = u_sem1.float()
            s0_f = h_sem_0.float()
            proj_c = (u_f * s0_f).sum(dim=-1, keepdim=True) / s0_f.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            h_sem_1 = F.rms_norm((u_f - proj_c * s0_f).to(work_dtype), (self.sem_real_dim,), eps=1e-6)
            h_sem_1_scaled = (h_sem_1 * sem_scale).contiguous()
        else:
            h_sem_1_scaled = torch.zeros_like(h_sem_0_scaled)

        # Build R candidate 64D identity branches before Gram-Schmidt Stiefel frame orthogonalization
        n_extra = self.num_branches - 1
        stiefel_w = self.mb_head.id_branch_stiefel.to(work_dtype)  # (R-1, 64, 512)
        src_state = F.rms_norm(0.75 * h_pre + 0.25 * h_mid, (self.embed_dim,), eps=1e-6)
        rot_id = torch.einsum("btd,rkd->btrk", src_state, stiefel_w)  # (B, T, R-1, 64)

        cand_list: list[torch.Tensor] = [h_id_0]
        for r_idx in range(n_extra):
            if r_idx == 0:
                # Branch 1: Pure Deep Contextual Trunk Identity + Stiefel rotation
                cand_list.append(F.rms_norm(h_id_tr + 0.50 * rot_id[:, :, 0, :], (self.id_real_dim,), eps=1e-6))
            elif r_idx == 1 and readout_id_64 is not None:
                # Branch 2: Pure Local N-Gram Identity + Stiefel rotation
                cand_list.append(F.rms_norm(readout_id_64 + 0.50 * rot_id[:, :, 1, :], (self.id_real_dim,), eps=1e-6))
            else:
                cand_list.append(F.rms_norm(rot_id[:, :, r_idx, :], (self.id_real_dim,), eps=1e-6))

        raw_id_stack = torch.stack(cand_list, dim=-2)  # (B, T, R, 64)
        ortho_id_frame = orthogonalize_identity_stiefel_frame(
            raw_id_stack,
            target_rms_norm=math.sqrt(float(self.id_real_dim)),
        )
        sin_th = float(DEFAULT_CONE_TANGENT_SIN)
        cos_th = math.sqrt(max(0.0, 1.0 - sin_th * sin_th))
        anchor_id_u0 = ortho_id_frame[:, :, :1, :]
        cone_explorers = cos_th * anchor_id_u0 + sin_th * ortho_id_frame[:, :, 1:, :]
        cone_id_frame = torch.cat([anchor_id_u0, cone_explorers], dim=-2)
        h_id_branches = (cone_id_frame * id_scale_scalar).contiguous()

        # Context-dependent branch gates and coupling scalars
        router_feat = torch.cat(
            [
                F.rms_norm(h_pre[:, :, :64], (64,), eps=1e-6),
                h_id_0,
            ],
            dim=-1,
        ).float()
        gate_logits = self.mb_head.branch_gate_proj(router_feat)
        if self.mb_head.use_coupled_semantic:
            sem1_raw = 0.35 * torch.tanh(self.mb_head.sem1_gate_proj(router_feat))
            # Keep Anchor Branch 0 strictly at beta_0 = 0.0 so Branch 0 preserves exact Step-0 calibration
            sem1_gates = torch.cat([torch.zeros_like(sem1_raw[:, :, :1]), sem1_raw[:, :, 1:]], dim=-1)
        else:
            sem1_gates = torch.zeros_like(gate_logits)

        id_scales_raw = 1.0 + 0.45 * torch.tanh(self.mb_head.id_scale_proj(router_feat))
        id_scales = torch.cat(
            [
                torch.ones_like(id_scales_raw[:, :, :1]),
                DEFAULT_EXPLORER_ID_TEMP_SCALE * id_scales_raw[:, :, 1:],
            ],
            dim=-1,
        )
        branch_controls = torch.stack([sem1_gates, id_scales, gate_logits], dim=-1).to(work_dtype)
        return (
            h_base_464,
            h_sem_1_scaled,
            h_id_branches,
            branch_controls,
            w_aug,
        )

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute single-branch or multi-branch wavepacket mixture CE + load-balancing regularizer."""
        if self.mb_head is None:
            return super().forward_loss(idx, targets)

        (
            h_base_464,
            h_sem1_448,
            h_id_branches,
            branch_controls,
            w_aug,
        ) = self.build_multibranch_readout_tensors(idx)
        nll_loss = _FactoredMultiBranchWavepacketCEFn.apply(
            h_base_464,
            h_sem1_448,
            h_id_branches,
            branch_controls,
            w_aug,
            targets,
        )
        if not self.training:
            return nll_loss

        # Gentle entropy floor on mean branch gate probabilities so explorer branches stay active
        gate_logits = branch_controls[:, :, :, 2].float()
        pi_mean = F.softmax(gate_logits, dim=-1).mean(dim=(0, 1))
        uniform_kl = (pi_mean * torch.log((pi_mean * float(self.num_branches)).clamp_min(1e-6))).sum()
        return nll_loss + DEFAULT_LOAD_BALANCE_WEIGHT * uniform_kl


def run_k_target_superposition_probe(
    w_det_512: torch.Tensor,
    k_values: tuple[int, ...] = (1, 2, 4, 8, 16),
    r_values: tuple[int, ...] = (1, 4, 8),
) -> dict[str, Any]:
    r"""Empirically verify the Johnson-Lindenstrauss `K_max ≈ 2.95` Identity Superposition Ceiling on `(50304, 8, 2, 32)`.

    Constructs contexts where the true next-token distribution is an equiprobable `K`-way ambiguity among
    semantically clustered tokens (tokens that share near-identical `448D` semantic wavepackets and MUST be
    resolved by the `64D` Identity wavepacket `Packet 7`), and measures the minimum achievable Cross-Entropy,
    excess CE over the Shannon entropy floor `\ln K`, and `Top-K` exact set recovery for `R in {1, 4, 8}`
    orthogonal identity branches against the full `50,304`-token vocabulary.
    """
    device = w_det_512.device
    v_sz, d_full = w_det_512.shape
    sem_dim = 448
    id_dim = d_full - sem_dim  # 64

    w_id = F.normalize(w_det_512[:4096, sem_dim:].detach().float(), p=2.0, dim=-1)  # (4096, 64) unit sphere
    n_pool = w_id.size(0)
    gen = torch.Generator(device=device)
    gen.manual_seed(2026)

    probe_results: dict[str, Any] = {
        "theoretical_jl_k_max_id64": round(float(id_dim) / (2.0 * math.log(float(v_sz))), 3),
        "theoretical_jl_k_max_sem448": round(float(sem_dim) / (2.0 * math.log(float(v_sz))), 3),
        "by_k": {},
    }

    n_trials = 32
    temp_id = 10.5

    for k_targets in k_values:
        shannon_floor = math.log(float(k_targets))
        target_idx = torch.randint(0, n_pool, (n_trials, k_targets), generator=gen, device=device)
        # Target vectors on S^{63}: (n_trials, K, 64)
        target_vecs = w_id[target_idx]

        k_entry: dict[str, Any] = {"shannon_floor_nats": round(shannon_floor, 4)}
        for num_r in r_values:
            # Partition the K target tokens across the R orthogonal branches (ceil(K / R) targets per branch)
            branch_list: list[torch.Tensor] = []
            for r_idx in range(num_r):
                start_k = (r_idx * k_targets) // num_r
                end_k = ((r_idx + 1) * k_targets) // num_r
                if end_k > start_k:
                    sub_sum = target_vecs[:, start_k:end_k, :].sum(dim=1)
                else:
                    sub_sum = target_vecs[:, 0, :]
                branch_list.append(F.normalize(sub_sum, p=2.0, dim=-1))

            # (n_trials, R, 64) -> logits against pool (n_trials, R, 4096)
            h_r = torch.stack(branch_list, dim=1)
            logits_r = temp_id * torch.matmul(h_r, w_id.t())  # (n_trials, R, n_pool)
            log_p_r = F.log_softmax(logits_r, dim=-1)
            log_mix = torch.logsumexp(log_p_r - math.log(float(num_r)), dim=1)  # (n_trials, n_pool)

            # Evaluate Cross-Entropy on the equiprobable K targets and Top-K exact set recall
            target_log_probs = log_mix.gather(dim=-1, index=target_idx)  # (n_trials, K)
            ce_val = float((-target_log_probs.mean()).item())
            excess_ce = max(0.0, ce_val - shannon_floor)

            topk_pred = torch.topk(log_mix, k=k_targets, dim=-1).indices  # (n_trials, K)
            hits = (topk_pred.unsqueeze(-1) == target_idx.unsqueeze(1)).any(dim=1).float().mean().item()

            k_entry[f"R{num_r}"] = {
                "ce_nats": round(ce_val, 4),
                "excess_ce_nats": round(excess_ce, 4),
                "topk_recall": round(float(hits), 4),
            }
        probe_results["by_k"][f"K{k_targets}"] = k_entry

    return probe_results


@torch.no_grad()
def evaluate_multibranch_model(
    model: MultiBranchWavepacketGPT,
    stream: LongSeqFineWebStream,
) -> dict[str, float]:
    """Evaluate `MultiBranchWavepacketGPT` on unseen Shard 0 (`65,536` tokens)."""
    if model.num_branches <= 1:
        from .shallow_longseq_wavepacket_lm_lab import evaluate_shallow_longseq_model

        return evaluate_shallow_longseq_model(model, stream)

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_neural_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_tokens = 0
    branch_entropy_sum = 0.0
    n_batches = 0

    use_synergy = model.mb_mode == MultiBranchMode.M4_COUPLED_R8_CALIBRATED_SYNERGY
    val_kn_list = _get_or_build_val_kn_log_priors(stream, model.vocab_size) if use_synergy else None

    sem_dim = model.sem_real_dim
    id_dim = model.id_real_dim
    car_dim = DEFAULT_CARRIER_BIAS_DIM
    cap = float(DEFAULT_SOFTCAP_LOGIT)
    inv_cap = 1.0 / cap
    log_alpha = math.log(0.55)
    log_one_minus_alpha = math.log(0.45)

    for b_idx, (idx_in, targets) in enumerate(stream.val_batches):
        (
            h_base_464,
            h_sem1_448,
            h_id_branches,
            branch_controls,
            w_aug,
        ) = model.build_multibranch_readout_tensors(idx_in)
        sem1_gates = branch_controls[:, :, :, 0]
        id_scales = branch_controls[:, :, :, 1]
        gate_logits = branch_controls[:, :, :, 2]

        bsz, seq_len = idx_in.shape
        n_tok = int(targets.numel())
        num_r = model.num_branches
        v_sz = model.vocab_size

        w_sem = w_aug[:, :sem_dim]
        w_id = w_aug[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous()
        w_sem_t = w_sem.t().contiguous()
        w_id_t = w_id.t().contiguous()

        pi_b = F.softmax(gate_logits.float(), dim=-1)
        ent_b = -(pi_b * torch.log(pi_b.clamp_min(1e-8))).sum(dim=-1).mean()
        branch_entropy_sum += float(ent_b.item())
        n_batches += 1

        kn_batch_cpu = val_kn_list[b_idx] if val_kn_list is not None else None
        for b0 in range(0, bsz, 1):
            b1 = b0 + 1
            c_len = seq_len
            yc = targets[b0:b1].reshape(-1)

            hb_c = h_base_464[b0:b1].reshape(c_len, sem_dim + car_dim)
            hs1_c = h_sem1_448[b0:b1].reshape(c_len, sem_dim)
            hid_c = h_id_branches[b0:b1].reshape(c_len * num_r, id_dim)
            b_c = sem1_gates[b0:b1].reshape(c_len, num_r, 1).float()
            t_c = id_scales[b0:b1].reshape(c_len, num_r, 1).float()
            log_pi = F.log_softmax(gate_logits[b0:b1].reshape(c_len, num_r).float(), dim=-1).unsqueeze(-1)

            z_base = torch.matmul(hb_c, w_base_t).float()
            z_sem1 = torch.matmul(hs1_c, w_sem_t).float()
            z_id = torch.matmul(hid_c, w_id_t).float().view(c_len, num_r, v_sz)

            raw_r = z_base.unsqueeze(1) + b_c * z_sem1.unsqueeze(1) + t_c * z_id
            z_r = cap * torch.tanh(raw_r * inv_cap)
            log_p_r = F.log_softmax(z_r, dim=-1)
            log_p_neural = torch.logsumexp(log_pi + log_p_r, dim=1)  # (c_len, V)
            del raw_r, z_r, log_p_r, z_id

            total_neural_loss += float(F.nll_loss(log_p_neural, yc, reduction="sum").item())

            if kn_batch_cpu is not None:
                log_kn = F.log_softmax(
                    kn_batch_cpu[b0:b1].to(device=h_base_464.device, dtype=torch.float32).reshape(c_len, v_sz),
                    dim=-1,
                )
                log_eval = torch.logaddexp(log_p_neural + log_one_minus_alpha, log_kn + log_alpha)
            else:
                log_eval = log_p_neural

            total_loss += float(F.nll_loss(log_eval, yc, reduction="sum").item())
            top5_idx = torch.topk(log_eval, k=5, dim=-1, largest=True, sorted=True).indices
            total_top1 += int((top5_idx[:, 0] == yc).sum().item())
            total_top5 += int((top5_idx == yc.unsqueeze(1)).any(dim=-1).sum().item())

        total_tokens += n_tok

    if stream.device.type == "cuda":
        torch.cuda.empty_cache()
    if was_training:
        model.train()

    val_ce = total_loss / float(max(1, total_tokens))
    neural_ce = total_neural_loss / float(max(1, total_tokens))
    mean_ent = branch_entropy_sum / float(max(1, n_batches))
    return {
        "val_ce": round(val_ce, 4),
        "neural_val_ce": round(neural_ce, 4),
        "val_ppl": round(float(math.exp(min(20.0, val_ce))), 2),
        "top1_acc": round(float(total_top1) / float(max(1, total_tokens)), 4),
        "top5_acc": round(float(total_top5) / float(max(1, total_tokens)), 4),
        "branch_gate_entropy_nats": round(mean_ent, 4),
        "effective_active_branches": round(float(math.exp(mean_ent)), 2),
    }


@torch.no_grad()
def calibrate_multibranch_residual_step0_(
    model: MultiBranchWavepacketGPT,
    stream: LongSeqFineWebStream,
) -> None:
    """Calibrate `sem1_bridge` and `id_branch_stiefel` on Shard-1 residuals orthogonal to Anchor Branch 0."""
    if model.mb_head is None:
        return
    device = stream.device
    d_emb = model.embed_dim
    sem_dim = model.sem_real_dim
    id_dim = model.id_real_dim
    n_extra = max(1, model.num_branches - 1)

    target_table = model.w_det_512.detach().float() * model.radial_scale.detach().float().unsqueeze(1)
    cov_hh = torch.zeros((d_emb, d_emb), device=device, dtype=torch.float32)
    cov_sem_hy = torch.zeros((d_emb, sem_dim), device=device, dtype=torch.float32)
    cov_id_hy = torch.zeros((n_extra, d_emb, id_dim), device=device, dtype=torch.float32)

    n_cal = 8
    max_start = stream.shard1_gpu.numel() - stream.seq_len - 2
    stride = max(1, max_start // (n_cal * stream.batch_size))
    cal_samples: list[tuple[torch.Tensor, torch.Tensor]] = []

    for b_idx in range(n_cal):
        base = b_idx * stream.batch_size * stride
        starts = (base + torch.arange(stream.batch_size, device=device) * stride) % max_start
        offsets = torch.arange(stream.seq_len + 1, device=device).unsqueeze(0)
        chunk = stream.shard1_gpu[starts.unsqueeze(1) + offsets]
        idx_in, targets = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
        if b_idx < 2:
            cal_samples.append((idx_in, targets))

        h_pre, h_mid, readout_sem_448, readout_id_64 = model._encode_multiscale_trunk(idx_in)
        h_bridge = model.exit_bridge(h_pre)
        h_sem_tr = F.rms_norm(h_bridge[:, :, :sem_dim], (sem_dim,), eps=1e-6)
        h_id_tr = F.rms_norm(h_bridge[:, :, sem_dim:], (id_dim,), eps=1e-6)
        if readout_sem_448 is not None and readout_id_64 is not None:
            g_sem, g_id = model.trunk_sem_gate.to(h_pre.dtype), model.trunk_id_gate.to(h_pre.dtype)
            h_sem_0 = F.rms_norm(g_sem * h_sem_tr + readout_sem_448, (sem_dim,), eps=1e-6).float()
            h_id_0 = F.rms_norm(g_id * h_id_tr + readout_id_64, (id_dim,), eps=1e-6).float()
        else:
            h_sem_0, h_id_0 = h_sem_tr.float(), h_id_tr.float()

        src_sem = F.rms_norm(0.70 * h_pre + 0.30 * h_mid, (d_emb,), eps=1e-6).reshape(-1, d_emb).float()
        src_id = F.rms_norm(0.75 * h_pre + 0.25 * h_mid, (d_emb,), eps=1e-6).reshape(-1, d_emb).float()

        y_flat = target_table[targets.reshape(-1)]
        y_sem = y_flat[:, :sem_dim]
        y_id = y_flat[:, sem_dim:]
        s0_flat = h_sem_0.reshape(-1, sem_dim)
        i0_flat = h_id_0.reshape(-1, id_dim)

        # Orthogonalize target semantic & identity vectors against Anchor Branch 0 predictions
        proj_s = (y_sem * s0_flat).sum(dim=-1, keepdim=True) / s0_flat.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        res_sem = y_sem - proj_s * s0_flat

        proj_i = (y_id * i0_flat).sum(dim=-1, keepdim=True) / i0_flat.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        res_id = y_id - proj_i * i0_flat

        cov_hh.add_(torch.matmul(src_sem.t(), src_sem))
        cov_sem_hy.add_(torch.matmul(src_sem.t(), res_sem))
        r_slot = b_idx % n_extra
        cov_id_hy[r_slot].add_(torch.matmul(src_id.t(), res_id))

    n_tok = float(n_cal * stream.batch_size * stream.seq_len)
    cov_hh.div_(n_tok)
    cov_sem_hy.div_(n_tok)
    reg_eye = 0.08 * torch.eye(d_emb, device=device, dtype=torch.float32)

    if model.mb_head.use_coupled_semantic:
        w_sem1 = torch.linalg.solve(cov_hh + reg_eye, cov_sem_hy)  # (512, 448)
        u_s, s_s, vh_s = torch.linalg.svd(w_sem1.t(), full_matrices=False)  # u_s: (448, 448), vh_s: (448, 512)
        s_rms = s_s.square().mean().sqrt().clamp_min(1e-6)
        model.mb_head.sem1_bridge.u_stiefel.copy_(u_s.contiguous())
        model.mb_head.sem1_bridge.v_stiefel.copy_(vh_s.t().contiguous())
        model.mb_head.sem1_bridge.sigma.copy_((s_s / s_rms).clamp(0.05, 4.0).contiguous())

    for r_idx in range(n_extra):
        w_id_r = torch.linalg.solve(cov_hh + reg_eye, cov_id_hy[r_idx] / float(max(1, n_cal // n_extra)))  # (512, 64)
        u_i, _, vh_i = torch.linalg.svd(w_id_r.t(), full_matrices=False)  # (64, 512)
        model.mb_head.id_branch_stiefel.data[r_idx].copy_(torch.matmul(u_i, vh_i))

    # Line-search initial Anchor Branch 0 gate bias on Shard-1 calibration batches
    was_training = model.training
    model.eval()
    best_bias, best_l = DEFAULT_ANCHOR_BRANCH_LOGIT_BIAS, float("inf")
    for cand_bias in (1.10, 1.55, 2.00):
        model.mb_head.branch_gate_proj.bias.zero_()
        model.mb_head.branch_gate_proj.bias[0] = cand_bias
        tot_l = sum(float(model.forward_loss(x_b, y_b).item()) for x_b, y_b in cal_samples)
        if tot_l < best_l:
            best_l, best_bias = tot_l, cand_bias
    model.mb_head.branch_gate_proj.bias.zero_()
    model.mb_head.branch_gate_proj.bias[0] = best_bias
    if was_training:
        model.train()
    del target_table, cov_hh, cov_sem_hy, cov_id_hy, cal_samples


def run_multibranch_wavepacket_experiment(
    spec: MultiBranchReadoutSpec,
    stream: LongSeqFineWebStream,
    steps: int = 300,
    eval_every: int = 100,
) -> dict[str, Any]:
    """Train and evaluate a `MultiBranchWavepacketGPT` variant on FineWeb (`L = 4, d = 512, T = 512`)."""
    device = stream.device
    torch.manual_seed(int(spec.seed))
    model = MultiBranchWavepacketGPT(spec).to(device)
    saved_mb_head = model.mb_head
    model.mb_head = None
    init_ms, meta = initialize_and_calibrate_shallow_model_(model, stream)
    model.mb_head = saved_mb_head
    if model.mb_head is not None:
        t_res0 = time.perf_counter()
        calibrate_multibranch_residual_step0_(model, stream)
        init_ms += (time.perf_counter() - t_res0) * 1000.0
    model.clamp_identity_dispersion_and_scalars_()

    # Build Riemannian Stiefel + Complex Unitary + Product-Sphere Optimizer Suite
    stiefel_params: list[nn.Parameter] = []
    unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for q_blk in model.quantum_blocks:
        stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
        unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]
    stiefel_params.extend(model.exit_bridge.real_stiefel_parameters())

    fast_temp_params: list[nn.Parameter] = [
        model.temp_sem_scale,
        model.temp_id_scale,
        model.trunk_sem_gate,
        model.trunk_id_gate,
    ]
    scalar_params: list[nn.Parameter] = [
        model.wp_causal_re,
        model.wp_causal_im,
        model.smear_gate.weight,
        model.smear_lambda,
        model.radial_scale,
        model.unigram_bias,
        model.prior_gate,
    ]
    scalar_params.extend(model.exit_bridge.scalar_parameters())
    for q_blk in model.quantum_blocks:
        scalar_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]

    mb_stiefel_params: list[nn.Parameter] = []
    if model.mb_head is not None:
        stiefel_params.extend(model.mb_head.real_stiefel_parameters())
        scalar_params.extend(model.mb_head.scalar_parameters())
        mb_stiefel_params.append(model.mb_head.id_branch_stiefel)

    q_trunk_opt = ModularRiemannianStiefelKrausMuon(
        real_stiefel_params=stiefel_params,
        complex_operator_pairs=unitary_pairs,
        stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR,
        unitary_lr=DEFAULT_SHALLOW_UNITARY_LR,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
    )
    wte_opt = BatchSubspaceTikhonovGramMuon(
        [model.wte_3d], lr=DEFAULT_SHALLOW_WTE_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA
    )
    head_opt = BatchSubspaceTikhonovGramMuon(
        [model.w_det_512], lr=DEFAULT_SHALLOW_READOUT_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA
    )

    adam_groups: list[dict[str, Any]] = [
        {"params": scalar_params, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0},
        {"params": fast_temp_params, "lr": 0.06, "weight_decay": 0.0},
    ]
    if model.sem_ngram is not None:
        ngram_tables = [
            model.sem_ngram.unigram_next_table,
            model.sem_ngram.bigram_table,
            model.sem_ngram.trigram_table,
            model.sem_ngram.skipgram_table,
            model.sem_ngram.u_sem,
        ]
        ngram_scalars = [
            model.sem_ngram.in_scale,
            model.sem_ngram.readout_sem_scale,
            model.sem_ngram.readout_id_scale,
            model.sem_ngram.order_logits,
        ]
        adam_groups.append({"params": ngram_tables, "lr": DEFAULT_SHALLOW_NGRAM_LR, "weight_decay": 0.0})
        adam_groups.append({"params": ngram_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0})
    if mb_stiefel_params:
        adam_groups.append({"params": mb_stiefel_params, "lr": DEFAULT_SHALLOW_STIEFEL_LR, "weight_decay": 0.0})

    adam_opt = torch.optim.AdamW(adam_groups, betas=(0.80, 0.95), eps=1e-8, fused=(device.type == "cuda"))

    checkpoints: list[dict[str, Any]] = []
    step0_eval = evaluate_multibranch_model(model, stream)
    step0_eval["step"] = 0
    step0_eval["tokens_M"] = 0.0
    checkpoints.append(step0_eval)
    print(
        f"[{spec.mode.value}] Step 0 (0.00M tok) | "
        f"Val CE={step0_eval['val_ce']:.4f} (Neural={step0_eval['neural_val_ce']:.4f}) "
        f"PPL={step0_eval['val_ppl']:.2f} Top1={step0_eval['top1_acc']*100:.2f}% "
        f"Top5={step0_eval['top5_acc']*100:.2f}% "
        f"ActiveBranches={step0_eval.get('effective_active_branches', 1.0):.2f}/{model.num_branches}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train0 = time.perf_counter()
    tok_per_step = spec.batch_size * spec.seq_len
    total_steps = int(steps)

    model.train()
    for step in range(1, total_steps + 1):
        warm_steps = max(1, int(0.08 * total_steps))
        if step <= warm_steps:
            lr_mult = 0.20 + 0.80 * (float(step - 1) / float(max(1, warm_steps - 1)))
        else:
            prog = float(step - warm_steps) / float(max(1, total_steps - warm_steps))
            lr_mult = 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * prog))
        mom_beta = compute_riemannian_momentum_beta(
            step, total_steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070
        )

        q_trunk_opt.zero_grad(set_to_none=True)
        wte_opt.zero_grad(set_to_none=True)
        head_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

        idx_in, targets = stream.sample_train_batch(step)
        loss = model.forward_loss(idx_in, targets)
        loss.backward()

        q_trunk_opt.set_momentum(mom_beta)
        q_trunk_opt.set_lr_scale(lr_mult)
        q_trunk_opt.step()

        active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
        wte_opt.set_momentum(mom_beta)
        wte_opt.step_active(active_ids, lr_scale=lr_mult)
        head_opt.set_momentum(mom_beta)
        head_opt.step_full(lr_scale=lr_mult, active_idx=active_ids)

        adam_opt.param_groups[0]["lr"] = DEFAULT_SHALLOW_SCALAR_LR * lr_mult
        adam_opt.param_groups[1]["lr"] = 0.06 * lr_mult
        if len(adam_opt.param_groups) > 2:
            adam_opt.param_groups[2]["lr"] = DEFAULT_SHALLOW_NGRAM_LR * lr_mult
            adam_opt.param_groups[3]["lr"] = DEFAULT_SHALLOW_SCALAR_LR * lr_mult
        if len(adam_opt.param_groups) > 4:
            adam_opt.param_groups[4]["lr"] = DEFAULT_SHALLOW_STIEFEL_LR * lr_mult
        adam_opt.step()

        model.clamp_identity_dispersion_and_scalars_()

        if step % int(eval_every) == 0 or step == total_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t_train0
            ev = evaluate_multibranch_model(model, stream)
            ev["step"] = step
            ev["tokens_M"] = round((step * tok_per_step) / 1e6, 3)
            ev["train_loss"] = round(float(loss.detach().item()), 4)
            ev["elapsed_s"] = round(elapsed_s, 1)
            checkpoints.append(ev)
            print(
                f"[{spec.mode.value}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok, {elapsed_s:5.1f}s) | "
                f"Train CE={ev['train_loss']:.4f} | Val CE={ev['val_ce']:.4f} (Neural={ev['neural_val_ce']:.4f}) "
                f"PPL={ev['val_ppl']:.2f} Top1={ev['top1_acc']*100:.2f}% Top5={ev['top5_acc']*100:.2f}% "
                f"ActiveBranches={ev.get('effective_active_branches', 1.0):.2f}/{model.num_branches}",
                flush=True,
            )

    total_wall_s = time.perf_counter() - t_train0
    result = {
        "mode": spec.mode.value,
        "num_branches": model.num_branches,
        "use_coupled_semantic": spec.use_coupled_semantic,
        "num_layers": spec.num_layers,
        "embed_dim": spec.embed_dim,
        "seq_len": spec.seq_len,
        "batch_size": spec.batch_size,
        "total_steps": total_steps,
        "total_tokens_M": round((total_steps * tok_per_step) / 1e6, 3),
        "init_ms": round(init_ms, 2),
        "train_wall_s": round(total_wall_s, 2),
        "meta": meta,
        "step0": checkpoints[0],
        "final": checkpoints[-1],
        "checkpoints": checkpoints,
    }
    del model, q_trunk_opt, wte_opt, head_opt, adam_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
