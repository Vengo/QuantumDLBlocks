"""Two 8-Qubit Registers ("Qubytes") $(\\mathbb{C}^2)^{\\otimes 8}_{\\text{hi}} \\otimes (\\mathbb{C}^2)^{\\otimes 8}_{\\text{lo}}$ with 8-Qubit Walsh-Hadamard Pauli-$Z$ Basis, Step-1 Multi-Mode Qubyte Distribution, Step-3 Latent `512D` Contraction, & Rank-$R$ Schmidt Entanglement (`EXPERIMENT_PLAN_082.md` / `Wave 62`).

Evaluates representing the `50,257` GPT-2 vocabulary as **Two 8-Qubit Registers ("Qubytes")** corresponding to the high and low bytes $(b_{\\text{hi}}, b_{\\text{lo}})$ of the deterministic `uint16` token decomposition, while slashing readout compute via:
1. **Step 3 — Latent `512D` Cholesky Mode Contraction Before Vocabulary Projection (`Q0`)**:
   Contracts the $R$ vacuum-gated Cholesky modes inside `512D` Hilbert space before multiplying by $W_{\\text{aug}} \\in \\mathbb{R}^{50304 \\times 528}$, reducing full `50,304`-vocabulary GEMMs from $2R + 1$ in forward (`4R + 2` in backward) down to **1 in forward (`2` in backward)**.
2. **Step 1 — Multi-Mode Cholesky Distribution Over the `(197, 256)` Qubyte Basis Before Vocabulary Expansion (`Q1`)**:
   Projects all $R$ Cholesky modes onto the `197` high-byte and `256` low-byte basis states (`453` dot products/mode — **`111x` fewer multi-mode FLOPs** than `50,304`), preserving nonlinear multi-mode mixture competition and log-density innovation $\\Delta \\Phi^{\\text{step1}}_t[b_{\\text{hi}}, b_{\\text{lo}}]$ while querying the `50,304` fine residual table only once via Step 3.
3. **Two 8-Qubit Registers with Exact 8-Qubit Walsh-Hadamard Pauli-$Z^{\\otimes S}$ Basis ($H^{\\otimes 8}$) & 16-Qubit Bloch-Ising Hamiltonian (`Q2`)**:
   Couples all 256 computational basis states $|q_7 \\dots q_0\\rangle$ of each 8-qubit register through the exact $256 \\times 256$ Walsh-Hadamard unitary $H^{\\otimes 8}_{b,S} = \\frac{1}{16}(-1)^{\\operatorname{popcount}(b \\,\\&\\, S)}$ gated by Pauli $k$-body Hamming weight $|S| \\in \\{0 \\dots 8\\}$, plus 16-qubit longitudinal Pauli-$Z_j$ fields and cross-register $Z_i^{\\text{hi}} Z_j^{\\text{lo}}$ Ising couplings.
4. **Rank-$R$ Bipartite Schmidt Entanglement Across the Two 8-Qubit Registers (`Q3`, `Q4`, `Q5`)**:
   Treats the $R$ Cholesky wavepacket modes as $R$ bipartite Schmidt entanglement modes $\\sum_{r=0}^{R-1} \\pi_{t,r}\\, |\\alpha_{t,r}\\rangle_{\\text{hi}} \\otimes |\\beta_{t,r}\\rangle_{\\text{lo}} \\in \\mathbb{R}^{197 \\times 256}$, evaluated via a single `(N, 197, R) @ (N, R, 256)` batched matmul.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from qdlblocks.benchmarks.noclick_vacuum_and_smith_sgd_lab import _apply_global_optimizer_step
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_NGRAM_LR,
    DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SOFTCAP_LOGIT,
    RAW_GPT2_VOCAB_SIZE,
    LongSeqFineWebStream,
)
from qdlblocks.benchmarks.vacuum_subspace_cholesky_lab import DEFAULT_VACUUM_SUBSPACE_DIM as DEFAULT_VACUUM_DIM
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    NUM_HIGH_BYTE_BUCKETS,
    NUM_LOW_BYTE_BUCKETS,
    ZeroPPMIBytePairCholeskyGPT,
    ZeroPPMIBytePairCholeskySpec,
    ZeroPPMIBytePairVariantMode,
    _compute_zero_ppmi_step_lr_and_momentum,
    build_zero_ppmi_byte_pair_optimizers,
    decompose_token_uint16_byte_pair,
    initialize_zero_ppmi_byte_pair_model_,
)
from qdlblocks.models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from qdlblocks.optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon

# -----------------------------------------------------------------------------
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_082.md Rule 7)
# -----------------------------------------------------------------------------
QUBYTE_NUM_QUBITS: int = 8
QUBYTE_HILBERT_DIM: int = 256
DEFAULT_PAULI_BASE_LOGIT: float = 0.80
DEFAULT_PAULI_HAMMING_DECAY: float = 0.35
DEFAULT_WALSH_BASIS_SCALE: float = 0.25
DEFAULT_STEP1_INNOVATION_SCALE: float = 0.35
DEFAULT_SCHMIDT_ENTANGLE_SCALE: float = 0.40
DEFAULT_SCHMIDT_TEMP: float = 2.50
DEFAULT_ISING_COUPLING_SCALE: float = 0.15
DEFAULT_QUBYTE_AUX_LOSS_WEIGHT: float = 0.08
SINGLE_GEMM_TOKEN_CHUNK: int = 2048


class TwoQubyteVariantMode(str, Enum):
    """Ablation and scaling modes for `EXPERIMENT_PLAN_082.md` (`Wave 62`)."""

    Q0_STEP3_ONLY_LATENT_512D_CONTRACTION = "q0_step3_only_latent_512d_contraction"
    Q1_STEP1_PLUS_STEP3_FACTORED_MULTIMODE = "q1_step1_plus_step3_factored_multimode"
    Q2_TWO_8QUBIT_REGISTERS_WALSH_PAULI_ISING = "q2_two_8qubit_registers_walsh_pauli_ising"
    Q3_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R8 = "q3_two_8qubit_schmidt_entangled_cholesky_r8"
    Q4_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R12 = "q4_two_8qubit_schmidt_entangled_cholesky_r12"
    Q5_TWO_8QUBIT_SCHMIDT_SCALING_1200 = "q5_two_8qubit_schmidt_scaling_1200"


@dataclass(frozen=True)
class TwoQubyteRegisterCholeskySpec:
    """Configuration for Two 8-Qubit Register ("Qubytes") & Step-1/Step-3 Cholesky GPT (`<= 8` fields)."""

    variant_mode: TwoQubyteVariantMode = TwoQubyteVariantMode.Q3_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R8
    num_cholesky_modes: int = 8
    vacuum_dim: int = DEFAULT_VACUUM_DIM
    enable_step1_multimode_qubyte: bool = True
    enable_walsh_pauli_qubytes: bool = True
    enable_schmidt_entanglement: bool = True
    enable_hierarchical_bhi_carrier: bool = True
    seed: int = 1337


def build_8qubit_walsh_hadamard_and_spins(
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Construct exact $256 \times 256$ 8-qubit Walsh-Hadamard Pauli-$Z^{\otimes S}$ unitary $H^{\otimes 8}$, Hamming weights $|S|$, and spins $s(b) \in \{\pm 1\}^8$."""
    states = torch.arange(QUBYTE_HILBERT_DIM, device=device, dtype=torch.long)
    bit_shifts = torch.arange(QUBYTE_NUM_QUBITS, device=device, dtype=torch.long)

    # Spins s_j(b) = 1 - 2 * ((b >> j) & 1) in {+1, -1} for qubits j = 0..7
    bits = (states.unsqueeze(1) >> bit_shifts.unsqueeze(0)) & 1
    qubit_spins = (1.0 - 2.0 * bits.float()).contiguous()

    # Exact 8-qubit Walsh-Hadamard matrix H^{\otimes 8}_{b, S} = (1 / 16) * (-1)^{popcount(b & S)}
    bitwise_and = states.unsqueeze(1) & states.unsqueeze(0)
    popcounts = torch.zeros_like(bitwise_and)
    for j in range(QUBYTE_NUM_QUBITS):
        popcounts = popcounts + ((bitwise_and >> j) & 1)
    parity_sign = 1.0 - 2.0 * (popcounts & 1).float()
    walsh_h8 = (parity_sign / math.sqrt(float(QUBYTE_HILBERT_DIM))).contiguous()

    # Hamming weight |S| = popcount(S) in {0, ..., 8}
    hamming_weights = bits.sum(dim=-1).float().contiguous()
    return walsh_h8, hamming_weights, qubit_spins


def contract_step3_latent_512d_modes(
    h_base_464: torch.Tensor,
    h_sem_branches: torch.Tensor,
    h_id_branches: torch.Tensor,
    branch_controls: torch.Tensor,
) -> torch.Tensor:
    r"""Contract $R$ Cholesky modes in `512D` latent space (**Step 3**) before the single `50,304` vocabulary GEMM (`<= 8` args).

    Given anchor semantic state $h_{t,\text{sem},0} \in \mathbb{R}^{448}$, carrier $h_{t,\text{car}} \in \mathbb{R}^{16}$,
    and $R$ Cholesky branches $(h_{t,\text{sem},r}, h_{t,\text{id},r})$ with controls $(\beta_{t,r}, \tau_{t,r}, \alpha_{t,r})$:
    $$\bar{h}_{t,\text{sem}} = h_{t,\text{sem},0} + \sum_{r=0}^{R-1} \pi_{t,r}\,\beta_{t,r}\,h_{t,\text{sem},r}, \qquad
      \bar{h}_{t,\text{id}} = \sum_{r=0}^{R-1} \pi_{t,r}\,\tau_{t,r}\,h_{t,\text{id},r}$$
    Returns `h_step3_528` of shape `(B, T, 528)`.
    """
    work_dtype = h_base_464.dtype
    sem_dim = h_sem_branches.size(-1)
    h_sem_0 = h_base_464[:, :, :sem_dim]
    h_car_16 = h_base_464[:, :, sem_dim:]

    beta = branch_controls[:, :, :, 0].float()
    tau = branch_controls[:, :, :, 1].float()
    pi = F.softmax(branch_controls[:, :, :, 2].float(), dim=-1)

    w_sem_r = (pi * beta).to(work_dtype).unsqueeze(-1)
    w_id_r = (pi * tau).to(work_dtype).unsqueeze(-1)

    if h_sem_branches.dim() == 4:
        h_sem_bar = h_sem_0 + (w_sem_r * h_sem_branches).sum(dim=2)
    else:
        w_sem_sum = (pi * beta).sum(dim=-1, keepdim=True).to(work_dtype)
        h_sem_bar = h_sem_0 + w_sem_sum * h_sem_branches

    h_id_bar = (w_id_r * h_id_branches).sum(dim=2)
    return torch.cat([h_sem_bar, h_id_bar, h_car_16], dim=-1).contiguous()


class _SingleGemmStep3QubyteCEFn(torch.autograd.Function):
    """Rule-9 single-pass chunked Cross-Entropy kernel executing **1 forward and 2 backward `50,304` GEMMs** + factored `(197, K) @ (K, 256)` Two-Qubyte bmm per token chunk."""

    @staticmethod
    def forward(
        ctx: Any,
        h_step3_528: torch.Tensor,
        q_hi_factors: torch.Tensor | None,
        q_lo_factors: torch.Tensor | None,
        w_aug_528: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exact softcapped Cross-Entropy over `50,304` tokens with 1 GEMM + factored `(c_len, K, 197)` x `(c_len, K, 256)` Two-Qubyte bmm."""
        bsz, seq_len, dim_aug = h_step3_528.shape
        n_total = bsz * seq_len
        v_sz = w_aug_528.size(0)
        n_valid_vocab = min(RAW_GPT2_VOCAB_SIZE, v_sz)

        h_flat = h_step3_528.reshape(n_total, dim_aug)
        has_qubyte = (q_hi_factors is not None) and (q_lo_factors is not None)
        if has_qubyte:
            assert q_hi_factors is not None and q_lo_factors is not None
            k_rank = q_hi_factors.size(2)
            q_hi_flat = q_hi_factors.reshape(n_total, k_rank, NUM_HIGH_BYTE_BUCKETS)
            q_lo_flat = q_lo_factors.reshape(n_total, k_rank, NUM_LOW_BYTE_BUCKETS)
        else:
            q_hi_flat = None
            q_lo_flat = None

        y_flat = targets.reshape(n_total)
        w_t = w_aug_528.t().contiguous()

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        total_nll = torch.zeros((), device=h_step3_528.device, dtype=torch.float32)
        n_valid = int((y_flat >= 0).sum().item())
        all_valid = bool(n_valid == n_total)

        for start in range(0, n_total, SINGLE_GEMM_TOKEN_CHUNK):
            end = min(start + SINGLE_GEMM_TOKEN_CHUNK, n_total)
            c_len = end - start
            yc = y_flat[start:end]
            if not all_valid and not bool((yc >= 0).any().item()):
                continue

            z_c = torch.matmul(h_flat[start:end], w_t).float()
            if q_hi_flat is not None and q_lo_flat is not None:
                q_grid_c = torch.bmm(q_hi_flat[start:end].transpose(1, 2), q_lo_flat[start:end]).float()
                z_c[:, :n_valid_vocab].add_(
                    q_grid_c.reshape(c_len, NUM_HIGH_BYTE_BUCKETS * NUM_LOW_BYTE_BUCKETS)[:, :n_valid_vocab]
                )
                del q_grid_c
            z_c.mul_(inv_cap).tanh_().mul_(cap)

            yc_safe = yc.clamp_min(0).unsqueeze(-1)
            target_log_p = z_c.gather(dim=-1, index=yc_safe).squeeze(-1) - torch.logsumexp(z_c, dim=-1)
            del z_c
            total_nll = total_nll - (target_log_p.sum() if all_valid else target_log_p[yc >= 0].sum())

        if has_qubyte:
            assert q_hi_factors is not None and q_lo_factors is not None
            ctx.save_for_backward(h_step3_528, q_hi_factors, q_lo_factors, w_aug_528, targets)
        else:
            ctx.save_for_backward(h_step3_528, w_aug_528, targets)
        ctx.has_qubyte = has_qubyte
        ctx.n_valid = max(1, n_valid)
        ctx.all_valid = all_valid
        return total_nll / float(max(1, n_valid))

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, None]:
        """Single-pass chunked backward computing `grad_h_step3`, `grad_q_hi`, `grad_q_lo`, and `grad_w_aug` inside one loop (Rule 9)."""
        has_qubyte = bool(ctx.has_qubyte)
        if has_qubyte:
            h_step3_528, q_hi_factors, q_lo_factors, w_aug_528, targets = ctx.saved_tensors
        else:
            h_step3_528, w_aug_528, targets = ctx.saved_tensors
            q_hi_factors = None
            q_lo_factors = None

        bsz, seq_len, dim_aug = h_step3_528.shape
        n_total = bsz * seq_len
        v_sz = w_aug_528.size(0)
        n_valid_vocab = min(RAW_GPT2_VOCAB_SIZE, v_sz)
        work_dtype = h_step3_528.dtype
        all_valid = bool(ctx.all_valid)

        h_flat = h_step3_528.reshape(n_total, dim_aug)
        if has_qubyte and q_hi_factors is not None and q_lo_factors is not None:
            k_rank = q_hi_factors.size(2)
            q_hi_flat = q_hi_factors.reshape(n_total, k_rank, NUM_HIGH_BYTE_BUCKETS)
            q_lo_flat = q_lo_factors.reshape(n_total, k_rank, NUM_LOW_BYTE_BUCKETS)
            grad_q_hi = torch.zeros_like(q_hi_flat)
            grad_q_lo = torch.zeros_like(q_lo_flat)
        else:
            q_hi_flat = None
            q_lo_flat = None
            grad_q_hi = None
            grad_q_lo = None

        y_flat = targets.reshape(n_total)
        w_t = w_aug_528.t().contiguous()

        grad_h = torch.zeros_like(h_flat)
        grad_w = torch.zeros_like(w_aug_528)

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        scale = float(grad_output.item()) / float(ctx.n_valid)
        n_qubyte_cells = NUM_HIGH_BYTE_BUCKETS * NUM_LOW_BYTE_BUCKETS

        for start in range(0, n_total, SINGLE_GEMM_TOKEN_CHUNK):
            end = min(start + SINGLE_GEMM_TOKEN_CHUNK, n_total)
            c_len = end - start
            yc = y_flat[start:end]
            valid_c = yc >= 0
            if not all_valid and not bool(valid_c.any().item()):
                continue

            h_c = h_flat[start:end]
            tanh_u = torch.matmul(h_c, w_t).float()
            if q_hi_flat is not None and q_lo_flat is not None:
                q_grid_c = torch.bmm(q_hi_flat[start:end].transpose(1, 2), q_lo_flat[start:end]).float()
                tanh_u[:, :n_valid_vocab].add_(q_grid_c.reshape(c_len, n_qubyte_cells)[:, :n_valid_vocab])
                del q_grid_c
            tanh_u.mul_(inv_cap).tanh_()

            p_c = F.softmax(tanh_u * cap, dim=-1)
            yc_idx = yc.clamp_min(0).unsqueeze(-1)
            p_c.scatter_add_(
                dim=-1,
                index=yc_idx,
                src=torch.full((c_len, 1), -1.0, device=p_c.device, dtype=p_c.dtype),
            )
            valid_f = (
                torch.full((c_len, 1), scale, device=p_c.device, dtype=torch.float32)
                if all_valid
                else (valid_c.float().unsqueeze(-1) * scale)
            )
            tanh_u.square_().neg_().add_(1.0)
            p_c.mul_(valid_f).mul_(tanh_u)
            del tanh_u

            g_w = p_c.to(work_dtype)
            del p_c

            if grad_q_hi is not None and grad_q_lo is not None and q_hi_flat is not None and q_lo_flat is not None:
                g_grid_flat = torch.zeros((c_len, n_qubyte_cells), device=g_w.device, dtype=work_dtype)
                g_grid_flat[:, :n_valid_vocab] = g_w[:, :n_valid_vocab]
                g_grid_c = g_grid_flat.view(c_len, NUM_HIGH_BYTE_BUCKETS, NUM_LOW_BYTE_BUCKETS)
                qh_c = q_hi_flat[start:end]
                ql_c = q_lo_flat[start:end]
                grad_q_hi[start:end] = torch.bmm(ql_c, g_grid_c.transpose(1, 2))
                grad_q_lo[start:end] = torch.bmm(qh_c, g_grid_c)
                del g_grid_flat, g_grid_c

            grad_h[start:end] = torch.matmul(g_w, w_aug_528)
            grad_w.add_(torch.matmul(g_w.t(), h_c))
            del g_w

        return (
            grad_h.view_as(h_step3_528),
            grad_q_hi.view_as(q_hi_factors) if (grad_q_hi is not None and q_hi_factors is not None) else None,
            grad_q_lo.view_as(q_lo_factors) if (grad_q_lo is not None and q_lo_factors is not None) else None,
            grad_w,
            None,
        )


class Two8QubitRegisterSynthesizer(nn.Module):
    r"""Two 8-Qubit Registers $(\mathbb{C}^2)^{\otimes 8}_{\text{hi}} \otimes (\mathbb{C}^2)^{\otimes 8}_{\text{lo}}$ with 8-Qubit Walsh-Hadamard Pauli-$Z^{\otimes S}$ Basis, 16-Qubit Ising Coupling, Step-1 Multi-Mode Projection, & Rank-$R$ Schmidt Entanglement."""

    def __init__(self, pkt_count: int = 8, pkt_dim: int = 64, sem_dim: int = 448, id_dim: int = 64) -> None:
        super().__init__()
        self.pkt_count = int(pkt_count)
        self.pkt_dim = int(pkt_dim)
        self.sem_dim = int(sem_dim)
        self.id_dim = int(id_dim)
        self.embed_dim = self.sem_dim + self.id_dim

        walsh_h8, hamming_w, qubit_spins = build_8qubit_walsh_hadamard_and_spins()
        self.register_buffer("walsh_h8", walsh_h8, persistent=False)
        self.register_buffer("hamming_weights", hamming_w, persistent=False)
        self.register_buffer("spins_hi", qubit_spins[:NUM_HIGH_BYTE_BUCKETS].contiguous(), persistent=False)
        self.register_buffer("spins_lo", qubit_spins.contiguous(), persistent=False)

        # 8-Qubit Pauli-Z^{\otimes S} spectral basis tables for input wavepackets and readout detectors
        self.v_in_pauli_hi = nn.Parameter(torch.zeros((QUBYTE_HILBERT_DIM, self.pkt_count, self.pkt_dim), dtype=torch.float32))
        self.v_in_pauli_lo = nn.Parameter(torch.zeros((QUBYTE_HILBERT_DIM, self.pkt_count, self.pkt_dim), dtype=torch.float32))
        self.v_det_pauli_hi = nn.Parameter(torch.zeros((QUBYTE_HILBERT_DIM, self.embed_dim), dtype=torch.float32))
        self.v_det_pauli_lo = nn.Parameter(torch.zeros((QUBYTE_HILBERT_DIM, self.embed_dim), dtype=torch.float32))

        init_pauli_logits = DEFAULT_PAULI_BASE_LOGIT - DEFAULT_PAULI_HAMMING_DECAY * hamming_w
        self.pauli_gate_logits_hi = nn.Parameter(init_pauli_logits.clone())
        self.pauli_gate_logits_lo = nn.Parameter(init_pauli_logits.clone())

        # 16-Qubit Bloch-Magnetization & Cross-Register Pauli-Z_i^{hi} Z_j^{lo} Ising Hamiltonian
        self.bloch_field_hi = nn.Parameter(torch.zeros((self.id_dim, QUBYTE_NUM_QUBITS), dtype=torch.float32))
        self.bloch_field_lo = nn.Parameter(torch.zeros((self.id_dim, QUBYTE_NUM_QUBITS), dtype=torch.float32))
        self.ising_coupling_j = nn.Parameter(torch.zeros((QUBYTE_NUM_QUBITS, QUBYTE_NUM_QUBITS), dtype=torch.float32))

        # Step-1 & Rank-R Schmidt Entanglement coupling scales
        self.walsh_basis_scale = nn.Parameter(torch.tensor(DEFAULT_WALSH_BASIS_SCALE, dtype=torch.float32))
        self.step1_innovation_scale = nn.Parameter(torch.tensor(DEFAULT_STEP1_INNOVATION_SCALE, dtype=torch.float32))
        self.schmidt_entangle_scale = nn.Parameter(torch.tensor(DEFAULT_SCHMIDT_ENTANGLE_SCALE, dtype=torch.float32))
        self.ising_coupling_scale = nn.Parameter(torch.tensor(DEFAULT_ISING_COUPLING_SCALE, dtype=torch.float32))
        self.qubyte_hi_temp = nn.Parameter(torch.tensor(1.60, dtype=torch.float32))
        self.qubyte_lo_temp = nn.Parameter(torch.tensor(1.60, dtype=torch.float32))

    def table_parameters(self) -> list[nn.Parameter]:
        """Return 8-qubit Pauli-$Z$ spectral tables and 16-qubit Bloch-Ising matrices."""
        return [
            self.v_in_pauli_hi,
            self.v_in_pauli_lo,
            self.v_det_pauli_hi,
            self.v_det_pauli_lo,
            self.bloch_field_hi,
            self.bloch_field_lo,
            self.ising_coupling_j,
        ]

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return Pauli Hamming-weight gate logits and Step-1 / Schmidt / Ising scales."""
        return [
            self.pauli_gate_logits_hi,
            self.pauli_gate_logits_lo,
            self.walsh_basis_scale,
            self.step1_innovation_scale,
            self.schmidt_entangle_scale,
            self.ising_coupling_scale,
            self.qubyte_hi_temp,
            self.qubyte_lo_temp,
        ]

    @torch.no_grad()
    def reset_parameters_(self, seed: int = 1337) -> None:
        """Initialize 8-qubit Pauli-$Z^{\\otimes S}$ spectral tables and 16-qubit Ising couplings deterministically."""
        device = self.v_det_pauli_hi.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 8282)

        for param in (self.v_in_pauli_hi, self.v_in_pauli_lo):
            raw = torch.randn(param.shape, generator=gen, device=device, dtype=torch.float32)
            param.copy_(F.normalize(raw, dim=-1, eps=1e-6))

        for param in (self.v_det_pauli_hi, self.v_det_pauli_lo):
            raw = torch.randn(param.shape, generator=gen, device=device, dtype=torch.float32)
            sem_p = F.rms_norm(raw[:, : self.sem_dim], (self.sem_dim,), eps=1e-6)
            id_p = F.rms_norm(raw[:, self.sem_dim :], (self.id_dim,), eps=1e-6)
            param.copy_(torch.cat([sem_p, id_p], dim=-1))

        init_logits = DEFAULT_PAULI_BASE_LOGIT - DEFAULT_PAULI_HAMMING_DECAY * self.hamming_weights
        self.pauli_gate_logits_hi.copy_(init_logits)
        self.pauli_gate_logits_lo.copy_(init_logits)

        nn.init.normal_(self.bloch_field_hi, std=0.08)
        nn.init.normal_(self.bloch_field_lo, std=0.08)
        nn.init.normal_(self.ising_coupling_j, std=0.05)

        self.walsh_basis_scale.fill_(DEFAULT_WALSH_BASIS_SCALE)
        self.step1_innovation_scale.fill_(DEFAULT_STEP1_INNOVATION_SCALE)
        self.schmidt_entangle_scale.fill_(DEFAULT_SCHMIDT_ENTANGLE_SCALE)
        self.ising_coupling_scale.fill_(DEFAULT_ISING_COUPLING_SCALE)
        self.qubyte_hi_temp.fill_(1.60)
        self.qubyte_lo_temp.fill_(1.60)

    def build_walsh_pauli_input_tables(self, work_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Synthesize `(197, 8, 64)` high-qubyte and `(256, 8, 64)` low-qubyte input wavepackets via $H^{\otimes 8}$."""
        h8 = self.walsh_h8.to(work_dtype)
        g_hi = torch.sigmoid(self.pauli_gate_logits_hi).to(work_dtype).view(QUBYTE_HILBERT_DIM, 1)
        g_lo = torch.sigmoid(self.pauli_gate_logits_lo).to(work_dtype).view(QUBYTE_HILBERT_DIM, 1)
        flat_dim = self.pkt_count * self.pkt_dim

        v_hi_flat = self.v_in_pauli_hi.to(work_dtype).view(QUBYTE_HILBERT_DIM, flat_dim) * g_hi
        v_lo_flat = self.v_in_pauli_lo.to(work_dtype).view(QUBYTE_HILBERT_DIM, flat_dim) * g_lo

        u_hi = torch.matmul(h8[:NUM_HIGH_BYTE_BUCKETS], v_hi_flat).view(NUM_HIGH_BYTE_BUCKETS, self.pkt_count, self.pkt_dim)
        u_lo = torch.matmul(h8, v_lo_flat).view(NUM_LOW_BYTE_BUCKETS, self.pkt_count, self.pkt_dim)
        return u_hi, u_lo

    def synthesize_qubyte_detector_bases(
        self,
        d_comp_hi: torch.Tensor,
        d_comp_lo: torch.Tensor,
        enable_walsh_pauli: bool,
        work_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Combine computational-basis `(197, 512)` & `(256, 512)` detectors with the 8-Qubit Walsh-Hadamard Pauli-$Z^{\otimes S}$ basis."""
        d_hi = d_comp_hi.to(work_dtype)
        d_lo = d_comp_lo.to(work_dtype)
        if enable_walsh_pauli:
            h8 = self.walsh_h8.to(work_dtype)
            w_sc = self.walsh_basis_scale.to(work_dtype)
            g_hi = torch.sigmoid(self.pauli_gate_logits_hi).to(work_dtype).unsqueeze(-1)
            g_lo = torch.sigmoid(self.pauli_gate_logits_lo).to(work_dtype).unsqueeze(-1)
            pauli_hi = torch.matmul(h8[:NUM_HIGH_BYTE_BUCKETS], self.v_det_pauli_hi.to(work_dtype) * g_hi)
            pauli_lo = torch.matmul(h8, self.v_det_pauli_lo.to(work_dtype) * g_lo)
            d_hi = d_hi + w_sc * pauli_hi
            d_lo = d_lo + w_sc * pauli_lo

        d_hi_norm = torch.cat(
            [
                F.rms_norm(d_hi[:, : self.sem_dim], (self.sem_dim,), eps=1e-6),
                F.rms_norm(d_hi[:, self.sem_dim :], (self.id_dim,), eps=1e-6),
            ],
            dim=-1,
        )
        d_lo_norm = torch.cat(
            [
                F.rms_norm(d_lo[:, : self.sem_dim], (self.sem_dim,), eps=1e-6),
                F.rms_norm(d_lo[:, self.sem_dim :], (self.id_dim,), eps=1e-6),
            ],
            dim=-1,
        )
        return d_hi_norm, d_lo_norm

    def compute_qubyte_step1_and_schmidt_factors(
        self,
        h_base_464: torch.Tensor,
        h_sem_branches: torch.Tensor,
        h_id_branches: torch.Tensor,
        branch_controls: torch.Tensor,
        d_qubyte_hi: torch.Tensor,
        d_qubyte_lo: torch.Tensor,
        log_p_hi_prior: torch.Tensor,
        spec: TwoQubyteRegisterCholeskySpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Distribute $R$ Cholesky modes over the `(197, 256)` Two-Qubyte basis (**Step 1**) and return compact factored tensors `(q_hi_factors, q_lo_factors)` of shapes `(B, T, K, 197)` and `(B, T, K, 256)` (`<= 8` args)."""
        bsz, seq_len, n_modes, _ = h_id_branches.shape
        n_pos = bsz * seq_len
        work_dtype = h_base_464.dtype

        h_sem_0 = h_base_464[:, :, : self.sem_dim].reshape(n_pos, self.sem_dim)
        h_id_flat = h_id_branches.reshape(n_pos * n_modes, self.id_dim)

        beta = branch_controls[:, :, :, 0].reshape(n_pos, n_modes, 1).float()
        tau = branch_controls[:, :, :, 1].reshape(n_pos, n_modes, 1).float()
        log_pi = F.log_softmax(branch_controls[:, :, :, 2].reshape(n_pos, n_modes).float(), dim=-1)
        pi = log_pi.exp()

        d_hi_sem, d_hi_id = d_qubyte_hi[:, : self.sem_dim], d_qubyte_hi[:, self.sem_dim :]
        d_lo_sem, d_lo_id = d_qubyte_lo[:, : self.sem_dim], d_qubyte_lo[:, self.sem_dim :]

        # Step 1: Project anchor & R Cholesky branches onto the 197 High-Qubyte and 256 Low-Qubyte bases (453 cols vs 50,304!)
        z_hi_s0 = torch.matmul(h_sem_0, d_hi_sem.t()).float().unsqueeze(1)
        z_lo_s0 = torch.matmul(h_sem_0, d_lo_sem.t()).float().unsqueeze(1)

        if h_sem_branches.dim() == 4:
            h_sem_r = h_sem_branches.reshape(n_pos * n_modes, self.sem_dim)
            z_hi_sr = torch.matmul(h_sem_r, d_hi_sem.t()).float().view(n_pos, n_modes, NUM_HIGH_BYTE_BUCKETS)
            z_lo_sr = torch.matmul(h_sem_r, d_lo_sem.t()).float().view(n_pos, n_modes, NUM_LOW_BYTE_BUCKETS)
        else:
            h_sem_1 = h_sem_branches.reshape(n_pos, self.sem_dim)
            z_hi_sr = torch.matmul(h_sem_1, d_hi_sem.t()).float().unsqueeze(1)
            z_lo_sr = torch.matmul(h_sem_1, d_lo_sem.t()).float().unsqueeze(1)

        z_hi_id = torch.matmul(h_id_flat, d_hi_id.t()).float().view(n_pos, n_modes, NUM_HIGH_BYTE_BUCKETS)
        z_lo_id = torch.matmul(h_id_flat, d_lo_id.t()).float().view(n_pos, n_modes, NUM_LOW_BYTE_BUCKETS)

        t_hi = self.qubyte_hi_temp.float().clamp(0.4, 4.0)
        t_lo = self.qubyte_lo_temp.float().clamp(0.4, 4.0)
        u_hi_r = t_hi * (z_hi_s0 + beta * z_hi_sr + tau * z_hi_id)
        u_lo_r = t_lo * (z_lo_s0 + beta * z_lo_sr + tau * z_lo_id)

        # 16-Qubit Bloch-Magnetization & Cross-Register Pauli-Z_i^{hi} Z_j^{lo} Ising Hamiltonian per mode
        if spec.enable_walsh_pauli_qubytes:
            s_hi = self.spins_hi.float()  # (197, 8)
            s_lo = self.spins_lo.float()  # (256, 8)
            m_hi_r = torch.matmul(h_id_flat.float(), self.bloch_field_hi.float()).view(n_pos, n_modes, QUBYTE_NUM_QUBITS)
            m_lo_r = torch.matmul(h_id_flat.float(), self.bloch_field_lo.float()).view(n_pos, n_modes, QUBYTE_NUM_QUBITS)
            ising_sc = self.ising_coupling_scale.float()
            u_hi_r = u_hi_r + ising_sc * torch.matmul(m_hi_r + torch.matmul(m_lo_r, self.ising_coupling_j.float().t()), s_hi.t())
            u_lo_r = u_lo_r + ising_sc * torch.matmul(m_lo_r + torch.matmul(m_hi_r, self.ising_coupling_j.float()), s_lo.t())

        log_p_hi_r = F.log_softmax(u_hi_r + 0.35 * log_p_hi_prior.float().view(1, 1, NUM_HIGH_BYTE_BUCKETS), dim=-1)
        log_p_lo_r = F.log_softmax(u_lo_r, dim=-1)

        p_hi_r = log_p_hi_r.exp()
        p_lo_r = log_p_lo_r.exp()

        # Multi-mode marginal mixture vs anchor mode-0 innovation (Rank-2 outer product in (197, 256))
        p_hi_mix = (pi.unsqueeze(-1) * p_hi_r).sum(dim=1, keepdim=True)
        p_lo_mix = (pi.unsqueeze(-1) * p_lo_r).sum(dim=1, keepdim=True)
        s1_sc = self.step1_innovation_scale.float()
        delta_hi = s1_sc * (torch.log(p_hi_mix.clamp_min(1e-7)) - log_p_hi_r[:, :1, :])
        delta_lo = s1_sc * (torch.log(p_lo_mix.clamp_min(1e-7)) - log_p_lo_r[:, :1, :])

        ones_hi = torch.ones_like(delta_hi)
        ones_lo = torch.ones_like(delta_lo)
        q_hi_list = [delta_hi.to(work_dtype), ones_hi.to(work_dtype)]
        q_lo_list = [ones_lo.to(work_dtype), delta_lo.to(work_dtype)]

        # Rank-R Bipartite Schmidt Entanglement across the Two 8-Qubit Registers: \sum_{r=0}^{R-1} \pi_{t,r} |\alpha_{t,r}\rangle_{hi} \langle\beta_{t,r}|_{lo}
        if spec.enable_schmidt_entanglement:
            inv_temp = 1.0 / float(DEFAULT_SCHMIDT_TEMP)
            sch_sc = self.schmidt_entangle_scale.float()
            alpha_hi = (sch_sc * pi.unsqueeze(-1) * torch.tanh(u_hi_r * inv_temp)).to(work_dtype)
            beta_lo = torch.tanh(u_lo_r * inv_temp).to(work_dtype)
            q_hi_list.append(alpha_hi)
            q_lo_list.append(beta_lo)
        else:
            # Centered multi-mode probability covariance across (b_hi, b_lo) for Step 1
            p_hi_dev = (s1_sc * pi.unsqueeze(-1) * (p_hi_r - p_hi_mix)).to(work_dtype)
            p_lo_dev = (p_lo_r - p_lo_mix).to(work_dtype)
            q_hi_list.append(p_hi_dev)
            q_lo_list.append(p_lo_dev)

        q_hi_factors = torch.cat(q_hi_list, dim=1).view(bsz, seq_len, -1, NUM_HIGH_BYTE_BUCKETS).contiguous()
        q_lo_factors = torch.cat(q_lo_list, dim=1).view(bsz, seq_len, -1, NUM_LOW_BYTE_BUCKETS).contiguous()
        return q_hi_factors, q_lo_factors, log_pi, log_p_hi_r, log_p_lo_r


def compute_qubyte_multimode_aux_loss(
    log_pi: torch.Tensor,
    log_p_hi_r: torch.Tensor,
    log_p_lo_r: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    r"""Compute `0`-matmul multi-mode Two-Qubyte EM responsibility NLL $-\ln \sum_r \pi_{t,r} P^{\text{hi}}_{t,r}(b_{\text{hi}}(y_t)) P^{\text{lo}}_{t,r}(b_{\text{lo}}(y_t))$."""
    y_flat = targets.reshape(-1)
    valid_mask = (y_flat >= 0) & (y_flat < RAW_GPT2_VOCAB_SIZE)
    y_safe = y_flat.clamp(0, RAW_GPT2_VOCAB_SIZE - 1)
    b_hi_y, b_lo_y = decompose_token_uint16_byte_pair(y_safe)

    n_pos, n_modes, _ = log_p_hi_r.shape
    hi_idx = b_hi_y.view(n_pos, 1, 1).expand(n_pos, n_modes, 1)
    lo_idx = b_lo_y.view(n_pos, 1, 1).expand(n_pos, n_modes, 1)

    target_hi_r = log_p_hi_r.gather(dim=-1, index=hi_idx).squeeze(-1)
    target_lo_r = log_p_lo_r.gather(dim=-1, index=lo_idx).squeeze(-1)
    nll_mix = -torch.logsumexp(log_pi + target_hi_r + target_lo_r, dim=-1)
    if bool(valid_mask.all()):
        return nll_mix.mean()
    return nll_mix[valid_mask].mean()


class TwoQubyteRegisterCholeskyGPT(ZeroPPMIBytePairCholeskyGPT):
    r"""Two 8-Qubit Register ("Qubytes") & Single-GEMM Step-1/Step-3 Quantum Cholesky GPT (`EXPERIMENT_PLAN_082.md` / `Wave 62`)."""

    def __init__(self, q_spec: TwoQubyteRegisterCholeskySpec) -> None:
        zp_spec = ZeroPPMIBytePairCholeskySpec(
            variant_mode=ZeroPPMIBytePairVariantMode.Z4_ZERO_PPMI_HIERARCHICAL_BYTE_PAIR_R12
            if q_spec.num_cholesky_modes >= 12
            else ZeroPPMIBytePairVariantMode.Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR,
            num_cholesky_modes=q_spec.num_cholesky_modes,
            vacuum_dim=q_spec.vacuum_dim,
            enable_online_ema=True,
            enable_byte_pair_schur=True,
            use_exact_chain_rule_readout=False,
            enable_hierarchical_bhi_carrier=q_spec.enable_hierarchical_bhi_carrier,
            seed=q_spec.seed,
        )
        super().__init__(zp_spec)
        self.q_spec = q_spec
        self.q_variant_mode = TwoQubyteVariantMode(q_spec.variant_mode)
        self.qubyte_synth = Two8QubitRegisterSynthesizer(
            pkt_count=self.pkt_count,
            pkt_dim=self.pkt_dim,
            sem_dim=self.sem_real_dim,
            id_dim=self.id_real_dim,
        )

    def build_step3_and_qubyte_readout_tensors(
        self, idx: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        """Build Step-3 contracted `(B, T, 528)` state, compact Two-Qubyte `(q_hi_factors, q_lo_factors)`, and `w_aug`."""
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug = self.build_multibranch_readout_tensors(idx)
        work_dtype = h_base_464.dtype

        if self.q_spec.enable_walsh_pauli_qubytes:
            d_q_hi, d_q_lo = self.qubyte_synth.synthesize_qubyte_detector_bases(
                self.byte_synth.d_out_hi, self.byte_synth.d_out_lo, enable_walsh_pauli=True, work_dtype=work_dtype
            )
            b_hi_v = self.byte_synth.vocab_b_hi
            b_lo_v = self.byte_synth.vocab_b_lo
            w_walsh_delta = 0.12 * (d_q_hi[b_hi_v] + d_q_lo[b_lo_v]) * self.radial_scale.to(work_dtype).unsqueeze(1)
            w_aug = torch.cat([w_aug[:, : self.embed_dim] + w_walsh_delta, w_aug[:, self.embed_dim :]], dim=-1).contiguous()
        else:
            d_q_hi, d_q_lo = self.qubyte_synth.synthesize_qubyte_detector_bases(
                self.byte_synth.d_out_hi, self.byte_synth.d_out_lo, enable_walsh_pauli=False, work_dtype=work_dtype
            )

        # Step 3: Contract R Cholesky branches in 512D latent space before the single 50,304 vocabulary GEMM
        h_step3_528 = contract_step3_latent_512d_modes(h_base_464, h_sem_branches, h_id_branches, branch_controls)

        if not self.q_spec.enable_step1_multimode_qubyte:
            return h_step3_528, None, None, w_aug, branch_controls, None

        _, log_p_hi_prior = self.online_zipf.current_log_priors()
        q_hi_factors, q_lo_factors, log_pi, log_p_hi_r, log_p_lo_r = (
            self.qubyte_synth.compute_qubyte_step1_and_schmidt_factors(
                h_base_464,
                h_sem_branches,
                h_id_branches,
                branch_controls,
                d_q_hi,
                d_q_lo,
                log_p_hi_prior,
                self.q_spec,
            )
        )
        return h_step3_528, q_hi_factors, q_lo_factors, w_aug, branch_controls, (log_pi, log_p_hi_r, log_p_lo_r)

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Single-GEMM Step-3 + Step-1 Two-Qubyte Schmidt-Entangled Cross-Entropy loss."""
        h_step3_528, q_hi_factors, q_lo_factors, w_aug, _, aux_tuple = self.build_step3_and_qubyte_readout_tensors(idx)
        main_ce = _SingleGemmStep3QubyteCEFn.apply(h_step3_528, q_hi_factors, q_lo_factors, w_aug, targets)
        if aux_tuple is not None:
            log_pi, log_p_hi_r, log_p_lo_r = aux_tuple
            aux_ce = compute_qubyte_multimode_aux_loss(log_pi, log_p_hi_r, log_p_lo_r, targets)
            return main_ce + float(DEFAULT_QUBYTE_AUX_LOSS_WEIGHT) * aux_ce
        return main_ce


def build_two_qubyte_register_optimizers(
    model: TwoQubyteRegisterCholeskyGPT,
) -> tuple[
    ModularRiemannianStiefelKrausMuon,
    BatchSubspaceTikhonovGramMuon,
    BatchSubspaceTikhonovGramMuon,
    torch.optim.AdamW,
]:
    """Build Riemannian Stiefel, Vocab Gram-Muon, and AdamW optimizers with Two-Qubyte Pauli/Ising groups (`<= 8` args)."""
    q_trunk_opt, wte_opt, head_opt, adam_opt = build_zero_ppmi_byte_pair_optimizers(model)
    adam_opt.add_param_group(
        {"params": model.qubyte_synth.table_parameters(), "lr": DEFAULT_SHALLOW_NGRAM_LR, "weight_decay": 0.0}
    )
    adam_opt.add_param_group(
        {"params": model.qubyte_synth.scalar_parameters(), "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0}
    )
    return q_trunk_opt, wte_opt, head_opt, adam_opt


@torch.no_grad()
def initialize_two_qubyte_register_model_(model: TwoQubyteRegisterCholeskyGPT) -> tuple[float, dict[str, Any]]:
    """Initialize `TwoQubyteRegisterCholeskyGPT` at Step 0 with **zero training tokens scanned** (`< 50 ms`)."""
    device = model.wte_3d.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    _, meta = initialize_zero_ppmi_byte_pair_model_(model)
    model.qubyte_synth.reset_parameters_(seed=model.q_spec.seed)

    if model.q_spec.enable_walsh_pauli_qubytes:
        u_hi_w, u_lo_w = model.qubyte_synth.build_walsh_pauli_input_tables(torch.float32)
        b_hi_v = model.byte_synth.vocab_b_hi
        b_lo_v = model.byte_synth.vocab_b_lo
        blended = F.normalize(model.wte_3d.float() + 0.08 * u_hi_w[b_hi_v] + 0.08 * u_lo_w[b_lo_v], dim=-1, eps=1e-6)
        model.wte_3d.copy_(blended)

    if device.type == "cuda":
        torch.cuda.synchronize()
    init_ms = (time.perf_counter() - t0) * 1000.0
    meta["init_ms"] = round(init_ms, 3)
    meta["qubyte_num_qubits_per_register"] = QUBYTE_NUM_QUBITS
    return init_ms, meta


@torch.no_grad()
def evaluate_two_qubyte_register_model(
    model: TwoQubyteRegisterCholeskyGPT, stream: LongSeqFineWebStream
) -> dict[str, float]:
    """Evaluate `TwoQubyteRegisterCholeskyGPT` on `10,485,760` Shard-0 validation tokens using **1 GEMM/chunk** (`<= 8` args)."""
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_top1 = 0
    total_top5 = 0
    total_tokens = 0
    gate_ent_sum = 0.0
    n_batches = 0
    anc_cos_sum, exc_cos_sum, core_cos_sum, high_cos_sum = 0.0, 0.0, 0.0, 0.0

    cap = float(DEFAULT_SOFTCAP_LOGIT)
    inv_cap = 1.0 / cap
    n_qubyte_cells = NUM_HIGH_BYTE_BUCKETS * NUM_LOW_BYTE_BUCKETS

    for idx_in, targets in stream.val_batches:
        bsz, seq_len = idx_in.shape
        n_pos = bsz * seq_len
        h_step3_528, q_hi_factors, q_lo_factors, w_aug, branch_controls, _ = (
            model.build_step3_and_qubyte_readout_tensors(idx_in)
        )

        gate_logits = branch_controls[:, :, :, 2].float()
        pi = F.softmax(gate_logits, dim=-1)
        log_pi = F.log_softmax(gate_logits, dim=-1)
        gate_ent_sum += float(-(pi * log_pi).sum(dim=-1).mean().item())
        anc_cos_sum += float(getattr(model, "last_anchor_id_cos_vac", 1.0))
        exc_cos_sum += float(getattr(model, "last_excited_id_cos_vac", 1.0))
        core_cos_sum += float(getattr(model, "last_core_excited_cos_vac", 1.0))
        high_cos_sum += float(getattr(model, "last_high_excited_cos_vac", 1.0))
        n_batches += 1

        h_flat = h_step3_528.reshape(n_pos, -1)
        if q_hi_factors is not None and q_lo_factors is not None:
            k_rank = q_hi_factors.size(2)
            q_hi_flat = q_hi_factors.reshape(n_pos, k_rank, NUM_HIGH_BYTE_BUCKETS)
            q_lo_flat = q_lo_factors.reshape(n_pos, k_rank, NUM_LOW_BYTE_BUCKETS)
        else:
            q_hi_flat = None
            q_lo_flat = None
        y_flat = targets.reshape(n_pos)
        w_t = w_aug[:RAW_GPT2_VOCAB_SIZE].t().contiguous()

        for start in range(0, n_pos, SINGLE_GEMM_TOKEN_CHUNK):
            end = min(start + SINGLE_GEMM_TOKEN_CHUNK, n_pos)
            c_len = end - start
            y_c = y_flat[start:end]

            z_c = torch.matmul(h_flat[start:end], w_t).float()
            if q_hi_flat is not None and q_lo_flat is not None:
                q_grid_c = torch.bmm(q_hi_flat[start:end].transpose(1, 2), q_lo_flat[start:end]).float()
                z_c.add_(q_grid_c.reshape(c_len, n_qubyte_cells)[:, :RAW_GPT2_VOCAB_SIZE])
                del q_grid_c
            z_c.mul_(inv_cap).tanh_().mul_(cap)

            log_z = torch.logsumexp(z_c, dim=-1)
            target_z = z_c.gather(dim=-1, index=y_c.unsqueeze(-1)).squeeze(-1)
            total_nll += float((log_z - target_z).sum().item())

            top5_idx = torch.topk(z_c, k=5, dim=-1).indices
            total_top1 += int((top5_idx[:, 0] == y_c).sum().item())
            total_top5 += int((top5_idx == y_c.unsqueeze(-1)).any(dim=-1).sum().item())
            total_tokens += c_len
            del z_c

        del h_step3_528, q_hi_factors, q_lo_factors, w_aug, branch_controls

    if stream.device.type == "cuda":
        torch.cuda.empty_cache()
    if was_training:
        model.train()

    denom = float(max(1, total_tokens))
    nb = float(max(1, n_batches))
    val_ce = total_nll / denom
    mean_ent = gate_ent_sum / nb
    ev = {
        "val_ce": round(val_ce, 4),
        "neural_val_ce": round(val_ce, 4),
        "val_ppl": round(math.exp(min(20.0, val_ce)), 2),
        "top1_acc": round(total_top1 / denom, 4),
        "top5_acc": round(total_top5 / denom, 4),
        "branch_gate_entropy_nats": round(mean_ent, 4),
        "effective_active_branches": round(math.exp(mean_ent), 2),
        "mean_anchor_id_cos_theta_vac": round(anc_cos_sum / nb, 4),
        "mean_excited_id_cos_theta_vac": round(exc_cos_sum / nb, 4),
        "mean_core_excited_cos_theta_vac": round(core_cos_sum / nb, 4),
        "mean_high_excited_cos_theta_vac": round(high_cos_sum / nb, 4),
        "online_batches_seen": float(model.online_zipf.batches_seen.item()),
    }
    ev.update(model.get_noclick_telemetry())
    return ev


def _train_and_eval_two_qubyte_variants(
    stream: LongSeqFineWebStream,
    variants: list[tuple[str, TwoQubyteRegisterCholeskySpec, str]],
    steps: int,
    eval_interval: int,
    wave_id: str = "W62",
) -> list[dict[str, Any]]:
    """Train and evaluate Two 8-Qubit Register & Step-1/Step-3 Cholesky variants (`<= 8` args)."""
    device = stream.device
    results: list[dict[str, Any]] = []
    tokens_per_step = stream.batch_size * stream.seq_len

    for variant_id, q_spec, desc in variants:
        torch.manual_seed(q_spec.seed)
        model = TwoQubyteRegisterCholeskyGPT(q_spec).to(device=device)
        init_ms, init_meta = initialize_two_qubyte_register_model_(model)

        run_key = f"plan082_{variant_id.lower()}_{steps}steps"
        intent = ExperimentIntentSpec(
            run_key=run_key,
            wave_id=wave_id,
            plan_id="EXPERIMENT_PLAN_082",
            task_domain="fineweb_speedrun",
            hypothesis_intent=desc,
            architecture_summary=(
                f"TwoQubyteRegisterCholeskyGPT(L=4, d=512, R={q_spec.num_cholesky_modes}, "
                f"step1={q_spec.enable_step1_multimode_qubyte}, walsh={q_spec.enable_walsh_pauli_qubytes}, "
                f"schmidt={q_spec.enable_schmidt_entanglement}, single_gemm=True)"
            ),
            hardware_budget="1x RTX 5060 Ti 16GB",
            batch_size=stream.batch_size,
            seq_len=stream.seq_len,
            total_steps=steps,
            target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_two_qubyte_register_optimizers(model)

        step0_eval = evaluate_two_qubyte_register_model(model, stream)
        step0_eval["step"], step0_eval["tokens_M"] = 0, 0.0
        exp_logger.log_eval(
            0, {**step0_eval, "top1_acc": round(100.0 * step0_eval["top1_acc"], 2), "top5_acc": round(100.0 * step0_eval["top5_acc"], 2)}
        )
        print(
            f"[{variant_id} (R={q_spec.num_cholesky_modes}, step1={q_spec.enable_step1_multimode_qubyte}, "
            f"walsh={q_spec.enable_walsh_pauli_qubytes}, schmidt={q_spec.enable_schmidt_entanglement}, init={init_ms:.1f}ms)] "
            f"Step    0 | Two-Qubyte Val CE: {step0_eval['val_ce']:.4f} (PPL {step0_eval['val_ppl']:.2f}) | "
            f"Top1: {100.0 * step0_eval['top1_acc']:.2f}% | Modes: {step0_eval['effective_active_branches']:.2f}/{q_spec.num_cholesky_modes}",
            flush=True,
        )

        curve: list[dict[str, Any]] = [step0_eval]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        model.train()

        for step in range(1, steps + 1):
            idx_in, targets = stream.sample_train_batch(step)
            lr_mult, mom_beta = _compute_zero_ppmi_step_lr_and_momentum(step, steps)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            loss = model.forward_loss(idx_in, targets)
            train_loss_val = float(loss.detach().item())
            loss.backward()
            del loss
            _apply_global_optimizer_step(model, optimizers, idx_in, targets, lr_mult=lr_mult, mom_beta=mom_beta)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_two_qubyte_register_model(model, stream)
                ev["step"] = step
                ev["tokens_M"] = round((step * tokens_per_step) / 1e6, 3)
                ev["train_loss"] = round(train_loss_val, 4)
                ev["lr_mult"] = round(lr_mult, 4)
                curve.append(ev)
                exp_logger.log_eval(
                    step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)}
                )
                print(
                    f"[{variant_id} (R={q_spec.num_cholesky_modes}, step1={q_spec.enable_step1_multimode_qubyte}, "
                    f"schmidt={q_spec.enable_schmidt_entanglement})] "
                    f"Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={lr_mult:.2f}x) | "
                    f"Train CE: {ev['train_loss']:.4f} | Two-Qubyte Val CE: {ev['val_ce']:.4f} (PPL {ev['val_ppl']:.2f}) | "
                    f"Top1: {100.0 * ev['top1_acc']:.2f}% | Top5: {100.0 * ev['top5_acc']:.2f}% | "
                    f"Modes: {ev['effective_active_branches']:.2f}/{q_spec.num_cholesky_modes} | "
                    f"p_noclick: {ev['mean_noclick_prob']:.3f}",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t_start
        results.append({
            "variant_id": variant_id,
            "variant_mode": q_spec.variant_mode.value,
            "description": desc,
            "num_cholesky_modes": q_spec.num_cholesky_modes,
            "enable_step1_multimode_qubyte": q_spec.enable_step1_multimode_qubyte,
            "enable_walsh_pauli_qubytes": q_spec.enable_walsh_pauli_qubytes,
            "enable_schmidt_entanglement": q_spec.enable_schmidt_entanglement,
            "init_tokens_scanned": init_meta["init_tokens_scanned"],
            "init_ms": round(init_ms, 2),
            "elapsed_sec": round(elapsed_s, 2),
            "step0": step0_eval,
            "final": curve[-1],
            "curve": curve,
        })
        del model, optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_two_qubyte_register_ablation_benchmark(
    stream: LongSeqFineWebStream, steps: int = 300, eval_interval: int = 100
) -> list[dict[str, Any]]:
    """Run the 300-step (`2.46M`-token) Two 8-Qubit Register & Step-1/Step-3 ablation suite (`Q0..Q4`, `<= 8` args)."""
    variants = [
        (
            "Q0_STEP3_ONLY_LATENT_512D_CONTRACTION",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q0_STEP3_ONLY_LATENT_512D_CONTRACTION,
                num_cholesky_modes=8,
                enable_step1_multimode_qubyte=False,
                enable_walsh_pauli_qubytes=False,
                enable_schmidt_entanglement=False,
                enable_hierarchical_bhi_carrier=False,
            ),
            "R=8 Step-3 Only: Contract R Cholesky Modes in 512D Latent Space Before Single 50,304 Vocab GEMM",
        ),
        (
            "Q1_STEP1_PLUS_STEP3_FACTORED_MULTIMODE",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q1_STEP1_PLUS_STEP3_FACTORED_MULTIMODE,
                num_cholesky_modes=8,
                enable_step1_multimode_qubyte=True,
                enable_walsh_pauli_qubytes=False,
                enable_schmidt_entanglement=False,
                enable_hierarchical_bhi_carrier=False,
            ),
            "R=8 Step 1 + Step 3: Multi-Mode Cholesky on (197 + 256) Byte-Pair Basis (111x Fewer FLOPs) + Single-GEMM Residual",
        ),
        (
            "Q2_TWO_8QUBIT_REGISTERS_WALSH_PAULI_ISING",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q2_TWO_8QUBIT_REGISTERS_WALSH_PAULI_ISING,
                num_cholesky_modes=8,
                enable_step1_multimode_qubyte=True,
                enable_walsh_pauli_qubytes=True,
                enable_schmidt_entanglement=False,
                enable_hierarchical_bhi_carrier=True,
            ),
            "R=8 Two 8-Qubit Registers (Qubytes) with Exact 8-Qubit Walsh-Hadamard Pauli-Z Basis & 16-Qubit Bloch-Ising Hamiltonian",
        ),
        (
            "Q3_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R8",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q3_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R8,
                num_cholesky_modes=8,
                enable_step1_multimode_qubyte=True,
                enable_walsh_pauli_qubytes=True,
                enable_schmidt_entanglement=True,
                enable_hierarchical_bhi_carrier=True,
            ),
            "R=8 Two 8-Qubit Registers + Rank-R Bipartite Schmidt Entanglement + Step 1 Multi-Mode + Step 3 Single GEMM",
        ),
        (
            "Q4_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R12",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q4_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R12,
                num_cholesky_modes=12,
                enable_step1_multimode_qubyte=True,
                enable_walsh_pauli_qubytes=True,
                enable_schmidt_entanglement=True,
                enable_hierarchical_bhi_carrier=True,
            ),
            "R=12 Two 8-Qubit Registers + Rank-12 Bipartite Schmidt Entanglement + Step 1 Multi-Mode + Step 3 Single GEMM",
        ),
    ]
    return _train_and_eval_two_qubyte_variants(stream, variants, steps=steps, eval_interval=eval_interval)


def run_two_qubyte_register_scaling_benchmark(
    stream: LongSeqFineWebStream, steps: int = 1200, eval_interval: int = 150
) -> list[dict[str, Any]]:
    """Run the 1,200-step (`9.83M`-token) Two 8-Qubit Register Schmidt-Entangled Scaling Run (`Q5`, `<= 8` args)."""
    variants = [
        (
            "Q5_TWO_8QUBIT_SCHMIDT_SCALING_1200",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q5_TWO_8QUBIT_SCHMIDT_SCALING_1200,
                num_cholesky_modes=12,
                enable_step1_multimode_qubyte=True,
                enable_walsh_pauli_qubytes=True,
                enable_schmidt_entanglement=True,
                enable_hierarchical_bhi_carrier=True,
            ),
            "1,200-Step (9.83M-Token) Two 8-Qubit Registers + R=12 Schmidt Entanglement + Step-1/Step-3 Single-GEMM Scaling",
        ),
    ]
    return _train_and_eval_two_qubyte_variants(stream, variants, steps=steps, eval_interval=eval_interval)
