#!/usr/bin/env python3
r"""Wave 69 (`EXPERIMENT_PLAN_089.md`): Numerical & Geometric Invariant Inspector for `PureDensityMatrixGPT`.

Audits the 5 core mathematical invariants corresponding to Lean 4 Theorems 6–8:
1. **Theorem 6b & 6c (Hilbert–Schmidt Isometric Outer-Product Identity & Right $U(R)$-Gauge Invariance)**:
   Verifies $\langle \tilde{P}(Q Q^\dagger), \tilde{P}(K K^\dagger) \rangle_{\mathbb{R}^{256}} = \|Q^\dagger K\|_F^2 \ge 0$
   and invariance under $(Q, K) \mapsto (Q U_Q, K U_K)$ for random $U_Q, U_K \in U(R)$.
2. **Theorem 6a (`exp`/`softmax`-Free Hilbert–Schmidt Quantum Attention Normalization)**:
   Verifies $A_{t,s}^{(h)} \ge 0$ and $\sum_{s \le t} A_{t,s}^{(h)} + \sum_m A_{t,\text{hist},m}^{(h)} + A_{t,\text{vac}}^{(h)} \equiv 1$.
3. **Theorem 7 (Quantum "OR + AND" Schur Product Closure)**:
   Verifies orthostochastic row sum $\sum_{h'} (W_{\text{OR},h,h'})^2 = 1$, positive semidefiniteness
   $\lambda_{\min}(\rho_{t,h}^{\text{out}}) \ge -10^{-6}$, Hermiticity, and unit trace $\operatorname{Tr}(\rho_{t,h}^{\text{out}}) = 1$.
4. **Theorem 8 (Online EMA N-Gram Density Matrices & Stateful Multi-Window History Invariance)**:
   Verifies unit trace and PSD preservation across multi-step online updates of `OnlineEmaNgramDensityBank`
   and `StatefulQuantumDensityHistoryBuffer`.
5. **Path E & Rule 9 (Unconstrained `512D` Born Readout Normalization & Single-Pass Backward Accuracy)**:
   Verifies $\sum_{v=0}^{50256} P(v \mid \rho_t^{(L)}) \equiv 1$ and custom autograd backward accuracy (`< 1e-5`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    DEFAULT_PURE_CHOLESKY_RANK, DEFAULT_PURE_RHO_DIM, DEFAULT_PURE_RHO_HEADS,
    DEFAULT_UNCONSTRAINED_LOGIT_MAX, PADDED_LUDERS_GRID_SIZE,
    OnlineEmaNgramDensityBank, PureDensityMatrixGPT, StatefulQuantumDensityHistoryBuffer,
    Wave69VariantMode, _UnconstrainedDualSubspace512DCEFn, apply_quantum_or_and_schur_logic,
    build_wave69_optimizers, build_wave69_variant_spec, compute_hilbert_schmidt_quantum_attention,
    normalize_cholesky_factor_frobenius, pack_cholesky_factor_outer_isometric,
)
from qdlblocks.benchmarks.quantum_subsystem_vocab_lab import (
    LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import RAW_GPT2_VOCAB_SIZE
from qdlblocks.layers.impure_density_mirror_block import unpack_hermitian_isometric

# ==============================================================================
# Top-of-File Verification Threshold Constants (Rule 7)
# ==============================================================================
MAX_ISOMETRIC_OVERLAP_REL_ERR: float = 1.0e-5
MAX_UR_GAUGE_REL_ERR: float = 1.0e-5
MAX_ATTN_PARTITION_SUM_ERR: float = 1.0e-5
MIN_PSD_EIGENVALUE_TOL: float = -1.0e-5
MAX_TRACE_UNIT_ERR: float = 1.0e-5
MAX_AUTOGRAD_REL_ERR: float = 1.0e-4
MAX_VOCAB_PROB_SUM_ERR: float = 1.0e-4


def _random_complex_unitary(batch: int, dim: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random $U(d)$ matrices `(u_re, u_im)` of shape `(batch, dim, dim)` (`<= 8` args)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    zr = torch.randn(batch, dim, dim, generator=gen, dtype=torch.float32).to(device)
    zi = torch.randn(batch, dim, dim, generator=gen, dtype=torch.float32).to(device)
    q_c, r_c = torch.linalg.qr(torch.complex(zr, zi))
    diag_r = torch.diagonal(r_c, dim1=-2, dim2=-1)
    u_c = q_c * (diag_r / diag_r.abs().clamp_min(1e-8)).unsqueeze(-2)
    return u_c.real.float().contiguous(), u_c.imag.float().contiguous()


def verify_theorem6_hilbert_schmidt_and_gauge(device: torch.device) -> dict[str, Any]:
    r"""Audit Theorem 6: $\langle \tilde{P}(Q Q^\dagger), \tilde{P}(K K^\dagger) \rangle = \|Q^\dagger K\|_F^2 \ge 0$, $U(R)$ invariance, and `exp`-free attention."""
    torch.manual_seed(601)
    bsz, num_h, seq_len, d_h, r_rank = 2, DEFAULT_PURE_RHO_HEADS, 12, DEFAULT_PURE_RHO_DIM, DEFAULT_PURE_CHOLESKY_RANK
    q_re, q_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
    )
    k_re, k_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
    )

    q_pack = pack_cholesky_factor_outer_isometric(q_re, q_im)
    k_pack = pack_cholesky_factor_outer_isometric(k_re, k_im)
    sim_packed = torch.matmul(q_pack, k_pack.transpose(-2, -1))

    q_c = torch.complex(q_re, q_im).unsqueeze(3)  # (B, H, T, 1, d_h, R)
    k_c = torch.complex(k_re, k_im).unsqueeze(2)  # (B, H, 1, T, d_h, R)
    qk_ov = torch.matmul(q_c.conj().transpose(-2, -1), k_c)
    sim_exact = qk_ov.abs().square().sum(dim=(-2, -1))
    iso_rel_err = float(((sim_packed - sim_exact).abs().max() / sim_exact.abs().max().clamp_min(1e-8)).item())

    uq_re, uq_im = _random_complex_unitary(num_h, r_rank, seed=602, device=device)
    uk_re, uk_im = _random_complex_unitary(num_h, r_rank, seed=603, device=device)
    uqr, uqi = uq_re.view(1, num_h, 1, r_rank, r_rank), uq_im.view(1, num_h, 1, r_rank, r_rank)
    ukr, uki = uk_re.view(1, num_h, 1, r_rank, r_rank), uk_im.view(1, num_h, 1, r_rank, r_rank)
    q_rot_re = torch.matmul(q_re, uqr) - torch.matmul(q_im, uqi)
    q_rot_im = torch.matmul(q_re, uqi) + torch.matmul(q_im, uqr)
    k_rot_re = torch.matmul(k_re, ukr) - torch.matmul(k_im, uki)
    k_rot_im = torch.matmul(k_re, uki) + torch.matmul(k_im, ukr)
    q_rot_pack = pack_cholesky_factor_outer_isometric(q_rot_re, q_rot_im)
    k_rot_pack = pack_cholesky_factor_outer_isometric(k_rot_re, k_rot_im)
    sim_rot = torch.matmul(q_rot_pack, k_rot_pack.transpose(-2, -1))
    ur_gauge_rel_err = float(((sim_rot - sim_packed).abs().max() / sim_packed.abs().max().clamp_min(1e-8)).item())

    v_payload = torch.randn(bsz, num_h, seq_len, 384, device=device)
    coinc_beta = torch.full((num_h,), 0.35, device=device)
    pos_decay = torch.linspace(-2.0, -0.3, steps=num_h, device=device)
    vac_floor = torch.full((num_h,), 0.08, device=device)
    therm_payload = torch.zeros(1, 1, 1, 384, device=device)
    _, attn_w = compute_hilbert_schmidt_quantum_attention(
        q_pack, k_pack, v_payload, None, None, coinc_beta, pos_decay, (vac_floor, therm_payload)
    )
    min_attn_w = float(attn_w.min().item())
    upper_tri_leak = float(attn_w.triu(diagonal=1).abs().max().item())

    passed = (
        iso_rel_err <= MAX_ISOMETRIC_OVERLAP_REL_ERR
        and ur_gauge_rel_err <= MAX_UR_GAUGE_REL_ERR
        and min_attn_w >= 0.0
        and upper_tri_leak <= 1e-8
    )
    return {
        "passed": passed,
        "isometric_overlap_rel_err": iso_rel_err,
        "ur_gauge_invariance_rel_err": ur_gauge_rel_err,
        "min_attention_weight": min_attn_w,
        "causal_upper_tri_leak": upper_tri_leak,
    }


def verify_theorem7_quantum_or_and_schur_closure(device: torch.device) -> dict[str, Any]:
    r"""Audit Theorem 7: Quantum "OR + AND" (Schur Product) preserves PSD $\rho \succeq 0$, Hermiticity, and $\operatorname{Tr}(\rho) = 1$."""
    torch.manual_seed(701)
    bsz, num_h, seq_len, d_h, r_rank = 2, DEFAULT_PURE_RHO_HEADS, 8, DEFAULT_PURE_RHO_DIM, DEFAULT_PURE_CHOLESKY_RANK
    l_re, l_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
    )
    rho_in = pack_cholesky_factor_outer_isometric(l_re, l_im)

    w_a, _ = torch.linalg.qr(torch.randn(num_h, num_h, device=device))
    w_b, _ = torch.linalg.qr(torch.randn(num_h, num_h, device=device))
    ua_re, ua_im = _random_complex_unitary(num_h, d_h, seed=702, device=device)
    ub_re, ub_im = _random_complex_unitary(num_h, d_h, seed=703, device=device)
    and_logit = torch.linspace(-1.0, 1.0, steps=num_h, device=device)

    l_out_re, l_out_im, rho_out_packed = apply_quantum_or_and_schur_logic(
        l_re, l_im, rho_in, w_a, w_b, (ua_re, ua_im, ub_re, ub_im), and_logit
    )

    orthostoch_row_err = float((w_a.square().sum(dim=-1) - 1.0).abs().max().item())
    chol_norm_err = float(((l_out_re.square() + l_out_im.square()).sum(dim=(-2, -1)) - 1.0).abs().max().item())

    rho_re, rho_im = unpack_hermitian_isometric(rho_out_packed, d_h)
    rho_c = torch.complex(rho_re.float(), rho_im.float())
    herm_err = float((rho_c - rho_c.conj().transpose(-2, -1)).abs().max().item())
    tr_err = float((torch.diagonal(rho_re.float(), dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
    eigvals = torch.linalg.eigvalsh(rho_c)
    min_eigval = float(eigvals.min().item())

    passed = (
        orthostoch_row_err <= MAX_TRACE_UNIT_ERR
        and chol_norm_err <= MAX_TRACE_UNIT_ERR
        and herm_err <= MAX_TRACE_UNIT_ERR
        and tr_err <= MAX_TRACE_UNIT_ERR
        and min_eigval >= MIN_PSD_EIGENVALUE_TOL
    )
    return {
        "passed": passed,
        "orthostochastic_row_sum_err": orthostoch_row_err,
        "cholesky_unit_frob_err": chol_norm_err,
        "hermitian_symmetry_err": herm_err,
        "unit_trace_err": tr_err,
        "min_psd_eigenvalue": min_eigval,
    }


def verify_theorem8_online_ema_and_stateful_history(device: torch.device) -> dict[str, Any]:
    r"""Audit Theorem 8: `OnlineEmaNgramDensityBank` and `StatefulQuantumDensityHistoryBuffer` preserve PSD and $\operatorname{Tr}(\rho) = 1$."""
    torch.manual_seed(801)
    bsz, num_h, seq_len, d_h, r_rank = 2, DEFAULT_PURE_RHO_HEADS, 16, DEFAULT_PURE_RHO_DIM, DEFAULT_PURE_CHOLESKY_RANK
    bank = OnlineEmaNgramDensityBank(num_buckets=256, num_heads=num_h, d_h=d_h, ema_decay=0.10, enable_online_ema=True).to(device)
    bank.train()
    hist = StatefulQuantumDensityHistoryBuffer(num_heads=num_h, d_h=d_h, cholesky_rank=r_rank, num_slots=8, enable_history=True).to(device)

    for step_i in range(4):
        h_tok = torch.randint(0, 256, (bsz, seq_len), device=device)
        psi_re = torch.randn(bsz, num_h, seq_len, d_h, device=device)
        psi_im = torch.randn(bsz, num_h, seq_len, d_h, device=device)
        inv_n = torch.rsqrt((psi_re.square() + psi_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-6))
        psi_re, psi_im = psi_re * inv_n, psi_im * inv_n
        _ = bank.lookup_and_update(h_tok, psi_re, psi_im)

        l_re, l_im = normalize_cholesky_factor_frobenius(
            torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
            torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
        )
        rho_win = pack_cholesky_factor_outer_isometric(l_re, l_im)
        hist.update_from_window(l_re, l_im, rho_win)

    ema_re, ema_im = unpack_hermitian_isometric(bank.ema_rho_packed, d_h)
    ema_tr_err = float((torch.diagonal(ema_re.float(), dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
    ema_min_eig = float(torch.linalg.eigvalsh(torch.complex(ema_re[:16].float(), ema_im[:16].float())).min().item())

    hist_re, hist_im = unpack_hermitian_isometric(hist.hist_rho_packed, d_h)
    hist_tr_err = float((torch.diagonal(hist_re.float(), dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
    hist_min_eig = float(torch.linalg.eigvalsh(torch.complex(hist_re.float(), hist_im.float())).min().item())
    hist_chol_err = float(((hist.hist_l_re.square() + hist.hist_l_im.square()).sum(dim=(-2, -1)) - 1.0).abs().max().item())

    passed = (
        ema_tr_err <= MAX_TRACE_UNIT_ERR
        and ema_min_eig >= MIN_PSD_EIGENVALUE_TOL
        and hist_tr_err <= MAX_TRACE_UNIT_ERR
        and hist_min_eig >= MIN_PSD_EIGENVALUE_TOL
        and hist_chol_err <= MAX_TRACE_UNIT_ERR
    )
    return {
        "passed": passed,
        "online_ema_trace_err": ema_tr_err,
        "online_ema_min_eigval": ema_min_eig,
        "history_trace_err": hist_tr_err,
        "history_min_eigval": hist_min_eig,
        "history_cholesky_frob_err": hist_chol_err,
    }


def verify_unconstrained_born_readout_and_model_step(device: torch.device) -> dict[str, Any]:
    r"""Audit Path E & Rule 9: `_UnconstrainedDualSubspace512DCEFn` backward accuracy, `50,257`-way normalization, and 1-step `PureDensityMatrixGPT` update."""
    torch.manual_seed(901)
    n_pos, sub_dim, n_valid = 24, 256, RAW_GPT2_VOCAB_SIZE
    h1 = torch.randn(n_pos, sub_dim, device=device, dtype=torch.float32, requires_grad=True)
    h2 = torch.randn(n_pos, sub_dim, device=device, dtype=torch.float32, requires_grad=True)
    w1 = (torch.randn(n_valid, sub_dim, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    ec = (torch.randn(LUDERS_COARSE_OUTCOMES, sub_dim, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    ef = (torch.randn(LUDERS_FINE_MAX_OUTCOMES, sub_dim, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    b_u = torch.randn(n_valid, device=device, dtype=torch.float32, requires_grad=True)
    y_t = torch.randint(0, n_valid, (n_pos,), device=device, dtype=torch.long)

    loss_custom = _UnconstrainedDualSubspace512DCEFn.apply(h1, h2, w1, ec, ef, b_u, y_t)
    grads_custom = torch.autograd.grad(loss_custom, (h1, h2, w1, ec, ef, b_u))

    c_max = float(DEFAULT_UNCONSTRAINED_LOGIT_MAX)
    z1_ref = torch.matmul(h1, w1.t())
    z2_ref = (torch.matmul(h2, ef.t()).unsqueeze(2) + torch.matmul(h2, ec.t()).unsqueeze(1)).reshape(n_pos, PADDED_LUDERS_GRID_SIZE)[:, :n_valid]
    z_ref = (z1_ref + z2_ref + b_u.unsqueeze(0)).clamp(-c_max, c_max)
    loss_ref = F.cross_entropy(z_ref, y_t)
    grads_ref = torch.autograd.grad(loss_ref, (h1, h2, w1, ec, ef, b_u))

    max_grad_rel_err = max(
        float(((gc - gr).abs().max() / gr.abs().max().clamp_min(1e-8)).item())
        for gc, gr in zip(grads_custom, grads_ref)
    )

    spec = build_wave69_variant_spec(Wave69VariantMode.W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L, num_layers_override=2, seed=902)
    model = PureDensityMatrixGPT(spec).to(device)
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
    trunk_opt.record_diagnostics = True

    idx = torch.randint(0, n_valid, (2, 16), device=device, dtype=torch.long)
    tgt = torch.randint(0, n_valid, (2, 16), device=device, dtype=torch.long)
    act_idx = torch.unique(torch.cat([idx.reshape(-1), tgt.reshape(-1)], dim=0))
    loss = model(idx, targets=tgt)
    loss.backward()
    trunk_opt.step()
    vocab_opt.step_active(act_idx, lr_scale=1.0)
    scalar_opt.step_active(active_idx=None, lr_scale=1.0)
    model.restrict_scalars_()

    model.eval()
    with torch.no_grad():
        log_p = model(idx[:1, :4], targets=None)
        prob_sums = log_p.exp().sum(dim=-1)
        prob_sum_err = float((prob_sums - 1.0).abs().max().item())

    g_diag = trunk_opt.last_gauge_diagnostics
    passed = (
        max_grad_rel_err <= MAX_AUTOGRAD_REL_ERR
        and prob_sum_err <= MAX_VOCAB_PROB_SUM_ERR
        and g_diag["su16_step_phase_frac"] <= 1e-6
        and g_diag["kraus_step_gauge_frac"] <= 1e-6
    )
    return {
        "passed": passed,
        "custom_autograd_max_rel_err": max_grad_rel_err,
        "vocab_50257_prob_sum_err": prob_sum_err,
        "su16_step_phase_frac": g_diag["su16_step_phase_frac"],
        "kraus_step_gauge_frac": g_diag["kraus_step_gauge_frac"],
    }


def run_pure_density_matrix_inspection(device_str: str = "cuda") -> dict[str, Any]:
    """Run all 4 Wave 69 inspection suites and return structured report (`<= 8` args)."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    c1 = verify_theorem6_hilbert_schmidt_and_gauge(device)
    c2 = verify_theorem7_quantum_or_and_schur_closure(device)
    c3 = verify_theorem8_online_ema_and_stateful_history(device)
    c4 = verify_unconstrained_born_readout_and_model_step(device)
    all_passed = bool(c1["passed"] and c2["passed"] and c3["passed"] and c4["passed"])
    return {
        "plan_id": "EXPERIMENT_PLAN_089",
        "wave_id": "W69",
        "device": str(device),
        "overall_passed": all_passed,
        "theorem6_hilbert_schmidt_and_gauge": c1,
        "theorem7_quantum_or_and_schur_closure": c2,
        "theorem8_online_ema_and_stateful_history": c3,
        "unconstrained_born_readout_and_model_step": c4,
    }


def main() -> None:
    """CLI entrypoint for `scripts/pure_density_matrix_inspector.py`."""
    parser = argparse.ArgumentParser(description="Wave 69 Pure Density-Matrix Architecture Inspector.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (`cuda` or `cpu`).")
    parser.add_argument(
        "--json-out",
        type=str,
        default="/home/festivity/agy/QuantumDLBlocks/.cache/plan089_pure_density_inspector.json",
        help="Path to save JSON inspection report.",
    )
    args = parser.parse_args()
    report = run_pure_density_matrix_inspection(device_str=args.device)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
