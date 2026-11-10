"""Wave 68 (`EXPERIMENT_PLAN_088.md`): Complementary `512D` ($256 \\oplus 256$), Lindblad Highway, & Parallel Transport Inspector.

Audits the 5 mathematical invariants of Wave 68:
1. `100.0%` Parseval `512D` norm preservation across $Q = [U^{(1)}, U^{(2)}] \\in \\operatorname{O}(512)$:
   $\\|h U^{(1)}\\|_2^2 + \\|h U^{(2)}\\|_2^2 \\equiv \\|h\\|_2^2$.
2. Exact `50,257`-way probability normalization $\\sum_{v=0}^{50256} \\exp(\\log P(v \\mid \\rho_t)) \\equiv 1.000000$
   for both `ComplementaryOrthogonalLudersBoundary` (`W68_L1`) and `ComplementaryDualSubspace512DBoundary` (`W68_L2`, `W68_L3`).
3. Exact custom-autograd gradient agreement between `_DualSubspace512DSoftcapCEFn` and a PyTorch reference implementation.
4. Exact density-matrix unit trace $\\operatorname{Tr}(\\rho_{t,h}^{(\\ell)}) = 1.000000$ and PSD eigenvalues $\\lambda_{\\min}(\\rho_{t,h}^{(\\ell)}) \\ge 0$
   under `_LindbladKrausDensityHighwayBlock`.
5. Exact norm conservation $\\|\\mathcal{T}_{W_t \\to W_{t+1}}(M_t)\\|_F / \\|M_t\\|_F = 1.000000$ and tangency under
   `parallel_transport_real_stiefel_momentum` and `parallel_transport_complex_quotient_momentum`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import (
    ComplementaryDualSubspace512DBoundary,
    ComplementaryOrthogonalLudersBoundary,
    _DualSubspace512DSoftcapCEFn,
    _LindbladKrausDensityHighwayBlock,
    parallel_transport_complex_quotient_momentum,
    parallel_transport_real_stiefel_momentum,
)
from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import (
    compute_kraus_branch_gauge_fraction,
    project_kraus_branch_gauge_horizontal,
)
from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import build_initial_kraus_stiefel_operators
from qdlblocks.benchmarks.selective_exact_quotient_lab import (
    _retract_complex_tall_stiefel_tensor_lr,
    _retract_real_tall_stiefel_tensor_lr,
    _whiten_tall_real_stiefel,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SOFTCAP_LOGIT,
    RAW_GPT2_VOCAB_SIZE,
)
from qdlblocks.layers.impure_density_mirror_block import (
    complex_l2_normalize,
    pack_rank1_ket_outer_isometric,
    unpack_hermitian_isometric,
)
from qdlblocks.layers.quantum_statmech_channels import (
    project_real_stiefel_tangent,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import NativeLeaderboardRotary

DEFAULT_INSPECTOR_JSON_OUT: str = ".cache/complementary_512d_lindblad_inspector.json"


def audit_complementary_512d_parseval_and_normalization(device: torch.device) -> dict[str, float]:
    """Audit `100.0%` Parseval `512D` norm preservation and `50,257`-way probability normalization (`<= 8` args)."""
    torch.manual_seed(1337)
    luders_b = ComplementaryOrthogonalLudersBoundary().to(device)
    luders_b.initialize_boundary_(seed=1337)
    dual_b = ComplementaryDualSubspace512DBoundary().to(device)
    dual_b.initialize_boundary_(seed=1337)

    _, _, u_c, u_f = luders_b.split_ortho_subspaces()
    h = torch.randn(32, 512, device=device, dtype=torch.float32)
    h_c = torch.matmul(h, u_c)
    h_f = torch.matmul(h, u_f)
    h_norm_sq = h.square().sum(dim=-1)
    split_norm_sq = h_c.square().sum(dim=-1) + h_f.square().sum(dim=-1)
    parseval_retention = float((split_norm_sq / h_norm_sq).mean().item())
    parseval_max_err = float((split_norm_sq - h_norm_sq).abs().max().item())
    cross_ortho_err = float(torch.matmul(u_c.t(), u_f).abs().max().item())

    u_vocab = torch.randn(2, 4, 512, device=device, dtype=torch.float32)
    u_vac = torch.randn(2, 4, 16, device=device, dtype=torch.float32)
    vac_mass = torch.tensor(0.25, device=device, dtype=torch.float32)
    beta_0 = torch.tensor(12.0, device=device, dtype=torch.float32)

    psi = torch.randn(2, 16, 4, 2, 16, device=device, dtype=torch.float32)
    pr, pi = complex_l2_normalize(psi[..., 0, :], psi[..., 1, :])
    rho_packed = pack_rank1_ket_outer_isometric(pr, pi)
    rho_re, rho_im = unpack_hermitian_isometric(rho_packed, 16)

    log_p_luders = luders_b.full_50257_log_probs(u_vocab, u_vac, vac_mass, rho_re, rho_im, beta_0)
    luders_prob_sum = log_p_luders.exp().sum(dim=-1)
    luders_norm_err = float((luders_prob_sum - 1.0).abs().max().item())

    log_p_dual = dual_b.full_50257_log_probs(u_vocab, u_vac, vac_mass, rho_re, rho_im, beta_0)
    dual_prob_sum = log_p_dual.exp().sum(dim=-1)
    dual_norm_err = float((dual_prob_sum - 1.0).abs().max().item())

    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 4), device=device)
    loss_luders_fwd, _ = luders_b.forward_nll_and_metrics(u_vocab, u_vac, vac_mass, rho_re, rho_im, beta_0, targets)
    loss_luders_ref = -log_p_luders.gather(-1, targets.reshape(-1, 1)).mean()
    luders_nll_match_err = float((loss_luders_fwd - loss_luders_ref).abs().item())

    loss_dual_fwd, _ = dual_b.forward_nll_and_metrics(u_vocab, u_vac, vac_mass, rho_re, rho_im, beta_0, targets)
    loss_dual_ref = -log_p_dual.gather(-1, targets.reshape(-1, 1)).mean()
    dual_nll_match_err = float((loss_dual_fwd - loss_dual_ref).abs().item())

    return {
        "parseval_512d_retention_ratio": round(parseval_retention, 6),
        "parseval_512d_max_abs_err": parseval_max_err,
        "subspace_cross_ortho_max_err": cross_ortho_err,
        "luders_50257_prob_sum_max_err": luders_norm_err,
        "dual_50257_prob_sum_max_err": dual_norm_err,
        "luders_nll_vs_full_probs_err": luders_nll_match_err,
        "dual_nll_vs_full_probs_err": dual_nll_match_err,
    }


def audit_dual_subspace_custom_autograd(device: torch.device) -> dict[str, float]:
    """Compare `_DualSubspace512DSoftcapCEFn` backward gradients against an exact PyTorch reference (`<= 8` args)."""
    torch.manual_seed(2026)
    n_pos, d_sub = 12, 256
    cap = float(DEFAULT_SOFTCAP_LOGIT)
    h1 = torch.randn(n_pos, d_sub, device=device, dtype=torch.float32, requires_grad=True)
    h2 = torch.randn(n_pos, d_sub, device=device, dtype=torch.float32, requires_grad=True)
    w_tok = torch.randn(RAW_GPT2_VOCAB_SIZE, d_sub, device=device, dtype=torch.float32, requires_grad=True)
    e_coarse = torch.randn(256, d_sub, device=device, dtype=torch.float32, requires_grad=True)
    e_fine = torch.randn(197, d_sub, device=device, dtype=torch.float32, requires_grad=True)
    bias_v = torch.randn(RAW_GPT2_VOCAB_SIZE, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (n_pos,), device=device, dtype=torch.long)

    loss_custom = _DualSubspace512DSoftcapCEFn.apply(h1, h2, w_tok, e_coarse, e_fine, bias_v, targets)
    grads_custom = torch.autograd.grad(loss_custom, (h1, h2, w_tok, e_coarse, e_fine, bias_v))

    h1_r = h1.detach().clone().requires_grad_(True)
    h2_r = h2.detach().clone().requires_grad_(True)
    wt_r = w_tok.detach().clone().requires_grad_(True)
    ec_r = e_coarse.detach().clone().requires_grad_(True)
    ef_r = e_fine.detach().clone().requires_grad_(True)
    bv_r = bias_v.detach().clone().requires_grad_(True)

    s_coarse = torch.matmul(h2_r, ec_r.t())  # (N, 256)
    s_fine = torch.matmul(h2_r, ef_r.t())    # (N, 197)
    s_comp_grid = (s_fine.unsqueeze(2) + s_coarse.unsqueeze(1)).reshape(n_pos, 197 * 256)[:, :RAW_GPT2_VOCAB_SIZE]
    s_tok = torch.matmul(h1_r, wt_r.t())
    raw_s = s_tok + s_comp_grid
    z = cap * torch.tanh(raw_s / cap) + bv_r.unsqueeze(0)
    loss_ref = F.cross_entropy(z, targets)
    grads_ref = torch.autograd.grad(loss_ref, (h1_r, h2_r, wt_r, ec_r, ef_r, bv_r))

    errs = [float((gc - gr).abs().max().item()) for gc, gr in zip(grads_custom, grads_ref)]
    return {
        "fwd_loss_abs_diff": float((loss_custom - loss_ref).abs().item()),
        "grad_h1_max_err": errs[0],
        "grad_h2_max_err": errs[1],
        "grad_w_tok_max_err": errs[2],
        "grad_e_coarse_max_err": errs[3],
        "grad_e_fine_max_err": errs[4],
        "grad_bias_max_err": errs[5],
    }


def audit_lindblad_highway_density_invariants(device: torch.device) -> dict[str, float]:
    """Verify unit trace $\\operatorname{Tr}(\\rho_{t,h}) = 1$ and PSD eigenvalues $\\lambda_{\\min}(\\rho_{t,h}) \\ge 0$ (`<= 8` args)."""
    torch.manual_seed(31415)
    blk = _LindbladKrausDensityHighwayBlock(enable_lindblad=True).to(device)
    kr, ki = build_initial_kraus_stiefel_operators(16, 16, kraus_rank=4, device=device)
    with torch.no_grad():
        blk.k_evol_re.copy_(kr)
        blk.k_evol_im.copy_(ki)

    bsz, seq_len = 2, 16
    x = torch.randn(bsz, seq_len, 512, device=device, dtype=torch.float32)
    psi = x.view(bsz, seq_len, 16, 2, 16).permute(0, 2, 1, 3, 4)
    pr, pi = complex_l2_normalize(psi[..., 0, :], psi[..., 1, :])
    rho_0 = pack_rank1_ket_outer_isometric(pr, pi)
    rope = NativeLeaderboardRotary(32, max_seq_len=64).to(device)
    cos_4d, sin_4d = rope.get_cos_sin(seq_len, dtype=torch.float32)

    with torch.no_grad():
        _, rho_next = blk(x, x.clone(), cos_4d, sin_4d, rho_0)
        rho_re, rho_im = unpack_hermitian_isometric(rho_next, 16)
        traces = torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1)
        trace_max_err = float((traces - 1.0).abs().max().item())
        rho_c = torch.complex(rho_re, rho_im)
        herm_err = float((rho_c - rho_c.conj().transpose(-1, -2)).abs().max().item())
        eigvals = torch.linalg.eigvalsh(rho_c)
        min_eig = float(eigvals.min().item())
        alpha_min = float(blk.highway_gamma.min().item())
        alpha_max = float(blk.highway_gamma.max().item())

    return {
        "lindblad_trace_max_err": trace_max_err,
        "lindblad_hermitian_max_err": herm_err,
        "lindblad_min_eigenvalue": min_eig,
        "highway_alpha_min": round(alpha_min, 4),
        "highway_alpha_max": round(alpha_max, 4),
    }


def audit_isometric_parallel_transport(device: torch.device) -> dict[str, float]:
    """Audit norm conservation and horizontal tangency of real and complex parallel transport (`<= 8` args)."""
    torch.manual_seed(4242)
    q_init, _ = torch.linalg.qr(torch.randn(64, 32, device=device, dtype=torch.float32))
    w_old = q_init.unsqueeze(0)
    m_raw = torch.randn_like(w_old)
    m_upd = project_real_stiefel_tangent(m_raw, w_old)
    raw_dir = project_real_stiefel_tangent(torch.randn_like(w_old), w_old)
    step_dir = project_real_stiefel_tangent(_whiten_tall_real_stiefel(raw_dir, ns_steps=5, tik_lam=1e-4), w_old)
    w_new = _retract_real_tall_stiefel_tensor_lr(w_old, step_dir, torch.tensor([[[0.08]]], device=device))

    m_trans = parallel_transport_real_stiefel_momentum(m_upd, w_old, w_new)
    real_norm_ratio = float((m_trans.norm() / m_upd.norm()).item())
    sym_leak = torch.matmul(w_new.transpose(-1, -2), m_trans)
    real_sym_err = float((0.5 * (sym_leak + sym_leak.transpose(-1, -2))).abs().max().item())

    kr, ki = build_initial_kraus_stiefel_operators(4, 16, kraus_rank=4, device=device)
    mr, mi = project_kraus_branch_gauge_horizontal(torch.randn_like(kr), torch.randn_like(ki), kr, ki)
    sr, si = project_kraus_branch_gauge_horizontal(torch.randn_like(kr), torch.randn_like(ki), kr, ki)
    kr_new, ki_new = _retract_complex_tall_stiefel_tensor_lr(
        kr, ki, sr, si, torch.full((4, 1, 1), 0.08, device=device)
    )
    mr_t, mi_t = parallel_transport_complex_quotient_momentum(
        mr, mi, kr, ki, kr_new, ki_new, project_kraus_branch_gauge_horizontal
    )
    old_c_norm = (mr.square() + mi.square()).sum().sqrt()
    new_c_norm = (mr_t.square() + mi_t.square()).sum().sqrt()
    complex_norm_ratio = float((new_c_norm / old_c_norm).item())
    kraus_gauge_leak = compute_kraus_branch_gauge_fraction(mr_t, mi_t, kr_new, ki_new)

    return {
        "real_pt_norm_ratio": round(real_norm_ratio, 6),
        "real_pt_sym_tangent_err": real_sym_err,
        "complex_pt_norm_ratio": round(complex_norm_ratio, 6),
        "complex_pt_kraus_gauge_leak": kraus_gauge_leak,
    }


def run_complementary_512d_lindblad_inspection(device: torch.device | None = None) -> dict[str, Any]:
    """Run all Wave 68 mathematical verification audits and return a structured report."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_rep = audit_complementary_512d_parseval_and_normalization(dev)
    g_rep = audit_dual_subspace_custom_autograd(dev)
    l_rep = audit_lindblad_highway_density_invariants(dev)
    t_rep = audit_isometric_parallel_transport(dev)

    all_passed = bool(
        abs(p_rep["parseval_512d_retention_ratio"] - 1.0) < 1e-5
        and p_rep["luders_50257_prob_sum_max_err"] < 1e-4
        and p_rep["dual_50257_prob_sum_max_err"] < 1e-4
        and g_rep["grad_h1_max_err"] < 1e-4
        and g_rep["grad_h2_max_err"] < 1e-4
        and l_rep["lindblad_trace_max_err"] < 1e-5
        and l_rep["lindblad_min_eigenvalue"] >= -1e-6
        and abs(t_rep["real_pt_norm_ratio"] - 1.0) < 1e-5
        and abs(t_rep["complex_pt_norm_ratio"] - 1.0) < 1e-5
        and t_rep["complex_pt_kraus_gauge_leak"] < 1e-6
    )
    return {
        "plan_id": "EXPERIMENT_PLAN_088",
        "wave_id": "W68",
        "device": str(dev),
        "all_checks_passed": all_passed,
        "parseval_and_normalization": p_rep,
        "dual_subspace_custom_autograd": g_rep,
        "lindblad_highway_invariants": l_rep,
        "isometric_parallel_transport": t_rep,
    }


def main() -> None:
    """CLI entrypoint for `complementary_512d_lindblad_inspector.py`."""
    parser = argparse.ArgumentParser(description="Inspect Wave 68 512D Complementary Readout & Lindblad Invariants.")
    parser.add_argument("--json-out", type=str, default=DEFAULT_INSPECTOR_JSON_OUT)
    args = parser.parse_args()

    report = run_complementary_512d_lindblad_inspection()
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
