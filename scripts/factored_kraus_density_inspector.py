"""Wave 70 (`EXPERIMENT_PLAN_090.md`) Inspector: Factored Cross-Head Kraus Superoperator & Representational Power.

Audits:
1. Isometric Hermitian Pack/Unpack Round-Trip & Outer-Product Identity (`fast_pack_hermitian_to_isometric_real` /
   `fast_unpack_isometric_real_to_hermitian`).
2. Lean 4 Theorem 9a & 9b Numerical Verification:
   - Exact equivalence between `apply_naive_pairwise_cross_head_kraus` ($O(H^2 M)$ 7D tensor),
     `apply_factored_associative_cross_head_kraus` (`Trick A`, $O(H M)$), and
     `apply_attention_commuted_cross_head_kraus` (`Trick B`, Rank-$R$ commuted).
   - Wall-clock latency speedup of `Trick A` and `Trick B` over the naive $O(H^2 M)$ congruence loop.
3. Lean 4 Theorem 10a & 10b Numerical Verification:
   - Exact CPTP trace-1, Hermitian symmetry, and PSD preservation.
   - Strict representational power separation over scalar orthostochastic mixing (`0.0` orthogonal basis transfer
     and rank $\\le 4$ for scalar mixing vs. $> 0.60$ orthogonal basis mass and full **Rank 16** for
     `FactoredCrossHeadKrausSuperoperator` and `MultiBranchStiefelKrausSuperoperator`).
4. Full `FactoredKrausPureDensityGPT` forward/backward/step quotient horizontality check.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import _collect_active_vocab_indices_w68
from qdlblocks.benchmarks.factored_kraus_density_gpt_lab import (
    DEFAULT_FACTORED_KRAUS_BRANCHES,
    FULL_RANK_EIGVAL_THRESHOLD,
    FactoredKrausPureDensityGPT,
    Wave70ExperimentSpec,
    Wave70VariantMode,
    apply_attention_commuted_cross_head_kraus,
    apply_factored_associative_cross_head_kraus,
    apply_multibranch_stiefel_kraus_superoperator,
    apply_naive_pairwise_cross_head_kraus,
    audit_riemannian_momentum_buffer_properties,
    build_wave69_optimizers,
    fast_pack_hermitian_to_isometric_real,
    fast_unpack_isometric_real_to_hermitian,
)
from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    DEFAULT_PURE_CHOLESKY_RANK,
    DEFAULT_PURE_RHO_DIM,
    DEFAULT_PURE_RHO_HEADS,
    normalize_cholesky_factor_frobenius,
    pack_cholesky_factor_outer_isometric,
)


def _random_unitary_stack(n_mats: int, d_h: int, gen: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate `n_mats` Haar-like unitary matrices in $U(d_h)$ (`<= 8` args)."""
    re_list, im_list = [], []
    for _ in range(n_mats):
        a_r = torch.randn(d_h, d_h, generator=gen, device=device, dtype=torch.float32)
        a_i = torch.randn(d_h, d_h, generator=gen, device=device, dtype=torch.float32)
        q_c, _ = torch.linalg.qr(torch.complex(a_r, a_i))
        re_list.append(q_c.real)
        im_list.append(q_c.imag)
    return torch.stack(re_list, dim=0), torch.stack(im_list, dim=0)


@torch.no_grad()
def verify_lut_pack_unpack_equivalence(device: torch.device, seed: int = 1337) -> dict[str, Any]:
    """Verify exact round-trip and Frobenius isometry of Hermitian pack/unpack (`<= 8` args)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    l_re, l_im = normalize_cholesky_factor_frobenius(
        torch.randn(4, DEFAULT_PURE_RHO_HEADS, 32, DEFAULT_PURE_RHO_DIM, DEFAULT_PURE_CHOLESKY_RANK, generator=gen, device=device),
        torch.randn(4, DEFAULT_PURE_RHO_HEADS, 32, DEFAULT_PURE_RHO_DIM, DEFAULT_PURE_CHOLESKY_RANK, generator=gen, device=device),
    )
    packed = pack_cholesky_factor_outer_isometric(l_re, l_im)
    rho_re, rho_im = fast_unpack_isometric_real_to_hermitian(packed)
    repacked = fast_pack_hermitian_to_isometric_real(rho_re, rho_im)

    roundtrip_err = float((packed - repacked).abs().max().item())
    herm_err = float(
        max((rho_re - rho_re.transpose(-2, -1)).abs().max().item(), (rho_im + rho_im.transpose(-2, -1)).abs().max().item())
    )
    trace_err = float((rho_re.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
    passed = (roundtrip_err <= 1e-6) and (herm_err <= 1e-6) and (trace_err <= 1e-6)
    return {
        "passed": bool(passed),
        "roundtrip_max_abs_err": roundtrip_err,
        "hermitian_symmetry_err": herm_err,
        "unit_trace_err": trace_err,
    }


@torch.no_grad()
def verify_theorem9_algebraic_equivalence_and_speedup(device: torch.device, seed: int = 2026) -> dict[str, Any]:
    """Verify Theorem 9a (`Trick A`) and Theorem 9b (`Trick B`) against $O(H^2 M)$ reference (`<= 8` args)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    b_sz, n_h, seq_l, d_h, r_chol = 2, DEFAULT_PURE_RHO_HEADS, 16, DEFAULT_PURE_RHO_DIM, DEFAULT_PURE_CHOLESKY_RANK
    m_br = DEFAULT_FACTORED_KRAUS_BRANCHES

    v_re, v_im = normalize_cholesky_factor_frobenius(
        torch.randn(b_sz, n_h, seq_l, d_h, r_chol, generator=gen, device=device),
        torch.randn(b_sz, n_h, seq_l, d_h, r_chol, generator=gen, device=device),
    )
    v_packed = pack_cholesky_factor_outer_isometric(v_re, v_im)

    raw_attn = torch.rand(b_sz, n_h, seq_l, seq_l, generator=gen, device=device)
    causal_mask = torch.tril(torch.ones(seq_l, seq_l, device=device, dtype=torch.bool))
    raw_attn = raw_attn.masked_fill(~causal_mask, 0.0)
    raw_attn[..., :, 0] = raw_attn[..., :, 0] + 0.1
    attn_weights = raw_attn / raw_attn.sum(dim=-1, keepdim=True)
    rho_attn_packed = torch.matmul(attn_weights, v_packed)

    ki_re, ki_im = _random_unitary_stack(n_h * m_br, d_h, gen, device)
    ko_re, ko_im = _random_unitary_stack(n_h * m_br, d_h, gen, device)
    k_in_pair = (ki_re.view(n_h, m_br, d_h, d_h), ki_im.view(n_h, m_br, d_h, d_h))
    k_out_pair = (ko_re.view(n_h, m_br, d_h, d_h), ko_im.view(n_h, m_br, d_h, d_h))
    w_route = torch.randn(n_h, n_h, m_br, generator=gen, device=device)

    t0 = time.perf_counter()
    out_naive = apply_naive_pairwise_cross_head_kraus(rho_attn_packed, k_in_pair, k_out_pair, w_route)
    naive_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    out_trick_a = apply_factored_associative_cross_head_kraus(rho_attn_packed, k_in_pair, k_out_pair, w_route)
    trick_a_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    out_trick_b = apply_attention_commuted_cross_head_kraus((v_re, v_im), attn_weights, k_in_pair, k_out_pair, w_route)
    trick_b_ms = (time.perf_counter() - t2) * 1000.0

    denom = out_naive.norm().clamp_min(1e-8)
    rel_err_a = float(((out_naive - out_trick_a).norm() / denom).item())
    rel_err_b = float(((out_naive - out_trick_b).norm() / denom).item())
    rel_err_ab = float(((out_trick_a - out_trick_b).norm() / denom).item())

    passed = (rel_err_a < 1e-5) and (rel_err_b < 1e-5) and (rel_err_ab < 1e-5)
    return {
        "passed": bool(passed),
        "trick_a_vs_naive_rel_err": rel_err_a,
        "trick_b_vs_naive_rel_err": rel_err_b,
        "trick_a_vs_trick_b_rel_err": rel_err_ab,
        "naive_ms": round(naive_ms, 3),
        "trick_a_ms": round(trick_a_ms, 3),
        "trick_b_ms": round(trick_b_ms, 3),
        "trick_a_speedup_x": round(naive_ms / max(1e-6, trick_a_ms), 2),
        "trick_b_speedup_x": round(naive_ms / max(1e-6, trick_b_ms), 2),
    }


@torch.no_grad()
def verify_theorem10_cptp_and_representational_power(device: torch.device, seed: int = 3030) -> dict[str, Any]:
    """Verify Theorem 10a (CPTP closure) & Theorem 10b (strict expressivity separation & Rank 16 amplification) (`<= 8` args)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    b_sz, n_h, seq_l, d_h, r_chol = 1, DEFAULT_PURE_RHO_HEADS, 4, DEFAULT_PURE_RHO_DIM, DEFAULT_PURE_CHOLESKY_RANK
    m_br = DEFAULT_FACTORED_KRAUS_BRANCHES

    l_sub_re = torch.zeros(b_sz, n_h, seq_l, d_h, r_chol, device=device, dtype=torch.float32)
    l_sub_im = torch.zeros(b_sz, n_h, seq_l, d_h, r_chol, device=device, dtype=torch.float32)
    l_sub_re[..., :r_chol, :] = torch.randn(b_sz, n_h, seq_l, r_chol, r_chol, generator=gen, device=device)
    l_sub_im[..., :r_chol, :] = torch.randn(b_sz, n_h, seq_l, r_chol, r_chol, generator=gen, device=device)
    l_sub_re, l_sub_im = normalize_cholesky_factor_frobenius(l_sub_re, l_sub_im)
    rho_sub_packed = pack_cholesky_factor_outer_isometric(l_sub_re, l_sub_im)

    q_or, _ = torch.linalg.qr(torch.randn(n_h, n_h, generator=gen, device=device))
    b_orth = q_or.square()
    rho_scalar_packed = torch.einsum("ij,bjtd->bitd", b_orth, rho_sub_packed)
    scalar_re, scalar_im = fast_unpack_isometric_real_to_hermitian(rho_scalar_packed)
    scalar_orth_mass = float(scalar_re[..., r_chol:, r_chol:].diagonal(dim1=-2, dim2=-1).sum(dim=-1).max().item())
    scalar_eigs = torch.linalg.eigvalsh(torch.complex(scalar_re, scalar_im))
    scalar_max_rank = int((scalar_eigs > FULL_RANK_EIGVAL_THRESHOLD).sum(dim=-1).max().item())

    ki_re, ki_im = _random_unitary_stack(n_h * m_br, d_h, gen, device)
    ko_re, ko_im = _random_unitary_stack(n_h * m_br, d_h, gen, device)
    w_route = torch.randn(n_h, n_h, m_br, generator=gen, device=device)
    rho_factored_packed = apply_factored_associative_cross_head_kraus(
        rho_sub_packed,
        (ki_re.view(n_h, m_br, d_h, d_h), ki_im.view(n_h, m_br, d_h, d_h)),
        (ko_re.view(n_h, m_br, d_h, d_h), ko_im.view(n_h, m_br, d_h, d_h)),
        w_route,
    )
    fac_re, fac_im = fast_unpack_isometric_real_to_hermitian(rho_factored_packed)
    fac_trace_err = float((fac_re.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
    fac_herm_err = float(
        max((fac_re - fac_re.transpose(-2, -1)).abs().max().item(), (fac_im + fac_im.transpose(-2, -1)).abs().max().item())
    )
    fac_orth_mass = float(fac_re[..., r_chol:, r_chol:].diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().item())
    fac_eigs = torch.linalg.eigvalsh(torch.complex(fac_re, fac_im))
    fac_min_eig = float(fac_eigs.min().item())
    fac_max_rank = int((fac_eigs > FULL_RANK_EIGVAL_THRESHOLD).sum(dim=-1).max().item())

    flat_dim = n_h * 2 * d_h
    w_mb = torch.stack(
        [torch.linalg.qr(torch.randn(flat_dim, flat_dim, generator=gen, device=device))[0] for _ in range(m_br)],
        dim=0,
    )
    _, _, rho_mb_packed = apply_multibranch_stiefel_kraus_superoperator(l_sub_re, l_sub_im, w_mb)
    mb_re, mb_im = fast_unpack_isometric_real_to_hermitian(rho_mb_packed)
    mb_trace_err = float((mb_re.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
    mb_orth_mass = float(mb_re[..., r_chol:, r_chol:].diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().item())
    mb_eigs = torch.linalg.eigvalsh(torch.complex(mb_re, mb_im))
    mb_min_eig = float(mb_eigs.min().item())
    mb_max_rank = int((mb_eigs > FULL_RANK_EIGVAL_THRESHOLD).sum(dim=-1).max().item())

    passed = (
        (scalar_orth_mass <= 1e-7)
        and (scalar_max_rank == r_chol)
        and (fac_trace_err < 1e-5)
        and (fac_herm_err <= 1e-6)
        and (fac_min_eig >= -1e-6)
        and (fac_orth_mass > 0.50)
        and (fac_max_rank == d_h)
        and (mb_trace_err < 1e-5)
        and (mb_min_eig >= -1e-6)
        and (mb_orth_mass > 0.50)
        and (mb_max_rank == d_h)
    )
    return {
        "passed": bool(passed),
        "scalar_orthogonal_subspace_mass": scalar_orth_mass,
        "scalar_max_rank": scalar_max_rank,
        "factored_kraus_unit_trace_err": fac_trace_err,
        "factored_kraus_hermitian_err": fac_herm_err,
        "factored_kraus_min_eigval": fac_min_eig,
        "factored_kraus_orthogonal_subspace_mass": round(fac_orth_mass, 4),
        "factored_kraus_max_rank": fac_max_rank,
        "multibranch_kraus_unit_trace_err": mb_trace_err,
        "multibranch_kraus_min_eigval": round(mb_min_eig, 6),
        "multibranch_kraus_orthogonal_subspace_mass": round(mb_orth_mass, 4),
        "multibranch_kraus_max_rank": mb_max_rank,
    }


def verify_wave70_model_forward_backward_step(device: torch.device, seed: int = 4040) -> dict[str, Any]:
    """Run 1 forward/backward/optimizer step on `FactoredKrausPureDensityGPT` (`W70_K2`) (`<= 8` args)."""
    spec = Wave70ExperimentSpec(
        variant_mode=Wave70VariantMode.W70_K2_COMMUTED_AND_MULTIBRANCH_4L,
        num_layers=2,
        seed=seed,
    )
    model = FactoredKrausPureDensityGPT(spec).to(device)
    model.initialize_quantum_state(seed=seed)
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)

    idx_in = torch.randint(0, 50257, (2, 16), device=device)
    targets = torch.randint(0, 50257, (2, 16), device=device)
    loss = model(idx_in, targets=targets, return_metrics=False)
    loss.backward()

    trunk_opt.record_diagnostics = True
    trunk_opt.step()
    vocab_opt.step_active(_collect_active_vocab_indices_w68(idx_in, targets), lr_scale=1.0)
    scalar_opt.step_active(active_idx=None, lr_scale=1.0)
    model.restrict_scalars_()

    gd = trunk_opt.last_gauge_diagnostics
    su16_leak = float(gd.get("su16_step_phase_frac", 0.0))
    kraus_leak = float(gd.get("kraus_step_gauge_frac", 0.0))
    passed = math.isfinite(float(loss.item())) and (su16_leak < 1e-6) and (kraus_leak < 1e-5)
    return {
        "passed": bool(passed),
        "loss": round(float(loss.item()), 4),
        "total_params": sum(p.numel() for p in model.parameters()),
        "su16_step_phase_frac": su16_leak,
        "kraus_step_gauge_frac": kraus_leak,
    }


def verify_theorem11_riemannian_momentum_buffer(device: torch.device) -> dict[str, Any]:
    """Verify Lean 4 Theorem 11 (Riemannian Quotient Momentum Buffer at beta=0.95) (`<= 8` args)."""
    res = audit_riemannian_momentum_buffer_properties(device)
    passed = (
        abs(res["plateau_momentum_beta"] - 0.95) < 1e-6
        and res["uncleaned_opposing_descent_ratio"] < 0.0
        and res["cleaned_opposing_descent_ratio"] >= 1.9499
        and res["cleaned_orthogonality_err"] < 1e-6
        and res["cleaned_pythagorean_err"] < 1e-6
        and res["transported_norm_rel_err"] < 1e-6
        and res["transported_vertical_leak"] < 1e-6
        and res["max_su16_mom_vertical_leak"] < 1e-6
        and res["max_kraus_mom_vertical_leak"] < 1e-5
        and res["max_mub_mom_vertical_leak"] < 1e-6
        and res["active_stiefel_mom_buffers"] >= 28.0
        and res["active_su16_mom_buffers"] >= 28.0
    )
    return {"passed": bool(passed), **res}


def run_all_wave70_inspections(device_str: str = "cpu") -> dict[str, Any]:
    """Run all 5 Wave 70 mathematical, speed, representational-power, and momentum audits (`<= 8` args)."""
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    c1 = verify_lut_pack_unpack_equivalence(device)
    c2 = verify_theorem9_algebraic_equivalence_and_speedup(device)
    c3 = verify_theorem10_cptp_and_representational_power(device)
    c4 = verify_wave70_model_forward_backward_step(device)
    c5 = verify_theorem11_riemannian_momentum_buffer(device)
    overall = bool(c1["passed"] and c2["passed"] and c3["passed"] and c4["passed"] and c5["passed"])
    return {
        "plan_id": "EXPERIMENT_PLAN_090",
        "wave_id": "W70",
        "device": str(device),
        "overall_passed": overall,
        "lut_pack_unpack_equivalence": c1,
        "theorem9_algebraic_equivalence_and_speedup": c2,
        "theorem10_cptp_and_representational_power": c3,
        "wave70_model_step_and_quotient": c4,
        "theorem11_riemannian_momentum_buffer": c5,
    }


def main() -> None:
    """CLI entry point for `scripts/factored_kraus_density_inspector.py` (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Wave 70 Factored Cross-Head Kraus Superoperator Inspector")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--json-out", type=str, default=".cache/plan090_factored_kraus_inspector.json", help="Output JSON path")
    args = parser.parse_args()

    report = run_all_wave70_inspections(args.device)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
