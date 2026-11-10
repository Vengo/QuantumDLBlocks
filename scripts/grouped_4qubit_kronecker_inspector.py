#!/usr/bin/env python3
"""Wave 72 (`EXPERIMENT_PLAN_092.md`) Grouped 4-Qubit Kronecker Density Inspector.

Verifies:
1. Theorem 13a (4-Qubit Segre Product Unit Norm, Exact Mutual Orthogonality in `C^{65,536}`, & Intra-Head Entanglement Generation under `SU(16)`).
2. Theorem 13b (Elimination of Mean-Field Superposition Collapse via `B`-Branch Kronecker Density Mixtures).
3. Theorem 13c (Zero-Materialization `65,536D` Hilbert-Schmidt Born Coincidence Identity).
4. `GroupedFourQubitKroneckerDensityGPT` Forward/Backward & `PureDensityQuotientMuon` Quotient Invariants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qdlblocks.benchmarks.anchored_bipartite_density_gpt_lab import _build_deterministic_haar_unitary_stack
from qdlblocks.benchmarks.factored_kraus_density_gpt_lab import compute_restored_riemannian_momentum_schedule
from qdlblocks.benchmarks.grouped_4qubit_kronecker_density_gpt_lab import (
    GroupedFourQubitKroneckerDensityGPT,
    Wave72VariantMode,
    build_deterministic_16qubit_binary_codebook,
    build_wave72_variant_spec,
    compute_four_qubit_partial_trace_purity,
    compute_grouped_4qubit_segre_product,
    evaluate_factorized_65536d_kronecker_born_overlap,
)
from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    _collect_active_vocab_indices_w68,
    apply_complex_unitary_to_cholesky,
    build_wave69_optimizers,
    normalize_cholesky_factor_frobenius,
)

TOL: float = 1e-5


def verify_theorem13a_segre_norm_orthogonality_and_entanglement(device: torch.device) -> dict[str, Any]:
    """Verify Theorem 13a: 4-qubit Segre norm preservation, 65,536D binary orthogonality, and SU(16) intra-head entanglement (`<= 8` args)."""
    torch.manual_seed(2026092)
    q_re = torch.randn(32, 4, 4, 2, device=device, dtype=torch.float32)
    q_im = torch.randn(32, 4, 4, 2, device=device, dtype=torch.float32)
    phi_re, phi_im = compute_grouped_4qubit_segre_product(q_re, q_im)

    # 1. Exact unit norm of 4-qubit Segre product in C^16
    segre_norm_err = float(((phi_re.square() + phi_im.square()).sum(dim=-1) - 1.0).abs().max().item())

    # 2. Layer-0 Segre product state has exact 2-qubit partial-trace purity == 1.0 (separable)
    l_seg_re, l_seg_im = phi_re.unsqueeze(2).unsqueeze(-1), phi_im.unsqueeze(2).unsqueeze(-1)
    separable_purity = compute_four_qubit_partial_trace_purity(l_seg_re, l_seg_im)
    separable_purity_err = float((separable_purity - 1.0).abs().max().item())

    # 3. Applying a Haar SU(16) unitary across the 4 qubits generates strictly entangled states (purity < 1.0)
    u16_re, u16_im = _build_deterministic_haar_unitary_stack(4, 16, 2026093)
    u16_re, u16_im = u16_re.to(device), u16_im.to(device)
    ent_re, ent_im = apply_complex_unitary_to_cholesky(l_seg_re, l_seg_im, u16_re, u16_im)
    entangled_purity_mean = float(compute_four_qubit_partial_trace_purity(ent_re, ent_im).mean().item())

    # 4. Exact mutual orthogonality of distinct binary 16-qubit codewords in (C^2)^{x16} = C^{65,536}
    cb_re, cb_im = build_deterministic_16qubit_binary_codebook(64, perturb_std=0.0, seed=1337, device=device)
    cb_phi_re, cb_phi_im = compute_grouped_4qubit_segre_product(cb_re, cb_im)  # (64, 4, 16)
    # Pairwise overlap in C^{65,536} = prod_{g=1}^4 |<Phi_{u,g} | Phi_{v,g}>|^2
    ov_re = torch.einsum("ugi,vgi->uvg", cb_phi_re, cb_phi_re) + torch.einsum("ugi,vgi->uvg", cb_phi_im, cb_phi_im)
    ov_im = torch.einsum("ugi,vgi->uvg", cb_phi_re, cb_phi_im) - torch.einsum("ugi,vgi->uvg", cb_phi_im, cb_phi_re)
    joint_gram = (ov_re.square() + ov_im.square()).prod(dim=-1)  # (64, 64)
    eye_64 = torch.eye(64, device=device, dtype=torch.float32)
    codebook_ortho_err = float((joint_gram - eye_64).abs().max().item())

    passed = (
        segre_norm_err < TOL
        and separable_purity_err < TOL
        and entangled_purity_mean < 0.65
        and codebook_ortho_err < TOL
    )
    return {
        "passed": bool(passed),
        "segre_norm_err": round(segre_norm_err, 8),
        "separable_purity_mean": round(float(separable_purity.mean().item()), 6),
        "separable_purity_err": round(separable_purity_err, 8),
        "entangled_post_su16_purity_mean": round(entangled_purity_mean, 6),
        "codebook_65536d_ortho_err": round(codebook_ortho_err, 8),
    }


def verify_theorem13bc_multibranch_ghz_and_65536d_factorization(device: torch.device) -> dict[str, Any]:
    """Verify Theorem 13b (mean-field collapse vs multi-branch exactness) & Theorem 13c (65,536D factorized identity) (`<= 8` args)."""
    # 1. Theorem 13b: Bimodal target distribution on 16 = 2^4 factor patterns (|0,0,0,0> and |1,1,1,1> in C^{16}^4)
    # Construct all 16 binary combinations (b1, b2, b3, b4) in {0, 1}^4 where state |0> is e_0 in C^{16} and |1> is e_1 in C^{16}
    patterns = torch.zeros(16, 4, 16, device=device, dtype=torch.float32)
    for idx in range(16):
        for g in range(4):
            bit = (idx >> g) & 1
            patterns[idx, g, bit] = 1.0
    patterns_im = torch.zeros_like(patterns)

    # Single-branch mean-field state (B=1): each factor g has marginal rho_g = 0.5 |0><0| + 0.5 |1><1| (Rank 2 Cholesky)
    l_mf_re = torch.zeros(1, 4, 16, 2, device=device, dtype=torch.float32)
    l_mf_re[0, :, 0, 0] = (0.5) ** 0.5
    l_mf_re[0, :, 1, 1] = (0.5) ** 0.5
    l_mf_im = torch.zeros_like(l_mf_re)
    probs_mf = evaluate_factorized_65536d_kronecker_born_overlap(
        (l_mf_re, l_mf_im), (patterns, patterns_im), torch.ones(1, device=device)
    )
    mf_target_0000 = float(probs_mf[0].item())
    mf_target_1111 = float(probs_mf[15].item())
    mf_cross_bleed = float(probs_mf[1:15].sum().item())

    # Multi-branch Kronecker mixture (B=2): branch 0 is |0,0,0,0>, branch 1 is |1,1,1,1>, weights w = [0.5, 0.5]
    l_mb_re = torch.zeros(2, 4, 16, 1, device=device, dtype=torch.float32)
    l_mb_re[0, :, 0, 0] = 1.0
    l_mb_re[1, :, 1, 0] = 1.0
    l_mb_im = torch.zeros_like(l_mb_re)
    w_mb = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)
    probs_mb = evaluate_factorized_65536d_kronecker_born_overlap((l_mb_re, l_mb_im), (patterns, patterns_im), w_mb)
    mb_target_0000 = float(probs_mb[0].item())
    mb_target_1111 = float(probs_mb[15].item())
    mb_cross_bleed = float(probs_mb[1:15].sum().item())

    # 2. Theorem 13c: Exact numerical match between explicit 65,536D tensor product and factorized formula
    torch.manual_seed(2026094)
    l_rand_re, l_rand_im = normalize_cholesky_factor_frobenius(
        torch.randn(4, 4, 16, 4, device=device), torch.randn(4, 4, 16, 4, device=device)
    )
    phi_v_re, phi_v_im = compute_grouped_4qubit_segre_product(
        torch.randn(3, 4, 4, 2, device=device), torch.randn(3, 4, 4, 2, device=device)
    )
    w_b = torch.softmax(torch.randn(4, device=device), dim=0)
    fast_ov = evaluate_factorized_65536d_kronecker_born_overlap((l_rand_re, l_rand_im), (phi_v_re, phi_v_im), w_b)

    # Explicit 65,536D ket |Phi_v> = Phi_{v,0} (x) Phi_{v,1} (x) Phi_{v,2} (x) Phi_{v,3} in C^{65,536}
    phi_c = torch.complex(phi_v_re, phi_v_im)  # (3, 4, 16)
    l_c = torch.complex(l_rand_re, l_rand_im)  # (4_B, 4_G, 16, 4_R)
    ref_ov = []
    for v_i in range(3):
        ket_65536 = torch.einsum("i,j,k,l->ijkl", phi_c[v_i, 0], phi_c[v_i, 1], phi_c[v_i, 2], phi_c[v_i, 3]).reshape(65536)
        val_v = 0.0
        for b_i in range(4):
            # Each factor density matrix rho_{b,g} = L_{b,g} L_{b,g}^dagger in C^{16 x 16}
            rhos = [l_c[b_i, g] @ l_c[b_i, g].conj().t() for g in range(4)]
            # Apply (rho_0 (x) rho_1 (x) rho_2 (x) rho_3) to ket_65536 via 4-index contraction
            ket_4d = ket_65536.view(16, 16, 16, 16)
            out_4d = torch.einsum("ai,bj,ck,dl,ijkl->abcd", rhos[0], rhos[1], rhos[2], rhos[3], ket_4d)
            val_v += float(w_b[b_i].item()) * float(torch.vdot(ket_65536, out_4d.reshape(65536)).real.item())
        ref_ov.append(val_v)
    ref_t = torch.tensor(ref_ov, device=device, dtype=torch.float32)
    factorized_65536d_max_err = float((fast_ov - ref_t).abs().max().item())

    passed = (
        abs(mf_target_0000 - 0.0625) < TOL
        and abs(mf_cross_bleed - 0.875) < TOL
        and abs(mb_target_0000 - 0.5) < TOL
        and abs(mb_target_1111 - 0.5) < TOL
        and mb_cross_bleed == 0.0
        and factorized_65536d_max_err < TOL
    )
    return {
        "passed": bool(passed),
        "mf_target_0000_prob": round(mf_target_0000, 6),
        "mf_target_1111_prob": round(mf_target_1111, 6),
        "mf_cross_bleed_prob": round(mf_cross_bleed, 6),
        "mb_target_0000_prob": round(mb_target_0000, 6),
        "mb_target_1111_prob": round(mb_target_1111, 6),
        "mb_cross_bleed_prob": round(mb_cross_bleed, 8),
        "factorized_65536d_max_err": round(factorized_65536d_max_err, 8),
    }


def verify_wave72_model_step_and_quotient(device: torch.device) -> dict[str, Any]:
    """Verify forward/backward + 2-step `PureDensityQuotientMuon` invariants on `GroupedFourQubitKroneckerDensityGPT` (`<= 8` args)."""
    spec = build_wave72_variant_spec(Wave72VariantMode.W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L, seed=1337)
    model = GroupedFourQubitKroneckerDensityGPT(spec).to(device)
    model.train()
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
    total_params = sum(p.numel() for p in model.parameters())

    idx = torch.randint(0, 4096, (2, 16), device=device)
    targets = torch.randint(0, 4096, (2, 16), device=device)
    losses: list[float] = []
    for step in (1, 2):
        for opt in (trunk_opt, vocab_opt, scalar_opt):
            opt.zero_grad(set_to_none=True)
        loss, metrics = model(idx, targets=targets, return_metrics=True)
        loss.backward()
        lr_mult, mom_beta = compute_restored_riemannian_momentum_schedule(step, 100)
        trunk_opt.record_diagnostics = True
        trunk_opt.set_momentum(mom_beta)
        trunk_opt.set_lr_scale(lr_mult)
        trunk_opt.step()
        vocab_opt.set_momentum(mom_beta)
        vocab_opt.step_active(_collect_active_vocab_indices_w68(idx, targets), lr_scale=lr_mult)
        scalar_opt.set_momentum(min(0.90, mom_beta))
        scalar_opt.step_active(active_idx=None, lr_scale=lr_mult)
        model.restrict_scalars_()
        losses.append(float(loss.detach().item()))

    gd = trunk_opt.last_gauge_diagnostics
    su16_phase = float(gd.get("su16_step_phase_frac", 0.0))
    kraus_gauge = float(gd.get("kraus_step_gauge_frac", 0.0))
    passed = su16_phase < 1e-6 and kraus_gauge < 1e-5 and all(v > 0.0 for v in losses)
    return {
        "passed": bool(passed),
        "total_params": total_params,
        "step1_loss": round(losses[0], 4),
        "step2_loss": round(losses[1], 4),
        "su16_step_phase_frac": round(su16_phase, 8),
        "kraus_step_gauge_frac": round(kraus_gauge, 8),
        "analytical_beta": round(float(metrics["analytical_beta"]), 4),
    }


def main() -> None:
    """CLI entrypoint for Wave 72 grouped 4-qubit Kronecker density verification (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Wave 72 Grouped 4-Qubit Kronecker Density Inspector")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=str, default=".cache/wave72_grouped_4qubit_kronecker_inspector.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    t13a = verify_theorem13a_segre_norm_orthogonality_and_entanglement(device)
    t13bc = verify_theorem13bc_multibranch_ghz_and_65536d_factorization(device)
    model_chk = verify_wave72_model_step_and_quotient(device)

    report = {
        "plan_id": "EXPERIMENT_PLAN_092",
        "wave_id": "W72",
        "device": str(device),
        "overall_passed": bool(t13a["passed"] and t13bc["passed"] and model_chk["passed"]),
        "theorem13a_segre_norm_orthogonality_and_entanglement": t13a,
        "theorem13bc_multibranch_ghz_and_65536d_factorization": t13bc,
        "wave72_model_step_and_quotient": model_chk,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
