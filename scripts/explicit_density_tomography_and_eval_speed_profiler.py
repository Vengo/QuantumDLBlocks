r"""Diagnostic & Throughput Profiler for `EXPERIMENT_PLAN_083.md` (`Wave 63`): Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density-Matrix Highway, `16 × 32 = 512D` Quantum State Tomography Readout ($\operatorname{Tr}(\rho_t E_v)$), & Zero-Synthesis Validation Speedup.

Verifies:
1. **Frobenius-Isometric Outer-Product Identity**:
   $\langle \tilde{P}(\rho_{t,h}), \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \rangle_{\mathbb{R}^{256}} \equiv \langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle$
   to `< 1e-5` max absolute error across all $H = 16$ heads and $K = 32$ probes/head.
2. **Tomographic Operator Trace Equivalence (`512D` Dot Product == `16 × 16` Matrix Trace)**:
   $\langle \mathbf{w}_v, \mathbf{q}_t \rangle \equiv \sum_{h=0}^{15} \operatorname{Tr}(\rho_{t,h} E_{v,h})$
   where $E_{v,h} = \sum_{k=0}^{31} w_{v,h,k} |c_{k,h}\rangle\langle c_{k,h}| \in \operatorname{Herm}(\mathbb{C}^{16})$.
3. **Rule-9 Single-Pass Custom Autograd Efficiency Gate**:
   Verifies `_SingleGemm512DTomographicCEFn` via `audit_chunked_nll_autograd_efficiency(..., forbid_exponential_ops=False)`.
4. **Validation & Training Step Wall-Clock Throughput Breakdown**:
   Compares `Wave 62` per-batch `50,304 × 512` detector re-synthesis against `Wave 63` zero-synthesis
   hoisted `512D` evaluation (`T2` and `T3`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from qdlblocks.benchmarks.autograd_efficiency_auditor import audit_chunked_nll_autograd_efficiency
from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (
    DEFAULT_PROBES_PER_HEAD,
    DEFAULT_REGISTER_BUCKETS,
    DEFAULT_TOMO_EMBED_DIM,
    DEFAULT_TOMO_HEAD_COMPLEX_DIM,
    DEFAULT_TOMO_NUM_HEADS,
    ExplicitDensityTomographicGPT,
    ExplicitDensityTomographicSpec,
    ExplicitDensityVariantMode,
    _SingleGemm512DTomographicCEFn,
    compute_tomographic_512d_probe_expectations,
    conjugate_packed_hermitian_by_unitary,
    initialize_explicit_density_tomographic_model_,
)
from qdlblocks.layers.impure_density_mirror_block import (
    complex_l2_normalize,
    pack_hermitian_isometric,
    unpack_hermitian_isometric,
)


def audit_isometric_and_tomographic_trace_identities(device: torch.device) -> dict[str, Any]:
    r"""Verify Frobenius-isometric probe expectation and $\langle \mathbf{w}_v, \mathbf{q}_t \rangle = \sum_h \operatorname{Tr}(\rho_{t,h} E_{v,h})$."""
    torch.manual_seed(2026)
    bsz, seq_len = 2, 8
    num_h = DEFAULT_TOMO_NUM_HEADS
    d_h = DEFAULT_TOMO_HEAD_COMPLEX_DIM
    num_k = DEFAULT_PROBES_PER_HEAD

    # Build a random Rank-3 mixed Hermitian density matrix per head: rho = sum_{m=1}^3 p_m |psi_m><psi_m|
    raw_kets = torch.randn(bsz, num_h, seq_len, 3, 2, d_h, device=device, dtype=torch.float32)
    k_re, k_im = complex_l2_normalize(raw_kets[..., 0, :], raw_kets[..., 1, :])
    weights = torch.tensor([0.55, 0.30, 0.15], device=device, dtype=torch.float32).view(1, 1, 1, 3, 1, 1)

    outer_re = (
        weights * (k_re.unsqueeze(-1) * k_re.unsqueeze(-2) + k_im.unsqueeze(-1) * k_im.unsqueeze(-2))
    ).sum(dim=3)
    outer_im = (
        weights * (k_im.unsqueeze(-1) * k_re.unsqueeze(-2) - k_re.unsqueeze(-1) * k_im.unsqueeze(-2))
    ).sum(dim=3)
    rho_packed = pack_hermitian_isometric(outer_re, outer_im)

    # Probe kets |c_{k,h}> in C^{16}
    raw_probes = torch.randn(num_h, num_k, 2, d_h, device=device, dtype=torch.float32)
    c_re, c_im = complex_l2_normalize(raw_probes[..., 0, :], raw_probes[..., 1, :])

    # 1. Fast packed tomographic expectation (B, T, 512)
    q_fast_512 = compute_tomographic_512d_probe_expectations(rho_packed, c_re, c_im)

    # 2. Explicit complex quadratic form <c_{k,h} | rho_{t,h} | c_{k,h}>
    cr = c_re.unsqueeze(0).unsqueeze(2)  # (1, H, 1, K, 16)
    ci = c_im.unsqueeze(0).unsqueeze(2)
    # rho * |c>: (B, H, T, 16, 16) @ (1, H, 1, 16, K) -> (B, H, T, 16, K)
    rc_re = torch.matmul(outer_re, cr.transpose(-1, -2)) - torch.matmul(outer_im, ci.transpose(-1, -2))
    rc_im = torch.matmul(outer_re, ci.transpose(-1, -2)) + torch.matmul(outer_im, cr.transpose(-1, -2))
    q_explicit_bhtk = (cr.transpose(-1, -2) * rc_re + ci.transpose(-1, -2) * rc_im).sum(dim=-2)  # (B, H, T, K)
    q_explicit_512 = q_explicit_bhtk.transpose(1, 2).reshape(bsz, seq_len, num_h * num_k)

    max_probe_err = float((q_fast_512 - q_explicit_512).abs().max().item())

    # 3. Verify <w_v, q_t> == sum_{h=0}^{15} Tr(rho_{t,h} E_{v,h}) for V_test = 64 tokens
    v_test = 64
    w_vocab_test = torch.randn(v_test, num_h * num_k, device=device, dtype=torch.float32)
    logits_dot = torch.matmul(q_fast_512, w_vocab_test.t())  # (B, T, V_test)

    # Explicitly construct E_{v,h} = sum_{k=0}^{31} w_{v,h,k} |c_{k,h}><c_{k,h}| in Herm(C^{16})
    w_vhk = w_vocab_test.view(v_test, num_h, num_k, 1, 1)
    c_out_re = (c_re.unsqueeze(-1) * c_re.unsqueeze(-2) + c_im.unsqueeze(-1) * c_im.unsqueeze(-2)).unsqueeze(0)
    c_out_im = (c_im.unsqueeze(-1) * c_re.unsqueeze(-2) - c_re.unsqueeze(-1) * c_im.unsqueeze(-2)).unsqueeze(0)
    e_vh_re = (w_vhk * c_out_re).sum(dim=2)  # (V_test, H, 16, 16)
    e_vh_im = (w_vhk * c_out_im).sum(dim=2)

    # For Hermitian A, B: Tr(A B) = sum_{i,j} (A_re_{ij} B_re_{ij} + A_im_{ij} B_im_{ij})
    logits_trace = torch.einsum("bhtij,vhij->btv", outer_re, e_vh_re) + torch.einsum(
        "bhtij,vhij->btv", outer_im, e_vh_im
    )
    max_trace_err = float((logits_dot - logits_trace).abs().max().item())

    # 4. Verify U(16) unitary conjugation preserves trace Tr(U rho U^\dagger) == Tr(rho) and purity Tr((U rho U^\dagger)^2) == Tr(rho^2)
    q_mat, _ = torch.linalg.qr(
        torch.complex(
            torch.randn(num_h, d_h, d_h, device=device, dtype=torch.float32),
            torch.randn(num_h, d_h, d_h, device=device, dtype=torch.float32),
        )
    )
    rot_packed, rot_re, _ = conjugate_packed_hermitian_by_unitary(rho_packed, q_mat.real, q_mat.imag)
    tr_before = torch.diagonal(outer_re, dim1=-2, dim2=-1).sum(dim=-1)
    tr_after = torch.diagonal(rot_re, dim1=-2, dim2=-1).sum(dim=-1)
    pur_before = rho_packed.square().sum(dim=-1)
    pur_after = rot_packed.square().sum(dim=-1)
    max_unitary_tr_err = float((tr_before - tr_after).abs().max().item())
    max_unitary_pur_err = float((pur_before - pur_after).abs().max().item())

    return {
        "max_isometric_probe_expectation_error": max_probe_err,
        "max_tomographic_operator_trace_error": max_trace_err,
        "max_unitary_trace_conservation_error": max_unitary_tr_err,
        "max_unitary_purity_conservation_error": max_unitary_pur_err,
        "passed": bool(
            max_probe_err < 1e-5
            and max_trace_err < 1e-4
            and max_unitary_tr_err < 1e-5
            and max_unitary_pur_err < 1e-5
        ),
    }


def audit_rule9_single_gemm_tomographic_autograd(device: torch.device) -> dict[str, Any]:
    """Audit `_SingleGemm512DTomographicCEFn` under `OpCountingDispatchMode` (Rule 9)."""
    torch.manual_seed(1337)
    bsz, seq_len = 2, 16
    v_sz = 4096
    h_512 = torch.randn(bsz, seq_len, DEFAULT_TOMO_EMBED_DIM, device=device, dtype=torch.float32, requires_grad=True)
    w_vocab = torch.randn(v_sz, DEFAULT_TOMO_EMBED_DIM, device=device, dtype=torch.float32, requires_grad=True)
    b_uni = torch.randn(v_sz, device=device, dtype=torch.float32, requires_grad=True)
    reg_hi = torch.randn(
        bsz, seq_len, DEFAULT_TOMO_NUM_HEADS, DEFAULT_REGISTER_BUCKETS, device=device, dtype=torch.float32, requires_grad=True
    )
    reg_lo = torch.randn(
        bsz, seq_len, DEFAULT_TOMO_NUM_HEADS, DEFAULT_REGISTER_BUCKETS, device=device, dtype=torch.float32, requires_grad=True
    )
    targets = torch.randint(0, v_sz, (bsz, seq_len), device=device, dtype=torch.long)

    report = audit_chunked_nll_autograd_efficiency(
        lambda: _SingleGemm512DTomographicCEFn.apply(h_512, w_vocab, b_uni, reg_hi, reg_lo, targets),
        fn_name="_SingleGemm512DTomographicCEFn",
        num_chunks=1,
        forbid_exponential_ops=False,
    )
    return {
        "fn_name": report.fn_name,
        "fwd_op_counts": report.fwd_op_counts,
        "bwd_op_counts": report.bwd_op_counts,
        "bwd_to_fwd_bmm_ratio": round(report.bwd_to_fwd_bmm_ratio, 3),
        "bwd_to_fwd_pow_ratio": round(report.bwd_to_fwd_pow_ratio, 3),
        "passed": bool(report.passed),
    }


def benchmark_step_and_eval_throughput(device: torch.device) -> dict[str, Any]:
    """Benchmark training step time and projected `1,280`-batch (`10.48M`-token) validation time for `T2` and `T3`."""
    if device.type != "cuda":
        return {"skipped_cuda": True}

    bsz, seq_len = 16, 512
    idx_in = torch.randint(0, 50257, (bsz, seq_len), device=device, dtype=torch.long)
    targets = torch.randint(0, 50257, (bsz, seq_len), device=device, dtype=torch.long)
    out: dict[str, Any] = {}

    for label, mode, use_reg in [
        ("T2_ZERO_BYTEPAIR_512D_TOMO", ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM, False),
        ("T3_TWO_REGISTER_BILINEAR", ExplicitDensityVariantMode.T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE, True),
    ]:
        spec = ExplicitDensityTomographicSpec(
            variant_mode=mode,
            enable_density_highway=True,
            enable_tomographic_readout=True,
            enable_noclick_and_vacuum=True,
            enable_two_register_bilinear=use_reg,
        )
        model = ExplicitDensityTomographicGPT(spec).to(device=device)
        initialize_explicit_density_tomographic_model_(model)

        # Warmup forward+backward
        model.train()
        for _ in range(2):
            loss = model.forward_loss(idx_in, targets)
            loss.backward()
            del loss
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        n_bench = 5
        for _ in range(n_bench):
            loss = model.forward_loss(idx_in, targets)
            loss.backward()
            del loss
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_ms = ((time.perf_counter() - t0) * 1000.0) / float(n_bench)

        # Benchmark eval batch with hoisted w_t and 2,048-token chunking
        model.eval()
        with torch.no_grad():
            w_vocab = torch.nn.functional.rms_norm(
                model.w_det_512.to(torch.bfloat16), (model.embed_dim,), eps=1e-6
            ) * model.radial_scale.to(torch.bfloat16).unsqueeze(1)
            w_t = w_vocab[:50257].t().contiguous()
            for _ in range(2):
                h_512, _, _, r_hi, r_lo = model.build_readout_tensors(idx_in, include_vocab_params=False)
                _ = torch.matmul(h_512.view(-1, 512)[:2048], w_t)
                del h_512, r_hi, r_lo
            torch.cuda.synchronize()

            t_ev0 = time.perf_counter()
            n_eval_batches = 10
            n_pos = bsz * seq_len
            for _ in range(n_eval_batches):
                h_512, _, _, r_hi, r_lo = model.build_readout_tensors(idx_in, include_vocab_params=False)
                h_flat = h_512.view(n_pos, 512)
                for start in range(0, n_pos, 2048):
                    end = min(start + 2048, n_pos)
                    z = torch.matmul(h_flat[start:end], w_t).float()
                    if r_hi is not None and r_lo is not None:
                        k_r = r_hi.size(2)
                        r_hi_c = r_hi.view(n_pos, k_r, 256)[start:end]
                        r_lo_c = r_lo.view(n_pos, k_r, 256)[start:end]
                        grid = torch.bmm(r_hi_c.transpose(1, 2), r_lo_c).float()
                        z.add_(grid.view(end - start, 65536)[:, :50257])
                        del grid
                    _ = torch.logsumexp(z, dim=-1)
                    del z
                del h_512, h_flat, r_hi, r_lo
            torch.cuda.synchronize()
            eval_batch_ms = ((time.perf_counter() - t_ev0) * 1000.0) / float(n_eval_batches)
            del w_vocab, w_t

        projected_1280_eval_sec = (eval_batch_ms * 1280.0) / 1000.0
        out[label] = {
            "train_step_ms": round(step_ms, 2),
            "eval_batch_ms": round(eval_batch_ms, 2),
            "projected_10m_val_sec": round(projected_1280_eval_sec, 2),
            "speedup_vs_wave62_230s_eval": round(230.0 / max(0.1, projected_1280_eval_sec), 2),
        }
        del model
        torch.cuda.empty_cache()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 63 Explicit Density & 512D Tomographic Readout Profiler")
    parser.add_argument(
        "--json-out",
        type=str,
        default=".cache/explicit_density_tomographic_profiler.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identities = audit_isometric_and_tomographic_trace_identities(device)
    autograd_audit = audit_rule9_single_gemm_tomographic_autograd(device)
    throughput = benchmark_step_and_eval_throughput(device)

    report = {
        "plan": "EXPERIMENT_PLAN_083",
        "wave": "W63",
        "device": str(device),
        "identities": identities,
        "rule9_autograd_audit": autograd_audit,
        "throughput": throughput,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
