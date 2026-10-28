"""Unit tests for Track 5 (Backward Deduplication, Fast Validation, & Autograd Efficiency Auditor)
and Kronecker model updates for Tracks 1, 2, 3, and 5 (`Wave 44`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.custom_autograd_efficiency_linter import (
    lint_autograd_source,
    scan_paths_for_autograd_functions,
)
from qdlblocks.benchmarks.autograd_efficiency_auditor import (
    audit_chunked_nll_autograd_efficiency,
)
import qdlblocks.models.calibrated_kronecker_born_gpt as cal_mod
from qdlblocks.models.calibrated_kronecker_born_gpt import (
    CalibratedKroneckerBornChunkedNLLFn,
    CalibratedKroneckerConfig,
    eval_calibrated_kronecker_born_probs,
)
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (
    DEFAULT_REGISTER_DIM,
    VALID_SELF_ECHO_MODES,
    KroneckerMultiMeaningBornChunkedNLLFn,
    KroneckerMultiMeaningConfig,
    _kronecker_chunk_bwd_step_with_prior_grad,
    apply_carrier_orthogonal_complex_self_echo,
    classify_kronecker_model_parameters,
    compute_large_batch_muon_lr_scale,
)
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    NgramCalibratedKroneckerBornGPT,
    NgramCalibratedKroneckerConfig,
    build_ngram_calibrated_kronecker_optimizers,
    evaluate_modded_nanogpt_validation,
)


def _make_normalized_complex_factors(
    shape: tuple[int, ...], seed: int, requires_grad: bool = True,
) -> torch.Tensor:
    """Create unit-sphere complex factor tensor `(..., 2*d_f)` in `float32`."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    raw = torch.randn(shape, generator=gen, dtype=torch.float32)
    d_f = shape[-1] // 2
    re, im = raw[..., :d_f], raw[..., d_f:]
    inv = torch.rsqrt(re.square().sum(dim=-1, keepdim=True) + im.square().sum(dim=-1, keepdim=True) + 1e-8)
    normed = torch.cat([re * inv, im * inv], dim=-1).detach().clone()
    normed.requires_grad_(requires_grad)
    return normed


def test_h5_1_single_pass_backward_bit_exact_and_finite_diff() -> None:
    """Verify `_kronecker_chunk_povm_grad_step` is removed and single-pass backward matches exact reference."""
    assert not hasattr(cal_mod, "_kronecker_chunk_povm_grad_step"), (
        "Legacy double-pass _kronecker_chunk_povm_grad_step must be deleted from calibrated_kronecker_born_gpt."
    )

    n_tok, v_eff, r_num, m_num, d_f = 12, 128, 2, 4, 16
    d_2 = 2 * d_f
    phi_0 = _make_normalized_complex_factors((n_tok, r_num, m_num, d_2), seed=101)
    psi_0 = _make_normalized_complex_factors((v_eff, r_num, m_num, d_2), seed=102)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(103)
    beta_raw = torch.rand((n_tok, r_num), generator=gen, dtype=torch.float32) + 0.2
    beta_0 = (beta_raw * torch.rsqrt(beta_raw.square().sum(dim=-1, keepdim=True))).detach().clone().requires_grad_(True)
    alpha_raw = torch.rand((v_eff, r_num), generator=gen, dtype=torch.float32) + 0.2
    alpha_0 = (alpha_raw * torch.rsqrt(alpha_raw.square().sum(dim=-1, keepdim=True))).detach().clone().requires_grad_(True)
    povm_0 = (torch.rand((v_eff,), generator=gen, dtype=torch.float32) + 0.5).detach().clone().requires_grad_(True)
    targets = torch.randint(0, v_eff, (n_tok,), generator=gen, dtype=torch.int64)
    token_weights = torch.rand((n_tok,), generator=gen, dtype=torch.float32) + 0.25
    hyperparams = torch.tensor([6.0, 0.85, 0.40, 0.35], dtype=torch.float32)

    CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = 32
    loss_chunked = CalibratedKroneckerBornChunkedNLLFn.apply(
        phi_0, psi_0, beta_0, alpha_0, povm_0, targets, hyperparams, token_weights
    )
    loss_chunked.backward()

    phi_ref = phi_0.detach().clone().requires_grad_(True)
    psi_ref = psi_0.detach().clone().requires_grad_(True)
    beta_ref = beta_0.detach().clone().requires_grad_(True)
    alpha_ref = alpha_0.detach().clone().requires_grad_(True)
    povm_ref = povm_0.detach().clone().requires_grad_(True)

    probs_ref = eval_calibrated_kronecker_born_probs(
        phi_ref.view(3, 4, r_num, m_num, d_2),
        psi_ref,
        beta_ref.view(3, 4, r_num),
        alpha_ref,
        povm_ref,
        hyperparams,
    )
    nll_ref = -torch.log(probs_ref.reshape(n_tok, v_eff).gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12))
    w_norm = token_weights / token_weights.sum()
    loss_ref = (w_norm * nll_ref).sum()
    loss_ref.backward()

    assert abs(float(loss_chunked.item()) - float(loss_ref.item())) < 1e-5
    for name, g_c, g_r in (
        ("phi_factors", phi_0.grad, phi_ref.grad),
        ("psi_factors", psi_0.grad, psi_ref.grad),
        ("branch_beta", beta_0.grad, beta_ref.grad),
        ("branch_alpha", alpha_0.grad, alpha_ref.grad),
        ("povm_weights", povm_0.grad, povm_ref.grad),
    ):
        assert g_c is not None and g_r is not None, f"Missing gradient for {name}"
        assert torch.isfinite(g_c).all(), f"Non-finite gradient for {name}"
        max_diff = float((g_c - g_r).abs().max().item())
        assert max_diff < 1e-5, f"{name} grad max_abs_diff={max_diff} >= 1e-5"


def test_h5_2_dispatch_mode_op_count_certification() -> None:
    """Verify `OpCountingDispatchMode` certifies single-pass backward (`pow` ratio `1.0`) and catches double-pass."""
    n_tok, v_eff, r_num, m_num, d_f = 8, 256, 2, 4, 16
    d_2 = 2 * d_f
    v_chunk = 64
    num_chunks = v_eff // v_chunk
    hyperparams = torch.tensor([6.0, 0.85, 0.40, 0.35], dtype=torch.float32)
    targets = torch.arange(n_tok, dtype=torch.int64) % v_eff

    def _run_calibrated() -> torch.Tensor:
        phi = _make_normalized_complex_factors((n_tok, r_num, m_num, d_2), seed=201)
        psi = _make_normalized_complex_factors((v_eff, r_num, m_num, d_2), seed=202)
        beta = torch.full((n_tok, r_num), 1.0 / (r_num ** 0.5), requires_grad=True)
        alpha = torch.full((v_eff, r_num), 1.0 / (r_num ** 0.5), requires_grad=True)
        povm = torch.ones((v_eff,), dtype=torch.float32, requires_grad=True)
        CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = v_chunk
        return CalibratedKroneckerBornChunkedNLLFn.apply(
            phi, psi, beta, alpha, povm, targets, hyperparams, None
        )

    rep_cal = audit_chunked_nll_autograd_efficiency(
        _run_calibrated, fn_name="CalibratedKroneckerBornChunkedNLLFn", num_chunks=num_chunks
    )
    assert rep_cal.passed is True
    assert rep_cal.fwd_op_counts["aten.bmm"] == num_chunks
    assert rep_cal.bwd_op_counts["aten.bmm"] == 3 * num_chunks
    assert rep_cal.fwd_op_counts["aten.pow"] == num_chunks
    assert rep_cal.bwd_op_counts["aten.pow"] == num_chunks
    assert rep_cal.bwd_to_fwd_pow_ratio == 1.0
    assert rep_cal.forbidden_ops_detected == ()

    def _run_multimeaning() -> torch.Tensor:
        phi = _make_normalized_complex_factors((n_tok, r_num, m_num, d_2), seed=203)
        psi = _make_normalized_complex_factors((v_eff, r_num, m_num, d_2), seed=204)
        beta = torch.full((n_tok, r_num), 1.0 / (r_num ** 0.5), requires_grad=True)
        alpha = torch.full((v_eff, r_num), 1.0 / (r_num ** 0.5), requires_grad=True)
        prior = torch.ones((v_eff,), dtype=torch.float32)
        KroneckerMultiMeaningBornChunkedNLLFn.vocab_chunk_size = v_chunk
        return KroneckerMultiMeaningBornChunkedNLLFn.apply(
            phi, psi, beta, alpha, targets, prior, hyperparams, None
        )

    rep_mm = audit_chunked_nll_autograd_efficiency(
        _run_multimeaning, fn_name="KroneckerMultiMeaningBornChunkedNLLFn", num_chunks=num_chunks
    )
    assert rep_mm.passed is True
    assert rep_mm.bwd_to_fwd_pow_ratio == 1.0
    assert rep_mm.forbidden_ops_detected == ()

    class _SyntheticDoublePassChunkedFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(u, w)  # type: ignore[attr-defined]
            z = torch.bmm(u, w)
            p = torch.pow(z.square() + 1e-6, 4.0)
            return p.sum()

        @staticmethod
        def backward(ctx: object, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            u, w = ctx.saved_tensors  # type: ignore[attr-defined]
            z = torch.bmm(u, w)
            g1 = torch.pow(z.square() + 1e-6, 3.0) * z * grad_out
            g2 = torch.pow(z.square() + 1e-6, 3.0) * z * grad_out
            du = torch.bmm(g1, w.transpose(1, 2))
            dw = torch.bmm(u.transpose(1, 2), g2)
            return du, dw

    def _run_bad_double_pass() -> torch.Tensor:
        u = torch.randn((4, 8, 16), requires_grad=True)
        w = torch.randn((4, 16, 32), requires_grad=True)
        return _SyntheticDoublePassChunkedFn.apply(u, w)

    rep_bad = audit_chunked_nll_autograd_efficiency(
        _run_bad_double_pass, fn_name="_SyntheticDoublePassChunkedFn", num_chunks=1
    )
    assert rep_bad.passed is False
    assert rep_bad.bwd_to_fwd_pow_ratio == 2.0


def test_h5_3_static_ast_linter_catches_bad_and_passes_codebase() -> None:
    """Verify `custom_autograd_efficiency_linter.py` catches R1–R4 on synthetic code and passes codebase."""
    bad_source = '''
import torch

class BadKroneckerBornFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, d, e, f, g, h, i, j, k):
        ctx.save_for_backward(a, b, c, d, e, f, g, h, i, j, k)
        for v0 in range(0, 128, 32):
            x = torch.exp(a)
        for v0 in range(0, 128, 32):
            y = b + 1
        return x.sum() + y.sum()

    @staticmethod
    def backward(ctx, grad_out):
        for v0 in range(0, 128, 32):
            z_cat = grad_out
            g1 = _kronecker_chunk_bwd_step(z_cat, grad_out)
            g2 = _kronecker_chunk_povm_grad_step(z_cat, grad_out)
        return (grad_out,) * 11
'''
    audits_bad, violations_bad = lint_autograd_source(bad_source, file_path="bad_kronecker.py")
    assert len(audits_bad) == 1
    v_joined = " | ".join(violations_bad)
    assert "R1(DuplicateChunkStepCall)" in v_joined
    assert "R2(RedundantChunkLoops)" in v_joined
    assert "R3(ForbiddenExponentialOp)" in v_joined
    assert "R4(OversizedSaveForBackward)" in v_joined

    audits_real, violations_real = scan_paths_for_autograd_functions(
        ["src/qdlblocks/models", "src/qdlblocks/layers"]
    )
    assert len(audits_real) >= 12
    assert violations_real == [], f"Unexpected autograd AST violations: {violations_real}"


def test_h5_4_fast_validation_parity_and_track3_self_echo_metrics() -> None:
    """Verify carrier-orthogonal self-echo, `d_register=16`, optimizer routing, and fast chunked validation parity."""
    cfg_base = KroneckerMultiMeaningConfig()
    assert cfg_base.d_register == DEFAULT_REGISTER_DIM == 16
    assert cfg_base.register_dim == 16

    n_tok, r_num, m_num, d_f = 10, 2, 4, 16
    gen = torch.Generator(device="cpu")
    gen.manual_seed(301)
    op_re = torch.randn((n_tok, r_num, m_num, d_f), generator=gen, dtype=torch.float32)
    op_im = torch.randn((n_tok, r_num, m_num, d_f), generator=gen, dtype=torch.float32)
    p0_re = torch.randn((n_tok, r_num, m_num, d_f), generator=gen, dtype=torch.float32)
    p0_im = torch.randn((n_tok, r_num, m_num, d_f), generator=gen, dtype=torch.float32)

    for mode in VALID_SELF_ECHO_MODES:
        gate_logits = torch.zeros((r_num, m_num), dtype=torch.float32, requires_grad=True)
        out_re, out_im, _ = apply_carrier_orthogonal_complex_self_echo(
            op_re, op_im, p0_re, p0_im, mode=mode, echo_gate_logits=gate_logits
        )
        if mode in ("none", "ortho_real_only", "ortho_complex_learned"):
            assert torch.allclose(out_re[..., :1], op_re[..., :1], atol=1e-6)
            assert torch.allclose(out_im[..., :1], op_im[..., :1], atol=1e-6)
        if mode == "ortho_complex_learned":
            loss_echo = (out_re.square() + out_im.square()).sum()
            loss_echo.backward()
            assert gate_logits.grad is not None and float(gate_logits.grad.abs().sum().item()) > 0.0

    ng_cfg = NgramCalibratedKroneckerConfig(
        vocab_size=256,
        valid_vocab_size=250,
        num_layers=2,
        n_registers=2,
        d_register=16,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        direct_bigram_slots=64,
        hashed_trigram_buckets=128,
        vocab_chunk_size=128,
    )
    model = NgramCalibratedKroneckerBornGPT(ng_cfg)
    assert model.echo_gate_logits is not None
    assert tuple(model.echo_gate_logits.shape) == (2, 4)

    part = classify_kronecker_model_parameters(model)
    assert len(part.unclassified_names) == 0, f"Unclassified parameters: {part.unclassified_names}"
    assert any(p.shape[-1] == 2 for p in part.muon_params), "Expected 2x2 branch_mixers in StiefelMuon group"
    assert abs(compute_large_batch_muon_lr_scale(32768, 8192) - 2.0) < 1e-6

    opt_muon, opt_adam = build_ngram_calibrated_kronecker_optimizers(model, batch_tokens=16384)
    assert len(opt_muon.param_groups) == 1
    assert len(opt_adam.param_groups) == 3

    val_stream = torch.randint(0, 250, (513,), generator=gen, dtype=torch.int64)
    val_stream[10] = val_stream[9]
    val_stream[25] = val_stream[24]

    res_small_chunk = evaluate_modded_nanogpt_validation(
        model, val_stream, max_eval_tokens=512, batch_size=4, seq_len=64, compute_top50=True, eval_vocab_chunk_size=256
    )
    res_large_chunk = evaluate_modded_nanogpt_validation(
        model, val_stream, max_eval_tokens=512, batch_size=4, seq_len=64, compute_top50=True, eval_vocab_chunk_size=8192
    )
    assert abs(res_small_chunk["val_ce"] - res_large_chunk["val_ce"]) < 1e-4
    assert res_small_chunk["top1_acc"] == res_large_chunk["top1_acc"]
    assert res_small_chunk["top50_acc"] == res_large_chunk["top50_acc"]
    for key in ("rep_val_ce", "non_rep_val_ce", "mean_self_echo_prob", "empirical_rep_rate", "rep_top1_acc"):
        assert key in res_large_chunk, f"Missing Track 3 repetition metric: {key}"
        assert abs(res_small_chunk[key] - res_large_chunk[key]) < 1e-4
