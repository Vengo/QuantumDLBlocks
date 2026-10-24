"""Unit, Mathematical, and Rule-6 AST Compliance Tests for Wave 28 (`EXPERIMENT_PLAN_043.md`)."""

from __future__ import annotations

import ast
from dataclasses import fields
import json
import math
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdlblocks.functional import l2_normalize  # noqa: E402
from qdlblocks.layers.parseval_mup_chunked_ce import (  # noqa: E402
    ParsevalChunkedCEReadoutSpec,
    eval_parseval_mup_full_logits,
    parseval_mup_dyadic_chunked_ce,
)
from qdlblocks.models.parseval_mup_dyadic_gpt import (  # noqa: E402
    ParsevalMuPDyadicConfig,
    ParsevalMuPDyadicGPT,
    build_parseval_mup_optimizers,
    initialize_parseval_mup_with_ppmi_svd,
)
from scripts.ast_signature_and_length_linter import LinterConfig, audit_python_file  # noqa: E402
from scripts.smith_clr_range_test_vs_wsd_profiler import compute_schedule_lr_and_momentum  # noqa: E402


def test_rule6_ast_flag_free_and_no_clamps() -> None:
    """Verify Rule-6 flag-free AST invariants on `parseval_mup_dyadic_gpt.py` and `parseval_mup_chunked_ce.py`."""
    model_path = ROOT / "src" / "qdlblocks" / "models" / "parseval_mup_dyadic_gpt.py"
    loss_path = ROOT / "src" / "qdlblocks" / "layers" / "parseval_mup_chunked_ce.py"

    for fpath in (model_path, loss_path):
        src = fpath.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(fpath))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert "PurityGate" not in names
        assert "ContextualBlochPurityGate" not in names
        assert "purity_gate" not in names
        assert "kboson_raw_k" not in names
        assert "k_copies" not in names
        assert "clamp" not in names
        assert "clamp_" not in names

    for cfg_cls in (ParsevalMuPDyadicConfig, ParsevalChunkedCEReadoutSpec):
        for f in fields(cfg_cls):
            assert not f.name.startswith("use_"), f"Forbidden boolean flag {f.name} on {cfg_cls.__name__}"
            assert not f.name.startswith("is_"), f"Forbidden boolean flag {f.name} on {cfg_cls.__name__}"
            assert f.type not in (bool, "bool"), f"Forbidden bool field {f.name} on {cfg_cls.__name__}"


def test_ast_linter_compliance_on_wave28_files() -> None:
    """Verify all new Wave 28 modules and scripts pass `scripts/ast_signature_and_length_linter.py`."""
    targets = [
        ROOT / "src" / "qdlblocks" / "kernels" / "fused_parseval_ce.py",
        ROOT / "src" / "qdlblocks" / "layers" / "parseval_mup_chunked_ce.py",
        ROOT / "src" / "qdlblocks" / "models" / "parseval_mup_dyadic_gpt.py",
        ROOT / "scripts" / "smith_clr_range_test_vs_wsd_profiler.py",
        ROOT / "examples" / "53_verify_plan043_parseval_sqrt_d_noclamp_readout.py",
    ]
    for path in targets:
        cfg = LinterConfig(paths=(str(path),), max_lines=1000, hard_ceiling_lines=2500, max_args=8, max_nested_lines=40)
        report = audit_python_file(str(path), cfg)
        assert report["passed"], f"{path.name} failed AST linter: {report['violations']}"


def test_parseval_sqrt_d_scaling_and_proportional_wavelet_ranks() -> None:
    """Verify `beta_0(d) = c* sqrt(d)` and proportional wavelet ranks `(d // 2, d // 4)` across widths."""
    for embed_dim in (128, 256, 384, 512):
        cfg = ParsevalMuPDyadicConfig(
            vocab_size=256,
            valid_vocab_size=250,
            block_size=32,
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=2,
            window_size=32,
            ambient_n=1024,
            mup_c_star=1.48,
        )
        model = ParsevalMuPDyadicGPT(cfg)
        expected_beta = 1.48 * math.sqrt(float(embed_dim))
        assert math.isclose(model.beta_0, expected_beta, rel_tol=1e-6)
        assert model.mid_detail_dim == embed_dim // 2
        assert model.fine_detail_dim == embed_dim // 4

        model.set_training_progress(0.0)
        assert math.isclose(model.current_beta(), 0.95 * expected_beta, rel_tol=1e-6)
        assert math.isclose(model.current_lambda_diff(), 0.20, rel_tol=1e-6)
        assert math.isclose(model.current_lambda_diff_scaled(), 0.95 * expected_beta * 0.20, rel_tol=1e-6)

        model.set_training_progress(1.0)
        assert math.isclose(model.current_beta(), 1.03 * expected_beta, rel_tol=1e-6)
        assert math.isclose(model.current_lambda_diff(), 0.05, rel_tol=1e-6)
        assert math.isclose(model.current_lambda_diff_scaled(), 1.03 * expected_beta * 0.05, rel_tol=1e-6)


def test_parseval_chunked_ce_matches_unchunked_logits_and_gradients() -> None:
    """Verify forward loss and backward gradients of `parseval_mup_dyadic_chunked_ce` vs full logits."""
    torch.manual_seed(42)
    b, t, d, v_tot, v_val = 2, 16, 32, 128, 120
    psi_s = l2_normalize(torch.randn(b, t, d, dtype=torch.float32), dim=-1).requires_grad_(True)
    psi_d = l2_normalize(torch.randn(b, t, d, dtype=torch.float32), dim=-1).requires_grad_(True)
    m_s = l2_normalize(torch.randn(v_tot, d, dtype=torch.float32), dim=-1).requires_grad_(True)
    m_d = l2_normalize(torch.randn(v_tot, d, dtype=torch.float32), dim=-1).requires_grad_(True)
    prior = (0.1 * torch.randn(v_tot, dtype=torch.float32)).requires_grad_(True)
    targets = torch.randint(0, v_val, (b, t), dtype=torch.long)
    targets[0, -1] = -100
    v_to_coarse = torch.arange(v_val, dtype=torch.long) % 16

    spec = ParsevalChunkedCEReadoutSpec(
        beta_0=1.48 * math.sqrt(float(d)),
        lambda_diff_scaled=1.48 * math.sqrt(float(d)) * 0.15,
        traceless_gamma=0.25,
        dyadic_coarse_weight=0.14,
        chunk_size=12,
        valid_vocab_size=v_val,
        ignore_index=-100,
        gemm_precision="fp32",
        num_coarse_buckets=16,
        v_to_coarse=v_to_coarse,
    )

    loss_chunked = parseval_mup_dyadic_chunked_ce(psi_s, psi_d, m_s, m_d, targets, spec, prior)
    grads_chunked = torch.autograd.grad(loss_chunked, (psi_s, psi_d, m_s, m_d, prior))

    psi_s2 = psi_s.detach().clone().requires_grad_(True)
    psi_d2 = psi_d.detach().clone().requires_grad_(True)
    m_s2 = m_s.detach().clone().requires_grad_(True)
    m_d2 = m_d.detach().clone().requires_grad_(True)
    prior2 = prior.detach().clone().requires_grad_(True)

    full_logits = eval_parseval_mup_full_logits(psi_s2, psi_d2, m_s2, m_d2, spec, prior2)
    loss_ref = F.cross_entropy(full_logits[..., :v_val].reshape(-1, v_val), targets.reshape(-1), ignore_index=-100)
    grads_ref = torch.autograd.grad(loss_ref, (psi_s2, psi_d2, m_s2, m_d2, prior2))

    assert float((loss_chunked - loss_ref).abs().item()) < 1e-5
    for g_c, g_r in zip(grads_chunked, grads_ref):
        assert float((g_c - g_r).abs().max().item()) < 1e-4


def test_post_norm_cholesky_metric_temperature_absorption_equivalence() -> None:
    """Verify `absorb_beta_into_readout_cholesky()` preserves exact logits and loss (`< 1e-4`)."""
    torch.manual_seed(99)
    cfg = ParsevalMuPDyadicConfig(
        vocab_size=256,
        valid_vocab_size=240,
        block_size=16,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        window_size=16,
        ambient_n=1024,
        readout_gemm_precision="fp32",
    )
    model = ParsevalMuPDyadicGPT(cfg)
    with torch.no_grad():
        model.pyramid.wav_256.normal_(0.0, 0.2)
        model.pyramid.wav_2048.normal_(0.0, 0.2)
        model.pyramid.wav_8192.normal_(0.0, 0.2)
        model.pyramid.wav_65536.normal_(0.0, 0.2)
        model.pyramid.readout_chol.raw_diag.normal_(0.0, 0.2)
        model.pyramid.readout_chol.raw_tril.normal_(0.0, 0.1)
    model.eval()

    idx = torch.randint(0, 240, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 240, (2, 16), dtype=torch.long)
    with torch.no_grad():
        logits_before, loss_before = model(idx, targets, return_logits=True)
        info = model.absorb_beta_into_readout_cholesky()
        logits_after, loss_after = model(idx, targets, return_logits=True)

    assert logits_before is not None and logits_after is not None
    assert loss_before is not None and loss_after is not None
    assert math.isclose(info["beta_0"], model.beta_0, rel_tol=1e-6)
    assert float((logits_before - logits_after).abs().max().item()) < 1e-4
    assert float((loss_before - loss_after).abs().item()) < 1e-5


def test_spectral_mup_optimizer_width_scaling() -> None:
    """Verify `build_parseval_mup_optimizers` scales LRs by `sqrt(256 / d)` across widths."""
    lrs_by_d: dict[int, tuple[float, float, float]] = {}
    for d in (128, 256, 384, 512):
        cfg = ParsevalMuPDyadicConfig(
            vocab_size=256, valid_vocab_size=256, block_size=16, embed_dim=d, num_heads=4, num_layers=2, ambient_n=1024
        )
        model = ParsevalMuPDyadicGPT(cfg)
        muon_opt, adam_opt = build_parseval_mup_optimizers(model, base_muon_lr=0.026, base_wav_lr=0.011, base_head_lr=0.008)
        m_lr = float(muon_opt.param_groups[0]["lr"])
        w_lr = float(next(g["lr"] for g in adam_opt.param_groups if g["group_name"] == "wavelet_2pow16"))
        h_lr = float(next(g["lr"] for g in adam_opt.param_groups if g["group_name"] == "cholesky_readout"))
        lrs_by_d[d] = (m_lr, w_lr, h_lr)

        scale = math.sqrt(256.0 / float(d))
        assert math.isclose(m_lr, 0.026 * scale, rel_tol=1e-6)
        assert math.isclose(w_lr, 0.011 * scale, rel_tol=1e-6)
        assert math.isclose(h_lr, 0.008 * scale, rel_tol=1e-6)

    assert lrs_by_d[512][0] < lrs_by_d[384][0] < lrs_by_d[256][0] < lrs_by_d[128][0]


def test_smith_clr_schedules_and_diagnostic_artifacts() -> None:
    """Verify Smith CLR schedule boundary properties and `.cache/` diagnostic JSON artifacts."""
    eta_base, eta_max = 0.008, 0.024
    lr_start, _ = compute_schedule_lr_and_momentum("smith_triangular2", 0, 48, eta_base, eta_max)
    lr_mid, _ = compute_schedule_lr_and_momentum("smith_triangular2", 24, 48, eta_base, eta_max)
    assert math.isclose(lr_start, eta_base, rel_tol=1e-5)
    assert math.isclose(lr_mid, eta_base, rel_tol=1e-5)

    p1 = json.loads((ROOT / ".cache" / "parseval_sqrt_d_scale_invariance_probe.json").read_text(encoding="utf-8"))
    p2 = json.loads((ROOT / ".cache" / "unit_sphere_mup_transfer_grid_search.json").read_text(encoding="utf-8"))
    p3 = json.loads((ROOT / ".cache" / "cholesky_metric_temperature_absorber.json").read_text(encoding="utf-8"))
    p4 = json.loads((ROOT / ".cache" / "smith_clr_range_test_vs_wsd_profiler.json").read_text(encoding="utf-8"))

    s1 = p1["summary_metrics"]
    s2 = p2["summary_metrics"]
    s3 = p3["summary_metrics"]

    assert s1["bg_logit_std_cv_parseval"] < 0.015
    assert s1["algebraic_unification_max_err"] < 1e-6
    assert s2["mup_zero_shot_transfer_regret_nats"] <= 0.01
    assert s3["max_logit_abs_err"] < 1e-4

    rt = p4["range_test"]
    assert 0.004 <= rt["eta_base"] <= 0.015
    assert 0.020 <= rt["eta_max"] <= 0.045
    assert 2.5 <= rt["eta_max_over_eta_base"] <= 6.0
    assert p4["comparisons"]["triangular2_beats_wsd_at_tau_0p50"] is True
    assert p4["comparisons"]["final_within_0p05_nats"] is True
    assert p4["comparisons"]["all_kraus_stiefel_err_below_0p02"] is True


class TestPlan043ParsevalMuPReadout(unittest.TestCase):
    """Unittest wrapper so tests run with both `pytest` and `python -m unittest`."""

    def test_all(self) -> None:
        test_rule6_ast_flag_free_and_no_clamps()
        test_ast_linter_compliance_on_wave28_files()
        test_parseval_sqrt_d_scaling_and_proportional_wavelet_ranks()
        test_parseval_chunked_ce_matches_unchunked_logits_and_gradients()
        test_post_norm_cholesky_metric_temperature_absorption_equivalence()
        test_spectral_mup_optimizer_width_scaling()
        test_smith_clr_schedules_and_diagnostic_artifacts()


if __name__ == "__main__":
    unittest.main()
