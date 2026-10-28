"""Unit tests for Track 1 (Joint-Complex StiefelMuon) and Track 2 (Universal Complex Muon Routing).

Provenance (`Wave 44` / `EXPERIMENT_PLAN_059.md`):
- Verifies `complex_polar_express_orth`, `zeropower_via_polarexpress5_complex_2d`,
  `zeropower_via_polarexpress5_complex_batched`, `normuon_equalize_complex`, and
  `retract_complex_stiefel_pair` in `src/qdlblocks/functional.py`.
- Verifies `StiefelMuon` (`is_complex_pair_group=True`), `StiefelMuonConfig`,
  `retract_complex_kraus_generators`, `compute_large_batch_muon_lr_scale`,
  `collect_named_complex_muon_pairs`, and `classify_kronecker_model_parameters`
  in `src/qdlblocks/optim/muon.py`.
- Verifies `build_gemini_complex_optimizers` in `src/qdlblocks/models/gemini_packed_complex_rope_gpt.py`.
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from qdlblocks.functional import (
    complex_polar_express_orth,
    retract_complex_stiefel_pair,
    zeropower_via_polarexpress5,
    zeropower_via_polarexpress5_complex_2d,
    zeropower_via_polarexpress5_complex_batched,
)
from qdlblocks.models.calibrated_kronecker_born_gpt import (
    CalibratedKroneckerBornGPT,
    CalibratedKroneckerConfig,
)
from qdlblocks.models.gemini_packed_complex_rope_gpt import (
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
    build_gemini_complex_optimizers,
)
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (
    KroneckerMultiMeaning16x4GPT,
    KroneckerMultiMeaningConfig,
)
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    NgramCalibratedKroneckerBornGPT,
    NgramCalibratedKroneckerConfig,
)
from qdlblocks.optim.muon import (
    DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
    MAX_MUON_LR_SCALE,
    MIN_MUON_LR_SCALE,
    StiefelMuon,
    StiefelMuonConfig,
    classify_kronecker_model_parameters,
    compute_large_batch_muon_lr_scale,
    retract_complex_kraus_generators,
)

# ==============================================================================
# Top-of-File Named Test Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
TEST_SEED: int = 42
UNITARITY_ERR_CEILING: float = 0.15
RECT_UNITARITY_ERR_CEILING: float = 0.05
SPLIT_REAL_ERR_FLOOR: float = 0.90
BF16_PHASE_EQUIV_TOL: float = 0.02
FP32_PHASE_EQUIV_TOL: float = 1e-5
IMAG_NOISE_SCALE: float = 1e-5
IMAG_NOISE_RATIO_CEILING: float = 1e-4
LARGE_BATCH_COSINE_GAIN_FLOOR: float = 0.10


def _complex_semi_unitarity_error(u_re: torch.Tensor, u_im: torch.Tensor) -> float:
    """Compute normalized semi-unitarity error `||U^H U - I||_F / sqrt(k)` or `||U U^H - I||_F / sqrt(k)`."""
    ur = u_re.float()
    ui = u_im.float()
    m, n = ur.size(-2), ur.size(-1)
    if m >= n:
        gram_re = ur.transpose(-2, -1) @ ur + ui.transpose(-2, -1) @ ui
        gram_im = ur.transpose(-2, -1) @ ui - ui.transpose(-2, -1) @ ur
        k = n
    else:
        gram_re = ur @ ur.transpose(-2, -1) + ui @ ui.transpose(-2, -1)
        gram_im = ui @ ur.transpose(-2, -1) - ur @ ui.transpose(-2, -1)
        k = m
    eye = torch.eye(k, device=ur.device, dtype=torch.float32)
    diff_re = gram_re - eye
    err = torch.sqrt((diff_re.square() + gram_im.square()).sum(dim=(-2, -1))) / math.sqrt(float(k))
    return float(err.mean().item())


def _complex_cos(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> float:
    """Compute real Frobenius cosine similarity between two complex matrices."""
    ar, ai, br, bi = a_re.float(), a_im.float(), b_re.float(), b_im.float()
    num = (ar * br + ai * bi).sum()
    den = torch.sqrt((ar.square() + ai.square()).sum() * (br.square() + bi.square()).sum())
    return float((num / den).item())


class _ToyComplexKrausModule(nn.Module):
    """Toy module with 3D paired complex Kraus generator parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_re = nn.Parameter(torch.randn(4, 16, 16) * 0.4)
        self.proj_im = nn.Parameter(torch.randn(4, 16, 16) * 0.4)


class _MockFlagshipKronecker(nn.Module):
    """Mock Wave 44 flagship Kronecker module covering all parameter classification buckets."""

    def __init__(self, r_branches: int) -> None:
        super().__init__()
        self.embed_re = nn.Parameter(torch.randn(128, 64))
        self.embed_im = nn.Parameter(torch.randn(128, 64))
        self.bi_ctx_wave_re = nn.Parameter(torch.randn(64, 64))
        self.bi_ctx_wave_im = nn.Parameter(torch.randn(64, 64))
        self.fourgram_ctx_wave_re = nn.Parameter(torch.randn(64, 64))
        self.fourgram_ctx_wave_im = nn.Parameter(torch.randn(64, 64))
        self.povm_scale_logits = nn.Parameter(torch.zeros(128, 3))
        self.router_w_re = nn.Parameter(torch.randn(max(2, r_branches), 64))
        self.router_w_im = nn.Parameter(torch.randn(max(2, r_branches), 64))
        self.echo_gate_logits = nn.Parameter(torch.zeros(4, 2))
        self.gram_eta = nn.Parameter(torch.zeros(4))
        self.meaning_mix_re = nn.Parameter(torch.eye(r_branches))
        self.meaning_mix_im = nn.Parameter(torch.zeros(r_branches, r_branches))
        self.stinespring_w_re = nn.Parameter(torch.randn(4, r_branches, r_branches))
        self.stinespring_w_im = nn.Parameter(torch.randn(4, r_branches, r_branches))
        self.factor_u_re = nn.Parameter(torch.randn(4 * r_branches, 16, 16))
        self.factor_u_im = nn.Parameter(torch.randn(4 * r_branches, 16, 16))
        self.out_proj_re = nn.Parameter(torch.randn(64, 64))
        self.out_proj_im = nn.Parameter(torch.randn(64, 64))


def test_joint_complex_polar_express_unitarity_across_shapes() -> None:
    """Verify complex semi-unitarity error `< 0.15` (`< 0.05` on rectangular/16x16) vs `> 0.90` for split-real."""
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((64, 64), (128, 64), (64, 128), (8, 16, 16)):
            torch.manual_seed(TEST_SEED)
            g_re = torch.randn(shape, dtype=dtype)
            g_im = torch.randn(shape, dtype=dtype)
            tol = UNITARITY_ERR_CEILING if shape == (64, 64) else RECT_UNITARITY_ERR_CEILING

            u_re, u_im = complex_polar_express_orth(g_re, g_im, steps=5, compute_dtype=dtype)
            assert u_re.dtype == dtype
            assert u_im.dtype == dtype
            err_joint = _complex_semi_unitarity_error(u_re, u_im)
            assert err_joint < tol

            if len(shape) == 2:
                u2_re, u2_im = zeropower_via_polarexpress5_complex_2d(g_re, g_im, steps=5)
                assert _complex_semi_unitarity_error(u2_re, u2_im) < tol
                s_re = zeropower_via_polarexpress5(g_re.float(), steps=5)
                s_im = zeropower_via_polarexpress5(g_im.float(), steps=5)
                err_split = _complex_semi_unitarity_error(s_re, s_im)
                assert err_split > SPLIT_REAL_ERR_FLOOR
            else:
                ub_re, ub_im = zeropower_via_polarexpress5_complex_batched(g_re, g_im, steps=5)
                assert _complex_semi_unitarity_error(ub_re, ub_im) < tol


def test_joint_complex_polar_express_u1_phase_equivariance() -> None:
    """Verify `Polar_C(e^{i theta} G) == e^{i theta} Polar_C(G)` in both `float32` and `bfloat16`."""
    for theta in (0.37, math.pi / 4.0, 1.23):
        torch.manual_seed(TEST_SEED)
        g_re = torch.randn(32, 32, dtype=torch.float32)
        g_im = torch.randn(32, 32, dtype=torch.float32)
        c_th = math.cos(theta)
        s_th = math.sin(theta)
        rot_g_re = c_th * g_re - s_th * g_im
        rot_g_im = s_th * g_re + c_th * g_im

        # 1. float32 equivariance (< 1e-5)
        u_re_32, u_im_32 = complex_polar_express_orth(g_re, g_im, steps=5, compute_dtype=torch.float32)
        ru_re_32, ru_im_32 = complex_polar_express_orth(rot_g_re, rot_g_im, steps=5, compute_dtype=torch.float32)
        exp_re_32 = c_th * u_re_32 - s_th * u_im_32
        exp_im_32 = s_th * u_re_32 + c_th * u_im_32
        num_32 = torch.sqrt((ru_re_32 - exp_re_32).square().sum() + (ru_im_32 - exp_im_32).square().sum())
        den_32 = torch.sqrt(u_re_32.square().sum() + u_im_32.square().sum())
        assert float((num_32 / den_32).item()) < FP32_PHASE_EQUIV_TOL

        # 2. bfloat16 equivariance (< 0.02)
        u_re_16, u_im_16 = complex_polar_express_orth(
            g_re, g_im, steps=5, compute_dtype=torch.bfloat16
        )
        ru_re_16, ru_im_16 = complex_polar_express_orth(
            rot_g_re, rot_g_im, steps=5, compute_dtype=torch.bfloat16
        )
        exp_re_16 = c_th * u_re_16.float() - s_th * u_im_16.float()
        exp_im_16 = s_th * u_re_16.float() + c_th * u_im_16.float()
        num_16 = torch.sqrt(
            (ru_re_16.float() - exp_re_16).square().sum() + (ru_im_16.float() - exp_im_16).square().sum()
        )
        den_16 = torch.sqrt(u_re_16.float().square().sum() + u_im_16.float().square().sum())
        assert float((num_16 / den_16).item()) < BF16_PHASE_EQUIV_TOL


def test_joint_complex_polar_express_suppresses_imaginary_noise_at_real_init() -> None:
    """Verify `||G_re||_F = 1.0, ||G_im||_F = 1e-5` yields `||U_im||_F / ||U_re||_F < 1e-4`."""
    torch.manual_seed(TEST_SEED)
    g_re = torch.randn(32, 32, dtype=torch.float32)
    g_re = g_re / g_re.norm()
    g_im = torch.randn(32, 32, dtype=torch.float32)
    g_im = IMAG_NOISE_SCALE * (g_im / g_im.norm())

    u_re, u_im = complex_polar_express_orth(g_re, g_im, steps=5, compute_dtype=torch.bfloat16)
    ratio = float(u_im.float().norm().item() / u_re.float().norm().item())
    assert ratio < IMAG_NOISE_RATIO_CEILING


def test_retract_complex_stiefel_pair_and_kraus_generators() -> None:
    """Verify `retract_complex_stiefel_pair` and `retract_complex_kraus_generators` reduce unitary error."""
    torch.manual_seed(TEST_SEED)
    for shape in [(32, 16), (16, 32), (4, 16, 16)]:
        u_re, u_im = complex_polar_express_orth(
            torch.randn(shape), torch.randn(shape), steps=5, compute_dtype=torch.float32
        )
        w_re = nn.Parameter(u_re + 0.15 * torch.randn(shape))
        w_im = nn.Parameter(u_im + 0.15 * torch.randn(shape))
        err_before = _complex_semi_unitarity_error(w_re, w_im)
        retract_complex_stiefel_pair(w_re, w_im, penalty=0.25)
        err_after = _complex_semi_unitarity_error(w_re, w_im)
        assert err_after < err_before

    toy = _ToyComplexKrausModule()
    err0 = _complex_semi_unitarity_error(toy.proj_re, toy.proj_im)
    n_retracted = retract_complex_kraus_generators(toy, stiefel_penalty=0.20)
    err1 = _complex_semi_unitarity_error(toy.proj_re, toy.proj_im)
    assert n_retracted == 1
    assert err1 < err0


def test_stiefel_muon_complex_pair_group_step_and_large_batch_cosine() -> None:
    """Verify `StiefelMuon` joint-complex pair group step, shared `lane_energy`, and large-batch SNR gain."""
    torch.manual_seed(TEST_SEED)
    w2d_re = nn.Parameter(torch.randn(32, 16))
    w2d_im = nn.Parameter(torch.randn(32, 16))
    w3d_re = nn.Parameter(torch.randn(3, 16, 16))
    w3d_im = nn.Parameter(torch.randn(3, 16, 16))

    cfg = StiefelMuonConfig(
        nesterov=True,
        polar_method="polar_express",
        stiefel_penalty=0.05,
        retract_every=1,
        use_normuon=True,
        use_twin_rail=True,
    )
    opt = StiefelMuon(
        [{"params": [(w2d_re, w2d_im), (w3d_re, w3d_im)], "is_complex_pair_group": True}],
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
        decay_mode="cautious",
        config=cfg,
    )

    for _ in range(3):
        for p in (w2d_re, w2d_im, w3d_re, w3d_im):
            p.grad = torch.randn_like(p)
        opt.step()

    assert "lane_energy" in opt.state[w2d_re]
    assert "lane_energy_0" in opt.state[w3d_re]
    assert "lane_energy" not in opt.state[w2d_im]

    w_wide_re = nn.Parameter(torch.randn(16, 32))
    w_wide_im = nn.Parameter(torch.randn(16, 32))
    opt_smooth = StiefelMuon(
        [{"params": [w_wide_re, w_wide_im], "is_complex_pair_group": True, "batched_ns": False}],
        lr=0.02,
        weight_decay=0.01,
        decay_mode="smooth",
        stiefel_penalty=0.05,
        retract_every=1,
        use_normuon=True,
    )
    w_wide_re.grad = torch.randn_like(w_wide_re)
    w_wide_im.grad = torch.randn_like(w_wide_im)
    opt_smooth.step()
    assert "lane_energy" in opt_smooth.state[w_wide_re]

    g_true_re = torch.randn(32, 32)
    g_true_im = torch.randn(32, 32)
    u_true_re, u_true_im = complex_polar_express_orth(g_true_re, g_true_im, steps=5, compute_dtype=torch.float32)

    noise_8k = 1.60
    noise_32k = noise_8k * math.sqrt(8192.0 / 32768.0)
    u8_re, u8_im = complex_polar_express_orth(
        g_true_re + noise_8k * torch.randn_like(g_true_re),
        g_true_im + noise_8k * torch.randn_like(g_true_im),
        steps=5,
        compute_dtype=torch.bfloat16,
    )
    u32_re, u32_im = complex_polar_express_orth(
        g_true_re + noise_32k * torch.randn_like(g_true_re),
        g_true_im + noise_32k * torch.randn_like(g_true_im),
        steps=5,
        compute_dtype=torch.bfloat16,
    )

    cos_8k = _complex_cos(u8_re, u8_im, u_true_re, u_true_im)
    cos_32k = _complex_cos(u32_re, u32_im, u_true_re, u_true_im)
    assert (cos_32k - cos_8k) >= LARGE_BATCH_COSINE_GAIN_FLOOR


def test_compute_large_batch_muon_lr_scale() -> None:
    """Verify square-root batch LR scaling across `{8192, 32768, 65536}` and clamping bounds."""
    assert math.isclose(compute_large_batch_muon_lr_scale(DEFAULT_MUON_REFERENCE_BATCH_TOKENS), 1.0, rel_tol=1e-6)
    assert math.isclose(compute_large_batch_muon_lr_scale(32768), 2.0, rel_tol=1e-6)
    assert math.isclose(compute_large_batch_muon_lr_scale(65536), math.sqrt(8.0), rel_tol=1e-6)
    assert math.isclose(compute_large_batch_muon_lr_scale(64), MIN_MUON_LR_SCALE, rel_tol=1e-6)
    assert math.isclose(compute_large_batch_muon_lr_scale(10_000_000), MAX_MUON_LR_SCALE, rel_tol=1e-6)


def test_classify_kronecker_model_parameters_mock_and_real_models() -> None:
    """Verify deterministic 100% parameter partitioning across mock and canonical Kronecker models."""
    for r_val in (1, 2, 4):
        mock = _MockFlagshipKronecker(r_val)
        part = classify_kronecker_model_parameters(mock)
        all_ids = (
            {id(p) for p in part.muon_complex_pairs}
            | {id(p) for p in part.embed_params}
            | {id(p) for p in part.povm_params}
            | {id(p) for p in part.gate_params}
        )
        expected_ids = {id(p) for p in mock.parameters() if p.requires_grad}
        assert all_ids == expected_ids
        assert id(mock.router_w_re) in {id(p) for p in part.gate_params}
        assert id(mock.echo_gate_logits) in {id(p) for p in part.gate_params}
        assert id(mock.gram_eta) in {id(p) for p in part.gate_params}
        if r_val == 1:
            assert id(mock.stinespring_w_re) in {id(p) for p in part.gate_params}
            assert id(mock.meaning_mix_re) in {id(p) for p in part.gate_params}
        else:
            assert id(mock.stinespring_w_re) in {id(p) for p in part.muon_complex_pairs}
            assert id(mock.meaning_mix_re) in {id(p) for p in part.muon_complex_pairs}

    for r_val in (1, 2, 4):
        m1 = KroneckerMultiMeaning16x4GPT(
            KroneckerMultiMeaningConfig(vocab_size=256, valid_vocab_size=256, num_layers=2, num_meanings=r_val)
        )
        m2 = CalibratedKroneckerBornGPT(
            CalibratedKroneckerConfig(vocab_size=256, valid_vocab_size=256, num_layers=2, num_meanings=r_val)
        )
        m3 = NgramCalibratedKroneckerBornGPT(
            NgramCalibratedKroneckerConfig(
                vocab_size=256,
                valid_vocab_size=256,
                num_layers=2,
                num_meanings=r_val,
                direct_bigram_slots=64,
                hashed_trigram_buckets=64,
            )
        )
        for model in (m1, m2, m3):
            part = classify_kronecker_model_parameters(model)
            total_classified = (
                len(part.muon_complex_pairs)
                + len(part.embed_params)
                + len(part.povm_params)
                + len(part.gate_params)
            )
            total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            assert total_classified == total_trainable


def test_build_gemini_complex_optimizers_joint_and_split() -> None:
    """Verify `build_gemini_complex_optimizers` with `use_joint_complex_muon=True/False` and `batch_tokens=32768`."""
    torch.manual_seed(TEST_SEED)
    cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=256,
        valid_vocab_size=256,
        d_complex=64,
        n_registers=2,
        d_register=32,
        num_layers=2,
        block_size=32,
    )
    model = GeminiPackedComplexRoPEGPT(cfg)
    opt_muon, opt_adam = build_gemini_complex_optimizers(
        model,
        use_joint_complex_muon=True,
        batch_tokens=32768,
    )
    assert opt_muon.param_groups[0]["is_complex_pair_group"] is True

    muon_ids = {id(p) for g in opt_muon.param_groups for p in g["params"]}
    adam_ids = {id(p) for g in opt_adam.param_groups for p in g["params"]}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert muon_ids.isdisjoint(adam_ids)
    assert (muon_ids | adam_ids) == model_ids

    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    _, loss = model(x, y)
    loss.backward()
    opt_muon.step()
    opt_adam.step()
    opt_muon.zero_grad(set_to_none=True)
    opt_adam.zero_grad(set_to_none=True)

    opt_muon_split, _ = build_gemini_complex_optimizers(model, use_joint_complex_muon=False)
    assert opt_muon_split.param_groups[0].get("is_complex_pair_group", False) is False


class Track1Track2JointComplexMuonTestSuite(unittest.TestCase):
    """Standard `unittest.TestCase` wrapper for environments without `pytest`."""

    def test_all(self) -> None:
        test_joint_complex_polar_express_unitarity_across_shapes()
        test_joint_complex_polar_express_u1_phase_equivariance()
        test_joint_complex_polar_express_suppresses_imaginary_noise_at_real_init()
        test_retract_complex_stiefel_pair_and_kraus_generators()
        test_stiefel_muon_complex_pair_group_step_and_large_batch_cosine()
        test_compute_large_batch_muon_lr_scale()
        test_classify_kronecker_model_parameters_mock_and_real_models()
        test_build_gemini_complex_optimizers_joint_and_split()


if __name__ == "__main__":
    unittest.main()
