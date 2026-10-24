"""CPU Unit Tests for Experiment Plan 040 (`Wave 25`):
1. Genuine `ILSVRC ImageNet-1K` (`1,281,167` train / `50,000` val / `100,000` test, `1,000` classes) loader.
2. `WaveletCholeskyVisionClassifier` quantum building-block ablation flags & exact probability normalization.
3. `Radical2Pow16WaveletCholeskyGPT` full 256-D Hilbert space `use_last_layer_measurement_operator=True`
   + `use_convex_traceless_born=True` forward/backward gradient flow.
4. Promoted Plan 040 diagnostic artifacts (`.cache/multiclass_ce_margin_decomposer.json` and
   `.cache/dyadic_tree_conditional_calibration_probe.json`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from qdlblocks.layers.wavelet_cholesky import WaveletCholeskyMultiscaleBranch
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
)
from qdlblocks.models.wavelet_cholesky_vision import (
    VisionWaveletCholeskyConfig,
    WaveletCholeskyVisionClassifier,
    load_or_download_imagenet1k_32x32,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_or_download_imagenet1k_32x32_integrity() -> None:
    """Verify `load_or_download_imagenet1k_32x32` loads genuine ILSVRC ImageNet-1K metadata and normalized tensors."""
    cache_dir = ROOT / ".cache" / "datasets" / "imagenet1k_32x32"
    bundle = load_or_download_imagenet1k_32x32(
        cache_dir=cache_dir,
        device="cpu",
        max_train_samples=2000,
    )
    assert bundle.name == "ilsvrc_imagenet1k"
    assert bundle.total_train_rows == 1281167
    assert bundle.total_val_rows == 50000
    assert bundle.total_test_rows == 100000
    assert bundle.num_classes == 1000
    assert bundle.num_coarse_classes == 32
    assert bundle.dyadic_dim == 1024

    assert bundle.train_x.shape == (2000, 3, 32, 32)
    assert bundle.train_y.shape == (2000,)
    assert bundle.train_coarse_y.shape == (2000,)
    assert bundle.test_x.shape == (50000, 3, 32, 32)
    assert bundle.test_y.shape == (50000,)
    assert bundle.test_coarse_y.shape == (50000,)
    assert bundle.class_to_coarse.shape == (1000,)

    assert int(bundle.test_y.min().item()) == 0
    assert int(bundle.test_y.max().item()) == 999
    assert int(bundle.test_coarse_y.min().item()) == 0
    assert int(bundle.test_coarse_y.max().item()) == 31
    assert torch.isfinite(bundle.train_x).all()


@pytest.mark.parametrize(
    "ablation_flags",
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
@pytest.mark.parametrize(
    "readout_mode",
    [
        "convex_traceless_born",
        "dyadic_hierarchical_born",
    ],
)
def test_vision_classifier_quantum_building_block_ablations(
    ablation_flags: tuple[bool, bool, bool],
    readout_mode: str,
) -> None:
    """Verify forward/backward and exact probability normalization across all quantum building-block ablation flags."""
    torch.manual_seed(2040)
    use_wav, use_mz, use_kraus = ablation_flags
    num_classes, num_coarse, dyadic_dim = 1000, 32, 1024
    class_to_coarse = (torch.arange(num_classes, dtype=torch.long) * num_coarse) // num_classes

    cfg = VisionWaveletCholeskyConfig(
        num_classes=num_classes,
        num_coarse_classes=num_coarse,
        dyadic_dim=dyadic_dim,
        d_model=64,
        num_layers=2,
        patch_size=4,
        kraus_rank=4,
        cholesky_block_size=16,
        readout_mode=readout_mode,
        dropout=0.0,
        use_wavelet_cholesky_branch=use_wav,
        use_mach_zehnder_interferometer=use_mz,
        use_kraus_diffuse_bath=use_kraus,
    )
    model = WaveletCholeskyVisionClassifier(cfg, class_to_coarse)
    if use_wav:
        assert isinstance(model.blocks[0].wav_branch, WaveletCholeskyMultiscaleBranch)
    else:
        assert isinstance(model.blocks[0].wav_branch, nn.Linear)

    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    log_probs, diag = model.compute_log_probs_and_diagnostics(images)
    assert log_probs.shape == (batch_size, num_classes)
    assert torch.isfinite(log_probs).all()

    probs = log_probs.exp()
    max_prob_sum_err = float((probs.sum(dim=-1) - 1.0).abs().max().item())
    assert max_prob_sum_err < 1e-5
    assert diag["psi_fine"].shape == (num_classes, 64)
    assert diag["psi_coarse"].shape == (num_coarse, 64)

    _, loss = model(images, targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

    if use_kraus:
        assert model.log_lambda_diff.grad is not None
        assert float(model.log_lambda_diff.grad.abs().item()) > 0.0
    else:
        assert model.log_lambda_diff.grad is None or float(model.log_lambda_diff.grad.abs().item()) == 0.0


def test_radical_2pow16_gpt_last_layer_meas_and_convex_traceless_256d_hilbert() -> None:
    """Verify `Radical2Pow16WaveletCholeskyGPT` with `use_last_layer_measurement_operator=True` and
    `use_convex_traceless_born=True` operates in the full 256-D Hilbert space and produces finite gradients.
    """
    torch.manual_seed(2040)
    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        vocab_size=1024,
        valid_vocab_size=1000,
        ambient_n=1024,
        num_layers=2,
        embed_dim=256,
        num_heads=4,
        block_size=16,
        window_size=16,
        fine_detail_dim=64,
        mid_detail_dim=128,
        diffuse_rank=8,
        use_kraus_retraction_and_lambda_anneal=True,
        use_last_layer_measurement_operator=True,
        use_convex_traceless_born=True,
        traceless_gamma=0.25,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg)
    with torch.no_grad():
        nn.init.normal_(model.pyramid.wav_256, std=0.15)
        nn.init.normal_(model.pyramid.wav_2048, std=0.15)
        nn.init.normal_(model.pyramid.wav_8192, std=0.15)
        nn.init.normal_(model.pyramid.wav_65536, std=0.15)

    idx = torch.randint(0, 1000, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 1000, (2, 8), dtype=torch.long)

    psi_in_0 = model.pyramid.embed_tokens(idx)
    psi_s_out, psi_d_out, psi_vocab, w_k = model.pyramid.apply_last_layer_measurement(
        psi_in_0, psi_in_0, psi_in_0, idx
    )
    assert psi_s_out.shape == (2, 8, 256)
    assert psi_d_out.shape == (2, 8, 256)
    assert psi_vocab.shape == (1024, 256)
    assert w_k.shape == (8,)
    assert torch.allclose(psi_s_out, psi_in_0)

    logits, loss = model(idx, targets, return_logits=True)
    assert logits is not None and logits.shape == (2, 8, 1024)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    assert model.pyramid.kraus_generators.grad is not None
    assert torch.isfinite(model.pyramid.kraus_generators.grad).all()
    assert float(model.pyramid.kraus_generators.grad.abs().max().item()) > 0.0

    assert model.pyramid.wav_65536.grad is not None
    assert torch.isfinite(model.pyramid.wav_65536.grad).all()
    assert float(model.pyramid.wav_65536.grad.abs().max().item()) > 0.0


def test_promoted_plan040_diagnostic_json_artifacts() -> None:
    """Verify existence, schema, and mathematical invariants of Plan 040 diagnostic JSON artifacts."""
    margin_path = ROOT / ".cache" / "multiclass_ce_margin_decomposer.json"
    cal_path = ROOT / ".cache" / "dyadic_tree_conditional_calibration_probe.json"
    assert margin_path.exists(), f"Missing {margin_path}"
    assert cal_path.exists(), f"Missing {cal_path}"

    margin_data = json.loads(margin_path.read_text(encoding="utf-8"))
    assert "config" in margin_data
    assert "summary" in margin_data
    assert "by_num_classes" in margin_data
    assert "rows" in margin_data

    for k_str in ("100", "1000", "50257"):
        assert k_str in margin_data["by_num_classes"]
        k_block = margin_data["by_num_classes"][k_str]
        for fam in (
            "naive_kboson_born",
            "traceless_kboson_born",
            "convex_traceless_born",
            "dyadic_hierarchical_born",
        ):
            assert fam in k_block
            row = k_block[fam]
            assert row["decomposition_residual"] < 1e-5
        assert k_block["convex_traceless_born"]["total_ce_nats"] < k_block["naive_kboson_born"]["total_ce_nats"]
        assert k_block["dyadic_hierarchical_born"]["total_ce_nats"] < k_block["naive_kboson_born"]["total_ce_nats"]

    cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert "config" in cal_data
    assert "summary" in cal_data
    assert "families" in cal_data
    assert "rows" in cal_data

    fams = cal_data["families"]
    for fam in (
        "naive_kboson_born",
        "traceless_kboson_born",
        "convex_traceless_born",
        "dyadic_hierarchical_born",
    ):
        assert fam in fams
        f_row = fams[fam]
        assert f_row["max_prob_sum_error"] < 1e-5
        assert f_row["chain_rule_nll_residual"] < 1e-5

    assert fams["convex_traceless_born"]["total_nll_nats"] < fams["naive_kboson_born"]["total_nll_nats"]
    assert fams["dyadic_hierarchical_born"]["total_nll_nats"] < fams["naive_kboson_born"]["total_nll_nats"]
    assert fams["convex_traceless_born"]["ece"] < fams["naive_kboson_born"]["ece"]
    assert fams["dyadic_hierarchical_born"]["ece"] < fams["naive_kboson_born"]["ece"]
