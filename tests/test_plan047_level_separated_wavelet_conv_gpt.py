"""Unit and hypothesis tests for Experiment Plan 047 (Wave 32):
Explicit Level-Separated Wavelet Tensors (`WaveletScaleState`), Multi-Rate 1D Wavefunction
Convolution Bank (`MultiRateWavefunctionConvBank`), and Strided vs. Level-Assigned
Subspace-Balanced Coincidence Readout (`LevelSeparatedWaveletConvGPT`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from qdlblocks.layers.multirate_wavefunction_conv import (
    MultiRateWavefunctionConvBank,
    WaveletScaleState,
)
from qdlblocks.layers.subspace_coincidence_born import (
    compute_subspace_cosines,
    strided_subspace_l2_normalize,
)
from qdlblocks.models.level_separated_wavelet_conv_gpt import (
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    build_level_separated_wavelet_conv_optimizers,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
TEST_EMBED_DIM: int = 128
TEST_NUM_SUBSPACES: int = 4
TEST_SUB_DIM: int = TEST_EMBED_DIM // TEST_NUM_SUBSPACES
TEST_BATCH_SIZE: int = 2
TEST_SEQ_LEN: int = 24
TEST_VOCAB_SIZE: int = 512
NORM_ATOL: float = 1e-5
CAUSALITY_ATOL: float = 1e-6


def test_wavelet_scale_state_independent_level_normalization() -> None:
    """Verify WaveletScaleState locks each wavelet level at 25% energy even under 8x coarse growth."""
    torch.manual_seed(47)
    c3 = torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_SUB_DIM) * 8.0
    c2 = torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_SUB_DIM) * 2.0
    c1 = torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_SUB_DIM) * 1.0
    c0 = torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_SUB_DIM) * 0.25

    state = WaveletScaleState(
        coarse_256=c3,
        mid_2048=c2,
        sub_8192=c1,
        fine_65536=c0,
    ).normalize_levels()

    expected_sub_norm = 1.0 / math.sqrt(float(TEST_NUM_SUBSPACES))
    for level_tensor in state.as_tuple():
        norms = torch.linalg.vector_norm(level_tensor, dim=-1)
        assert torch.allclose(norms, torch.full_like(norms, expected_sub_norm), atol=NORM_ATOL)

    concat = state.to_product_of_spheres()
    total_norms = torch.linalg.vector_norm(concat, dim=-1)
    assert torch.allclose(total_norms, torch.ones_like(total_norms), atol=NORM_ATOL)

    energies = state.level_energy_fractions()
    for key in ("coarse_256", "mid_2048", "sub_8192", "fine_65536"):
        assert abs(energies[key] - 0.25) < 1e-4

    roundtrip = WaveletScaleState.from_concat(concat)
    for orig_t, rt_t in zip(state.as_tuple(), roundtrip.as_tuple()):
        assert torch.allclose(orig_t, rt_t, atol=NORM_ATOL)


def test_strided_vs_contiguous_subspace_l2_normalize() -> None:
    """Verify strided_subspace_l2_normalize enforces ||x[..., m::M]||_2 = 1/sqrt(M) and ||x||_2 = 1."""
    torch.manual_seed(147)
    x = torch.randn(16, TEST_EMBED_DIM)
    decay = torch.pow(torch.arange(1, TEST_EMBED_DIM + 1, dtype=torch.float32), -0.85)
    x = x * decay.unsqueeze(0)

    x_strided = strided_subspace_l2_normalize(x, num_subspaces=TEST_NUM_SUBSPACES)
    total_norms = torch.linalg.vector_norm(x_strided, dim=-1)
    assert torch.allclose(total_norms, torch.ones_like(total_norms), atol=NORM_ATOL)

    expected_sub = 1.0 / math.sqrt(float(TEST_NUM_SUBSPACES))
    for m in range(TEST_NUM_SUBSPACES):
        sub_slice = x_strided[:, m::TEST_NUM_SUBSPACES]
        sub_norms = torch.linalg.vector_norm(sub_slice, dim=-1)
        assert torch.allclose(sub_norms, torch.full_like(sub_norms, expected_sub), atol=NORM_ATOL)

    cos_self = compute_subspace_cosines(
        x_strided[:4],
        x_strided[:4],
        num_subspaces=TEST_NUM_SUBSPACES,
        partition_mode="strided",
    )
    diag_cos = torch.diagonal(cos_self, dim1=0, dim2=1)
    assert torch.allclose(diag_cos, torch.ones_like(diag_cos), atol=1e-4)


def test_multirate_wavefunction_conv_bank_strict_causality() -> None:
    """Verify MultiRateWavefunctionConvBank has zero future leakage across dilations (4, 2, 1, 1)."""
    torch.manual_seed(247)
    conv_bank = MultiRateWavefunctionConvBank(
        embed_dim=TEST_EMBED_DIM,
        temporal_strides=(4, 2, 1, 1),
    )

    x_a = torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_EMBED_DIM)
    x_b = x_a.clone()
    cutoff_t = 11
    x_b[:, cutoff_t + 1 :, :] = torch.randn_like(x_b[:, cutoff_t + 1 :, :]) * 5.0

    state_a = WaveletScaleState.from_concat(x_a).normalize_levels()
    state_b = WaveletScaleState.from_concat(x_b).normalize_levels()

    out_a = conv_bank.forward_levels(state_a).to_product_of_spheres()
    out_b = conv_bank.forward_levels(state_b).to_product_of_spheres()

    max_past_diff = float((out_a[:, : cutoff_t + 1, :] - out_b[:, : cutoff_t + 1, :]).abs().max().item())
    assert max_past_diff < CAUSALITY_ATOL, f"Causal leakage detected: max_past_diff={max_past_diff}"

    total_norms = torch.linalg.vector_norm(out_a, dim=-1)
    assert torch.allclose(total_norms, torch.ones_like(total_norms), atol=NORM_ATOL)


def test_level_separated_wavelet_conv_gpt_forward_backward_and_eval() -> None:
    """Verify LevelSeparatedWaveletConvGPT forward, backward, optimizer step, and eval logits."""
    torch.manual_seed(347)
    cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=1024,
        valid_vocab_size=1000,
        embed_dim=TEST_EMBED_DIM,
        num_layers=2,
        num_heads=4,
        block_size=TEST_SEQ_LEN,
        num_subspaces=TEST_NUM_SUBSPACES,
        partition_mode="strided",
    )
    model = LevelSeparatedWaveletConvGPT(cfg)
    stiefel_opt, adamw_opt = build_level_separated_wavelet_conv_optimizers(model)

    idx = torch.randint(0, cfg.valid_vocab_size, (TEST_BATCH_SIZE, TEST_SEQ_LEN))
    targets = torch.randint(0, cfg.valid_vocab_size, (TEST_BATCH_SIZE, TEST_SEQ_LEN))

    logits, loss = model(idx, targets=targets, return_logits=True)
    assert loss is not None and torch.isfinite(loss)
    assert logits is not None and logits.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN, cfg.valid_vocab_size)
    loss.backward()

    for name in ("wav_256", "wav_2048", "wav_8192", "wav_65536"):
        param = getattr(model.pyramid, name)
        assert param.grad is not None and torch.isfinite(param.grad).all(), f"Missing/NaN grad on {name}"

    assert model.conv_bank.conv_mix_gates.grad is not None
    assert torch.isfinite(model.conv_bank.conv_mix_gates.grad).all()

    diag = model.inspect_level_energy_and_rank()
    assert abs(diag["energy_fine_65536"] - 0.25) < 1e-3
    assert abs(diag["energy_coarse_256"] - 0.25) < 1e-3

    stiefel_opt.step()
    adamw_opt.step()
    model.retract_kraus_generators()


def test_subspace_specialization_probe_hypothesis_h1() -> None:
    """Verify .cache/subspace_specialization_mutual_information_probe.json passes H1."""
    probe_path = Path("/home/festivity/agy/QuantumDLBlocks/.cache/subspace_specialization_mutual_information_probe.json")
    assert probe_path.exists(), "Run scripts/subspace_specialization_mutual_information_probe.py first"
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert data["H1_cross_level_energy_preservation_passed"] is True
    assert data["H1_strided_svd_balance_passed"] is True
    cannib = data["cross_level_norm_cannibalization"]
    assert cannib["merged_global_l2_fine_energy_fraction"] < 0.05
    assert abs(cannib["explicit_level_separated_fine_energy_fraction"] - 0.25) < 1e-4


if __name__ == "__main__":
    for fn in (
        test_wavelet_scale_state_independent_level_normalization,
        test_strided_vs_contiguous_subspace_l2_normalize,
        test_multirate_wavefunction_conv_bank_strict_causality,
        test_level_separated_wavelet_conv_gpt_forward_backward_and_eval,
        test_subspace_specialization_probe_hypothesis_h1,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 5 Plan 047 unit & hypothesis tests PASSED.")
