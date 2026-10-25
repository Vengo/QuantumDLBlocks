"""Unit and hypothesis tests for Experiment Plan 048 (Wave 33):
Progress-Ramped Coincidence Warmup (`coincidence_gamma(tau) = 0.22 * tau`,
`subspace_var_penalty(tau) = 0.14 * tau`) and Per-Layer Pre-Attention
`MultiRateWavefunctionConvBank` (`PreAttnMultiRateWaveletConvBlock`) replacing the
post-attention channel U-Net on `LevelSeparatedWaveletConvGPT`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from qdlblocks.layers.multirate_wavefunction_conv import WaveletScaleState
from qdlblocks.layers.subspace_coincidence_born import (
    DEFAULT_COINCIDENCE_GAMMA,
    DEFAULT_SUBSPACE_VAR_PENALTY,
    blend_subspace_l2_normalize,
)
from qdlblocks.models.level_separated_wavelet_conv_gpt import (
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    PreAttnMultiRateWaveletConvBlock,
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
NORM_ATOL: float = 1e-5
CAUSALITY_ATOL: float = 1e-6


def test_wavelet_scale_state_interleaved_roundtrip() -> None:
    """Verify WaveletScaleState.to_interleaved_product_of_spheres and from_interleaved roundtrip."""
    torch.manual_seed(48)
    state = WaveletScaleState(
        coarse_256=torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_SUB_DIM),
        mid_2048=torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_SUB_DIM),
        sub_8192=torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_SUB_DIM),
        fine_65536=torch.randn(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_SUB_DIM),
    ).normalize_levels()

    interleaved = state.to_interleaved_product_of_spheres()
    assert interleaved.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_EMBED_DIM)
    assert torch.allclose(torch.linalg.vector_norm(interleaved, dim=-1), torch.ones(TEST_BATCH_SIZE, TEST_SEQ_LEN), atol=NORM_ATOL)

    unpacked = WaveletScaleState.from_interleaved(interleaved)
    for orig_t, rec_t in zip(state.as_tuple(), unpacked.as_tuple()):
        assert torch.allclose(orig_t, rec_t, atol=NORM_ATOL)


def test_progress_ramped_coincidence_warmup_schedule() -> None:
    """Verify coincidence_gamma(tau) = 0.22 * tau and subspace_var_penalty(tau) = 0.14 * tau."""
    cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=512,
        valid_vocab_size=500,
        embed_dim=TEST_EMBED_DIM,
        num_layers=2,
        num_heads=4,
        block_size=TEST_SEQ_LEN,
        num_subspaces=TEST_NUM_SUBSPACES,
        coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
        subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        use_coincidence_warmup=True,
    )
    model = LevelSeparatedWaveletConvGPT(cfg)

    for tau in (0.0, 0.25, 0.50, 0.75, 1.0):
        model.set_training_progress(tau)
        spec = model.build_coincidence_spec()
        assert abs(spec.coincidence_gamma - DEFAULT_COINCIDENCE_GAMMA * tau) < 1e-6
        assert abs(spec.subspace_var_penalty - DEFAULT_SUBSPACE_VAR_PENALTY * tau) < 1e-6
        assert abs(spec.subspace_balance_blend - cfg.subspace_balance_blend_max * tau) < 1e-6

    x = torch.randn(8, TEST_EMBED_DIM)
    for b_val in (0.0, 0.25, 0.5, 1.0):
        x_b = blend_subspace_l2_normalize(x, num_subspaces=TEST_NUM_SUBSPACES, blend=b_val, partition_mode="strided")
        assert torch.allclose(torch.linalg.vector_norm(x_b, dim=-1), torch.ones(8), atol=NORM_ATOL)


def test_per_layer_pre_attn_conv_block_replaces_channel_unet_and_is_causal() -> None:
    """Verify every backbone block in LevelSeparatedWaveletConvGPT is PreAttnMultiRateWaveletConvBlock and strictly causal."""
    torch.manual_seed(148)
    cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=512,
        valid_vocab_size=500,
        embed_dim=TEST_EMBED_DIM,
        num_layers=3,
        num_heads=4,
        block_size=TEST_SEQ_LEN,
        use_per_layer_pre_attn_conv=True,
    )
    model = LevelSeparatedWaveletConvGPT(cfg)
    model.eval()

    assert len(model.blocks) == 3
    for blk in model.blocks:
        assert isinstance(blk, PreAttnMultiRateWaveletConvBlock)
        assert not hasattr(blk, "wavelet_branch"), "Legacy post-attention channel U-Net must be removed"

    idx_a = torch.randint(0, cfg.valid_vocab_size, (TEST_BATCH_SIZE, TEST_SEQ_LEN))
    idx_b = idx_a.clone()
    cutoff = 10
    idx_b[:, cutoff + 1 :] = (idx_b[:, cutoff + 1 :] + 17) % cfg.valid_vocab_size

    with torch.no_grad():
        logits_a, _ = model(idx_a, return_logits=True)
        logits_b, _ = model(idx_b, return_logits=True)
    assert logits_a is not None and logits_b is not None
    max_past_diff = float((logits_a[:, : cutoff + 1, :] - logits_b[:, : cutoff + 1, :]).abs().max().item())
    assert max_past_diff < 1e-4, f"Future token leakage across stacked PreAttnMultiRateWaveletConvBlocks: {max_past_diff}"


def test_per_layer_pre_attn_conv_gradients_and_optimizer_step() -> None:
    """Verify all per-layer pre_attn_conv banks receive finite gradients and update cleanly."""
    torch.manual_seed(248)
    cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=512,
        valid_vocab_size=500,
        embed_dim=TEST_EMBED_DIM,
        num_layers=2,
        num_heads=4,
        block_size=TEST_SEQ_LEN,
        use_coincidence_warmup=True,
        use_per_layer_pre_attn_conv=True,
    )
    model = LevelSeparatedWaveletConvGPT(cfg)
    model.set_training_progress(0.4)
    stiefel_opt, adamw_opt = build_level_separated_wavelet_conv_optimizers(model)

    idx = torch.randint(0, cfg.valid_vocab_size, (TEST_BATCH_SIZE, TEST_SEQ_LEN))
    targets = torch.randint(0, cfg.valid_vocab_size, (TEST_BATCH_SIZE, TEST_SEQ_LEN))

    logits, loss = model(idx, targets=targets, return_logits=True)
    assert loss is not None and torch.isfinite(loss)
    assert logits is not None and logits.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN, cfg.valid_vocab_size)
    loss.backward()

    for l_idx, blk in enumerate(model.blocks):
        assert isinstance(blk, PreAttnMultiRateWaveletConvBlock)
        g_mix = blk.pre_attn_conv.conv_mix_gates.grad
        assert g_mix is not None and torch.isfinite(g_mix).all(), f"Missing grad on block {l_idx} pre_attn_conv"

    stiefel_opt.step()
    adamw_opt.step()
    iso_err = model.retract_kraus_generators()
    assert iso_err < 0.05


def test_receptive_field_profiler_hypothesis_h1() -> None:
    """Verify .cache/multirate_wavefunction_conv_receptive_field_profiler.json passes H1."""
    probe_path = Path("/home/festivity/agy/QuantumDLBlocks/.cache/multirate_wavefunction_conv_receptive_field_profiler.json")
    assert probe_path.exists(), "Run scripts/multirate_wavefunction_conv_receptive_field_profiler.py first"
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert data["H1_pre_attn_receptive_field_and_causality_passed"] is True
    assert data["H1_coincidence_warmup_schedule_passed"] is True


if __name__ == "__main__":
    for fn in (
        test_wavelet_scale_state_interleaved_roundtrip,
        test_progress_ramped_coincidence_warmup_schedule,
        test_per_layer_pre_attn_conv_block_replaces_channel_unet_and_is_causal,
        test_per_layer_pre_attn_conv_gradients_and_optimizer_step,
        test_receptive_field_profiler_hypothesis_h1,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 5 Plan 048 unit & hypothesis tests PASSED.")
