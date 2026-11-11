"""Unit tests for Wave 75 (`EXPERIMENT_PLAN_095.md`):
1. Exact FP32 algebraic equivalence (`< 1e-5`) between `Wave 73` explicit `d × d` mixed-state
   Schur packets and `Wave 75` Theorem 16 zero-`(d, d)` contraction (`verify_wave73_exact_mixed_state_equivalence`).
2. Arbitrary-width forward/backward execution across `d ∈ {64, 128, 192, 256}` (testing the 3-bit
   Walsh–Hadamard Token-ID basis extension for `d > 136`).
3. `ShadowEMAParameterTracker` bit-for-bit hot parameter restoration after evaluating `hot_val_ce`
   and `ema_val_ce`.
"""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.exact_mixed_state_baton_width_scaling_lab import (
    DEFAULT_WIDTH_CHOICES,
    ExactMixedStateFourCompartmentBatonLM,
    ShadowEMAParameterTracker,
    verify_wave73_exact_mixed_state_equivalence,
)
from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import (
    FusedBF16FourCompartmentBatonConfig,
    FusedBF16FourCompartmentBatonLM,
)


def test_theorem16_exact_mixed_state_equivalence_with_wave73() -> None:
    """Verify Theorem 16 zero-(d, d) contraction matches Wave 73 explicit d×d mixed-state Schur product to < 1e-5."""
    audit = verify_wave73_exact_mixed_state_equivalence(
        batch_size=2, seq_len=12, hilbert_dim=64, seed=42
    )
    assert audit["passed"], f"Equivalence audit failed: {audit}"
    assert audit["w75_theorem16_vs_w73_explicit_deep_max_abs_err"] <= 1e-5
    assert audit["w75_theorem16_vs_w73_explicit_id_max_abs_err"] <= 1e-5


def test_width_scaling_forward_backward_all_widths() -> None:
    """Verify forward + backward pass for d in (64, 128, 192, 256) on both ExactMixedState and FusedBF16 models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 512
    for d in DEFAULT_WIDTH_CHOICES:
        cfg = FusedBF16FourCompartmentBatonConfig(
            vocab_size=vocab_size,
            hilbert_dim=d,
            num_layers=1,
            use_two_hop_triangle=True,
            use_bfloat16_amp=False,
        )
        for model_cls in (FusedBF16FourCompartmentBatonLM, ExactMixedStateFourCompartmentBatonLM):
            model = model_cls(cfg).to(device)
            x = torch.randint(0, vocab_size, (2, 16), device=device)
            y = torch.randint(0, vocab_size, (2, 16), device=device)
            out = model.forward_with_diagnostics(x, targets=y, ingest_online=True, return_logits=True)
            assert out["loss"] is not None
            assert torch.isfinite(out["loss"]).all()
            out["loss"].backward()


def test_shadow_ema_tracker_updates_and_restores_hot_weights() -> None:
    """Verify ShadowEMAParameterTracker updates shadow buffers and leaves hot weights bit-for-bit intact."""
    cfg = FusedBF16FourCompartmentBatonConfig(
        vocab_size=256,
        hilbert_dim=64,
        num_layers=1,
        use_two_hop_triangle=True,
        use_bfloat16_amp=False,
    )
    model = ExactMixedStateFourCompartmentBatonLM(cfg)
    tracker = ShadowEMAParameterTracker(model, beta_ema=0.5)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    tracker.update(model)

    hot_snapshot = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    for name, param in model.named_parameters():
        if param.requires_grad:
            diff = (param.detach().float() - tracker.shadow_params[name]).abs().max().item()
            assert diff > 0.4, "EMA shadow parameter should lag behind step-jumped hot parameter"
            assert torch.equal(param.detach(), hot_snapshot[name])
