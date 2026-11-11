"""Unit tests for `Wave 76` Cross-Architecture X-Ray & Mechanistic Probes (`EXPERIMENT_PLAN_096.md`)."""

from __future__ import annotations

import math

import torch

from qdlblocks.benchmarks.architecture_mechanistic_probes import (
    DualHorizonShadowEMATracker,
    fit_early_powerlaw_asymptote,
    generate_fuzzy_induction_batch,
    generate_mqar_batch,
    run_baton_compartment_ablation_audit,
)
from qdlblocks.benchmarks.architecture_xray_diagnostics import (
    FINEWEB_BOS_TOKEN_ID,
    VALID_FINEWEB_VOCAB_SIZE,
    Baton4CompAdapter,
    SharpenedPBosonDocMaskedBatonLM,
    compute_document_horizon_and_copy_masks,
    compute_expected_calibration_error,
    compute_stratified_nat_gap_xray,
)
from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import (
    FusedBF16FourCompartmentBatonConfig,
    FusedBF16FourCompartmentBatonLM,
)


def test_document_horizon_and_copy_masks_respect_bos_boundaries() -> None:
    """Verify `<BOS>` (`50256`) resets intra-document position and blocks cross-document copy flags."""
    # Row 0: [10, 20, BOS(50256), 30, 40] -> targets [20, 50256, 30, 10, 30]
    x = torch.tensor([[10, 20, FINEWEB_BOS_TOKEN_ID, 30, 40]], dtype=torch.long)
    y = torch.tensor([[20, FINEWEB_BOS_TOKEN_ID, 30, 10, 30]], dtype=torch.long)

    pos_in_doc, in_doc_copy, self_echo = compute_document_horizon_and_copy_masks(x, y)
    assert pos_in_doc.tolist() == [[0, 1, 0, 1, 2]]
    # At index 3 (`x=30`, `y=10`), token `10` appeared BEFORE `<BOS>` at index 0, NOT in current doc!
    assert bool(in_doc_copy[0, 3].item()) is False
    # At index 4 (`x=40`, `y=30`), token `30` appeared AFTER `<BOS>` at index 3 in current doc!
    assert bool(in_doc_copy[0, 4].item()) is True
    assert self_echo.tolist() == [[False, False, False, False, False]]


def test_sharpened_pboson_doc_masked_baton_zero_bos_leakage_and_grad() -> None:
    """Verify `SharpenedPBosonDocMaskedBatonLM` achieves 0.0 cross-`<BOS>` leakage and clean backward."""
    device = torch.device("cpu")
    cfg = FusedBF16FourCompartmentBatonConfig(
        hilbert_dim=32, num_layers=2, use_bfloat16_amp=False
    )
    model = SharpenedPBosonDocMaskedBatonLM(cfg, boson_power=3.0, mask_bos_documents=True).to(device)
    adapter = Baton4CompAdapter("test_sharp", "Test Sharp Baton", model, device=device)

    gen = torch.Generator().manual_seed(77)
    xb, yb, q_mask = generate_fuzzy_induction_batch(
        batch_size=4, seq_len=32, vocab_size=VALID_FINEWEB_VOCAB_SIZE, gen=gen, device=device
    )
    assert int(q_mask.sum().item()) == 4
    loss = adapter.train_step(xb, yb, step=1, total_steps=10)
    assert math.isfinite(loss) and loss > 0.0

    audit = run_baton_compartment_ablation_audit(model, xb, yb, max_batches=1)
    assert audit["attn_telemetry"]["cross_bos_leakage_fraction"] == 0.0
    assert "no_bath_log_prior" in audit["leave_one_out_delta_ce"]
    assert math.isfinite(audit["uncollapsed_ridge_mup_heldout_ce"])


def test_stratified_xray_and_dual_horizon_shadow_ema_restoration() -> None:
    """Verify `compute_stratified_nat_gap_xray` and exact parameter restoration in `DualHorizonShadowEMATracker`."""
    device = torch.device("cpu")
    cfg = FusedBF16FourCompartmentBatonConfig(
        hilbert_dim=32, num_layers=1, use_bfloat16_amp=False
    )
    model = FusedBF16FourCompartmentBatonLM(cfg).to(device)
    adapter = Baton4CompAdapter("test_w74", "Test W74 1L", model, device=device)
    tracker = DualHorizonShadowEMATracker(model, fast_decay=0.5, slow_decay=0.9)

    x_eval = torch.randint(0, 1000, (4, 16), dtype=torch.long)
    x_eval[:, 8] = FINEWEB_BOS_TOKEN_ID
    y_eval = torch.randint(0, 1000, (4, 16), dtype=torch.long)

    adapter.train_step(x_eval, y_eval, step=1, total_steps=5)
    tracker.update(model)
    before_snap = {k: p.detach().clone() for k, p in model.named_parameters()}
    ema_res = tracker.evaluate_hot_fast_slow(adapter, x_eval, y_eval, micro_batch=2)
    for k, p in model.named_parameters():
        assert torch.allclose(p, before_snap[k], atol=0.0, rtol=0.0)
    assert math.isfinite(ema_res["hot_val_ce"])
    assert math.isfinite(ema_res["slow_ema_val_ce"])

    decile_map = torch.arange(VALID_FINEWEB_VOCAB_SIZE, dtype=torch.long) % 10
    xray = compute_stratified_nat_gap_xray(adapter, x_eval, y_eval, decile_map=decile_map, micro_batch=2)
    assert xray["eval_tokens"] == 64
    assert len(xray["deciles"]) == 10
    assert 0.0 <= compute_expected_calibration_error(torch.full((20,), 0.4), torch.zeros(20)) <= 1.0


def test_powerlaw_asymptote_and_mqar_generator() -> None:
    """Verify `fit_early_powerlaw_asymptote` recovers synthetic $L_\\infty$ and `generate_mqar_batch` shapes."""
    d_vals = [0.5, 1.0, 2.0, 4.0, 8.0]
    # Exact curve: L(D) = 3.40 + 2.20 * D^(-0.35)
    y_vals = [3.40 + 2.20 * (d ** (-0.35)) for d in d_vals]
    fit = fit_early_powerlaw_asymptote(d_vals, y_vals)
    assert abs(fit["l_inf_asymptote"] - 3.40) < 0.08
    assert fit["r2_fit"] > 0.999

    gen = torch.Generator().manual_seed(99)
    xb, yb, q_mask = generate_mqar_batch(
        batch_size=3, seq_len=64, num_pairs=4, vocab_size=VALID_FINEWEB_VOCAB_SIZE, gen=gen, device=torch.device("cpu")
    )
    assert xb.shape == (3, 64)
    assert yb.shape == (3, 64)
    assert int(q_mask.sum().item()) == 12
