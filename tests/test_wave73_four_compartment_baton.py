"""Unit and integration tests for Wave 73 (`EXPERIMENT_PLAN_093.md`):
4-Compartment Quantum Statistical Mechanics Baton Architecture (`four_compartment_baton_lm_lab.py`).
"""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.four_compartment_baton_lm_lab import (
    ALL_WAVE73_VARIANTS,
    DEFAULT_HILBERT_DIM,
    DEFAULT_NUM_PACKETS,
    FourCompartmentBatonConfig,
    FourCompartmentBatonHighwayLM,
    build_wave73_config_for_variant,
    run_five_regime_baton_probe,
    trace_of_density,
)


def test_four_compartment_8packet_geometry_and_online_ingestion() -> None:
    """Verify exact `(B, T, 8, 64, 64)` Hermitian unit-trace density packets and online bath ingestion."""
    torch.manual_seed(42)
    cfg = FourCompartmentBatonConfig(
        vocab_size=256,
        hilbert_dim=DEFAULT_HILBERT_DIM,
        num_layers=2,
        use_schur_intersection=True,
        use_two_hop_triangle=True,
    )
    model = FourCompartmentBatonHighwayLM(cfg)
    model.reset_online_baths_to_cold_step0()
    assert int(model.online_ingested_tokens.item()) == 0

    idx = torch.randint(0, 256, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 16), dtype=torch.long)
    out = model.forward_with_diagnostics(idx, targets, ingest_online=True)

    assert out["logits"].shape == (2, 16, 256)
    assert out["loss"] is not None and torch.isfinite(out["loss"])
    assert out["packets_re"].shape == (
        2,
        16,
        DEFAULT_NUM_PACKETS,
        DEFAULT_HILBERT_DIM,
        DEFAULT_HILBERT_DIM,
    )
    assert out["packets_im"].shape == (
        2,
        16,
        DEFAULT_NUM_PACKETS,
        DEFAULT_HILBERT_DIM,
        DEFAULT_HILBERT_DIM,
    )

    pkt_traces = trace_of_density(out["packets_re"])
    out_traces = trace_of_density(out["rho_out_re"])
    assert torch.allclose(pkt_traces, torch.ones_like(pkt_traces), atol=1e-4)
    assert torch.allclose(out_traces, torch.ones_like(out_traces), atol=1e-4)
    assert int(model.online_ingested_tokens.item()) == 32

    out["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(g >= 0.0 for g in grad_norms)


def test_all_wave73_variant_configs_build_cleanly() -> None:
    """Verify all 4 canonical Wave 73 variants construct valid models."""
    for variant_id in ALL_WAVE73_VARIANTS:
        cfg = build_wave73_config_for_variant(variant_id, vocab_size=128)
        model = FourCompartmentBatonHighwayLM(cfg)
        idx = torch.randint(0, 128, (2, 8), dtype=torch.long)
        logits, loss = model(idx, idx)
        assert logits.shape == (2, 8, 128)
        assert loss is not None and torch.isfinite(loss)


def test_five_regime_quantum_statistical_mechanics_probe() -> None:
    """Verify all 5 physical regimes from the Gemini notebook pass deterministically."""
    report = run_five_regime_baton_probe()
    assert report["overall_passed"] is True
    assert report["regime1_cold_bos_and_online_unigram_bath"]["passed"] is True
    assert report["regime2_mps_stop_codon_destructive_cancellation"]["passed"] is True
    assert report["regime3_in_context_lindblad_transition_induction"]["passed"] is True
    assert report["regime4_coding_hairpin_schur_vs_additive"]["passed"] is True
    assert report["regime5_zero_shot_walsh_hadamard_token_id_reversal"]["passed"] is True
