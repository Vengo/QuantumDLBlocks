"""Unit tests for `Wave 78` Copy/Schur Diagnostics & `R4c + R5B` Unification (`EXPERIMENT_PLAN_098.md`)."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.architecture_xray_diagnostics import (
    FINEWEB_BOS_TOKEN_ID,
    VALID_FINEWEB_VOCAB_SIZE,
    TeacherReferenceProfile,
)
from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import normalize_complex_ket
from qdlblocks.benchmarks.wave77_quantum_routing_primitives import build_bos_anchor_sink_doc_masks
from qdlblocks.benchmarks.wave78_routing_and_copy_diagnostics import (
    inspect_copy_gate_confidence_and_entity_calibration,
    profile_complementary_schur_pand_and_purity_layers,
)
from qdlblocks.benchmarks.wave78_unified_mup_carbon_baton import (
    PurityConditionedHybridWalshCopyGate,
    Wave78VariantKind,
    build_symmetric_walsh_hadamard_codebook,
    build_wave78_variant_adapter,
)


def test_symmetric_walsh_hadamard_codebook_and_purity_copy_gate() -> None:
    """Verify exact self-overlap, bounded worst-case collision, `<BOS>` firewall, & diffuse-repeat suppression."""
    codebook = build_symmetric_walsh_hadamard_codebook(VALID_FINEWEB_VOCAB_SIZE).float()
    assert codebook.shape == (VALID_FINEWEB_VOCAB_SIZE, 64)
    row_norms = codebook.norm(dim=-1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1.0e-3)

    # Check exact worst-case p=3 collision across all 50,256 non-zero comparisons against token 0:
    dots_vs_0 = torch.matmul(codebook[1:], codebook[0])
    max_p3_collision = float(dots_vs_0.square().pow(3.0).amax().item())
    assert max_p3_collision < 0.025, f"Worst-case WH p=3 collision too high: {max_p3_collision}"

    gate = PurityConditionedHybridWalshCopyGate(
        sem_ket_dim=16, ctx_dim=32, vocab_size=4096, copy_mode="hybrid_walsh_purity"
    )
    # Seq 0: unique unseen rare entity match (token 3777 appears once at pos 3, queried at pos 15)
    # Seq 1: 10-way diffuse function-word repeat (token 12 appears 10 times before pos 15)
    idx = torch.arange(100, 116, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    idx[0, 1] = 3777
    idx[0, 5] = FINEWEB_BOS_TOKEN_ID
    idx[0, 8] = 3777  # in-doc single match -> should copy pos 9 with zero pre-BOS leak from pos 2!
    idx[0, 15] = 3777

    idx[1, 5] = FINEWEB_BOS_TOKEN_ID
    for pos in (6, 7, 8, 9, 10, 11, 12, 13):
        idx[1, pos] = 12
    idx[1, 15] = 12

    torch.manual_seed(7)
    raw = torch.randn(4096, 32)
    sem_emb = raw[idx.clamp(max=4095)]
    sem_re, sem_im = normalize_complex_ket(sem_emb[..., :16], sem_emb[..., 16:])
    ctx = torch.zeros(2, 16, 32)
    mask_pack = build_bos_anchor_sink_doc_masks(idx)

    copy_lo, copy_conf = gate(idx, sem_re, sem_im, ctx, mask_pack.doc_same)
    # Pre-<BOS> position s=2 (following pre-BOS 3777 at s-1=1) must receive strictly 0.0 copy log-odds:
    assert float(copy_lo[0, 15, 2].abs().item()) == 0.0
    # In-doc unique match at s=9 (following 3777 at s-1=8) must dominate:
    unique_peak = float(copy_lo[0, 15, 9].item())
    diffuse_max_impostor = float(copy_lo[1, 15, :15].amax().item())
    assert unique_peak > 1.0
    assert diffuse_max_impostor < 0.25 * unique_peak
    assert float(copy_conf[0, 15].item()) > 0.50


def test_horizon_pooled_mtp_mps_100pct_active_gradients_and_u0_cleanliness() -> None:
    """Verify every parameter in `U3` (including all slices of `HorizonPooledCausalMTPMPSSweep`) gets grad."""
    dev = torch.device("cpu")
    u0 = build_wave78_variant_adapter(
        Wave78VariantKind.U0_R4C_BASE_4L_D256, device=dev, seed=11, complex_dim_override=32
    )
    assert u0.unified_model.copy_gate.wh_metric_logits is None
    assert u0.unified_model.copy_gate.wh_blend_logit is None

    u3 = build_wave78_variant_adapter(
        Wave78VariantKind.U3_UNIFIED_R4C_R5B_4L_D256, device=dev, seed=11, complex_dim_override=32
    )
    gen = torch.Generator().manual_seed(99)
    # Run 2 warmup steps so zero-init residual projections (lm_head, out_proj, mtp_out_proj, even_odd_out_proj) activate:
    for step in (1, 2):
        xb = torch.randint(0, 2048, (4, 24), generator=gen)
        yb = torch.randint(0, 2048, (4, 24), generator=gen)
        xb[:, 10] = FINEWEB_BOS_TOKEN_ID
        yb[:, 9] = FINEWEB_BOS_TOKEN_ID
        u3.train_step(xb, yb, step=step, total_steps=20)
        u0.train_step(xb, yb, step=step, total_steps=20)

    # On Step 3, inspect gradients BEFORE step clear:
    for opt in u3.optimizers:
        opt.zero_grad(set_to_none=True)
    xb3 = torch.randint(0, 2048, (4, 24), generator=gen)
    yb3 = torch.randint(0, 2048, (4, 24), generator=gen)
    _, loss3 = u3.unified_model(xb3, targets=yb3, ingest_online=True, return_logits=False)
    assert loss3 is not None and torch.isfinite(loss3)
    loss3.backward()

    for name, p in u3.unified_model.named_parameters():
        assert p.grad is not None, f"Missing gradient on {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient on {name}"
        assert float(p.grad.abs().max().item()) > 0.0, f"Zero gradient on {name}"

    mps = u3.unified_model.causal_mtp_sweep
    assert mps is not None
    for k in range(mps.fwd_site_to_bond.size(0)):
        assert float(mps.fwd_site_to_bond.grad[k].abs().max().item()) > 0.0  # type: ignore[index]
    for k in range(mps.fwd_bond_to_site.size(0)):
        assert float(mps.fwd_bond_to_site.grad[k].abs().max().item()) > 0.0  # type: ignore[index]


def test_u0_to_u4_strict_causality_no_future_leakage() -> None:
    """Ensure mutating future tokens `idx[:, t+1:]` has `< 1e-5` effect on `logits[:, :t+1]` across `U0..U4`."""
    dev = torch.device("cpu")
    gen = torch.Generator().manual_seed(42)
    for variant in Wave78VariantKind:
        adapter = build_wave78_variant_adapter(
            variant, device=dev, seed=21, complex_dim_override=32
        )
        # Take 2 steps so all zero-initialized output projections become active non-zero matrices!
        for step in (1, 2):
            xb = torch.randint(0, 1024, (2, 20), generator=gen)
            yb = torch.randint(0, 1024, (2, 20), generator=gen)
            adapter.train_step(xb, yb, step=step, total_steps=10)

        adapter.unified_model.eval()
        x_a = torch.randint(0, 1024, (2, 20), generator=gen)
        x_a[0, 6] = FINEWEB_BOS_TOKEN_ID
        x_b = x_a.clone()
        t_cut = 11
        x_b[:, t_cut + 1 :] = torch.randint(2000, 4000, (2, 20 - (t_cut + 1)), generator=gen)
        x_b[1, t_cut + 2] = FINEWEB_BOS_TOKEN_ID

        with torch.no_grad():
            lg_a, _ = adapter.unified_model(x_a, targets=None, ingest_online=False, return_logits=True)
            lg_b, _ = adapter.unified_model(x_b, targets=None, ingest_online=False, return_logits=True)
        assert lg_a is not None and lg_b is not None
        max_diff = float((lg_a[:, : t_cut + 1] - lg_b[:, : t_cut + 1]).abs().max().item())
        assert max_diff < 1.0e-5, f"Future leakage ({max_diff}) detected in {variant.value}"


def test_wave78_copy_and_complementary_schur_diagnostics_smoke() -> None:
    """Verify both `Wave 78` diagnostic functions execute cleanly on `UnifiedMuPCarbonBatonLM`."""
    dev = torch.device("cpu")
    adapter = build_wave78_variant_adapter(
        Wave78VariantKind.U3_UNIFIED_R4C_R5B_4L_D256, device=dev, seed=5, complex_dim_override=32
    )
    gen = torch.Generator().manual_seed(123)
    x_val = torch.randint(0, 512, (4, 32), generator=gen)
    y_val = torch.randint(0, 512, (4, 32), generator=gen)
    x_val[:, 12] = FINEWEB_BOS_TOKEN_ID
    # Guarantee repeated tokens in-doc for multiplicity buckets:
    x_val[0, 14] = 77
    x_val[0, 15] = 88
    x_val[0, 20] = 77
    y_val[0, 20] = 88

    adapter.train_step(x_val, y_val, step=1, total_steps=10)
    adapter.train_step(x_val, y_val, step=2, total_steps=10)

    decile_map = torch.arange(VALID_FINEWEB_VOCAB_SIZE, dtype=torch.long) % 10
    copy_rep = inspect_copy_gate_confidence_and_entity_calibration(
        adapter.model, x_val, y_val, decile_map=decile_map, model_label="u3_smoke", micro_batch=2
    )
    assert copy_rep["eval_tokens"] == 128
    assert "D6_D9_rare_entities_copyable" in copy_rep["by_zipf_and_entity_tier"]
    assert "m_0_out_of_doc_recall" in copy_rep["by_prior_doc_multiplicity"]

    ref_prof = TeacherReferenceProfile(
        anchor_name="smoke_teacher",
        anchor_val_ce=3.5,
        token_nll=torch.full((4, 32), 3.5),
        token_entropy=torch.linspace(0.5, 5.5, steps=128).view(4, 32),
    )
    schur_rep = profile_complementary_schur_pand_and_purity_layers(
        adapter.model,
        x_val,
        y_val,
        decile_map=decile_map,
        ref_profile=ref_prof,
        model_label="u3_smoke",
        micro_batch=2,
    )
    assert len(schur_rep["per_layer_stats"]) == 4
    assert "delta_ce_drop_odd_overall" in schur_rep["packet_ablation"]
    for blk in adapter.unified_model.blocks:
        assert blk.schur_gate is not None
        assert len(blk.schur_gate._forward_hooks) == 0
        assert len(blk.schur_gate.even_odd_out_proj._forward_pre_hooks) == 0
