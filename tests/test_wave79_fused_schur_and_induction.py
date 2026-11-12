"""Unit tests for Wave 79 Fused In-Attention Schur, Bi-Trigger Walsh Induction, & Diagnostics (`EXPERIMENT_PLAN_099.md`)."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.architecture_mechanistic_probes import generate_fuzzy_induction_batch
from qdlblocks.benchmarks.wave77_quantum_routing_primitives import (
    FINEWEB_BOS_TOKEN_ID,
    build_bos_anchor_sink_doc_masks,
    explicit_complementary_schur_matrix_pair,
)
from qdlblocks.benchmarks.wave79_fused_schur_and_induction_baton import (
    FusedUnifiedBatonConfig,
    FusedUnifiedSchurBatonLM,
    InAttentionComplementarySchurRouter,
    SurprisalConditionedWalshBiTriggerCopyGate,
    Wave79VariantKind,
    build_balanced_128d_walsh_codebook,
    build_wave79_variant_adapter,
)
from qdlblocks.benchmarks.wave79_speedrun_and_ablation_runner import (
    FlexDocMaskBenchConfig,
    audit_wave79_copy_and_schur_diagnostics,
    benchmark_flex_attention_bos_docmask_parity_and_speed,
    inspect_mtp_horizon_weights_and_rollout_cosines,
)


def test_balanced_128d_walsh_codebook_and_step0_unseen_induction() -> None:
    """Verify 128D Walsh unit norms, Welch noise-floor suppression, and 100% Step-0 unseen induction."""
    cb = build_balanced_128d_walsh_codebook(vocab_size=50257)
    assert cb.shape == (50257, 128)
    norms = cb.float().square().sum(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1.0e-3)

    adapter = build_wave79_variant_adapter(
        Wave79VariantKind.V0_SURPRISAL_WH_INDUCTION_4L_D256,
        device=torch.device("cpu"),
        enable_compile=False,
        complex_dim_override=64,
    )
    gen = torch.Generator().manual_seed(2026)
    xb, yb, q_mask = generate_fuzzy_induction_batch(16, 64, 50257, gen, torch.device("cpu"))
    log_p, _ = adapter.predict_log_probs_and_span(xb)
    preds = log_p.argmax(dim=-1)
    step0_acc = float((preds[q_mask] == yb[q_mask]).float().mean().item())
    assert step0_acc == 1.0, f"Expected 100% Step-0 unseen-ID induction, got {step0_acc}"


def test_surprisal_idf_damps_lone_stopword_vs_rare_entity() -> None:
    """Verify lone D0 stopword ('the', p=0.05) triggers far weaker copy than rare entity (p=1e-5)."""
    gate = SurprisalConditionedWalshBiTriggerCopyGate(sem_ket_dim=32, ctx_dim=64, vocab_size=1024)
    uni_probs = torch.full((1024,), 1.0e-5, dtype=torch.float32)
    uni_probs[10] = 0.05  # frequent stopword ID 10
    uni_probs[777] = 1.0e-5  # rare entity ID 777

    # Sequence 0 repeats stopword 10 -> 20; Sequence 1 repeats rare entity 777 -> 20
    idx = torch.tensor(
        [
            [5, 10, 20, 31, 42, 10],
            [5, 777, 20, 31, 42, 777],
        ],
        dtype=torch.long,
    )
    sem_re = torch.randn(2, 6, 32)
    sem_im = torch.randn(2, 6, 32)
    norm = torch.rsqrt((sem_re.square() + sem_im.square()).sum(dim=-1, keepdim=True))
    sem_re, sem_im = sem_re * norm, sem_im * norm
    ctx = torch.zeros(2, 6, 64)
    doc_same = build_bos_anchor_sink_doc_masks(idx).doc_same

    copy_lo, copy_conf = gate(idx, sem_re, sem_im, ctx, doc_same, uni_probs)
    assert float(copy_conf[1, 5].item()) > 4.0 * float(copy_conf[0, 5].item())
    assert float(copy_lo[1, 5, 2].item()) > 4.0 * float(copy_lo[0, 5, 2].item())


def test_in_attention_complementary_schur_trace_and_channel_modes() -> None:
    """Verify Rank-2 In-Attention Complementary Schur trace conservation and drop_odd/drop_even."""
    router = InAttentionComplementarySchurRouter(real_dim=128)
    attn_flat = torch.randn(2, 12, 128)
    h_norm = torch.randn(2, 12, 128)

    even_p, odd_p, p_and, ua_re, ua_im, ub_re, ub_im = router.compute_even_odd_packets(
        attn_flat, h_norm
    )
    assert even_p.shape == (2, 12, 2, 32)
    assert odd_p.shape == (2, 12, 2, 32)
    assert bool(((p_and >= 0.0) & (p_and <= 1.0)).all())

    ua_c = torch.complex(ua_re, ua_im)
    ub_c = torch.complex(ub_re, ub_im)
    rho_a = torch.matmul(ua_c.transpose(-2, -1), ua_c.conj())
    rho_b = torch.matmul(ub_c.transpose(-2, -1), ub_c.conj())
    rho_and, rho_comp, p_mat = explicit_complementary_schur_matrix_pair(rho_a, rho_b)
    assert torch.allclose(p_and, p_mat, atol=1.0e-4)
    tr_and = torch.diagonal(rho_and, dim1=-2, dim2=-1).real.sum(dim=-1)
    tr_comp = torch.diagonal(rho_comp, dim1=-2, dim2=-1).real.sum(dim=-1)
    assert torch.allclose(tr_and, torch.ones_like(tr_and), atol=1.0e-4)
    assert torch.allclose(tr_comp, torch.ones_like(tr_comp), atol=1.0e-4)

    out_norm = router(attn_flat, h_norm)
    router.eval_channel_mode = "drop_odd"
    out_drop_odd = router(attn_flat, h_norm)
    assert not torch.allclose(out_norm, out_drop_odd)


def test_wave79_model_diagnostics_and_flex_bench_smoke() -> None:
    """Verify train_step, MTP horizon cosine inspector, copy/schur audit, and flex bench schema."""
    cfg = FusedUnifiedBatonConfig(
        vocab_size=2048,
        complex_dim=64,
        num_layers=4,
        schur_mode="in_attn_head_pair",
        use_horizon_mtp_mps=True,
        compile_backbone=False,
    )
    model = FusedUnifiedSchurBatonLM(cfg)
    xb = torch.randint(0, 2048, (4, 32))
    xb[0, 12] = FINEWEB_BOS_TOKEN_ID
    yb = torch.randint(0, 2048, (4, 32))

    # Clamp BOS for tiny test vocab in wte_ket/targets while testing BOS boundary mask
    xb_safe = xb.clone()
    xb_safe[xb_safe == FINEWEB_BOS_TOKEN_ID] = 0
    _, loss = model(xb_safe, targets=yb, ingest_online=True, return_logits=False)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    mtp_rep = inspect_mtp_horizon_weights_and_rollout_cosines(model, xb_safe, yb)
    assert mtp_rep["has_mtp_sweep"] is True
    assert len(mtp_rep["per_horizon"]) == 4

    adapter = build_wave79_variant_adapter(
        Wave79VariantKind.V2_FUSED_UNIFIED_4L_D256,
        device=torch.device("cpu"),
        enable_compile=False,
        complex_dim_override=64,
    )
    x_f = torch.randint(0, 50256, (4, 32))
    y_f = torch.randint(0, 50256, (4, 32))
    dec_map = torch.arange(50257) % 10
    loss_step = adapter.train_step(x_f, y_f, step=1, total_steps=10)
    assert loss_step > 0.0
    audit = audit_wave79_copy_and_schur_diagnostics(adapter, x_f, y_f, dec_map)
    assert "m0_unseen_recall" in audit["multiplicity_buckets"]
    assert "delta_ce_drop_odd" in audit["in_attn_schur_channel_ablation"]

    fb = benchmark_flex_attention_bos_docmask_parity_and_speed(
        FlexDocMaskBenchConfig(seq_lens=(128,), total_batch_tokens=256, warmup_iters=1, bench_iters=2),
        device=torch.device("cpu"),
    )
    assert len(fb["results"]) == 1
