"""Unit and Hypothesis Verification Tests for Experiment Plan 029 (`synthetic_fineweb_walk.py`)."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.synthetic_fineweb_walk import (
    FineWebSyntheticWalkConfig,
    NaiveQuadraticBornAttnLM,
    build_fast_quantum_born_synth_lm,
    evaluate_oracle_markov_bigram_floor,
    generate_fineweb_synthetic_random_walk,
    run_task_b_variable_lag_hub_benchmark,
    train_and_evaluate_task_a_arm,
)


def test_synthetic_fineweb_walk_statistical_invariants() -> None:
    """Verify Hypothesis H1: `V = 50,257` FineWeb-matched synthetic random-walk statistical fidelity."""
    cfg = FineWebSyntheticWalkConfig(seed=1337)
    out = generate_fineweb_synthetic_random_walk(num_seqs=64, seq_len=128, config=cfg)
    st = out["stats"]

    assert st["vocab_valid"] == 50257
    assert st["total_tokens"] == 8192
    assert 7.00 <= st["empirical_unigram_entropy_nats"] <= 8.20
    assert 7.40 <= st["theoretical_zipf_entropy_nats"] <= 7.85
    assert 0.35 <= st["top100_mass_fineweb_rank"] <= 0.52
    assert 0.35 <= st["top100_mass_empirical"] <= 0.52
    assert 0.55 <= st["top1000_mass_fineweb_rank"] <= 0.75
    assert 0.55 <= st["top1000_mass_empirical"] <= 0.75
    assert st["unique_tokens"] >= 2000
    assert st["max_non_eot_token_id"] >= 50000
    assert st["eot_token_count"] >= 8
    assert st["empirical_bigram_cond_entropy_nats"] >= 1.35
    assert st["oracle_markov_bigram_top1_acc"] <= 48.0


def test_fast_quantum_attention_overfits_full_50304_vocab_walk() -> None:
    """Verify Hypothesis H2: `FastQuantumBornAttentionSynthLM` overfits `V = 50,304` walk below Markov floor."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)

    cfg = FineWebSyntheticWalkConfig(seed=1337)
    walk = generate_fineweb_synthetic_random_walk(num_seqs=64, seq_len=128, config=cfg)
    tokens = walk["tokens"]
    oracle = evaluate_oracle_markov_bigram_floor(tokens)

    # 1. Naive k=1 Linear Born + Quadratic Attention stays trapped >= 8.50 nats
    naive = NaiveQuadraticBornAttnLM(vocab_size=50304, embed_dim=256, num_heads=4, num_layers=2).to(dev)
    res_naive = train_and_evaluate_task_a_arm(
        "test_naive", naive, tokens, steps=40, batch_size=16, eval_every=40
    )
    assert res_naive["final_ce"] >= 8.50

    # 2. FastQuantumNoAttnMLP (skip_attention=True, use_embed_causal_memory=False) is bounded by Markov floor
    no_attn = build_fast_quantum_born_synth_lm(
        vocab_size=50304,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        block_size=128,
        skip_attention=True,
        use_embed_causal_memory=False,
        seed=1337,
        device=dev,
    )
    res_no_attn = train_and_evaluate_task_a_arm(
        "test_no_attn", no_attn, tokens, steps=160, batch_size=16, eval_every=160
    )
    assert res_no_attn["final_ce"] >= oracle["cond_entropy_nats"] - 0.02

    # 3. FastQuantumBornAttentionSynthLM overfits to <= 0.10 nats and >= 98.5% Top-1
    q_attn = build_fast_quantum_born_synth_lm(
        vocab_size=50304,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        block_size=128,
        skip_attention=False,
        use_embed_causal_memory=False,
        seed=1337,
        device=dev,
    )
    res_q_attn = train_and_evaluate_task_a_arm(
        "test_q_attn",
        q_attn,
        tokens,
        steps=240,
        batch_size=16,
        lr_embed_readout=0.018,
        lr_backbone=0.007,
        eval_every=240,
    )
    assert res_q_attn["final_ce"] <= 0.10
    assert res_q_attn["top1_acc"] >= 98.5
    assert res_q_attn["top5_acc"] >= 99.8
    assert res_q_attn["beam_ortho"] < 1e-5
    assert res_q_attn["peak_vram_gb"] <= 8.0


def test_position_shifted_variable_lag_hub_and_radial_quantum_attention() -> None:
    """Verify Hypothesis H3: Position-shifted variable-lag (`Delta in [8, 32]`) hub & collinear radial disambiguation."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)

    res_b = run_task_b_variable_lag_hub_benchmark(steps=120, batch_size=32, seq_len=64, seed=2029, device=dev)
    assert res_b["OracleMarkovBigramFloor"]["hub_top1_acc"] <= 20.0
    assert res_b["FastQuantumNoAttnMLP"]["hub_top1_acc"] <= 25.0
    assert res_b["FastQuantumBornAttentionSynthLM"]["hub_top1_acc"] >= 95.0
    assert res_b["FastQuantumBornAttentionSynthLM"]["hub_ce"] <= 0.25

    rad = res_b["TaskB_CollinearRadialSubProbe"]
    assert rad["stereographic_lift_true"]["eval_acc"] >= 95.0
    assert rad["naive_radial_erasing_false"]["eval_acc"] <= 58.0
