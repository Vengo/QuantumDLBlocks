"""Unit tests for Experiment Plan 025: Target-Rank Cross-Entropy Decomposition, Layerwise Training Dynamics, & Embed-Only Causal Hierarchy Memory."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks import (
    RANK_BUCKETS,
    compute_calibration_rank_decomposition,
    compute_layerwise_dynamics_profile,
    parse_cli_args_to_card_sections,
)
from qdlblocks.layers.hierarchical_qubit import Hierarchical16QubitEmbedder
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT


def test_compute_calibration_rank_decomposition_exact_partition_identity() -> None:
    torch.manual_seed(2025)
    n_tokens = 512
    vocab_padded = 2048
    valid_vocab = 1800

    logits = torch.randn(n_tokens, vocab_padded, dtype=torch.float32) * 3.5
    targets = torch.randint(0, valid_vocab, (n_tokens,), dtype=torch.long)
    # Ensure every rank bucket is populated
    for i in range(0, 64):
        logits[i, targets[i]] = float(logits[i, :valid_vocab].max().item() + 2.0)

    res = compute_calibration_rank_decomposition(
        logits,
        targets,
        valid_vocab_size=valid_vocab,
        num_ece_bins=10,
        chunk_size=128,
    )

    assert res["num_tokens"] == n_tokens
    assert res["valid_vocab_size"] == valid_vocab

    buckets = res["buckets"]
    assert set(buckets.keys()) == {b[0] for b in RANK_BUCKETS}

    frac_sum = sum(float(b["fraction"]) for b in buckets.values())
    cnt_sum = sum(int(b["count"]) for b in buckets.values())
    ce_sum = sum(float(b["ce_contribution"]) for b in buckets.values())

    assert cnt_sum == n_tokens
    assert abs(frac_sum - 1.0) < 1e-6

    ref_ce = float(F.cross_entropy(logits[:, :valid_vocab], targets, reduction="mean").item())
    assert abs(res["total_ce"] - ref_ce) < 1e-5
    assert abs(ce_sum - res["total_ce"]) < 1e-5
    assert res["ce_decomposition_error"] < 1e-5

    prob_mass_sum = (
        res["mean_top1_prob_mass"]
        + res["mean_ranks_2_5_prob_mass"]
        + res["mean_ranks_6_50_prob_mass"]
        + res["mean_tail_gt_50_prob_mass"]
    )
    assert abs(prob_mass_sum - 1.0) < 1e-4

    assert 0.0 <= res["ece"] <= 1.0
    assert abs(res["top5_minus_top1_gap_pct"] - (res["top5_acc_pct"] - res["top1_acc_pct"])) < 1e-5
    assert res["mean_prediction_entropy_nats"] > 0.0
    assert res["logit_std"] > 0.0


def test_compute_layerwise_dynamics_profile_classical_and_quantum() -> None:
    torch.manual_seed(42)
    vocab_size = 128
    embed_dim = 64
    num_heads = 4
    num_layers = 3
    seq_len = 16

    x = torch.randint(0, vocab_size, (4, seq_len), dtype=torch.long)
    y = torch.randint(0, vocab_size, (4, seq_len), dtype=torch.long)

    classical_model = ClassicalModdedGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=256,
        block_size=seq_len,
        leaderboard_exact=True,
        num_value_embeddings=1,
    )
    c_dyn = compute_layerwise_dynamics_profile(classical_model, x, y, valid_vocab_size=vocab_size)
    assert c_dyn["num_layers"] == num_layers
    assert len(c_dyn["layer_angular_velocity_deg"]) == num_layers
    assert len(c_dyn["layer_effective_ranks"]) == num_layers
    assert len(c_dyn["layer_participation_ratios"]) == num_layers
    assert all(math.isfinite(v) and v >= 0.0 for v in c_dyn["layer_angular_velocity_deg"])
    assert all(math.isfinite(r) and r >= 1.0 for r in c_dyn["layer_effective_ranks"])
    assert c_dyn["grad_norm_embed_readout"] > 0.0
    assert c_dyn["grad_norm_backbone_muon"] > 0.0
    assert all(p.grad is None for p in classical_model.parameters())

    quantum_model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=32,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=seq_len,
        readout_mode="asymmetric",
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=2,
        num_higher_states=16,
        use_block_hierarchy_bridge=True,
        bridge_causal_memory=False,
        use_last_layer_softmax=True,
        logit_softcap=15.0,
    )
    q_dyn = compute_layerwise_dynamics_profile(quantum_model, x, y, valid_vocab_size=vocab_size)
    assert q_dyn["num_layers"] == num_layers
    assert len(q_dyn["layer_angular_velocity_deg"]) == num_layers
    assert all(math.isfinite(v) and v > 0.0 for v in q_dyn["layer_angular_velocity_deg"])
    assert all(math.isfinite(r) and r >= 1.0 for r in q_dyn["layer_effective_ranks"])
    assert q_dyn["grad_norm_embed_readout"] > 0.0
    assert q_dyn["grad_norm_backbone_muon"] > 0.0
    assert q_dyn["embed_to_backbone_grad_ratio"] > 0.0
    assert all(p.grad is None for p in quantum_model.parameters())


def test_bridge_causal_memory_flag_and_strict_causality() -> None:
    torch.manual_seed(77)
    vocab_size = 64
    embed_dim = 64
    seq_len = 12

    emb_true = Hierarchical16QubitEmbedder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_registers=16,
        register_dim=2,
        num_higher_states=16,
        num_layers=2,
        use_ortho_decouple=True,
        use_block_hierarchy_bridge=True,
        bridge_causal_memory=True,
        seed=99,
    )
    emb_false = Hierarchical16QubitEmbedder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_registers=16,
        register_dim=2,
        num_higher_states=16,
        num_layers=2,
        use_ortho_decouple=True,
        use_block_hierarchy_bridge=True,
        bridge_causal_memory=False,
        seed=99,
    )

    idx = torch.randint(0, vocab_size, (2, seq_len), dtype=torch.long)
    # Initial embedding forward always applies causal memory for both
    psi0_true = emb_true(idx)
    psi0_false = emb_false(idx)
    assert torch.allclose(psi0_true, psi0_false, atol=1e-6)

    # Intermediate block bridge differs for t >= 1 because bridge_causal_memory=False skips the 4-tap shift
    psi_in = F.normalize(torch.randn(2, seq_len, embed_dim), p=2, dim=-1)
    bridge_true = emb_true.forward_block_bridge(psi_in, layer_idx=0)
    bridge_false = emb_false.forward_block_bridge(psi_in, layer_idx=0)
    # At t=0 there is no prior token, wait: in route_through_hierarchy with include_causal_memory=True,
    # c_mem at t=0 is 0, so c_bar = (1 - alpha_mem) * c_t != c_t, making even t=0 distinct!
    max_bridge_diff = float((bridge_true - bridge_false).abs().max().item())
    assert max_bridge_diff > 1e-4

    # When bridge_causal_memory=False, forward_block_bridge is strictly position-local (token-wise):
    # perturbing token t >= 4 has exact 0.0 effect on positions 0..3 inside forward_block_bridge!
    psi_pert = psi_in.clone()
    psi_pert[:, 6:, :] = F.normalize(torch.randn_like(psi_pert[:, 6:, :]), p=2, dim=-1)
    bridge_false_pert = emb_false.forward_block_bridge(psi_pert, layer_idx=0)
    assert float((bridge_false[:, :6, :] - bridge_false_pert[:, :6, :]).abs().max().item()) == 0.0

    # Full QuantumBornGPT with bridge_causal_memory=False preserves strict autoregressive causality
    q_model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=32,
        num_heads=4,
        num_layers=2,
        block_size=seq_len,
        readout_mode="asymmetric",
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=2,
        num_higher_states=16,
        use_block_hierarchy_bridge=True,
        bridge_causal_memory=False,
        use_last_layer_softmax=True,
        use_activation_checkpointing=True,
    )
    assert q_model.hierarchical_embed is not None
    assert q_model.hierarchical_embed.bridge_causal_memory is False

    q_model.eval()
    idx_pert = idx.clone()
    idx_pert[:, 6:] = (idx_pert[:, 6:] + 7) % vocab_size
    with torch.no_grad():
        logits_a, _ = q_model(idx)
        logits_b, _ = q_model(idx_pert)
    prefix_leakage = float((logits_a[:, :6, :] - logits_b[:, :6, :]).abs().max().item())
    assert prefix_leakage < 1e-5

    # Verify CLI flag parsing for --no-bridge-causal-memory / --embed-only-causal-memory
    arch, _, _ = parse_cli_args_to_card_sections(
        "--architecture quantum_born --hierarchical-16qubit --embed-only-causal-memory --last-layer-softmax"
    )
    assert arch["bridge_causal_memory"] is False


def test_calibration_rank_histogram_probe_cli(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "calibration_rank_histogram_probe.py"
    assert script_path.exists()

    help_res = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--quick-test" in help_res.stdout
    assert "--max-vram-gb" in help_res.stdout
    assert "--out-json" in help_res.stdout

    out_json = tmp_path / "probe_quick.json"
    run_res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--quick-test",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--out-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Target-Rank Cross-Entropy Decomposition" in run_res.stdout
    assert out_json.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["quick_test"] is True
    assert "classical_modded" in payload["variants"]
    assert "quantum_povm" in payload["variants"]
    assert "quantum_softmax" in payload["variants"]
    assert "quantum_w13w15_unified" in payload["variants"]
    for v_data in payload["variants"].values():
        assert v_data["calibration"]["ce_decomposition_error"] < 1e-5
