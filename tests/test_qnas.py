"""Unit tests for Track 2 Quantum-Geometric Neural Architecture Search (qdlblocks.qnas)."""

from __future__ import annotations

import numpy as np
import torch

from qdlblocks.qnas import (
    OP_NAMES,
    ArchConfig,
    DiscreteQuantumBackbone,
    EntropyAdaptiveNetwork,
    QuantumSearchCell,
    QuantumZeroShotNAS,
    SuperpositionDARTSNetwork,
    anneal_purification_power,
    compute_qnas_proxy_score,
    kendall_tau_corr,
    run_qnas_comparison,
    sample_random_arch,
    spearman_rank_corr,
)


def test_search_space_and_discrete_backbone():
    torch.manual_seed(42)
    cfg = ArchConfig(
        in_features=64,
        num_tokens=8,
        embed_dim=32,
        num_heads=4,
        num_classes=4,
        cell_ops=("born_attn_real", "sparse_born_moe", "projective_mlp"),
    )
    model = DiscreteQuantumBackbone(cfg)
    x = torch.randn(6, 64)
    y = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)

    logits, layer_states = model(x, return_layer_states=True)
    assert logits.shape == (6, 4)
    assert len(layer_states) == 3
    norms = torch.linalg.vector_norm(layer_states[-1], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    loss = torch.nn.functional.cross_entropy(logits, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_superposition_darts_and_purification_collapse():
    torch.manual_seed(11)
    cell = QuantumSearchCell(
        embed_dim=32,
        num_heads=4,
        block_size=16,
        mixing_mode="quantum_superposition",
    )
    probs_p1 = cell.get_op_probabilities(purification_power=1.0)
    probs_p4 = cell.get_op_probabilities(purification_power=4.0)
    assert probs_p1.shape == (len(OP_NAMES),)
    assert torch.allclose(probs_p1.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(probs_p4.sum(), torch.tensor(1.0), atol=1e-5)

    with torch.no_grad():
        cell.alpha_raw.copy_(torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1]))
    p1 = cell.get_op_probabilities(purification_power=1.0)
    p3 = cell.get_op_probabilities(purification_power=3.0)
    assert p3[0] > p1[0]

    supernet = SuperpositionDARTSNetwork(
        in_features=64,
        num_tokens=8,
        embed_dim=32,
        num_heads=4,
        num_cells=2,
        num_classes=3,
        mixing_mode="quantum_superposition",
    )
    x = torch.randn(5, 64)
    y = torch.tensor([0, 1, 2, 0, 1])
    logits = supernet(x, purification_power=anneal_purification_power(5, 10))
    loss = torch.nn.functional.cross_entropy(logits, y)
    assert logits.shape == (5, 3) and torch.isfinite(loss)
    loss.backward()

    disc = supernet.export_discrete_model()
    assert isinstance(disc, DiscreteQuantumBackbone)
    assert len(disc.config.cell_ops) == 2


def test_entropy_adaptive_kraus_network():
    torch.manual_seed(19)
    net = EntropyAdaptiveNetwork(
        in_features=16,
        stem_dim=6,
        init_dims=(6, 6, 6),
        num_classes=3,
        num_kraus=2,
        min_dim=4,
        max_dim=12,
        delta_dim=2,
        expand_entropy_ratio=0.30,  # Low threshold to trigger expansion on mixed states
    )
    x = torch.randn(8, 16)
    logits, densities = net(x, return_densities=True)
    assert logits.shape == (8, 3)
    assert len(densities) == 3

    rec = net.step_topology_adaptation(x)
    assert "events" in rec
    assert len(net.adaptation_history) == 1

    logits_after = net(x)
    assert logits_after.shape == (8, 3)
    assert torch.all(torch.isfinite(logits_after))


def test_zero_shot_proxy_and_comparison_runner():
    torch.manual_seed(23)
    rng = np.random.default_rng(23)
    x = torch.randn(12, 64)
    y = torch.randint(0, 3, (12,))
    cfg = sample_random_arch(rng, in_features=64, num_tokens=8, embed_dim=32, num_heads=4, num_classes=3)
    model = DiscreteQuantumBackbone(cfg)
    metrics = compute_qnas_proxy_score(model, x, y)
    assert "total_score" in metrics
    assert "norm_effective_rank" in metrics
    assert "log_synflow" in metrics
    assert "isometry_error" in metrics

    searcher = QuantumZeroShotNAS(in_features=64, num_tokens=8, embed_dim=32, num_heads=4, num_classes=3)
    best_cfg, history = searcher.search(x, y, num_candidates=4, seed=23)
    assert isinstance(best_cfg, ArchConfig)
    assert len(history) == 4

    assert abs(spearman_rank_corr([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) - 1.0) < 1e-6
    assert abs(kendall_tau_corr([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) - 1.0) < 1e-6

    out = run_qnas_comparison(
        device="cpu",
        epochs=2,
        num_zero_shot_candidates=4,
        seed=42,
    )
    assert len(out["results"]) == 4
    assert "spearman_rho" in out["proxy_rank_correlations"]
