"""Unit tests for StiefelMuon optimizer and Quantum-Geometric Telemetry Probe."""

from __future__ import annotations

import torch
import torch.nn as nn

from qdlblocks import (
    QuantumBornGPT,
    StiefelMuon,
    inspect_quantum_model,
    polar_snap_weights,
    split_muon_adamw_params,
    stiefel_isometry_error,
)


def test_stiefel_muon_riemannian_retraction_reduces_isometry_error() -> None:
    torch.manual_seed(20)
    # Create a badly scaled non-orthogonal matrix (singular values ~ 3.0)
    W = nn.Parameter(torch.randn(32, 32) * 0.6)
    err_initial = stiefel_isometry_error(W)
    assert err_initial > 1.0

    opt = StiefelMuon(
        [W],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0,
        decay_mode="none",
        stiefel_penalty=0.08,
    )

    for _ in range(60):
        opt.zero_grad()
        # Small dummy gradient
        W.grad = torch.randn_like(W) * 0.01
        opt.step()

    err_final = stiefel_isometry_error(W)
    assert err_final < err_initial * 0.05
    assert err_final < 0.15


def test_stiefel_muon_cautious_and_smooth_modes() -> None:
    torch.manual_seed(21)
    for mode in ("smooth", "cautious"):
        W = nn.Parameter(torch.randn(24, 16) * 0.2)
        opt = StiefelMuon(
            [W],
            lr=0.01,
            weight_decay=0.05,
            decay_mode=mode,  # type: ignore[arg-type]
            aspect_scale=True,
        )
        opt.zero_grad()
        W.grad = torch.randn_like(W)
        w_before = W.detach().clone()
        opt.step()
        assert not torch.allclose(W, w_before)


def test_inspect_quantum_model_and_param_splitting() -> None:
    torch.manual_seed(22)
    model = QuantumBornGPT(
        vocab_size=64,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        block_size=16,
    )

    muon_params, adamw_params = split_muon_adamw_params(model)
    assert len(muon_params) > 0
    assert len(adamw_params) > 0
    assert all(p.ndim == 2 for p in muon_params)

    x = torch.randint(0, 64, (2, 12))
    y = torch.randint(0, 64, (2, 12))

    report = inspect_quantum_model(model, x, y, print_report=False)
    assert "mean_isometry_error" in report
    assert "layer_effective_ranks" in report
    assert "Layer_00" in report["layer_effective_ranks"]
    assert "logit_stats" in report
    assert report["logit_stats"]["eval_cross_entropy"] > 0.0

    # Test polar_snap_weights
    polar_snap_weights(model)
    assert "layer_bipartite_entropies" in report
    assert "mean_bipartite_entropy_nats" in report
    assert "quantum_scales" in report
    assert "gamma_head_mean" in report["quantum_scales"]
    assert "povm_beta" in report["quantum_scales"]


def test_wsd_schedule_and_cooldown_branching() -> None:
    from qdlblocks import (
        ProgressiveSequenceCurriculum,
        branch_wsd_cooldown_multiplier,
        cosine_lr_multiplier,
        wsd_lr_multiplier,
    )

    # Warmup phase
    assert abs(wsd_lr_multiplier(50, 1000, warmup_steps=100, decay_fraction=0.2) - 0.5) < 1e-6
    # Stable plateau phase (steps 100..800)
    for s in (100, 300, 500, 800):
        assert abs(wsd_lr_multiplier(s, 1000, warmup_steps=100, decay_fraction=0.2) - 1.0) < 1e-6
    # Radical 1 - sqrt(t / T_decay) cooldown phase (steps 800..1000)
    m_mid = wsd_lr_multiplier(900, 1000, warmup_steps=100, decay_fraction=0.2, min_lr_ratio=0.1)
    assert 0.1 < m_mid < 1.0
    m_end = wsd_lr_multiplier(1000, 1000, warmup_steps=100, decay_fraction=0.2, min_lr_ratio=0.1)
    assert abs(m_end - 0.1) < 1e-6

    # Branching cooldown over 200 steps
    assert abs(branch_wsd_cooldown_multiplier(0, 200, min_lr_ratio=0.1) - 1.0) < 1e-6
    assert abs(branch_wsd_cooldown_multiplier(200, 200, min_lr_ratio=0.1) - 0.1) < 1e-6
    assert 0.1 < cosine_lr_multiplier(500, 1000, warmup_steps=100, min_lr_ratio=0.1) <= 1.0

    # Progressive sequence curriculum preserves B * T == tokens_per_step
    curr = ProgressiveSequenceCurriculum(tokens_per_step=65536, seq_lengths=(256, 512, 1024, 2048))
    for ep in range(4):
        b_ep, t_ep = curr.for_epoch(ep, 4)
        assert b_ep * t_ep == 65536
        assert t_ep == (256, 512, 1024, 2048)[ep]


def test_distributed_stiefel_muon_and_10qubit_entanglement() -> None:
    from qdlblocks import (
        DistributedStiefelMuon,
        bipartite_entanglement_entropy,
        l2_normalize,
        partial_trace_qubits,
    )

    torch.manual_seed(77)
    p1 = nn.Parameter(torch.randn(32, 16) * 0.5)
    p2 = nn.Parameter(torch.randn(16, 32) * 0.5)
    opt = DistributedStiefelMuon([p1, p2], lr=0.02, stiefel_penalty=0.05, ns_steps=5)
    opt.zero_grad()
    p1.grad = torch.randn_like(p1)
    p2.grad = torch.randn_like(p2)
    p1_init = p1.detach().clone()
    opt.step()
    assert not torch.allclose(p1, p1_init)

    # Fast 10-qubit (d=1024, 5+5 split -> 32x32 reduced density matrix) pure-state partial trace
    psi_10q = l2_normalize(torch.randn(4, 1024), dim=-1)
    rho_a = partial_trace_qubits(psi_10q, num_qubits=10, keep_qubits=(0, 1, 2, 3, 4))
    assert rho_a.shape == (4, 32, 32)
    ent = bipartite_entanglement_entropy(psi_10q, num_qubits=10, keep_qubits=(0, 1, 2, 3, 4))
    assert ent.shape == (4,)
    assert torch.all((ent >= 0.0) & (ent <= 5.0 * 0.69315 + 1e-3))

