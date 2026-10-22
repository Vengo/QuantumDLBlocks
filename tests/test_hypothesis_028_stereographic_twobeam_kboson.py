"""Unit tests for Plan 028: Stereographic Radial Lift, Cosine Phase-Interference Two-Beam Readout, Positive-Part Quadratic Kraus Fidelity, Fisher's Iris & 4D Concentric Shells, and Wave 17 2-G4 Fleet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.model_card import (
    WAVE_DEFAULT_LABELS,
    WAVE_DEFAULT_PLANS,
    parse_cli_args_to_card_sections,
)
from qdlblocks.layers.direct_chunked_ce import (
    compute_two_beam_kboson_logits,
    direct_chunked_two_beam_kboson_ce,
)
from qdlblocks.layers.hierarchical_qubit import Hierarchical16QubitPOVMReadout
from qdlblocks.layers.overcomplete_two_beam import OvercompleteTwoBeamInterference
from qdlblocks.layers.stereographic_lift import (
    StereographicHilbertLift,
    run_iris_and_shells_benchmark,
)
from qdlblocks.models.born_gpt import QuantumBornGPT


def test_fisher_iris_and_concentric_shells_stereographic_vs_naive_born() -> None:
    """Verify H1: StereographicTwoBeamKBoson resolves radial norm erasure on 4D Concentric Shells (>=95%) & Fisher's Iris (>=96%)."""
    lift = StereographicHilbertLift(dim=4, init_tau=1.0)
    x = torch.randn(16, 4)
    psi_s = lift(x)
    assert psi_s.shape == (16, 5)
    assert torch.allclose(psi_s.norm(p=2, dim=-1), torch.ones(16), atol=1e-6)
    # Collinear vectors 2x and 0.5x must map to distinct points on S^4
    psi_2x = lift(2.0 * x)
    psi_half = lift(0.5 * x)
    assert float((psi_2x - psi_half).norm(p=2, dim=-1).mean().item()) > 0.25

    res = run_iris_and_shells_benchmark(epochs=220, seed=42, device="cpu")
    shells = res["concentric_shells_4d"]
    iris = res["fisher_iris_4d"]

    assert shells["naive_born_test_acc_pct"] <= 45.0
    assert shells["stereographic_two_beam_test_acc_pct"] >= 95.0
    assert shells["gain_percentage_points"] >= 50.0

    assert iris["naive_born_acc_pct"] <= 86.7
    assert iris["stereographic_two_beam_acc_pct"] >= 96.0
    assert res["max_unit_norm_err"] < 1e-5
    assert res["max_beam_orthogonality_abs"] < 1e-5
    assert res["passed"] is True


def test_quadratic_diffuse_kboson_ce_autograd_parity() -> None:
    """Verify H2.1: _DirectChunkedTwoBeamKBosonCEFn with quad_diffuse_weight=0.50 matches materialized logits + autograd (< 1e-5)."""
    torch.manual_seed(108)
    b, t, d, r, v = 2, 7, 32, 8, 96
    targets = torch.randint(0, v, (b, t), dtype=torch.long)

    for per_token_lam in (False, True):
        psi_sharp_1 = torch.randn(b, t, d, dtype=torch.float32, requires_grad=True)
        psi_diff_1 = torch.randn(b, t, r, dtype=torch.float32, requires_grad=True)
        m_sharp_1 = torch.randn(v, d, dtype=torch.float32, requires_grad=True)
        m_diff_1 = torch.randn(v, r, dtype=torch.float32, requires_grad=True)
        beta_1 = (3.5 + 3.0 * torch.rand(b, t, 1, dtype=torch.float32)).requires_grad_(True)
        k_1 = torch.tensor(12.0, dtype=torch.float32, requires_grad=True)
        if per_token_lam:
            lam_1 = (0.12 + 0.20 * torch.rand(b, t, 1, dtype=torch.float32)).requires_grad_(True)
        else:
            lam_1 = torch.tensor(0.22, dtype=torch.float32, requires_grad=True)
        prior_1 = (0.2 * torch.randn(v, dtype=torch.float32)).requires_grad_(True)

        psi_sharp_2 = psi_sharp_1.detach().clone().requires_grad_(True)
        psi_diff_2 = psi_diff_1.detach().clone().requires_grad_(True)
        m_sharp_2 = m_sharp_1.detach().clone().requires_grad_(True)
        m_diff_2 = m_diff_1.detach().clone().requires_grad_(True)
        beta_2 = beta_1.detach().clone().requires_grad_(True)
        k_2 = k_1.detach().clone().requires_grad_(True)
        lam_2 = lam_1.detach().clone().requires_grad_(True)
        prior_2 = prior_1.detach().clone().requires_grad_(True)

        loss_chunked = direct_chunked_two_beam_kboson_ce(
            psi_sharp_1,
            psi_diff_1,
            m_sharp_1,
            m_diff_1,
            beta_1,
            k_1,
            lam_1,
            targets,
            log_prior=prior_1,
            chunk_size=5,
            quad_diffuse_weight=0.50,
        )
        loss_chunked.backward()

        logits_ref = compute_two_beam_kboson_logits(
            psi_sharp_2,
            psi_diff_2,
            m_sharp_2,
            m_diff_2,
            beta_2,
            k_2,
            lam_2,
            log_prior=prior_2,
            quad_diffuse_weight=0.50,
        )
        loss_ref = F.cross_entropy(logits_ref.reshape(-1, v), targets.reshape(-1))
        loss_ref.backward()

        assert abs(float(loss_chunked.item()) - float(loss_ref.item())) < 1e-5
        for g1, g2, name in (
            (psi_sharp_1.grad, psi_sharp_2.grad, "dpsi_sharp"),
            (psi_diff_1.grad, psi_diff_2.grad, "dpsi_diff"),
            (m_sharp_1.grad, m_sharp_2.grad, "dm_sharp"),
            (m_diff_1.grad, m_diff_2.grad, "dm_diff"),
            (beta_1.grad, beta_2.grad, "dbeta"),
            (k_1.grad, k_2.grad, "dk"),
            (lam_1.grad, lam_2.grad, "dlam"),
            (prior_1.grad, prior_2.grad, "dlog_prior"),
        ):
            assert g1 is not None and g2 is not None, f"Missing gradient for {name}"
            max_diff = float((g1 - g2).abs().max().item())
            assert max_diff < 1e-5, f"Gradient mismatch for {name}: {max_diff}"


def test_stereographic_cosine_phase_two_beam_orthogonality_and_radial_sensitivity() -> None:
    """Verify H2.2: OvercompleteTwoBeamInterference with stereographic lift & cosine phase gate gives unit-norm orthogonal beams and radial sensitivity."""
    torch.manual_seed(77)
    layer = OvercompleteTwoBeamInterference(
        embed_dim=64,
        frame_expansion=2.0,
        use_stereographic_lift=True,
        use_cosine_phase_beam_gate=True,
    )
    x = torch.randn(3, 11, 64)
    x_unit = x / x.norm(p=2, dim=-1, keepdim=True)

    psi_main, psi_beam2 = layer(x_unit, return_second_beam=True)
    err_main = float((psi_main.norm(p=2, dim=-1) - 1.0).abs().max().item())
    err_beam2 = float((psi_beam2.norm(p=2, dim=-1) - 1.0).abs().max().item())
    ortho_abs = float((psi_main * psi_beam2).sum(dim=-1).abs().max().item())

    assert err_main < 1e-5
    assert err_beam2 < 1e-5
    assert ortho_abs < 1e-5

    # Verify non-zero radial sensitivity when input scale changes (2.0 * x vs 0.5 * x)
    psi_hi, _ = layer(2.0 * x_unit, return_second_beam=True)
    psi_lo, _ = layer(0.5 * x_unit, return_second_beam=True)
    radial_diff = float((psi_hi - psi_lo).norm(p=2, dim=-1).mean().item())
    assert radial_diff > 0.02


def test_batch_centered_contextual_purity_and_wave17_two_g4_manifest() -> None:
    """Verify H2.3: Batch-centered contextual_purity_complement preserves batch mean lam_base at high beta_t and wave17_fleet.json has ONLY 2 G4 runs."""
    readout = Hierarchical16QubitPOVMReadout(
        vocab_size=128,
        embed_dim=64,
        num_registers=8,
        register_dim=8,
        num_higher_states=16,
        use_two_beam_kboson_readout=True,
        diffuse_rank=8,
        diffuse_schedule="contextual_purity_complement",
        lambda_diff_min=0.05,
        lambda_diff_max=0.40,
        lambda_diff_start=0.22,
        lambda_diff_end=0.12,
        contextual_purity_gain=0.15,
        use_stereographic_lift=True,
        quadratic_diffuse_fidelity=0.50,
    )
    with torch.no_grad():
        readout.kboson_copies.fill_(40.0)

    # High-purity late-training regime: beta_t in [32.0, 44.0] => r_tilde in [0.80, 0.95]
    beta_high = torch.linspace(32.0, 44.0, steps=32).reshape(4, 8, 1)
    for step, total in ((10, 100), (35, 100), (80, 100)):
        readout.set_diffuse_schedule_progress(step, total)
        lam_ctx = readout.compute_scheduled_lambda_diff(beta_t=beta_high)

        # Compute exact warmup_hold_anneal base value
        readout.diffuse_schedule = "warmup_hold_anneal"
        lam_base = float(readout.compute_scheduled_lambda_diff().item())
        readout.diffuse_schedule = "contextual_purity_complement"

        assert abs(float(lam_ctx.mean().item()) - lam_base) < 1e-3
        # Lowest beta token (32.0) must receive higher diffuse bath weight than highest beta token (44.0)
        assert float(lam_ctx.reshape(-1)[0].item()) > float(lam_ctx.reshape(-1)[-1].item()) + 0.02

    # Verify Wave 17 manifest strictly has ONLY 2 runs on gpu_g4_w6_1 and gpu_g4_w6_2
    assert "W17" in WAVE_DEFAULT_LABELS
    assert WAVE_DEFAULT_PLANS["W17"] == "EXPERIMENT_PLAN_028"

    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "wave17_fleet.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["wave_id"] == "W17"
    assert data["plan_id"] == "EXPERIMENT_PLAN_028"
    runs = data["runs"]
    assert len(runs) == 2, f"Expected strictly 2 G4 runs in wave17_fleet.json, got {len(runs)}"
    sessions = [r["session"] for r in runs]
    assert set(sessions) == {"gpu_g4_w6_1", "gpu_g4_w6_2"}

    for r in runs:
        arch, _, _ = parse_cli_args_to_card_sections(r["cli_args"])
        assert arch["readout_mode"] == "two_beam_kboson_povm"
        assert arch["use_stereographic_lift"] is True
        assert arch["quadratic_diffuse_fidelity"] == pytest.approx(0.50)

    # Also verify QuantumBornGPT end-to-end forward+backward with all Plan 028 flags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantumBornGPT(
        vocab_size=256,
        seed_dim=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        block_size=32,
        readout_mode="asymmetric",
        use_two_beam_mlp=True,
        use_overcomplete_two_beam=True,
        frame_expansion=2.0,
        use_dynamic_povm_beta=True,
        use_hierarchical_16qubit=True,
        qubit_registers=8,
        qubit_amplitudes=8,
        num_higher_states=32,
        bridge_causal_memory=False,
        use_two_beam_kboson_readout=True,
        diffuse_rank=8,
        diffuse_schedule="contextual_purity_complement",
        contextual_purity_gain=0.15,
        use_stereographic_lift=True,
        use_cosine_phase_beam_gate=True,
        quadratic_diffuse_fidelity=0.50,
    ).to(device)
    idx = torch.randint(0, 256, (2, 16), device=device)
    targets = torch.randint(0, 256, (2, 16), device=device)
    _, loss = model(idx, targets=targets, direct_chunked_ce=True)
    assert loss is not None and torch.isfinite(loss).item()
    loss.backward()
    assert model.hierarchical_povm_readout.readout_stereo_pole.grad is not None
