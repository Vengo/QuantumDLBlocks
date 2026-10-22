"""Unit tests for Plan 027: Diffuse Kraus Bath Weight Schedule, Per-Token Chunked k-Boson CE, Stiefel Retraction, and Wave 16 Fleet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

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
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import polar_snap_weights


def test_per_token_lambda_diff_chunked_kboson_ce_parity() -> None:
    """Verify forward loss (< 1e-5) and backward gradient parity (< 1e-5) for per-token and scalar lambda_diff."""
    torch.manual_seed(42)
    b, t, d, r, v = 2, 7, 32, 8, 96
    n_tok = b * t
    targets = torch.randint(0, v, (b, t), dtype=torch.long)

    for per_token in (False, True):
        psi_sharp_1 = torch.randn(b, t, d, dtype=torch.float32, requires_grad=True)
        psi_diff_1 = torch.randn(b, t, r, dtype=torch.float32, requires_grad=True)
        m_sharp_1 = torch.randn(v, d, dtype=torch.float32, requires_grad=True)
        m_diff_1 = torch.randn(v, r, dtype=torch.float32, requires_grad=True)
        beta_1 = (3.5 + 3.0 * torch.rand(b, t, dtype=torch.float32)).detach().requires_grad_(True)
        log_prior_1 = torch.randn(v, dtype=torch.float32, requires_grad=True)
        k_eff_1 = torch.tensor(34.0, dtype=torch.float32, requires_grad=True)
        if per_token:
            lam_1 = (0.10 + 0.25 * torch.rand(n_tok, 1, dtype=torch.float32)).detach().requires_grad_(True)
        else:
            lam_1 = torch.tensor(0.24, dtype=torch.float32, requires_grad=True)

        psi_sharp_2 = psi_sharp_1.detach().clone().requires_grad_(True)
        psi_diff_2 = psi_diff_1.detach().clone().requires_grad_(True)
        m_sharp_2 = m_sharp_1.detach().clone().requires_grad_(True)
        m_diff_2 = m_diff_1.detach().clone().requires_grad_(True)
        beta_2 = beta_1.detach().clone().requires_grad_(True)
        log_prior_2 = log_prior_1.detach().clone().requires_grad_(True)
        k_eff_2 = k_eff_1.detach().clone().requires_grad_(True)
        lam_2 = lam_1.detach().clone().requires_grad_(True)

        loss_chunked = direct_chunked_two_beam_kboson_ce(
            psi_sharp=psi_sharp_1,
            psi_diff=psi_diff_1,
            m_sharp=m_sharp_1,
            m_diff=m_diff_1,
            beta_t=beta_1,
            k_copies=k_eff_1,
            lambda_diff=lam_1,
            targets=targets,
            log_prior=log_prior_1,
            chunk_size=5,
        )
        loss_chunked.backward()

        logits_ref = compute_two_beam_kboson_logits(
            psi_sharp=psi_sharp_2,
            psi_diff=psi_diff_2,
            m_sharp=m_sharp_2,
            m_diff=m_diff_2,
            beta_t=beta_2,
            k_copies=k_eff_2,
            lambda_diff=lam_2.reshape(b, t, 1) if per_token else lam_2,
            log_prior=log_prior_2,
        )
        loss_ref = torch.nn.functional.cross_entropy(
            logits_ref.reshape(-1, v),
            targets.reshape(-1),
            reduction="mean",
        )
        loss_ref.backward()

        assert abs(float(loss_chunked.item()) - float(loss_ref.item())) < 1e-5
        assert torch.max(torch.abs(psi_sharp_1.grad - psi_sharp_2.grad)).item() < 1e-5
        assert torch.max(torch.abs(psi_diff_1.grad - psi_diff_2.grad)).item() < 1e-5
        assert torch.max(torch.abs(m_sharp_1.grad - m_sharp_2.grad)).item() < 1e-5
        assert torch.max(torch.abs(m_diff_1.grad - m_diff_2.grad)).item() < 1e-5
        assert torch.max(torch.abs(beta_1.grad - beta_2.grad)).item() < 1e-4
        assert torch.max(torch.abs(log_prior_1.grad - log_prior_2.grad)).item() < 1e-5
        assert torch.max(torch.abs(k_eff_1.grad - k_eff_2.grad)).item() < 1e-4
        assert lam_1.grad.shape == lam_2.grad.shape
        assert torch.max(torch.abs(lam_1.grad - lam_2.grad)).item() < 1e-4


def test_kraus_stiefel_retraction_and_schedule_modes() -> None:
    """Verify Stiefel retraction of kraus_u_diff/kraus_v_diff and all 5 diffuse_schedule modes."""
    torch.manual_seed(123)
    schedules = (
        "free_decay",
        "bounded_floor",
        "cosine_broad_to_sharp",
        "warmup_hold_anneal",
        "contextual_purity_complement",
    )
    for sched in schedules:
        readout = Hierarchical16QubitPOVMReadout(
            vocab_size=128,
            embed_dim=64,
            num_registers=8,
            register_dim=8,
            num_higher_states=32,
            diffuse_rank=8,
            diffuse_schedule=sched,
            lambda_diff_min=0.08,
            lambda_diff_max=0.35,
            lambda_diff_start=0.22,
            lambda_diff_end=0.12,
            contextual_purity_gain=0.20,
        )
        # Perturb kraus_u_diff and kraus_v_diff off the Stiefel manifold
        with torch.no_grad():
            readout.kraus_u_diff.add_(0.25 * torch.randn_like(readout.kraus_u_diff))
            readout.kraus_v_diff.add_(0.25 * torch.randn_like(readout.kraus_v_diff))

        err = readout.retract_kraus_stiefel()
        assert err < 1e-5
        eye_r = torch.eye(8)
        u_err = torch.linalg.norm(readout.kraus_u_diff.T @ readout.kraus_u_diff - eye_r).item()
        v_err = torch.linalg.norm(readout.kraus_v_diff.T @ readout.kraus_v_diff - eye_r).item()
        assert u_err < 1e-5
        assert v_err < 1e-5

        # Also verify polar_snap_weights invokes retract_kraus_stiefel
        with torch.no_grad():
            readout.kraus_u_diff.mul_(1.3)
        polar_snap_weights(readout)
        u_err_after = torch.linalg.norm(readout.kraus_u_diff.T @ readout.kraus_u_diff - eye_r).item()
        assert u_err_after < 1e-5

        beta_t = torch.tensor([[3.8, 5.2, 6.8], [4.1, 5.9, 7.2]], dtype=torch.float32)
        vals = []
        for step, total in ((0, 100), (15, 100), (35, 100), (70, 100), (100, 100)):
            p = readout.set_diffuse_schedule_progress(step, total)
            assert 0.0 <= p <= 1.0
            lam = readout.compute_scheduled_lambda_diff(beta_t=beta_t)
            if sched == "contextual_purity_complement":
                assert lam.numel() == beta_t.numel()
                assert float(lam.min().item()) >= 0.08 - 1e-5
                assert float(lam.max().item()) <= 0.45 + 1e-5
                # Higher purity beta_t should give smaller lambda_diff than lower purity beta_t
                lam_2d = lam.reshape_as(beta_t)
                assert float(lam_2d[0, 0].item()) > float(lam_2d[0, 2].item())
            else:
                assert lam.numel() == 1
                vals.append(float(lam.item()))

        if sched == "bounded_floor":
            assert all(0.08 - 1e-5 <= v <= 0.35 + 1e-5 for v in vals)
        elif sched == "cosine_broad_to_sharp":
            assert vals[0] > vals[2] > vals[-1]
        elif sched == "warmup_hold_anneal":
            # Step 0 ->Step 15/35 hold -> Step 100 anneal
            assert abs(vals[1] - vals[2]) < 1e-6
            assert vals[2] > vals[-1]


def test_wave16_manifest_and_model_card_parsing() -> None:
    """Verify configs/wave16_fleet.json has 5 valid W16 runs and model_card parses diffuse schedule flags."""
    assert "W16" in WAVE_DEFAULT_LABELS
    assert WAVE_DEFAULT_PLANS["W16"] == "EXPERIMENT_PLAN_027"

    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "wave16_fleet.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["wave_id"] == "W16"
    assert data["plan_id"] == "EXPERIMENT_PLAN_027"
    runs = data["runs"]
    assert len(runs) == 5
    sessions = {r["session"] for r in runs}
    assert sessions == {"gpu_g4_w6_1", "gpu_g4_w6_2", "gpu_g4_w6_3", "gpu_a100_w6_1", "gpu_a100_w6_2"}

    for r in runs:
        arch, _, _ = parse_cli_args_to_card_sections(r["cli_args"])
        assert arch["readout_mode"] == "two_beam_kboson_povm"
        assert arch["two_beam_kboson_readout"] is True
        assert arch["diffuse_schedule"] in ("warmup_hold_anneal", "contextual_purity_complement")
        assert arch["diffuse_rank"] in (16, 32)
        assert arch["lambda_diff_min"] == pytest.approx(0.08)
        assert arch["lambda_diff_max"] == pytest.approx(0.35)


def test_torch_compile_two_beam_kboson_step() -> None:
    """Verify QuantumBornGPT with Two-Beam k-Boson Readout and contextual_purity_complement runs cleanly."""
    torch.manual_seed(7)
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
        contextual_purity_gain=0.20,
    ).to(device)

    model.hierarchical_povm_readout.set_diffuse_schedule_progress(10, 100)
    idx = torch.randint(0, 256, (2, 16), device=device)
    targets = torch.randint(0, 256, (2, 16), device=device)

    _, loss = model(idx, targets=targets, direct_chunked_ce=True)
    assert loss is not None
    loss.backward()
    assert torch.isfinite(loss).item()
    assert model.hierarchical_povm_readout.kraus_u_diff.grad is not None
    assert torch.isfinite(model.hierarchical_povm_readout.kraus_u_diff.grad).all().item()
