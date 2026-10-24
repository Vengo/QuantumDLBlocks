"""CPU Unit Tests for Experiment Plan 039 (`Wave 24`):
Fast-Loop Multi-Class Vision Benchmarks (`CIFAR-100` $K=100, N=128$ & `1,000`-Class $K=1000, N=1024$)
and Convex-Peak Traceless Born / Dyadic Hierarchical Readouts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from qdlblocks.layers.direct_chunked_ce import (
    compute_two_beam_kboson_logits,
    direct_chunked_two_beam_kboson_ce,
)
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
)
from qdlblocks.models.wavelet_cholesky_vision import (
    VisionWaveletCholeskyConfig,
    WaveletCholeskyVisionClassifier,
    build_vision_wavelet_optimizers,
)

ROOT = Path(__file__).resolve().parents[1]

READOUT_MODES = (
    "classical_softmax",
    "naive_kboson_born",
    "last_layer_meas_traceless_kboson",
    "convex_traceless_born",
    "dyadic_hierarchical_born",
)


@pytest.mark.parametrize("readout_mode", READOUT_MODES)
@pytest.mark.parametrize(
    "hierarchy_spec",
    [
        (100, 20, 128),
        (1000, 32, 1024),
    ],
)
def test_vision_classifier_readouts_and_exact_prob_normalization(
    readout_mode: str,
    hierarchy_spec: tuple[int, int, int],
) -> None:
    """Verify all 5 readout modes produce exact probability normalization and finite gradients on K=100 and K=1000."""
    torch.manual_seed(2039)
    num_classes, num_coarse, dyadic_dim = hierarchy_spec
    class_to_coarse = (torch.arange(num_classes, dtype=torch.long) * num_coarse) // num_classes

    cfg = VisionWaveletCholeskyConfig(
        num_classes=num_classes,
        num_coarse_classes=num_coarse,
        dyadic_dim=dyadic_dim,
        d_model=64,
        num_layers=2,
        patch_size=4,
        kraus_rank=4,
        cholesky_block_size=16,
        readout_mode=readout_mode,
        dropout=0.0,
    )
    model = WaveletCholeskyVisionClassifier(cfg, class_to_coarse)
    batch_size = 6
    images = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    log_probs, diag = model.compute_log_probs_and_diagnostics(images)
    assert log_probs.shape == (batch_size, num_classes)
    assert torch.isfinite(log_probs).all()

    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    assert diag["psi_fine"].shape == (num_classes, 64)
    assert diag["psi_coarse"].shape == (num_coarse, 64)

    _, loss = model(images, targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    grad_count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
            grad_count += 1
    assert grad_count > 0


def test_build_vision_wavelet_optimizers_routes_stiefel_matrices_to_muon() -> None:
    """Verify `kraus_generators`, `up_fine`, and `token_mixer` are routed to `StiefelMuon`."""
    class_to_coarse = (torch.arange(100, dtype=torch.long) * 20) // 100
    cfg = VisionWaveletCholeskyConfig(num_classes=100, num_coarse_classes=20, dyadic_dim=128, d_model=64, num_layers=2)
    model = WaveletCholeskyVisionClassifier(cfg, class_to_coarse)
    muon_opt, adam_opt = build_vision_wavelet_optimizers(model)

    muon_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    adam_ids = {id(p) for g in adam_opt.param_groups for p in g["params"]}
    assert len(muon_ids & adam_ids) == 0
    assert len(muon_ids | adam_ids) == sum(1 for p in model.parameters() if p.requires_grad)

    assert id(model.pyramid.kraus_generators) in muon_ids
    assert id(model.pyramid.up_fine) in muon_ids
    assert id(model.blocks[0].token_mixer.weight) in muon_ids
    assert id(model.pyramid.wav_coarse) in adam_ids
    assert id(model.pyramid.wav_fine) in adam_ids
    assert model.pyramid.kraus_stiefel_error() < 1e-5


def test_direct_chunked_convex_traceless_born_autograd_equivalence() -> None:
    """Verify `direct_chunked_two_beam_kboson_ce(..., convex_traceless=True)` matches PyTorch autograd."""
    torch.manual_seed(2039)
    n_tok, d, v_vocab, v_act = 32, 24, 96, 80
    psi_s = F.normalize(torch.randn(n_tok, d, dtype=torch.float32), dim=-1).requires_grad_(True)
    psi_d = F.normalize(torch.randn(n_tok, d, dtype=torch.float32), dim=-1).requires_grad_(True)
    m_s = F.normalize(torch.randn(v_vocab, d, dtype=torch.float32), dim=-1).requires_grad_(True)
    m_d = F.normalize(torch.randn(v_vocab, d, dtype=torch.float32), dim=-1).requires_grad_(True)
    beta_t = (torch.rand(n_tok, 1, dtype=torch.float32) * 8.0 + 12.0).requires_grad_(True)
    k_cop = torch.tensor(24.0, dtype=torch.float32, requires_grad=True)
    lam_d = torch.tensor(0.15, dtype=torch.float32, requires_grad=True)
    prior = (torch.randn(v_vocab, dtype=torch.float32) * 0.2).requires_grad_(True)
    targets = torch.randint(0, v_act, (n_tok,), dtype=torch.long)

    loss_chunk = direct_chunked_two_beam_kboson_ce(
        psi_sharp=psi_s,
        psi_diff=psi_d,
        m_sharp=m_s,
        m_diff=m_d,
        beta_t=beta_t,
        k_copies=k_cop,
        lambda_diff=lam_d,
        targets=targets,
        log_prior=prior,
        chunk_size=16,
        valid_vocab_size=v_act,
        quadratic_diffuse_fidelity=0.5,
        traceless_gamma=0.75,
        convex_traceless=True,
    )
    grads_chunk = torch.autograd.grad(loss_chunk, (psi_s, psi_d, m_s, m_d, beta_t, k_cop, lam_d, prior))

    logits_ref = compute_two_beam_kboson_logits(
        psi_sharp=psi_s,
        psi_diff=psi_d,
        m_sharp=m_s,
        m_diff=m_d,
        beta_t=beta_t,
        k_copies=k_cop,
        lambda_diff=lam_d,
        log_prior=prior,
        quadratic_diffuse_fidelity=0.5,
        traceless_gamma=0.75,
        convex_traceless=True,
        valid_vocab_size=v_act,
    )
    loss_ref = F.cross_entropy(logits_ref[:, :v_act], targets)
    grads_ref = torch.autograd.grad(loss_ref, (psi_s, psi_d, m_s, m_d, beta_t, k_cop, lam_d, prior))

    fwd_diff = float((loss_chunk - loss_ref).abs().item())
    max_grad_diff = max(float((gc - gr).abs().max().item()) for gc, gr in zip(grads_chunk, grads_ref))
    assert fwd_diff < 1e-5
    assert max_grad_diff < 1e-4


def test_radical_2pow16_gpt_convex_traceless_born_cpu_smoke() -> None:
    """Verify `Radical2Pow16WaveletCholeskyGPT` supports `use_convex_traceless_born=True` on CPU."""
    torch.manual_seed(2039)
    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        vocab_size=1024,
        valid_vocab_size=1000,
        ambient_n=1024,
        num_layers=1,
        embed_dim=64,
        num_heads=4,
        block_size=16,
        window_size=16,
        fine_detail_dim=16,
        mid_detail_dim=32,
        diffuse_rank=4,
        use_convex_traceless_born=True,
        traceless_gamma=0.65,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg)
    idx = torch.randint(0, 512, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 512, (2, 8), dtype=torch.long)
    _, loss = model(idx, targets, return_logits=False)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()


def test_promoted_plan039_diagnostic_json_artifacts() -> None:
    """Verify `.cache/last_layer_measurement_adjoint_bench.json` and `.cache/traceless_observable_rank_bucket_probe.json`."""
    adj_path = ROOT / ".cache" / "last_layer_measurement_adjoint_bench.json"
    probe_path = ROOT / ".cache" / "traceless_observable_rank_bucket_probe.json"
    assert adj_path.exists(), f"Missing {adj_path}"
    assert probe_path.exists(), f"Missing {probe_path}"

    adj = json.loads(adj_path.read_text(encoding="utf-8"))
    assert adj["all_checks_passed"] is True
    for k_str in ("100", "1000", "50304"):
        row = adj["classes"][k_str]
        assert row["adjoint_equivalence_passed"] is True
        assert row["self_loop_zero_passed"] is True
        assert row["adjoint_max_abs_diff"] < 1e-5
    assert adj["classes"]["50304"]["measurement_flop_ratio"] > 20.0

    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    summary = probe["summary"]
    assert summary["convex_traceless_ce_at_max_gamma"] < summary["naive_kboson_ce_at_max_gamma"]
    assert summary["dyadic_hierarchical_ce_at_max_gamma"] < summary["naive_kboson_ce_at_max_gamma"]
    assert summary["convex_ce_gain_vs_naive_nats"] > 0.50
    assert summary["dyadic_ce_gain_vs_naive_nats"] > 0.50
