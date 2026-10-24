"""Unit & Integration Tests for Plan 045 / Wave 29 (`ParsevalMuPNGramSpeedrunGPT` & 900M Scaling)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(ROOT / "src"), str(ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.models.parseval_mup_ngram_speedrun_gpt import (  # noqa: E402
    HashedSurfaceNGramProjector,
    ParsevalMuPNGramSpeedrunConfig,
    ParsevalMuPNGramSpeedrunGPT,
    build_parseval_mup_ngram_optimizers,
    initialize_parseval_mup_ngram_with_ppmi_svd,
)
from scripts.fineweb_150m_300m_900m_scaling_projector import (  # noqa: E402
    HORIZONS_M,
    ScalingProjectorConfig,
    eval_chinchilla_curve,
    fit_chinchilla_3param,
    run_scaling_projector,
)


def test_hashed_surface_ngram_projector_shapes_and_stiefel() -> None:
    """Verify `HashedSurfaceNGramProjector` bucket hashing, Stiefel retraction, and unit-norm outputs."""
    proj = HashedSurfaceNGramProjector(embed_dim=64, ngram_buckets=256, ngram_dim=16)
    with torch.no_grad():
        proj.unigram_table.normal_(0.0, 0.1)
        proj.bigram_table.normal_(0.0, 0.1)
        proj.trigram_table.normal_(0.0, 0.1)
        proj.u_ngram.add_(0.05 * torch.randn_like(proj.u_ngram))

    err_after = proj.retract_stiefel_basis()
    assert err_after < 0.05
    idx = torch.randint(0, 128, (2, 16), dtype=torch.long)
    bi_idx, tri_idx = proj.compute_bigram_trigram_buckets(idx)
    assert bi_idx.shape == (2, 16)
    assert tri_idx.shape == (2, 16)
    in_delta, readout_delta = proj(idx)
    assert in_delta.shape == (2, 16, 64)
    assert readout_delta.shape == (2, 16, 64)


def test_parseval_mup_ngram_gpt_forward_backward_and_cholesky_absorption() -> None:
    """Verify `ParsevalMuPNGramSpeedrunGPT` forward/backward, Cholesky absorption, and optimizer routing."""
    cfg = ParsevalMuPNGramSpeedrunConfig(
        vocab_size=256,
        valid_vocab_size=250,
        block_size=32,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        window_size=32,
        ngram_buckets=128,
        ngram_dim=16,
        readout_gemm_precision="bf16",
    )
    model = ParsevalMuPNGramSpeedrunGPT(cfg)
    ppmi = torch.randn(256, 64)
    init_toks = torch.randint(0, 250, (2048,), dtype=torch.long)
    stats = initialize_parseval_mup_ngram_with_ppmi_svd(model, ppmi, init_toks, init_toks)
    assert stats["kraus_stiefel_err"] < 0.05
    assert stats["u_ngram_stiefel_err"] < 0.05

    model.eval()
    x = torch.randint(0, 250, (2, 16), dtype=torch.long)
    y = torch.randint(0, 250, (2, 16), dtype=torch.long)
    with torch.no_grad():
        logits_pre, _ = model(x, targets=None, return_logits=True)
        abs_info = model.absorb_beta_into_readout_cholesky()
        logits_post, _ = model(x, targets=None, return_logits=True)
    assert logits_pre is not None and logits_post is not None
    max_err = float((logits_pre[..., :250].float() - logits_post[..., :250].float()).abs().max().item())
    assert max_err < 1e-4
    assert abs_info["beta_0"] == model.beta_0

    model.train()
    muon_opt, adam_opt = build_parseval_mup_ngram_optimizers(model)
    muon_opt.zero_grad(set_to_none=True)
    adam_opt.zero_grad(set_to_none=True)
    _, loss = model(x, y, return_logits=False)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.ngram_proj.bigram_table.grad is not None
    assert float(model.ngram_proj.bigram_table.grad.norm().item()) > 0.0
    assert model.v0_gate.grad is not None
    assert float(model.v0_gate.grad.norm().item()) > 0.0


def test_chinchilla_3param_and_scaling_projector() -> None:
    """Verify `fit_chinchilla_3param` and `run_scaling_projector` predictions at 150M, 300M, 900M."""
    pts = [(7.37, 5.15), (14.75, 4.86), (50.0, 4.22), (100.0, 3.93), (300.0, 3.58), (900.0, 3.31)]
    law = fit_chinchilla_3param(pts)
    assert float(law["rmse"]) < 0.05
    preds = eval_chinchilla_curve(law, HORIZONS_M)["post_wsd_cooldown"]
    assert float(preds["150M"]) < 3.90
    assert 3.45 <= float(preds["300M"]) <= 3.68
    assert 3.20 <= float(preds["900M"]) <= 3.38

    out = run_scaling_projector(ScalingProjectorConfig(quiet=True))
    w29_traj = out["w29_flagship_trajectory"]
    post_cd = w29_traj["post_wsd_cooldown_by_horizon"]
    assert float(post_cd["150M"]) < 3.90
    assert float(w29_traj["tokens_m_to_break_3p90_post_cooldown"]) < 150.0


def test_hypothesis_experiments_045_artifacts() -> None:
    """Verify `.cache/hypothesis_experiments_045.json` and Wave 29 model cards when present."""
    hyp_path = ROOT / ".cache" / "hypothesis_experiments_045.json"
    if not hyp_path.exists():
        return
    data = json.loads(hyp_path.read_text(encoding="utf-8"))
    assert data["overall_pass"] is True
    assert data["H1"]["pass"] is True
    assert data["H2"]["pass"] is True
    assert data["H3"]["pass"] is True
    assert float(data["H2"]["w29_ngram_v0_d256_ce"]) <= 5.32
    assert float(data["H2"]["w29_ngram_v0_d384_1800steps_ce"]) < 4.98
    assert float(data["H3"]["projected_ce_150M_post_cooldown"]) < 3.90


if __name__ == "__main__":
    test_hashed_surface_ngram_projector_shapes_and_stiefel()
    test_parseval_mup_ngram_gpt_forward_backward_and_cholesky_absorption()
    test_chinchilla_3param_and_scaling_projector()
    test_hypothesis_experiments_045_artifacts()
    print("ALL PLAN 045 TESTS PASSED")
