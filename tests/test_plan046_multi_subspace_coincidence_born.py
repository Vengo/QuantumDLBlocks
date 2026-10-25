"""Unit and hypothesis tests for Experiment Plan 046 (Wave 31, Lever 3):
Product-of-Spheres Multi-Subspace Coincidence Born Readout.
"""

from __future__ import annotations

import math
import torch

from qdlblocks.layers.parseval_mup_chunked_ce import ParsevalChunkedCEReadoutSpec
from qdlblocks.layers.subspace_coincidence_born import (
    SubspaceCoincidenceSpec,
    compute_coarse_cluster_centers,
    compute_multi_subspace_coincidence_observable,
    compute_subspace_cosines,
    eval_subspace_coincidence_full_logits,
    subspace_coincidence_dyadic_chunked_ce,
    subspace_l2_normalize,
)
from qdlblocks.models.parseval_mup_subspace_coincidence_gpt import (
    ParsevalMuPSubspaceCoincidenceConfig,
    ParsevalMuPSubspaceCoincidenceGPT,
    build_subspace_coincidence_optimizers,
)
from scripts.multi_subspace_coincidence_margin_probe import (
    compute_analytical_parseval_beta,
    run_probe_grid,
)


def test_product_of_spheres_unit_and_subspace_norms() -> None:
    """Verify subspace_l2_normalize enforces ||x^(m)||_2 = 1/sqrt(M) and ||x||_2 = 1."""
    torch.manual_seed(46)
    x = torch.randn(32, 512)
    x[:, :128] *= 10.0

    for m in (1, 2, 4, 8):
        x_norm = subspace_l2_normalize(x, num_subspaces=m)
        total_norms = torch.linalg.vector_norm(x_norm, dim=-1)
        assert torch.allclose(total_norms, torch.ones_like(total_norms), atol=1e-5)

        sub_norms = torch.linalg.vector_norm(x_norm.view(32, m, 512 // m), dim=-1)
        expected = torch.full_like(sub_norms, 1.0 / math.sqrt(float(m)))
        assert torch.allclose(sub_norms, expected, atol=1e-5)


def test_traceless_coincidence_observable_sum_zero_and_impostor_suppression() -> None:
    """Verify the multi-subspace coincidence observable is exactly traceless and suppresses single-subspace impostors."""
    torch.manual_seed(146)
    vocab_size = 256
    embed_dim = 128
    m = 4

    m_sharp = subspace_l2_normalize(torch.randn(vocab_size, embed_dim), num_subspaces=m)
    psi = m_sharp[0:1].clone()
    impostor = torch.randn(1, embed_dim)
    impostor[0, :32] = psi[0, :32] * 4.0
    impostor[0, 32:] = -psi[0, 32:] * 0.2
    m_sharp[1:2] = subspace_l2_normalize(impostor, num_subspaces=m)

    cos_m = compute_subspace_cosines(psi, m_sharp, num_subspaces=m)
    obs_dict = compute_multi_subspace_coincidence_observable(
        cos_m,
        traceless_gamma=0.25,
        coincidence_gamma=0.45,
        subspace_var_penalty=0.25,
    )
    obs = obs_dict["observable"]

    assert abs(float(obs.sum().item())) < 1e-4
    assert float(obs[0, 0].item()) > float(obs[0, 1].item()) + 0.5


def test_subspace_coincidence_chunked_ce_and_full_logits() -> None:
    """Verify subspace_coincidence_dyadic_chunked_ce and eval_subspace_coincidence_full_logits."""
    torch.manual_seed(246)
    vocab_size = 512
    valid_vocab_size = 500
    embed_dim = 128
    num_coarse = 16
    beta_0 = compute_analytical_parseval_beta(embed_dim)
    v_to_coarse = torch.arange(vocab_size, dtype=torch.long) % num_coarse

    readout_spec = ParsevalChunkedCEReadoutSpec(
        beta_0=beta_0,
        lambda_diff_scaled=beta_0 * 0.06,
        traceless_gamma=0.25,
        chunk_size=128,
        valid_vocab_size=valid_vocab_size,
        num_coarse_buckets=num_coarse,
        v_to_coarse=v_to_coarse,
    )
    coinc_spec = SubspaceCoincidenceSpec(
        num_subspaces=4,
        coincidence_gamma=0.22,
        subspace_var_penalty=0.14,
        coarse_coincidence_weight=0.16,
        full_vocab_coincidence_eval=True,
    )

    psi = subspace_l2_normalize(torch.randn(2, 8, embed_dim), num_subspaces=4).requires_grad_(True)
    psi_diff = subspace_l2_normalize(torch.randn(2, 8, embed_dim), num_subspaces=4).requires_grad_(True)
    m_sharp = subspace_l2_normalize(torch.randn(vocab_size, embed_dim), num_subspaces=4).requires_grad_(True)
    m_diff = subspace_l2_normalize(torch.randn(vocab_size, embed_dim), num_subspaces=4).requires_grad_(True)
    log_prior = torch.zeros(vocab_size)
    targets = torch.randint(0, valid_vocab_size, (2, 8))

    centers, v_map = compute_coarse_cluster_centers(m_sharp, readout_spec, coinc_spec)
    assert centers.shape == (num_coarse, embed_dim)
    assert v_map.shape == (valid_vocab_size,)

    ce_chunked = subspace_coincidence_dyadic_chunked_ce(
        psi_sharp=psi,
        psi_diff=psi_diff,
        m_sharp=m_sharp,
        m_diff=m_diff,
        targets=targets,
        readout_spec=readout_spec,
        coinc_spec=coinc_spec,
        log_prior=log_prior,
    )
    ce_chunked.backward()
    assert psi.grad is not None and torch.isfinite(psi.grad).all()
    assert m_sharp.grad is not None and torch.isfinite(m_sharp.grad).all()

    logits_full = eval_subspace_coincidence_full_logits(
        psi_sharp=psi.detach(),
        psi_diff=psi_diff.detach(),
        m_sharp=m_sharp.detach(),
        m_diff=m_diff.detach(),
        readout_spec=readout_spec,
        coinc_spec=coinc_spec,
        log_prior=log_prior,
    )
    assert logits_full.shape == (2, 8, valid_vocab_size)
    assert torch.isfinite(logits_full).all()


def test_parseval_mup_subspace_coincidence_gpt_forward_backward() -> None:
    """Verify ParsevalMuPSubspaceCoincidenceGPT forward, backward, and Cholesky absorption."""
    torch.manual_seed(346)
    cfg = ParsevalMuPSubspaceCoincidenceConfig(
        vocab_size=1024,
        valid_vocab_size=1000,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        block_size=16,
        num_subspaces=4,
    )
    model = ParsevalMuPSubspaceCoincidenceGPT(cfg)
    opt_muon, opt_adam = build_subspace_coincidence_optimizers(model)

    x = torch.randint(0, 1000, (2, 16))
    y = torch.randint(0, 1000, (2, 16))
    logits, loss = model(x, targets=y, return_logits=True)
    assert logits is not None and logits.shape == (2, 16, 1000)
    assert loss is not None and torch.isfinite(loss)

    loss.backward()
    opt_muon.step()
    opt_adam.step()
    scale_info = model.absorb_beta_into_readout_cholesky()
    assert isinstance(scale_info, dict) and scale_info.get("absorbed_scale", 1.0) > 0.0
    assert model.pyramid.compute_kraus_stiefel_err() < 0.05


def test_multi_subspace_coincidence_probe_hypotheses_h1_and_h2() -> None:
    """Verify H1 (R_sharp retention >= 2.2x) and H2 (runner-up CE tax reduction >= 0.12 nats)."""
    report = run_probe_grid(
        dims=[256, 512],
        subspaces=[1, 4],
        vocab_size=12288,
        device="cpu",
    )
    assert report["H1_product_of_spheres_rank_preservation"]["passed"] is True
    assert report["H2_coincidence_runnerup_suppression"]["passed"] is True


if __name__ == "__main__":
    for fn in (
        test_product_of_spheres_unit_and_subspace_norms,
        test_traceless_coincidence_observable_sum_zero_and_impostor_suppression,
        test_subspace_coincidence_chunked_ce_and_full_logits,
        test_parseval_mup_subspace_coincidence_gpt_forward_backward,
        test_multi_subspace_coincidence_probe_hypotheses_h1_and_h2,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 5 Plan 046 unit & hypothesis tests PASSED.")
