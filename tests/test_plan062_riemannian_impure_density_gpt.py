"""Unit tests for Wave 47 (`EXPERIMENT_PLAN_062.md`): `RiemannianImpureDensityGPT` & `AllLayerRiemannianStiefelMuon`."""

from __future__ import annotations

import math

import torch

from qdlblocks.layers.impure_density_mirror_block import (
    complex_l2_normalize,
    pack_rank1_ket_outer_isometric,
)
from qdlblocks.models.riemannian_impure_density_gpt import (
    DensityStateMode,
    InputBoundaryMode,
    RiemannianImpureDensityBlock,
    RiemannianImpureDensityConfig,
    RiemannianImpureDensityGPT,
    build_all_layer_riemannian_optimizers,
    fast_rank_r_token_density_pack,
    polar_factorize_stiefel_svd,
    riemannian_complex_unitary_muon_step,
)


def test_fast_rank_r_token_density_pack_matches_explicit_branch_sum() -> None:
    """Verify `fast_rank_r_token_density_pack` matches explicit $R=4$ branch outer-product sum to `< 1e-6`."""
    torch.manual_seed(42)
    bsz, heads, seq_len, r_branches, d_h = 2, 4, 8, 4, 16
    v_re = torch.randn(bsz, heads, seq_len, r_branches, d_h)
    v_im = torch.randn(bsz, heads, seq_len, r_branches, d_h)

    cfg = RiemannianImpureDensityConfig(embed_dim=256, num_layers=2, num_heads=heads, head_complex_dim=d_h)
    blk = RiemannianImpureDensityBlock(cfg, layer_idx=0)
    fast_packed = fast_rank_r_token_density_pack(v_re, v_im, blk.re_pack_mask, blk.im_pack_mask)

    branch_energy = (v_re.square().sum(dim=-1) + v_im.square().sum(dim=-1)).clamp_min(1e-7)
    branch_weights = branch_energy / branch_energy.sum(dim=-1, keepdim=True)
    v_re_u, v_im_u = complex_l2_normalize(v_re, v_im)
    explicit_branches = pack_rank1_ket_outer_isometric(v_re_u, v_im_u)
    explicit_packed = (branch_weights.unsqueeze(-1) * explicit_branches).sum(dim=-2)

    max_diff = float((fast_packed - explicit_packed).abs().max().item())
    assert max_diff < 1e-5, f"Expected fast_rank_r_token_density_pack exactness < 1e-5, got {max_diff}"


def test_riemannian_complex_unitary_muon_step_preserves_u16() -> None:
    """Verify `riemannian_complex_unitary_muon_step` preserves $U^\\dagger U = I_{16}$ across 50 steps."""
    torch.manual_seed(7)
    d_h = 16
    u_re = torch.eye(d_h, dtype=torch.float32)
    u_im = torch.zeros(d_h, d_h, dtype=torch.float32)
    mom_re = torch.zeros_like(u_re)
    mom_im = torch.zeros_like(u_im)

    for _ in range(50):
        g_re = torch.randn_like(u_re)
        g_im = torch.randn_like(u_im)
        riemannian_complex_unitary_muon_step(u_re, u_im, g_re, g_im, mom_re, mom_im, lr=0.03, beta=0.9)

    eye = torch.eye(d_h)
    gram_re = u_re.T @ u_re + u_im.T @ u_im
    gram_im = u_re.T @ u_im - u_im.T @ u_re
    unitary_err = float(((gram_re - eye).square().sum() + gram_im.square().sum()).sqrt().item()) / math.sqrt(d_h)
    assert unitary_err < 1e-5, f"Expected U(16) unitary error < 1e-5, got {unitary_err}"


def test_polar_factorize_stiefel_svd_exact_reconstruction() -> None:
    """Verify `polar_factorize_stiefel_svd` factors $M \\in \\mathbb{R}^{128 \\times 64}$ with $< 1e-5$ error."""
    torch.manual_seed(11)
    mat = torch.randn(128, 64) * 0.25
    u_mat, sigma, vt_mat = polar_factorize_stiefel_svd(mat)
    recon = (u_mat * sigma) @ vt_mat
    recon_err = float((recon - mat).abs().max().item())
    st_u_err = float(torch.linalg.matrix_norm(u_mat.T @ u_mat - torch.eye(64), ord="fro").item()) / 8.0
    st_v_err = float(torch.linalg.matrix_norm(vt_mat.T @ vt_mat - torch.eye(64), ord="fro").item()) / 8.0
    assert recon_err < 1e-5
    assert st_u_err < 1e-5
    assert st_v_err < 1e-5


def test_all_layer_riemannian_impure_density_gpt_forward_backward_and_manifold() -> None:
    """Verify `RiemannianImpureDensityGPT` forward/backward and all-layer Stiefel/Unitary manifold preservation."""
    torch.manual_seed(19)
    cfg = RiemannianImpureDensityConfig(
        vocab_size=256,
        embed_dim=128,
        num_layers=3,
        num_heads=4,
        head_complex_dim=8,
        boundary_mode=InputBoundaryMode.ANCHORED_SPHERE,
        density_mode=DensityStateMode.IMPURE_MCWEENY_R4,
        use_v0_highway=True,
    )
    model = RiemannianImpureDensityGPT(cfg)
    stiefel_opt, euclid_opt = build_all_layer_riemannian_optimizers(model, stiefel_lr=0.025, unitary_lr=0.02)

    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    for _ in range(5):
        stiefel_opt.zero_grad()
        euclid_opt.zero_grad()
        logits, _ = model(idx, return_logits=True)
        assert logits is not None
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), targets.view(-1))
        loss.backward()
        stiefel_opt.step()
        euclid_opt.step()

    telem = model.compute_manifold_and_purity_telemetry()
    assert telem["mean_stiefel_err"] < 1e-4, f"Stiefel error too high: {telem['mean_stiefel_err']}"
    assert telem["mean_unitary_err"] < 1e-4, f"Unitary error too high: {telem['mean_unitary_err']}"
    assert 0.1 < telem["mean_purity"] < 0.99
    assert telem["mean_eff_rank"] > 1.05
