"""Unit tests for Wave 46 (`EXPERIMENT_PLAN_061.md`) `impure_density_mirror_block.py`."""

from __future__ import annotations

import math

import torch

from qdlblocks.layers.impure_density_mirror_block import (
    ImpureQuantumDensityMirrorBlock,
    PureQuantumKetMirrorBlock,
    QuantumMirrorBlockConfig,
    RiemannianStiefelMuon,
    bures_wasserstein_mcweeny_step,
    complex_l2_normalize,
    compute_layer_mirroring_loss,
    extract_rank2_krylov_eigenpair,
    pack_hermitian_isometric,
    pack_rank1_ket_outer_isometric,
    riemannian_stiefel_muon_step,
    unpack_hermitian_isometric,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)


def _random_hermitian_pair(batch: int, d_h: int) -> tuple[torch.Tensor, torch.Tensor]:
    a_re = torch.randn(batch, d_h, d_h, dtype=torch.float32)
    a_im = torch.randn(batch, d_h, d_h, dtype=torch.float32)
    h_re = 0.5 * (a_re + a_re.transpose(-2, -1))
    h_im = 0.5 * (a_im - a_im.transpose(-2, -1))
    return h_re, h_im


def test_hermitian_isometric_packing_preserves_exact_trace_and_inverse() -> None:
    """Verify `<P(A), P(B)> == Tr(AB)` and `P^{-1}(P(A)) == A` to machine precision."""
    torch.manual_seed(4601)
    d_h = 16
    a_re, a_im = _random_hermitian_pair(8, d_h)
    b_re, b_im = _random_hermitian_pair(8, d_h)

    pa = pack_hermitian_isometric(a_re, a_im)
    pb = pack_hermitian_isometric(b_re, b_im)
    assert pa.shape == (8, d_h * d_h)

    dot_packed = (pa * pb).sum(dim=-1)
    ab_re = torch.matmul(a_re, b_re) - torch.matmul(a_im, b_im)
    exact_trace = torch.diagonal(ab_re, dim1=-2, dim2=-1).sum(dim=-1)
    assert torch.allclose(dot_packed, exact_trace, atol=1e-4, rtol=1e-4)

    rec_re, rec_im = unpack_hermitian_isometric(pa, d_h)
    assert torch.allclose(rec_re, a_re, atol=1e-5)
    assert torch.allclose(rec_im, a_im, atol=1e-5)

    v_re, v_im = complex_l2_normalize(torch.randn(8, d_h), torch.randn(8, d_h))
    pv = pack_rank1_ket_outer_isometric(v_re, v_im)
    norms = pv.square().sum(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_mcweeny_preserves_rank2_fixed_point_and_suppresses_midpoint_hallucination() -> None:
    """Verify McWeeny squaring preserves polysemous Rank-2 impurity and eliminates midpoint hallucination."""
    torch.manual_seed(4602)
    d_h = 16
    ket_a_re = torch.zeros(1, d_h)
    ket_a_im = torch.zeros(1, d_h)
    ket_a_re[0, 0] = 1.0

    ket_b_re = torch.zeros(1, d_h)
    ket_b_im = torch.zeros(1, d_h)
    ket_b_re[0, 1] = 1.0

    ket_c_re = torch.zeros(1, d_h)
    ket_c_im = torch.zeros(1, d_h)
    ket_c_re[0, 0] = 1.0 / math.sqrt(2.0)
    ket_c_re[0, 1] = 1.0 / math.sqrt(2.0)

    pa = pack_rank1_ket_outer_isometric(ket_a_re, ket_a_im)
    pb = pack_rank1_ket_outer_isometric(ket_b_re, ket_b_im)
    pc = pack_rank1_ket_outer_isometric(ket_c_re, ket_c_im)

    p_mix = 0.5 * pa + 0.5 * pb
    rho_re, rho_im = unpack_hermitian_isometric(p_mix, d_h)
    rho_mc_re, rho_mc_im, purity, eff_rank = bures_wasserstein_mcweeny_step(rho_re, rho_im, num_steps=1)

    assert abs(float(purity.item()) - 0.5) < 1e-5
    assert abs(float(eff_rank.item()) - 2.0) < 1e-5

    p_mc = pack_hermitian_isometric(rho_mc_re, rho_mc_im)
    prob_impure_a = float((p_mc * pa).sum().item()) ** 2
    prob_impure_b = float((p_mc * pb).sum().item()) ** 2
    prob_impure_c = float((p_mc * pc).sum().item()) ** 2

    psi_pure_re, psi_pure_im = complex_l2_normalize(0.5 * (ket_a_re + ket_b_re), 0.5 * (ket_a_im + ket_b_im))
    p_pure = pack_rank1_ket_outer_isometric(psi_pure_re, psi_pure_im)
    prob_pure_a = float((p_pure * pa).sum().item()) ** 2
    prob_pure_c = float((p_pure * pc).sum().item()) ** 2

    assert prob_pure_c / max(1e-8, prob_pure_a) >= 3.99
    assert abs(prob_impure_a - prob_impure_b) < 1e-6
    assert abs(prob_impure_c - prob_impure_a) < 1e-6

    krylov = extract_rank2_krylov_eigenpair(rho_mc_re, rho_mc_im)
    ov_re = (krylov["phi1_re"] * krylov["phi2_re"] + krylov["phi1_im"] * krylov["phi2_im"]).sum(dim=-1)
    ov_im = (krylov["phi1_re"] * krylov["phi2_im"] - krylov["phi1_im"] * krylov["phi2_re"]).sum(dim=-1)
    ortho_err = float(torch.sqrt(ov_re.square() + ov_im.square()).max().item())
    assert ortho_err < 1e-5


def test_riemannian_stiefel_muon_geodesic_vs_euclidean_drift() -> None:
    """Verify `riemannian_stiefel_muon_step` preserves $W^T W = I_d$ and matches Cayley geodesic."""
    torch.manual_seed(4603)
    d = 32
    w_riem = torch.empty(d, d, dtype=torch.float32)
    torch.nn.init.orthogonal_(w_riem)
    w_euc = w_riem.clone()
    buf_riem = torch.zeros_like(w_riem)
    buf_euc = torch.zeros_like(w_euc)

    lr = 0.03
    for step_idx in range(30):
        grad = torch.randn(d, d, dtype=torch.float32)
        if step_idx == 0:
            w_cayley_ref = w_riem.clone()
            buf_c = torch.zeros_like(w_riem)
            riemannian_stiefel_muon_step(w_riem, grad, buf_riem, lr=lr, beta=0.9)
            omega_hat = torch.matmul(w_cayley_ref.T, (w_cayley_ref - w_riem) * math.sqrt(1.0 + lr * lr) / lr)
            omega_hat = 0.5 * (omega_hat - omega_hat.T)
            eye = torch.eye(d, dtype=torch.float32)
            cayley_next = torch.matmul(
                w_cayley_ref,
                torch.linalg.solve(eye + 0.5 * lr * omega_hat, eye - 0.5 * lr * omega_hat),
            )
            cos_cayley = torch.nn.functional.cosine_similarity(
                w_riem.flatten(), cayley_next.flatten(), dim=0
            )
            assert float(cos_cayley.item()) > 0.99999
            _ = buf_c
        else:
            riemannian_stiefel_muon_step(w_riem, grad, buf_riem, lr=lr, beta=0.9)

        buf_euc.mul_(0.9).add_(grad)
        u_euc, _, vt_euc = torch.linalg.svd(grad + 0.9 * buf_euc, full_matrices=False)
        w_euc = w_euc - lr * torch.matmul(u_euc, vt_euc)

    eye_d = torch.eye(d, dtype=torch.float32)
    riem_err = float(torch.linalg.matrix_norm(w_riem.T @ w_riem - eye_d).item()) / math.sqrt(d)
    euc_err = float(torch.linalg.matrix_norm(w_euc.T @ w_euc - eye_d).item()) / math.sqrt(d)
    assert riem_err < 1e-4
    assert euc_err > 0.15


def test_impure_and_pure_mirror_blocks_forward_backward_and_grafting() -> None:
    """Verify `ImpureQuantumDensityMirrorBlock` gradient flow, mirroring loss, and surgical grafting."""
    torch.manual_seed(4604)
    cfg = QuantumMirrorBlockConfig(
        embed_dim=64,
        num_heads=4,
        head_complex_dim=8,
        channel_hidden_dim=128,
        max_seq_len=32,
    )
    impure_blk = ImpureQuantumDensityMirrorBlock(cfg)
    pure_blk = PureQuantumKetMirrorBlock(cfg)

    x_in = torch.randn(2, 16, 64)
    target_out = x_in + 0.2 * torch.randn(2, 16, 64)

    out_impure, _ = impure_blk(x_in)
    loss_dict = compute_layer_mirroring_loss(out_impure, x_in, target_out)
    loss_dict["total_loss"].backward()

    opt = RiemannianStiefelMuon(impure_blk.stiefel_parameters(), lr=0.02)
    opt.step()
    assert 0.05 <= impure_blk.last_telemetry["mean_purity"] <= 1.0
    assert impure_blk.last_telemetry["mean_eff_rank"] >= 1.0

    out_pure, _ = pure_blk(x_in)
    assert out_pure.shape == (2, 16, 64)

    t_cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=256,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        hidden_dim=128,
        max_seq_len=32,
        num_value_embeddings=1,
        skip_attn_layers=(),
        bfloat16_weights=False,
    )
    teacher = CompiledLeaderboardClassicalGPT(t_cfg)
    teacher.blocks[1] = ImpureQuantumDensityMirrorBlock(cfg)
    idx = torch.randint(0, 200, (2, 16), dtype=torch.long)
    logits, loss = teacher(idx, idx, return_logits=True)
    assert logits is not None and loss is not None
    assert math.isfinite(float(loss.item()))
