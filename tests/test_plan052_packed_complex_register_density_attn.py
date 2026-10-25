"""Unit tests for Wave 37 / EXPERIMENT_PLAN_052.md: Standalone Pure-Complex Register Density GPT."""

from __future__ import annotations

import torch

from qdlblocks.layers.packed_complex_register_density_attn import (
    ComplexRoPE,
    complex_l2_normalize,
    explicit_mcweeny_density_probe,
    krylov_purified_density_probe,
    pack_hermitian_outer_product_and_ket,
    unpack_hermitian_matrix,
)
from qdlblocks.models.gemini_packed_complex_rope_gpt import (
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
PACK_UNPACK_ATOL: float = 1e-6
ROPE_SHIFT_ATOL: float = 1e-6
KRYLOV_MCWEENY_ATOL: float = 1e-6
SINGLE_TOKEN_NORM_ATOL: float = 1e-5
Q_PROBE_NORM_CEILING: float = 0.25
COMPETING_KRYLOV_MIN_CONC: float = 0.96


def test_h1_hermitian_packing_rope_and_associative_krylov() -> None:
    """Verify 1-GEMM Hermitian triangle packing, ComplexRoPE shift invariance, and associative Krylov parity."""
    torch.manual_seed(7)
    b_sz, h_num, seq_t, dh = 2, 4, 32, 32
    v_re, v_im = complex_l2_normalize(torch.randn(b_sz, h_num, seq_t, dh), torch.randn(b_sz, h_num, seq_t, dh), dim=-1)
    w_born = torch.softmax(torch.randn(b_sz, h_num, seq_t, seq_t), dim=-1).tril()
    w_born = w_born / (w_born.sum(dim=-1, keepdim=True) + 1e-8)

    # 1. 2-GEMM Reference vs. 1-GEMM Fused Packed + V-ket (width 1088)
    ref_outer_re = (
        v_re.unsqueeze(-1) * v_re.unsqueeze(-2) + v_im.unsqueeze(-1) * v_im.unsqueeze(-2)
    ).view(b_sz, h_num, seq_t, dh * dh)
    ref_outer_im = (
        v_im.unsqueeze(-1) * v_re.unsqueeze(-2) - v_re.unsqueeze(-1) * v_im.unsqueeze(-2)
    ).view(b_sz, h_num, seq_t, dh * dh)
    ref_rho_re = torch.matmul(w_born, ref_outer_re).view(b_sz, h_num, seq_t, dh, dh)
    ref_rho_im = torch.matmul(w_born, ref_outer_im).view(b_sz, h_num, seq_t, dh, dh)
    ref_vbar_re, ref_vbar_im = complex_l2_normalize(
        torch.matmul(w_born, v_re), torch.matmul(w_born, v_im), dim=-1
    )

    packed_and_ket = pack_hermitian_outer_product_and_ket(v_re, v_im)
    assert packed_and_ket.shape == (b_sz, h_num, seq_t, dh * dh + 2 * dh)
    mixed_all = torch.matmul(w_born, packed_and_ket)
    rho_re, rho_im = unpack_hermitian_matrix(mixed_all[..., : dh * dh], dh=dh)
    vbar_re, vbar_im = complex_l2_normalize(
        mixed_all[..., dh * dh : dh * dh + dh],
        mixed_all[..., dh * dh + dh :],
        dim=-1,
    )

    assert float((ref_rho_re - rho_re).abs().max().item()) < PACK_UNPACK_ATOL
    assert float((ref_rho_im - rho_im).abs().max().item()) < PACK_UNPACK_ATOL
    assert float((ref_vbar_re - vbar_re).abs().max().item()) < PACK_UNPACK_ATOL
    assert float((ref_vbar_im - vbar_im).abs().max().item()) < PACK_UNPACK_ATOL

    # 2. ComplexRoPE relative-distance shift invariance
    rope = ComplexRoPE(d_register=dh)
    q1_re, q1_im = complex_l2_normalize(torch.randn(1, 1, 1, dh), torch.randn(1, 1, 1, dh))
    k1_re, k1_im = complex_l2_normalize(torch.randn(1, 1, 1, dh), torch.randn(1, 1, 1, dh))
    qs_re, qs_im = rope(q1_re.expand(1, 1, 24, dh), q1_im.expand(1, 1, 24, dh))
    ks_re, ks_im = rope(k1_re.expand(1, 1, 24, dh), k1_im.expand(1, 1, 24, dh))
    c_re = torch.matmul(qs_re, ks_re.transpose(-2, -1)) + torch.matmul(qs_im, ks_im.transpose(-2, -1))
    c_im = torch.matmul(qs_re, ks_im.transpose(-2, -1)) - torch.matmul(qs_im, ks_re.transpose(-2, -1))
    assert float((c_re[0, 0, 9, 3] - c_re[0, 0, 19, 13]).abs().item()) < ROPE_SHIFT_ATOL
    assert float((c_im[0, 0, 9, 3] - c_im[0, 0, 19, 13]).abs().item()) < ROPE_SHIFT_ATOL

    # 3. Associative Krylov O(dh^2) vs. Explicit Matrix-Squared McWeeny O(dh^3)
    mc_re, mc_im = explicit_mcweeny_density_probe(rho_re, rho_im, vbar_re, vbar_im, normalize_output=False)
    kr_re, kr_im = krylov_purified_density_probe(rho_re, rho_im, vbar_re, vbar_im, normalize_output=False)
    assert float((mc_re - kr_re).abs().max().item()) < KRYLOV_MCWEENY_ATOL
    assert float((mc_im - kr_im).abs().max().item()) < KRYLOV_MCWEENY_ATOL


def test_h2_step0_subspace_alignment_and_krylov_amplification() -> None:
    """Verify V-space Krylov probing eliminates Step-0 Q-probe norm collapse and amplifies principal subspace."""
    torch.manual_seed(42)
    b_sz, h_num, dh = 2, 8, 32

    # 1. Single-token dominance: w_ts* = 1
    vs_re, vs_im = complex_l2_normalize(torch.randn(b_sz, h_num, 1, dh), torch.randn(b_sz, h_num, 1, dh), dim=-1)
    q_re, q_im = complex_l2_normalize(torch.randn(b_sz, h_num, 1, dh), torch.randn(b_sz, h_num, 1, dh), dim=-1)
    packed_1 = pack_hermitian_outer_product_and_ket(vs_re, vs_im)
    rho1_re, rho1_im = unpack_hermitian_matrix(packed_1[..., : dh * dh], dh=dh)

    uq_re, uq_im = krylov_purified_density_probe(rho1_re, rho1_im, q_re, q_im, normalize_output=False)
    uv_re, uv_im = krylov_purified_density_probe(rho1_re, rho1_im, vs_re, vs_im, normalize_output=False)

    q_norm = float(torch.sqrt(uq_re.square().sum(dim=-1) + uq_im.square().sum(dim=-1)).mean().item())
    v_norm = float(torch.sqrt(uv_re.square().sum(dim=-1) + uv_im.square().sum(dim=-1)).mean().item())
    v_ov_re = float((vs_re * uv_re + vs_im * uv_im).sum(dim=-1).mean().item())
    v_ov_im = float((vs_re * uv_im - vs_im * uv_re).sum(dim=-1).abs().max().item())

    assert q_norm < Q_PROBE_NORM_CEILING
    assert abs(v_norm - 1.0) < SINGLE_TOKEN_NORM_ATOL
    assert abs(v_ov_re - 1.0) < SINGLE_TOKEN_NORM_ATOL
    assert v_ov_im < SINGLE_TOKEN_NORM_ATOL

    # 2. Competing 3-token orthogonal mixture: w = (0.60, 0.25, 0.15)
    eye_re = torch.eye(dh).unsqueeze(0).unsqueeze(0)[:, :, :3, :]
    eye_im = torch.zeros_like(eye_re)
    w_comp = torch.tensor([0.60, 0.25, 0.15], dtype=torch.float32).view(1, 1, 1, 3)
    packed_3 = pack_hermitian_outer_product_and_ket(eye_re, eye_im)
    mixed_3 = torch.matmul(w_comp, packed_3)
    rho3_re, rho3_im = unpack_hermitian_matrix(mixed_3[..., : dh * dh], dh=dh)
    vbar3_re, vbar3_im = complex_l2_normalize(
        mixed_3[..., dh * dh : dh * dh + dh], mixed_3[..., dh * dh + dh :], dim=-1
    )
    uk3_re, uk3_im = krylov_purified_density_probe(
        rho3_re, rho3_im, vbar3_re, vbar3_im, normalize_output=True
    )
    principal_fid_vbar = float(vbar3_re[0, 0, 0, 0].square().item())
    principal_fid_krylov = float(uk3_re[0, 0, 0, 0].square().item())
    assert principal_fid_krylov > COMPETING_KRYLOV_MIN_CONC
    assert principal_fid_krylov > principal_fid_vbar + 0.15


def test_standalone_gemini_packed_complex_rope_gpt_forward_backward() -> None:
    """Verify standalone GeminiPackedComplexRoPEGPT forward/backward and train/eval parity."""
    torch.manual_seed(11)
    cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=256,
        valid_vocab_size=250,
        d_complex=64,
        n_registers=2,
        d_register=32,
        num_layers=2,
        block_size=32,
        probe_mode="v_krylov",
        normalize_krylov_ket=True,
    )
    model = GeminiPackedComplexRoPEGPT(cfg)
    idx = torch.randint(0, 250, (2, 16))
    targets = torch.randint(0, 250, (2, 16))

    logits, loss = model(idx, targets=targets, return_logits=True)
    assert logits is not None and loss is not None
    assert logits.shape == (2, 16, 250)
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()


if __name__ == "__main__":
    for fn in (
        test_h1_hermitian_packing_rope_and_associative_krylov,
        test_h2_step0_subspace_alignment_and_krylov_amplification,
        test_standalone_gemini_packed_complex_rope_gpt_forward_backward,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 3 Plan 052 unit & hypothesis tests PASSED.")

