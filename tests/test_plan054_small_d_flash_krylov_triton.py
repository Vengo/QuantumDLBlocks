"""Unit & Hypothesis Tests for `EXPERIMENT_PLAN_054.md` (`Wave 39` — Arbitrary Multiples of 32 & Scale-Invariant Krylov)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qdlblocks.kernels.flash_complex_krylov_triton import (
    flash_complex_krylov_attention,
    fused_complex_norm_rope_triton,
)
from qdlblocks.layers.packed_complex_register_density_attn import (
    ComplexRoPE,
    complex_l2_normalize,
    krylov_purified_density_probe,
    pack_hermitian_outer_product_and_ket,
    unpack_hermitian_matrix,
)
from qdlblocks.models.gemini_packed_complex_rope_gpt import (
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
    build_gemini_complex_optimizers,
)


def _reference_hermitian_unit_krylov(
    q_re: torch.Tensor,
    q_im: torch.Tensor,
    k_re: torch.Tensor,
    k_im: torch.Tensor,
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference explicit Hermitian triangle-pack + V-space Krylov + unit-sphere normalization."""
    seq_t = q_re.size(2)
    dh = v_re.size(-1)
    c_re = torch.matmul(q_re, k_re.transpose(-2, -1)) + torch.matmul(q_im, k_im.transpose(-2, -1))
    c_im = torch.matmul(q_re, k_im.transpose(-2, -1)) - torch.matmul(q_im, k_re.transpose(-2, -1))
    idx_t = torch.arange(seq_t, device=q_re.device)
    causal = (idx_t[:, None] >= idx_t[None, :]).view(1, 1, seq_t, seq_t)
    u = (c_re.square() + c_im.square()).square().masked_fill(~causal, 0.0)
    w = u / (u.sum(dim=-1, keepdim=True) + eps)

    v_pk = pack_hermitian_outer_product_and_ket(v_re, v_im)
    m_all = torch.matmul(w, v_pk)
    d_sq = dh * dh
    r_re, r_im = unpack_hermitian_matrix(m_all[..., :d_sq], dh)
    vb_re = m_all[..., d_sq : d_sq + dh]
    vb_im = m_all[..., d_sq + dh : d_sq + 2 * dh]
    pr_re, pr_im = complex_l2_normalize(vb_re, vb_im, dim=-1, eps=eps)
    u_raw_re, u_raw_im = krylov_purified_density_probe(r_re, r_im, pr_re, pr_im, eps=eps)
    n_sq = u_raw_re.square().sum(dim=-1, keepdim=True) + u_raw_im.square().sum(dim=-1, keepdim=True)
    inv_n = torch.rsqrt(n_sq + eps)
    return u_raw_re * inv_n, u_raw_im * inv_n


def test_h1_nonpow2_norm_rope_and_scale_invariant_krylov_parity() -> None:
    """Verify masked Triton Norm+RoPE across D in {16..256} and scale-invariant `_norm` Krylov parity."""
    torch.manual_seed(5401)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_norm_err = 0.0
    for d_dim in (16, 24, 32, 64, 96, 128, 160, 192, 256):
        z_re = torch.randn(2, 3, 32, d_dim, device=device, dtype=torch.float32, requires_grad=True)
        z_im = torch.randn(2, 3, 32, d_dim, device=device, dtype=torch.float32, requires_grad=True)
        rope = ComplexRoPE(d_register=d_dim).to(device)
        cos_2d, sin_2d = rope.get_cos_sin(32, device, torch.float32)

        tr_re, tr_im = fused_complex_norm_rope_triton(z_re, z_im, cos_2d, sin_2d, eps=1e-7)
        n_sq = z_re.square().sum(dim=-1, keepdim=True) + z_im.square().sum(dim=-1, keepdim=True)
        inv_n = torch.rsqrt(n_sq + 1e-7)
        nr, ni = z_re * inv_n, z_im * inv_n
        c = cos_2d.view(1, 1, 32, d_dim)
        s = sin_2d.view(1, 1, 32, d_dim)
        ref_re = nr * c - ni * s
        ref_im = nr * s + ni * c

        err = max(float((tr_re - ref_re).abs().max().item()), float((tr_im - ref_im).abs().max().item()))
        max_norm_err = max(max_norm_err, err)
    assert max_norm_err < 1e-6, f"max_nonpow2_norm_rope_err_fp32={max_norm_err}"

    max_krylov_err = 0.0
    min_grad_cos = 1.0
    for dh, h_num in ((16, 2), (32, 3), (32, 5)):
        seq_t = 48
        rope = ComplexRoPE(d_register=dh).to(device)
        cos_2d, sin_2d = rope.get_cos_sin(seq_t, device, torch.float32)
        q_re, q_im = fused_complex_norm_rope_triton(
            torch.randn(2, h_num, seq_t, dh, device=device),
            torch.randn(2, h_num, seq_t, dh, device=device),
            cos_2d,
            sin_2d,
        )
        k_re, k_im = fused_complex_norm_rope_triton(
            torch.randn(2, h_num, seq_t, dh, device=device),
            torch.randn(2, h_num, seq_t, dh, device=device),
            cos_2d,
            sin_2d,
        )
        v_re, v_im = fused_complex_norm_rope_triton(
            torch.randn(2, h_num, seq_t, dh, device=device),
            torch.randn(2, h_num, seq_t, dh, device=device),
            None,
            None,
        )

        qr1, qi1 = q_re.detach().clone().requires_grad_(True), q_im.detach().clone().requires_grad_(True)
        kr1, ki1 = k_re.detach().clone().requires_grad_(True), k_im.detach().clone().requires_grad_(True)
        vr1, vi1 = v_re.detach().clone().requires_grad_(True), v_im.detach().clone().requires_grad_(True)

        qr2, qi2 = q_re.detach().clone().requires_grad_(True), q_im.detach().clone().requires_grad_(True)
        kr2, ki2 = k_re.detach().clone().requires_grad_(True), k_im.detach().clone().requires_grad_(True)
        vr2, vi2 = v_re.detach().clone().requires_grad_(True), v_im.detach().clone().requires_grad_(True)

        u_re_fast, u_im_fast, _, _ = flash_complex_krylov_attention(
            qr1, qi1, kr1, ki1, vr1, vi1, probe_mode="v_krylov_norm", eps=1e-7
        )
        u_re_ref, u_im_ref = _reference_hermitian_unit_krylov(qr2, qi2, kr2, ki2, vr2, vi2, eps=1e-7)

        fwd_err = max(
            float((u_re_fast - u_re_ref).abs().max().item()),
            float((u_im_fast - u_im_ref).abs().max().item()),
        )
        max_krylov_err = max(max_krylov_err, fwd_err)

        up_re, up_im = torch.randn_like(u_re_fast), torch.randn_like(u_im_fast)
        (u_re_fast * up_re + u_im_fast * up_im).sum().backward()
        (u_re_ref * up_re + u_im_ref * up_im).sum().backward()

        for g1, g2 in ((qr1.grad, qr2.grad), (kr1.grad, kr2.grad), (vr1.grad, vr2.grad)):
            assert g1 is not None and g2 is not None
            cos_sim = float(F.cosine_similarity(g1.reshape(1, -1), g2.reshape(1, -1)).item())
            min_grad_cos = min(min_grad_cos, cos_sim)

    assert max_krylov_err < 1e-5, f"max_fast_krylov_unit_err_fp32={max_krylov_err}"
    assert min_grad_cos > 0.999, f"min_fast_krylov_grad_cos={min_grad_cos}"


def test_h2_small_d_multiples_of_32_forward_backward() -> None:
    """Verify forward+backward+step for d_complex in {32, 64, 96, 128, 160, 192} under bfloat16 autocast."""
    torch.manual_seed(5402)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = (
        (32, 2, 16),
        (64, 2, 32),
        (96, 3, 32),
        (128, 4, 32),
        (160, 5, 32),
        (192, 6, 32),
    )
    for d_c, h_num, dh in specs:
        cfg = GeminiPackedComplexRoPEConfig(
            vocab_size=256,
            valid_vocab_size=250,
            d_complex=d_c,
            n_registers=h_num,
            d_register=dh,
            num_layers=2,
            block_size=32,
            vocab_chunk_size=128,
        )
        model = GeminiPackedComplexRoPEGPT(cfg).to(device)
        opt_muon, opt_adam = build_gemini_complex_optimizers(model)
        idx = torch.randint(0, 250, (2, 32), device=device)
        targets = torch.randint(0, 250, (2, 32), device=device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            det_re, _ = model.synthesize_complex_detectors()
            if device.type == "cuda":
                assert det_re.dtype == torch.bfloat16
            _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        opt_muon.step()
        opt_adam.step()
        stiefel_err = model.retract_complex_unitary_weights(rate=0.05)
        assert stiefel_err < 0.25


if __name__ == "__main__":
    test_h1_nonpow2_norm_rope_and_scale_invariant_krylov_parity()
    print("PASS: test_h1_nonpow2_norm_rope_and_scale_invariant_krylov_parity")
    test_h2_small_d_multiples_of_32_forward_backward()
    print("PASS: test_h2_small_d_multiples_of_32_forward_backward")
    print("All Plan 054 unit tests PASSED.")
