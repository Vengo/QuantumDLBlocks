"""Unit tests for `EXPERIMENT_PLAN_053.md` (`Wave 38`): Custom Triton Flash-Krylov & Strict `bfloat16` Scaling."""

from __future__ import annotations

import torch

from qdlblocks.kernels import (
    factored_complex_krylov_step,
    flash_complex_krylov_attention,
    fused_causal_pboson_born_triton,
    fused_complex_norm_rope_triton,
)
from qdlblocks.layers.packed_complex_register_density_attn import (
    ComplexRoPE,
    _mix_and_krylov_probe_step,
    complex_l2_normalize,
)
from qdlblocks.models import (
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
    build_gemini_complex_optimizers,
)


def _reference_born_weights(
    q_re: torch.Tensor,
    q_im: torch.Tensor,
    k_re: torch.Tensor,
    k_im: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Reference explicit causal p=2 Born weight calculation."""
    seq_t = q_re.size(2)
    c_re = torch.matmul(q_re, k_re.transpose(-2, -1)) + torch.matmul(q_im, k_im.transpose(-2, -1))
    c_im = torch.matmul(q_re, k_im.transpose(-2, -1)) - torch.matmul(q_im, k_re.transpose(-2, -1))
    mag_sq = c_re.float().square() + c_im.float().square()
    idx_t = torch.arange(seq_t, device=q_re.device)
    causal_mask = (idx_t[:, None] >= idx_t[None, :]).view(1, 1, seq_t, seq_t)
    u = mag_sq.square().masked_fill(~causal_mask, 0.0)
    return u / (u.sum(dim=-1, keepdim=True) + eps)


def test_h1_fused_complex_norm_rope_and_associative_krylov_parity() -> None:
    """Verify `H1` forward & backward parity for Triton norm+RoPE, causal Born weights, and associative Krylov."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(53)
    b_sz, h_num, seq_t, dh = 2, 4, 64, 32
    rope = ComplexRoPE(d_register=dh).to(device)
    cos_2d, sin_2d = rope.get_cos_sin(seq_t, device, torch.float32)

    z_re = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32, requires_grad=True)
    z_im = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32, requires_grad=True)
    z_re_ref = z_re.detach().clone().requires_grad_(True)
    z_im_ref = z_im.detach().clone().requires_grad_(True)

    # 1. Fused Norm + RoPE parity (float32)
    out_re_tri, out_im_tri = fused_complex_norm_rope_triton(z_re, z_im, cos_2d, sin_2d, eps=1e-7)
    norm_sq = z_re_ref.square().sum(dim=-1, keepdim=True) + z_im_ref.square().sum(dim=-1, keepdim=True)
    inv_n = torch.rsqrt(norm_sq + 1e-7)
    nr_ref, ni_ref = z_re_ref * inv_n, z_im_ref * inv_n
    out_re_ref, out_im_ref = rope(nr_ref, ni_ref)

    max_norm_rope_err_fp32 = max(
        float((out_re_tri - out_re_ref).abs().max().item()),
        float((out_im_tri - out_im_ref).abs().max().item()),
    )
    assert max_norm_rope_err_fp32 < 1e-6

    grad_probe_re = torch.randn_like(out_re_tri)
    grad_probe_im = torch.randn_like(out_im_tri)
    (out_re_tri * grad_probe_re + out_im_tri * grad_probe_im).sum().backward()
    (out_re_ref * grad_probe_re + out_im_ref * grad_probe_im).sum().backward()
    assert z_re.grad is not None and z_re_ref.grad is not None
    assert float((z_re.grad - z_re_ref.grad).abs().max().item()) < 1e-5
    assert float((z_im.grad - z_im_ref.grad).abs().max().item()) < 1e-5

    # 2. Associative Krylov vs Reference Explicit 32x32 Hermitian Matrix Path (float32 & bfloat16)
    q_re = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32)
    q_im = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32)
    k_re = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32)
    k_im = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32)
    v_re = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32)
    v_im = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32)

    q_re, q_im = fused_complex_norm_rope_triton(q_re, q_im, cos_2d, sin_2d)
    k_re, k_im = fused_complex_norm_rope_triton(k_re, k_im, cos_2d, sin_2d)
    v_re, v_im = fused_complex_norm_rope_triton(v_re, v_im, None, None)

    qr1 = q_re.detach().clone().requires_grad_(True)
    qi1 = q_im.detach().clone().requires_grad_(True)
    kr1 = k_re.detach().clone().requires_grad_(True)
    ki1 = k_im.detach().clone().requires_grad_(True)
    vr1 = v_re.detach().clone().requires_grad_(True)
    vi1 = v_im.detach().clone().requires_grad_(True)

    qr2 = q_re.detach().clone().requires_grad_(True)
    qi2 = q_im.detach().clone().requires_grad_(True)
    kr2 = k_re.detach().clone().requires_grad_(True)
    ki2 = k_im.detach().clone().requires_grad_(True)
    vr2 = v_re.detach().clone().requires_grad_(True)
    vi2 = v_im.detach().clone().requires_grad_(True)

    u_re_new, u_im_new, tr_new, _ = flash_complex_krylov_attention(
        qr1, qi1, kr1, ki1, vr1, vi1, probe_mode="v_krylov", eps=1e-7
    )
    w_ref = _reference_born_weights(qr2, qi2, kr2, ki2, eps=1e-7)
    u_re_ref, u_im_ref, tr_ref, _ = _mix_and_krylov_probe_step(
        w_ref, vr2, vi2, qr2, qi2, dh=dh, use_vkrylov=True, eps=1e-7
    )

    max_krylov_fwd_err_fp32 = max(
        float((u_re_new - u_re_ref).abs().max().item()),
        float((u_im_new - u_im_ref).abs().max().item()),
    )
    assert max_krylov_fwd_err_fp32 < 1e-5
    assert abs(float(tr_new.item()) - float(tr_ref.item())) < 1e-5

    up_re = torch.randn_like(u_re_new)
    up_im = torch.randn_like(u_im_new)
    (u_re_new * up_re + u_im_new * up_im).sum().backward()
    (u_re_ref * up_re + u_im_ref * up_im).sum().backward()

    for g_new, g_ref in [(qr1.grad, qr2.grad), (kr1.grad, kr2.grad), (vr1.grad, vr2.grad)]:
        assert g_new is not None and g_ref is not None
        cos_sim = float(
            torch.nn.functional.cosine_similarity(g_new.reshape(1, -1), g_ref.reshape(1, -1)).item()
        )
        assert cos_sim > 0.999


def _collect_module_output_dtypes(model: GeminiPackedComplexRoPEGPT, idx: torch.Tensor, targets: torch.Tensor) -> tuple[list[torch.dtype], torch.Tensor, torch.Tensor]:
    """Run forward pass and return `(dtypes, loss_train, loss_eval)` without nested hook closures."""
    det_re, det_im = model.synthesize_complex_detectors()
    psi0_re = model.embed_re.index_select(0, idx.reshape(-1)).view(idx.size(0), idx.size(1), model.cfg.d_complex)
    psi0_im = model.embed_im.index_select(0, idx.reshape(-1)).view(idx.size(0), idx.size(1), model.cfg.d_complex)
    psi_re, psi_im = complex_l2_normalize(psi0_re, psi0_im, dim=-1)
    dtypes = [det_re.dtype, det_im.dtype, psi_re.dtype, psi_im.dtype]
    v0_re, v0_im = None, None
    for l_idx, block in enumerate(model.blocks):
        psi_re, psi_im, vr, vi = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
        dtypes.extend([psi_re.dtype, psi_im.dtype, vr.dtype, vi.dtype])
        if l_idx == 0:
            v0_re, v0_im = vr, vi
    _, loss_train = model(idx, targets=targets, return_logits=False)
    _, loss_eval = model(idx, targets=targets, return_logits=True)
    assert loss_train is not None and loss_eval is not None
    return dtypes, loss_train, loss_eval


def test_h2_strict_bf16_no_float32_leaks_and_complex_linear_autograd() -> None:
    """Verify `float32_activation_leak_count == 0` under CUDA `bfloat16` autocast and readout parity."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(5302)
    cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=512,
        valid_vocab_size=500,
        d_complex=64,
        n_registers=2,
        d_register=32,
        num_layers=2,
        block_size=32,
    )
    model = GeminiPackedComplexRoPEGPT(cfg).to(device)
    idx = torch.randint(0, 500, (2, 32), device=device)
    targets = torch.randint(0, 500, (2, 32), device=device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        recorded_dtypes, loss_train, loss_eval = _collect_module_output_dtypes(model, idx, targets)

    float32_leak_count = sum(1 for dt in recorded_dtypes if dt == torch.float32)
    assert float32_leak_count == 0
    assert abs(float(loss_train.item()) - float(loss_eval.item())) < 5e-3


def test_h3_checkpointed_multi_register_scaling_forward_backward() -> None:
    """Verify `GeminiPackedComplexRoPEGPT` with `checkpoint_blocks=True` at `H = 8` registers (`d_complex = 256`)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(5303)
    cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=512,
        valid_vocab_size=500,
        d_complex=256,
        n_registers=8,
        d_register=32,
        num_layers=2,
        block_size=32,
        checkpoint_blocks=True,
    )
    model = GeminiPackedComplexRoPEGPT(cfg).to(device)
    opt_muon, opt_adam = build_gemini_complex_optimizers(model)
    idx = torch.randint(0, 500, (2, 32), device=device)
    targets = torch.randint(0, 500, (2, 32), device=device)

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )
    with ctx:
        _, loss = model(idx, targets=targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    opt_muon.step()
    opt_adam.step()


if __name__ == "__main__":
    for fn in (
        test_h1_fused_complex_norm_rope_and_associative_krylov_parity,
        test_h2_strict_bf16_no_float32_leaks_and_complex_linear_autograd,
        test_h3_checkpointed_multi_register_scaling_forward_backward,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 3 Plan 053 unit & hypothesis tests PASSED.")

