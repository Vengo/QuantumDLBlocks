"""Unit tests for Plan 016: Speedrun Systems, Architectural Synergy, and Optimizer Ablations."""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from qdlblocks.functional import (
    apply_unitary_rope,
    l2_normalize,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)
from qdlblocks.kernels.flash_born import (
    flash_born_attention,
    flash_purified_born_attention,
)
from qdlblocks.layers.attention import (
    SubspaceProjectorBornAttention,
    compute_document_ids,
)
from qdlblocks.layers.direct_chunked_ce import (
    direct_chunked_povm_ce,
    direct_chunked_softcap_linear_ce,
)
from qdlblocks.layers.early_epoch_quantum import purified_born_attention_weights
from qdlblocks.layers.isometric import UnitaryRoPE
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks.optim.muon import StiefelMuon
from qdlblocks_jax.speedrun_primitives import (
    compute_document_ids_jax,
    half_truncated_rope_cos_sin_jax,
    zeropower_via_newtonschulz5_batched_jax,
)


def test_direct_chunked_ce_gradients_and_single_pass_gemm() -> None:
    """Verify exact loss/gradient parity and single-pass F.linear invocation per chunk."""
    torch.manual_seed(0)
    n_tok, d, v = 64, 32, 128
    chunk_size = 16
    expected_chunks = n_tok // chunk_size

    # 1. Softcap Linear CE
    x_ref = torch.randn(n_tok, d, dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(v, d, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, v, (n_tok,), dtype=torch.long)
    softcap = 30.0

    raw_ref = F.linear(x_ref, w_ref)
    logits_ref = softcap * torch.tanh(raw_ref / softcap)
    loss_ref = F.cross_entropy(logits_ref, targets)
    loss_ref.backward()

    x_test = x_ref.detach().clone().requires_grad_(True)
    w_test = w_ref.detach().clone().requires_grad_(True)

    orig_linear = F.linear
    call_count = 0

    def counting_linear(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_linear(*args, **kwargs)

    with patch("torch.nn.functional.linear", side_effect=counting_linear):
        loss_test = direct_chunked_softcap_linear_ce(
            x_test, w_test, targets, softcap=softcap, chunk_size=chunk_size
        )
        loss_test.backward()

    assert call_count == expected_chunks, f"Expected {expected_chunks} F.linear calls, got {call_count}"
    assert abs(loss_test.item() - loss_ref.item()) < 1e-5
    assert torch.max(torch.abs(x_test.grad - x_ref.grad)).item() < 1e-5
    assert torch.max(torch.abs(w_test.grad - w_ref.grad)).item() < 1e-5

    # 2. POVM CE with per-token inv_t and log_prior
    phi_ref = l2_normalize(torch.randn(n_tok, d, dtype=torch.float32), dim=-1).detach().requires_grad_(True)
    m_seed_ref = torch.randn(v, d, dtype=torch.float32, requires_grad=True)
    inv_t_ref = (20.0 + 5.0 * torch.randn(n_tok, 1, dtype=torch.float32)).detach().requires_grad_(True)
    prior_ref = (0.1 * torch.randn(v, dtype=torch.float32)).detach().requires_grad_(True)

    m_norm_ref = l2_normalize(m_seed_ref, dim=1, eps=1e-8)
    povm_logits_ref = F.linear(phi_ref, m_norm_ref) * inv_t_ref + prior_ref
    povm_loss_ref = F.cross_entropy(povm_logits_ref, targets)
    povm_loss_ref.backward()

    phi_test = phi_ref.detach().clone().requires_grad_(True)
    m_seed_test = m_seed_ref.detach().clone().requires_grad_(True)
    inv_t_test = inv_t_ref.detach().clone().requires_grad_(True)
    prior_test = prior_ref.detach().clone().requires_grad_(True)

    call_count_povm = 0

    def counting_linear_povm(*args, **kwargs):
        nonlocal call_count_povm
        call_count_povm += 1
        return orig_linear(*args, **kwargs)

    with patch("torch.nn.functional.linear", side_effect=counting_linear_povm):
        povm_loss_test = direct_chunked_povm_ce(
            phi_test,
            m_seed_test,
            inv_t_test,
            targets,
            log_prior=prior_test,
            chunk_size=chunk_size,
            eps=1e-8,
        )
        povm_loss_test.backward()

    assert call_count_povm == expected_chunks
    assert abs(povm_loss_test.item() - povm_loss_ref.item()) < 1e-5
    assert torch.max(torch.abs(phi_test.grad - phi_ref.grad)).item() < 1e-5
    assert torch.max(torch.abs(m_seed_test.grad - m_seed_ref.grad)).item() < 1e-5
    assert torch.max(torch.abs(inv_t_test.grad - inv_t_ref.grad)).item() < 1e-5
    assert torch.max(torch.abs(prior_test.grad - prior_ref.grad)).item() < 1e-5


def test_batched_newton_schulz_and_momentum_warmup() -> None:
    """Verify 3D batched Newton-Schulz parity and StiefelMuon linear momentum warmup."""
    torch.manual_seed(1)
    for shape in [(6, 64, 64), (4, 96, 64), (4, 64, 96)]:
        g_stack = torch.randn(*shape, dtype=torch.float32)
        u_batch = zeropower_via_newtonschulz5_batched(g_stack, steps=5)
        u_seq = torch.stack([zeropower_via_newtonschulz5(g_stack[i], steps=5) for i in range(shape[0])], dim=0)
        max_diff = torch.max(torch.abs(u_batch - u_seq)).item()
        assert max_diff < 1e-4, f"Batched NS mismatch {max_diff} for shape {shape}"

    p1 = torch.nn.Parameter(torch.randn(32, 32))
    p2 = torch.nn.Parameter(torch.randn(32, 32))
    opt = StiefelMuon(
        [p1, p2],
        lr=0.02,
        momentum=0.95,
        momentum_init=0.85,
        momentum_warmup_steps=300,
        batched_ns=True,
    )
    group = opt.param_groups[0]
    m_step1 = opt.get_effective_momentum(1, group)
    m_step150 = opt.get_effective_momentum(150, group)
    m_step300 = opt.get_effective_momentum(300, group)
    m_step500 = opt.get_effective_momentum(500, group)
    assert abs(m_step1 - 0.85) < 1e-6
    assert 0.89 < m_step150 < 0.91
    assert abs(m_step300 - 0.95) < 1e-6
    assert abs(m_step500 - 0.95) < 1e-6

    p1.grad = torch.randn_like(p1)
    p2.grad = torch.randn_like(p2)
    opt.step()
    assert torch.isfinite(p1).all() and torch.isfinite(p2).all()


def test_sliding_window_and_document_mask_born_attention() -> None:
    """Verify EOT document IDs and sliding-window + document masking in Born attention."""
    idx = torch.tensor(
        [
            [10, 20, 50256, 30, 40, 50256, 50, 60],
            [11, 50256, 22, 33, 44, 55, 66, 77],
        ],
        dtype=torch.long,
    )
    doc_ids = compute_document_ids(idx, eot_token_id=50256)
    expected_doc_ids = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(doc_ids, expected_doc_ids)

    torch.manual_seed(2)
    b, h, t, d_h = 2, 4, 8, 16
    q = l2_normalize(torch.randn(b, h, t, d_h), dim=-1)
    k = l2_normalize(0.85 * q + 0.15 * torch.randn(b, h, t, d_h), dim=-1)
    v = l2_normalize(torch.randn(b, h, t, d_h), dim=-1)

    w_std = SubspaceProjectorBornAttention._born_attention_weights(
        q, k, scale=10.0, window_size=4, doc_ids=doc_ids
    )
    gamma_h = torch.full((h,), 8.0)
    tau_h = torch.zeros((h,))
    w_pur = purified_born_attention_weights(
        q, k, gamma=gamma_h, tau=tau_h, window_size=4, doc_ids=doc_ids
    )

    for weights in (w_std, w_pur):
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        for bi in range(b):
            for qi in range(t):
                for ki in range(t):
                    if ki > qi or (qi - ki) >= 4 or doc_ids[bi, qi] != doc_ids[bi, ki]:
                        assert torch.all(weights[bi, :, qi, ki] == 0.0), (
                            f"Non-zero weight at b={bi}, q={qi}, k={ki}: {weights[bi, :, qi, ki]}"
                        )

    out_fb = flash_born_attention(q, k, v, scale=10.0, window_size=4, doc_ids=doc_ids)
    out_fp = flash_purified_born_attention(
        q, k, v, scale=gamma_h, tau=tau_h, window_size=4, doc_ids=doc_ids
    )
    assert torch.allclose(out_fb, torch.matmul(w_std, v), atol=1e-4)
    assert torch.allclose(out_fp, torch.matmul(w_pur, v), atol=1e-4)

    if torch.cuda.is_available():
        q_cu = q.cuda().detach().requires_grad_(True)
        k_cu = k.cuda().detach().requires_grad_(True)
        v_cu = v.cuda().detach().requires_grad_(True)
        doc_cu = doc_ids.cuda()
        out_fb_cu = flash_born_attention(q_cu, k_cu, v_cu, scale=10.0, window_size=4, doc_ids=doc_cu)
        assert torch.allclose(out_fb_cu.cpu(), out_fb, atol=1e-3)
        out_fb_cu.sum().backward()
        assert q_cu.grad is not None and torch.isfinite(q_cu.grad).all()

        q_cu2 = q.cuda().detach().requires_grad_(True)
        k_cu2 = k.cuda().detach().requires_grad_(True)
        v_cu2 = v.cuda().detach().requires_grad_(True)
        out_fp_cu = flash_purified_born_attention(
            q_cu2, k_cu2, v_cu2, scale=gamma_h.cuda(), tau=tau_h.cuda(), window_size=4, doc_ids=doc_cu
        )
        assert torch.allclose(out_fp_cu.cpu(), out_fp, atol=1e-3)
        out_fp_cu.sum().backward()
        assert q_cu2.grad is not None and torch.isfinite(q_cu2.grad).all()


def test_half_truncated_rope_invariance() -> None:
    """Verify half-truncated RoPE sets upper-half frequencies to zero and preserves position invariance."""
    rope = UnitaryRoPE(head_dim=64, max_seq_len=32, zero_half_freq=True)
    cos, sin = rope.get_cos_sin(16)
    assert cos.shape == (1, 1, 16, 32)
    assert torch.all(cos[..., 16:] == 1.0)
    assert torch.all(sin[..., 16:] == 0.0)
    assert not torch.all(cos[:, :, 1:, :16] == 1.0)

    # Identical token vector repeated across all 16 positions
    v_single = torch.randn(1, 2, 1, 64)
    x_rep = v_single.expand(1, 2, 16, 64).clone()
    x_rope = apply_unitary_rope(x_rep, cos, sin)

    # Upper half of each half-block (indices 16:32 and 48:64) must be strictly position-invariant
    diff_upper_1 = (x_rope[:, :, :, 16:32] - x_rope[:, :, :1, 16:32]).abs().max().item()
    diff_upper_2 = (x_rope[:, :, :, 48:64] - x_rope[:, :, :1, 48:64]).abs().max().item()
    assert diff_upper_1 == 0.0
    assert diff_upper_2 == 0.0
    # Lower half rotates with position
    diff_lower_1 = (x_rope[:, :, 1:, :16] - x_rope[:, :, :1, :16]).abs().max().item()
    assert diff_lower_1 > 1e-2


def test_speedrun_backbone_features_classical_and_quantum() -> None:
    """Verify scalar U-Net skips, fused QKV, value embeddings, skip_attn_layers, half-RoPE, window & doc mask."""
    torch.manual_seed(3)
    idx = torch.randint(0, 256, (2, 16), dtype=torch.long)
    idx[0, 5] = 50256
    targets = torch.randint(0, 256, (2, 16), dtype=torch.long)

    c_model = ClassicalModdedGPT(
        vocab_size=50304,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        hidden_dim=128,
        block_size=32,
        use_unet_interferometry=True,
        unet_skip_mode="scalar",
        use_fused_qkv=True,
        num_value_embeddings=2,
        skip_attn_layers=(2,),
        zero_half_rope=True,
        window_size=8,
        doc_mask_attn=True,
        zero_init_residual_and_head=True,
        true_layer0_v_residual=True,
    )
    assert c_model.blocks[2].attn is None
    assert c_model.unet_lambdas is not None
    assert c_model.value_embeds is not None and len(c_model.value_embeds) == 2
    assert c_model.blocks[0].attn.w_qkv is not None

    _, c_loss = c_model(idx, targets=targets)
    assert c_loss is not None and torch.isfinite(c_loss)
    c_loss.backward()
    assert c_model.unet_lambdas.grad is not None and torch.isfinite(c_model.unet_lambdas.grad).all()
    assert c_model.value_embeds[0].weight.grad is not None
    assert c_model.blocks[0].attn.w_qkv.weight.grad is not None

    q_model = QuantumBornGPT(
        vocab_size=50304,
        embed_dim=64,
        seed_dim=32,
        num_heads=4,
        num_layers=4,
        hidden_dim=128,
        block_size=32,
        use_unet_interferometry=True,
        unet_skip_mode="scalar",
        use_fused_qkv=True,
        num_value_embeddings=2,
        skip_attn_layers=(2,),
        zero_half_rope=True,
        window_size=8,
        doc_mask_attn=True,
        use_ortho_decouple=True,
        use_povm_log_prior=True,
        use_gated_spherical_residual=True,
        use_purified_attn=True,
        true_layer0_v_residual=True,
        use_flash_born=False,
    )
    assert q_model.blocks[2].attn is None
    assert q_model.unet_lambdas is not None
    assert q_model.value_embeds is not None and len(q_model.value_embeds) == 2
    assert q_model.blocks[0].attn.w_qkv is not None

    _, q_loss = q_model(idx, targets=targets)
    assert q_loss is not None and torch.isfinite(q_loss)
    q_loss.backward()
    assert q_model.unet_lambdas.grad is not None and torch.isfinite(q_model.unet_lambdas.grad).all()
    assert q_model.value_embeds[0].weight.grad is not None
    assert q_model.blocks[0].attn.w_qkv.weight.grad is not None


def test_jax_speedrun_primitives_parity() -> None:
    """Verify JAX speedrun primitives match PyTorch counterparts."""
    torch.manual_seed(4)
    g_pt = torch.randn(4, 32, 32, dtype=torch.float32)
    g_jax = jnp.asarray(g_pt.numpy())

    u_pt = zeropower_via_newtonschulz5_batched(g_pt, steps=5).numpy()
    u_jax = np.asarray(zeropower_via_newtonschulz5_batched_jax(g_jax, steps=5))
    assert np.max(np.abs(u_pt - u_jax)) < 1e-4

    rope_pt = UnitaryRoPE(head_dim=64, max_seq_len=32, zero_half_freq=True)
    cos_pt, sin_pt = rope_pt.get_cos_sin(16)
    cos_jax, sin_jax = half_truncated_rope_cos_sin_jax(16, 64)
    assert np.max(np.abs(cos_pt.numpy() - np.asarray(cos_jax))) < 1e-6
    assert np.max(np.abs(sin_pt.numpy() - np.asarray(sin_jax))) < 1e-6

    idx_pt = torch.tensor([[1, 50256, 2, 3, 50256, 4]], dtype=torch.long)
    idx_jax = jnp.asarray(idx_pt.numpy())
    doc_pt = compute_document_ids(idx_pt, eot_token_id=50256).numpy()
    doc_jax = np.asarray(compute_document_ids_jax(idx_jax, eot_token_id=50256))
    assert np.array_equal(doc_pt, doc_jax)
