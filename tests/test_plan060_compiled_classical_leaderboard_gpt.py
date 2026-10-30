"""Unit tests for Wave 45 (`EXPERIMENT_PLAN_060.md`): `CompiledLeaderboardClassicalGPT`."""

from __future__ import annotations

import math

import torch

from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    DEFAULT_LEADERBOARD_VOCAB_SIZE,
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
    CompiledLeaderboardOptimizerConfig,
    NativeLeaderboardRotary,
    apply_native_leaderboard_rotary,
    build_compiled_leaderboard_optimizers,
    evaluate_compiled_leaderboard_validation,
)


def test_native_leaderboard_rotary_dtype_and_zero_half_invariance() -> None:
    """Verify NativeLeaderboardRotary preserves bfloat16/float32 without casting and leaves upper-quarter freqs invariant."""
    rope = NativeLeaderboardRotary(head_dim=128, max_seq_len=64, base=1024.0)
    for dtype in (torch.float32, torch.bfloat16):
        cos_4d, sin_4d = rope.get_cos_sin(16, dtype=dtype)
        assert cos_4d.dtype == dtype
        assert sin_4d.dtype == dtype
        assert cos_4d.shape == (1, 16, 1, 64)
        assert torch.allclose(cos_4d[..., 32:], torch.ones_like(cos_4d[..., 32:]))
        assert torch.allclose(sin_4d[..., 32:], torch.zeros_like(sin_4d[..., 32:]))

        x = torch.randn(2, 16, 6, 128, dtype=dtype)
        y = apply_native_leaderboard_rotary(x, cos_4d, sin_4d)
        assert y.dtype == dtype
        assert y.shape == x.shape
        assert torch.allclose(y[..., 32:64], x[..., 32:64], atol=1e-5)
        assert torch.allclose(y[..., 96:128], x[..., 96:128], atol=1e-5)


def test_step0_exact_ln50304_and_single_pass_chunked_ce_grad_parity() -> None:
    """Verify Step-0 exact zero logits, ln(50304) loss, and bit-accurate chunked vs full softcap CE gradients."""
    torch.manual_seed(123)
    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=2048,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        hidden_dim=128,
        max_seq_len=32,
        num_value_embeddings=1,
        skip_attn_layers=(2,),
        ce_chunk_tokens=16,
        bfloat16_weights=False,
    )
    model_chunk = CompiledLeaderboardClassicalGPT(cfg)
    model_full = CompiledLeaderboardClassicalGPT(cfg)
    model_full.load_state_dict(model_chunk.state_dict())

    idx = torch.randint(0, 2000, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 2000, (2, 16), dtype=torch.long)

    # Step 0 check
    _, loss0_chunk = model_chunk(idx, targets, return_logits=False)
    logits0_full, loss0_full = model_full(idx, targets, return_logits=True)
    assert loss0_chunk is not None and loss0_full is not None and logits0_full is not None
    assert float(logits0_full.abs().max().item()) == 0.0
    assert abs(float(loss0_chunk.item()) - math.log(2048.0)) < 1e-5
    assert abs(float(loss0_chunk.item()) - float(loss0_full.item())) < 1e-6

    # Non-zero lm_head gradient parity check
    with torch.no_grad():
        w_rand = torch.randn_like(model_chunk.lm_head.weight) * 0.04
        model_chunk.lm_head.weight.copy_(w_rand)
        model_full.lm_head.weight.copy_(w_rand)
        for b_c, b_f in zip(model_chunk.blocks, model_full.blocks):
            if b_c.out_proj is not None and b_f.out_proj is not None:
                w_o = torch.randn_like(b_c.out_proj.weight) * 0.04
                b_c.out_proj.weight.copy_(w_o)
                b_f.out_proj.weight.copy_(w_o)

    _, loss_c = model_chunk(idx, targets, return_logits=False)
    _, loss_f = model_full(idx, targets, return_logits=True)
    assert loss_c is not None and loss_f is not None
    loss_c.backward()
    loss_f.backward()

    assert abs(float(loss_c.item()) - float(loss_f.item())) < 1e-5
    for (n_c, p_c), (n_f, p_f) in zip(model_chunk.named_parameters(), model_full.named_parameters()):
        assert n_c == n_f
        assert p_c.grad is not None and p_f.grad is not None
        assert torch.allclose(p_c.grad, p_f.grad, atol=1e-4, rtol=1e-3), f"Grad mismatch on {n_c}"


def test_optimizer_partition_and_evaluator() -> None:
    """Verify 100% parameter coverage in build_compiled_leaderboard_optimizers and fast validation."""
    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=512,
        embed_dim=64,
        num_heads=4,
        num_layers=6,
        hidden_dim=128,
        max_seq_len=32,
        num_value_embeddings=2,
        skip_attn_layers=(3,),
        bfloat16_weights=False,
    )
    model = CompiledLeaderboardClassicalGPT(cfg)
    muon_opt, adam_opt = build_compiled_leaderboard_optimizers(
        model,
        CompiledLeaderboardOptimizerConfig(muon_lr=0.04, muon_warmup_steps=10),
    )

    seen_ids: set[int] = set()
    for grp in muon_opt.param_groups:
        for p in grp["params"]:
            seen_ids.add(id(p))
    for grp in adam_opt.param_groups:
        for p in grp["params"]:
            seen_ids.add(id(p))

    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert seen_ids == all_ids

    val_stream = torch.randint(0, 500, (256,), dtype=torch.long)
    metrics = evaluate_compiled_leaderboard_validation(
        model,
        val_stream,
        val_tokens_budget=128,
        batch_size=2,
        seq_len=16,
        valid_vocab_size=500,
        chunk_tokens=64,
    )
    assert abs(metrics["val_ce"] - math.log(500.0)) < 1e-4
    assert metrics["eval_tokens"] == 128.0


def test_cuda_triton_inplace_softcap_ce_parity() -> None:
    """Verify the in-place CUDA Triton softcap CE kernel matches full float32 softcap CE loss and gradients."""
    if not torch.cuda.is_available():
        return
    torch.manual_seed(999)
    device = torch.device("cuda")
    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=4096,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        max_seq_len=32,
        num_value_embeddings=1,
        skip_attn_layers=(),
        ce_chunk_tokens=16,
        bfloat16_weights=False,
    )
    m_triton = CompiledLeaderboardClassicalGPT(cfg).to(device)
    m_ref = CompiledLeaderboardClassicalGPT(cfg).to(device)
    with torch.no_grad():
        m_triton.lm_head.weight.normal_(0.0, 0.05)
        for blk in m_triton.blocks:
            if blk.out_proj is not None:
                blk.out_proj.weight.normal_(0.0, 0.05)
    m_ref.load_state_dict(m_triton.state_dict())

    idx = torch.randint(0, 4000, (2, 16), device=device, dtype=torch.long)
    targets = torch.randint(0, 4000, (2, 16), device=device, dtype=torch.long)

    _, loss_t = m_triton(idx, targets, return_logits=False)
    _, loss_r = m_ref(idx, targets, return_logits=True)
    assert loss_t is not None and loss_r is not None
    loss_t.backward()
    loss_r.backward()

    assert abs(float(loss_t.item()) - float(loss_r.item())) < 2e-2
    assert m_triton.lm_head.weight.grad is not None and m_ref.lm_head.weight.grad is not None
    cos_sim = torch.nn.functional.cosine_similarity(
        m_triton.lm_head.weight.grad.flatten().float(),
        m_ref.lm_head.weight.grad.flatten().float(),
        dim=0,
    )
    assert float(cos_sim.item()) > 0.999


def test_leaderboard_yarn_and_gates_gradient_flow() -> None:
    """Verify YaRN frequency/temperature scaling, sliding-window buffer updates, and smear/attn/ve/XSA gate gradients."""
    torch.manual_seed(2026)
    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=1024,
        embed_dim=64,
        num_heads=4,
        num_layers=6,
        hidden_dim=128,
        max_seq_len=128,
        num_value_embeddings=2,
        skip_attn_layers=(3,),
        interior_value_mode="unit_v",
        initial_ws_blocks=3,
        bfloat16_weights=False,
    )
    model = CompiledLeaderboardClassicalGPT(cfg)
    assert model.current_ws_blocks == 3
    assert int(model.long_window_buf.item()) == 384
    assert int(model.short_window_buf.item()) == 128

    model.set_sliding_window_blocks(6, apply_yarn=True)
    assert model.current_ws_blocks == 6
    assert int(model.long_window_buf.item()) == 768
    assert int(model.short_window_buf.item()) == 384
    expected_q_scale = 1.0 + 0.2 * math.log(6.0 / 3.0)
    assert abs(float(model.rope.yarn_q_scale.item()) - expected_q_scale) < 1e-5

    model.set_sliding_window_blocks(3, apply_yarn=True)
    assert abs(float(model.rope.yarn_q_scale.item()) - 1.0) < 1e-6

    with torch.no_grad():
        model.lm_head.weight.normal_(0.0, 0.03)
        if model.smear_lambda is not None:
            model.smear_lambda.fill_(0.5)
        for blk in model.blocks:
            if blk.out_proj is not None:
                blk.out_proj.weight.normal_(0.0, 0.03)

    idx = torch.randint(0, 1000, (2, 32), dtype=torch.long)
    targets = torch.randint(0, 1000, (2, 32), dtype=torch.long)
    _, loss = model(idx, targets, return_logits=False)
    assert loss is not None
    loss.backward()

    assert model.smear_gate is not None and model.smear_gate.weight.grad is not None
    assert float(model.smear_gate.weight.grad.abs().max().item()) > 0.0
    assert model.blocks[0].attn_gate is not None and model.blocks[0].attn_gate.weight.grad is not None
    assert model.blocks[0].ve_gate is not None and model.blocks[0].ve_gate.weight.grad is not None
    assert model.blocks[0].xsa_alpha is not None and model.blocks[0].xsa_alpha.grad is not None

    with torch.no_grad():
        traj = model.extract_layer_trajectories(idx)
        feat = model.forward_features(idx)
        assert len(traj["layer_inputs"]) == cfg.num_layers
        assert len(traj["layer_outputs"]) == cfg.num_layers
        assert torch.allclose(traj["final_norm_x"], feat, atol=1e-6)



