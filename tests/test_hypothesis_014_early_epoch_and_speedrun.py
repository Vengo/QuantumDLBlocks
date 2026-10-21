"""Unit Tests for Plan 014: Early-Epoch Quantum Convergence & Classical Speedrun Upgrades."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from qdlblocks.layers.attention import SubspaceProjectorBornAttention
from qdlblocks.layers.early_epoch_quantum import (
    compute_born_attention_snr_metrics,
    compute_self_repetition_metrics,
    purified_born_attention_weights,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks_jax.early_epoch_quantum import (
    compute_self_repetition_metrics_jax,
    purified_born_attention_weights_jax,
)


def test_ortho_decouple_eliminates_step0_self_repetition() -> None:
    """Verify that orthogonal highway decoupling eliminates Step-0 self-repetition bias."""
    torch.manual_seed(42)
    vocab_size = 512
    embed_dim = 128
    seed_dim = 64
    num_heads = 4
    num_layers = 4

    x = torch.randint(0, vocab_size, (4, 64))

    torch.manual_seed(101)
    model_base = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=seed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=256,
        block_size=64,
        readout_mode="tied_stiefel",
        use_flash_born=False,
        use_ortho_decouple=False,
    )
    logits_base, _ = model_base(x)
    metrics_base = compute_self_repetition_metrics(logits_base, x)

    torch.manual_seed(101)
    model_ortho = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=seed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=256,
        block_size=64,
        readout_mode="tied_stiefel",
        use_flash_born=False,
        use_ortho_decouple=True,
    )
    logits_ortho, _ = model_ortho(x)
    metrics_ortho = compute_self_repetition_metrics(logits_ortho, x)

    assert metrics_base["mean_self_repeat_prob"] > 0.15, (
        f"Expected baseline Step-0 mean_self_repeat_prob > 0.15, got {metrics_base['mean_self_repeat_prob']:.6f}"
    )
    assert metrics_ortho["mean_self_repeat_prob"] < 0.01, (
        f"Expected ortho_decouple Step-0 mean_self_repeat_prob < 0.01, got {metrics_ortho['mean_self_repeat_prob']:.6f}"
    )


def test_purified_born_attention_non_zero_gamma_and_tau_gradients() -> None:
    """Verify non-zero gamma_h / tau_h gradients and monotonic entropy reduction in Purified Born Attention."""
    torch.manual_seed(42)
    b, t, d = 4, 64, 128
    h = 4
    x = F.normalize(torch.randn(b, t, d), p=2, dim=-1)
    target = torch.randn(b, t, d)

    # 1. Standard Born Attention: qk_gain_param.grad is identically 0 due to scale-invariant L1 norm
    torch.manual_seed(7)
    attn_std = SubspaceProjectorBornAttention(
        embed_dim=d,
        num_heads=h,
        block_size=t,
        use_flash_born=False,
        use_purified_attn=False,
    )
    out_std = attn_std(x)
    loss_std = (out_std * target).sum(dim=-1).mean()
    loss_std.backward()
    assert attn_std.qk_gain_param is not None and attn_std.qk_gain_param.grad is not None
    std_gamma_grad = float(attn_std.qk_gain_param.grad.norm().item())
    assert std_gamma_grad < 1e-7, f"Expected standard Born qk_gain_param grad < 1e-7, got {std_gamma_grad}"

    # 2. Purified Born Attention: both qk_gain_param.grad and coherence_tau_raw.grad > 1e-4
    torch.manual_seed(7)
    attn_pur = SubspaceProjectorBornAttention(
        embed_dim=d,
        num_heads=h,
        block_size=t,
        use_flash_born=False,
        use_purified_attn=True,
    )
    out_pur = attn_pur(x)
    loss_pur = (out_pur * target).sum(dim=-1).mean()
    loss_pur.backward()
    assert attn_pur.qk_gain_param is not None and attn_pur.qk_gain_param.grad is not None
    assert attn_pur.coherence_tau_raw is not None and attn_pur.coherence_tau_raw.grad is not None
    pur_gamma_grad = float(attn_pur.qk_gain_param.grad.norm().item())
    pur_tau_grad = float(attn_pur.coherence_tau_raw.grad.norm().item())
    assert pur_gamma_grad > 1e-4, f"Expected purified qk_gain_param grad > 1e-4, got {pur_gamma_grad}"
    assert pur_tau_grad > 1e-4, f"Expected purified coherence_tau_raw grad > 1e-4, got {pur_tau_grad}"

    # 3. Monotonic Shannon entropy reduction across gamma in [2.0, 4.0, 8.0, 12.0]
    d_h = d // h
    q = torch.zeros(b, h, t, d_h)
    k = torch.zeros(b, h, t, d_h)
    # Coherent key j=0 along e_0; sub-threshold distractor keys j>=1 with overlap 0.04 <= tau=0.0625
    k[:, :, 0, 0] = 1.0
    k[:, :, 1:, 1] = 0.05
    k[:, :, 1:, 3:] = F.normalize(torch.randn(b, h, t - 1, d_h - 3), p=2, dim=-1) * ((1.0 - 0.05**2) ** 0.5)
    q[:, :, :, 0] = 0.15
    q[:, :, :, 1] = 0.80
    q[:, :, :, 2] = (1.0 - 0.15**2 - 0.80**2) ** 0.5

    entropies: list[float] = []
    for gamma_val in [2.0, 4.0, 8.0, 12.0]:
        w = purified_born_attention_weights(q, k, gamma=gamma_val, tau=0.0625, eps_base=0.05, causal=True)
        # Evaluate row-wise Shannon entropy on the second half of the sequence (queries i >= t//2)
        w_half = w[:, :, t // 2 :, :].clamp_min(1e-12)
        ent = float(-(w_half * w_half.log()).sum(dim=-1).mean().item())
        entropies.append(ent)

    for idx in range(len(entropies) - 1):
        assert entropies[idx] > entropies[idx + 1], f"Entropy not strictly monotonic: {entropies}"
    entropy_drop = entropies[0] - entropies[-1]
    assert entropy_drop >= 0.80, f"Expected entropy(2.0) - entropy(12.0) >= 0.80 nats, got {entropy_drop:.4f}"


def test_purified_born_attention_snr_gain_at_t512() -> None:
    """Verify Purified Born Attention signal-to-noise ratio at T=512, d_h=64."""
    torch.manual_seed(42)
    b, h, t, d_h = 2, 4, 512, 64
    q = F.normalize(torch.randn(b, h, t, d_h), p=2, dim=-1)
    k = F.normalize(torch.randn(b, h, t, d_h), p=2, dim=-1)

    metrics = compute_born_attention_snr_metrics(
        q,
        k,
        gamma_h=8.0,
        tau_h=0.0625,
        eps_base=0.05,
        target_overlap=0.85,
    )
    assert metrics["std_target_weight"] < 0.18, f"Expected std_target_weight < 0.18, got {metrics['std_target_weight']}"
    assert metrics["purified_target_weight"] >= 0.65, (
        f"Expected purified_target_weight >= 0.65, got {metrics['purified_target_weight']}"
    )
    assert metrics["snr_gain_ratio"] >= 3.5, f"Expected snr_gain_ratio >= 3.5, got {metrics['snr_gain_ratio']}"


def test_gated_spherical_residual_and_povm_log_prior_forward_backward() -> None:
    """Verify forward/backward gradient flow across all 4 early-epoch quantum mechanisms."""
    torch.manual_seed(42)
    vocab_size = 256
    x = torch.randint(0, vocab_size, (2, 32))
    y = torch.randint(0, vocab_size, (2, 32))

    for readout_mode in ("tied_stiefel", "asymmetric"):
        model = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=64,
            seed_dim=32,
            num_heads=4,
            num_layers=2,
            hidden_dim=128,
            block_size=32,
            readout_mode=readout_mode,  # type: ignore[arg-type]
            use_flash_born=False,
            use_ortho_decouple=True,
            use_povm_log_prior=True,
            use_gated_spherical_residual=True,
            use_purified_attn=True,
        )
        logits, loss = model(x, targets=y)
        assert logits.shape == (2, 32, vocab_size)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()

        assert model.povm_log_prior is not None and model.povm_log_prior.grad is not None
        assert float(model.povm_log_prior.grad.abs().max().item()) > 0.0

        for block in model.blocks:
            assert block.attn_res_gate is not None and block.attn_res_gate.grad is not None
            assert block.mlp_res_gate is not None and block.mlp_res_gate.grad is not None
            assert block.attn.coherence_tau_raw is not None and block.attn.coherence_tau_raw.grad is not None
            assert float(block.attn_res_gate.grad.abs().item()) > 0.0
            assert float(block.mlp_res_gate.grad.abs().item()) > 0.0
            assert float(block.attn.coherence_tau_raw.grad.abs().max().item()) > 0.0


def test_classical_zero_init_and_jax_parity() -> None:
    """Verify ClassicalModdedGPT zero-init uniform CE at Step 0 and PyTorch-vs-JAX parity."""
    torch.manual_seed(42)
    vocab_size = 512
    x = torch.randint(0, vocab_size, (4, 32))
    y = torch.randint(0, vocab_size, (4, 32))

    model = ClassicalModdedGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        hidden_dim=256,
        block_size=32,
        tie_embeddings=True,
        zero_init_residual_and_head=True,
    )
    assert model.tie_embeddings is False
    logits, loss = model(x, targets=y)
    assert loss is not None
    expected_ce = math.log(vocab_size)
    assert abs(float(loss.item()) - expected_ce) < 1e-5, (
        f"Expected exact ln({vocab_size})={expected_ce:.6f}, got {loss.item():.6f}"
    )
    assert float(logits.abs().max().item()) == 0.0

    # PyTorch-vs-JAX parity for purified_born_attention_weights_jax and compute_self_repetition_metrics_jax
    q_pt = F.normalize(torch.randn(2, 4, 16, 32), p=2, dim=-1)
    k_pt = F.normalize(torch.randn(2, 4, 16, 32), p=2, dim=-1)
    w_pt = purified_born_attention_weights(q_pt, k_pt, gamma_h=8.0, tau_h=0.0625, eps_base=0.05, causal=True)

    q_jax = jnp.asarray(q_pt.numpy())
    k_jax = jnp.asarray(k_pt.numpy())
    w_jax = purified_born_attention_weights_jax(q_jax, k_jax, gamma_h=8.0, tau_h=0.0625, eps_base=0.05, causal=True)
    max_diff_w = float(np.max(np.abs(w_pt.numpy() - np.asarray(w_jax))))
    assert max_diff_w < 1e-5, f"Max diff in purified_born_attention_weights_jax: {max_diff_w}"

    rand_logits_pt = torch.randn(2, 16, vocab_size)
    ids_pt = torch.randint(0, vocab_size, (2, 16))
    rep_pt = compute_self_repetition_metrics(rand_logits_pt, ids_pt)
    rep_jax = compute_self_repetition_metrics_jax(
        jnp.asarray(rand_logits_pt.numpy()),
        jnp.asarray(ids_pt.numpy()),
    )
    for key, val_pt in rep_pt.items():
        val_jax = float(rep_jax[key])
        assert abs(val_pt - val_jax) < 1e-5, f"Mismatch on {key}: pt={val_pt}, jax={val_jax}"
