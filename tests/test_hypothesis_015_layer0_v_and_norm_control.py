"""Unit Tests for Plan 015: True Cross-Layer First-Layer Value Residual (V^(0)),
Cubic ReLU^2 Norm Control, and Fused Triton Purified Born Attention.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.kernels.flash_born import flash_purified_born_attention
from qdlblocks.layers.attention import SubspaceProjectorBornAttention
from qdlblocks.layers.early_epoch_quantum import purified_born_attention_weights
from qdlblocks.layers.isometric import UnitaryRoPE
from qdlblocks.layers.value_residual_and_norm_control import (
    compute_cross_layer_v_orthogonality_metrics,
    compute_mlp_attn_homogeneity_metrics,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedBlock, ClassicalModdedGPT
from qdlblocks_jax.value_residual_and_norm_control import (
    compute_cross_layer_v_orthogonality_metrics_jax,
    compute_mlp_attn_homogeneity_metrics_jax,
)


def _count_v_proj_calls_and_orth_metrics_born(
    true_layer0_v_residual: bool,
    num_layers: int = 6,
    embed_dim: int = 128,
    num_heads: int = 4,
    batch_size: int = 4,
    seq_len: int = 64,
) -> tuple[int, list[float]]:
    torch.manual_seed(42)
    model = QuantumBornGPT(
        vocab_size=256,
        embed_dim=embed_dim,
        seed_dim=64,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=256,
        block_size=seq_len,
        readout_mode="tied_stiefel",
        value_residual_mixing=True,
        input_highway_mixing=True,
        use_flash_born=False,
        true_layer0_v_residual=true_layer0_v_residual,
    )
    v_proj_calls = 0

    def _hook(_mod: nn.Module, _inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        nonlocal v_proj_calls
        v_proj_calls += 1

    handles = [blk.attn.v_proj.register_forward_hook(_hook) for blk in model.blocks]
    torch.manual_seed(99)
    idx = torch.randint(0, 256, (batch_size, seq_len))
    _ = model(idx)
    for h in handles:
        h.remove()

    with torch.no_grad():
        psi_0 = model._last_psi_0
        assert psi_0 is not None
        orth_ratios: list[float] = []
        for l_idx in range(1, num_layers):
            blk = model.blocks[l_idx]
            assert blk.attn.last_v_mixed is not None and blk.attn.last_psi_in is not None
            metrics = compute_cross_layer_v_orthogonality_metrics(
                v_mixed=blk.attn.last_v_mixed,
                x_in=blk.attn.last_psi_in,
                x_0=psi_0,
                w_v_layer=blk.attn.v_proj.weight,
            )
            orth_ratios.append(metrics["orth_residual_ratio"])

    return v_proj_calls, orth_ratios


def _count_v_proj_calls_and_orth_metrics_classical(
    true_layer0_v_residual: bool,
    num_layers: int = 6,
    embed_dim: int = 128,
    num_heads: int = 4,
    batch_size: int = 4,
    seq_len: int = 64,
) -> tuple[int, list[float]]:
    torch.manual_seed(42)
    model = ClassicalModdedGPT(
        vocab_size=256,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=256,
        block_size=seq_len,
        value_residual_mixing=True,
        input_highway_mixing=True,
        true_layer0_v_residual=true_layer0_v_residual,
    )
    # Give each layer distinct random orthogonal v_proj weights so Layer 0 and Layer l >= 1 differ
    for l_idx, blk in enumerate(model.blocks):
        torch.manual_seed(1000 + l_idx)
        nn.init.orthogonal_(blk.v_proj.weight)

    v_proj_calls = 0

    def _hook(_mod: nn.Module, _inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        nonlocal v_proj_calls
        v_proj_calls += 1

    handles = [blk.v_proj.register_forward_hook(_hook) for blk in model.blocks]
    torch.manual_seed(99)
    idx = torch.randint(0, 256, (batch_size, seq_len))
    _ = model(idx)
    for h in handles:
        h.remove()

    with torch.no_grad():
        h_0 = model._last_x_0
        assert h_0 is not None
        orth_ratios: list[float] = []
        for l_idx in range(1, num_layers):
            blk = model.blocks[l_idx]
            assert blk.last_v_mixed is not None and blk.last_h_in is not None
            metrics = compute_cross_layer_v_orthogonality_metrics(
                v_mixed=blk.last_v_mixed,
                x_in=blk.last_h_in,
                x_0=h_0,
                w_v_layer=blk.v_proj.weight,
            )
            orth_ratios.append(metrics["orth_residual_ratio"])

    return v_proj_calls, orth_ratios


def test_cross_layer_v_orthogonality_and_gemm_reduction() -> None:
    """Verify H1: cross-layer V^(0) orthogonality residual ratio and v_proj GEMM call savings."""
    calls_false_q, orth_false_q = _count_v_proj_calls_and_orth_metrics_born(true_layer0_v_residual=False)
    calls_true_q, orth_true_q = _count_v_proj_calls_and_orth_metrics_born(true_layer0_v_residual=True)

    assert calls_false_q == 12, f"Expected 2*L=12 v_proj calls when true_layer0_v_residual=False, got {calls_false_q}"
    assert calls_true_q == 6, f"Expected L=6 v_proj calls when true_layer0_v_residual=True, got {calls_true_q}"
    assert calls_false_q - calls_true_q == 6
    assert max(orth_false_q) < 1e-5, f"Expected max orth_residual_ratio < 1e-5, got {max(orth_false_q):.3e}"
    assert float(np.mean(orth_true_q)) >= 0.15, (
        f"Expected mean orth_residual_ratio >= 0.15 for QuantumBornGPT, got {float(np.mean(orth_true_q)):.4f}"
    )

    calls_false_c, orth_false_c = _count_v_proj_calls_and_orth_metrics_classical(true_layer0_v_residual=False)
    calls_true_c, orth_true_c = _count_v_proj_calls_and_orth_metrics_classical(true_layer0_v_residual=True)

    assert calls_false_c == 12, f"Expected 2*L=12 v_proj calls when true_layer0_v_residual=False, got {calls_false_c}"
    assert calls_true_c == 6, f"Expected L=6 v_proj calls when true_layer0_v_residual=True, got {calls_true_c}"
    assert calls_false_c - calls_true_c == 6
    assert max(orth_false_c) < 1e-5, f"Expected max orth_residual_ratio < 1e-5, got {max(orth_false_c):.3e}"
    assert float(np.mean(orth_true_c)) >= 0.15, (
        f"Expected mean orth_residual_ratio >= 0.15 for ClassicalModdedGPT, got {float(np.mean(orth_true_c)):.4f}"
    )


def test_classical_relu2_cubic_vs_attn_quadratic_homogeneity() -> None:
    """Verify H2: exact c^2 attention scaling vs. c^3 ReLU^2 MLP scaling under weight scale c=4.0."""
    torch.manual_seed(42)
    b, t, d, h = 4, 64, 128, 4
    block = ClassicalModdedBlock(
        embed_dim=d,
        num_heads=h,
        hidden_dim=256,
        value_residual_mixing=True,
        input_highway_mixing=True,
    )
    # Give non-zero weights to out_proj and fc2 so norms are non-zero
    nn.init.orthogonal_(block.out_proj.weight)
    nn.init.orthogonal_(block.fc2.weight)

    x = torch.randn(b, t, d)
    rope = UnitaryRoPE(head_dim=d // h, max_seq_len=t)
    cos, sin = rope.get_cos_sin(t)
    c = 4.0
    metrics = compute_mlp_attn_homogeneity_metrics(block, x, cos, sin, scale_factor=c)

    assert abs(metrics["attn_scale_ratio"] - (c**2)) / (c**2) < 1e-4, (
        f"Expected attn_scale_ratio={c**2}, got {metrics['attn_scale_ratio']:.6f}"
    )
    assert abs(metrics["mlp_scale_ratio"] - (c**3)) / (c**3) < 1e-4, (
        f"Expected mlp_scale_ratio={c**3}, got {metrics['mlp_scale_ratio']:.6f}"
    )
    assert abs(metrics["dominance_growth_ratio"] - c) / c < 1e-4, (
        f"Expected dominance_growth_ratio={c}, got {metrics['dominance_growth_ratio']:.6f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton flash_purified_born_attention")
def test_flash_purified_born_attention_parity_and_gradients() -> None:
    """Verify H3: CUDA bfloat16 forward/backward parity between flash_purified_born_attention and PyTorch ref."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    b, h, t, d_h = 4, 8, 128, 64

    q0 = F.normalize(torch.randn(b, h, t, d_h, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    k0 = F.normalize(torch.randn(b, h, t, d_h, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    v0 = F.normalize(torch.randn(b, h, t, d_h, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    gamma0 = torch.linspace(4.0, 11.0, h, device=device, dtype=torch.float32).view(1, h, 1, 1)
    tau0 = torch.linspace(0.03, 0.09, h, device=device, dtype=torch.float32).view(1, h, 1, 1)
    grad_out = torch.randn(b, h, t, d_h, device=device, dtype=torch.bfloat16)

    # Triton path
    q_tri = q0.clone().detach().requires_grad_(True)
    k_tri = k0.clone().detach().requires_grad_(True)
    v_tri = v0.clone().detach().requires_grad_(True)
    gamma_tri = gamma0.clone().detach().requires_grad_(True)
    tau_tri = tau0.clone().detach().requires_grad_(True)

    out_tri = flash_purified_born_attention(
        q_tri, k_tri, v_tri, gamma_h=gamma_tri, tau_h=tau_tri, eps_base=0.05, causal=True
    )
    (out_tri * grad_out).sum().backward()

    # Reference path
    q_ref = q0.clone().detach().requires_grad_(True)
    k_ref = k0.clone().detach().requires_grad_(True)
    v_ref = v0.clone().detach().requires_grad_(True)
    gamma_ref = gamma0.clone().detach().requires_grad_(True)
    tau_ref = tau0.clone().detach().requires_grad_(True)

    weights_ref = purified_born_attention_weights(
        q_ref, k_ref, gamma_h=gamma_ref, tau_h=tau_ref, eps_base=0.05, causal=True
    )
    out_ref = torch.matmul(weights_ref, v_ref)
    (out_ref * grad_out).sum().backward()

    max_abs_diff = float((out_tri.float() - out_ref.float()).abs().max().item())
    assert max_abs_diff < 5e-3, f"Expected max_abs_diff < 5e-3, got {max_abs_diff:.6f}"

    def _cos(a: torch.Tensor | None, b_t: torch.Tensor | None) -> float:
        assert a is not None and b_t is not None
        af = a.detach().float().reshape(-1)
        bf = b_t.detach().float().reshape(-1)
        return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())

    for name, g_t, g_r in [
        ("dQ", q_tri.grad, q_ref.grad),
        ("dK", k_tri.grad, k_ref.grad),
        ("dV", v_tri.grad, v_ref.grad),
        ("d_gamma", gamma_tri.grad, gamma_ref.grad),
        ("d_tau", tau_tri.grad, tau_ref.grad),
    ]:
        cos_sim = _cos(g_t, g_r)
        assert cos_sim >= 0.995, f"Expected {name} cosine similarity >= 0.995, got {cos_sim:.6f}"


def test_jax_torch_parity_015() -> None:
    """Verify numerical parity between PyTorch and JAX diagnostic functions in value_residual_and_norm_control."""
    torch.manual_seed(42)
    b, h, t, d_h = 2, 4, 32, 32
    d = h * d_h

    x_in = torch.randn(b, t, d)
    x_0 = torch.randn(b, t, d)
    w_v_l = torch.randn(d, d)
    v_mixed = torch.randn(b, h, t, d_h)

    pt_orth = compute_cross_layer_v_orthogonality_metrics(v_mixed, x_in, x_0, w_v_l)
    jax_orth = compute_cross_layer_v_orthogonality_metrics_jax(
        jnp.asarray(v_mixed.numpy()),
        jnp.asarray(x_in.numpy()),
        jnp.asarray(x_0.numpy()),
        jnp.asarray(w_v_l.numpy()),
    )
    for k in ("orth_residual_ratio", "parallel_energy_ratio", "total_v_rms"):
        diff = abs(pt_orth[k] - jax_orth[k])
        assert diff < 1e-5, f"Mismatch on {k}: pt={pt_orth[k]}, jax={jax_orth[k]} (diff={diff})"

    block = ClassicalModdedBlock(
        embed_dim=d,
        num_heads=h,
        hidden_dim=256,
        value_residual_mixing=True,
        input_highway_mixing=True,
    )
    nn.init.orthogonal_(block.out_proj.weight)
    nn.init.orthogonal_(block.fc2.weight)
    rope = UnitaryRoPE(head_dim=d_h, max_seq_len=t)
    cos, sin = rope.get_cos_sin(t)

    pt_hom = compute_mlp_attn_homogeneity_metrics(block, x_in, cos, sin, scale_factor=4.0)
    jax_params = {
        "w_q": jnp.asarray(block.q_proj.weight.detach().numpy()),
        "w_k": jnp.asarray(block.k_proj.weight.detach().numpy()),
        "w_v": jnp.asarray(block.v_proj.weight.detach().numpy()),
        "w_o": jnp.asarray(block.out_proj.weight.detach().numpy()),
        "w_1": jnp.asarray(block.fc1.weight.detach().numpy()),
        "w_2": jnp.asarray(block.fc2.weight.detach().numpy()),
        "num_heads": h,
    }
    jax_hom = compute_mlp_attn_homogeneity_metrics_jax(
        jax_params,
        jnp.asarray(x_in.numpy()),
        jnp.asarray(cos.numpy()),
        jnp.asarray(sin.numpy()),
        scale_factor=4.0,
    )
    for k in ("attn_scale_ratio", "mlp_scale_ratio", "dominance_growth_ratio"):
        rel_diff = abs(pt_hom[k] - jax_hom[k]) / max(abs(pt_hom[k]), 1e-8)
        assert rel_diff < 1e-4, f"Mismatch on {k}: pt={pt_hom[k]}, jax={jax_hom[k]} (rel_diff={rel_diff})"
