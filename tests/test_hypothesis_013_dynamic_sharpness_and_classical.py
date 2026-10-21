"""Unit tests for EXPERIMENT_PLAN_013: Contextual Bloch-Purity Dynamic POVM Sharpness & Classical Modded-NanoGPT Baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks import (
    ClassicalModdedBlock,
    ClassicalModdedGPT,
    ContextualBlochPurityGate,
    QuantumBornGPT,
    compute_logit_norm_variation_metrics,
    l2_normalize,
)
from qdlblocks_jax import (
    compute_logit_norm_variation_metrics_jax,
    contextual_bloch_purity_beta_jax,
)


def test_constant_logit_norm_vs_dynamic_purity_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Constant-Logit-Norm theorem vs. ContextualBlochPurityGate identity init & dynamic modulation."""
    torch.manual_seed(42)
    B, T, d, D_out, V = 8, 32, 64, 32, 512
    base_beta = 30.0

    # 1. Construct isotropic unit-norm state kets and tight-frame vocabulary POVM vectors
    psi = l2_normalize(torch.randn(B, T, d), dim=-1)
    phi = l2_normalize(torch.randn(B, T, D_out), dim=-1)
    q_frame, _ = torch.linalg.qr(torch.randn(V, D_out))
    m_frame = l2_normalize(q_frame, dim=-1)

    cos_logits = F.linear(phi, m_frame)
    static_logits = cos_logits * base_beta
    static_metrics = compute_logit_norm_variation_metrics(static_logits)
    assert static_metrics["logit_norm_cv"] < 0.08, (
        f"Expected static global-beta spherical readout CV(||ell||_2) < 0.08, got {static_metrics['logit_norm_cv']:.4f}"
    )

    # 2. Verify exact step-0 identity initialization of ContextualBlochPurityGate
    gate = ContextualBlochPurityGate(
        embed_dim=d,
        hidden_dim=32,
        r_min=0.25,
        r_max=4.0,
        init_bias=1.0,
        max_beta=160.0,
    )
    r0 = gate.purity_radius(psi)
    beta0 = gate(psi, base_beta)
    assert r0.shape == (B, T, 1)
    assert beta0.shape == (B, T, 1)
    assert torch.allclose(r0, torch.ones_like(r0), atol=1e-6)
    assert torch.allclose(beta0, torch.full_like(beta0, base_beta), atol=1e-5)

    scaled_0, _ = gate.scale_logits(psi, cos_logits, base_beta)
    assert torch.allclose(scaled_0, static_logits, atol=1e-5)

    # 3. After gradient updates (or trained gate weights), CV(||ell||_2) > 0.10 and beta is clamped in [beta_0*r_min, min(beta_0*r_max, max_beta)]
    with torch.no_grad():
        nn.init.normal_(gate.fc2.weight, mean=0.0, std=2.5)
    scaled_dyn, beta_dyn = gate.scale_logits(psi, cos_logits, base_beta)
    dyn_metrics = compute_logit_norm_variation_metrics(scaled_dyn)
    assert dyn_metrics["logit_norm_cv"] > 0.10, (
        f"Expected dynamic purity gate CV(||ell||_2) > 0.10, got {dyn_metrics['logit_norm_cv']:.4f}"
    )
    min_bound = max(1.0, base_beta * gate.r_min)
    max_bound = min(gate.max_beta, base_beta * gate.r_max)
    assert float(beta_dyn.min().item()) >= min_bound - 1e-5
    assert float(beta_dyn.max().item()) <= max_bound + 1e-5

    # 4. Verify CLI flags --architecture and --dynamic-povm-beta in examples/11_fineweb_100m_multiepoch_gpu_tpu.py
    script_path = Path(__file__).resolve().parents[1] / "examples" / "11_fineweb_100m_multiepoch_gpu_tpu.py"
    spec = importlib.util.spec_from_file_location("ex11_plan013", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--architecture",
            "classical_modded",
            "--dynamic-povm-beta",
        ],
    )
    args = mod.parse_args()
    assert args.architecture == "classical_modded"
    assert args.dynamic_povm_beta is True


def test_quantum_born_gpt_dynamic_povm_beta_forward_backward() -> None:
    """Verify QuantumBornGPT(..., use_dynamic_povm_beta=True) across tied_stiefel and asymmetric modes."""
    V, d, d_seed = 128, 64, 24
    idx = torch.randint(0, V, (4, 16), dtype=torch.long)
    targets = torch.randint(0, V, (4, 16), dtype=torch.long)

    for mode in ("tied_stiefel", "asymmetric"):
        torch.manual_seed(42)
        m_static = QuantumBornGPT(
            vocab_size=V,
            embed_dim=d,
            seed_dim=d_seed,
            num_heads=4,
            num_layers=2,
            hidden_dim=96,
            block_size=16,
            readout_mode=mode,
            use_dynamic_povm_beta=False,
        )
        torch.manual_seed(42)
        m_dyn = QuantumBornGPT(
            vocab_size=V,
            embed_dim=d,
            seed_dim=d_seed,
            num_heads=4,
            num_layers=2,
            hidden_dim=96,
            block_size=16,
            readout_mode=mode,
            use_dynamic_povm_beta=True,
        )
        assert m_dyn.purity_gate is not None

        # At step 0, dynamic purity gate is exact identity (r == 1.0)
        logits_s, loss_s = m_static(idx, targets=targets)
        logits_d0, loss_d0 = m_dyn(idx, targets=targets)
        assert torch.allclose(logits_s, logits_d0, atol=1e-5)
        assert loss_d0 is not None and torch.isfinite(loss_d0)

        # Step 0 backward updates fc2; step 1 backward updates both fc1 and fc2
        opt = torch.optim.AdamW(m_dyn.purity_gate.parameters(), lr=1e-2)
        opt.zero_grad()
        loss_d0.backward()
        assert m_dyn.purity_gate.fc2.weight.grad is not None
        assert float(m_dyn.purity_gate.fc2.weight.grad.abs().max().item()) > 0.0
        opt.step()

        opt.zero_grad()
        logits_d1, loss_d1 = m_dyn(idx, targets=targets)
        assert logits_d1.shape == (4, 16, V)
        assert loss_d1 is not None and torch.isfinite(loss_d1)
        loss_d1.backward()
        assert m_dyn.purity_gate.fc1.weight.grad is not None
        assert m_dyn.purity_gate.fc2.weight.grad is not None
        assert float(m_dyn.purity_gate.fc1.weight.grad.abs().max().item()) > 0.0
        assert float(m_dyn.purity_gate.fc2.weight.grad.abs().max().item()) > 0.0


def test_classical_modded_gpt_forward_backward_and_softcap() -> None:
    """Verify ClassicalModdedGPT forward/backward shapes, logit softcapping <= 30.0, U-Net skips, and value mixing."""
    torch.manual_seed(42)
    V, d, H, L = 256, 64, 4, 4
    model = ClassicalModdedGPT(
        vocab_size=V,
        embed_dim=d,
        num_heads=H,
        num_layers=L,
        hidden_dim=128,
        block_size=32,
        use_bigram_smear=True,
        use_unet_interferometry=True,
        value_residual_mixing=True,
        input_highway_mixing=True,
        depth_scaled_residual=True,
        logit_softcap=30.0,
    )
    assert isinstance(model.blocks[0], ClassicalModdedBlock)
    assert model.unet_gates is not None and len(model.unet_gates) == L // 2
    assert model.unet_scales is not None and len(model.unet_scales) == L // 2

    # Scale lm_head large enough to saturate softcap and verify |logits| <= 30.0
    with torch.no_grad():
        model.lm_head.weight.mul_(25.0)

    idx = torch.randint(0, V, (4, 16), dtype=torch.long)
    targets = torch.randint(0, V, (4, 16), dtype=torch.long)
    logits, loss, layer_states = model(idx, targets=targets, return_layer_states=True)

    assert logits.shape == (4, 16, V)
    assert len(layer_states) == L
    assert loss is not None and torch.isfinite(loss)
    assert float(logits.abs().max().item()) <= 30.0 + 1e-5

    loss.backward()
    assert model.wte.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.smear_embed_gate is not None and model.smear_embed_gate.grad is not None
    for blk in model.blocks:
        assert blk.q_proj.weight.grad is not None
        assert blk.fc1.weight.grad is not None
        assert blk.v_mix_gate is not None and blk.v_mix_gate.grad is not None
        assert blk.highway_gate is not None and blk.highway_gate.grad is not None
    for u_scale in model.unet_scales:
        assert u_scale.grad is not None


def test_jax_dynamic_purity_parity() -> None:
    """Verify numerical parity (max_abs_diff < 1e-5) between PyTorch and JAX dynamic purity & logit-norm metrics."""
    torch.manual_seed(42)
    gate = ContextualBlochPurityGate(
        embed_dim=32,
        hidden_dim=16,
        r_min=0.25,
        r_max=4.0,
        init_bias=1.0,
        max_beta=160.0,
    )
    with torch.no_grad():
        nn.init.normal_(gate.fc1.weight, std=0.4)
        nn.init.normal_(gate.fc1.bias, std=0.2)
        nn.init.normal_(gate.fc2.weight, std=0.6)
        nn.init.constant_(gate.fc2.bias, 0.85)

    psi_pt = torch.randn(4, 12, 32, dtype=torch.float32)
    beta_pt = gate(psi_pt, base_inv_temp=32.5)

    psi_jax = jnp.asarray(psi_pt.numpy(), dtype=jnp.float32)
    w1_jax = jnp.asarray(gate.fc1.weight.detach().numpy(), dtype=jnp.float32)
    b1_jax = jnp.asarray(gate.fc1.bias.detach().numpy(), dtype=jnp.float32)
    w2_jax = jnp.asarray(gate.fc2.weight.detach().numpy(), dtype=jnp.float32)
    b2_jax = jnp.asarray(gate.fc2.bias.detach().numpy(), dtype=jnp.float32)

    beta_jax = contextual_bloch_purity_beta_jax(
        psi_jax,
        w1_jax,
        b1_jax,
        w2_jax,
        b2_jax,
        base_inv_temp=32.5,
        r_min=0.25,
        r_max=4.0,
        init_bias=1.0,
        max_beta=160.0,
    )
    max_diff = float(torch.from_numpy(beta_jax.__array__().copy()).sub(beta_pt).abs().max().item())
    assert max_diff < 1e-5, f"Expected PyTorch/JAX beta max_abs_diff < 1e-5, got {max_diff:.2e}"

    m_frame_pt = l2_normalize(torch.randn(64, 32, dtype=torch.float32), dim=-1)
    logits_pt = F.linear(l2_normalize(psi_pt, dim=-1), m_frame_pt) * (beta_pt / 10.0)
    logits_jax = jnp.asarray(logits_pt.detach().numpy(), dtype=jnp.float32)
    m_pt = compute_logit_norm_variation_metrics(logits_pt)
    m_jax = compute_logit_norm_variation_metrics_jax(logits_jax)
    for k in m_pt:
        assert abs(m_pt[k] - m_jax[k]) < 1e-5, f"Metric {k} mismatch: pt={m_pt[k]} vs jax={m_jax[k]}"


def test_heteroscedastic_sharpness_separation() -> None:
    """Verify ContextualBlochPurityGate assigns higher beta to deterministic contexts than branching contexts."""
    torch.manual_seed(42)
    V = 128
    d = 64
    K_branch = 16

    # Deterministic contexts: token u in [0, V//2) -> unique target with moderate cosine margin (~0.24)
    # requiring high beta (~80-120) to drive entropy to zero.
    # Branching contexts: token u in [V//2, V) -> uniform over K_branch=16 targets with cosine spread
    # [0.02, 0.32], requiring low beta (~12-18) to avoid over-sharpening the dominant branch.
    psi_table = l2_normalize(torch.randn(V, d), dim=-1)
    q_orth, _ = torch.linalg.qr(torch.randn(V, d))
    m_frame = l2_normalize(q_orth, dim=-1)

    det_targets = torch.tensor([(u * 3 + 1) % V for u in range(V // 2)], dtype=torch.long)
    branch_targets = torch.stack(
        [torch.tensor([(u * 5 + k * 7 + 3) % V for k in range(K_branch)], dtype=torch.long) for u in range(V // 2, V)],
        dim=0,
    )

    with torch.no_grad():
        # Encode context type in subspace so gate can distinguish deterministic vs branching states,
        # while keeping directional cosine margins in the realistic spherical crowding regime (~0.22)
        context_dir = l2_normalize(torch.randn(1, d), dim=-1)
        for u in range(V // 2):
            tgt = int(det_targets[u].item())
            noise_ket = l2_normalize(psi_table[u] + 0.6 * context_dir.squeeze(0), dim=-1)
            psi_table[u] = l2_normalize(0.24 * m_frame[tgt] + 0.76 * noise_ket, dim=-1)

        weights = torch.exp(-torch.linspace(0.0, 2.2, K_branch)).unsqueeze(1)
        for idx_u, u in enumerate(range(V // 2, V)):
            tgts = branch_targets[idx_u]
            superpos = l2_normalize((weights * m_frame[tgts]).sum(dim=0), dim=-1)
            noise_ket = l2_normalize(psi_table[u] - 0.6 * context_dir.squeeze(0), dim=-1)
            psi_table[u] = l2_normalize(0.42 * superpos + 0.58 * noise_ket, dim=-1)

    cos_all = F.linear(psi_table, m_frame)  # (V, V)

    # Static global beta optimization (60 steps)
    static_beta = nn.Parameter(torch.tensor(28.57))
    opt_s = torch.optim.Adam([static_beta], lr=0.5)

    # Dynamic purity gate optimization (60 steps)
    gate = ContextualBlochPurityGate(embed_dim=d, hidden_dim=32, r_min=0.20, r_max=4.0, max_beta=160.0)
    dyn_base_beta = nn.Parameter(torch.tensor(35.0))
    opt_d = torch.optim.Adam(list(gate.parameters()) + [dyn_base_beta], lr=0.04)

    cos_det = cos_all[: V // 2]
    cos_branch = cos_all[V // 2 :].unsqueeze(1).expand(-1, K_branch, -1).reshape(-1, V)
    psi_det = psi_table[: V // 2]
    psi_branch = psi_table[V // 2 :]
    y_branch_flat = branch_targets.reshape(-1)

    for _ in range(60):
        opt_s.zero_grad()
        b_s = torch.clamp(static_beta, min=1.0, max=60.0)
        loss_s = 0.5 * (
            F.cross_entropy(cos_det * b_s, det_targets)
            + F.cross_entropy(cos_branch * b_s, y_branch_flat)
        )
        loss_s.backward()
        opt_s.step()

        opt_d.zero_grad()
        b_base = torch.clamp(dyn_base_beta, min=1.0, max=60.0)
        logits_det, _ = gate.scale_logits(psi_det, cos_det, b_base)
        beta_br = gate(psi_branch, b_base)  # (V//2, 1)
        logits_br = cos_branch * beta_br.unsqueeze(1).expand(-1, K_branch, 1).reshape(-1, 1)
        loss_d = 0.5 * (
            F.cross_entropy(logits_det, det_targets)
            + F.cross_entropy(logits_br, y_branch_flat)
        )
        loss_d.backward()
        opt_d.step()

    with torch.no_grad():
        b_s_final = torch.clamp(static_beta, min=1.0, max=60.0)
        ce_static = float(
            0.5
            * (
                F.cross_entropy(cos_det * b_s_final, det_targets)
                + F.cross_entropy(cos_branch * b_s_final, y_branch_flat)
            ).item()
        )
        b_base_final = torch.clamp(dyn_base_beta, min=1.0, max=60.0)
        beta_det_mean = float(gate(psi_det, b_base_final).mean().item())
        beta_branch_mean = float(gate(psi_branch, b_base_final).mean().item())
        logits_det_f, _ = gate.scale_logits(psi_det, cos_det, b_base_final)
        beta_br_f = gate(psi_branch, b_base_final)
        logits_br_f = cos_branch * beta_br_f.unsqueeze(1).expand(-1, K_branch, 1).reshape(-1, 1)
        ce_dyn = float(
            0.5
            * (
                F.cross_entropy(logits_det_f, det_targets)
                + F.cross_entropy(logits_br_f, y_branch_flat)
            ).item()
        )

    assert beta_det_mean > 1.35 * beta_branch_mean, (
        f"Expected beta_det ({beta_det_mean:.2f}) > 1.35 * beta_branch ({beta_branch_mean:.2f})"
    )
    assert ce_dyn < ce_static - 0.15, (
        f"Expected dynamic purity gate CE ({ce_dyn:.4f}) < static CE ({ce_static:.4f}) - 0.15"
    )
