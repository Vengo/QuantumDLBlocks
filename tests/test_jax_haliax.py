"""Unit tests for Track 3 JAX, Equinox, Haliax, Optax, and Levanter integration (qdlblocks_jax)."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

import qdlblocks as qdl
import qdlblocks_jax as qjax
import haliax as hax


def test_jax_functional_parity_with_pytorch():
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((4, 8)).astype(np.float32)

    # 1. l2_normalize
    t_norm = qdl.l2_normalize(torch.from_numpy(x_np), dim=-1, eps=1e-12).numpy()
    j_norm = np.asarray(qjax.l2_normalize(jnp.asarray(x_np), axis=-1, eps=1e-12))
    np.testing.assert_allclose(j_norm, t_norm, atol=1e-5)

    # 2. state_to_density & power_purify
    psi_np = (
        rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    ).astype(np.complex64)
    t_rho = qdl.state_to_density(torch.from_numpy(psi_np), eps=1e-8)
    j_rho = qjax.state_to_density(jnp.asarray(psi_np), eps=1e-8)
    np.testing.assert_allclose(np.asarray(j_rho), t_rho.numpy(), atol=1e-5)

    # Mix two pure states and test power_purify + von_neumann_entropy
    psi2_np = (
        rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    ).astype(np.complex64)
    t_rho2 = qdl.state_to_density(torch.from_numpy(psi2_np), eps=1e-8)
    t_mix = 0.6 * t_rho + 0.4 * t_rho2
    j_mix = jnp.asarray(t_mix.numpy())

    t_pur = qdl.power_purify(t_mix, power=2, eps=1e-8).numpy()
    j_pur = np.asarray(qjax.power_purify(j_mix, power=2, eps=1e-8))
    np.testing.assert_allclose(j_pur, t_pur, atol=1e-5)

    t_vne = qdl.von_neumann_entropy(t_mix, eps=1e-8, reduction="none").numpy()
    j_vne = np.asarray(qjax.von_neumann_entropy(j_mix, eps=1e-8))
    np.testing.assert_allclose(j_vne, t_vne, atol=1e-5)

    # 3. kraus_transform & povm_measure
    k_np = (
        rng.standard_normal((2, 4, 4)) + 1j * rng.standard_normal((2, 4, 4))
    ).astype(np.complex64)
    t_kr = qdl.kraus_transform(t_mix, torch.from_numpy(k_np), eps=1e-8).numpy()
    j_kr = np.asarray(qjax.kraus_transform(j_mix, jnp.asarray(k_np), eps=1e-8))
    np.testing.assert_allclose(j_kr, t_kr, atol=1e-5)

    m_np = (
        rng.standard_normal((5, 4, 4)) + 1j * rng.standard_normal((5, 4, 4))
    ).astype(np.complex64)
    t_povm = qdl.povm_measure(t_mix, torch.from_numpy(m_np)).numpy()
    j_povm = np.asarray(qjax.povm_measure(j_mix, jnp.asarray(m_np)))
    np.testing.assert_allclose(j_povm, t_povm, atol=1e-5)

    # 4. zeropower_via_newtonschulz5
    g_np = rng.standard_normal((16, 8)).astype(np.float32)
    t_ns = qdl.zeropower_via_newtonschulz5(torch.from_numpy(g_np), steps=5, eps=1e-7).numpy()
    j_ns = np.asarray(qjax.zeropower_via_newtonschulz5(jnp.asarray(g_np), steps=5, eps=1e-7))
    np.testing.assert_allclose(j_ns, t_ns, atol=1e-4)


def test_flash_born_attention_jax_forward_and_backward_parity():
    rng = np.random.default_rng(123)
    B, H, T, D = 2, 2, 16, 8
    q_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
    k_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
    v_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
    do_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
    scale = math.sqrt(D)

    # PyTorch reference
    qt = torch.from_numpy(q_np).requires_grad_(True)
    kt = torch.from_numpy(k_np).requires_grad_(True)
    vt = torch.from_numpy(v_np).requires_grad_(True)
    dot = torch.from_numpy(do_np)
    ot = qdl.flash_born_attention(qt, kt, vt, scale=scale, is_causal=True, eps=1e-6)
    (ot * dot).sum().backward()

    # JAX scan backend
    qj = jnp.asarray(q_np)
    kj = jnp.asarray(k_np)
    vj = jnp.asarray(v_np)
    doj = jnp.asarray(do_np)

    oj_scan = qjax.flash_born_attention_jax(
        qj, kj, vj, scale=scale, is_causal=True, eps=1e-6, block_size=8, backend="scan"
    )
    oj_pallas = qjax.flash_born_attention_jax(
        qj, kj, vj, scale=scale, is_causal=True, eps=1e-6, block_size=8, backend="pallas"
    )
    np.testing.assert_allclose(np.asarray(oj_scan), ot.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(oj_pallas), ot.detach().numpy(), atol=1e-5)

    def loss_fn(q, k, v):
        out = qjax.flash_born_attention_jax(
            q, k, v, scale=scale, is_causal=True, eps=1e-6, block_size=8, backend="scan"
        )
        return jnp.sum(out * doj)

    dq_j, dk_j, dv_j = jax.grad(loss_fn, argnums=(0, 1, 2))(qj, kj, vj)
    assert qt.grad is not None and kt.grad is not None and vt.grad is not None
    np.testing.assert_allclose(np.asarray(dq_j), qt.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(np.asarray(dk_j), kt.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(np.asarray(dv_j), vt.grad.numpy(), atol=1e-4)

    # Test Pallas backward kernel + differentiable per-head scale (H,) parity against scan backend
    scale_vec = jnp.array([2.5, 3.2], dtype=jnp.float32)

    def loss_pallas(q, k, v, sc):
        out = qjax.flash_born_attention_jax(
            q, k, v, scale=sc, is_causal=True, eps=1e-6, block_size=8, backend="pallas"
        )
        return jnp.sum(out * doj)

    def loss_scan(q, k, v, sc):
        out = qjax.flash_born_attention_jax(
            q, k, v, scale=sc, is_causal=True, eps=1e-6, block_size=8, backend="scan"
        )
        return jnp.sum(out * doj)

    dq_p, dk_p, dv_p, dsc_p = jax.grad(loss_pallas, argnums=(0, 1, 2, 3))(qj, kj, vj, scale_vec)
    dq_s, dk_s, dv_s, dsc_s = jax.grad(loss_scan, argnums=(0, 1, 2, 3))(qj, kj, vj, scale_vec)
    np.testing.assert_allclose(np.asarray(dq_p), np.asarray(dq_s), atol=1e-4)
    np.testing.assert_allclose(np.asarray(dk_p), np.asarray(dk_s), atol=1e-4)
    np.testing.assert_allclose(np.asarray(dv_p), np.asarray(dv_s), atol=1e-4)
    np.testing.assert_allclose(np.asarray(dsc_p), np.asarray(dsc_s), atol=1e-4)


def test_partial_trace_and_bipartite_entanglement():
    # 1. Product state |00> has zero entanglement entropy
    psi_prod = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex64)
    ent_prod = qjax.bipartite_entanglement_entropy(psi_prod, num_qubits=2, keep_qubits=(0,))
    np.testing.assert_allclose(float(ent_prod), 0.0, atol=1e-5)

    # 2. Maximally entangled Bell state (|00> + |11>) / sqrt(2) has S(rho_A) = ln(2)
    psi_bell = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)
    ent_bell = qjax.bipartite_entanglement_entropy(psi_bell, num_qubits=2, keep_qubits=(0,))
    np.testing.assert_allclose(float(ent_bell), math.log(2.0), atol=1e-5)

    # 3. Non-contiguous keep_qubits on 3-qubit GHZ state (|000> + |111>) / sqrt(2)
    psi_ghz = jnp.zeros((8,), dtype=jnp.complex64).at[0].set(1.0).at[7].set(1.0) / jnp.sqrt(2.0)
    rho_02 = qjax.partial_trace_qubits(psi_ghz, num_qubits=3, keep_qubits=(2, 0))
    assert rho_02.shape == (4, 4)
    ent_ghz = qjax.bipartite_entanglement_entropy(psi_ghz, num_qubits=3, keep_qubits=(2, 0))
    np.testing.assert_allclose(float(ent_ghz), math.log(2.0), atol=1e-5)

    # 4. Fast 10-qubit (d=1024, 5+5 split) pure-state Schmidt partial trace parity with PyTorch
    rng = np.random.default_rng(99)
    psi_10_np = rng.standard_normal((3, 1024)).astype(np.float32)
    ent_jax = np.asarray(
        qjax.bipartite_entanglement_entropy(jnp.asarray(psi_10_np), num_qubits=10, keep_qubits=(0, 1, 2, 3, 4))
    )
    ent_pt = qdl.bipartite_entanglement_entropy(
        torch.from_numpy(psi_10_np), num_qubits=10, keep_qubits=(0, 1, 2, 3, 4)
    ).numpy()
    np.testing.assert_allclose(ent_jax, ent_pt, atol=1e-4)


def test_haliax_quantum_born_gpt_and_stiefel_muon_optimizer():
    key = jax.random.PRNGKey(0)
    cfg = qjax.HaxQuantumBornLMConfig(
        vocab_size=32,
        seq_len=8,
        seed_dim=8,
        embed_dim=16,  # 4 qubits (2^4 = 16)
        num_heads=2,
        mlp_dim=32,
        num_layers=2,
        num_experts=4,
        top_k_experts=2,
        use_flash_born=True,
        mtp_horizons=3,
        mtp_loss_weight=0.25,
    )
    model = cfg.build(key=key)
    assert len(model.mtp_rotations) == 2

    Batch = hax.Axis("batch", 2)
    Seq = model.Seq
    tok_arr = jax.random.randint(key, (Batch.size, Seq.size), 0, cfg.vocab_size)
    tgt_arr = jax.random.randint(key, (Batch.size, Seq.size), 0, cfg.vocab_size)
    tokens = hax.NamedArray(tok_arr, (Batch, Seq))
    targets = hax.NamedArray(tgt_arr, (Batch, Seq))

    psi = model.encode(tokens)
    assert psi.axes == (Batch, Seq, model.Hilbert)

    # Test factorize_into_qubits (16 -> 4 qubits of size 2)
    psi_qubits = model.factorize_into_qubits(psi)
    assert len(psi_qubits.axes) == 2 + 4
    assert all(ax.size == 2 for ax in psi_qubits.axes[2:])

    # Test multi-horizon POVM logits
    mtp_logits = model.forward_mtp_logits(tokens)
    assert len(mtp_logits) == 3

    # Test Optax Stiefel-Muon training step
    tx = qjax.build_quantum_optax_optimizer(
        model,
        muon_lr=0.02,
        adamw_lr=1e-3,
        stiefel_penalty=0.05,
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = tx.init(params)

    @eqx.filter_value_and_grad
    def compute_loss(m):
        _, loss = m(tokens, targets)
        assert loss is not None
        return loss

    loss_initial, grads = compute_loss(model)
    updates, opt_state = tx.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    loss_after, _ = compute_loss(model)

    assert jnp.isfinite(loss_initial) and jnp.isfinite(loss_after)
    iso_err = qjax.stiefel_isometry_error(model.blocks[0].attn.w_q.array)
    assert float(iso_err) < 0.05


def test_jax_distributed_sharding_and_wsd_schedule():
    key = jax.random.PRNGKey(7)
    cfg = qjax.HaxQuantumBornLMConfig(
        vocab_size=32,
        seq_len=8,
        seed_dim=8,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        num_layers=1,
    )
    model = cfg.build(key=key)

    mapping = qjax.build_quantum_axis_mapping()
    assert mapping["param_mapping"]["hilbert"] == "data"
    assert mapping["param_mapping"]["heads"] == "model"

    mesh = qjax.make_quantum_device_mesh()
    sharded_model = qjax.shard_quantum_model(model, mesh=mesh)
    part_info = qjax.partition_stiefel_leaves(sharded_model, world_size=2)
    assert part_info["total_stiefel_matrices"] > 0
    assert len(part_info["assignments"]) == part_info["total_stiefel_matrices"]

    # Test distributed_stiefel_muon_step
    weights_list = [jnp.eye(16, dtype=jnp.float32), jnp.eye(16, dtype=jnp.float32)]
    grads_list = [jnp.ones((16, 16), dtype=jnp.float32) * 0.01, jnp.ones((16, 16), dtype=jnp.float32) * 0.02]
    mom_list = [jnp.zeros((16, 16), dtype=jnp.float32), jnp.zeros((16, 16), dtype=jnp.float32)]
    new_w, new_mom, meta = qjax.distributed_stiefel_muon_step(
        grads_list, mom_list, weights_list, world_size=2, lr=0.02
    )
    assert len(new_w) == 2 and len(new_mom) == 2
    assert meta["world_size"] == 2

    # Test wsd_lr_schedule_jax
    sched = qjax.wsd_lr_schedule_jax(peak_lr=0.02, total_steps=100, warmup_steps=10, decay_fraction=0.2, min_lr_ratio=0.1)
    assert abs(float(sched(5)) - 0.01) < 1e-5
    assert abs(float(sched(50)) - 0.02) < 1e-5
    assert abs(float(sched(100)) - 0.002) < 1e-5


