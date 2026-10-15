#!/usr/bin/env python3
"""Example 09: JAX + Equinox + Haliax + Optax Named-Axis Quantum Born GPT.

Demonstrates:
- Named-axis `HaxQuantumBornGPT` with `HaxIsometricHilbertLift`, `HaxSubspaceBornAttention`
  (backed by O(T) `flash_born_attention_jax`), optional `HaxQuantumBornMoE`, and `HaxProjectivePOVMHead`
- Optax `scale_by_stiefel_muon` + `build_quantum_optax_optimizer` with Continuous Riemannian Stiefel Penalty
- Explicit multi-qubit tensor factorization (`factorize_into_qubits`) and bipartite von Neumann
  entanglement entropy (`bipartite_entanglement_entropy`) across a 3+3 qubit partition (2^6 = 64).
"""

from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp

import qdlblocks_jax as qjax
import haliax as hax


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX + Haliax Quantum Born GPT Demo")
    parser.add_argument("--steps", type=int, default=15, help="Number of training steps.")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length.")
    parser.add_argument(
        "--moe",
        action="store_true",
        help="Enable zero-Softmax HaxQuantumBornMoE routing.",
    )
    args = parser.parse_args()

    key = jax.random.PRNGKey(42)
    cfg = qjax.HaxQuantumBornLMConfig(
        vocab_size=128,
        seq_len=args.seq_len,
        seed_dim=16,
        embed_dim=64,  # 6 qubits: 2^6 = 64
        num_heads=4,
        mlp_dim=128,
        num_layers=2,
        num_experts=4 if args.moe else None,
        top_k_experts=2,
        povm_temperature=0.035,
        use_flash_born=True,
    )
    k_model, k_data = jax.random.split(key)
    model = cfg.build(key=k_model)

    Batch = hax.Axis("batch", 8)
    Seq = model.Seq

    # Construct a deterministic repeating pattern so loss converges rapidly
    base_seq = jnp.arange(Batch.size * Seq.size, dtype=jnp.int32).reshape(Batch.size, Seq.size) % 16
    tgt_seq = (base_seq + 1) % 16
    tokens = hax.NamedArray(base_seq, (Batch, Seq))
    targets = hax.NamedArray(tgt_seq, (Batch, Seq))

    tx = qjax.build_quantum_optax_optimizer(
        model,
        muon_lr=0.02,
        adamw_lr=3e-3,
        stiefel_penalty=0.05,
    )
    opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(m: qjax.HaxQuantumBornGPT, state):
        @eqx.filter_value_and_grad
        def loss_fn(mod: qjax.HaxQuantumBornGPT):
            _, loss = mod(tokens, targets)
            assert loss is not None
            return loss

        loss_val, grads = loss_fn(m)
        params = eqx.filter(m, eqx.is_inexact_array)
        updates, new_state = tx.update(grads, state, params)
        new_m = eqx.apply_updates(m, updates)
        return new_m, new_state, loss_val

    mode_str = "Sparse Born MoE (4 Experts, Top-2)" if args.moe else "Dense Projective MLP"
    print("=" * 84)
    print(f"HaxQuantumBornGPT (6-Qubit Hilbert Space d=64) | {mode_str}")
    print("=" * 84)

    for step in range(1, args.steps + 1):
        model, opt_state, loss_val = train_step(model, opt_state)
        if step == 1 or step % 5 == 0 or step == args.steps:
            iso_q = float(qjax.stiefel_isometry_error(model.blocks[0].attn.w_q.array))
            psi = model.encode(tokens)
            ent_3q = qjax.bipartite_entanglement_entropy(
                psi.array,
                num_qubits=6,
                keep_qubits=(0, 1, 2),
            )
            mean_ent = float(jnp.mean(ent_3q))
            print(
                f"Step {step:02d}/{args.steps:02d} | Loss: {float(loss_val):.4f} | "
                f"Stiefel Isometry Error: {iso_q:.5f} | "
                f"3+3 Qubit Bipartite Entanglement S(rho_A): {mean_ent:.4f} nats"
            )

    psi_final = model.encode(tokens)
    psi_qubits = model.factorize_into_qubits(psi_final)
    print("-" * 84)
    print(f"Factorized Named Qubit Axes: {[f'{ax.name}:{ax.size}' for ax in psi_qubits.axes]}")
    print("=" * 84)


if __name__ == "__main__":
    main()
