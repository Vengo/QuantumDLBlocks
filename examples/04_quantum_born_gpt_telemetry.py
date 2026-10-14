"""Example 4: Quantum-Geometric Born GPT, StiefelMuon Ablation, & Telemetry Inspector.

Demonstrates:
1. How unconstrained Muon accumulates Stiefel isometry drift (`||W^T W - I||_F / sqrt(d)`).
2. Why periodic hard polar snapping (`polar_snap_weights`) causes sudden validation loss spikes.
3. How `StiefelMuon(..., stiefel_penalty=0.02)` keeps Mean Isometry Error locked near 0.0
   smoothly and continuously without loss shocks, while also enabling the research extensions
   (`learnable_temperature`, `value_residual_mixing`, `input_highway_mixing`).
4. Full telemetry extraction via `inspect_quantum_model`.
"""

from __future__ import annotations

import copy

import torch

from qdlblocks import (
    QuantumBornGPT,
    StiefelMuon,
    inspect_quantum_model,
    model_isometry_error,
    polar_snap_weights,
    split_muon_adamw_params,
)


def make_synthetic_markov_sequences(
    num_sequences: int = 256,
    seq_len: int = 65,
    vocab_size: int = 256,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate structured multi-order induction/Markov token sequences for fast local testing."""
    gen = torch.Generator().manual_seed(seed)
    data = torch.zeros((num_sequences, seq_len), dtype=torch.long)
    data[:, 0] = torch.randint(0, vocab_size, (num_sequences,), generator=gen)
    data[:, 1] = torch.randint(0, vocab_size, (num_sequences,), generator=gen)
    for t in range(2, seq_len):
        # Deterministic + noisy induction rule: token_t depends on (token_{t-1}, token_{t-2})
        noise = (torch.rand(num_sequences, generator=gen) < 0.10).long() * torch.randint(
            1, 5, (num_sequences,), generator=gen
        )
        data[:, t] = (data[:, t - 1] * 3 + data[:, t - 2] * 5 + 7 + noise) % vocab_size

    split = int(0.8 * num_sequences)
    return data[:split], data[split:]


def run_ablation_experiment() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==========================================================================")
    print("   Quantum-Geometric Born GPT: Stiefel Isometry & Polar Snap Ablation")
    print(f"   Device: {device}")
    print("==========================================================================")

    vocab_size = 32
    block_size = 32
    train_seqs, val_seqs = make_synthetic_markov_sequences(
        num_sequences=400, seq_len=block_size + 1, vocab_size=vocab_size
    )
    train_seqs = train_seqs.to(device)
    val_seqs = val_seqs.to(device)

    vx, vy = val_seqs[:, :-1], val_seqs[:, 1:]

    # Base 6-Qubit (d=64) QuantumBornGPT model for rapid ablation comparison
    base_model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=64,
        seed_dim=16,
        num_heads=4,
        num_layers=4,
        hidden_dim=160,
        block_size=block_size,
        povm_temperature=0.05,
        readout_mode="asymmetric",
        learnable_temperature=True,
        value_residual_mixing=True,
        input_highway_mixing=True,
    ).to(device)

    configs = [
        ("1. Unconstrained Muon (wd=0, no retraction)", "none", 0.0, 0.0, False),
        ("2. Muon + Hard Polar Snap at Step 40", "none", 0.0, 0.0, True),
        ("3. StiefelMuon + Continuous Riemannian Penalty", "smooth", 0.01, 0.05, False),
    ]

    steps = 60
    batch_size = 32

    for title, decay_mode, wd, stiefel_pen, hard_snap in configs:
        torch.manual_seed(42)
        model = copy.deepcopy(base_model)
        muon_params, adamw_params = split_muon_adamw_params(model)

        opt_muon = StiefelMuon(
            muon_params,
            lr=0.04,
            momentum=0.90,
            weight_decay=wd,
            decay_mode=decay_mode,  # type: ignore[arg-type]
            stiefel_penalty=stiefel_pen,
        )
        opt_adam = torch.optim.AdamW(adamw_params, lr=0.01, weight_decay=1e-4)

        print(f"\n--- {title} ---")
        print("Step | Train Loss | Val Loss | Mean Isometry Error | Note")
        print("-" * 68)

        for step in range(1, steps + 1):
            model.train()
            ix = torch.randint(0, len(train_seqs), (batch_size,), device=device)
            bx, by = train_seqs[ix, :-1], train_seqs[ix, 1:]

            opt_muon.zero_grad()
            opt_adam.zero_grad()
            _, loss = model(bx, by)
            loss.backward()
            opt_muon.step()
            opt_adam.step()

            note = ""
            if hard_snap and step == 40:
                polar_snap_weights(model)
                note = "<-- HARD POLAR SNAP (Shock!)"

            if step in (1, 20, 39, 40, 45, 60):
                model.eval()
                with torch.no_grad():
                    _, vloss = model(vx, vy)
                    iso_err, _ = model_isometry_error(model)
                print(
                    f"{step:4d} | {loss.item():10.4f} | {vloss.item():8.4f} | "
                    f"{iso_err:19.4f} | {note}"
                )

    # Run full diagnostic telemetry probe on the final Riemannian-stabilized model
    print("\nRunning `inspect_quantum_model` on the Riemannian-stabilized Born GPT:")
    inspect_quantum_model(model, vx, vy, print_report=True)


if __name__ == "__main__":
    run_ablation_experiment()
