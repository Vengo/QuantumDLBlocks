"""Example 3: Custom Deep Hilbert-Space Pipeline & Quantum Entropy/Purity Trajectory Study.

Demonstrates how to compose custom quantum deep learning pipelines using modular `qdlblocks` layers:
- Incoherent token superposition -> mixed state
- Dimension-expanding Kraus channel (C^{12x12} -> C^{20x20})
- Non-linear Power State Purification (p=2)
- Dimension-compressing Kraus channel (C^{20x20} -> C^{12x12})
- Second Power State Purification (p=2)
- POVM Measurement readout

Tracks how von Neumann entropy S(rho) and state purity Tr(rho^2) evolve layer-by-layer!
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from qdlblocks import (
    ComplexSuperpositionAttention,
    KrausChannel,
    POVMMeasurement,
    PowerPurification,
    SPAMNoise,
    is_valid_density_matrix,
    purity,
    von_neumann_entropy,
)


class CustomBottleneckQuantumNet(nn.Module):
    """Custom multi-stage Hilbert-space network with dimension expansion/compression."""

    def __init__(self, patch_dim: int = 16, num_classes: int = 10) -> None:
        super().__init__()
        # Stage 1: Token superposition over 4 quadrants (L=4, D=16) -> mixed rho in C^{12x12}
        self.attn = ComplexSuperpositionAttention(
            in_features=patch_dim,
            hilbert_dim=12,
            num_heads=4,
            pooling_mode="incoherent",       # Mixed state across spatial quadrants
            purification_placement="none",   # We will track purification explicitly below!
        )
        self.purify1 = PowerPurification(power=2)

        # Stage 2: Expand Hilbert space C^{12x12} -> C^{20x20} via rectangular Kraus operators
        self.kraus_expand = KrausChannel(in_hilbert_dim=12, out_hilbert_dim=20, num_kraus=4)
        self.purify2 = PowerPurification(power=2)

        # Stage 3: Compress Hilbert space C^{20x20} -> C^{12x12} via rectangular Kraus operators
        self.kraus_compress = KrausChannel(in_hilbert_dim=20, out_hilbert_dim=12, num_kraus=4)
        self.purify3 = PowerPurification(power=2)

        # Stage 4: POVM readout
        self.povm = POVMMeasurement(hilbert_dim=12, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Reshape 8x8 image into 4 quadrant patches of 4x4=16 features: (B, L=4, D=16)
        b = x.shape[0]
        img = x.view(b, 2, 4, 2, 4).permute(0, 1, 3, 2, 4).reshape(b, 4, 16)

        rho_attn = self.attn(img)                 # (B, 12, 12) mixed state
        rho_p1 = self.purify1(rho_attn)           # (B, 12, 12) purified
        rho_exp = self.kraus_expand(rho_p1)       # (B, 20, 20) expanded Kraus state
        rho_p2 = self.purify2(rho_exp)            # (B, 20, 20) purified
        rho_comp = self.kraus_compress(rho_p2)    # (B, 12, 12) compressed Kraus state
        rho_p3 = self.purify3(rho_comp)           # (B, 12, 12) final purified state

        logits = self.povm(rho_p3)
        states = {
            "1_post_incoherent_attn (12x12)": rho_attn,
            "2_post_purify_1        (12x12)": rho_p1,
            "3_post_kraus_expand    (20x20)": rho_exp,
            "4_post_purify_2        (20x20)": rho_p2,
            "5_post_kraus_compress  (12x12)": rho_comp,
            "6_post_purify_3        (12x12)": rho_p3,
        }
        return logits, states


def run_entropy_study(epochs: int = 20, seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("==========================================================================")
    print("   Deep Quantum Kraus Bottleneck & Power Purification Entropy Study")
    print("==========================================================================")

    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    model = CustomBottleneckQuantumNet()
    spam = SPAMNoise(std=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    num_batches = int(np.ceil(len(X_tr) / batch_size))

    for epoch in range(1, epochs + 1):
        model.train()
        spam.train()
        perm = torch.randperm(len(X_tr))
        for i in range(num_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            bx, by = spam(X_tr[idx]), y_tr[idx]
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                te_logits, _ = model(X_te)
                te_acc = (te_logits.argmax(dim=-1) == y_te).float().mean().item() * 100.0
            print(f"Epoch {epoch:2d}/{epochs} | Test Accuracy: {te_acc:.2f}%")

    # Inspect layer-by-layer quantum information trajectory on the test set
    model.eval()
    print("\nLayer-by-Layer Quantum State Trajectory on Test Set:")
    print("-" * 78)
    print(f"{'Stage':32s} | {'S(rho) [nats]':14s} | {'Purity Tr(rho^2)':16s} | {'Valid Density?'}")
    print("-" * 78)
    with torch.no_grad():
        _, states = model(X_te)
        for stage_name, rho in states.items():
            s_val = von_neumann_entropy(rho).item()
            p_val = purity(rho).item()
            valid, _ = is_valid_density_matrix(rho, atol=1e-4)
            print(f"{stage_name:32s} | {s_val:14.4f} | {p_val:16.4f} | {str(valid)}")
    print("-" * 78)


if __name__ == "__main__":
    run_entropy_study()
