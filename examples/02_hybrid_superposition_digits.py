"""Example 2: Hybrid Spatial CNN + Complex Superposition Quantum Attention Classifier.

Reproduces and extends the multi-token complex superposition prototype using `qdlblocks`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from qdlblocks import HybridQuantumClassifier, SPAMNoise, purity, von_neumann_entropy


def run_hybrid_superposition_benchmark(epochs: int = 25, seed: int = 42) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("==========================================================================")
    print("   Hybrid Spatial Token + Complex Superposition Attention (qdlblocks)")
    print("==========================================================================")

    digits = load_digits()
    X, y = digits.images, digits.target
    X = X[:, np.newaxis, :, :] / 16.0  # Shape: (1797, 1, 8, 8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    spam_noise = SPAMNoise(std=0.02, clamp_range=(0.0, 1.0))
    model = HybridQuantumClassifier(
        in_channels=1,
        token_dim=32,
        hilbert_dim=16,
        num_heads=4,
        num_classes=10,
        pooling_mode="coherent",
        purification_placement="post_fusion",
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    print("\nEpoch | Train Loss | Train Acc (%) | Test Acc (%) | Avg S(rho) | Purity | LR")
    print("-" * 80)

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        spam_noise.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for bx, by in train_loader:
            bx_aug = spam_noise(bx)

            optimizer.zero_grad()
            logits, _ = model(bx_aug)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_train += (preds == by).sum().item()
            total_train += bx.size(0)

        scheduler.step()

        model.eval()
        spam_noise.eval()
        correct_test = 0
        total_test = 0
        entropies = []
        purities = []
        with torch.no_grad():
            for bx, by in test_loader:
                logits, rho = model(bx)
                preds = torch.argmax(logits, dim=-1)
                correct_test += (preds == by).sum().item()
                total_test += bx.size(0)
                entropies.append(von_neumann_entropy(rho).item())
                purities.append(purity(rho).item())

        train_acc = 100.0 * correct_train / total_train
        test_acc = 100.0 * correct_test / total_test
        avg_loss = total_loss / total_train
        avg_s = float(np.mean(entropies))
        avg_p = float(np.mean(purities))

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(
            f"{epoch:5d} | {avg_loss:10.4f} | {train_acc:13.2f} | {test_acc:12.2f} | "
            f"{avg_s:10.4f} | {avg_p:6.4f} | {scheduler.get_last_lr()[0]:.5f}"
        )

    print("-" * 80)
    print(f"Peak Test Accuracy Reached: {best_test_acc:.2f}%\n")
    return best_test_acc


if __name__ == "__main__":
    run_hybrid_superposition_benchmark()
