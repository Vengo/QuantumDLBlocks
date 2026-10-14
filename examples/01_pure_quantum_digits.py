"""Example 1: Multi-Layer Pure Quantum Attention Network on Digits (No Conv, No ReLU, No Softmax).

Reproduces and extends the pure Hilbert-space network from the notebook prototype using `qdlblocks`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from qdlblocks import DeepPureQuantumNetwork, SPAMNoise, purity, von_neumann_entropy


def run_pure_quantum_benchmark(epochs: int = 25, seed: int = 42) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("==========================================================================")
    print("   Multi-Layer Pure Quantum Attention Network Benchmark (qdlblocks)")
    print("   (Zero Conv, Zero ReLU, Zero MaxPool, Zero Attention Softmax)")
    print("==========================================================================")

    digits = load_digits()
    X, y = digits.data / 16.0, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)

    in_dim = X.shape[1]  # 64
    spam_noise = SPAMNoise(std=0.02)
    model = DeepPureQuantumNetwork(
        in_features=in_dim,
        hilbert_dim=16,
        num_classes=10,
        num_heads=4,
        num_kraus=4,
        num_kraus_layers=1,
        purification_placement="post_fusion",
        purify_after_kraus=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    num_batches = int(np.ceil(len(X_tr_t) / batch_size))

    print(f"\nDataset size: {len(X_train)} Train, {len(X_test)} Test | Feature Dim: {in_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("\nEpoch | Train Loss | Train Acc (%) | Test Acc (%) | Avg S(rho) | Purity Tr(rho^2) | LR")
    print("-" * 88)

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        spam_noise.train()
        perm = torch.randperm(len(X_tr_t))
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i in range(num_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            bx, by = X_tr_t[idx], y_tr_t[idx]

            bx_noisy = spam_noise(bx)

            optimizer.zero_grad()
            logits, _ = model(bx_noisy)
            loss = criterion(logits, by)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            correct_train += (preds == by).sum().item()
            total_train += len(by)

        scheduler.step()

        model.eval()
        spam_noise.eval()
        with torch.no_grad():
            test_logits, test_rho = model(X_te_t)
            test_preds = test_logits.argmax(dim=-1)
            test_acc = (test_preds == y_te_t).float().mean().item() * 100.0
            avg_entropy = von_neumann_entropy(test_rho).item()
            avg_purity = purity(test_rho).item()

        train_acc = (correct_train / total_train) * 100.0
        epoch_loss = running_loss / total_train
        current_lr = scheduler.get_last_lr()[0]

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(
            f"{epoch:5d} | {epoch_loss:10.4f} | {train_acc:13.2f} | {test_acc:12.2f} | "
            f"{avg_entropy:10.4f} | {avg_purity:16.4f} | {current_lr:.5f}"
        )

    print("-" * 88)
    print(f"Peak Test Accuracy Reached: {best_test_acc:.2f}%\n")
    return best_test_acc


if __name__ == "__main__":
    run_pure_quantum_benchmark()
