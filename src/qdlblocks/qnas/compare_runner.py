"""Head-to-Head Comparison Harness for all Three Quantum-Geometric NAS Paradigms."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .entropy_adaptive import EntropyAdaptiveNetwork
from .search_space import DiscreteQuantumBackbone
from .superposition_darts import SuperpositionDARTSNetwork, anneal_purification_power
from .zero_shot_proxy import QuantumZeroShotNAS, kendall_tau_corr, spearman_rank_corr


@dataclass
class QNASMethodResult:
    method_name: str
    discovered_topology: str
    params: int
    search_time_sec: float
    test_acc_pct: float
    extra_metrics: dict[str, float]


def load_qnas_dataset(
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load 8x8 Digits dataset (or deterministic synthetic fallback) with 80/20 train/test split."""
    rng = np.random.default_rng(seed)
    try:
        from sklearn.datasets import load_digits

        digits = load_digits()
        x_np = digits.data.astype(np.float32)
        y_np = digits.target.astype(np.int64)
    except Exception:
        # Deterministic 10-class prototype fallback
        prototypes = rng.normal(0.0, 1.0, size=(10, 64)).astype(np.float32)
        y_np = np.arange(1000, dtype=np.int64) % 10
        x_np = prototypes[y_np] + 0.35 * rng.normal(0.0, 1.0, size=(1000, 64)).astype(np.float32)

    mean = x_np.mean(axis=0, keepdims=True)
    std = x_np.std(axis=0, keepdims=True) + 1e-6
    x_np = (x_np - mean) / std

    perm = rng.permutation(len(x_np))
    x_np, y_np = x_np[perm], y_np[perm]
    split = int(0.8 * len(x_np))

    return (
        torch.from_numpy(x_np[:split]),
        torch.from_numpy(y_np[:split]),
        torch.from_numpy(x_np[split:]),
        torch.from_numpy(y_np[split:]),
    )


def _train_model_epochs(
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    epochs: int,
    lr: float = 0.008,
    batch_size: int = 64,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    n = x_tr.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=x_tr.device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(x_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            loss.backward()
            opt.step()


def _eval_acc(model: nn.Module, x_te: torch.Tensor, y_te: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x_te)
        return float((logits.argmax(dim=-1) == y_te).float().mean().item() * 100.0)


def format_qnas_table(results: list[QNASMethodResult]) -> str:
    lines = [
        "| Q-NAS Method | Discovered Topology | Params | Search Time | Test Acc (%) | Extra Telemetry |",
        "| :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for r in results:
        extra = ", ".join(f"`{k}={v:.3f}`" for k, v in r.extra_metrics.items())
        lines.append(
            f"| **{r.method_name}** | `{r.discovered_topology}` | {r.params:,} "
            f"| {r.search_time_sec:.2f}s | **{r.test_acc_pct:.2f}%** | {extra} |"
        )
    return "\n".join(lines)


def run_qnas_comparison(
    device: str = "cuda",
    epochs: int = 20,
    num_zero_shot_candidates: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all three Q-NAS paradigms plus Classical Softmax DARTS head-to-head."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    x_tr, y_tr, x_te, y_te = load_qnas_dataset(seed=seed)
    x_tr, y_tr, x_te, y_te = x_tr.to(dev), y_tr.to(dev), x_te.to(dev), y_te.to(dev)

    results: list[QNASMethodResult] = []
    ft_epochs = max(2, epochs // 2)

    # 1. Classical Softmax DARTS vs 2. Quantum Superposition DARTS
    for mode_label, mix_mode in [
        ("Classical Softmax DARTS", "classical_softmax"),
        ("Quantum Superposition DARTS", "quantum_superposition"),
    ]:
        torch.manual_seed(seed)
        t0 = time.time()
        supernet = SuperpositionDARTSNetwork(
            in_features=64,
            num_tokens=8,
            embed_dim=32,
            num_heads=4,
            num_cells=3,
            num_classes=10,
            mixing_mode=mix_mode,  # type: ignore[arg-type]
        ).to(dev)
        opt = torch.optim.AdamW(supernet.parameters(), lr=0.008, weight_decay=1e-4)
        supernet.train()
        n = x_tr.size(0)
        for ep in range(epochs):
            supernet.purification_power = anneal_purification_power(ep, epochs, 1.0, 4.0)
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 64):
                idx = perm[i : i + 64]
                opt.zero_grad(set_to_none=True)
                logits = supernet(x_tr[idx])
                loss = F.cross_entropy(logits, y_tr[idx])
                loss.backward()
                opt.step()

        search_elapsed = time.time() - t0
        disc_model = supernet.export_discrete_model()
        _train_model_epochs(disc_model, x_tr, y_tr, epochs=ft_epochs, lr=0.006)
        acc = _eval_acc(disc_model, x_te, y_te)
        max_probs = [
            float(c.get_op_probabilities(supernet.purification_power).max().item())
            for c in supernet.cells
        ]
        results.append(
            QNASMethodResult(
                method_name=mode_label,
                discovered_topology=" -> ".join(supernet.selected_ops()),
                params=sum(p.numel() for p in disc_model.parameters()),
                search_time_sec=search_elapsed,
                test_acc_pct=acc,
                extra_metrics={"mean_winner_prob": float(np.mean(max_probs))},
            )
        )

    # 3. Thermodynamic Entropy-Adaptive Kraus NAS
    torch.manual_seed(seed)
    t0 = time.time()
    adaptive_net = EntropyAdaptiveNetwork(
        in_features=64,
        stem_dim=8,
        init_dims=(8, 8, 8),
        num_classes=10,
        num_kraus=3,
        expand_entropy_ratio=0.58,
        purify_purity_threshold=0.45,
    ).to(dev)
    opt_ad = torch.optim.AdamW(adaptive_net.parameters(), lr=0.008, weight_decay=1e-4)
    adapt_interval = max(2, epochs // 4)
    n = x_tr.size(0)
    total_events = 0

    for ep in range(1, epochs + 1):
        adaptive_net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            opt_ad.zero_grad(set_to_none=True)
            logits = adaptive_net(x_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            loss.backward()
            opt_ad.step()

        if ep % adapt_interval == 0:
            rec = adaptive_net.step_topology_adaptation(x_tr[:128])
            if len(rec["events"]) > 0:
                total_events += len(rec["events"])
                opt_ad = torch.optim.AdamW(adaptive_net.parameters(), lr=0.008, weight_decay=1e-4)

    adapt_elapsed = time.time() - t0
    acc_ad = _eval_acc(adaptive_net, x_te, y_te)
    last_hist = (
        adaptive_net.adaptation_history[-1]
        if adaptive_net.adaptation_history
        else {"entropies": [0.0], "purities": [1.0]}
    )
    results.append(
        QNASMethodResult(
            method_name="Entropy-Adaptive Kraus NAS",
            discovered_topology=f"dims={adaptive_net.layer_dims}, p={adaptive_net.purification_powers}",
            params=sum(p.numel() for p in adaptive_net.parameters()),
            search_time_sec=adapt_elapsed,
            test_acc_pct=acc_ad,
            extra_metrics={
                "adapt_events": float(total_events),
                "final_entropy": float(np.mean(last_hist["entropies"])),
                "final_purity": float(np.mean(last_hist["purities"])),
            },
        )
    )

    # 4. Zero-Shot Quantum Proxy NAS
    torch.manual_seed(seed)
    t0 = time.time()
    zs_nas = QuantumZeroShotNAS(
        in_features=64, num_tokens=8, embed_dim=32, num_heads=4, num_classes=10
    )
    best_cfg, ranked = zs_nas.search(
        x_tr[:64], y_tr[:64], num_candidates=num_zero_shot_candidates, seed=seed
    )
    zs_search_elapsed = time.time() - t0

    best_model = DiscreteQuantumBackbone(best_cfg).to(dev)
    _train_model_epochs(best_model, x_tr, y_tr, epochs=epochs, lr=0.008)
    acc_zs = _eval_acc(best_model, x_te, y_te)

    # Evaluate correlation between Zero-Shot Proxy Score and trained accuracy across subset
    eval_k = min(6, len(ranked))
    proxy_scores: list[float] = []
    trained_accs: list[float] = []
    short_ep = max(3, epochs // 3)
    for item in ranked[:eval_k]:
        proxy_scores.append(float(item["total_score"]))
        cand_m = DiscreteQuantumBackbone(item["config"]).to(dev)
        _train_model_epochs(cand_m, x_tr, y_tr, epochs=short_ep, lr=0.008)
        trained_accs.append(_eval_acc(cand_m, x_te, y_te))

    sp_rho = spearman_rank_corr(proxy_scores, trained_accs)
    kt_tau = kendall_tau_corr(proxy_scores, trained_accs)

    results.append(
        QNASMethodResult(
            method_name="Quantum Zero-Shot Proxy NAS",
            discovered_topology=" -> ".join(best_cfg.cell_ops),
            params=sum(p.numel() for p in best_model.parameters()),
            search_time_sec=zs_search_elapsed,
            test_acc_pct=acc_zs,
            extra_metrics={
                "top_proxy_score": float(ranked[0]["total_score"]),
                "spearman_rho": sp_rho,
            },
        )
    )

    return {
        "results": results,
        "proxy_rank_correlations": {"spearman_rho": sp_rho, "kendall_tau": kt_tau},
        "markdown_table": format_qnas_table(results),
    }
