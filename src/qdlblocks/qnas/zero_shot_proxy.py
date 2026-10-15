"""Method 3: Zero-Shot Quantum-Geometric Proxy Search & Rank Correlation Metrics."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..layers.attention import SubspaceProjectorBornAttention
from ..metrics import attention_entropy_and_sparsity, effective_rank, model_isometry_error
from .search_space import ArchConfig, DiscreteQuantumBackbone, sample_random_arch


def compute_qnas_proxy_score(
    model: DiscreteQuantumBackbone,
    x_calib: torch.Tensor,
    y_calib: torch.Tensor,
    w_rank: float = 1.0,
    w_grad: float = 0.5,
    w_entropy: float = 0.25,
    w_iso: float = 0.5,
) -> dict[str, float]:
    """Compute training-free Quantum-Geometric NAS proxy score on a single minibatch."""
    model.zero_grad(set_to_none=True)

    for m in model.modules():
        if isinstance(m, SubspaceProjectorBornAttention):
            m.record_attention = True
            m.use_flash_born = False

    logits, layer_states = model(x_calib, return_layer_states=True)
    loss = F.cross_entropy(logits, y_calib)
    loss.backward()

    r_list: list[float] = []
    for psi_l in layer_states:
        psi_flat = psi_l.detach().reshape(-1, psi_l.size(-1))
        n_tok, d = psi_flat.shape
        r_val = effective_rank(psi_flat) / float(max(min(n_tok, d), 1))
        r_list.append(r_val)
    mean_norm_rank = float(np.mean(r_list)) if r_list else 0.0

    synflow_norms: list[float] = []
    for p in model.parameters():
        if p.ndim == 2 and p.grad is not None:
            syn_val = float(torch.norm(p.detach() * p.grad.detach(), p="fro").item())
            synflow_norms.append(syn_val)
    g_syn = float(np.mean(synflow_norms)) if synflow_norms else 0.0
    log_synflow = math.log(1.0 + g_syn)

    attn_entropies: list[float] = []
    for m in model.modules():
        if isinstance(m, SubspaceProjectorBornAttention):
            if m.last_born_weights is not None:
                stats_a = attention_entropy_and_sparsity(m.last_born_weights)
                attn_entropies.append(stats_a["entropy"])
            m.record_attention = False
    born_attn_entropy = float(np.mean(attn_entropies)) if attn_entropies else 0.5

    iso_err, _ = model_isometry_error(model)

    total_score = (
        w_rank * mean_norm_rank
        + w_grad * log_synflow
        + w_entropy * born_attn_entropy
        - w_iso * iso_err
    )
    model.zero_grad(set_to_none=True)

    return {
        "total_score": float(total_score),
        "norm_effective_rank": mean_norm_rank,
        "log_synflow": log_synflow,
        "born_attn_entropy": born_attn_entropy,
        "isometry_error": iso_err,
    }


class QuantumZeroShotNAS:
    """Training-Free Quantum-Geometric Architecture Search Runner."""

    def __init__(
        self,
        in_features: int = 64,
        num_tokens: int = 8,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_classes: int = 10,
    ) -> None:
        self.in_features = in_features
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

    def search(
        self,
        x_calib: torch.Tensor,
        y_calib: torch.Tensor,
        num_candidates: int = 12,
        seed: int = 42,
    ) -> tuple[ArchConfig, list[dict[str, Any]]]:
        rng = np.random.default_rng(seed)
        seen: set[tuple[str, ...]] = set()
        candidates: list[ArchConfig] = []
        attempts = 0
        max_attempts = 10 * num_candidates

        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            cfg = sample_random_arch(
                rng,
                in_features=self.in_features,
                num_tokens=self.num_tokens,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_classes=self.num_classes,
            )
            if cfg.cell_ops not in seen:
                seen.add(cfg.cell_ops)
                candidates.append(cfg)

        ranked_history: list[dict[str, Any]] = []
        for idx, cfg in enumerate(candidates):
            torch.manual_seed(seed + idx)
            model = DiscreteQuantumBackbone(cfg).to(x_calib.device)
            scores = compute_qnas_proxy_score(model, x_calib, y_calib)
            ranked_history.append(
                {
                    "config": cfg,
                    "cell_ops": cfg.cell_ops,
                    **scores,
                }
            )

        ranked_history.sort(key=lambda d: d["total_score"], reverse=True)
        return ranked_history[0]["config"], ranked_history


def _rank_array(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    return ranks


def spearman_rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman's rank correlation rho in pure NumPy."""
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    rx = _rank_array(np.asarray(x, dtype=np.float64))
    ry = _rank_array(np.asarray(y, dtype=np.float64))
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt((rx_c**2).sum() * (ry_c**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((rx_c * ry_c).sum() / denom)


def kendall_tau_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Kendall's rank correlation tau in pure NumPy."""
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            prod = (xa[i] - xa[j]) * (ya[i] - ya[j])
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2.0
    return float((concordant - discordant) / max(total_pairs, 1.0))
