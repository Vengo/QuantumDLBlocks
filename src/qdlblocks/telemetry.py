"""Quantum-Geometric Model Telemetry & Diagnostic Inspector."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .functional import bipartite_entanglement_entropy
from .metrics import (
    attention_entropy_and_sparsity,
    effective_rank,
    logit_margin_stats,
    model_isometry_error,
    model_quantum_scale_telemetry,
)


@torch.no_grad()
def inspect_quantum_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    print_report: bool = True,
) -> dict[str, Any]:
    """Extract deep geometric, spectral, and information-theoretic telemetry from a quantum model.

    Probes six core diagnostic dimensions:
    1. **Stiefel Isometry Error**: Deviation from orthogonality `||W^T W - I||_F / sqrt(d)`
       across 2D isometric weight matrices.
    2. **Effective State Rank**: Singular value entropy effective rank `exp(-sum p_i ln p_i)`
       of intermediate layer activations (detecting hyperspherical dimensional collapse).
    3. **Bipartite von Neumann Entanglement Entropy**: Layer-wise reduced density matrix
       entanglement entropy `S(rho_A) = -Tr(rho_A log rho_A)` across the `(n//2) + (n - n//2)`
       qubit bipartition (when `embed_dim = 2^n`).
    4. **Attention Entropy & Sparsity**: Shannon entropy `H(A)` and exact zero fraction
       from positive-cone Born overlaps `max(0, S)^2`.
    5. **Quantum Scale Telemetry**: Per-head Born attention scales `gamma_h^(l)`, POVM readout
       inverse temperature `beta = 1/tau`, and Mach-Zehnder U-Net interferometry gate scales.
    6. **Logit Margin & Target Rank Spectrum**: Top-1/Top-2 margin, Median Target Token Rank,
       Top-1 accuracy, and Top-5 accuracy (detecting readout rank bottlenecks).

    Args:
        model: `QuantumBornGPT` or compatible model with a `.blocks` `nn.ModuleList`.
        x: Input batch tensor (e.g., token IDs `(B, T)`).
        y: Target batch tensor (e.g., next-token IDs `(B, T)`).
        print_report: Whether to print a formatted telemetry table to stdout.

    Returns:
        Structured dictionary containing all telemetry metrics.
    """
    was_training = model.training
    model.eval()
    raw = getattr(model, "_orig_mod", model)

    # 1. Stiefel Isometry Error across Muon-optimized 2D Layers
    mean_iso_err, layer_iso_errors = model_isometry_error(raw)

    # 2. Hook intermediate blocks for Effective Rank, Entanglement, and Attention Sparsity/Entropy
    hidden_states: dict[int, torch.Tensor] = {}
    hooks = []
    blocks = getattr(raw, "blocks", None)

    def make_activation_hook(layer_idx: int):
        def _hook(module: nn.Module, inp: Any, out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
            out_t = out[0] if isinstance(out, tuple) else out
            hidden_states[layer_idx] = out_t.detach()
        return _hook

    if blocks is not None:
        for idx, block in enumerate(blocks):
            hooks.append(block.register_forward_hook(make_activation_hook(idx)))
            attn = getattr(block, "attn", None)
            if attn is not None and hasattr(attn, "record_attention"):
                attn.record_attention = True

    try:
        out = model(x, y)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
    finally:
        for h in hooks:
            h.remove()

    effective_ranks: dict[str, float] = {}
    bipartite_entropies: dict[str, float] = {}
    attn_telemetry: dict[str, dict[str, float]] = {}

    if blocks is not None:
        for idx, block in enumerate(blocks):
            layer_key = f"Layer_{idx:02d}"
            if idx in hidden_states:
                h_act = hidden_states[idx]
                effective_ranks[layer_key] = round(effective_rank(h_act), 1)
                d_state = int(h_act.size(-1))
                if d_state >= 4 and (d_state & (d_state - 1)) == 0:
                    n_q = int(round(float(np.log2(d_state))))
                    keep_q = tuple(range(n_q // 2))
                    flat_psi = h_act.reshape(-1, d_state)[:128].float()
                    ent_vals = bipartite_entanglement_entropy(
                        flat_psi, num_qubits=n_q, keep_qubits=keep_q
                    )
                    bipartite_entropies[layer_key] = round(float(ent_vals.mean().item()), 4)
            attn = getattr(block, "attn", None)
            if attn is not None and getattr(attn, "last_born_weights", None) is not None:
                attn_telemetry[layer_key] = attention_entropy_and_sparsity(attn.last_born_weights)
                attn.record_attention = False
                attn.last_born_weights = None

    # 3. Logit Margin and Output Bottleneck Diagnostics + Quantum Scales
    l_stats = logit_margin_stats(logits, y)
    q_scales = model_quantum_scale_telemetry(raw)

    mean_attn_sparsity = (
        float(np.mean([v["sparsity"] for v in attn_telemetry.values()])) * 100.0
        if attn_telemetry
        else 0.0
    )
    mean_attn_entropy = (
        float(np.mean([v["entropy"] for v in attn_telemetry.values()]))
        if attn_telemetry
        else 0.0
    )
    mean_bipartite_ent = (
        round(float(np.mean(list(bipartite_entropies.values()))), 4)
        if bipartite_entropies
        else 0.0
    )

    report: dict[str, Any] = {
        "mean_isometry_error": mean_iso_err,
        "layer_isometry_errors": layer_iso_errors,
        "layer_effective_ranks": effective_ranks,
        "layer_bipartite_entropies": bipartite_entropies,
        "mean_bipartite_entropy_nats": mean_bipartite_ent,
        "quantum_scales": q_scales,
        "layer_attention_telemetry": attn_telemetry,
        "mean_attention_sparsity_pct": mean_attn_sparsity,
        "mean_attention_entropy_nats": mean_attn_entropy,
        "logit_stats": l_stats,
    }

    if print_report:
        embed_dim = getattr(raw, "embed_dim", logits.size(-1))
        print("\n" + "=" * 68)
        print("         QUANTUM-GEOMETRIC MODEL TELEMETRY REPORT")
        print("=" * 68)
        print(f"Validation Loss:          {l_stats['eval_cross_entropy']:.4f}")
        print(f"Top-1 Token Accuracy:     {l_stats['top1_accuracy']:.2f}%")
        print(f"Top-5 Token Accuracy:     {l_stats['top5_accuracy']:.2f}%")
        print(f"Median Target Token Rank: {l_stats['median_target_rank']}")
        print(f"Mean Logit Margin:        {l_stats['mean_top1_top2_margin']:.2f}")
        print(f"Mean Isometry Error:      {mean_iso_err:.4f}")
        print(
            f"Born Scale gamma_h Mean:  {q_scales['gamma_head_mean']:.2f} "
            f"(POVM beta: {q_scales['povm_beta']:.2f})"
        )
        if bipartite_entropies:
            print(f"Mean Bipartite S(rho_A):  {mean_bipartite_ent:.4f} nats")
        if attn_telemetry:
            print(
                f"Born Attention Sparsity:  {mean_attn_sparsity:.1f}% zeros "
                f"(Entropy: {mean_attn_entropy:.2f} nats)"
            )
        if effective_ranks:
            print("-" * 68)
            print(f"Effective Rank across Layers (Capacity = {embed_dim}):")
            scale_div = max(1.0, embed_dim / 30.0)
            for layer_name, rank_val in effective_ranks.items():
                bar = "█" * max(1, int(rank_val / scale_div))
                ent_suffix = (
                    f"  [S(rho_A)={bipartite_entropies[layer_name]:.3f} nats]"
                    if layer_name in bipartite_entropies
                    else ""
                )
                print(f"  {layer_name}: {rank_val:6.1f} / {embed_dim:<4d}  {bar}{ent_suffix}")
        print("=" * 68)

    if was_training:
        model.train()

    return report

