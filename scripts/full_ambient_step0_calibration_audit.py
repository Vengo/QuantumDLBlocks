#!/usr/bin/env python3
"""Step-0 Inner-Product & Calibration Audit CLI for Full-Ambient vs. Parity-Split Qubits (Plan 033).

Promoted from `scripts/BACKLOG.md` (identified in Plan 032, implemented in Plan 033).
Audits Step-0 inner-product distributions (`c_sharp` mean/std/max, `c_diff` mean/std/max,
`max_self_overlap`, `self_register_overlap_max`, `directed_bigram_overlap_ratio`, and
`Step 0 Val CE / ECE`) across `(Q, A, use_ortho_decouple, decouple_self_in_full_ambient)`
configurations in `< 5s` on CPU or CUDA.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.calibration_probe import compute_calibration_rank_decomposition
from qdlblocks.benchmarks.config_models import initialize_povm_log_prior
from qdlblocks.layers.hierarchical_qubit import initialize_hierarchical_16qubit_with_ppmi_svd
from qdlblocks.models.born_gpt import QuantumBornGPT


@dataclass(frozen=True)
class Step0AuditConfig:
    """Configuration for a Step-0 inner-product and calibration audit."""

    embed_dim: int = 256
    qubit_registers: int = 16
    qubit_amplitudes: int = 16
    num_higher_states: int = 64
    use_ortho_decouple: bool = False
    decouple_self_in_full_ambient: bool = True
    device: str = "cpu"
    out_json: str | None = None


def build_synthetic_bigram_corpus(
    vocab_size: int = 512,
    num_tokens: int = 16384,
    seed: int = 2033,
) -> tuple[np.ndarray, torch.Tensor]:
    """Generate a deterministic Zipfian + directed-bigram token stream and spectral basis."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
    probs = (1.0 / (ranks**1.05))
    probs = probs / probs.sum()

    base_tokens = torch.multinomial(probs, num_tokens, replacement=True, generator=gen)
    for idx in range(1, num_tokens, 2):
        prev_tok = int(base_tokens[idx - 1].item())
        base_tokens[idx] = (prev_tok * 7 + 3) % min(128, vocab_size)

    tokens_np = base_tokens.numpy().astype(np.uint16)
    ppmi_basis = torch.randn((vocab_size, 512), generator=gen, dtype=torch.float32)
    ppmi_basis = torch.nn.functional.normalize(ppmi_basis, p=2.0, dim=-1)
    return tokens_np, ppmi_basis


def build_audit_model(
    cfg: Step0AuditConfig,
    vocab_size: int = 512,
    seed: int = 2033,
) -> QuantumBornGPT:
    """Construct a compact 2-layer `QuantumBornGPT` matching `cfg` for Step-0 calibration auditing."""
    torch.manual_seed(seed)
    dev = torch.device(cfg.device)
    num_heads = 4 if cfg.embed_dim < 512 else 8
    return QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        seed_dim=cfg.embed_dim,
        num_heads=num_heads,
        num_layers=2,
        block_size=128,
        readout_mode="asymmetric",
        povm_temperature=0.030,
        use_overcomplete_two_beam=True,
        frame_expansion=1.0,
        use_purified_attn=True,
        true_layer0_v_residual=True,
        use_gated_spherical_residual=True,
        use_dynamic_povm_beta=True,
        use_povm_log_prior=True,
        use_ortho_decouple=cfg.use_ortho_decouple,
        use_hierarchical_16qubit=True,
        qubit_registers=cfg.qubit_registers,
        qubit_amplitudes=cfg.qubit_amplitudes,
        num_higher_states=cfg.num_higher_states,
        use_block_hierarchy_bridge=True,
        bridge_causal_memory=False,
        use_two_beam_kboson_readout=True,
        diffuse_rank=16,
        diffuse_schedule="warmup_hold_anneal",
        use_stereographic_lift=True,
        quadratic_diffuse_fidelity=0.50,
        kraus_spectral_floor=6.5,
        num_value_embeddings=2,
        zero_half_rope=True,
        window_size=64,
    ).to(dev)


@torch.no_grad()
def audit_single_step0_config(
    cfg: Step0AuditConfig,
    tokens_np: np.ndarray,
    ppmi_basis: torch.Tensor,
    vocab_size: int = 512,
) -> dict[str, Any]:
    """Run Step-0 PPMI-SVD initialization and measure `c_sharp`, `c_diff`, overlaps, Val CE, and ECE."""
    dev = torch.device(cfg.device)
    flat_dim = cfg.qubit_registers * cfg.qubit_amplitudes
    model = build_audit_model(cfg, vocab_size=vocab_size)
    prior_t = torch.from_numpy(tokens_np.astype(np.int64)).to(dev)
    initialize_povm_log_prior(model, prior_t, alpha=0.62, bose_saturate_top_k=16)

    init_stats = initialize_hierarchical_16qubit_with_ppmi_svd(
        model,
        ppmi_embeddings=ppmi_basis[:, :flat_dim].to(dev),
        token_shards=[tokens_np],
        alpha=0.75,
        kmeans_iters=4,
        decouple_self_in_full_ambient=cfg.decouple_self_in_full_ambient,
        directed_predecessor_mix=0.35 if cfg.decouple_self_in_full_ambient else 0.0,
        device=dev,
    )

    model.eval()
    seq_len = 64
    batch_size = 4
    span = batch_size * seq_len
    x = prior_t[:span].view(batch_size, seq_len)
    y = prior_t[1 : span + 1].view(batch_size, seq_len)

    psi_sharp, _ = model.forward_features(x, return_layer_states=False)
    psi_beam2 = getattr(model, "_last_psi_beam2", psi_sharp)
    embedder = model.hierarchical_qubit_embed
    readout = model.hierarchical_povm_readout
    pr_mod = getattr(model, "povm_readout", None)
    base_h = getattr(pr_mod, "povm_head", None) if pr_mod is not None else None

    psi_s_norm, psi_d_r, m_s_norm, m_d_r, _, _ = readout.get_two_beam_kboson_factors(
        embedder=embedder,
        psi_sharp=psi_sharp,
        psi_beam2=psi_beam2,
        base_povm_head=base_h,
    )
    c_sharp = psi_s_norm.reshape(-1, cfg.embed_dim).float() @ m_s_norm.float().t()
    c_diff = (psi_d_r.reshape(-1, psi_d_r.size(-1)).float() @ m_d_r.float().t()).clamp(-0.99, 0.99)

    logits = model.compute_logits(psi_sharp)
    cal = compute_calibration_rank_decomposition(
        logits_2d=logits.reshape(-1, vocab_size),
        targets_1d=y.reshape(-1),
        valid_vocab_size=vocab_size,
    )

    mode_tag = (
        "Parity-Split"
        if cfg.use_ortho_decouple
        else ("Full-Ambient J_anti" if cfg.decouple_self_in_full_ambient else "Full-Ambient Legacy")
    )
    return {
        "mode": mode_tag,
        "config": asdict(cfg),
        "q_x_a": f"{cfg.qubit_registers}x{cfg.qubit_amplitudes}={flat_dim}",
        "embed_dim": cfg.embed_dim,
        "c_sharp_mean": round(float(c_sharp.mean().item()), 5),
        "c_sharp_std": round(float(c_sharp.std().item()), 5),
        "c_sharp_max": round(float(c_sharp.abs().max().item()), 5),
        "c_diff_mean": round(float(c_diff.mean().item()), 5),
        "c_diff_std": round(float(c_diff.std().item()), 5),
        "c_diff_max": round(float(c_diff.abs().max().item()), 5),
        "max_self_overlap": round(float(init_stats["max_self_overlap"]), 6),
        "self_register_overlap_max": round(float(init_stats["self_register_overlap_max"]), 6),
        "directed_bigram_overlap_ratio": round(float(init_stats["directed_bigram_overlap_ratio"]), 4),
        "step0_val_ce": round(float(cal["total_ce"]), 4),
        "step0_ece": round(float(cal["ece"]), 4),
    }


def run_step0_audit_table(base_cfg: Step0AuditConfig) -> dict[str, Any]:
    """Run the canonical 4-row Step-0 calibration comparison table and optionally save JSON."""
    t0 = time.perf_counter()
    tokens_np, ppmi_basis = build_synthetic_bigram_corpus(vocab_size=512, num_tokens=16384)
    half_a = max(4, base_cfg.qubit_amplitudes // 2)

    sweep = [
        Step0AuditConfig(
            embed_dim=base_cfg.embed_dim,
            qubit_registers=base_cfg.qubit_registers,
            qubit_amplitudes=half_a,
            num_higher_states=base_cfg.num_higher_states,
            use_ortho_decouple=True,
            decouple_self_in_full_ambient=False,
            device=base_cfg.device,
        ),
        Step0AuditConfig(
            embed_dim=base_cfg.embed_dim,
            qubit_registers=base_cfg.qubit_registers,
            qubit_amplitudes=base_cfg.qubit_amplitudes,
            num_higher_states=base_cfg.num_higher_states,
            use_ortho_decouple=False,
            decouple_self_in_full_ambient=False,
            device=base_cfg.device,
        ),
        Step0AuditConfig(
            embed_dim=base_cfg.embed_dim,
            qubit_registers=base_cfg.qubit_registers,
            qubit_amplitudes=base_cfg.qubit_amplitudes,
            num_higher_states=base_cfg.num_higher_states,
            use_ortho_decouple=False,
            decouple_self_in_full_ambient=True,
            device=base_cfg.device,
        ),
    ]

    rows = [audit_single_step0_config(c, tokens_np, ppmi_basis, vocab_size=512) for c in sweep]
    elapsed_s = round(time.perf_counter() - t0, 3)

    print("\n### Step-0 Full-Ambient vs. Parity-Split Calibration Audit Table")
    print(
        "| Mode | QxA | d | c_sharp (std/max) | c_diff (std/max) | Self-Reg Ov | Dir Bigram Ratio | Step-0 CE | Step-0 ECE |"
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        print(
            f"| {r['mode']} | `{r['q_x_a']}` | `{r['embed_dim']}` | "
            f"`{r['c_sharp_std']:.4f}` / `{r['c_sharp_max']:.3f}` | "
            f"`{r['c_diff_std']:.4f}` / `{r['c_diff_max']:.3f}` | "
            f"`{r['self_register_overlap_max']:.6f}` | "
            f"`{r['directed_bigram_overlap_ratio']:.2f}x` | "
            f"`{r['step0_val_ce']:.4f}` | `{r['step0_ece']:.4f}` |"
        )

    report = {"elapsed_sec": elapsed_s, "rows": rows}
    if base_cfg.out_json:
        out_p = Path(base_cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> Step0AuditConfig:
    """Parse CLI arguments into a frozen `Step0AuditConfig`."""
    p = argparse.ArgumentParser(description="Audit Step-0 inner-product and calibration distributions.")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--qubit-registers", type=int, default=16)
    p.add_argument("--qubit-amplitudes", type=int, default=16)
    p.add_argument("--num-higher-states", type=int, default=64)
    p.add_argument("--use-ortho-decouple", action="store_true")
    p.add_argument("--no-decouple-self", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default=None)
    ns = p.parse_args(argv)
    return Step0AuditConfig(
        embed_dim=ns.embed_dim,
        qubit_registers=ns.qubit_registers,
        qubit_amplitudes=ns.qubit_amplitudes,
        num_higher_states=ns.num_higher_states,
        use_ortho_decouple=bool(ns.use_ortho_decouple),
        decouple_self_in_full_ambient=not bool(ns.no_decouple_self),
        device=ns.device,
        out_json=ns.out_json,
    )


if __name__ == "__main__":
    run_step0_audit_table(parse_args())
