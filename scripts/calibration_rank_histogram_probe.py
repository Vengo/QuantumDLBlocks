#!/usr/bin/env python3
"""Target-Rank Cross-Entropy Decomposition & Layerwise Dynamics CLI Probe (Plan 025).

Promoted from `scripts/BACKLOG.md` (identified in Plan 024, implemented in Plan 025).
Evaluates and compares:
  1. `classical_modded`: Leaderboard-Exact `ClassicalModdedGPT`
  2. `quantum_povm`: 16-Qubit + 256-State Hierarchical `QuantumBornGPT` (Pure Unit-Sphere POVM)
  3. `quantum_softmax`: 16-Qubit + 256-State Hierarchical `QuantumBornGPT` (Calibrated Last-Layer Softmax)
  4. `quantum_w13w15_unified`: Unified Wave 13 + Wave 15 + Plan 024 `QuantumBornGPT` (`bridge_causal_memory=False`, `VE=3`, `ZeroHalfRoPE`, `DepthScaledRes`, `Win=384`, `Hashed N-Gram`, `Last-Layer Softmax`)

Outputs:
  - Markdown summary table of Validation CE, Top-1 / Top-5 / Gap, Median Rank, ECE, and Peak VRAM
  - 5-Bucket Target-Rank Cross-Entropy Decomposition table (`Rank 1`, `Ranks 2-5`, `Ranks 6-50`, `Ranks 51-1000`, `Ranks > 1000`)
  - Layerwise angular state velocity and spectral effective rank profile
  - Optional JSON report (`--out-json`)
"""

from __future__ import annotations

import argparse
import gc
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

from qdlblocks.benchmarks.calibration_probe import (
    RANK_BUCKETS,
    compute_calibration_rank_decomposition,
    compute_layerwise_dynamics_profile,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks.optim.muon import StiefelMuon, polar_snap_weights, split_muon_adamw_params


def enforce_vram_cap(max_vram_gb: float = 8.0) -> float:
    """Enforce a strict hardware VRAM cap on CUDA device 0 via `set_per_process_memory_fraction`."""
    if not torch.cuda.is_available():
        return 0.0
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0**3)
    frac = min(0.50 if max_vram_gb <= 8.0 else 0.95, max(0.05, float(max_vram_gb) / max(1.0, total_gb)))
    torch.cuda.set_per_process_memory_fraction(frac, device=0)
    return frac


def _build_synthetic_zipf_stream(
    vocab_size: int = 256,
    batch_size: int = 8,
    seq_len: int = 64,
    num_batches: int = 8,
    device: torch.device = torch.device("cpu"),
    seed: int = 2025,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
    probs = (1.0 / (ranks**1.10))
    probs = probs / probs.sum()

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_batches + 1):
        x = torch.multinomial(probs, batch_size * seq_len, replacement=True, generator=gen).view(batch_size, seq_len)
        z = torch.multinomial(probs, batch_size * seq_len, replacement=True, generator=gen).view(batch_size, seq_len)
        mask = torch.rand((batch_size, seq_len), generator=gen) < 0.55
        y = torch.where(mask, (x * 5 + 3) % min(64, vocab_size), z)
        batches.append((x.to(device), y.to(device)))
    return batches[:-1], batches[-1]


def run_calibration_probe_suite(
    steps: int = 12,
    quick_test: bool = False,
    max_vram_gb: float = 8.0,
    device_str: str | None = None,
    include_twobeam_kboson: bool = True,
    diffuse_rank: int = 16,
) -> dict[str, Any]:
    """Run controlled Classical vs. Quantum POVM vs. Quantum Softmax vs. Two-Beam k-Boson calibration rank decomposition."""
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_fraction = enforce_vram_cap(max_vram_gb) if device.type == "cuda" else 0.0

    if quick_test:
        vocab_size = 256
        valid_vocab_size = 256
        embed_dim = 64
        num_heads = 4
        num_layers = 2
        qubit_registers = 16
        qubit_amplitudes = 2
        num_higher_states = 16
        batch_size = 8
        seq_len = 32
        eff_steps = max(2, min(steps, 4))
        eff_diffuse_rank = min(8, diffuse_rank)
    else:
        vocab_size = 2048
        valid_vocab_size = 2048
        embed_dim = 256
        num_heads = 4
        num_layers = 4
        qubit_registers = 16
        qubit_amplitudes = 8
        num_higher_states = 64
        batch_size = 8
        seq_len = 128
        eff_steps = max(2, int(steps))
        eff_diffuse_rank = int(diffuse_rank)

    train_batches, (val_x, val_y) = _build_synthetic_zipf_stream(
        vocab_size=vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        num_batches=eff_steps,
        device=device,
    )

    variant_specs = [
        ("classical_modded", "ClassicalModdedGPT (Leaderboard-Exact)"),
        ("quantum_povm", "QuantumBornGPT (16-Qubit + 256-Hier Pure POVM)"),
        ("quantum_softmax", "QuantumBornGPT (16-Qubit + 256-Hier Calibrated Softmax)"),
        ("quantum_w13w15_unified", "QuantumBornGPT (W13+W15 Unified + Embed-Only Causal Mem)"),
    ]
    if include_twobeam_kboson:
        variant_specs.append(
            ("quantum_twobeam_kboson", "QuantumBornGPT (16-Qubit + 256-Hier Two-Beam k-Boson Readout)")
        )

    variants_out: dict[str, dict[str, Any]] = {}

    for v_key, v_label in variant_specs:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        torch.manual_seed(1337)
        t0 = time.perf_counter()

        if v_key == "classical_modded":
            model: nn.Module = ClassicalModdedGPT(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dim=4 * embed_dim,
                block_size=seq_len,
                leaderboard_exact=True,
                num_value_embeddings=min(2, num_layers // 2),
                logit_softcap=15.0,
            ).to(device)
        elif v_key == "quantum_povm":
            model = QuantumBornGPT(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                seed_dim=embed_dim // 2,
                num_heads=num_heads,
                num_layers=num_layers,
                block_size=seq_len,
                readout_mode="asymmetric",
                use_overcomplete_two_beam=True,
                frame_expansion=2.0,
                use_purified_attn=True,
                true_layer0_v_residual=True,
                use_gated_spherical_residual=True,
                use_dynamic_povm_beta=True,
                use_povm_log_prior=True,
                use_ortho_decouple=True,
                use_hierarchical_16qubit=True,
                qubit_registers=qubit_registers,
                qubit_amplitudes=qubit_amplitudes,
                num_higher_states=num_higher_states,
                use_block_hierarchy_bridge=True,
                bridge_causal_memory=True,
                use_last_layer_softmax=False,
            ).to(device)
        elif v_key == "quantum_softmax":
            model = QuantumBornGPT(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                seed_dim=embed_dim // 2,
                num_heads=num_heads,
                num_layers=num_layers,
                block_size=seq_len,
                readout_mode="asymmetric",
                use_overcomplete_two_beam=True,
                frame_expansion=2.0,
                use_purified_attn=True,
                true_layer0_v_residual=True,
                use_gated_spherical_residual=True,
                use_dynamic_povm_beta=True,
                use_povm_log_prior=True,
                use_ortho_decouple=True,
                use_hierarchical_16qubit=True,
                qubit_registers=qubit_registers,
                qubit_amplitudes=qubit_amplitudes,
                num_higher_states=num_higher_states,
                use_block_hierarchy_bridge=True,
                bridge_causal_memory=True,
                use_last_layer_softmax=True,
                logit_softcap=15.0,
            ).to(device)
        elif v_key == "quantum_twobeam_kboson":
            model = QuantumBornGPT(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                seed_dim=embed_dim // 2,
                num_heads=num_heads,
                num_layers=num_layers,
                block_size=seq_len,
                readout_mode="asymmetric",
                use_overcomplete_two_beam=True,
                frame_expansion=2.0,
                use_purified_attn=True,
                true_layer0_v_residual=True,
                use_gated_spherical_residual=True,
                use_dynamic_povm_beta=True,
                use_povm_log_prior=True,
                use_ortho_decouple=True,
                use_hierarchical_16qubit=True,
                qubit_registers=qubit_registers,
                qubit_amplitudes=qubit_amplitudes,
                num_higher_states=num_higher_states,
                use_block_hierarchy_bridge=True,
                bridge_causal_memory=True,
                use_last_layer_softmax=False,
                use_two_beam_kboson_readout=True,
                diffuse_rank=eff_diffuse_rank,
            ).to(device)
        else:
            model = QuantumBornGPT(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                seed_dim=embed_dim // 2,
                num_heads=num_heads,
                num_layers=num_layers,
                block_size=seq_len,
                readout_mode="asymmetric",
                use_overcomplete_two_beam=True,
                frame_expansion=2.0,
                use_purified_attn=True,
                true_layer0_v_residual=True,
                use_gated_spherical_residual=True,
                use_dynamic_povm_beta=True,
                use_povm_log_prior=True,
                use_ortho_decouple=True,
                use_hierarchical_16qubit=True,
                qubit_registers=qubit_registers,
                qubit_amplitudes=qubit_amplitudes,
                num_higher_states=num_higher_states,
                use_block_hierarchy_bridge=True,
                bridge_causal_memory=False,
                use_last_layer_softmax=True,
                logit_softcap=15.0,
                num_value_embeddings=min(2, num_layers // 2),
                zero_half_rope=True,
                depth_scaled_residual=True,
                window_size=min(384, seq_len),
                use_ngram_hash=True,
                ngram_vocab_size=min(8192, 4 * vocab_size),
            ).to(device)

        muon_params, adam_params = split_muon_adamw_params(model)
        opt_m = StiefelMuon(muon_params, lr=0.02, polar_method="polar_express") if muon_params else None
        opt_a = torch.optim.AdamW(adam_params, lr=0.008, weight_decay=0.0)

        for step_idx in range(eff_steps):
            x_b, y_b = train_batches[step_idx % len(train_batches)]
            if opt_m is not None:
                opt_m.zero_grad(set_to_none=True)
            opt_a.zero_grad(set_to_none=True)
            _, loss = model(x_b, y_b)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if opt_m is not None:
                opt_m.step()
            opt_a.step()
            if isinstance(model, QuantumBornGPT):
                polar_snap_weights(model, polar_method="polar_express")

        model.eval()
        with torch.no_grad():
            logits_val, _ = model(val_x, None)

        cal_decomp = compute_calibration_rank_decomposition(
            logits_val,
            val_y,
            valid_vocab_size=valid_vocab_size,
        )
        dyn_prof = compute_layerwise_dynamics_profile(
            model,
            val_x,
            val_y,
            valid_vocab_size=valid_vocab_size,
        )
        peak_vram_gb = (
            round(torch.cuda.max_memory_allocated(device) / 1e9, 4)
            if device.type == "cuda"
            else 0.0
        )
        wall_s = round(time.perf_counter() - t0, 3)

        variants_out[v_key] = {
            "label": v_label,
            "params": sum(p.numel() for p in model.parameters()),
            "steps": eff_steps,
            "wall_s": wall_s,
            "peak_vram_gb": peak_vram_gb,
            "calibration": cal_decomp,
            "dynamics": dyn_prof,
        }

        del model, opt_m, opt_a, logits_val
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "device": str(device),
        "max_vram_gb": max_vram_gb,
        "cuda_memory_fraction": round(mem_fraction, 4),
        "quick_test": bool(quick_test),
        "steps": eff_steps,
        "rank_buckets": [b[0] for b in RANK_BUCKETS],
        "variants": variants_out,
    }


def format_probe_markdown(report: dict[str, Any]) -> str:
    """Format the calibration rank decomposition & dynamics report as Markdown tables."""
    lines: list[str] = []
    lines.append("# Target-Rank Cross-Entropy Decomposition & Layerwise Dynamics Probe")
    lines.append("")
    lines.append(
        f"- **Device**: `{report['device']}` | **VRAM Cap**: `{report['max_vram_gb']:.2f} GB` "
        f"(`fraction={report['cuda_memory_fraction']}`) | **Steps**: `{report['steps']}`"
    )
    lines.append("")
    lines.append("## 1. Overall Calibration & Top-k Summary")
    lines.append("")
    lines.append(
        "| Variant | Val CE | Val PPL | Top-1 (%) | Top-5 (%) | `Top-5 - Top-1` Gap (%) | Med Rank | ECE | Peak VRAM (GB) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for v_key, v_data in report["variants"].items():
        c = v_data["calibration"]
        lines.append(
            f"| `{v_key}` | `{c['total_ce']:.4f}` | `{c['total_ppl']:.2f}` | `{c['top1_acc_pct']:.2f}%` | "
            f"`{c['top5_acc_pct']:.2f}%` | `{c['top5_minus_top1_gap_pct']:.2f}%` | `{c['median_rank']:.1f}` | "
            f"`{c['ece']:.4f}` | `{v_data['peak_vram_gb']:.3f}` |"
        )

    lines.append("")
    lines.append("## 2. 5-Bucket Target-Rank Cross-Entropy Contributions (`p(b) * NLL(b)`)")
    lines.append("")
    lines.append(
        "| Variant | `rank_1` CE (Share) | `ranks_2_5` CE (Share) | `ranks_6_50` CE (Share) | "
        "`ranks_51_1000` CE (Share) | `ranks_gt_1000` CE (Share) | Sum Error |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for v_key, v_data in report["variants"].items():
        c = v_data["calibration"]
        b = c["buckets"]
        lines.append(
            f"| `{v_key}` | `{b['rank_1']['ce_contribution']:.4f}` ({100*b['rank_1']['fraction']:.1f}%) | "
            f"`{b['ranks_2_5']['ce_contribution']:.4f}` ({100*b['ranks_2_5']['fraction']:.1f}%) | "
            f"`{b['ranks_6_50']['ce_contribution']:.4f}` ({100*b['ranks_6_50']['fraction']:.1f}%) | "
            f"`{b['ranks_51_1000']['ce_contribution']:.4f}` ({100*b['ranks_51_1000']['fraction']:.1f}%) | "
            f"`{b['ranks_gt_1000']['ce_contribution']:.4f}` ({100*b['ranks_gt_1000']['fraction']:.1f}%) | "
            f"`{c['ce_decomposition_error']:.2e}` |"
        )

    lines.append("")
    lines.append("## 3. Layerwise Geometry & Gradient Ratio Profile")
    lines.append("")
    lines.append(
        "| Variant | Mean Angular Vel (deg) | Final Spectral Rank $R_\\sigma(h_L)$ | "
        "Embed/Readout Grad Norm | Backbone Muon Grad Norm | Grad Ratio | Stiefel IsoErr |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for v_key, v_data in report["variants"].items():
        d = v_data["dynamics"]
        lines.append(
            f"| `{v_key}` | `{d['mean_angular_velocity_deg']:.2f}` | `{d['final_layer_effective_rank']:.2f}` | "
            f"`{d['grad_norm_embed_readout']:.4f}` | `{d['grad_norm_backbone_muon']:.4f}` | "
            f"`{d['embed_to_backbone_grad_ratio']:.3f}` | `{d['stiefel_iso_err']:.4f}` |"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose Validation Cross-Entropy across target-rank buckets and profile layerwise dynamics."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of optimization steps per model variant (default: 12).",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a fast micro-scale verification pass (<2s).",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=8.0,
        help="Strict CUDA VRAM cap in GB enforced via set_per_process_memory_fraction (default: 8.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--include-twobeam-kboson",
        action="store_true",
        default=True,
        help="Include Two-Beam k-Boson Mixed-State Readout variant (default: True).",
    )
    parser.add_argument(
        "--diffuse-rank",
        type=int,
        default=16,
        help="Rank of the Kraus diffuse semantic bath (default: 16).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to save the full JSON decomposition report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = run_calibration_probe_suite(
        steps=args.steps,
        quick_test=args.quick_test,
        max_vram_gb=args.max_vram_gb,
        device_str=args.device,
        include_twobeam_kboson=args.include_twobeam_kboson,
        diffuse_rank=args.diffuse_rank,
    )
    md = format_probe_markdown(report)
    print(md)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON calibration rank decomposition report -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
