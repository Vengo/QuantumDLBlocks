#!/usr/bin/env python3
"""Architecture Parameter Breakdown & 2-Step Smoke-Test Inspector (`scripts/inspect_model.py`).

Replaces ad-hoc `python -c` snippets used by planning and implementation agents to:
1. Count total and module-group parameters (`seed_proj`, `interference_frames`, `mlp`,
   `attention`, `lm_head`, `purity_gate`, `other`) for `QuantumBornGPT` or `ClassicalModdedGPT`.
2. Compare a preset or custom model configuration against the 124M / 162M NanoGPT budgets.
3. Run an optional N-step forward+backward synthetic smoke test on CUDA/CPU and report
   throughput (`tok/s`), peak VRAM (`MB`), and logit-norm variation metrics (`CV( ||l||_2 )`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.models.born_gpt import QuantumBornGPT


def categorize_parameters(model: torch.nn.Module) -> dict[str, int]:
    buckets = {
        "seed_and_embed": 0,
        "attention": 0,
        "interference_frames": 0,
        "mlp": 0,
        "readout_and_adapters": 0,
        "purity_and_gates": 0,
        "other": 0,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if (
            ("seed" in name and "adapter" not in name)
            or "token_embed" in name
            or "wte" in name
            or "sparse_qubit_embed" in name
            or "hierarchical_qubit_embed" in name
            or "hierarchical_embed" in name
            or "ngram_embed" in name
            or "value_embeds" in name
        ):
            buckets["seed_and_embed"] += n
        elif "attn" in name or "qkv" in name or "c_proj" in name or "out_proj" in name:
            buckets["attention"] += n
        elif "frame" in name or "interf" in name:
            buckets["interference_frames"] += n
        elif "sparse_density_layer" in name or "mlp" in name or "fc" in name or ("beam" in name and "adapter" not in name):
            buckets["mlp"] += n
        elif (
            "adapter" in name
            or "lm_head" in name
            or "povm" in name
            or "readout" in name
            or "hierarchical_povm_readout" in name
            or "hierarchical_readout" in name
            or "kraus" in name
            or "kboson" in name
            or "diffuse" in name
        ):
            buckets["readout_and_adapters"] += n
        elif (
            "purity" in name
            or "beta" in name
            or "gate" in name
            or "stereo" in name
            or "phase" in name
            or "lambdas" in name
            or "coherence_tau" in name
            or "mudd" in name
            or "smear" in name
        ):
            buckets["purity_and_gates"] += n
        else:
            buckets["other"] += n
    buckets["total"] = sum(buckets.values())
    return buckets


def build_model_from_args(args: argparse.Namespace) -> torch.nn.Module:
    if args.architecture == "classical_modded":
        from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT

        return ClassicalModdedGPT(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            max_seq_len=args.seq_len,
            zero_init_residual_and_head=args.zero_init_classical,
        )

    eff_embed_dim = 2048 if args.sparse_64qubit and args.embed_dim == 1024 else args.embed_dim
    kwargs: dict[str, Any] = dict(
        vocab_size=args.vocab_size,
        seed_dim=args.seed_dim,
        embed_dim=eff_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        readout_mode=args.readout_mode,
        use_two_beam_mlp=args.two_beam_mlp or args.overcomplete_two_beam or args.overcomplete_frame,
        use_overcomplete_two_beam=args.overcomplete_two_beam,
        frame_expansion=args.frame_expansion,
        use_overcomplete_seed_adapter=args.overcomplete_seed_adapter,
        use_two_beam_seed_adapter=args.two_beam_seed_adapter,
        depth_scaled_residual=args.depth_scaled_residual,
        use_dynamic_povm_beta=args.dynamic_povm_beta,
        use_ortho_decouple=args.ortho_decouple,
        use_povm_log_prior=args.povm_log_prior,
        use_gated_spherical_residual=args.gated_spherical_residual,
        use_purified_attn=args.purified_born_attn,
        true_layer0_v_residual=args.true_layer0_v_residual,
        unet_skip_mode=args.unet_skip_mode,
        use_fused_qkv=args.fused_qkv,
        num_value_embeddings=args.num_value_embeddings,
        zero_half_rope=args.zero_half_rope,
        use_ngram_hash=args.ngram_hash,
        ngram_vocab_size=args.ngram_vocab_size,
        use_mudd_and_gates=args.mudd_and_gates,
        use_smear_gate=args.smear_gate,
        use_sparse_64qubit=args.sparse_64qubit,
        num_qubits=args.num_qubits,
        sparse_ngram_k=args.sparse_ngram_k,
        use_sparse_density_layer=args.sparse_density_layer,
        density_matrix_rank=args.density_matrix_rank,
        use_fock_coupler=not args.no_fock_coupler,
        use_hierarchical_16qubit=args.hierarchical_16qubit,
        qubit_registers=args.qubit_registers,
        qubit_amplitudes=args.qubit_amplitudes,
        num_higher_states=args.num_higher_states,
        use_block_hierarchy_bridge=not args.no_block_hierarchy_bridge,
        bridge_causal_memory=not args.no_bridge_causal_memory,
        use_last_layer_softmax=args.last_layer_softmax,
        use_two_beam_kboson_readout=args.two_beam_kboson_readout,
        diffuse_rank=args.diffuse_rank,
        diffuse_schedule=args.diffuse_schedule,
        lambda_diff_min=args.lambda_diff_min,
        lambda_diff_max=args.lambda_diff_max,
        lambda_diff_start=args.lambda_diff_start,
        lambda_diff_end=args.lambda_diff_end,
        contextual_purity_gain=args.contextual_purity_gain,
        use_stereographic_lift=args.stereographic_lift,
        use_cosine_phase_beam_gate=args.cosine_phase_beam_gate,
        quadratic_diffuse_fidelity=args.quadratic_diffuse_fidelity,
        use_bipolar_kboson=args.bipolar_kboson,
        bipolar_hole_weight=args.bipolar_hole_weight,
        logit_softcap=args.logit_softcap,
    )
    model = QuantumBornGPT(**kwargs)
    if args.overcomplete_frame and not args.overcomplete_two_beam and not args.sparse_density_layer:
        from qdlblocks.layers.overcomplete_frame import OvercompleteFrameInterference

        for blk in model.blocks:
            blk.two_beam_mlp = OvercompleteFrameInterference(
                embed_dim=eff_embed_dim,
                frame_expansion=args.frame_expansion,
            )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect QuantumBornGPT / ClassicalModdedGPT parameters & smoke test.")
    parser.add_argument("--architecture", choices=["quantum_born", "classical_modded"], default="quantum_born")
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--seed-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--readout-mode", type=str, default="tied_stiefel")
    parser.add_argument("--two-beam-mlp", action="store_true")
    parser.add_argument("--overcomplete-frame", action="store_true")
    parser.add_argument("--overcomplete-two-beam", action="store_true")
    parser.add_argument("--frame-expansion", type=float, default=2.0)
    parser.add_argument("--overcomplete-seed-adapter", action="store_true")
    parser.add_argument("--two-beam-seed-adapter", action="store_true")
    parser.add_argument("--depth-scaled-residual", action="store_true")
    parser.add_argument("--dynamic-povm-beta", action="store_true")
    parser.add_argument("--ortho-decouple", action="store_true")
    parser.add_argument("--povm-log-prior", action="store_true")
    parser.add_argument("--gated-spherical-residual", action="store_true")
    parser.add_argument("--purified-born-attn", action="store_true")
    parser.add_argument("--true-layer0-v-residual", action="store_true")
    parser.add_argument("--unet-skip-mode", type=str, default="matrix")
    parser.add_argument("--fused-qkv", action="store_true")
    parser.add_argument("--num-value-embeddings", type=int, default=0)
    parser.add_argument("--zero-half-rope", action="store_true")
    parser.add_argument("--ngram-hash", action="store_true")
    parser.add_argument("--ngram-vocab-size", type=int, default=262144)
    parser.add_argument("--mudd-and-gates", action="store_true")
    parser.add_argument("--smear-gate", action="store_true")
    parser.add_argument("--sparse-64qubit", action="store_true")
    parser.add_argument("--num-qubits", type=int, default=64)
    parser.add_argument("--sparse-ngram-k", type=int, default=16)
    parser.add_argument("--sparse-density-layer", action="store_true")
    parser.add_argument("--density-matrix-rank", type=int, default=4)
    parser.add_argument("--no-fock-coupler", action="store_true")
    parser.add_argument("--hierarchical-16qubit", action="store_true")
    parser.add_argument("--qubit-registers", type=int, default=16)
    parser.add_argument("--qubit-amplitudes", type=int, default=32)
    parser.add_argument("--num-higher-states", type=int, default=256)
    parser.add_argument("--no-block-hierarchy-bridge", action="store_true")
    parser.add_argument("--no-bridge-causal-memory", "--embed-only-causal-memory", dest="no_bridge_causal_memory", action="store_true")
    parser.add_argument("--last-layer-softmax", action="store_true")
    parser.add_argument("--two-beam-kboson-readout", "--two-beam-kboson", dest="two_beam_kboson_readout", action="store_true")
    parser.add_argument("--diffuse-rank", "--diffuse-bath-rank", dest="diffuse_rank", type=int, default=16)
    parser.add_argument("--diffuse-schedule", type=str, default="free_decay")
    parser.add_argument("--lambda-diff-min", type=float, default=0.08)
    parser.add_argument("--lambda-diff-max", type=float, default=0.35)
    parser.add_argument("--lambda-diff-start", type=float, default=0.22)
    parser.add_argument("--lambda-diff-end", type=float, default=0.12)
    parser.add_argument("--contextual-purity-gain", type=float, default=0.20)
    parser.add_argument("--stereographic-lift", action="store_true")
    parser.add_argument("--cosine-phase-beam-gate", action="store_true")
    parser.add_argument("--quadratic-diffuse-fidelity", "--quad-diffuse-weight", dest="quadratic_diffuse_fidelity", nargs="?", const=0.50, type=float, default=0.0)
    parser.add_argument("--bipolar-kboson", action="store_true")
    parser.add_argument("--bipolar-hole-weight", type=float, default=0.85)
    parser.add_argument("--logit-softcap", type=float, default=15.0)
    parser.add_argument("--zero-init-classical", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Run N forward+backward steps on CUDA/CPU.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    args = parser.parse_args()

    model = build_model_from_args(args)
    buckets = categorize_parameters(model)

    smoke_metrics: dict[str, Any] = {}
    if args.smoke_steps > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        losses = []
        logit_norm_cv = 0.0
        for _ in range(args.smoke_steps):
            x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
            y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                norms = logits.detach().float().norm(dim=-1)
                logit_norm_cv = float(norms.std() / norms.mean().clamp_min(1e-8))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        total_tok = args.smoke_steps * args.batch_size * args.seq_len
        smoke_metrics = {
            "device": str(device),
            "steps": args.smoke_steps,
            "losses": [round(v, 4) for v in losses],
            "tok_per_s": round(total_tok / max(dt, 1e-6), 1),
            "logit_l2_norm_cv": round(logit_norm_cv, 6),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 1) if device.type == "cuda" else 0.0,
        }

    result = {
        "architecture": args.architecture,
        "total_params": buckets["total"],
        "total_params_m": round(buckets["total"] / 1e6, 3),
        "breakdown": buckets,
        "smoke_test": smoke_metrics,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"=== Model Inspection: {args.architecture} ({result['total_params_m']:.2f}M params) ===")
        for k, v in buckets.items():
            if k != "total":
                pct = 100.0 * v / max(buckets["total"], 1)
                print(f"  {k:<24}: {v:>12,} ({pct:5.1f}%)")
        print(f"  {'TOTAL':<24}: {buckets['total']:>12,} (100.0%)")
        if smoke_metrics:
            print(f"\n=== Smoke Test ({smoke_metrics['steps']} steps on {smoke_metrics['device']}) ===")
            print(f"  Losses         : {smoke_metrics['losses']}")
            print(f"  Throughput     : {smoke_metrics['tok_per_s']:,.1f} tok/s")
            print(f"  Logit Norm CV  : {smoke_metrics['logit_l2_norm_cv']:.6f}")
            print(f"  Peak VRAM (MB) : {smoke_metrics['peak_vram_mb']:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
