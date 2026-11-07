#!/usr/bin/env python3
"""CLI Benchmark for Multi-Branch Wavepacket Readout (`(50304, 8, 2, 32)` Stiefel Identity Frames R=4, R=8).

Runs:
1. The K-Target Identity Superposition Capacity Probe (`K in {1, 2, 4, 8, 16}` vs `R in {1, 4, 8}`),
   verifying how orthogonal identity branches overcome the `K_max = d_id / (2 ln V) ≈ 2.95`
   Johnson-Lindenstrauss single-vector superposition ceiling.
2. Full 300-step (`2.46M`-token, `B=16, T=512, L=4, d=512`) FineWeb training and unseen Shard-0
   evaluation across `M1_FIBER_STIEFEL_FRAME_R4`, `M2_COUPLED_SEM_ID_WAVEPACKETS_R4`,
   `M3_COUPLED_SEM_ID_WAVEPACKETS_R8`, and `M4_COUPLED_R8_CALIBRATED_SYNERGY`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qdlblocks.benchmarks.multibranch_wavepacket_readout_lab import (
    DEFAULT_SHALLOW_BATCH_SIZE,
    DEFAULT_SHALLOW_EMBED_DIM,
    DEFAULT_SHALLOW_NUM_HEADS,
    DEFAULT_SHALLOW_NUM_LAYERS,
    DEFAULT_SHALLOW_SEQ_LEN,
    DEFAULT_SHALLOW_VOCAB_SIZE,
    MultiBranchMode,
    MultiBranchReadoutSpec,
    MultiBranchWavepacketGPT,
     initialize_and_calibrate_shallow_model_,
     LongSeqFineWebStream,
    run_k_target_superposition_probe,
    run_multibranch_wavepacket_experiment,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Multi-Branch Wavepacket Readout benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        type=str,
        default="m1_fiber_stiefel_frame_r4,m2_coupled_sem_id_wavepackets_r4,m3_coupled_sem_id_wavepackets_r8,m4_coupled_r8_calibrated_synergy",
        help="Comma-separated MultiBranchMode values to evaluate.",
    )
    parser.add_argument("--layers", type=int, default=DEFAULT_SHALLOW_NUM_LAYERS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SHALLOW_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SHALLOW_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--skip-probe", action="store_true", help="Skip the K-target superposition probe.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=str,
        default=".cache/multibranch_wavepacket_readout_results.json",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the K-Target Superposition Probe and Multi-Branch FineWeb training benchmark."""
    args = parse_args()
    device = torch.device(args.device)
    mode_list = [MultiBranchMode(m.strip().lower()) for m in args.modes.split(",") if m.strip()]

    print(
        f"[multibranch_wavepacket_readout_bench] Initializing FineWeb stream "
        f"(B={args.batch_size}, T={args.seq_len}, L={args.layers}) on {device}..."
    )
    stream = LongSeqFineWebStream(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    payload: dict[str, Any] = {
        "script": "multibranch_wavepacket_readout_bench.py",
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "num_layers": args.layers,
        "steps": args.steps,
        "superposition_probe": {},
        "results": {},
    }

    if not args.skip_probe:
        probe_spec = MultiBranchReadoutSpec(
            mode=MultiBranchMode.M0_SINGLE_BRANCH_REF,
            vocab_size=DEFAULT_SHALLOW_VOCAB_SIZE,
            embed_dim=DEFAULT_SHALLOW_EMBED_DIM,
            num_heads=DEFAULT_SHALLOW_NUM_HEADS,
            num_layers=args.layers,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        probe_model = MultiBranchWavepacketGPT(probe_spec).to(device)
        initialize_and_calibrate_shallow_model_(probe_model, stream)
        probe_res = run_k_target_superposition_probe(probe_model.w_det_512)
        payload["superposition_probe"] = probe_res
        print(
            f"[SuperpositionProbe] Theoretical JL Ceiling: K_max(id=64D)={probe_res['theoretical_jl_k_max_id64']}, "
            f"K_max(sem=448D)={probe_res['theoretical_jl_k_max_sem448']}"
        )
        for k_key, k_dict in probe_res["by_k"].items():
            r1 = k_dict["R1"]
            r4 = k_dict["R4"]
            r8 = k_dict["R8"]
            print(
                f"  {k_key:3s} (Floor={k_dict['shannon_floor_nats']:.3f}) | "
                f"R=1: ExcessCE={r1['excess_ce_nats']:.4f} Recall={r1['topk_recall']*100:5.1f}% | "
                f"R=4: ExcessCE={r4['excess_ce_nats']:.4f} Recall={r4['topk_recall']*100:5.1f}% | "
                f"R=8: ExcessCE={r8['excess_ce_nats']:.4f} Recall={r8['topk_recall']*100:5.1f}%"
            )
        del probe_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for mode in mode_list:
        spec = MultiBranchReadoutSpec(
            mode=mode,
            vocab_size=DEFAULT_SHALLOW_VOCAB_SIZE,
            embed_dim=DEFAULT_SHALLOW_EMBED_DIM,
            num_heads=DEFAULT_SHALLOW_NUM_HEADS,
            num_layers=args.layers,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        res = run_multibranch_wavepacket_experiment(
            spec=spec,
            stream=stream,
            steps=args.steps,
            eval_every=args.eval_every,
        )
        payload["results"][mode.value] = res
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"[multibranch_wavepacket_readout_bench] Saved results to {out_path}")


if __name__ == "__main__":
    main()
