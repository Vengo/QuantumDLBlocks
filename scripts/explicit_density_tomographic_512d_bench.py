r"""CLI Benchmark Runner for `EXPERIMENT_PLAN_083.md` (`Wave 63`): Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density-Matrix Highway & `512D` Tomographic Readout (`T0..T4`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (
    DEFAULT_TOMO_NUM_LAYERS,
    ExplicitDensityTomographicSpec,
    ExplicitDensityVariantMode,
    _train_and_eval_explicit_density_variants,
    run_explicit_density_tomographic_ablation_benchmark,
    run_explicit_density_tomographic_scaling_benchmark,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_BATCH_SIZE,
    DEFAULT_SHALLOW_SEQ_LEN,
    DEFAULT_SHALLOW_VAL_TOKENS,
    LongSeqFineWebStream,
)


def _resolve_requested_variants(variant_filter: str) -> list[tuple[str, ExplicitDensityTomographicSpec, str]]:
    """Build filtered variant list from comma-separated IDs (`T0,T1,T2,T3`)."""
    all_variants = [
        (
            "T0_PURE_WAVEPACKET_RANK1_BASELINE",
            ExplicitDensityTomographicSpec(
                variant_mode=ExplicitDensityVariantMode.T0_PURE_WAVEPACKET_RANK1_BASELINE,
                enable_density_highway=False,
                enable_tomographic_readout=False,
                enable_noclick_and_vacuum=False,
                enable_two_register_bilinear=False,
                num_layers=DEFAULT_TOMO_NUM_LAYERS,
            ),
            "Pure Rank-1 Complex Wavepacket (H=16xC16, Single 512D GEMM, Zero Density Highway, Zero Vacuum)",
        ),
        (
            "T1_EXPLICIT_OUTER_DENSITY_TOMOGRAPHIC_512D",
            ExplicitDensityTomographicSpec(
                variant_mode=ExplicitDensityVariantMode.T1_EXPLICIT_OUTER_DENSITY_TOMOGRAPHIC_512D,
                enable_density_highway=True,
                enable_tomographic_readout=True,
                enable_noclick_and_vacuum=False,
                enable_two_register_bilinear=False,
                num_layers=DEFAULT_TOMO_NUM_LAYERS,
            ),
            "Explicit C^{16x16} Outer-Product Density Highway + 16x32=512D Probe Tomography Tr(rho_t E_v) Readout",
        ),
        (
            "T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM",
            ExplicitDensityTomographicSpec(
                variant_mode=ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM,
                enable_density_highway=True,
                enable_tomographic_readout=True,
                enable_noclick_and_vacuum=True,
                enable_two_register_bilinear=False,
                num_layers=DEFAULT_TOMO_NUM_LAYERS,
            ),
            "Explicit C^{16x16} Density Tomography + Trunk No-Click POVM Sink + Unified 512D Vacuum Subspace (Zero Byte-Pairs)",
        ),
        (
            "T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE",
            ExplicitDensityTomographicSpec(
                variant_mode=ExplicitDensityVariantMode.T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE,
                enable_density_highway=True,
                enable_tomographic_readout=True,
                enable_noclick_and_vacuum=True,
                enable_two_register_bilinear=True,
                num_layers=DEFAULT_TOMO_NUM_LAYERS,
            ),
            "T2 + Zero-Synthesis 256x256 Two-Register Bilinear Density Trace <phi_hi | rho_t | phi_lo> (Zero 50304x512 Re-Synthesis)",
        ),
    ]
    if not variant_filter.strip():
        return all_variants
    tokens = {tok.strip().upper() for tok in variant_filter.split(",") if tok.strip()}
    return [item for item in all_variants if item[0].upper() in tokens or item[0].split("_")[0].upper() in tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 63 Explicit Density & 512D Tomographic Readout Benchmark")
    parser.add_argument("--mode", choices=["ablation", "scaling", "both"], default="ablation")
    parser.add_argument("--variants", type=str, default="", help="Optional comma-separated filter e.g. T0,T1,T2,T3")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--scaling-steps", type=int, default=1200)
    parser.add_argument("--scaling-eval-interval", type=int, default=150)
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_SHALLOW_VAL_TOKENS)
    parser.add_argument(
        "--json-out",
        type=str,
        default=".cache/plan083_explicit_density_tomographic_results.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(
        seq_len=DEFAULT_SHALLOW_SEQ_LEN,
        batch_size=DEFAULT_SHALLOW_BATCH_SIZE,
        val_tokens=args.val_tokens,
        device=device,
    )

    payload: dict[str, Any] = {
        "plan": "EXPERIMENT_PLAN_083",
        "wave": "W63",
        "val_tokens": args.val_tokens,
    }

    if args.mode in ("ablation", "both"):
        if args.variants.strip():
            selected = _resolve_requested_variants(args.variants)
            payload["ablation_results"] = _train_and_eval_explicit_density_variants(
                stream, selected, steps=args.steps, eval_interval=args.eval_interval
            )
        else:
            payload["ablation_results"] = run_explicit_density_tomographic_ablation_benchmark(
                stream, steps=args.steps, eval_interval=args.eval_interval
            )

    if args.mode in ("scaling", "both"):
        use_reg = True
        if "ablation_results" in payload and len(payload["ablation_results"]) >= 2:
            best_ab = min(payload["ablation_results"], key=lambda r: r["final"]["val_ce"])
            use_reg = bool(best_ab.get("enable_two_register_bilinear", True))
        payload["scaling_results"] = run_explicit_density_tomographic_scaling_benchmark(
            stream,
            steps=args.scaling_steps,
            eval_interval=args.scaling_eval_interval,
            enable_two_register_bilinear=use_reg,
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Wave 63 results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
