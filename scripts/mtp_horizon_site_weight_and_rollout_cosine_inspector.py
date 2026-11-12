#!/usr/bin/env python3
"""Inspect Horizon-Pooled Causal MTP MPS Weights & Future Rollout Cosines (`Wave 79` / `EXPERIMENT_PLAN_099.md`).

Promoted from `scripts/TOOLING_BACKLOG.md` (`Wave 78` candidate). Loads a saved `Wave 79` `.pt`
checkpoint (or runs a quick FineWeb calibration sprint) and reports learned horizon pooling weights
$w_0..w_{K-1}$, per-site Schur coincidence $p_{\\text{mps}, k}$, and smoothed vs unsmoothed future
cosine alignment $\\cos(\\tilde{\\phi}_{t,k}, x_{t+k})$ across $k \\in \\{1..K\\}$.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.architecture_xray_diagnostics import (  # noqa: E402
    ShortTrainConfig,
    load_fineweb_token_slice,
)
from qdlblocks.benchmarks.wave79_fused_schur_and_induction_baton import (  # noqa: E402
    Wave79VariantKind,
    build_wave79_variant_adapter,
)
from qdlblocks.benchmarks.wave79_speedrun_and_ablation_runner import (  # noqa: E402
    inspect_mtp_horizon_weights_and_rollout_cosines,
    run_wave79_controlled_sprint,
)

DEFAULT_JSON_OUT: Path = REPO_ROOT / ".cache" / "wave79_mtp_horizon_inspector.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a saved Wave 79 `.pt` checkpoint.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=Wave79VariantKind.V2_FUSED_UNIFIED_4L_D256.value,
        choices=[v.value for v in Wave79VariantKind],
    )
    parser.add_argument(
        "--sprint-steps",
        type=int,
        default=120,
        help="Calibration steps on FineWeb Shard-1 if `--checkpoint` is not provided.",
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    return parser.parse_args()


def main() -> None:
    """Execute MTP horizon pooling weight & rollout cosine inspection."""
    args = parse_args()
    device = torch.device(args.device)
    seq_len = int(args.seq_len)
    val_raw = load_fineweb_token_slice("fineweb_val_000000.bin", num_tokens=32 * seq_len + 8)
    x_val = val_raw[: 32 * seq_len].view(32, seq_len)
    y_val = val_raw[1 : 32 * seq_len + 1].view(32, seq_len)

    if args.checkpoint is not None and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        var = Wave79VariantKind(str(ckpt.get("variant", args.variant)))
        adapter = build_wave79_variant_adapter(var, device=device, enable_compile=False)
        adapter.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        source_desc = str(args.checkpoint)
    elif int(args.sprint_steps) > 0:
        var = Wave79VariantKind(args.variant)
        cfg = ShortTrainConfig(
            steps=int(args.sprint_steps),
            batch_size=32,
            seq_len=seq_len,
            eval_interval=max(20, int(args.sprint_steps)),
            eval_tokens=int(x_val.numel()),
            wave_id="W79MTP",
            plan_id="EXPERIMENT_PLAN_099",
        )
        adapter, _ = run_wave79_controlled_sprint(var, cfg, x_val, y_val, device=device)
        source_desc = f"{var.value}_sprint_{args.sprint_steps}s"
    else:
        var = Wave79VariantKind(args.variant)
        adapter = build_wave79_variant_adapter(var, device=device, enable_compile=False)
        source_desc = f"{var.value}_step0"

    report = inspect_mtp_horizon_weights_and_rollout_cosines(adapter.fused_model, x_val, y_val)
    report["source"] = source_desc
    report["variant"] = adapter.key

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== Wave 79 Causal K=4 MTP Bwd-Fwd MPS Horizon Inspector ({source_desc}) ===")
    print(f"mtp_out_proj ||W||_F = {report.get('mtp_out_proj_frobenius_norm', 0.0):.4f}")
    print(
        "| Horizon t+k | Pool w_k | MPS p_and | Smoothed Cos | Raw Cos | Persist Cos | MPS Gain ΔCos |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for h in report.get("per_horizon", []):
        print(
            f"| t+{int(h['horizon_step_ahead'])} | {h['pool_weight']:.4f} | "
            f"{h['mps_schur_p_and']:.4f} | {h['smoothed_rollout_cosine']:.4f} | "
            f"{h['unsmoothed_raw_cosine']:.4f} | {h['static_persistence_cosine']:.4f} | "
            f"{h['mps_smoothing_cosine_gain']:+.4f} |"
        )
    print(f"\nSaved JSON report -> {args.json_out}")


if __name__ == "__main__":
    main()
