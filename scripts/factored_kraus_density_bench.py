"""CLI benchmark runner for Wave 70 (`EXPERIMENT_PLAN_090.md`): Factored Cross-Head Kraus Pure Density-Matrix GPT."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import torch

from qdlblocks.benchmarks.factored_kraus_density_gpt_lab import (
    Wave70VariantMode,
    build_wave70_variant_spec,
    run_wave70_variant,
)
from qdlblocks.benchmarks.micro_reversal_readout_lab import (
    DEFAULT_FINEWEB_DIR,
    DEFAULT_VAL_SHARD_PATH,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream

_HF_FINEWEB_BASE_URL: str = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main"
_VAL_TARGET_BYTES: int = 21_103_616
_TRAIN_SHARD_BYTES: int = 200_001_024

_VARIANT_LOOKUP: dict[str, Wave70VariantMode] = {
    "W70_K0": Wave70VariantMode.W70_K0_LUT_PACKED_BASELINE_4L,
    "W70_K0_LUT_PACKED_BASELINE_4L": Wave70VariantMode.W70_K0_LUT_PACKED_BASELINE_4L,
    "W70_K1": Wave70VariantMode.W70_K1_FACTORED_KRAUS_TRICK_A_4L,
    "W70_K1_FACTORED_KRAUS_TRICK_A_4L": Wave70VariantMode.W70_K1_FACTORED_KRAUS_TRICK_A_4L,
    "W70_K2": Wave70VariantMode.W70_K2_COMMUTED_AND_MULTIBRANCH_4L,
    "W70_K2_COMMUTED_AND_MULTIBRANCH_4L": Wave70VariantMode.W70_K2_COMMUTED_AND_MULTIBRANCH_4L,
    "W70_K3": Wave70VariantMode.W70_K3_DEEP_8L_FACTORED_KRAUS,
    "W70_K3_DEEP_8L_FACTORED_KRAUS": Wave70VariantMode.W70_K3_DEEP_8L_FACTORED_KRAUS,
}


def ensure_fineweb_shards_present(num_train_shards: int = 2) -> None:
    """Ensure `fineweb_val_000000.bin` and at least `num_train_shards` training shards exist (for Colab VMs) (`<= 8` args)."""
    DEFAULT_FINEWEB_DIR.mkdir(parents=True, exist_ok=True)
    targets: list[tuple[str, Path, int]] = [
        (f"{_HF_FINEWEB_BASE_URL}/fineweb_val_000000.bin", DEFAULT_VAL_SHARD_PATH, _VAL_TARGET_BYTES)
    ]
    for idx in range(1, max(1, num_train_shards) + 1):
        name = f"fineweb_train_{idx:06d}.bin"
        targets.append((f"{_HF_FINEWEB_BASE_URL}/{name}", DEFAULT_FINEWEB_DIR / name, _TRAIN_SHARD_BYTES))

    for url, dest, n_bytes in targets:
        if dest.exists() and dest.stat().st_size >= min(n_bytes, 10_000_000):
            continue
        tmp_dest = dest.with_suffix(".bin.part")
        print(f"Downloading {dest.name} ({n_bytes / 1e6:.1f} MB) from HuggingFace FineWeb10B...", flush=True)
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n_bytes - 1}", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_dest, "wb") as f:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp_dest.replace(dest)


def resolve_wave70_variants(variants_csv: str) -> list[Wave70VariantMode]:
    """Parse comma-separated Wave 70 variant keys (`<= 8` args)."""
    modes: list[Wave70VariantMode] = []
    for token in variants_csv.split(","):
        key = token.strip().upper()
        if not key:
            continue
        if key not in _VARIANT_LOOKUP:
            raise ValueError(f"Unknown Wave 70 variant {key!r}. Valid keys: {sorted(_VARIANT_LOOKUP)}")
        modes.append(_VARIANT_LOOKUP[key])
    return modes


def main() -> None:
    """Run Wave 70 Factored Cross-Head Kraus Pure Density-Matrix GPT variants (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Wave 70 Factored Cross-Head Kraus Pure Density-Matrix GPT Benchmark")
    parser.add_argument("--variants", type=str, default="W70_K0,W70_K1,W70_K2,W70_K3", help="Comma-separated Wave 70 variants")
    parser.add_argument("--steps", type=int, default=200, help="Training steps per variant")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--eval-interval", type=int, default=25, help="Evaluation cadence in steps")
    parser.add_argument("--eval-batches", type=int, default=8, help="Number of Shard-0 validation batches")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--json-out",
        "--out-json",
        dest="json_out",
        type=str,
        default=".cache/plan090_wave70_factored_kraus_summary.json",
        help="Summary JSON path",
    )
    args = parser.parse_args()

    ensure_fineweb_shards_present(num_train_shards=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(seq_len=args.seq_len, batch_size=args.batch_size, val_tokens=65_536, device=device)
    modes = resolve_wave70_variants(args.variants)
    results = []
    for mode in modes:
        spec = build_wave70_variant_spec(mode, seed=args.seed)
        res = run_wave70_variant(
            spec,
            stream,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_interval=args.eval_interval,
        )
        results.append(res)

    payload = {
        "plan_id": "EXPERIMENT_PLAN_090",
        "wave_id": "W70",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "results": results,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Wave 70 benchmark summary to {out_path}")


if __name__ == "__main__":
    main()
