"""Example 5: 10-Qubit (d=1024, ~124.1M Params) Quantum-Geometric Born GPT for FineWeb.

Refactored standalone script using `qdlblocks` with all the next-generation enhancements:
- 10-Qubit Hilbert space (d=1024, 16 heads x 64, 7 layers, hidden_dim=2560 -> ~124.14M params)
- Asymmetric Hilbert Lift (seed_dim=128 in -> full-rank d=1024 POVM readout out)
- StiefelMuon with Continuous Riemannian Stiefel Penalty (`stiefel_penalty=0.005`)
- Learnable per-head sharpness & POVM inverse temperature (`learnable_temperature=True`)
- Coherent reference-beam interference (`value_residual_mixing=True`, `input_highway_mixing=True`)
- Optional Complex Phase-Invariant Born Attention (`--complex-phase`)
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import time
import urllib.request

import numpy as np
import torch

from qdlblocks import (
    QuantumBornGPT,
    StiefelMuon,
    inspect_quantum_model,
    split_muon_adamw_params,
)


TRAIN_URL = (
    "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_train_000001.bin"
)
VAL_URL = (
    "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin"
)


def download_shard_slice(url: str, dest_path: str, max_bytes: int = 250_000_000) -> None:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) >= max_bytes:
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    filename = os.path.basename(dest_path)
    print(f"[Dataset] Downloading {max_bytes / 1e6:.1f} MB for {filename}...")
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    print(f"[Dataset] Cached {os.path.getsize(dest_path):,} bytes.")


def load_uint16_tokens(bin_path: str) -> np.ndarray:
    with open(bin_path, "rb") as f:
        header = f.read(1024)
        header_vals = struct.unpack("256i", header)
        magic = header_vals[0]
        f.seek(1024 if magic in (20240520, 20240327) else 0)
        return np.fromfile(f, dtype=np.uint16)


def build_10qubit_born_gpt(
    block_size: int = 1024,
    complex_phase: bool = False,
    use_flash_born: bool = True,
) -> QuantumBornGPT:
    """Instantiate the 124.14M-parameter 10-Qubit (d=1024) QuantumBornGPT model."""
    return QuantumBornGPT(
        vocab_size=50304,
        embed_dim=1024,
        seed_dim=128,
        num_heads=16,
        num_layers=7,
        hidden_dim=2560,
        block_size=block_size,
        povm_temperature=0.035,
        readout_mode="asymmetric",
        strict_spherical_residual=True,
        learnable_temperature=True,
        value_residual_mixing=True,
        input_highway_mixing=True,
        complex_phase_attention=complex_phase,
        use_flash_born=use_flash_born,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="10-Qubit QuantumBornGPT Training & Dry-Run")
    parser.add_argument("--dry-run", action="store_true", help="Run parameter summary & 2 synthetic steps")
    parser.add_argument("--data-dir", type=str, default="./data/fineweb10B")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--stiefel-penalty", type=float, default=0.005)
    parser.add_argument("--complex-phase", action="store_true")
    parser.add_argument("--no-flash-born", action="store_true", help="Disable fused Triton Flash-Born kernel")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = build_10qubit_born_gpt(
        block_size=args.block_size,
        complex_phase=args.complex_phase,
        use_flash_born=not args.no_flash_born,
    ).to(device)

    muon_params, adamw_params = split_muon_adamw_params(raw_model)
    muon_count = sum(p.numel() for p in muon_params)
    adamw_count = sum(p.numel() for p in adamw_params)
    total_count = muon_count + adamw_count

    print("==========================================================================")
    print("   10-Qubit (d = 1024) Quantum-Geometric Born GPT (qdlblocks)")
    print("==========================================================================")
    print(f"[Model] Total Parameters:         {total_count:,} (Target ~124.1M)")
    print(f"[Model] Active Muon 2D Isometries: {muon_count:,} ({muon_count / total_count * 100:.1f}%)")
    print(f"[Model] AdamW Head & Embeddings:   {adamw_count:,} ({adamw_count / total_count * 100:.1f}%)")

    optimizer_muon = StiefelMuon(
        muon_params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
        decay_mode="smooth",
        stiefel_penalty=args.stiefel_penalty,
    )
    optimizer_adamw = torch.optim.AdamW(
        adamw_params, lr=0.003, weight_decay=1e-4, betas=(0.9, 0.95)
    )

    if args.dry_run:
        print("\n[Dry Run] Executing 2 synthetic optimization steps & telemetry check...")
        seq_len = min(64, args.block_size)
        xb = torch.randint(0, 50304, (2, seq_len), device=device)
        yb = torch.randint(0, 50304, (2, seq_len), device=device)
        for s in range(1, 3):
            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adamw.zero_grad(set_to_none=True)
            _, loss = raw_model(xb, yb)
            loss.backward()
            optimizer_muon.step()
            optimizer_adamw.step()
            print(f"  Step {s}: loss = {loss.item():.4f}")
        inspect_quantum_model(raw_model, xb, yb, print_report=True)
        return

    assert torch.cuda.is_available(), "CUDA GPU required for full FineWeb training (use --dry-run on CPU)."
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_bin = os.path.join(args.data_dir, "train_slice_100M.bin")
    val_bin = os.path.join(args.data_dir, "val_slice_10M.bin")
    download_shard_slice(TRAIN_URL, train_bin, max_bytes=250_000_000)
    download_shard_slice(VAL_URL, val_bin, max_bytes=25_000_000)

    train_data = torch.from_numpy(load_uint16_tokens(train_bin).astype(np.int64))
    val_data = torch.from_numpy(load_uint16_tokens(val_bin).astype(np.int64))

    sched_muon = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_muon, T_max=args.max_steps, eta_min=0.002
    )
    sched_adamw = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adamw, T_max=args.max_steps, eta_min=3e-4
    )
    model = torch.compile(raw_model)

    def sample_batch(token_tensor: torch.Tensor, bsz: int) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(token_tensor) - args.block_size, (bsz,))
        x = torch.stack([token_tensor[i : i + args.block_size] for i in ix])
        y = torch.stack([token_tensor[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    t0 = time.time()
    tokens_processed = 0

    for step in range(1, args.max_steps + 1):
        optimizer_muon.zero_grad(set_to_none=True)
        optimizer_adamw.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.accum_steps):
            xb, yb = sample_batch(train_data, args.batch_size)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, loss = model(xb, yb)
                loss_scaled = loss / args.accum_steps
            loss_scaled.backward()
            accum_loss += loss.item() / args.accum_steps
            tokens_processed += args.batch_size * args.block_size

        torch.nn.utils.clip_grad_norm_(adamw_params, max_norm=1.0)
        optimizer_muon.step()
        optimizer_adamw.step()
        sched_muon.step()
        sched_adamw.step()

        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / max(elapsed, 1e-5)
            vx, vy = sample_batch(val_data, 8)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                report = inspect_quantum_model(raw_model, vx, vy, print_report=False)
            l_stats = report["logit_stats"]
            val_loss = l_stats["eval_cross_entropy"]
            val_ppl = math.exp(min(val_loss, 20.0))
            print(
                f"Step {step:5d}/{args.max_steps} | Train: {accum_loss:.4f} | Val: {val_loss:.4f} "
                f"| PPL: {val_ppl:.2f} | Top-1: {l_stats['top1_accuracy']:.2f}% "
                f"| MedRank: {l_stats['median_target_rank']} "
                f"| IsoErr: {report['mean_isometry_error']:.4f} | {tok_per_sec:.0f} tok/s"
            )


if __name__ == "__main__":
    main()
