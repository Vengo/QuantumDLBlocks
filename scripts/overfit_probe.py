#!/usr/bin/env python3
"""Karpathy Memorization Capacity & Logit-Sharpness Diagnostic CLI (`scripts/overfit_probe.py`).

Promoted from scratch to a permanent CLI diagnostic tool. Tests how accurately and sharply
candidate QuantumBornGPT variants (including `--dynamic-povm-beta`) and ClassicalModdedGPT
baselines can overfit N FineWeb sequences (512..65,536+ tokens), reporting:
- Final Training Cross-Entropy (`final_ce`)
- Next-Token Top-1 Memorization Accuracy (`top1_acc`)
- Output Probability Entropy (`mean_output_entropy_nats`)
- Mean Maximum Token Probability (`mean_max_prob`)
- Logit L2 Norm Coefficient of Variation (`logit_norm_cv`)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.models.born_gpt import QuantumBornGPT


def load_real_fineweb_sequences(num_records: int, seq_len: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    val_bin = REPO_ROOT / ".cache" / "fineweb10B" / "fineweb_val_000000.bin"
    total_tokens = num_records * seq_len + 1
    if val_bin.exists():
        raw = np.memmap(val_bin, dtype=np.uint16, mode="r", offset=256 * 4)
        tokens = torch.from_numpy(np.array(raw[:total_tokens], dtype=np.int64))
    else:
        g = torch.Generator().manual_seed(42)
        ranks = torch.arange(1, 50258, dtype=torch.float64)
        probs = (ranks ** -1.05)
        probs = (probs / probs.sum()).to(torch.float32)
        tokens = torch.multinomial(probs, num_samples=total_tokens, replacement=True, generator=g)
    x = tokens[:-1].reshape(num_records, seq_len)
    y = tokens[1:].reshape(num_records, seq_len)
    return x, y


def build_variant(variant: str, vocab_size: int = 50304, seq_len: int = 512) -> torch.nn.Module:
    if variant == "q_seed128_base":
        return QuantumBornGPT(
            vocab_size=vocab_size,
            seed_dim=128,
            embed_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=1024,
            use_two_beam_mlp=True,
        )
    if variant == "q_adapter384_2beam":
        return QuantumBornGPT(
            vocab_size=vocab_size,
            seed_dim=384,
            embed_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=1024,
            use_two_beam_mlp=True,
            use_overcomplete_two_beam=True,
            frame_expansion=3.0,
            use_two_beam_seed_adapter=True,
        )
    if variant == "q_adapter384_2beam_dynbeta":
        kwargs: dict[str, Any] = dict(
            vocab_size=vocab_size,
            seed_dim=384,
            embed_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=1024,
            use_two_beam_mlp=True,
            use_overcomplete_two_beam=True,
            frame_expansion=3.0,
            use_two_beam_seed_adapter=True,
            use_dynamic_povm_beta=True,
        )
        return QuantumBornGPT(**kwargs)
    if variant == "classical_modded":
        from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT

        return ClassicalModdedGPT(
            vocab_size=vocab_size,
            embed_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=1024,
            max_seq_len=seq_len,
        )
    raise ValueError(f"Unknown model variant: {variant}")


def run_overfit_trial(
    variant: str,
    num_records: int,
    steps: int,
    seq_len: int,
    batch_cap: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(1337)
    x_all, y_all = load_real_fineweb_sequences(num_records, seq_len=seq_len)
    x_all, y_all = x_all.to(device), y_all.to(device)
    model = build_variant(variant, seq_len=seq_len).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    high_lr, base_lr = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("povm_beta", "adapter", "purity_gate", "lm_head", "lambdas")):
            high_lr.append(p)
        else:
            base_lr.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": base_lr, "lr": 1.5e-3, "weight_decay": 0.0}, {"params": high_lr, "lr": 6.0e-3, "weight_decay": 0.0}],
        betas=(0.9, 0.95),
    )

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if num_records <= batch_cap:
            bx, by = x_all, y_all
        else:
            idx = torch.randint(0, num_records, (batch_cap,), device=device)
            bx, by = x_all[idx], y_all[idx]
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(bx)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), by.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    max_probs, entropies, logit_cvs = [], [], []
    with torch.no_grad():
        for i in range(0, num_records, batch_cap):
            bx = x_all[i : i + batch_cap]
            by = y_all[i : i + batch_cap]
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(bx)
                logits = (out[0] if isinstance(out, tuple) else out).float()
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), by.reshape(-1), reduction="sum")
                preds = logits.argmax(dim=-1)
                probs = F.softmax(logits, dim=-1)
                max_p = probs.max(dim=-1).values
                ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                l_norms = logits.norm(dim=-1)
                l_cv = float(l_norms.std() / l_norms.mean().clamp_min(1e-8))
            total_loss += float(ce.item())
            total_correct += int((preds == by).sum().item())
            total_count += int(by.numel())
            max_probs.append(float(max_p.mean().item()))
            entropies.append(float(ent.mean().item()))
            logit_cvs.append(l_cv)

    return {
        "variant": variant,
        "num_records": num_records,
        "total_tokens": total_count,
        "num_params_m": round(num_params / 1e6, 2),
        "final_ce": round(total_loss / total_count, 4),
        "top1_acc": round(total_correct / total_count, 4),
        "mean_max_prob": round(float(np.mean(max_probs)), 4),
        "mean_output_entropy_nats": round(float(np.mean(entropies)), 4),
        "logit_norm_cv": round(float(np.mean(logit_cvs)), 4),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Karpathy Memorization & Logit-Sharpness Diagnostic Probe.")
    parser.add_argument(
        "--variants",
        type=str,
        default="q_seed128_base,q_adapter384_2beam,q_adapter384_2beam_dynbeta,classical_modded",
        help="Comma-separated model variants to probe.",
    )
    parser.add_argument("--records", type=str, default="1,8,32", help="Comma-separated record counts N.")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps per trial.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length T per record.")
    parser.add_argument("--batch-cap", type=int, default=8, help="Max microbatch size per step.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save JSON report.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    records_list = [int(r.strip()) for r in args.records.split(",") if r.strip()]

    results = []
    print(f"{'Variant':<28} | {'Recs':<5} | {'Tokens':<7} | {'Final CE':<9} | {'Top-1':<7} | {'MaxProb':<8} | {'Entropy':<8} | {'LogitCV'}")
    print("-" * 102)
    for n_rec in records_list:
        for var in variants:
            res = run_overfit_trial(var, n_rec, args.steps, args.seq_len, args.batch_cap, device)
            results.append(res)
            print(
                f"{var:<28} | {n_rec:<5} | {res['total_tokens']:<7} | {res['final_ce']:<9.4f} | "
                f"{res['top1_acc']*100:>5.2f}% | {res['mean_max_prob']:<8.4f} | "
                f"{res['mean_output_entropy_nats']:<8.4f} | {res['logit_norm_cv']:.4f}"
            )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
