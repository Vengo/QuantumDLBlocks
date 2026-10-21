"""Example 21: Controlled Verification of Observable Hypotheses on Single-Pass 9-Shard
Streaming vs. 9-Pass Single-Shard Repetition (H1: Memorization Gap Elimination, Bigram
Coverage & Validation Scaling) and Warmup-Stable-Decay Schedule Recovery + Zero-Stall
Asynchronous Shard Prefetching (H2) in `QuantumBornGPT`.

Implements Section 3.4 of EXPERIMENT_PLAN_011.md and writes empirical results to
`.cache/hypothesis_experiments_011.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Literal

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.fineweb_stream import (
    FineWebMultiShardStream,
    compute_empirical_bigram_coverage,
    compute_memorization_gap_metrics,
    generate_contextual_zipf_shard,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params
from qdlblocks.optim.schedules import cosine_lr_multiplier, wsd_lr_multiplier


def _shard_to_batches(
    shard_tokens: np.ndarray,
    batch_size: int = 16,
    seq_len: int = 64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a token shard into `(num_batches, batch_size, seq_len)` `(x, y)` tensors."""
    tok_per_step = batch_size * seq_len
    n_steps = (len(shard_tokens) - 1) // tok_per_step
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(n_steps):
        start = i * tok_per_step
        buf = shard_tokens[start : start + tok_per_step + 1].astype(np.int64)
        xs.append(buf[:-1].reshape(batch_size, seq_len))
        ys.append(buf[1:].reshape(batch_size, seq_len))
    x_arr = torch.from_numpy(np.stack(xs, axis=0)).to(device)
    y_arr = torch.from_numpy(np.stack(ys, axis=0)).to(device)
    return x_arr, y_arr


@torch.no_grad()
def _evaluate_batches(
    model: QuantumBornGPT,
    x_batches: torch.Tensor,
    y_batches: torch.Tensor,
) -> dict[str, float]:
    """Evaluate average cross-entropy, perplexity, and top-1 accuracy over a set of batches."""
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for i in range(x_batches.size(0)):
        xb = x_batches[i]
        yb = y_batches[i]
        logits, loss = model(xb, yb)
        assert loss is not None
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        correct += int((preds == yb).sum().item())
        total += int(yb.numel())
    ce = float(np.mean(losses))
    return {
        "ce": ce,
        "ppl": float(math.exp(min(ce, 20.0))),
        "top1_acc": 100.0 * correct / max(1, total),
    }


def _build_reference_model(
    vocab_size: int = 512,
    embed_dim: int = 64,
    seed_dim: int = 24,
    num_layers: int = 2,
    num_heads: int = 4,
    hidden_dim: int = 96,
    block_size: int = 64,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> QuantumBornGPT:
    torch.manual_seed(seed)
    return QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=seed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        block_size=block_size,
        readout_mode="tied_stiefel",
        learnable_temperature=True,
        use_two_beam_mlp=True,
        use_overcomplete_seed_adapter=True,
    ).to(device)


def run_streaming_arm(
    arm_name: str,
    ref_state: dict[str, torch.Tensor],
    shard_batches: list[tuple[torch.Tensor, torch.Tensor]],
    val_batches: tuple[torch.Tensor, torch.Tensor],
    schedule: Literal["cosine", "wsd"] = "cosine",
    wsd_decay_fraction: float = 0.20,
    base_muon_lr: float = 0.025,
    base_adamw_lr: float = 0.006,
    vocab_size: int = 512,
    embed_dim: int = 64,
    seed_dim: int = 24,
    num_layers: int = 2,
    num_heads: int = 4,
    hidden_dim: int = 96,
    block_size: int = 64,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Train one `QuantumBornGPT` arm across 9 shard passes (either 1x9 repeated or 9x1 unique)."""
    model = _build_reference_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=seed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        block_size=block_size,
        device=device,
        seed=42,
    )
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(ref_state[k])

    muon_params, adamw_params = split_muon_adamw_params(
        model,
        adamw_name_patterns=("wte", "povm_head", "embed", "unet_scales", "smear", "inv_temp", "seed_adapter"),
    )
    opt_muon = StiefelMuon(
        muon_params,
        lr=base_muon_lr,
        momentum=0.95,
        weight_decay=0.01,
        decay_mode="cautious",
        stiefel_penalty=0.005,
        ns_steps=5,
        retract_every=4,
    )
    opt_adamw = torch.optim.AdamW(adamw_params, lr=base_adamw_lr, betas=(0.9, 0.95), weight_decay=1e-4)

    num_passes = len(shard_batches)
    steps_per_shard = int(shard_batches[0][0].size(0))
    total_steps = num_passes * steps_per_shard
    warmup_steps = max(4, total_steps // 12)

    x_val, y_val = val_batches
    train_ce_at_evals: list[float] = []
    val_ce_at_evals: list[float] = []
    boundary_pre_train_ce: list[float] = []
    boundary_post_train_ce: list[float] = []
    history: list[dict[str, float]] = []

    global_step = 0
    prev_pass_end_ce: float | None = None

    for pass_idx, (x_sh, y_sh) in enumerate(shard_batches):
        rng = np.random.default_rng(100 + pass_idx)
        order = rng.permutation(steps_per_shard)
        pass_step_losses: list[float] = []

        for s_idx in range(steps_per_shard):
            global_step += 1
            if schedule == "wsd":
                lr_mult = wsd_lr_multiplier(
                    global_step,
                    total_steps,
                    warmup_steps=warmup_steps,
                    decay_fraction=wsd_decay_fraction,
                    min_lr_ratio=0.08,
                )
            else:
                lr_mult = cosine_lr_multiplier(
                    global_step,
                    total_steps,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=0.10,
                )

            for g in opt_muon.param_groups:
                g["lr"] = base_muon_lr * lr_mult
            for g in opt_adamw.param_groups:
                g["lr"] = base_adamw_lr * lr_mult

            b_idx = int(order[s_idx])
            xb = x_sh[b_idx]
            yb = y_sh[b_idx]

            model.train()
            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adamw.step()

            step_ce = float(loss.detach().item())
            pass_step_losses.append(step_ce)

        first_2_ce = float(np.mean(pass_step_losses[:2]))
        last_2_ce = float(np.mean(pass_step_losses[-2:]))
        if prev_pass_end_ce is not None:
            boundary_pre_train_ce.append(prev_pass_end_ce)
            boundary_post_train_ce.append(first_2_ce)
        prev_pass_end_ce = last_2_ce

        train_eval = _evaluate_batches(model, x_sh, y_sh)
        val_eval = _evaluate_batches(model, x_val, y_val)

        train_ce = float(train_eval["ce"])
        val_ce = float(val_eval["ce"])
        train_ce_at_evals.append(train_ce)
        val_ce_at_evals.append(val_ce)

        history.append(
            {
                "pass_idx": float(pass_idx + 1),
                "step": float(global_step),
                "rolling_train_ce": round(float(np.mean(pass_step_losses)), 4),
                "train_ce": round(train_ce, 4),
                "val_ce": round(val_ce, 4),
                "val_ppl": round(float(val_eval["ppl"]), 2),
                "val_top1_acc": round(float(val_eval["top1_acc"]), 2),
                "memorization_gap": round(val_ce - train_ce, 4),
            }
        )

    gap_metrics = compute_memorization_gap_metrics(
        train_ce_at_evals=train_ce_at_evals,
        val_ce_at_evals=val_ce_at_evals,
        boundary_pre_train_ce=boundary_pre_train_ce,
        boundary_post_train_ce=boundary_post_train_ce,
    )
    final_h = history[-1]
    print(
        f"  [{arm_name:<26}] Final Train CE: {gap_metrics['final_train_ce']:.4f} | "
        f"Final Val CE: {gap_metrics['final_val_ce']:.4f} (PPL: {final_h['val_ppl']:6.2f}, Top-1: {final_h['val_top1_acc']:5.2f}%) | "
        f"MemGap (Val-Train): {gap_metrics['final_memorization_gap']:+.4f} nats | "
        f"BoundaryDrop: {gap_metrics['mean_boundary_drop']:+.4f} nats"
    )
    return {
        "arm_name": arm_name,
        "schedule": schedule,
        "total_steps": total_steps,
        **{k: round(v, 4) for k, v in gap_metrics.items()},
        "final_val_ppl": final_h["val_ppl"],
        "final_val_top1_acc": final_h["val_top1_acc"],
        "history": history,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXPERIMENT_PLAN_011 Controlled Verification on device={device}...")

    vocab_size = 512
    n_shard_tokens = 16385  # 16 steps/shard * (B=16 * T=64 = 1024 tok/step) + 1
    num_shards = 9
    batch_size = 16
    seq_len = 64

    # 1. Generate ground-truth validation shard (shard_idx=0) and 9 training shards (shard_idx=1..9)
    val_shard = generate_contextual_zipf_shard(
        n_shard_tokens, vocab_size=vocab_size, shard_idx=0, base_seed=42
    )
    unique_shards = [
        generate_contextual_zipf_shard(
            n_shard_tokens, vocab_size=vocab_size, shard_idx=s_idx, base_seed=42
        )
        for s_idx in range(1, num_shards + 1)
    ]
    repeated_shards = [unique_shards[0] for _ in range(num_shards)]

    cov_1x9 = compute_empirical_bigram_coverage(repeated_shards, max_vocab=vocab_size)
    cov_9x1 = compute_empirical_bigram_coverage(unique_shards, max_vocab=vocab_size)
    bigram_ratio = cov_9x1["unique_bigrams"] / max(1.0, cov_1x9["unique_bigrams"])

    val_batches = _shard_to_batches(val_shard, batch_size=batch_size, seq_len=seq_len, device=device)
    unique_batches = [
        _shard_to_batches(sh, batch_size=batch_size, seq_len=seq_len, device=device)
        for sh in unique_shards
    ]
    repeated_batches = [unique_batches[0] for _ in range(num_shards)]

    ref_model = _build_reference_model(vocab_size=vocab_size, device=device, seed=42)
    ref_state = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}

    print("\n" + "=" * 100)
    print("EXPERIMENT 1 (H1): 1x9 Single-Shard Repetition vs. 9x1 Multi-Shard Streaming (Cosine Schedule)")
    print("=" * 100)
    print(
        f"  [Context Coverage] 1x9 Unique Bigrams: {int(cov_1x9['unique_bigrams']):,d} | "
        f"9x1 Unique Bigrams: {int(cov_9x1['unique_bigrams']):,d} | "
        f"Coverage Ratio: {bigram_ratio:.2f}x (>= 2.50x)"
    )

    arm_a_cos = run_streaming_arm(
        arm_name="repeated_1x9_cosine",
        ref_state=ref_state,
        shard_batches=repeated_batches,
        val_batches=val_batches,
        schedule="cosine",
        vocab_size=vocab_size,
        device=device,
    )
    arm_b_cos = run_streaming_arm(
        arm_name="streaming_9x1_cosine",
        ref_state=ref_state,
        shard_batches=unique_batches,
        val_batches=val_batches,
        schedule="cosine",
        vocab_size=vocab_size,
        device=device,
    )

    gap_1x9 = arm_a_cos["final_memorization_gap"]
    gap_9x1 = arm_b_cos["final_memorization_gap"]
    gap_reduction_ratio = gap_1x9 / max(abs(gap_9x1), 1e-4)
    gap_diff = gap_1x9 - gap_9x1
    val_ce_gain_cos = arm_a_cos["final_val_ce"] - arm_b_cos["final_val_ce"]

    print("  " + "-" * 96)
    print(
        f"  [H1 Summary] Memorization Gap: 1x9={gap_1x9:+.4f} vs 9x1={gap_9x1:+.4f} "
        f"(Diff: {gap_diff:+.4f} >= +0.20, Ratio: {gap_reduction_ratio:.2f}x >= 3.0x) | "
        f"Val CE Improvement (9x1 vs 1x9): {val_ce_gain_cos:+.4f} nats (>= +0.12)"
    )

    assert bigram_ratio >= 2.5, (
        f"H1 Failed: expected 9x1 unique bigrams >= 2.5x 1x9, got {bigram_ratio:.2f}x"
    )
    assert gap_9x1 <= gap_1x9 / 3.0 and gap_diff >= 0.20, (
        f"H1 Failed: expected gap_9x1 <= gap_1x9 / 3.0 and gap_diff >= 0.20, "
        f"got gap_1x9={gap_1x9:.4f}, gap_9x1={gap_9x1:.4f}, diff={gap_diff:.4f}"
    )
    assert val_ce_gain_cos >= 0.12, (
        f"H1 Failed: expected 9x1 Cosine Val CE at least 0.12 nats below 1x9 Cosine, "
        f"got {arm_b_cos['final_val_ce']:.4f} vs {arm_a_cos['final_val_ce']:.4f} (gain={val_ce_gain_cos:+.4f})"
    )

    print("\n" + "=" * 100)
    print("EXPERIMENT 2 (H2): WSD Schedule Recovery on Unique Shards & Zero-Stall Async Prefetching")
    print("=" * 100)

    arm_c_wsd = run_streaming_arm(
        arm_name="repeated_1x9_wsd",
        ref_state=ref_state,
        shard_batches=repeated_batches,
        val_batches=val_batches,
        schedule="wsd",
        wsd_decay_fraction=0.20,
        vocab_size=vocab_size,
        device=device,
    )
    arm_d_wsd = run_streaming_arm(
        arm_name="streaming_9x1_wsd",
        ref_state=ref_state,
        shard_batches=unique_batches,
        val_batches=val_batches,
        schedule="wsd",
        wsd_decay_fraction=0.20,
        vocab_size=vocab_size,
        device=device,
    )

    val_ce_gain_wsd = arm_c_wsd["final_val_ce"] - arm_d_wsd["final_val_ce"]
    wsd_vs_cos_diff = arm_d_wsd["final_val_ce"] - arm_b_cos["final_val_ce"]

    # Benchmark FineWebMultiShardStream asynchronous prefetching across 9 shards
    with tempfile.TemporaryDirectory() as tmp_dir:
        streamer = FineWebMultiShardStream(
            data_dir=Path(tmp_dir),
            num_shards=num_shards,
            prefetch_async=True,
            synthetic_tokens_per_shard=n_shard_tokens,
            vocab_size=vocab_size,
            seed=42,
        )
        streamer.prefetch_shard(1)
        time.sleep(0.015)  # allow initial prefetch to complete before step 1
        wait_times_ms: list[float] = []
        bit_exact_all = True
        for _, s_idx, sh_arr, w_ms in streamer.iter_shards(num_passes=1):
            wait_times_ms.append(w_ms)
            if not np.array_equal(sh_arr, unique_shards[s_idx - 1]):
                bit_exact_all = False
            time.sleep(0.010)  # simulate accelerator step compute overlapping next shard prefetch
        streamer.close()

    max_prefetch_wait_ms = float(max(wait_times_ms))
    mean_prefetch_wait_ms = float(np.mean(wait_times_ms))

    print("  " + "-" * 96)
    print(
        f"  [H2 Summary] WSD 9x1 vs 1x9 Val CE Gain: {val_ce_gain_wsd:+.4f} nats (>= +0.15) | "
        f"9x1 WSD vs 9x1 Cosine: {wsd_vs_cos_diff:+.4f} nats (<= +0.03) | "
        f"Async Prefetch Max Wait: {max_prefetch_wait_ms:.3f} ms (< 25.0 ms), Bit-Exact={bit_exact_all}"
    )

    assert val_ce_gain_wsd >= 0.15, (
        f"H2 Failed: expected 9x1 WSD Val CE at least 0.15 nats below 1x9 WSD, "
        f"got {arm_d_wsd['final_val_ce']:.4f} vs {arm_c_wsd['final_val_ce']:.4f} (gain={val_ce_gain_wsd:+.4f})"
    )
    assert arm_d_wsd["final_val_ce"] <= arm_b_cos["final_val_ce"] + 0.03, (
        f"H2 Failed: expected 9x1 WSD Val CE <= 9x1 Cosine Val CE + 0.03, "
        f"got {arm_d_wsd['final_val_ce']:.4f} vs {arm_b_cos['final_val_ce']:.4f}"
    )
    assert bit_exact_all, "H2 Failed: async prefetched shards did not match sequential shards bit-for-bit"
    assert max_prefetch_wait_ms < 25.0, (
        f"H2 Failed: expected max_prefetch_wait_ms < 25.0 ms, got {max_prefetch_wait_ms:.3f} ms"
    )

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_011",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "h1_multishard_streaming_vs_repetition": {
            "coverage_1x9": {k: round(v, 6) for k, v in cov_1x9.items()},
            "coverage_9x1": {k: round(v, 6) for k, v in cov_9x1.items()},
            "unique_bigram_ratio_9x1_vs_1x9": round(bigram_ratio, 4),
            "repeated_1x9_cosine": arm_a_cos,
            "streaming_9x1_cosine": arm_b_cos,
            "memorization_gap_reduction_ratio": round(gap_reduction_ratio, 4),
            "memorization_gap_diff_nats": round(gap_diff, 4),
            "val_ce_improvement_9x1_vs_1x9_cosine": round(val_ce_gain_cos, 4),
            "passed": True,
        },
        "h2_wsd_schedule_and_async_prefetch": {
            "repeated_1x9_wsd": arm_c_wsd,
            "streaming_9x1_wsd": arm_d_wsd,
            "val_ce_improvement_9x1_vs_1x9_wsd": round(val_ce_gain_wsd, 4),
            "val_ce_diff_9x1_wsd_minus_cosine": round(wsd_vs_cos_diff, 4),
            "async_prefetch": {
                "wait_times_ms": [round(x, 4) for x in wait_times_ms],
                "mean_prefetch_wait_ms": round(mean_prefetch_wait_ms, 4),
                "max_prefetch_wait_ms": round(max_prefetch_wait_ms, 4),
                "bit_exact": bit_exact_all,
            },
            "passed": True,
        },
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_011.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll H1 & H2 quantitative assertions PASSED in {elapsed:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
