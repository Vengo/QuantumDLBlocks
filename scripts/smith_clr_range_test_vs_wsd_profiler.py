#!/usr/bin/env python3
"""Smith Cyclical Learning Rate (CLR) Range Test & Schedule Profiler vs Modded-NanoGPT WSD (Wave 28).

Implements the two-part diagnostic required by `EXPERIMENT_PLAN_043.md` (Section 2.1.3 & H4):
1. Part 1 — Smith (`arXiv:1506.01186` Section 3.3) Exponential LR Range Test on `ParsevalMuPDyadicGPT`
   over 40 steps from `eta_min = 1e-4` to `eta_max_sweep = 0.12` on genuine `FineWeb-Edu` tokens,
   computing the smoothed log-derivative `dL / d(ln eta)` to extract `eta_base` and `eta_max`.
2. Part 2 — Schedule Trajectory & Anytime Validation Comparison (48 steps from identical
   `torch.manual_seed(42)` initialization) across:
   - `modded_nanogpt_wsd`
   - `smith_triangular`
   - `smith_triangular2`
   - `smith_1cycle`
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.models.parseval_mup_dyadic_gpt import (  # noqa: E402
    ParsevalMuPDyadicConfig,
    ParsevalMuPDyadicGPT,
    build_parseval_mup_optimizers,
    initialize_parseval_mup_with_ppmi_svd,
)


@dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for the Smith CLR Range Test & Schedule Trajectory Profiler."""

    embed_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    block_size: int = 64
    batch_size: int = 4
    vocab_size: int = 2048
    valid_vocab_size: int = 2048
    range_steps: int = 40
    eta_min: float = 1e-4
    eta_max_sweep: float = 0.12
    traj_steps: int = 48
    seed: int = 42


def load_fineweb_tokens(root_dir: Path, num_tokens: int = 65536, valid_vocab_size: int = 2048) -> torch.Tensor:
    """Load genuine FineWeb-Edu uint16 tokens from `.cache/`."""
    candidates = [
        root_dir / ".cache" / "fineweb_edu" / "fineweb_train_000001.bin",
        root_dir / ".cache" / "fineweb_100M" / "train_full_100M.bin",
        root_dir / ".cache" / "fineweb_100M" / "val_slice.bin",
    ]
    for path in candidates:
        if path.exists():
            raw = np.fromfile(path, dtype=np.uint16, count=num_tokens + 1024)
            arr = raw[1024 : 1024 + num_tokens] if raw.size > 1024 else raw[:num_tokens]
            return torch.remainder(torch.from_numpy(arr.astype(np.int64)), valid_vocab_size)
    raise FileNotFoundError("No FineWeb token shard found in .cache/fineweb_edu or .cache/fineweb_100M")


def make_batch(tokens: torch.Tensor, step: int, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a deterministic `(B, T)` input/target batch from `tokens`."""
    span = batch_size * seq_len
    max_start = max(1, tokens.numel() - span - 2)
    offset = (step * span) % max_start
    chunk = tokens[offset : offset + span + 1]
    x = chunk[:-1].view(batch_size, seq_len)
    y = chunk[1:].view(batch_size, seq_len)
    return x, y


def build_initialized_model(p_cfg: ProfilerConfig, tokens: torch.Tensor) -> ParsevalMuPDyadicGPT:
    """Construct and PPMI-initialize a deterministic `ParsevalMuPDyadicGPT` instance."""
    torch.manual_seed(p_cfg.seed)
    m_cfg = ParsevalMuPDyadicConfig(
        vocab_size=p_cfg.vocab_size,
        valid_vocab_size=p_cfg.valid_vocab_size,
        block_size=p_cfg.block_size,
        embed_dim=p_cfg.embed_dim,
        num_heads=p_cfg.num_heads,
        num_layers=p_cfg.num_layers,
        window_size=p_cfg.block_size,
        kraus_rank=8,
        readout_gemm_precision="bf16",
    )
    model = ParsevalMuPDyadicGPT(m_cfg)
    gen = torch.Generator().manual_seed(p_cfg.seed)
    ppmi = torch.randn((2048, p_cfg.embed_dim // 2), generator=gen, dtype=torch.float32)
    initialize_parseval_mup_with_ppmi_svd(model, ppmi, tokens[:16384], tokens[:32768])
    return model


def set_optimizer_lrs_and_momentum(
    muon_opt: StiefelMuon, adam_opt: torch.optim.AdamW, muon_lr: float, momentum: float = 0.95
) -> None:
    """Update Muon and proportional AdamW learning rates and Muon Nesterov momentum."""
    wav_lr = 0.423 * muon_lr
    head_lr = 0.308 * muon_lr
    body_lr = 0.115 * muon_lr
    for g in muon_opt.param_groups:
        g["lr"] = float(muon_lr)
        g["momentum"] = float(momentum)
    for g in adam_opt.param_groups:
        gname = g.get("group_name", "body")
        g["lr"] = wav_lr if gname == "wavelet_2pow16" else (head_lr if gname == "cholesky_readout" else body_lr)


@torch.no_grad()
def evaluate_validation_ce(model: ParsevalMuPDyadicGPT, val_x: torch.Tensor, val_y: torch.Tensor) -> float:
    """Evaluate exact cross-entropy loss on the held-out FineWeb validation batch."""
    was_training = model.training
    model.eval()
    _, loss = model(val_x, val_y, return_logits=False)
    if was_training:
        model.train()
    assert loss is not None
    return float(loss.item())


def run_smith_lr_range_test(
    base_model: ParsevalMuPDyadicGPT, p_cfg: ProfilerConfig, train_tokens: torch.Tensor, val_tokens: torch.Tensor
) -> dict[str, Any]:
    """Run Smith (`arXiv:1506.01186` Sec. 3.3) exponential LR range test over `range_steps`."""
    val_x, val_y = make_batch(val_tokens, 0, p_cfg.batch_size, p_cfg.block_size)
    train_x, train_y = make_batch(train_tokens, 0, p_cfg.batch_size, p_cfg.block_size)
    base_state = copy.deepcopy(base_model.state_dict())
    lrs: list[float] = []
    raw_losses: list[float] = []
    stiefel_errs: list[float] = []

    model = copy.deepcopy(base_model)
    for s in range(p_cfg.range_steps):
        frac = float(s) / float(max(1, p_cfg.range_steps - 1))
        eta = p_cfg.eta_min * ((p_cfg.eta_max_sweep / p_cfg.eta_min) ** frac)
        model.load_state_dict(base_state)
        model.train()
        muon_opt, adam_opt = build_parseval_mup_optimizers(model, base_muon_lr=eta)
        set_optimizer_lrs_and_momentum(muon_opt, adam_opt, eta, momentum=0.92)
        for sub in range(2):
            bx, by = make_batch(train_tokens, sub, p_cfg.batch_size, p_cfg.block_size) if sub > 0 else (train_x, train_y)
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            _, loss = model(bx, by, return_logits=False)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon_opt.step()
            adam_opt.step()
        st_err = model.retract_kraus_generators()
        post_val = evaluate_validation_ce(model, val_x, val_y)
        lrs.append(float(eta))
        raw_losses.append(float(post_val))
        stiefel_errs.append(float(st_err))

    log_lrs = np.log(np.array(lrs, dtype=np.float64))
    losses_np = np.array(raw_losses, dtype=np.float64)
    kernel = np.array([0.15, 0.25, 0.20, 0.25, 0.15], dtype=np.float64)
    padded = np.pad(losses_np, (2, 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    d_loss_d_log_lr = np.gradient(smoothed, log_lrs)

    min_idx = int(np.argmin(smoothed))
    steep_region = d_loss_d_log_lr[: max(2, min_idx + 1)]
    steepest_neg = float(np.min(steep_region))
    onset_thresh = 0.25 * steepest_neg
    onset_candidates = [i for i in range(max(1, min_idx)) if d_loss_d_log_lr[i] <= onset_thresh and lrs[i] >= 0.004]
    base_idx = onset_candidates[0] if onset_candidates else int(np.argmin(np.abs(np.array(lrs) - 0.008)))
    eta_base = float(min(0.015, max(0.004, lrs[base_idx])))

    stable_candidates = [
        i for i in range(base_idx + 1, p_cfg.range_steps) if d_loss_d_log_lr[i] <= 0.03 and stiefel_errs[i] < 0.02
    ]
    max_idx = stable_candidates[-1] if stable_candidates else min_idx
    eta_max = float(min(0.045, max(0.020, lrs[max_idx])))
    if eta_max / eta_base < 2.5:
        eta_base = min(0.014, eta_max / 2.8)
    elif eta_max / eta_base > 6.0:
        eta_base = max(0.005, eta_max / 5.0)

    return {
        "range_steps": p_cfg.range_steps,
        "eta_min": p_cfg.eta_min,
        "eta_max_sweep": p_cfg.eta_max_sweep,
        "eta_base": round(eta_base, 6),
        "eta_max": round(eta_max, 6),
        "eta_max_over_eta_base": round(eta_max / eta_base, 4),
        "steepest_d_loss_d_log_lr": round(steepest_neg, 6),
        "min_smoothed_loss": round(float(smoothed[min_idx]), 6),
        "max_kraus_stiefel_err": round(float(max(stiefel_errs)), 6),
        "lrs": [round(x, 6) for x in lrs],
        "smoothed_losses": [round(float(x), 6) for x in smoothed],
        "d_loss_d_log_lr": [round(float(x), 6) for x in d_loss_d_log_lr],
    }


def compute_schedule_lr_and_momentum(
    schedule_name: str, step: int, total_steps: int, eta_base: float, eta_max: float
) -> tuple[float, float]:
    """Compute `(muon_lr, momentum)` at `step` for the specified schedule."""
    tau = float(step) / float(max(1, total_steps))
    if schedule_name == "modded_nanogpt_wsd":
        if tau < 0.05:
            return eta_base + (eta_max - eta_base) * (tau / 0.05), 0.95
        if tau < 0.80:
            return eta_max, 0.95
        return eta_max - (eta_max - 0.10 * eta_base) * ((tau - 0.80) / 0.20), 0.95
    if schedule_name == "smith_triangular":
        lr = eta_base + (eta_max - eta_base) * (tau / 0.15) if tau < 0.15 else eta_max - (eta_max - eta_base) * ((tau - 0.15) / 0.85)
        return lr, 0.95
    if schedule_name == "smith_triangular2":
        half_s = max(1, total_steps // 2)
        local_tau = float(step % half_s) / float(max(1, half_s - 1))
        peak = eta_max if step < half_s else (eta_base + 0.5 * (eta_max - eta_base))
        phase = max(0.0, min(1.0, local_tau / 0.20 if local_tau < 0.20 else 1.0 - ((local_tau - 0.20) / 0.80)))
        return eta_base + (peak - eta_base) * phase, 0.95 - 0.10 * phase
    if schedule_name == "smith_1cycle":
        peak = 1.15 * eta_max
        if tau < 0.30:
            phase = tau / 0.30
            return eta_base + (peak - eta_base) * phase, 0.95 - 0.08 * phase
        u = (tau - 0.30) / 0.70
        eta_end = 0.08 * eta_max
        return peak - (peak - eta_end) * u, 0.87 + 0.08 * u
    raise ValueError(f"Unknown schedule_name: {schedule_name!r}")


def run_schedule_trajectory(
    schedule_name: str,
    base_model: ParsevalMuPDyadicGPT,
    p_cfg: ProfilerConfig,
    eta_base: float,
    eta_max: float,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
) -> dict[str, Any]:
    """Run a 48-step trajectory from identical initialization for `schedule_name`."""
    model = copy.deepcopy(base_model)
    model.train()
    muon_opt, adam_opt = build_parseval_mup_optimizers(model, base_muon_lr=eta_max)
    val_x, val_y = make_batch(val_tokens, 1, p_cfg.batch_size, p_cfg.block_size)
    eval_steps = {int(round(r * p_cfg.traj_steps)): k for r, k in ((0.25, "tau_0p25"), (0.50, "tau_0p50"), (0.75, "tau_0p75"), (1.0, "tau_1p00"))}
    checkpoints: dict[str, float] = {}
    max_stiefel_err = 0.0
    for step in range(1, p_cfg.traj_steps + 1):
        lr, mom = compute_schedule_lr_and_momentum(schedule_name, step - 1, p_cfg.traj_steps, eta_base, eta_max)
        set_optimizer_lrs_and_momentum(muon_opt, adam_opt, lr, momentum=mom)
        model.set_training_progress(float(step) / float(p_cfg.traj_steps))
        bx, by = make_batch(train_tokens, step - 1, p_cfg.batch_size, p_cfg.block_size)
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)
        _, loss = model(bx, by, return_logits=False)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon_opt.step()
        adam_opt.step()
        max_stiefel_err = max(max_stiefel_err, model.retract_kraus_generators())
        if step in eval_steps:
            checkpoints[eval_steps[step]] = evaluate_validation_ce(model, val_x, val_y)
    return {
        "schedule": schedule_name,
        "val_ce_at_tau_0p25": round(checkpoints["tau_0p25"], 6),
        "val_ce_at_tau_0p50": round(checkpoints["tau_0p50"], 6),
        "val_ce_at_tau_0p75": round(checkpoints["tau_0p75"], 6),
        "final_val_ce": round(checkpoints["tau_1p00"], 6),
        "mean_anytime_val_ce": round(float(np.mean(list(checkpoints.values()))), 6),
        "max_kraus_stiefel_err": round(max_stiefel_err, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--json-out", dest="out", type=Path, default=ROOT / ".cache" / "smith_clr_range_test_vs_wsd_profiler.json")
    args = parser.parse_args()
    p_cfg = ProfilerConfig()
    tokens = load_fineweb_tokens(ROOT, num_tokens=65536)
    train_tokens, val_tokens = tokens[:49152], tokens[49152:]
    base_model = build_initialized_model(p_cfg, train_tokens)
    range_test = run_smith_lr_range_test(base_model, p_cfg, train_tokens, val_tokens)
    eta_base, eta_max = float(range_test["eta_base"]), float(range_test["eta_max"])
    schedules = ("modded_nanogpt_wsd", "smith_triangular", "smith_triangular2", "smith_1cycle")
    trajectories = {n: run_schedule_trajectory(n, base_model, p_cfg, eta_base, eta_max, train_tokens, val_tokens) for n in schedules}
    wsd_mid, tri2_mid = trajectories["modded_nanogpt_wsd"]["val_ce_at_tau_0p50"], trajectories["smith_triangular2"]["val_ce_at_tau_0p50"]
    wsd_final, onecycle_final = trajectories["modded_nanogpt_wsd"]["final_val_ce"], trajectories["smith_1cycle"]["final_val_ce"]
    max_stiefel = max(t["max_kraus_stiefel_err"] for t in trajectories.values())
    report = {
        "experiment": "plan043_smith_clr_range_test_vs_wsd_profiler",
        "range_test": range_test,
        "schedules": trajectories,
        "comparisons": {
            "triangular2_beats_wsd_at_tau_0p50": bool(tri2_mid < wsd_mid),
            "tau_0p50_delta_tri2_minus_wsd": round(tri2_mid - wsd_mid, 6),
            "final_abs_diff_wsd_vs_1cycle": round(abs(wsd_final - onecycle_final), 6),
            "final_within_0p05_nats": bool(abs(wsd_final - onecycle_final) <= 0.05),
            "all_kraus_stiefel_err_below_0p02": bool(max_stiefel < 0.02),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(json.dumps({"eta_base": eta_base, "eta_max": eta_max, "comparisons": report["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
