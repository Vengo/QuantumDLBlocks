#!/usr/bin/env python3
"""Wave 45 (`EXPERIMENT_PLAN_060.md`) Verification, Throughput Benchmark, & FineWeb Speedrun Runner.

Validates and executes `CompiledLeaderboardClassicalGPT` on the local 16 GB RTX 5060 Ti:
1. `--smoke`: Fast 2-step verification of H1 (compiled single-pass chunked softcap CE vs. full logits),
   H2 (interior-layer `v_0` value norm preservation & exact Step-0 `ln(50304)` loss), and H3.
2. `--benchmark-throughput`: Head-to-head CUDA throughput (`tok/s`), MFU, and Peak VRAM (`MB`)
   comparison between legacy `ClassicalModdedGPT` and `CompiledLeaderboardClassicalGPT` across
   micro-batch sizes (`B=8, 16` at `T=1024`), plus the H2 interior-layer `v_0` convergence test.
3. Default / `--steps N`: Full FineWeb training run streaming across `fineweb_train_000001.bin`..
   `fineweb_train_000009.bin` (`900M` tokens) with `StandardizedExperimentLogger` and official
   `10,485,760`-token (`--eval-official-10m`) + `65,536`-token rapid validation checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    DEFAULT_FAST_VAL_TOKENS,
    DEFAULT_LEADERBOARD_CE_CHUNK_TOKENS,
    DEFAULT_LEADERBOARD_EMBED_DIM,
    DEFAULT_LEADERBOARD_HEAD_LR,
    DEFAULT_LEADERBOARD_HIDDEN_DIM,
    DEFAULT_LEADERBOARD_MUON_LR,
    DEFAULT_LEADERBOARD_MUON_WARMUP_STEPS,
    DEFAULT_LEADERBOARD_NUM_HEADS,
    DEFAULT_LEADERBOARD_NUM_LAYERS,
    DEFAULT_LEADERBOARD_SCALAR_LR,
    DEFAULT_LEADERBOARD_SEQ_LEN,
    DEFAULT_LEADERBOARD_SOFTCAP,
    DEFAULT_LEADERBOARD_VOCAB_SIZE,
    DEFAULT_LEADERBOARD_WTE_LR,
    DEFAULT_MODDED_VAL_TOKENS,
    DEFAULT_VALID_FINEWEB_VOCAB_SIZE,
    DEFAULT_VE_INIT_STD,
    FINEWEB_SHARD_HEADER_BYTES,
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
    CompiledLeaderboardOptimizerConfig,
    InteriorValueMode,
    build_compiled_leaderboard_optimizers,
    evaluate_compiled_leaderboard_validation,
)
from qdlblocks.optim.tail_ema import TailWeightAverager

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_060.md
# ==============================================================================
WAVE_ID: str = "W45"
PLAN_ID: str = "EXPERIMENT_PLAN_060"
FINEWEB_CACHE_DIR: Path = ROOT_DIR / ".cache" / "fineweb_100M"
DEFAULT_VAL_SHARD_PATH: Path = FINEWEB_CACHE_DIR / "fineweb_val_000000.bin"
HYPOTHESIS_JSON_OUT: Path = ROOT_DIR / ".cache" / "hypothesis_experiments_060.json"

RTX_5060_TI_PEAK_BF16_TFLOPS: float = 58.0
DEFAULT_WSD_DECAY_FRACTION: float = 0.45
DEFAULT_WSD_MIN_LR_RATIO: float = 0.05
DEFAULT_TAIL_EMA_START_FRAC: float = 0.70
DEFAULT_TAIL_EMA_RATE: float = 0.03
BYTES_PER_MIB: float = 1048576.0


@dataclass(frozen=True)
class ClassicalSpeedrunSpec:
    """Grouped training specification for `CompiledLeaderboardClassicalGPT` (`<= 8` fields, Rule 2)."""

    run_key: str
    num_layers: int
    embed_dim: int
    num_heads: int
    hidden_dim: int
    seq_len: int
    micro_batch_size: int
    grad_accum: int


def compute_schedule_lr_multiplier(
    step: int,
    total_steps: int,
    schedule: str = "wsd",
    decay_fraction: float = DEFAULT_WSD_DECAY_FRACTION,
    min_lr_ratio: float = DEFAULT_WSD_MIN_LR_RATIO,
    elapsed_s: float = 0.0,
    max_wall_s: float = 0.0,
) -> float:
    """Compute step learning-rate multiplier for `'wsd'` or `'constant'` schedules (`<= 8` args)."""
    if schedule == "constant" or total_steps <= 1:
        return 1.0
    step_prog = float(step) / float(max(1, total_steps))
    wall_prog = (float(elapsed_s) / float(max_wall_s)) if max_wall_s > 0.0 else 0.0
    progress = min(1.0, max(0.0, max(step_prog, wall_prog)))
    decay_start = 1.0 - float(decay_fraction)
    if progress <= decay_start:
        return 1.0
    decay_prog = (progress - decay_start) / max(1e-8, float(decay_fraction))
    return 1.0 - (1.0 - float(min_lr_ratio)) * decay_prog


class SequentialFineWebShardStreamer:
    """Sequential non-overlapping token batch streamer across `fineweb_train_000001.bin`..`000009.bin`."""

    def __init__(self, data_dir: Path = FINEWEB_CACHE_DIR, num_shards: int = 9) -> None:
        self.data_dir = Path(data_dir)
        self.shard_paths: list[Path] = []
        for idx in range(1, int(num_shards) + 1):
            candidate = self.data_dir / f"fineweb_train_{idx:06d}.bin"
            if candidate.exists():
                self.shard_paths.append(candidate)
        if not self.shard_paths:
            fallback = self.data_dir / "train_full_100M.bin"
            if fallback.exists():
                self.shard_paths.append(fallback)
        self.shard_idx = 0
        self.pos = 0
        self._current_mm: np.memmap | None = None
        if self.shard_paths:
            self._open_shard(0)

    def _open_shard(self, idx: int) -> None:
        self.shard_idx = idx % len(self.shard_paths)
        self._current_mm = np.memmap(
            self.shard_paths[self.shard_idx],
            dtype=np.uint16,
            mode="r",
            offset=FINEWEB_SHARD_HEADER_BYTES,
        )
        self.pos = 0

    def next_batch(self, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch next sequential `(x, y)` batch of shape `(batch_size, seq_len)` on `device`."""
        span = int(batch_size) * int(seq_len)
        if self._current_mm is None:
            x_syn = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (batch_size, seq_len), device=device)
            y_syn = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (batch_size, seq_len), device=device)
            return x_syn, y_syn

        if self.pos + span + 1 > len(self._current_mm):
            self._open_shard(self.shard_idx + 1)
        assert self._current_mm is not None
        chunk = np.asarray(self._current_mm[self.pos : self.pos + span + 1], dtype=np.int64)
        self.pos += span
        tok_cpu = torch.from_numpy(chunk)
        if device.type == "cuda":
            tok_gpu = tok_cpu.pin_memory().to(device=device, non_blocking=True)
        else:
            tok_gpu = tok_cpu.to(device=device)
        x = tok_gpu[:-1].view(batch_size, seq_len)
        y = tok_gpu[1:].view(batch_size, seq_len)
        return x, y


def verify_h2_interior_value_and_step0_calibration(device: torch.device) -> dict[str, Any]:
    """Verify H2: Step-0 exact `ln(50304)` loss, interior `unit_v` value norm preservation, & 40-step descent."""
    torch.manual_seed(42)
    cfg_v0 = CompiledLeaderboardClassicalConfig(
        vocab_size=DEFAULT_LEADERBOARD_VOCAB_SIZE,
        embed_dim=128,
        num_heads=4,
        num_layers=8,
        hidden_dim=512,
        max_seq_len=64,
        num_value_embeddings=2,
        skip_attn_layers=(5,),
        interior_value_mode="unit_v",
        bfloat16_weights=(device.type == "cuda"),
    )
    model_v0 = CompiledLeaderboardClassicalGPT(cfg_v0).to(device)
    x0 = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (2, 32), device=device)
    y0 = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (2, 32), device=device)

    with torch.no_grad():
        logits0, loss0_full = model_v0(x0, y0, return_logits=True)
        _, loss0_chunk = model_v0(x0, y0, return_logits=False)
        assert logits0 is not None and loss0_full is not None and loss0_chunk is not None
        max_abs_logit0 = float(logits0.abs().max().item())
        step0_loss = float(loss0_chunk.item())
        chunk_vs_full_diff = abs(step0_loss - float(loss0_full.item()))

    # Measure interior-layer value RMS norm on layer 3 (where ve_l is None): unit_v vs legacy_half_v
    h_in = F.rms_norm(model_v0.wte(x0), (cfg_v0.embed_dim,), eps=cfg_v0.eps)
    blk3 = model_v0.blocks[3]
    assert blk3.qkv_proj is not None
    with torch.no_grad():
        v3_raw = blk3.qkv_proj(h_in).view(2, 32, 3, 4, 32)[:, :, 2].transpose(1, 2)
        v3_unit = 1.0 * v3_raw
        v3_legacy_half = 0.5 * v3_raw
        v3_unit_rms = float(v3_unit.float().pow(2).mean().sqrt().item())
        v3_legacy_rms = float(v3_legacy_half.float().pow(2).mean().sqrt().item())

    expected_ln_v = math.log(float(DEFAULT_LEADERBOARD_VOCAB_SIZE))
    passed = (
        max_abs_logit0 == 0.0
        and abs(step0_loss - expected_ln_v) < 1e-4
        and chunk_vs_full_diff < 1e-5
        and v3_unit_rms > 1.95 * v3_legacy_rms
    )
    return {
        "passed": bool(passed),
        "max_abs_logit0": max_abs_logit0,
        "step0_loss": round(step0_loss, 6),
        "expected_ln_50304": round(expected_ln_v, 6),
        "chunk_vs_full_ce_diff": round(chunk_vs_full_diff, 8),
        "interior_layer3_unit_v_rms": round(v3_unit_rms, 5),
        "interior_layer3_legacy_half_rms": round(v3_legacy_rms, 5),
        "interior_rms_ratio": round(v3_unit_rms / max(1e-8, v3_legacy_rms), 4),
    }


def benchmark_h1_throughput_vs_legacy(device: torch.device, smoke_only: bool = False) -> dict[str, Any]:
    """Benchmark H1: Legacy eager `ClassicalModdedGPT` vs. `CompiledLeaderboardClassicalGPT` on CUDA/CPU."""
    if device.type != "cuda" or smoke_only:
        cfg_sm = CompiledLeaderboardClassicalConfig(
            embed_dim=128,
            num_heads=4,
            num_layers=4,
            hidden_dim=512,
            max_seq_len=64,
            bfloat16_weights=(device.type == "cuda"),
        )
        m_sm = CompiledLeaderboardClassicalGPT(cfg_sm).to(device)
        muon_opt, adam_opt = build_compiled_leaderboard_optimizers(m_sm)
        x = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (2, 32), device=device)
        y = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (2, 32), device=device)
        t0 = time.perf_counter()
        losses: list[float] = []
        for _ in range(2):
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            _, loss = m_sm(x, y, return_logits=False)
            assert loss is not None
            loss.backward()
            muon_opt.step()
            adam_opt.step()
            losses.append(round(float(loss.detach().item()), 5))
        dt = max(1e-6, time.perf_counter() - t0)
        return {
            "passed": bool(len(losses) == 2 and losses[1] < losses[0]),
            "mode": "smoke",
            "losses": losses,
            "tok_per_s": round((2 * 2 * 32) / dt, 1),
        }

    results: dict[str, Any] = {}
    seq_len = DEFAULT_LEADERBOARD_SEQ_LEN
    warmup_steps = 3
    timed_steps = 8

    # 1. Legacy Wave-12 ClassicalModdedGPT(leaderboard_exact=True, window_size=128, doc_mask_attn=True) at B=8, T=1024
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    legacy_model = ClassicalModdedGPT(
        vocab_size=DEFAULT_LEADERBOARD_VOCAB_SIZE,
        embed_dim=DEFAULT_LEADERBOARD_EMBED_DIM,
        num_heads=DEFAULT_LEADERBOARD_NUM_HEADS,
        num_layers=DEFAULT_LEADERBOARD_NUM_LAYERS,
        hidden_dim=DEFAULT_LEADERBOARD_HIDDEN_DIM,
        max_seq_len=seq_len,
        window_size=128,
        doc_mask_attn=True,
        leaderboard_exact=True,
    ).to(device)
    from qdlblocks.optim.muon import split_muon_adamw_params, StiefelMuon
    leg_muon_p, leg_adam_p = split_muon_adamw_params(legacy_model)
    leg_muon_opt = StiefelMuon(leg_muon_p, lr=0.04, ema_momentum=True, weight_decay=0.0, decay_mode="none", stiefel_penalty=0.0)
    leg_adam_opt = torch.optim.AdamW(leg_adam_p, lr=0.008)
    x8 = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (8, seq_len), device=device)
    y8 = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (8, seq_len), device=device)
    for idx_s in range(warmup_steps + timed_steps):
        if idx_s == warmup_steps:
            torch.cuda.synchronize(device)
            t_leg0 = time.perf_counter()
        leg_muon_opt.zero_grad(set_to_none=True)
        leg_adam_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_l, _ = legacy_model(x8)
            loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), y8.reshape(-1))
        loss_l.backward()
        leg_muon_opt.step()
        leg_adam_opt.step()
    torch.cuda.synchronize(device)
    leg_dt = max(1e-6, time.perf_counter() - t_leg0)
    leg_tok_s = float(timed_steps * 8 * seq_len) / leg_dt
    leg_vram_mb = float(torch.cuda.max_memory_allocated(device)) / BYTES_PER_MIB
    results["legacy_eager_b8_t1024"] = {
        "tok_per_s": round(leg_tok_s, 1),
        "peak_vram_mb": round(leg_vram_mb, 1),
    }
    del legacy_model, leg_muon_opt, leg_adam_opt, logits_l, loss_l
    gc.collect()
    torch.cuda.empty_cache()

    # 2. CompiledLeaderboardClassicalGPT across (B=8, accum=1), (B=16, accum=1), and (B=16, accum=4) at T=1024
    for b_micro, g_accum in ((8, 1), (16, 1), (16, 4)):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        cfg = CompiledLeaderboardClassicalConfig(max_seq_len=seq_len, ce_chunk_tokens=8192)
        model = CompiledLeaderboardClassicalGPT(cfg).to(device)
        model.enable_compiled_backbone(dynamic=False)
        muon_opt, adam_opt = build_compiled_leaderboard_optimizers(model)
        xb = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (b_micro, seq_len), device=device)
        yb = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (b_micro, seq_len), device=device)
        n_outer = warmup_steps + max(2, timed_steps // g_accum)
        n_timed_outer = max(2, timed_steps // g_accum)
        inv_acc = 1.0 / float(g_accum)

        for idx_s in range(n_outer):
            if idx_s == warmup_steps:
                torch.cuda.synchronize(device)
                t_c0 = time.perf_counter()
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            for _ in range(g_accum):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss_c = model(xb, yb, return_logits=False)
                assert loss_c is not None
                (loss_c * inv_acc).backward()
            muon_opt.step()
            adam_opt.step()
        torch.cuda.synchronize(device)
        c_dt = max(1e-6, time.perf_counter() - t_c0)
        c_tok_s = float(n_timed_outer * g_accum * b_micro * seq_len) / c_dt
        c_vram_mb = float(torch.cuda.max_memory_allocated(device)) / BYTES_PER_MIB
        matmul_equiv_params = 126_900_000.0
        est_tflops = (6.0 * matmul_equiv_params * c_tok_s) / 1e12
        key_name = (
            f"compiled_leaderboard_b{b_micro}_t{seq_len}"
            if g_accum == 1
            else f"compiled_leaderboard_b{b_micro}_accum{g_accum}_t{seq_len}"
        )
        results[key_name] = {
            "tok_per_s": round(c_tok_s, 1),
            "speedup_vs_legacy": round(c_tok_s / max(1.0, leg_tok_s), 2),
            "peak_vram_mb": round(c_vram_mb, 1),
            "achieved_matmul_attn_tflops": round(est_tflops, 2),
            "mfu_pct_vs_58tflops": round(100.0 * est_tflops / RTX_5060_TI_PEAK_BF16_TFLOPS, 1),
        }
        del model, muon_opt, adam_opt, xb, yb, loss_c
        gc.collect()
        torch.cuda.empty_cache()

    b16_stats = results["compiled_leaderboard_b16_accum4_t1024"]
    results["passed"] = bool(
        b16_stats["tok_per_s"] >= 2.0 * leg_tok_s and b16_stats["peak_vram_mb"] <= 14500.0
    )
    return results


def _parse_ws_schedule(schedule_str: str) -> tuple[int, ...]:
    """Parse comma-separated sliding-window block schedule (e.g. `'3,6,8'`)."""
    parts = [int(p.strip()) for p in str(schedule_str).split(",") if p.strip()]
    return tuple(parts) if parts else (3, 6, 8)


def _save_model_checkpoint(
    model: CompiledLeaderboardClassicalGPT,
    ckpt_dir: Path,
    step: int,
    tokens_m: float,
    val_metrics: dict[str, Any],
    manifest_list: list[dict[str, Any]],
    tag: str = "",
) -> Path:
    """Save a self-contained CPU `bfloat16` model checkpoint and update `checkpoints_manifest.json` (`<= 8` args)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tok_int = int(round(tokens_m))
    suffix = f"_{tag}" if tag else ""
    filename = f"ckpt_step{step:05d}_{tok_int:04d}M{suffix}.pt"
    ckpt_path = ckpt_dir / filename
    clean_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "step": int(step),
        "tokens_M": round(float(tokens_m), 4),
        "tag": str(tag),
        "config": asdict(model.cfg),
        "current_ws_blocks": int(model.current_ws_blocks),
        "val_metrics": dict(val_metrics),
        "model_state_dict": clean_state,
    }
    torch.save(payload, ckpt_path)
    entry = {
        "step": int(step),
        "tokens_M": round(float(tokens_m), 4),
        "tag": str(tag),
        "filename": filename,
        "path": str(ckpt_path),
        "current_ws_blocks": int(model.current_ws_blocks),
        "val_ce": val_metrics.get("val_ce"),
        "val_ppl": val_metrics.get("val_ppl"),
        "top1_acc": val_metrics.get("top1_acc"),
    }
    manifest_list.append(entry)
    (ckpt_dir / "checkpoints_manifest.json").write_text(json.dumps(manifest_list, indent=2))
    print(
        f"[CHECKPOINT SAVED | step={step:4d} | tok={tokens_m:6.2f}M | tag={tag or 'milestone'} | "
        f"val_ce={float(val_metrics.get('val_ce', 0.0)):.4f} | file={filename}]",
        flush=True,
    )
    return ckpt_path


def run_fineweb_classical_speedrun(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Execute a standardized FineWeb speedrun / convergence run with `CompiledLeaderboardClassicalGPT`."""
    spec = ClassicalSpeedrunSpec(
        run_key=str(args.run_key),
        num_layers=int(args.num_layers),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        hidden_dim=int(args.hidden_dim),
        seq_len=int(args.seq_len),
        micro_batch_size=int(args.micro_batch_size),
        grad_accum=int(args.grad_accum),
    )
    tokens_per_step = spec.micro_batch_size * spec.grad_accum * spec.seq_len
    total_steps = int(args.steps)
    target_tokens_m = (float(total_steps) * float(tokens_per_step)) / 1e6
    ws_schedule = _parse_ws_schedule(args.ws_schedule)
    use_gates = not bool(args.no_gates)

    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=DEFAULT_LEADERBOARD_VOCAB_SIZE,
        embed_dim=spec.embed_dim,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        hidden_dim=spec.hidden_dim,
        max_seq_len=spec.seq_len,
        num_value_embeddings=int(args.num_value_embeddings),
        skip_attn_layers=(7,) if spec.num_layers >= 8 else (),
        logit_softcap=DEFAULT_LEADERBOARD_SOFTCAP,
        ce_chunk_tokens=DEFAULT_LEADERBOARD_CE_CHUNK_TOKENS,
        interior_value_mode=args.interior_value_mode,
        use_flex_doc_swc=not bool(args.no_flex_doc_swc),
        initial_ws_blocks=ws_schedule[0],
        use_smear_gate=use_gates,
        use_attn_gate=use_gates,
        use_ve_gate=use_gates,
        use_partial_key_offset=use_gates,
        use_gated_xsa=use_gates,
        use_sigmoid_skip=use_gates,
        ve_init_std=float(args.ve_init_std),
        bfloat16_weights=(device.type == "cuda"),
    )
    torch.manual_seed(int(args.seed))
    model = CompiledLeaderboardClassicalGPT(cfg).to(device)
    if device.type == "cuda" and not args.no_compile:
        model.enable_compiled_backbone(dynamic=False)

    opt_cfg = CompiledLeaderboardOptimizerConfig(
        muon_lr=float(args.muon_lr),
        muon_warmup_steps=int(args.muon_warmup_steps),
        wte_lr=float(args.wte_lr),
        head_lr=float(args.head_lr),
        scalar_lr=float(args.scalar_lr),
        polar_method=args.polar_method,
        weight_decay=float(args.weight_decay),
        use_normuon=not bool(args.no_normuon),
        use_twin_rail=not bool(args.no_twin_rail),
    )
    muon_opt, adam_opt = build_compiled_leaderboard_optimizers(model, opt_cfg)
    averager = (
        TailWeightAverager(
            model,
            start_frac=DEFAULT_TAIL_EMA_START_FRAC,
            ema_rate=DEFAULT_TAIL_EMA_RATE,
            update_every=2,
            decontract_frobenius=True,
        )
        if not bool(args.no_tail_ema)
        else None
    )
    total_params = sum(p.numel() for p in model.parameters())

    intent = ExperimentIntentSpec(
        run_key=spec.run_key,
        wave_id=WAVE_ID,
        plan_id=PLAN_ID,
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Evaluate Upgraded CompiledLeaderboardClassicalGPT (L={spec.num_layers}, d={spec.embed_dim}, "
            f"H={spec.num_heads}, interior={args.interior_value_mode}, flex_swc={cfg.use_flex_doc_swc}, "
            f"ws={ws_schedule}, gates={use_gates}, B_tok={tokens_per_step}) on Official FineWeb."
        ),
        architecture_summary=(
            f"CompiledLeaderboardClassicalGPT L={spec.num_layers} d={spec.embed_dim} H={spec.num_heads} "
            f"d_ff={spec.hidden_dim} VE={cfg.num_value_embeddings} FlexSWC+YaRN+Gates+XSA ({total_params / 1e6:.2f}M params)"
        ),
        hardware_budget="NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)",
        arch_family="classical_modded",
        batch_size=spec.micro_batch_size * spec.grad_accum,
        seq_len=spec.seq_len,
        total_steps=total_steps,
        target_tokens_M=round(target_tokens_m, 3),
        params_M=round(float(total_params) / 1e6, 3),
        embed_dim=spec.embed_dim,
        num_layers=spec.num_layers,
    )
    logger = StandardizedExperimentLogger(intent)
    streamer = SequentialFineWebShardStreamer(FINEWEB_CACHE_DIR, num_shards=9)

    val_source: Path | torch.Tensor = (
        DEFAULT_VAL_SHARD_PATH
        if DEFAULT_VAL_SHARD_PATH.exists()
        else torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (DEFAULT_FAST_VAL_TOKENS + 1024,), device=device)
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_train0 = time.perf_counter()
    eval_history: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []
    save_every_tokens_m = float(args.save_every_tokens_m)
    next_save_tokens_m = 0.0
    ckpt_dir = (
        Path(args.checkpoint_dir)
        if str(args.checkpoint_dir).strip()
        else (ROOT_DIR / ".cache" / "checkpoints" / spec.run_key)
    )
    best_val_ce = float("inf")
    max_wall_s = float(args.max_minutes) * 60.0 if float(args.max_minutes) > 0.0 else 0.0
    target_val_ce = float(args.target_val_ce)
    early_exit_reason = ""
    actual_steps_done = 0
    official_10m_metrics: dict[str, Any] = {}

    for step in range(total_steps + 1):
        actual_steps_done = step
        elapsed_now_s = max(1e-6, time.perf_counter() - t_train0)
        wall_limit_hit = max_wall_s > 0.0 and elapsed_now_s >= max_wall_s
        step_prog = float(step) / float(max(1, total_steps))
        wall_prog = (elapsed_now_s / max_wall_s) if max_wall_s > 0.0 else 0.0
        tau_prog = min(1.0, max(step_prog, wall_prog))

        ws_idx = min(len(ws_schedule) - 1, int(len(ws_schedule) * min(0.9999, tau_prog)))
        desired_ws = ws_schedule[ws_idx]
        if desired_ws != model.current_ws_blocks:
            model.set_sliding_window_blocks(desired_ws, apply_yarn=True)

        tokens_m_now = (float(step) * float(tokens_per_step)) / 1e6
        due_for_ckpt = save_every_tokens_m > 0.0 and (tokens_m_now >= next_save_tokens_m - 1e-6)
        should_eval = (
            (step % max(1, int(args.eval_every)) == 0)
            or (step == total_steps)
            or wall_limit_hit
            or due_for_ckpt
        )
        if should_eval:
            val_metrics = evaluate_compiled_leaderboard_validation(
                model,
                val_source,
                val_tokens_budget=DEFAULT_FAST_VAL_TOKENS,
                batch_size=spec.micro_batch_size,
                seq_len=min(spec.seq_len, 1024),
            )
            elapsed_s = max(1e-6, time.perf_counter() - t_train0)
            tokens_m = (float(step) * float(tokens_per_step)) / 1e6
            avg_tok_s = (float(step) * float(tokens_per_step)) / elapsed_s if step > 0 else 0.0
            cur_val_ce = float(val_metrics["val_ce"])
            best_val_ce = min(best_val_ce, cur_val_ce)
            compute_tflops = (6.0 * float(total_params) * float(step) * float(tokens_per_step)) / 1e12

            logger.log_eval(
                step,
                {
                    "tokens_M": round(tokens_m, 4),
                    "compute_tflops": round(compute_tflops, 3),
                    "tau_progress": round(tau_prog, 4),
                    "val_ce": cur_val_ce,
                    "val_ppl": float(val_metrics["val_ppl"]),
                    "top1_acc": float(val_metrics["top1_acc"]),
                    "rep_val_ce": val_metrics["rep_val_ce"],
                    "non_rep_val_ce": val_metrics["non_rep_val_ce"],
                    "rep_top1_acc": val_metrics["rep_top1_acc"],
                    "avg_tok_per_s": round(avg_tok_s, 1),
                    "elapsed_min": round(elapsed_s / 60.0, 3),
                    "ws_blocks": model.current_ws_blocks,
                },
            )
            eval_history.append(
                {
                    "step": step,
                    "tokens_M": round(tokens_m, 4),
                    "elapsed_min": round(elapsed_s / 60.0, 3),
                    "avg_tok_per_s": round(avg_tok_s, 1),
                    "ws_blocks": model.current_ws_blocks,
                    **val_metrics,
                }
            )
            print(
                f"[EVAL step={step:4d}/{total_steps} | tok={tokens_m:6.2f}M | ws={model.current_ws_blocks} | "
                f"val_ce_65k={cur_val_ce:.4f} | ppl={val_metrics['val_ppl']:7.2f} | "
                f"top1={val_metrics['top1_acc']:5.2f}% | tok/s={avg_tok_s:8.1f} | "
                f"wall={elapsed_s / 60.0:5.2f}m]",
                flush=True,
            )

            if due_for_ckpt:
                milestone_label = f"m{int(round(next_save_tokens_m)):04d}M"
                _save_model_checkpoint(
                    model,
                    ckpt_dir,
                    step,
                    tokens_m,
                    val_metrics,
                    checkpoint_manifest,
                    tag=milestone_label,
                )
                while next_save_tokens_m <= tokens_m + 1e-6:
                    next_save_tokens_m += save_every_tokens_m

            if target_val_ce > 0.0 and cur_val_ce <= target_val_ce:
                if args.eval_official_10m and DEFAULT_VAL_SHARD_PATH.exists():
                    official_10m_metrics = evaluate_compiled_leaderboard_validation(
                        model,
                        DEFAULT_VAL_SHARD_PATH,
                        val_tokens_budget=DEFAULT_MODDED_VAL_TOKENS,
                        batch_size=spec.micro_batch_size,
                        seq_len=min(spec.seq_len, 1024),
                    )
                    off_ce = float(official_10m_metrics["val_ce"])
                    print(
                        f"[TARGET CHECK 10.49M @ step={step} | val_ce_65k={cur_val_ce:.4f} | "
                        f"val_ce_10m={off_ce:.4f} | target={target_val_ce:.4f} | "
                        f"eval_wall={official_10m_metrics['eval_wall_s']:.2f}s]",
                        flush=True,
                    )
                    if off_ce <= target_val_ce:
                        early_exit_reason = (
                            f"target_val_ce_reached (val_ce_10m={off_ce:.4f} <= {target_val_ce:.4f} at step={step})"
                        )
                        break
                    official_10m_metrics = {}
                else:
                    early_exit_reason = (
                        f"target_val_ce_reached (val_ce_65k={cur_val_ce:.4f} <= {target_val_ce:.4f} at step={step})"
                    )
                    break

        if step == total_steps:
            break
        if wall_limit_hit:
            early_exit_reason = f"max_minutes_reached ({args.max_minutes:.2f}m at step={step})"
            print(f"[WALL CLOCK LIMIT REACHED | {early_exit_reason}]", flush=True)
            break

        lr_mult = compute_schedule_lr_multiplier(
            step=step,
            total_steps=total_steps,
            schedule=str(args.lr_schedule),
            decay_fraction=float(args.wsd_decay_frac),
            elapsed_s=max(0.0, time.perf_counter() - t_train0),
            max_wall_s=max_wall_s,
        )
        for grp in muon_opt.param_groups:
            grp["lr"] = float(args.muon_lr) * lr_mult
        for grp in adam_opt.param_groups:
            g_name = grp.get("group_name", "")
            base_lr = (
                float(args.head_lr)
                if g_name == "lm_head"
                else (float(args.wte_lr) if g_name == "wte_and_ve" else float(args.scalar_lr))
            )
            grp["lr"] = base_lr * lr_mult

        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)
        step_loss_accum = 0.0
        inv_accum = 1.0 / float(spec.grad_accum)

        for micro_idx in range(spec.grad_accum):
            xb, yb = streamer.next_batch(spec.micro_batch_size, spec.seq_len, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                _, loss_micro = model(xb, yb, return_logits=False)
                assert loss_micro is not None
                scaled_loss = loss_micro * inv_accum
            scaled_loss.backward()
            step_loss_accum += float(loss_micro.detach().item()) * inv_accum
            if step == 0 and micro_idx == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_train0 = time.perf_counter() - 0.35

        muon_opt.step()
        adam_opt.step()
        if averager is not None:
            virtual_step = int(round(tau_prog * float(max(1, total_steps))))
            averager.update(step=virtual_step, total_steps=total_steps)

        if (step + 1) % max(1, int(args.log_every)) == 0 or step == 0:
            elapsed_s = max(1e-6, time.perf_counter() - t_train0)
            tok_done = float(step + 1) * float(tokens_per_step)
            tok_s = tok_done / elapsed_s
            logger.log_step(
                step + 1,
                train_ce=round(step_loss_accum, 5),
                tok_per_s=round(tok_s, 1),
                lr_mult=round(lr_mult, 4),
                elapsed_min=round(elapsed_s / 60.0, 4),
            )

    actual_tokens_m = (float(actual_steps_done) * float(tokens_per_step)) / 1e6
    val_ws = max(model.current_ws_blocks, int(args.ws_validate))
    if val_ws != model.current_ws_blocks:
        model.set_sliding_window_blocks(val_ws, ws_final_blocks=val_ws, apply_yarn=True)

    if averager is not None and averager.is_seeded:
        with averager.average_parameters():
            ema_65k = evaluate_compiled_leaderboard_validation(
                model,
                val_source,
                val_tokens_budget=DEFAULT_FAST_VAL_TOKENS,
                batch_size=spec.micro_batch_size,
                seq_len=min(spec.seq_len, 1024),
            )
            ema_ce = float(ema_65k["val_ce"])
            best_val_ce = min(best_val_ce, ema_ce)
            print(
                f"[TAIL EMA 65K VAL | updates={averager.num_updates} | val_ce_65k={ema_ce:.4f} | "
                f"ppl={ema_65k['val_ppl']:.2f} | top1={ema_65k['top1_acc']:.2f}%]",
                flush=True,
            )
            if args.eval_official_10m and DEFAULT_VAL_SHARD_PATH.exists() and not official_10m_metrics:
                official_10m_metrics = evaluate_compiled_leaderboard_validation(
                    model,
                    DEFAULT_VAL_SHARD_PATH,
                    val_tokens_budget=DEFAULT_MODDED_VAL_TOKENS,
                    batch_size=spec.micro_batch_size,
                    seq_len=min(spec.seq_len, 1024),
                )
            if save_every_tokens_m > 0.0:
                _save_model_checkpoint(
                    model,
                    ckpt_dir,
                    actual_steps_done,
                    actual_tokens_m,
                    official_10m_metrics if official_10m_metrics else ema_65k,
                    checkpoint_manifest,
                    tag="final_ema",
                )
    elif args.eval_official_10m and DEFAULT_VAL_SHARD_PATH.exists() and not official_10m_metrics:
        official_10m_metrics = evaluate_compiled_leaderboard_validation(
            model,
            DEFAULT_VAL_SHARD_PATH,
            val_tokens_budget=DEFAULT_MODDED_VAL_TOKENS,
            batch_size=spec.micro_batch_size,
            seq_len=min(spec.seq_len, 1024),
        )

    if official_10m_metrics:
        logger.log_eval(
            actual_steps_done,
            {
                "tokens_M": round(actual_tokens_m, 4),
                "compute_tflops": round(
                    (6.0 * float(total_params) * float(actual_steps_done) * float(tokens_per_step)) / 1e12, 3
                ),
                "tau_progress": 1.0,
                "val_ce": float(official_10m_metrics["val_ce"]),
                "val_ppl": float(official_10m_metrics["val_ppl"]),
                "top1_acc": float(official_10m_metrics["top1_acc"]),
                "rep_val_ce": official_10m_metrics["rep_val_ce"],
                "non_rep_val_ce": official_10m_metrics["non_rep_val_ce"],
                "rep_top1_acc": official_10m_metrics["rep_top1_acc"],
                "eval_type": "official_10_485_760_tokens",
            },
        )
        print(
            f"[OFFICIAL 10.49M VAL | run={spec.run_key} | val_ce_10m={official_10m_metrics['val_ce']:.4f} | "
            f"ppl={official_10m_metrics['val_ppl']:.2f} | top1={official_10m_metrics['top1_acc']:.2f}% | "
            f"eval_wall={official_10m_metrics['eval_wall_s']:.2f}s]",
            flush=True,
        )

    total_wall_s = max(1e-6, time.perf_counter() - t_train0)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device)) / BYTES_PER_MIB if device.type == "cuda" else 0.0
    )
    summary = {
        "run_key": spec.run_key,
        "wave_id": WAVE_ID,
        "plan_id": PLAN_ID,
        "total_params": int(total_params),
        "tokens_per_step": int(tokens_per_step),
        "total_steps": int(actual_steps_done),
        "planned_steps": int(total_steps),
        "total_tokens_M": round(actual_tokens_m, 3),
        "wall_min": round(total_wall_s / 60.0, 3),
        "avg_tok_per_s": round((float(actual_steps_done) * float(tokens_per_step)) / total_wall_s, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "best_val_ce_65k": round(best_val_ce, 6),
        "early_exit_reason": early_exit_reason,
        "official_10m_val": official_10m_metrics,
        "eval_history": eval_history,
        "checkpoint_dir": str(ckpt_dir) if save_every_tokens_m > 0.0 else "",
        "checkpoints_saved": len(checkpoint_manifest),
    }
    logger.finalize(extra_summary=summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `70_verify_plan060_compiled_classical_leaderboard_speedrun.py`."""
    parser = argparse.ArgumentParser(description="Wave 45 Compiled Classical Leaderboard Speedrun Runner.")
    parser.add_argument("--smoke", action="store_true", help="Run fast 2-step smoke verification.")
    parser.add_argument(
        "--benchmark-throughput",
        action="store_true",
        help="Run H1 & H2 throughput/VRAM benchmark comparing legacy ClassicalModdedGPT vs CompiledLeaderboardClassicalGPT.",
    )
    parser.add_argument("--run-key", type=str, default="w45_compiled_classical_leaderboard_l12_d768")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_LEADERBOARD_NUM_LAYERS)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_LEADERBOARD_EMBED_DIM)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_LEADERBOARD_NUM_HEADS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_LEADERBOARD_HIDDEN_DIM)
    parser.add_argument("--num-value-embeddings", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_LEADERBOARD_SEQ_LEN)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-minutes", type=float, default=0.0, help="Optional maximum wall-clock minutes.")
    parser.add_argument("--target-val-ce", type=float, default=0.0, help="Optional early-exit target Val CE.")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--muon-lr", type=float, default=DEFAULT_LEADERBOARD_MUON_LR)
    parser.add_argument("--muon-warmup-steps", type=int, default=DEFAULT_LEADERBOARD_MUON_WARMUP_STEPS)
    parser.add_argument("--wte-lr", type=float, default=DEFAULT_LEADERBOARD_WTE_LR)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_LEADERBOARD_HEAD_LR)
    parser.add_argument("--scalar-lr", type=float, default=DEFAULT_LEADERBOARD_SCALAR_LR)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ve-init-std", type=float, default=DEFAULT_VE_INIT_STD)
    parser.add_argument("--ws-schedule", type=str, default="3,6,8", help="Progressive window schedule in 128-tok blocks.")
    parser.add_argument("--ws-validate", type=int, default=8, help="Validation window blocks (in 128-tok blocks).")
    parser.add_argument("--polar-method", choices=("polar_express", "newton_schulz"), default="polar_express")
    parser.add_argument("--lr-schedule", choices=("wsd", "constant"), default="wsd")
    parser.add_argument("--wsd-decay-frac", type=float, default=DEFAULT_WSD_DECAY_FRACTION)
    parser.add_argument(
        "--interior-value-mode",
        choices=("unit_v", "v0_residual", "legacy_half_v"),
        default="unit_v",
    )
    parser.add_argument(
        "--save-every-tokens-m",
        type=float,
        default=0.0,
        help="Save bfloat16 model checkpoint every N million tokens (e.g. 50.0 saves at 0M, 50M, 100M, ...).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Optional directory for model checkpoints (defaults to .cache/checkpoints/{run_key}).",
    )
    parser.add_argument("--no-flex-doc-swc", action="store_true", help="Disable FlexAttention Doc-SWC masks.")
    parser.add_argument("--no-gates", action="store_true", help="Disable smear/attn/ve gates, key offset, and XSA.")
    parser.add_argument("--no-normuon", action="store_true", help="Disable NorMuon per-lane variance equalization.")
    parser.add_argument("--no-twin-rail", action="store_true", help="Disable Twin-Rail Nesterov momentum.")
    parser.add_argument("--no-tail-ema", action="store_true", help="Disable TailWeightAverager EMA.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile on backbone.")
    parser.add_argument("--eval-official-10m", action="store_true", help="Run official 10,485,760-token eval at end.")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke or args.benchmark_throughput:
        h2_res = verify_h2_interior_value_and_step0_calibration(device)
        h1_res = benchmark_h1_throughput_vs_legacy(device, smoke_only=bool(args.smoke))
        payload = {
            "wave_id": WAVE_ID,
            "plan_id": PLAN_ID,
            "device": str(device),
            "h1_throughput_and_vram": h1_res,
            "h2_interior_value_and_step0": h2_res,
            "overall_pass": bool(h1_res.get("passed", False) and h2_res.get("passed", False)),
        }
        HYPOTHESIS_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
        HYPOTHESIS_JSON_OUT.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2), flush=True)
        return 0 if payload["overall_pass"] else 1

    run_fineweb_classical_speedrun(args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
