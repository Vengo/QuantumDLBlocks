"""Wave 52A (`EXPERIMENT_PLAN_065.md`) Runner: Surgical Quantum Front-End Graft Tournament on Frozen Classical GPT.

Executes the controlled tournament (`Arms T0..T11`) and scaling experiments grafting a trainable
Quantum Front-End (`q_wte` + Quantum Layer 0) onto a frozen pretrained `CompiledLeaderboardClassicalGPT`
teacher (`ckpt_step01145_0300M_m0300M.pt`, reference `val_ce = 3.5842`, or `ckpt_step00191_0050M_m0050M.pt`,
reference `val_ce = 4.2466`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from qdlblocks.layers.isometric_born_trace_readout import evaluate_readout_divergence_batch
from qdlblocks.layers.quantum_frontend_graft import (
    DEFAULT_GRAFT_INTERFACE_ALIGN_WEIGHT,
    DEFAULT_GRAFT_SCALAR_LR,
    DEFAULT_GRAFT_STIEFEL_LR,
    DEFAULT_GRAFT_UNITARY_LR,
    DEFAULT_GRAFT_WTE_LR,
    DEFAULT_TEACHER_300M_CKPT_PATH,
    FrozenTrunkQuantumFrontEndGPT,
    QuantumAttnInteriorMode,
    QuantumFrontEndGraftConfig,
    QuantumReplacementScope,
    QuantumStateIOMode,
    QuantumWavefunctionShape,
    build_frozen_trunk_quantum_frontend_optimizers,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
CACHE_DIR: Path = REPO_ROOT / ".cache"
FINEWEB_DIR: Path = CACHE_DIR / "fineweb_100M"
DEFAULT_JSON_OUT: Path = CACHE_DIR / "w52a_plan065_tournament.json"
SHARD_HEADER_INTS: int = 256
FAST_SCREENING_VAL_TOKENS: int = 65_536
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_SEQ_LEN: int = 1024
DEFAULT_TOURNAMENT_STEPS: int = 500
DEFAULT_EVAL_EVERY: int = 100


class MultiShardTokenStream:
    """Streams contiguous `(B, T)` token batches sequentially across local FineWeb training shards."""

    def __init__(self, shard_paths: list[Path], batch_size: int, seq_len: int, vocab_size: int) -> None:
        if not shard_paths:
            raise FileNotFoundError("No FineWeb training shards found.")
        self.shard_paths = list(shard_paths)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.span = self.batch_size * (self.seq_len + 1)
        self.shard_idx = 0
        self.offset = self.span
        self._tokens = self._load_shard(self.shard_paths[0])

    @staticmethod
    def _load_shard(path: Path) -> np.ndarray:
        raw = np.memmap(path, dtype=np.uint16, mode="r")
        header = np.fromfile(path, dtype=np.int32, count=SHARD_HEADER_INTS)
        if len(header) >= 3 and int(header[0]) == 20240520:
            return raw[SHARD_HEADER_INTS * 2 :]
        return raw

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return next `(idx, targets)` batch of shape `(B, T)` on `device`."""
        if self.offset + self.span > len(self._tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.shard_paths)
            self._tokens = self._load_shard(self.shard_paths[self.shard_idx])
            self.offset = 0
        chunk = np.asarray(self._tokens[self.offset : self.offset + self.span], dtype=np.int64)
        self.offset += self.span
        t_cpu = torch.from_numpy(chunk).clamp_(0, self.vocab_size - 1).view(self.batch_size, self.seq_len + 1)
        x = t_cpu[:, :-1].contiguous().to(device, non_blocking=True)
        y = t_cpu[:, 1:].contiguous().to(device, non_blocking=True)
        return x, y


VALID_FINEWEB_VOCAB_SIZE: int = 50_257


def load_validation_batches(
    val_path: Path,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    total_val_tokens: int = FAST_SCREENING_VAL_TOKENS,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load deterministic validation `(idx, targets)` batches matching `evaluate_compiled_leaderboard_validation`."""
    tokens = MultiShardTokenStream._load_shard(val_path)
    n_avail = min(len(tokens) - 1, int(total_val_tokens))
    n_seqs = max(1, n_avail // int(seq_len))
    total_eval_tokens = n_seqs * int(seq_len)
    flat_cpu = torch.from_numpy(np.asarray(tokens[: total_eval_tokens + 1], dtype=np.int64)).clamp_(0, vocab_size - 1)
    x_all = flat_cpu[:total_eval_tokens].view(n_seqs, int(seq_len))
    y_all = flat_cpu[1 : total_eval_tokens + 1].view(n_seqs, int(seq_len))
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for b_start in range(0, n_seqs, max(1, int(batch_size))):
        b_end = min(n_seqs, b_start + max(1, int(batch_size)))
        batches.append((x_all[b_start:b_end].contiguous(), y_all[b_start:b_end].contiguous()))
    return batches


@torch.no_grad()
def evaluate_graft_model(
    model: FrozenTrunkQuantumFrontEndGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate validation cross-entropy, Top-1/Top-5 accuracy, and quantum front-end telemetry."""
    was_training = model.training
    model.eval()
    accum: dict[str, float] = {}
    v_valid = min(VALID_FINEWEB_VOCAB_SIZE, model.vocab_size)
    use_bf16 = device.type == "cuda"
    for x_cpu, y_cpu in val_batches:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(x, return_logits=True)
        assert logits is not None
        b_metrics = evaluate_readout_divergence_batch(logits[:, :, :v_valid], y)
        for k, v in b_metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v)
    n_b = float(max(1, len(val_batches)))
    out = {k: v / n_b for k, v in accum.items()}
    out.update(model.compute_quantum_graft_telemetry())
    if was_training:
        model.train()
    return out


def resolve_tournament_arm_config(
    arm_id: str,
    warmup_steps_override: int | None = None,
) -> QuantumFrontEndGraftConfig:
    """Map tournament arm identifier (`T0..T11`) to its `QuantumFrontEndGraftConfig`."""
    key = arm_id.strip().upper()
    presets: dict[str, QuantumFrontEndGraftConfig] = {
        "T0_TEACHER_REF": QuantumFrontEndGraftConfig(
            replacement_scope=QuantumReplacementScope.TEACHER_PASS_THROUGH
        ),
        "T1_WP_24X2X16_BASE": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T2_WP_1X2X256": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x256,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T3_WP_1X2X512": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x512,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T4_QUBIT_256X2X2": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_256x2x2,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T5_DM_IO_24X2X16": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T6_HYBRID_OBS_OUT": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_DENSITY_OBS_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T7_INT_LOCAL_V": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T8_INT_KV_CROSS": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.KV_CROSS_OUTER_ON_Q,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T9_INT_PRE_ATTN": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.PRE_ATTN_LOCAL_OUTER_ON_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        ),
        "T10_SCOPE_KEEP_MLP0": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        ),
        "T11_ALIGN_WARMUP": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
            interface_warmup_steps=150,
            interface_align_weight=DEFAULT_GRAFT_INTERFACE_ALIGN_WEIGHT,
        ),
    }
    if key not in presets:
        raise ValueError(f"Unknown tournament arm: {arm_id}. Valid arms: {list(presets.keys())}")
    cfg = presets[key]
    if warmup_steps_override is not None:
        cfg.interface_warmup_steps = int(warmup_steps_override)
    return cfg


def run_single_graft_arm(
    arm_id: str,
    graft_cfg: QuantumFrontEndGraftConfig,
    teacher_ckpt: Path,
    steps: int,
    eval_every: int,
    batch_size: int,
    grad_accum: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate a single surgical quantum front-end graft arm (`<= 8` args)."""
    torch.manual_seed(2026065)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(2026065)

    model = FrozenTrunkQuantumFrontEndGPT.from_teacher_checkpoint(
        ckpt_path=teacher_ckpt,
        graft_cfg=graft_cfg,
        device=device,
    )
    train_shards = sorted(FINEWEB_DIR.glob("fineweb_train_*.bin"))
    val_path = FINEWEB_DIR / "fineweb_val_000000.bin"
    val_batches = load_validation_batches(val_path, batch_size=batch_size, seq_len=DEFAULT_SEQ_LEN, vocab_size=model.vocab_size)

    active_q_params = sum(p.numel() for p in model.active_quantum_parameters())
    q_l0_params = (
        sum(p.numel() for p in model.quantum_block0.parameters() if p.requires_grad)
        if model.quantum_block0 is not None
        else 0
    )

    if graft_cfg.replacement_scope == QuantumReplacementScope.TEACHER_PASS_THROUGH or steps <= 0:
        val_0 = evaluate_graft_model(model, val_batches, device)
        print(
            f"[{arm_id}] step=0 val_ce={val_0['val_ce']:.4f} top1={val_0['top1_acc']*100:.2f}% "
            f"top5={val_0['top5_acc']*100:.2f}% (Teacher Reference)"
        )
        return {
            "arm_id": arm_id,
            "config": {
                "wf_shape": graft_cfg.wf_shape.value,
                "state_io_mode": graft_cfg.state_io_mode.value,
                "attn_interior_mode": graft_cfg.attn_interior_mode.value,
                "replacement_scope": graft_cfg.replacement_scope.value,
                "interface_warmup_steps": graft_cfg.interface_warmup_steps,
                "active_quantum_params": active_q_params,
                "quantum_l0_params": q_l0_params,
            },
            "final": val_0,
            "history": [{"step": 0, **val_0}],
        }

    opt = build_frozen_trunk_quantum_frontend_optimizers(model)
    stream = MultiShardTokenStream(train_shards, batch_size=batch_size, seq_len=DEFAULT_SEQ_LEN, vocab_size=model.vocab_size)
    tokens_per_step = batch_size * DEFAULT_SEQ_LEN * grad_accum

    val_0 = evaluate_graft_model(model, val_batches, device)
    print(
        f"[{arm_id}] step=0 tok=0.00M val_ce={val_0['val_ce']:.4f} top1={val_0['top1_acc']*100:.2f}% "
        f"top5={val_0['top5_acc']*100:.2f}% q_l0_params={q_l0_params/1e6:.2f}M"
    )
    history: list[dict[str, Any]] = [{"step": 0, "tokens": 0, **val_0}]
    t0 = time.perf_counter()
    ema_loss = float(val_0["val_ce"])

    for step in range(1, steps + 1):
        progress = float(step - 1) / float(max(1, steps))
        lr_mult = 1.0 if progress < 0.70 else 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * (progress - 0.70) / 0.30))
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        last_idx: torch.Tensor | None = None
        for _ in range(grad_accum):
            idx, targets = stream.next_batch(device)
            last_idx = idx
            _, loss = model(idx, targets=targets)
            assert loss is not None
            (loss / float(grad_accum)).backward()
            step_loss += float(loss.detach().item()) / float(grad_accum)
        assert last_idx is not None
        opt.step(idx=last_idx, lr_mult=lr_mult)
        ema_loss = 0.95 * ema_loss + 0.05 * step_loss

        if step % eval_every == 0 or step == steps:
            elapsed = max(1e-6, time.perf_counter() - t0)
            tok_done = step * tokens_per_step
            tok_per_s = tok_done / elapsed
            val_m = evaluate_graft_model(model, val_batches, device)
            rec = {
                "step": step,
                "tokens": tok_done,
                "tokens_m": tok_done / 1e6,
                "train_ce_ema": ema_loss,
                "tok_per_s": tok_per_s,
                **val_m,
            }
            history.append(rec)
            print(
                f"[{arm_id}] step={step}/{steps} ({tok_done/1e6:.2f}M tok, {tok_per_s:.0f} tok/s) "
                f"train_ce={ema_loss:.4f} val_ce={val_m['val_ce']:.4f} "
                f"top1={val_m['top1_acc']*100:.2f}% top5={val_m['top5_acc']*100:.2f}% "
                f"q_pur={val_m.get('q_l0_purity', 1.0):.3f} q_rank={val_m.get('q_l0_eff_rank', 1.0):.2f}"
            )

    return {
        "arm_id": arm_id,
        "config": {
            "wf_shape": graft_cfg.wf_shape.value,
            "state_io_mode": graft_cfg.state_io_mode.value,
            "attn_interior_mode": graft_cfg.attn_interior_mode.value,
            "replacement_scope": graft_cfg.replacement_scope.value,
            "interface_warmup_steps": graft_cfg.interface_warmup_steps,
            "active_quantum_params": active_q_params,
            "quantum_l0_params": q_l0_params,
        },
        "steps": steps,
        "tokens_processed": steps * tokens_per_step,
        "final": history[-1],
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 52A (Plan 065) Frozen-Trunk Quantum Front-End Graft Runner")
    parser.add_argument("--arm", type=str, default="T1_WP_24x2x16_BASE")
    parser.add_argument("--arms", type=str, default="")
    parser.add_argument("--teacher-ckpt", type=Path, default=DEFAULT_TEACHER_300M_CKPT_PATH)
    parser.add_argument("--steps", type=int, default=DEFAULT_TOURNAMENT_STEPS)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--interface-warmup-steps", type=int, default=-1)
    parser.add_argument("--wf-shape", type=str, default="")
    parser.add_argument("--state-io-mode", type=str, default="")
    parser.add_argument("--attn-interior-mode", type=str, default="")
    parser.add_argument("--replacement-scope", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    warmup_override = args.interface_warmup_steps if args.interface_warmup_steps >= 0 else None

    if args.arms:
        arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
        results: dict[str, Any] = {}
        for arm_id in arm_list:
            cfg = resolve_tournament_arm_config(arm_id, warmup_steps_override=warmup_override)
            res = run_single_graft_arm(
                arm_id=arm_id,
                graft_cfg=cfg,
                teacher_ckpt=args.teacher_ckpt,
                steps=args.steps,
                eval_every=args.eval_every,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                device=device,
            )
            results[arm_id] = res
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return

    cfg = resolve_tournament_arm_config(args.arm, warmup_steps_override=warmup_override)
    if args.wf_shape:
        cfg.wf_shape = QuantumWavefunctionShape(args.wf_shape)
    if args.state_io_mode:
        cfg.state_io_mode = QuantumStateIOMode(args.state_io_mode)
    if args.attn_interior_mode:
        cfg.attn_interior_mode = QuantumAttnInteriorMode(args.attn_interior_mode)
    if args.replacement_scope:
        cfg.replacement_scope = QuantumReplacementScope(args.replacement_scope)

    res = run_single_graft_arm(
        arm_id=args.arm,
        graft_cfg=cfg,
        teacher_ckpt=args.teacher_ckpt,
        steps=args.steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        device=device,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
