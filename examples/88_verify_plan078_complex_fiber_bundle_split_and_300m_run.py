#!/usr/bin/env python3
r"""Verification Runner for `EXPERIMENT_PLAN_078.md` (`Wave 57`).

Executes:
1. **Part 1 — Cold-Start Complex Wavefunction Fiber-Bundle Split & Readout Ablation (`N78_1..N78_3`)**:
   - **`N78_1_FIBER_SPLIT_11SEM_1ID_PKT32`**:
     4D Complex Wavefunction `(50304, 12, 2, 32)` $\cong$ `(50304, 12, 64)` joint `64D` complex sphere layout with
     **`11 Semantic + 1 Identity` (`704D / 64D`)** split (`COMPLEX_FIBER_BUNDLE_11SEM_1ID`), Support-Preserving Directed
     Multi-Lag ($k \in \{1,2,4,8\}$) PPMI CountSketch (`Forward -> Re`, `Reverse -> Im`, Active-Only Centering,
     Haar Unitary Mode Balancing $U_M \in U(11)$, Complex Hermitian Gram-Schmidt orthogonal blend), `q_value_embeds[0..2]`
     fiber-bundle initialization, and `id_packet_lr_scale = 0.25` with `STANDARD_LINEAR_READOUT`.
   - **`N78_2_FIBER_SPLIT_10SEM_2ID_DYNAMIC_GATE_PKT32`**:
     4D Complex Wavefunction `(50304, 12, 2, 32)` with **`10 Semantic + 2 Identity` (`640D / 128D`)** split
     (`COMPLEX_FIBER_BUNDLE_10SEM_2ID`), `id_packet_lr_scale = 0.25`, and `DYNAMIC_COMPLEX_PACKET_GATED_READOUT`
     (`DynamicComplexPacketReadoutModulator`).
   - **`N78_3_FIBER_SPLIT_11SEM_1ID_STEP0_CALIB_PKT32`**:
     Same as `N78_1` plus `DYNAMIC_COMPLEX_PACKET_GATED_READOUT` and Step-0 training-shard Ridge-Regression readout
     transition calibration (`calibrate_cold_start_readout_from_shard_` over `fineweb_train_000001.bin`).
2. **Part 2 — Full `300M`-Token Cold-Start From-Scratch Run (`N78_4_LARGE_FROM_SCRATCH_300M_FIBER_SPLIT_QUANTUM_GPT`)**:
   - Full `300.15M`-token cold-start from-scratch 12-layer Quantum Wavepacket GPT (`1,145` steps at `262,144` tok/step
     across `2x NVIDIA GeForce RTX 3090` via `nccl` DDP, `ws_blocks: 3 -> 6` at `150M` tokens, Tail EMA at `tau >= 0.70`)
     using the winning Fiber-Bundle configuration from Part 1.
"""

from __future__ import annotations

import argparse
import gc
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "examples"))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from qdlblocks.kernels.fused_quantum_wavepacket_attn import (  # noqa: E402
    DEFAULT_LOW_RANK_CHOLESKY_K,
    QuantumAttnKernelMode,
)
from qdlblocks.layers.complex_fiber_bundle_init import (  # noqa: E402
    DEFAULT_COMPLEX_PACKET_COUNT,
    DEFAULT_COMPLEX_PACKET_HILBERT_DIM,
    DEFAULT_EB_ALPHA_MAX,
    DEFAULT_EB_PRIOR_COUNT_K0,
    DEFAULT_FIBER_SEMANTIC_PACKETS_10,
    DEFAULT_FIBER_SEMANTIC_PACKETS_11,
    DEFAULT_IDENTITY_GATE_BIAS_INIT,
    DEFAULT_IDENTITY_PACKET_LR_SCALE,
    DEFAULT_SEMANTIC_GATE_BIAS_INIT,
    QuantumPacketReadoutMode,
    build_or_load_directed_complex_ppmi_sketch,
    calibrate_cold_start_readout_from_shard_,
)
from qdlblocks.layers.isometric_born_trace_readout import evaluate_readout_divergence_batch  # noqa: E402
from qdlblocks.layers.quantum_frontend_graft import (  # noqa: E402
    DEFAULT_TEACHER_50M_CKPT_PATH,
    DEFAULT_TEACHER_300M_CKPT_PATH,
    QuantumAttnInteriorMode,
    QuantumQKPhaseMode,
    QuantumRecoveryScheduleMode,
    QuantumSelfOrthoMode,
    QuantumStateIOMode,
    QuantumStiefelProjMode,
    QuantumValueEmbedCouplingMode,
    QuantumWavefunctionShape,
    QuantumWteInitMode,
)
from qdlblocks.layers.quantum_statmech_channels import (  # noqa: E402
    PurificationMode,
    QuantumChannelMode,
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (  # noqa: E402
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)
from qdlblocks.models.quantum_full_backbone_graft_gpt import (  # noqa: E402
    DEFAULT_FROM_SCRATCH_SCALAR_LR,
    DEFAULT_FROM_SCRATCH_STIEFEL_LR,
    DEFAULT_FROM_SCRATCH_UNITARY_LR,
    DEFAULT_FROM_SCRATCH_WTE_LR,
    DEFAULT_QUANTUM_TAIL_EMA_BLEND,
    DEFAULT_QUANTUM_TAIL_EMA_RATE,
    DEFAULT_QUANTUM_TAIL_EMA_START_FRAC,
    DEFAULT_TEACHER_0M_CKPT_PATH,
    DEFAULT_TEACHER_100M_CKPT_PATH,
    QuantumBackboneDepthScope,
    QuantumDepthLRMode,
    QuantumDocSWCMode,
    QuantumFullBackboneGraftConfig,
    QuantumFullBackboneGraftGPT,
    QuantumKetAmplitudeMode,
    QuantumMLPChannelMode,
    build_quantum_full_backbone_optimizers,
    compute_quantum_backbone_lr_multiplier,
    retract_quantum_backbone_manifolds_,
)
from qdlblocks.optim.tail_ema import TailWeightAverager  # noqa: E402

_ex75 = import_module("75_verify_plan065_frozen_trunk_quantum_frontend_tournament")
FINEWEB_DIR: Path = _ex75.FINEWEB_DIR
DEFAULT_SEQ_LEN: int = _ex75.DEFAULT_SEQ_LEN
VALID_FINEWEB_VOCAB_SIZE: int = _ex75.VALID_FINEWEB_VOCAB_SIZE
MultiShardTokenStream = _ex75.MultiShardTokenStream
load_validation_batches = _ex75.load_validation_batches

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_078.md (Section 2.5)
# ==============================================================================
DEFAULT_OUT_JSON: Path = _REPO_ROOT / ".cache" / "w57_plan078_complex_fiber_bundle_split_and_300m_run.json"
DEFAULT_PART1_50M_STEPS: int = 191
DEFAULT_FROM_SCRATCH_300M_STEPS: int = 1145
DEFAULT_FROM_SCRATCH_GRAD_ACCUM: int = 32
DEFAULT_FROM_SCRATCH_BATCH_TOKENS: int = 262_144
DEFAULT_WS_EXPAND_STEP_150M: int = 574
FROM_SCRATCH_300M_EVAL_STEPS: tuple[int, ...] = (0, 48, 95, 191, 286, 382, 477, 573, 668, 763, 859, 954, 1050, 1145)
CKPT_DIR_W45B: Path = DEFAULT_TEACHER_300M_CKPT_PATH.parent


def _make_w57_fiber_cfg(
    wte_init_mode: QuantumWteInitMode = QuantumWteInitMode.COMPLEX_FIBER_BUNDLE_11SEM_1ID,
    packet_readout_mode: QuantumPacketReadoutMode = QuantumPacketReadoutMode.STANDARD_LINEAR_READOUT,
    id_packet_lr_scale: float = DEFAULT_IDENTITY_PACKET_LR_SCALE,
    attn_kernel_mode: QuantumAttnKernelMode = QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY,
) -> QuantumFullBackboneGraftConfig:
    """Build a calibrated cold-start `QuantumFullBackboneGraftConfig` for Wave 57 (`<= 8` args)."""
    return QuantumFullBackboneGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
        ket_amplitude_mode=QuantumKetAmplitudeMode.ENTROPY_PRESERVING_COHERENT_KET,
        mlp_channel_mode=QuantumMLPChannelMode.FROZEN_CLASSICAL_MLP,
        doc_swc_mode=QuantumDocSWCMode.TEACHER_DOC_SWC_MASK,
        depth_lr_mode=QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN,
        attn_kernel_mode=attn_kernel_mode,
        stiefel_momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        momentum_schedule_mode=QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070,
        state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
        attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        purification_mode=PurificationMode.QUADRATIC_MCWEENY,
        channel_mode=QuantumChannelMode.UNITARY_U16,
        wte_init_mode=wte_init_mode,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        mlp_stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        qk_phase_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        packet_readout_mode=packet_readout_mode,
        id_packet_lr_scale=id_packet_lr_scale,
        interface_warmup_steps=0,
        interface_align_weight=0.0,
        wte_lr=DEFAULT_FROM_SCRATCH_WTE_LR,
        stiefel_lr=DEFAULT_FROM_SCRATCH_STIEFEL_LR,
        unitary_lr=DEFAULT_FROM_SCRATCH_UNITARY_LR,
        scalar_lr=DEFAULT_FROM_SCRATCH_SCALAR_LR,
    )


ARM_SPECS: dict[str, tuple[Path, QuantumFullBackboneGraftConfig, bool, bool]] = {
    "N78_1_FIBER_SPLIT_11SEM_1ID_PKT32": (
        DEFAULT_TEACHER_0M_CKPT_PATH,
        _make_w57_fiber_cfg(
            wte_init_mode=QuantumWteInitMode.COMPLEX_FIBER_BUNDLE_11SEM_1ID,
            packet_readout_mode=QuantumPacketReadoutMode.STANDARD_LINEAR_READOUT,
            id_packet_lr_scale=DEFAULT_IDENTITY_PACKET_LR_SCALE,
        ),
        True,
        False,
    ),
    "N78_2_FIBER_SPLIT_10SEM_2ID_DYNAMIC_GATE_PKT32": (
        DEFAULT_TEACHER_0M_CKPT_PATH,
        _make_w57_fiber_cfg(
            wte_init_mode=QuantumWteInitMode.COMPLEX_FIBER_BUNDLE_10SEM_2ID,
            packet_readout_mode=QuantumPacketReadoutMode.DYNAMIC_COMPLEX_PACKET_GATED_READOUT,
            id_packet_lr_scale=DEFAULT_IDENTITY_PACKET_LR_SCALE,
        ),
        True,
        False,
    ),
    "N78_3_FIBER_SPLIT_11SEM_1ID_STEP0_CALIB_PKT32": (
        DEFAULT_TEACHER_0M_CKPT_PATH,
        _make_w57_fiber_cfg(
            wte_init_mode=QuantumWteInitMode.COMPLEX_FIBER_BUNDLE_11SEM_1ID,
            packet_readout_mode=QuantumPacketReadoutMode.DYNAMIC_COMPLEX_PACKET_GATED_READOUT,
            id_packet_lr_scale=DEFAULT_IDENTITY_PACKET_LR_SCALE,
        ),
        True,
        True,
    ),
    "N78_4_LARGE_FROM_SCRATCH_300M_FIBER_SPLIT_QUANTUM_GPT": (
        DEFAULT_TEACHER_0M_CKPT_PATH,
        _make_w57_fiber_cfg(
            wte_init_mode=QuantumWteInitMode.COMPLEX_FIBER_BUNDLE_11SEM_1ID,
            packet_readout_mode=QuantumPacketReadoutMode.DYNAMIC_COMPLEX_PACKET_GATED_READOUT,
            id_packet_lr_scale=DEFAULT_IDENTITY_PACKET_LR_SCALE,
        ),
        True,
        True,
    ),
}


def resolve_plan078_arm_config(arm_id: str) -> QuantumFullBackboneGraftConfig:
    """Resolve `arm_id` (`N78_1..N78_4`) to its `QuantumFullBackboneGraftConfig`."""
    key = arm_id.strip().upper()
    if key not in ARM_SPECS:
        raise ValueError(f"Unknown Plan 078 arm: {arm_id}. Valid: {list(ARM_SPECS.keys())}")
    return ARM_SPECS[key][1]


@torch.no_grad()
def evaluate_full_backbone_model(
    model: QuantumFullBackboneGraftGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    wte_step0: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate validation cross-entropy, Top-1/Top-5 accuracy, and fiber-bundle telemetry (`<= 8` args)."""
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

    m_sem = (
        DEFAULT_FIBER_SEMANTIC_PACKETS_10
        if model.graft_cfg.wte_init_mode == QuantumWteInitMode.COMPLEX_FIBER_BUNDLE_10SEM_2ID
        else DEFAULT_FIBER_SEMANTIC_PACKETS_11
    )
    if wte_step0 is not None and model.q_wte.shape == wte_step0.shape and model.q_wte.size(1) > m_sem:
        diff = (model.q_wte.detach().float().cpu() - wte_step0)
        sem_drift = float(diff[:, :m_sem, :].pow(2).mean().sqrt().item())
        id_drift = float(diff[:, m_sem:, :].pow(2).mean().sqrt().item())
        out["wte_sem_drift_rms"] = sem_drift
        out["wte_id_drift_rms"] = id_drift
        out["wte_id_over_sem_drift_ratio"] = id_drift / max(sem_drift, 1e-8)

    if model.packet_readout_modulator is not None:
        gb = model.packet_readout_modulator.gate_bias.detach().float()
        out["readout_gate_bias_sem_mean"] = float(gb[:m_sem].mean().item())
        out["readout_gate_bias_id_mean"] = float(gb[m_sem:].mean().item())

    if was_training:
        model.train()
    return out


@torch.no_grad()
def evaluate_classical_teacher_checkpoints(
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluate all saved `0M..300M` `w45b` classical checkpoints on `val_batches`."""
    ckpts = {
        "0M": DEFAULT_TEACHER_0M_CKPT_PATH,
        "50M": DEFAULT_TEACHER_50M_CKPT_PATH,
        "100M": DEFAULT_TEACHER_100M_CKPT_PATH,
        "150M": CKPT_DIR_W45B / "ckpt_step00573_0150M_m0150M.pt",
        "200M": CKPT_DIR_W45B / "ckpt_step00763_0200M_m0200M.pt",
        "250M": CKPT_DIR_W45B / "ckpt_step00954_0250M_m0250M.pt",
        "300M": DEFAULT_TEACHER_300M_CKPT_PATH,
    }
    out: dict[str, dict[str, float]] = {}
    for label, path in ckpts.items():
        if not path.exists():
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        raw_cfg = dict(ckpt["config"])
        for k_tup in ("skip_attn_layers", "long_window_layers"):
            if k_tup in raw_cfg:
                raw_cfg[k_tup] = tuple(raw_cfg[k_tup])
        c_cfg = CompiledLeaderboardClassicalConfig(**raw_cfg)
        t_model = CompiledLeaderboardClassicalGPT(c_cfg).to(device)
        t_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        ws_blocks = int(ckpt.get("current_ws_blocks", c_cfg.initial_ws_blocks))
        t_model.set_sliding_window_blocks(ws_blocks, apply_yarn=True)
        t_model.eval()
        accum: dict[str, float] = {}
        v_valid = min(VALID_FINEWEB_VOCAB_SIZE, t_model.vocab_size)
        for x_cpu, y_cpu in val_batches:
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = t_model(x, return_logits=True)
            assert logits is not None
            bm = evaluate_readout_divergence_batch(logits[:, :, :v_valid], y)
            for k, v in bm.items():
                accum[k] = accum.get(k, 0.0) + float(v)
        n_b = float(max(1, len(val_batches)))
        out[label] = {
            "step": int(ckpt.get("step", 0)),
            "tokens_M": float(ckpt.get("tokens_M", 0.0)),
            "ws_blocks": float(ws_blocks),
            **{k: v / n_b for k, v in accum.items()},
        }
        del t_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _save_partial_arm_json(out_path: Path, arm_id: str, res_payload: dict[str, Any]) -> None:
    """Atomically merge `res_payload` into `out_path` under `results[arm_id]`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            existing = {}
    existing[arm_id] = res_payload
    tmp_p = out_path.with_suffix(".tmp.json")
    tmp_p.write_text(json.dumps({"plan": "EXPERIMENT_PLAN_078", "results": existing}, indent=2), encoding="utf-8")
    tmp_p.replace(out_path)


def run_single_full_backbone_arm(
    arm_id: str,
    graft_cfg: QuantumFullBackboneGraftConfig,
    teacher_ckpt: Path,
    steps: int,
    eval_every: int,
    batch_size: int,
    grad_accum: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate a single Wave 57 arm (`N78_1..N78_4`) on `device` (`<= 8` args)."""
    torch.manual_seed(2026078)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(2026078)

    _, _, use_tail_ema, apply_step0_calib = ARM_SPECS.get(
        arm_id, (teacher_ckpt, graft_cfg, True, False)
    )
    model = QuantumFullBackboneGraftGPT.from_scratch(
        ckpt0_path=teacher_ckpt, graft_cfg=graft_cfg, device=device
    )

    train_shards = sorted(FINEWEB_DIR.glob("fineweb_train_*.bin"))
    val_path = FINEWEB_DIR / "fineweb_val_000000.bin"
    val_batches = load_validation_batches(
        val_path, batch_size=batch_size, seq_len=DEFAULT_SEQ_LEN, vocab_size=model.vocab_size
    )

    calib_info: dict[str, float] = {}
    if apply_step0_calib:
        calib_info = calibrate_cold_start_readout_from_shard_(
            model=model,
            shard_path=train_shards[0],
        )

    wte_step0 = model.q_wte.detach().float().cpu()
    torch.cuda.empty_cache()
    active_q_params = sum(p.numel() for p in model.active_quantum_parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_attn_params = sum(
        p.numel() for blk in model.quantum_attn_blocks.values() for p in blk.parameters() if p.requires_grad
    )
    q_mlp_params = sum(
        p.numel() for blk in model.quantum_mlp_blocks.values() for p in blk.parameters() if p.requires_grad
    )

    opt = build_quantum_full_backbone_optimizers(model)
    averager = (
        TailWeightAverager(
            model,
            start_frac=DEFAULT_QUANTUM_TAIL_EMA_START_FRAC,
            ema_rate=DEFAULT_QUANTUM_TAIL_EMA_RATE,
            blend_alpha=DEFAULT_QUANTUM_TAIL_EMA_BLEND,
            update_every=2,
            decontract_frobenius=False,
            shadow_device="cpu",
        )
        if use_tail_ema
        else None
    )
    stream = MultiShardTokenStream(train_shards, batch_size=batch_size, seq_len=DEFAULT_SEQ_LEN, vocab_size=model.vocab_size)
    tokens_per_step = batch_size * DEFAULT_SEQ_LEN * grad_accum
    sched_total_steps = (
        DEFAULT_FROM_SCRATCH_300M_STEPS
        if (tokens_per_step == DEFAULT_FROM_SCRATCH_BATCH_TOKENS and steps <= DEFAULT_FROM_SCRATCH_300M_STEPS)
        else steps
    )

    val_0 = evaluate_full_backbone_model(model, val_batches, device, wte_step0=wte_step0)
    print(
        f"[{arm_id}] step=0 tok=0.00M val_ce={val_0['val_ce']:.4f} top1={val_0['top1_acc']*100:.2f}% "
        f"top5={val_0['top5_acc']*100:.2f}% wte_shape={tuple(model.q_wte.shape)} "
        f"q_attn={q_attn_params/1e6:.2f}M trainable={total_trainable_params/1e6:.2f}M",
        flush=True,
    )
    history: list[dict[str, Any]] = [{"step": 0, "tokens": 0, **calib_info, **val_0}]
    t0 = time.perf_counter()
    ema_loss = float(val_0["val_ce"])

    for step in range(1, steps + 1):
        if step == DEFAULT_WS_EXPAND_STEP_150M:
            model.set_sliding_window_blocks(6, apply_yarn=True)
        lr_mult = compute_quantum_backbone_lr_multiplier(step, sched_total_steps, graft_cfg.depth_lr_mode)
        mom_beta = compute_riemannian_momentum_beta(step, sched_total_steps, graft_cfg.momentum_schedule_mode)
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
        opt.step(last_idx, lr_mult=lr_mult, momentum_beta=mom_beta)
        if averager is not None:
            averager.update(step=step, total_steps=sched_total_steps)
        ema_loss = 0.95 * ema_loss + 0.05 * step_loss

        should_eval = (
            (step in FROM_SCRATCH_300M_EVAL_STEPS or step == steps)
            if tokens_per_step == DEFAULT_FROM_SCRATCH_BATCH_TOKENS
            else (step % eval_every == 0 or step == steps)
        )
        if should_eval:
            val_m = evaluate_full_backbone_model(model, val_batches, device, wte_step0=wte_step0)
            if step == steps and averager is not None and averager.is_seeded:
                with averager.average_parameters(model):
                    retract_quantum_backbone_manifolds_(model)
                    val_ema = evaluate_full_backbone_model(model, val_batches, device, wte_step0=wte_step0)
                if float(val_ema["val_ce"]) <= float(val_m["val_ce"]):
                    val_m = {**val_ema, "raw_last_val_ce": float(val_m["val_ce"]), "used_tail_ema": 1.0}
                else:
                    val_m["tail_ema_val_ce"] = float(val_ema["val_ce"])
                    val_m["used_tail_ema"] = 0.0
            elapsed = max(1e-6, time.perf_counter() - t0)
            toks = step * tokens_per_step
            tok_s = toks / elapsed
            rec = {
                "step": step,
                "tokens": toks,
                "train_ema_loss": ema_loss,
                "lr_mult": lr_mult,
                "mom_beta": mom_beta,
                "tok_per_s": tok_s,
                **val_m,
            }
            history.append(rec)
            print(
                f"[{arm_id}] step={step}/{steps} tok={toks/1e6:.2f}M lr_m={lr_mult:.3f} mom={mom_beta:.2f} "
                f"train_ema={ema_loss:.4f} val_ce={val_m['val_ce']:.4f} ppl={val_m['val_ppl']:.2f} "
                f"top1={val_m['top1_acc']*100:.2f}% top5={val_m['top5_acc']*100:.2f}% "
                f"drift_id/sem={val_m.get('wte_id_over_sem_drift_ratio', 0.0):.3f} "
                f"st_err={val_m['q_max_stiefel_err']:.2e} ({tok_s:.0f} tok/s)",
                flush=True,
            )

    best_rec = min(history[1:], key=lambda r: float(r["val_ce"])) if len(history) > 1 else history[0]
    final_rec = history[-1]
    res = {
        "arm_id": arm_id,
        "teacher_ckpt": str(teacher_ckpt.name),
        "wf_shape": graft_cfg.wf_shape.value,
        "wte_shape": list(model.q_wte.shape),
        "wte_init_mode": graft_cfg.wte_init_mode.value,
        "packet_readout_mode": graft_cfg.packet_readout_mode.value,
        "id_packet_lr_scale": float(graft_cfg.id_packet_lr_scale),
        "apply_step0_calib": bool(apply_step0_calib),
        "depth_scope": graft_cfg.depth_scope.value,
        "mlp_channel_mode": graft_cfg.mlp_channel_mode.value,
        "doc_swc_mode": graft_cfg.doc_swc_mode.value,
        "depth_lr_mode": graft_cfg.depth_lr_mode.value,
        "attn_kernel_mode": graft_cfg.attn_kernel_mode.value,
        "low_rank_cholesky_k": DEFAULT_LOW_RANK_CHOLESKY_K,
        "stiefel_momentum_mode": graft_cfg.stiefel_momentum_mode.value,
        "momentum_schedule_mode": graft_cfg.momentum_schedule_mode.value,
        "stiefel_proj_mode": graft_cfg.stiefel_proj_mode.value,
        "use_tail_ema": bool(use_tail_ema),
        "from_scratch": True,
        "active_quantum_params": active_q_params,
        "total_trainable_params": total_trainable_params,
        "quantum_attn_params": q_attn_params,
        "quantum_mlp_params": q_mlp_params,
        "step0_val_ce": float(val_0["val_ce"]),
        "best_val_ce": float(best_rec["val_ce"]),
        "best_val_ppl": float(best_rec["val_ppl"]),
        "best_top1_acc": float(best_rec["top1_acc"]),
        "best_top5_acc": float(best_rec["top5_acc"]),
        "final_val_ce": float(final_rec["val_ce"]),
        "final_val_ppl": float(final_rec["val_ppl"]),
        "final_top1_acc": float(final_rec["top1_acc"]),
        "final_top5_acc": float(final_rec["top5_acc"]),
        "final_stiefel_err": float(final_rec["q_max_stiefel_err"]),
        "final_unitary_err": float(final_rec["q_max_unitary_err"]),
        "final_sphere_err": float(final_rec["q_wte_sphere_err"]),
        "final_q_purity": float(final_rec["q_l0_purity"]),
        "final_q_eff_rank": float(final_rec["q_l0_eff_rank"]),
        "history": history,
    }
    del model, opt, averager
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def _ddp_from_scratch_worker(
    rank: int,
    world_size: int,
    arm_id: str,
    steps: int,
    batch_size: int,
    total_grad_accum: int,
    out_path_str: str,
    master_port: int,
) -> None:
    """Execute 2-GPU `nccl` data-parallel training for a Wave 57 cold-start arm (`<= 8` args)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(2026078)
    torch.cuda.manual_seed_all(2026078)

    teacher_ckpt, graft_cfg, use_tail_ema, apply_step0_calib = ARM_SPECS[arm_id]
    model = QuantumFullBackboneGraftGPT.from_scratch(ckpt0_path=teacher_ckpt, graft_cfg=graft_cfg, device=device)

    train_shards = sorted(FINEWEB_DIR.glob("fineweb_train_*.bin"))
    val_path = FINEWEB_DIR / "fineweb_val_000000.bin"
    val_batches = load_validation_batches(
        val_path, batch_size=batch_size, seq_len=DEFAULT_SEQ_LEN, vocab_size=model.vocab_size
    )

    calib_info: dict[str, float] = {}
    if apply_step0_calib:
        calib_info = calibrate_cold_start_readout_from_shard_(
            model=model,
            shard_path=train_shards[0],
        )

    wte_step0 = model.q_wte.detach().float().cpu()
    torch.cuda.empty_cache()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    active_q_params = sum(p.numel() for p in model.active_quantum_parameters())
    total_trainable_params = sum(p.numel() for p in trainable_params)
    q_attn_params = sum(
        p.numel() for blk in model.quantum_attn_blocks.values() for p in blk.parameters() if p.requires_grad
    )
    q_mlp_params = sum(
        p.numel() for blk in model.quantum_mlp_blocks.values() for p in blk.parameters() if p.requires_grad
    )

    opt = build_quantum_full_backbone_optimizers(model)
    averager = (
        TailWeightAverager(
            model,
            start_frac=DEFAULT_QUANTUM_TAIL_EMA_START_FRAC,
            ema_rate=DEFAULT_QUANTUM_TAIL_EMA_RATE,
            blend_alpha=DEFAULT_QUANTUM_TAIL_EMA_BLEND,
            update_every=6,
            decontract_frobenius=False,
            shadow_device="cpu",
        )
        if (use_tail_ema and rank == 0)
        else None
    )
    stream = MultiShardTokenStream(train_shards, batch_size=batch_size, seq_len=DEFAULT_SEQ_LEN, vocab_size=model.vocab_size)
    local_accum = max(1, total_grad_accum // world_size)
    tokens_per_step = batch_size * DEFAULT_SEQ_LEN * local_accum * world_size
    sched_total_steps = (
        DEFAULT_FROM_SCRATCH_300M_STEPS
        if (tokens_per_step == DEFAULT_FROM_SCRATCH_BATCH_TOKENS and steps <= DEFAULT_FROM_SCRATCH_300M_STEPS)
        else steps
    )
    out_p = Path(out_path_str)

    teacher_refs = (
        evaluate_classical_teacher_checkpoints(val_batches, device)
        if (rank == 0 and steps == DEFAULT_FROM_SCRATCH_300M_STEPS)
        else {}
    )
    val_0 = evaluate_full_backbone_model(model, val_batches, device, wte_step0=wte_step0) if rank == 0 else {}
    if rank == 0:
        print(
            f"[{arm_id} DDP-{world_size}GPU] step=0 tok=0.00M val_ce={val_0['val_ce']:.4f} "
            f"top1={val_0['top1_acc']*100:.2f}% top5={val_0['top5_acc']*100:.2f}% "
            f"wte_shape={tuple(model.q_wte.shape)} id_lr_scale={graft_cfg.id_packet_lr_scale:.2f} "
            f"readout={graft_cfg.packet_readout_mode.value} "
            f"q_attn={q_attn_params/1e6:.2f}M trainable={total_trainable_params/1e6:.2f}M",
            flush=True,
        )
    history: list[dict[str, Any]] = [{"step": 0, "tokens": 0, **calib_info, **val_0}] if rank == 0 else []
    t0 = time.perf_counter()
    ema_loss = float(val_0.get("val_ce", 10.8249))

    for step in range(1, steps + 1):
        if step == DEFAULT_WS_EXPAND_STEP_150M:
            model.set_sliding_window_blocks(6, apply_yarn=True)
            if rank == 0:
                print(
                    f"[{arm_id} DDP-{world_size}GPU] step={step} tok=150.21M expanded sliding window ws_blocks: 3 -> 6 (YaRN active)",
                    flush=True,
                )

        lr_mult = compute_quantum_backbone_lr_multiplier(step, sched_total_steps, graft_cfg.depth_lr_mode)
        mom_beta = compute_riemannian_momentum_beta(step, sched_total_steps, graft_cfg.momentum_schedule_mode)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        local_idx_list: list[torch.Tensor] = []

        for m_idx in range(total_grad_accum):
            idx_b, targets_b = stream.next_batch(device)
            if (m_idx % world_size) != rank:
                continue
            local_idx_list.append(idx_b)
            _, loss = model(idx_b, targets=targets_b)
            assert loss is not None
            (loss / float(local_accum)).backward()
            step_loss += float(loss.detach().item()) / float(local_accum)

        for p in trainable_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        local_flat = torch.unique(torch.cat([t.reshape(-1) for t in local_idx_list], dim=0))
        pad_n = batch_size * DEFAULT_SEQ_LEN * local_accum
        padded = torch.full((pad_n,), int(local_flat[0].item()), device=device, dtype=torch.long)
        padded[: local_flat.numel()] = local_flat
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        all_idx = torch.cat(gathered, dim=0)

        opt.step(all_idx, lr_mult=lr_mult, momentum_beta=mom_beta)
        if averager is not None:
            averager.update(step=step, total_steps=sched_total_steps)
        ema_loss = 0.95 * ema_loss + 0.05 * step_loss

        should_eval = (
            (step in FROM_SCRATCH_300M_EVAL_STEPS or step == steps)
            if tokens_per_step == DEFAULT_FROM_SCRATCH_BATCH_TOKENS
            else (step == steps or step % max(1, steps // 5) == 0)
        )
        if should_eval and rank == 0:
            val_m = evaluate_full_backbone_model(model, val_batches, device, wte_step0=wte_step0)
            if step in (954, 1050, steps) and averager is not None and averager.is_seeded:
                with averager.average_parameters(model):
                    retract_quantum_backbone_manifolds_(model)
                    val_ema = evaluate_full_backbone_model(model, val_batches, device, wte_step0=wte_step0)
                if float(val_ema["val_ce"]) <= float(val_m["val_ce"]):
                    val_m = {**val_ema, "raw_last_val_ce": float(val_m["val_ce"]), "used_tail_ema": 1.0}
                else:
                    val_m["tail_ema_val_ce"] = float(val_ema["val_ce"])
                    val_m["used_tail_ema"] = 0.0
            elapsed = max(1e-6, time.perf_counter() - t0)
            toks = step * tokens_per_step
            tok_s = toks / elapsed
            rec = {
                "step": step,
                "tokens": toks,
                "ws_blocks": int(model.classical.current_ws_blocks),
                "train_ema_loss": ema_loss,
                "lr_mult": lr_mult,
                "mom_beta": mom_beta,
                "tok_per_s": tok_s,
                **val_m,
            }
            history.append(rec)
            print(
                f"[{arm_id} DDP-{world_size}GPU] step={step}/{steps} tok={toks/1e6:.2f}M ws={rec['ws_blocks']} "
                f"lr_m={lr_mult:.3f} mom={mom_beta:.2f} train_ema={ema_loss:.4f} val_ce={val_m['val_ce']:.4f} "
                f"ppl={val_m['val_ppl']:.2f} top1={val_m['top1_acc']*100:.2f}% top5={val_m['top5_acc']*100:.2f}% "
                f"drift_id/sem={val_m.get('wte_id_over_sem_drift_ratio', 0.0):.3f} "
                f"st_err={val_m['q_max_stiefel_err']:.2e} ({tok_s:.0f} tok/s)",
                flush=True,
            )
            best_rec = min(history[1:], key=lambda r: float(r["val_ce"])) if len(history) > 1 else history[0]
            final_rec = history[-1]
            partial_res = {
                "arm_id": arm_id,
                "world_size": world_size,
                "completed_steps": step,
                "target_steps": steps,
                "schedule_total_steps": sched_total_steps,
                "teacher_ckpt": str(teacher_ckpt.name),
                "wf_shape": graft_cfg.wf_shape.value,
                "wte_shape": list(model.q_wte.shape),
                "wte_init_mode": graft_cfg.wte_init_mode.value,
                "packet_readout_mode": graft_cfg.packet_readout_mode.value,
                "id_packet_lr_scale": float(graft_cfg.id_packet_lr_scale),
                "apply_step0_calib": bool(apply_step0_calib),
                "depth_scope": graft_cfg.depth_scope.value,
                "mlp_channel_mode": graft_cfg.mlp_channel_mode.value,
                "doc_swc_mode": graft_cfg.doc_swc_mode.value,
                "depth_lr_mode": graft_cfg.depth_lr_mode.value,
                "attn_kernel_mode": graft_cfg.attn_kernel_mode.value,
                "stiefel_momentum_mode": graft_cfg.stiefel_momentum_mode.value,
                "momentum_schedule_mode": graft_cfg.momentum_schedule_mode.value,
                "stiefel_proj_mode": graft_cfg.stiefel_proj_mode.value,
                "eff_stiefel_lr": float(opt.q_trunk_opt.param_groups[0]["lr"]),
                "eff_unitary_lr": float(opt.q_trunk_opt.param_groups[0]["unitary_lr"]),
                "eff_wte_lr": float(opt.q_wte_opt.param_groups[0]["lr"]),
                "eff_scalar_lr": float(opt.base_scalar_lr),
                "use_tail_ema": bool(use_tail_ema),
                "from_scratch": True,
                "active_quantum_params": active_q_params,
                "total_trainable_params": total_trainable_params,
                "quantum_attn_params": q_attn_params,
                "quantum_mlp_params": q_mlp_params,
                "step0_val_ce": float(val_0["val_ce"]),
                "best_val_ce": float(best_rec["val_ce"]),
                "best_val_ppl": float(best_rec["val_ppl"]),
                "best_top1_acc": float(best_rec["top1_acc"]),
                "best_top5_acc": float(best_rec["top5_acc"]),
                "final_val_ce": float(final_rec["val_ce"]),
                "final_val_ppl": float(final_rec["val_ppl"]),
                "final_top1_acc": float(final_rec["top1_acc"]),
                "final_top5_acc": float(final_rec["top5_acc"]),
                "final_stiefel_err": float(final_rec["q_max_stiefel_err"]),
                "final_unitary_err": float(final_rec["q_max_unitary_err"]),
                "final_sphere_err": float(final_rec["q_wte_sphere_err"]),
                "final_q_purity": float(final_rec["q_l0_purity"]),
                "final_q_eff_rank": float(final_rec["q_l0_eff_rank"]),
                "classical_teacher_references": teacher_refs,
                "history": history,
            }
            _save_partial_arm_json(out_p, arm_id, partial_res)
        if should_eval:
            dist.barrier(device_ids=[rank])

    dist.barrier(device_ids=[rank])
    del model, opt, averager
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Plan 078 Complex Wavefunction Fiber-Bundle Split & 300M From-Scratch Quantum GPT."
    )
    parser.add_argument("--arms", type=str, default="ALL", help="Comma-separated arm IDs or ALL.")
    parser.add_argument("--steps", type=int, default=DEFAULT_PART1_50M_STEPS)
    parser.add_argument("--eval-every", type=int, default=95)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_FROM_SCRATCH_GRAD_ACCUM)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-2gpu-ddp", action="store_true", help="Run selected arms across cuda:0 and cuda:1 via NCCL DDP.")
    parser.add_argument("--prebuild-sketch-only", action="store_true", help="Build/cache the Directed Complex PPMI sketch and exit.")
    parser.add_argument("--master-port", type=int, default=29578)
    parser.add_argument("--out-json", type=str, default=str(DEFAULT_OUT_JSON))
    args = parser.parse_args()

    if args.prebuild_sketch_only:
        sketch = build_or_load_directed_complex_ppmi_sketch()
        print(
            f"[prebuild-sketch] Cached directed complex PPMI sketch ready: "
            f"s_fwd={tuple(sketch['s_fwd_raw'].shape)} s_rev={tuple(sketch['s_rev_raw'].shape)} "
            f"active={int(sketch['active_mask'].sum().item())}/{int(sketch['active_mask'].numel())}",
            flush=True,
        )
        return

    if args.arms.strip().upper() == "ALL":
        selected = list(ARM_SPECS.keys())
    else:
        selected = [a.strip() for a in args.arms.split(",") if a.strip()]

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Pre-warm the cached sketch in the parent process before spawning workers
    build_or_load_directed_complex_ppmi_sketch()

    if args.use_2gpu_ddp and torch.cuda.device_count() >= 2:
        for idx_arm, arm_id in enumerate(selected):
            if arm_id not in ARM_SPECS:
                raise ValueError(f"Unknown arm `{arm_id}`. Valid: {list(ARM_SPECS.keys())}")
            mp.spawn(
                _ddp_from_scratch_worker,
                args=(
                    2,
                    arm_id,
                    int(args.steps),
                    int(args.batch_size),
                    int(args.grad_accum),
                    str(out_path),
                    int(args.master_port) + idx_arm,
                ),
                nprocs=2,
                join=True,
            )
        return

    for arm_id in selected:
        if arm_id not in ARM_SPECS:
            raise ValueError(f"Unknown arm `{arm_id}`. Valid: {list(ARM_SPECS.keys())}")
        ckpt_path, cfg, _, _ = ARM_SPECS[arm_id]
        res = run_single_full_backbone_arm(
            arm_id=arm_id,
            graft_cfg=cfg,
            teacher_ckpt=ckpt_path,
            steps=int(args.steps),
            eval_every=int(args.eval_every),
            batch_size=int(args.batch_size),
            grad_accum=int(args.grad_accum),
            device=device,
        )
        _save_partial_arm_json(out_path, arm_id, res)


if __name__ == "__main__":
    main()
