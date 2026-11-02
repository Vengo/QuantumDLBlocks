#!/usr/bin/env python3
r"""Verification Runner for `EXPERIMENT_PLAN_074.md` (`Wave 53B`).

Evaluates **Document-Boundary Sliding-Window Quantum Wavepacket Attention (`Doc-SWC`) & Depth-Scaled
$1/\sqrt{N_{\text{sub}}}$ Geodesic Trust-Region Optimization Across All 12 Layers** (`8` arms `J74_1..J74_8`)
on the Pretrained `300M` Classical Teacher (`val_ce = 3.5845`, `top1 = 37.31%`, `top5 = 58.26%`):
- **`J74_1_DEPTH3_L0_TO_L2_DOC_SWC_R384`**: `EMB_PLUS_L0_TO_L2_ATTN` (`wte + L0..L2` at `1:1` parity `R384`)
- **`J74_2_DEPTH6_ENCODER_L0_TO_L5_DOC_SWC_R384`**: `EMB_PLUS_L0_TO_L5_ATTN` (`wte + L0..L5` 6-layer encoder at `1:1` parity `R384`)
- **`J74_3_ALL12_ATTN_DOC_SWC_R768`**: `EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11` at `R768`)
- **`J74_4_ALL12_ATTN_DOC_SWC_R384_PARITY`**: `EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11` at `1:1` parity `R384`)
- **`J74_5_ALL12_ATTN_DOC_SWC_R192_2X`**: `EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11` at `2x` compression `R192`)
- **`J74_6_ALL12_ATTN_DOC_SWC_R128_3X`**: `EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11` at `3x` compression `R128`)
- **`J74_7_ALL12_FULL_BLOCKS_MZ_DOC_SWC_R768`**: `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ` (`100%` Backbone `wte + ve + L0..L11 Attn & MZ-MLP` at `R768`)
- **`J74_8_ALL12_FULL_BLOCKS_MZ_DOC_SWC_R384_COMPRESSED`**: `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ` (`100%` Backbone `wte + ve + L0..L11 Attn & MZ-MLP` at `R384`)
"""

from __future__ import annotations

import argparse
import gc
from importlib import import_module
import json
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

from qdlblocks.layers.isometric_born_trace_readout import evaluate_readout_divergence_batch  # noqa: E402
from qdlblocks.layers.quantum_frontend_graft import (  # noqa: E402
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
)
from qdlblocks.models.quantum_full_backbone_graft_gpt import (  # noqa: E402
    DEFAULT_FULL_BACKBONE_WARMUP_STEPS,
    QuantumBackboneDepthScope,
    QuantumDepthLRMode,
    QuantumDocSWCMode,
    QuantumFullBackboneGraftConfig,
    QuantumFullBackboneGraftGPT,
    QuantumKetAmplitudeMode,
    QuantumMLPChannelMode,
    build_quantum_full_backbone_optimizers,
    compute_quantum_backbone_lr_multiplier,
)

_ex75 = import_module("75_verify_plan065_frozen_trunk_quantum_frontend_tournament")
FINEWEB_DIR: Path = _ex75.FINEWEB_DIR
DEFAULT_SEQ_LEN: int = _ex75.DEFAULT_SEQ_LEN
VALID_FINEWEB_VOCAB_SIZE: int = _ex75.VALID_FINEWEB_VOCAB_SIZE
MultiShardTokenStream = _ex75.MultiShardTokenStream
load_validation_batches = _ex75.load_validation_batches

DEFAULT_OUT_JSON: Path = _REPO_ROOT / ".cache" / "w53b_plan074_doc_swc_and_depth_scaled_full_backbone.json"
DEFAULT_REF2L_WTE_LR: float = 0.003
DEFAULT_REF2L_STIEFEL_LR: float = 0.006
DEFAULT_REF2L_UNITARY_LR: float = 0.010
DEFAULT_REF2L_SCALAR_LR: float = 0.005


def _make_w53b_cfg(
    depth_scope: QuantumBackboneDepthScope = QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
    attn_rank: QuantumStiefelProjMode = QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
    mlp_mode: QuantumMLPChannelMode = QuantumMLPChannelMode.FROZEN_CLASSICAL_MLP,
    mlp_rank: QuantumStiefelProjMode = QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
    doc_swc_mode: QuantumDocSWCMode = QuantumDocSWCMode.TEACHER_DOC_SWC_MASK,
    depth_lr_mode: QuantumDepthLRMode = QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE,
    warmup_steps: int = DEFAULT_FULL_BACKBONE_WARMUP_STEPS,
) -> QuantumFullBackboneGraftConfig:
    """Build a calibrated `QuantumFullBackboneGraftConfig` for Wave 53B (`<= 8` args)."""
    return QuantumFullBackboneGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        depth_scope=depth_scope,
        ket_amplitude_mode=QuantumKetAmplitudeMode.ENTROPY_PRESERVING_COHERENT_KET,
        mlp_channel_mode=mlp_mode,
        doc_swc_mode=doc_swc_mode,
        depth_lr_mode=depth_lr_mode,
        state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
        attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        purification_mode=PurificationMode.QUADRATIC_MCWEENY,
        channel_mode=QuantumChannelMode.UNITARY_U16,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
        stiefel_proj_mode=attn_rank,
        mlp_stiefel_proj_mode=mlp_rank,
        self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        qk_phase_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        interface_warmup_steps=warmup_steps,
        interface_align_weight=0.50,
        wte_lr=DEFAULT_REF2L_WTE_LR,
        stiefel_lr=DEFAULT_REF2L_STIEFEL_LR,
        unitary_lr=DEFAULT_REF2L_UNITARY_LR,
        scalar_lr=DEFAULT_REF2L_SCALAR_LR,
    )


ARM_SPECS: dict[str, tuple[Path, QuantumFullBackboneGraftConfig]] = {
    "J74_1_DEPTH3_L0_TO_L2_DOC_SWC_R384": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w53b_cfg(
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_L0_TO_L2_ATTN,
            attn_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        ),
    ),
    "J74_2_DEPTH6_ENCODER_L0_TO_L5_DOC_SWC_R384": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w53b_cfg(
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_L0_TO_L5_ATTN,
            attn_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        ),
    ),
    "J74_3_ALL12_ATTN_DOC_SWC_R768": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w53b_cfg(
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
            attn_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
        ),
    ),
    "J74_4_ALL12_ATTN_DOC_SWC_R384_PARITY": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w53b_cfg(
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
            attn_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        ),
    ),
    "J74_5_ALL12_ATTN_DOC_SWC_R192_2X": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w53b_cfg(
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
            attn_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R192,
        ),
    ),
    "J74_6_ALL12_ATTN_DOC_SWC_R128_3X": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w53b_cfg(
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
            attn_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128,
        ),
    ),
    "J74_7_ALL12_FULL_BLOCKS_MZ_DOC_SWC_R768": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w53b_cfg(
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_FULL_BLOCKS_MZ,
            attn_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            mlp_mode=QuantumMLPChannelMode.SVD_STIEFEL_MACH_ZEHNDER,
            mlp_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
        ),
    ),
    "J74_8_ALL12_FULL_BLOCKS_MZ_DOC_SWC_R384_COMPRESSED": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w53b_cfg(
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_FULL_BLOCKS_MZ,
            attn_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            mlp_mode=QuantumMLPChannelMode.SVD_STIEFEL_MACH_ZEHNDER,
            mlp_rank=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        ),
    ),
}


def resolve_plan074_arm_config(arm_id: str) -> QuantumFullBackboneGraftConfig:
    """Resolve `arm_id` (`J74_1..J74_8`) to its `QuantumFullBackboneGraftConfig`."""
    key = arm_id.strip().upper()
    if key not in ARM_SPECS:
        raise ValueError(f"Unknown Plan 074 arm: {arm_id}. Valid: {list(ARM_SPECS.keys())}")
    return ARM_SPECS[key][1]


@torch.no_grad()
def evaluate_full_backbone_model(
    model: QuantumFullBackboneGraftGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate validation cross-entropy, Top-1/Top-5 accuracy, and quantum backbone telemetry."""
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
    """Train and evaluate a single full-backbone quantum wavepacket replacement arm (`<= 8` args)."""
    torch.manual_seed(2026074)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(2026074)

    model = QuantumFullBackboneGraftGPT.from_teacher_checkpoint(
        ckpt_path=teacher_ckpt,
        graft_cfg=graft_cfg,
        device=device,
    )
    train_shards = sorted(FINEWEB_DIR.glob("fineweb_train_*.bin"))
    val_path = FINEWEB_DIR / "fineweb_val_000000.bin"
    val_batches = load_validation_batches(
        val_path, batch_size=batch_size, seq_len=DEFAULT_SEQ_LEN, vocab_size=model.vocab_size
    )

    active_q_params = sum(p.numel() for p in model.active_quantum_parameters())
    q_attn_params = sum(
        p.numel() for blk in model.quantum_attn_blocks.values() for p in blk.parameters() if p.requires_grad
    )
    q_mlp_params = sum(
        p.numel() for blk in model.quantum_mlp_blocks.values() for p in blk.parameters() if p.requires_grad
    )

    opt = build_quantum_full_backbone_optimizers(model)
    stream = MultiShardTokenStream(train_shards, batch_size=batch_size, seq_len=DEFAULT_SEQ_LEN, vocab_size=model.vocab_size)
    tokens_per_step = batch_size * DEFAULT_SEQ_LEN * grad_accum

    val_0 = evaluate_full_backbone_model(model, val_batches, device)
    print(
        f"[{arm_id}] step=0 tok=0.00M val_ce={val_0['val_ce']:.4f} top1={val_0['top1_acc']*100:.2f}% "
        f"top5={val_0['top5_acc']*100:.2f}% q_attn={q_attn_params/1e6:.2f}M q_mlp={q_mlp_params/1e6:.2f}M"
    )
    history: list[dict[str, Any]] = [{"step": 0, "tokens": 0, **val_0}]
    t0 = time.perf_counter()
    ema_loss = float(val_0["val_ce"])

    for step in range(1, steps + 1):
        lr_mult = compute_quantum_backbone_lr_multiplier(step, steps, graft_cfg.depth_lr_mode)
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
            val_m = evaluate_full_backbone_model(model, val_batches, device)
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

    rep_attn = list(model.replaced_attn_layers)
    rep_mlp = list(model.replaced_mlp_layers)
    del model, opt, stream, val_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "arm_id": arm_id,
        "config": {
            "wf_shape": graft_cfg.wf_shape.value,
            "depth_scope": graft_cfg.depth_scope.value,
            "ket_amplitude_mode": graft_cfg.ket_amplitude_mode.value,
            "mlp_channel_mode": graft_cfg.mlp_channel_mode.value,
            "doc_swc_mode": graft_cfg.doc_swc_mode.value,
            "depth_lr_mode": graft_cfg.depth_lr_mode.value,
            "stiefel_proj_mode": graft_cfg.stiefel_proj_mode.value,
            "mlp_stiefel_proj_mode": graft_cfg.mlp_stiefel_proj_mode.value,
            "attn_interior_mode": graft_cfg.attn_interior_mode.value,
            "qk_phase_mode": graft_cfg.qk_phase_mode.value,
            "self_ortho_mode": graft_cfg.self_ortho_mode.value,
            "interface_warmup_steps": graft_cfg.interface_warmup_steps,
            "replaced_attn_layers": rep_attn,
            "replaced_mlp_layers": rep_mlp,
            "active_quantum_params": active_q_params,
            "quantum_attn_params": q_attn_params,
            "quantum_mlp_params": q_mlp_params,
        },
        "steps": steps,
        "tokens_processed": steps * tokens_per_step,
        "final": history[-1],
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 074 Doc-SWC & Depth-Scaled Full-Backbone Quantum Runner")
    parser.add_argument("--arms", type=str, default="J74_4_ALL12_ATTN_DOC_SWC_R384_PARITY")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-json", type=str, default=str(DEFAULT_OUT_JSON))
    args = parser.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm_id in selected:
        ckpt_path, cfg = ARM_SPECS[arm_id]
        res = run_single_full_backbone_arm(
            arm_id=arm_id,
            graft_cfg=cfg,
            teacher_ckpt=ckpt_path,
            steps=args.steps,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            device=device,
        )
        results[arm_id] = res
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
