"""Wave 52C (`EXPERIMENT_PLAN_067.md`) Runner:
Outer-Product Hilbert Granularity ($d_{\\text{pkt}} \\in \\{8, 16, 32\\}$), Rank-3 Causal Ensemble DM I/O, and Persistent Subspace Recovery.

Executes Phase-1 screening arms (`G1..G4`) and Phase-2 Sub-`300M`-Token Recovery runs
(`P1_SUB50M_RECOVERY_VS_50M_TEACHER`, `P2_SUB300M_RECOVERY_VS_300M_TEACHER`).
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "examples"))

import torch

from qdlblocks.layers.quantum_frontend_graft import (
    DEFAULT_TEACHER_300M_CKPT_PATH,
    DEFAULT_TEACHER_50M_CKPT_PATH,
    QuantumAttnInteriorMode,
    QuantumFrontEndGraftConfig,
    QuantumRecoveryScheduleMode,
    QuantumReplacementScope,
    QuantumStateIOMode,
    QuantumValueEmbedCouplingMode,
    QuantumWavefunctionShape,
    QuantumWteInitMode,
)

_ex75 = import_module("75_verify_plan065_frozen_trunk_quantum_frontend_tournament")
run_single_graft_arm = _ex75.run_single_graft_arm

CACHE_DIR: Path = REPO_ROOT / ".cache"
DEFAULT_JSON_OUT: Path = CACHE_DIR / "w52c_plan067_hilbert_granularity.json"


def resolve_plan067_arm_config(
    arm_id: str,
    wte_lr_override: float | None = None,
    stiefel_lr_override: float | None = None,
    warmup_steps_override: int | None = None,
) -> QuantumFrontEndGraftConfig:
    """Map Plan 067 arm identifier (`G1..G4`, `P1`, `P2`) to its `QuantumFrontEndGraftConfig`."""
    key = arm_id.strip().upper()
    presets: dict[str, QuantumFrontEndGraftConfig] = {
        "G1_PKT8_48X2X8_DM": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_48x2x8,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=300,
            interface_align_weight=1.5,
            wte_lr=0.006,
            stiefel_lr=0.035,
        ),
        "G2_PKT16_24X2X16_DM": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=300,
            interface_align_weight=1.5,
            wte_lr=0.006,
            stiefel_lr=0.035,
        ),
        "G3_PKT32_12X2X32_DM": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=300,
            interface_align_weight=1.5,
            wte_lr=0.006,
            stiefel_lr=0.035,
        ),
        "G4_PKT32_1X2X512_DM": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x512_PKT32,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=300,
            interface_align_weight=1.5,
            wte_lr=0.006,
            stiefel_lr=0.035,
        ),
        "P0_50M_TEACHER_REF": QuantumFrontEndGraftConfig(
            replacement_scope=QuantumReplacementScope.TEACHER_PASS_THROUGH,
        ),
        "P1_SUB50M_RECOVERY_VS_50M_TEACHER": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x512,
            state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=400,
            interface_align_weight=1.5,
            wte_lr=0.006,
            stiefel_lr=0.035,
        ),
        "P2_SUB300M_RECOVERY_VS_300M_TEACHER": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x512,
            state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=600,
            interface_align_weight=1.5,
            wte_lr=0.006,
            stiefel_lr=0.035,
        ),
        "S1_EXCEED_50M_TEACHER": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x512,
            state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE,
            interface_warmup_steps=80,
            interface_align_weight=1.0,
            wte_lr=0.025,
            stiefel_lr=0.035,
        ),
        "S2_G4_PKT32_300M_RECOVERY": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x512_PKT32,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=800,
            interface_align_weight=2.0,
            wte_lr=0.003,
            stiefel_lr=0.030,
        ),
    }
    if key not in presets:
        raise ValueError(f"Unknown Plan 067 arm: {arm_id}. Valid arms: {list(presets.keys())}")
    cfg = presets[key]
    if wte_lr_override is not None and wte_lr_override > 0.0:
        cfg.wte_lr = float(wte_lr_override)
    if stiefel_lr_override is not None and stiefel_lr_override > 0.0:
        cfg.stiefel_lr = float(stiefel_lr_override)
    if warmup_steps_override is not None and warmup_steps_override >= 0:
        cfg.interface_warmup_steps = int(warmup_steps_override)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 52C (Plan 067) Hilbert Granularity & Sub-300M Recovery Runner")
    parser.add_argument("--arm", type=str, default="G2_PKT16_24X2X16_DM")
    parser.add_argument("--arms", type=str, default="")
    parser.add_argument("--teacher-ckpt", type=Path, default=DEFAULT_TEACHER_300M_CKPT_PATH)
    parser.add_argument("--use-50m-teacher", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--wte-lr", type=float, default=-1.0)
    parser.add_argument("--stiefel-lr", type=float, default=-1.0)
    parser.add_argument("--interface-warmup-steps", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    teacher_ckpt = DEFAULT_TEACHER_50M_CKPT_PATH if args.use_50m_teacher else args.teacher_ckpt
    wte_lr_ov = args.wte_lr if args.wte_lr > 0.0 else None
    stiefel_lr_ov = args.stiefel_lr if args.stiefel_lr > 0.0 else None
    warmup_ov = args.interface_warmup_steps if args.interface_warmup_steps >= 0 else None

    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else [args.arm]
    results: dict[str, Any] = {}
    for arm_id in arm_list:
        cfg = resolve_plan067_arm_config(arm_id, wte_lr_ov, stiefel_lr_ov, warmup_ov)
        eff_ckpt = DEFAULT_TEACHER_50M_CKPT_PATH if "50M_TEACHER" in arm_id.upper() else teacher_ckpt
        res = run_single_graft_arm(
            arm_id=arm_id,
            graft_cfg=cfg,
            teacher_ckpt=eff_ckpt,
            steps=args.steps,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            device=device,
        )
        res["config"]["wte_init_mode"] = cfg.wte_init_mode.value
        res["config"]["ve_coupling_mode"] = cfg.ve_coupling_mode.value
        res["config"]["recovery_schedule_mode"] = cfg.recovery_schedule_mode.value
        res["config"]["wte_lr"] = cfg.wte_lr
        res["config"]["stiefel_lr"] = cfg.stiefel_lr
        results[arm_id] = res
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results if args.arms else res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
