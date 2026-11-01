"""Wave 52B (`EXPERIMENT_PLAN_066.md`) Runner: Teacher-Subspace Spherical Warm-Start, VE0 Coupling & Sub-300M Recovery.

Executes the Phase-1 ablation arms (`R1..R5`) and Phase-2 long-horizon sub-`300M` token recovery runs
(`S1_RECOVERY_VS_50M_TEACHER`, `S2_RECOVERY_VS_300M_TEACHER`) grafting a trainable Quantum Front-End
onto a frozen pretrained `CompiledLeaderboardClassicalGPT` teacher.
"""

from __future__ import annotations

import argparse
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

# Reuse the verified training & evaluation harness from example 75
from importlib import import_module

_ex75 = import_module("75_verify_plan065_frozen_trunk_quantum_frontend_tournament")
run_single_graft_arm = _ex75.run_single_graft_arm

CACHE_DIR: Path = REPO_ROOT / ".cache"
DEFAULT_JSON_OUT: Path = CACHE_DIR / "w52b_plan066_recovery.json"


def resolve_plan066_arm_config(
    arm_id: str,
    wf_shape_override: str = "",
    attn_interior_override: str = "",
    warmup_steps_override: int | None = None,
) -> QuantumFrontEndGraftConfig:
    """Map Plan 066 arm identifier (`R1..R5`, `S1`, `S2`) to its `QuantumFrontEndGraftConfig`."""
    key = arm_id.strip().upper()
    presets: dict[str, QuantumFrontEndGraftConfig] = {
        "R1_TEACHER_SPHERE_24X2X16": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.NONE,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        ),
        "R2_TEACHER_SPHERE_VE0_24X2X16": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        ),
        "R3_TEACHER_SPHERE_VE0_KEEP_MLP0": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        ),
        "R4_TEACHER_SPHERE_VE0_2STAGE": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE,
            interface_warmup_steps=150,
            interface_align_weight=1.5,
        ),
        "R5_BEST_P065_SHAPE_UPGRADED": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x512,
            state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        ),
        "R6_COMPACT_1X2X256_DM_UPGRADED": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x256,
            state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        ),
        "S1_RECOVERY_VS_50M_TEACHER": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE,
            interface_warmup_steps=200,
            interface_align_weight=1.5,
        ),
        "S2_RECOVERY_VS_300M_TEACHER": QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE,
            interface_warmup_steps=300,
            interface_align_weight=1.5,
        ),
    }
    if key not in presets:
        raise ValueError(f"Unknown Plan 066 arm: {arm_id}. Valid arms: {list(presets.keys())}")
    cfg = presets[key]
    if wf_shape_override:
        cfg.wf_shape = QuantumWavefunctionShape(wf_shape_override)
    if attn_interior_override:
        cfg.attn_interior_mode = QuantumAttnInteriorMode(attn_interior_override)
    if warmup_steps_override is not None and warmup_steps_override >= 0:
        cfg.interface_warmup_steps = int(warmup_steps_override)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 52B (Plan 066) Sub-300M Quantum Front-End Recovery Runner")
    parser.add_argument("--arm", type=str, default="R2_TEACHER_SPHERE_VE0_24X2X16")
    parser.add_argument("--arms", type=str, default="")
    parser.add_argument("--teacher-ckpt", type=Path, default=DEFAULT_TEACHER_300M_CKPT_PATH)
    parser.add_argument("--use-50m-teacher", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--wf-shape", type=str, default="")
    parser.add_argument("--attn-interior-mode", type=str, default="")
    parser.add_argument("--replacement-scope", type=str, default="")
    parser.add_argument("--interface-warmup-steps", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    teacher_ckpt = DEFAULT_TEACHER_50M_CKPT_PATH if args.use_50m_teacher else args.teacher_ckpt
    warmup_override = args.interface_warmup_steps if args.interface_warmup_steps >= 0 else None

    if args.arms:
        arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
        results: dict[str, Any] = {}
        for arm_id in arm_list:
            cfg = resolve_plan066_arm_config(
                arm_id,
                wf_shape_override=args.wf_shape,
                attn_interior_override=args.attn_interior_mode,
                warmup_steps_override=warmup_override,
            )
            if args.replacement_scope:
                cfg.replacement_scope = QuantumReplacementScope(args.replacement_scope)
            res = run_single_graft_arm(
                arm_id=arm_id,
                graft_cfg=cfg,
                teacher_ckpt=teacher_ckpt,
                steps=args.steps,
                eval_every=args.eval_every,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                device=device,
            )
            res["config"]["wte_init_mode"] = cfg.wte_init_mode.value
            res["config"]["ve_coupling_mode"] = cfg.ve_coupling_mode.value
            res["config"]["recovery_schedule_mode"] = cfg.recovery_schedule_mode.value
            results[arm_id] = res
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return

    cfg = resolve_plan066_arm_config(
        args.arm,
        wf_shape_override=args.wf_shape,
        attn_interior_override=args.attn_interior_mode,
        warmup_steps_override=warmup_override,
    )
    if args.replacement_scope:
        cfg.replacement_scope = QuantumReplacementScope(args.replacement_scope)

    res = run_single_graft_arm(
        arm_id=args.arm,
        graft_cfg=cfg,
        teacher_ckpt=teacher_ckpt,
        steps=args.steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        device=device,
    )
    res["config"]["wte_init_mode"] = cfg.wte_init_mode.value
    res["config"]["ve_coupling_mode"] = cfg.ve_coupling_mode.value
    res["config"]["recovery_schedule_mode"] = cfg.recovery_schedule_mode.value
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
