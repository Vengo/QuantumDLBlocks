"""Wave 52D (`EXPERIMENT_PLAN_068.md`) Runner:
Biorthogonal SVD-Stiefel Wavepacket Projections & Fubini-Study Self-State Orthogonalization.

Executes Phase-1 rank & orthogonalization ablation arms (`D0..D7`) and Phase-2 reproduction/exceeding
arms (`E1_EXCEED_50M_TEACHER_24X2X16`, `E2_REPRODUCE_300M_TEACHER_24X2X16`, `E3_REPRODUCE_300M_TEACHER_12X2X32`).
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
    QuantumSelfOrthoMode,
    QuantumStateIOMode,
    QuantumStiefelProjMode,
    QuantumValueEmbedCouplingMode,
    QuantumWavefunctionShape,
    QuantumWteInitMode,
)

_ex75 = import_module("75_verify_plan065_frozen_trunk_quantum_frontend_tournament")
run_single_graft_arm = _ex75.run_single_graft_arm

CACHE_DIR: Path = REPO_ROOT / ".cache"
DEFAULT_JSON_OUT: Path = CACHE_DIR / "w52d_plan068_svd_stiefel_fubini_study.json"


def _make_svd_graft_cfg(
    wf_shape: QuantumWavefunctionShape,
    proj_mode: QuantumStiefelProjMode,
    ortho_mode: QuantumSelfOrthoMode,
    state_io: QuantumStateIOMode = QuantumStateIOMode.DM_IN_DM_OUT,
    warmup_steps: int = 300,
    align_weight: float = 0.5,
    wte_lr: float = 0.004,
    stiefel_lr: float = 0.012,
) -> QuantumFrontEndGraftConfig:
    """Construct a `QuantumFrontEndGraftConfig` for Wave 52D arms (`<= 8` args)."""
    return QuantumFrontEndGraftConfig(
        wf_shape=wf_shape,
        state_io_mode=state_io,
        attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
        interface_warmup_steps=warmup_steps,
        interface_align_weight=align_weight,
        wte_lr=wte_lr,
        stiefel_lr=stiefel_lr,
        stiefel_proj_mode=proj_mode,
        self_ortho_mode=ortho_mode,
    )


def resolve_plan068_arm_config(
    arm_id: str,
    wte_lr_override: float | None = None,
    stiefel_lr_override: float | None = None,
    warmup_steps_override: int | None = None,
) -> QuantumFrontEndGraftConfig:
    """Map Plan 068 arm identifier (`D0..D7`, `E1..E3`) to its `QuantumFrontEndGraftConfig`."""
    key = arm_id.strip().upper()
    presets: dict[str, QuantumFrontEndGraftConfig] = {
        "D0_KRONEKER_BASELINE": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumStiefelProjMode.KRONEKER_FACTORED_STIEFEL,
            QuantumSelfOrthoMode.NONE,
            warmup_steps=300,
            align_weight=1.5,
            wte_lr=0.006,
            stiefel_lr=0.035,
        ),
        "D1_SVD_STIEFEL_R128": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128,
            QuantumSelfOrthoMode.NONE,
        ),
        "D2_SVD_STIEFEL_R256": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R256,
            QuantumSelfOrthoMode.NONE,
        ),
        "D3_SVD_STIEFEL_R512": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R512,
            QuantumSelfOrthoMode.NONE,
        ),
        "D4_SVD_STIEFEL_R768_NO_ORTHO": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumSelfOrthoMode.NONE,
        ),
        "D5_SVD_STIEFEL_R768_FUBINI_ORTHO": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        ),
        "D6_SVD_STIEFEL_R768_PKT32": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_12x2x32,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
            state_io=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
        ),
        "D7_SVD_STIEFEL_R768_512_PKT32": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_1x2x512_PKT32,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
            state_io=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
        ),
        "E1_EXCEED_50M_TEACHER_24X2X16": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
            warmup_steps=400,
        ),
        "E2_REPRODUCE_300M_TEACHER_24X2X16": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
            warmup_steps=400,
        ),
        "E3_REPRODUCE_300M_TEACHER_12X2X32": _make_svd_graft_cfg(
            QuantumWavefunctionShape.SHAPE_12x2x32,
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
            state_io=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            warmup_steps=400,
        ),
    }
    if key not in presets:
        raise ValueError(f"Unknown Plan 068 arm: {arm_id}. Valid arms: {list(presets.keys())}")
    cfg = presets[key]
    if wte_lr_override is not None and wte_lr_override > 0.0:
        cfg.wte_lr = float(wte_lr_override)
    if stiefel_lr_override is not None and stiefel_lr_override > 0.0:
        cfg.stiefel_lr = float(stiefel_lr_override)
    if warmup_steps_override is not None and warmup_steps_override >= 0:
        cfg.interface_warmup_steps = int(warmup_steps_override)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 52D (Plan 068) SVD-Stiefel & Fubini-Study Graft Runner")
    parser.add_argument("--arm", type=str, default="D5_SVD_STIEFEL_R768_FUBINI_ORTHO")
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
        cfg = resolve_plan068_arm_config(arm_id, wte_lr_ov, stiefel_lr_ov, warmup_ov)
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
        res["config"]["stiefel_proj_mode"] = cfg.stiefel_proj_mode.value
        res["config"]["self_ortho_mode"] = cfg.self_ortho_mode.value
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
