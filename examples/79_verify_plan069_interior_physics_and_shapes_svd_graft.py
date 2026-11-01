"""Wave 52E (`EXPERIMENT_PLAN_069.md`) Runner:
Interior Quantum Purification, CPTP Kraus Channels, Outer-Product Coupling Modes, and
Multi-Granularity Hilbert Shapes (`(V, 2, 256)`, `(V, 48, 2, 8)`, `(V, 256, 2, 2)`) under SVD-Stiefel Calibration.
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
from qdlblocks.layers.quantum_statmech_channels import PurificationMode, QuantumChannelMode

_ex75 = import_module("75_verify_plan065_frozen_trunk_quantum_frontend_tournament")
run_single_graft_arm = _ex75.run_single_graft_arm

CACHE_DIR: Path = REPO_ROOT / ".cache"
DEFAULT_JSON_OUT: Path = CACHE_DIR / "w52e_plan069_interior_physics_and_shapes.json"


def _make_w52e_cfg(
    wf_shape: QuantumWavefunctionShape = QuantumWavefunctionShape.SHAPE_24x2x16,
    state_io: QuantumStateIOMode = QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
    interior_mode: QuantumAttnInteriorMode = QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
    purify_mode: PurificationMode = PurificationMode.QUADRATIC_MCWEENY,
    channel_mode: QuantumChannelMode = QuantumChannelMode.UNITARY_U16,
    warmup_steps: int = 300,
) -> QuantumFrontEndGraftConfig:
    """Construct a calibrated `QuantumFrontEndGraftConfig` for Wave 52E (`<= 8` args)."""
    return QuantumFrontEndGraftConfig(
        wf_shape=wf_shape,
        state_io_mode=state_io,
        attn_interior_mode=interior_mode,
        purification_mode=purify_mode,
        channel_mode=channel_mode,
        replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
        interface_warmup_steps=warmup_steps,
        interface_align_weight=0.5,
        wte_lr=0.004,
        stiefel_lr=0.012,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
        self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
    )


def resolve_plan069_arm_config(
    arm_id: str,
    wte_lr_override: float | None = None,
    stiefel_lr_override: float | None = None,
    warmup_steps_override: int | None = None,
) -> QuantumFrontEndGraftConfig:
    """Map Plan 069 arm identifier (`W1..W12`) to its `QuantumFrontEndGraftConfig`."""
    key = arm_id.strip().upper()
    presets: dict[str, QuantumFrontEndGraftConfig] = {
        "W1_MCWEENY_U16_CAUSAL_ATT_V": _make_w52e_cfg(),
        "W2_CUBIC_FD_U16_CAUSAL_ATT_V": _make_w52e_cfg(
            purify_mode=PurificationMode.CUBIC_FERMI_DIRAC,
        ),
        "W3_LEARN_TEMP_U16_CAUSAL_ATT_V": _make_w52e_cfg(
            purify_mode=PurificationMode.LEARNABLE_TEMP_MCWEENY,
        ),
        "W4_MCWEENY_KRAUS_CAUSAL_ATT_V": _make_w52e_cfg(
            channel_mode=QuantumChannelMode.CPTP_KRAUS_RANK_K,
        ),
        "W5_PURE_WF_IO_ATT_V": _make_w52e_cfg(
            state_io=QuantumStateIOMode.WF_IN_WF_OUT,
        ),
        "W6_HYBRID_BORN_OBS_ATT_V": _make_w52e_cfg(
            state_io=QuantumStateIOMode.WF_IN_DENSITY_OBS_OUT,
        ),
        "W7_KV_CROSS_OUTER_ON_Q": _make_w52e_cfg(
            state_io=QuantumStateIOMode.WF_IN_WF_OUT,
            interior_mode=QuantumAttnInteriorMode.KV_CROSS_OUTER_ON_Q,
        ),
        "W8_SHAPE_1X2X256_SVD": _make_w52e_cfg(
            wf_shape=QuantumWavefunctionShape.SHAPE_1x2x256,
        ),
        "W9_SHAPE_48X2X8_SVD": _make_w52e_cfg(
            wf_shape=QuantumWavefunctionShape.SHAPE_48x2x8,
        ),
        "W10_SHAPE_256X2X2_QUBIT_SVD": _make_w52e_cfg(
            wf_shape=QuantumWavefunctionShape.SHAPE_256x2x2,
            state_io=QuantumStateIOMode.WF_IN_WF_OUT,
        ),
        "W11_CHAMPION_300M_RECOVERY_1200STEPS": _make_w52e_cfg(
            warmup_steps=500,
        ),
        "W12_CHAMPION_50M_EXCEED_800STEPS": _make_w52e_cfg(
            warmup_steps=400,
        ),
    }
    if key not in presets:
        raise ValueError(f"Unknown Plan 069 arm: {arm_id}. Valid arms: {list(presets.keys())}")
    cfg = presets[key]
    if wte_lr_override is not None and wte_lr_override > 0.0:
        cfg.wte_lr = float(wte_lr_override)
    if stiefel_lr_override is not None and stiefel_lr_override > 0.0:
        cfg.stiefel_lr = float(stiefel_lr_override)
    if warmup_steps_override is not None and warmup_steps_override >= 0:
        cfg.interface_warmup_steps = int(warmup_steps_override)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 52E (Plan 069) Interior Physics & Shapes SVD Graft Runner")
    parser.add_argument("--arm", type=str, default="W1_MCWEENY_U16_CAUSAL_ATT_V")
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
        cfg = resolve_plan069_arm_config(arm_id, wte_lr_ov, stiefel_lr_ov, warmup_ov)
        eff_ckpt = DEFAULT_TEACHER_50M_CKPT_PATH if "50M_" in arm_id.upper() else teacher_ckpt
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
        res["config"]["purification_mode"] = cfg.purification_mode.value
        res["config"]["channel_mode"] = cfg.channel_mode.value
        res["config"]["stiefel_proj_mode"] = cfg.stiefel_proj_mode.value
        res["config"]["self_ortho_mode"] = cfg.self_ortho_mode.value
        res["config"]["wte_lr"] = cfg.wte_lr
        res["config"]["stiefel_lr"] = cfg.stiefel_lr
        results[arm_id] = res
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results if args.arms else res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
