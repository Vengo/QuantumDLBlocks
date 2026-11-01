r"""Wave 52F (`EXPERIMENT_PLAN_070.md`) Runner:
Sub-Parity & 1:1 Parameter-Matched SVD-Stiefel Ranks ($r \in \{64, 128, 192, 256, 384\}$) on `(V, 12, 2, 32)`,
Interior Liouville-von Neumann Commutator & Local-Value Couplings, and Multi-Layer (`Emb + L0 + L1`) Calibrated Grafting.
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
DEFAULT_JSON_OUT: Path = CACHE_DIR / "w52f_plan070_subparity_commutator_multilayer.json"


def _make_w52f_cfg(
    wf_shape: QuantumWavefunctionShape = QuantumWavefunctionShape.SHAPE_12x2x32,
    proj_mode: QuantumStiefelProjMode = QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
    interior_mode: QuantumAttnInteriorMode = QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
    scope: QuantumReplacementScope = QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
    purify_mode: PurificationMode = PurificationMode.QUADRATIC_MCWEENY,
    state_io: QuantumStateIOMode = QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
    warmup_steps: int = 300,
) -> QuantumFrontEndGraftConfig:
    """Construct a calibrated `QuantumFrontEndGraftConfig` for Wave 52F (`<= 8` args)."""
    return QuantumFrontEndGraftConfig(
        wf_shape=wf_shape,
        state_io_mode=state_io,
        attn_interior_mode=interior_mode,
        purification_mode=purify_mode,
        channel_mode=QuantumChannelMode.UNITARY_U16,
        replacement_scope=scope,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
        interface_warmup_steps=warmup_steps,
        interface_align_weight=0.5,
        wte_lr=0.004,
        stiefel_lr=0.012,
        stiefel_proj_mode=proj_mode,
        self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
    )


def resolve_plan070_arm_config(
    arm_id: str,
    wte_lr_override: float | None = None,
    stiefel_lr_override: float | None = None,
    warmup_steps_override: int | None = None,
) -> QuantumFrontEndGraftConfig:
    """Map Plan 070 arm identifier (`F1..F11`) to its `QuantumFrontEndGraftConfig`."""
    key = arm_id.strip().upper()
    presets: dict[str, QuantumFrontEndGraftConfig] = {
        "F1_R64_SUBPARITY_6X_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
        ),
        "F2_R128_SUBPARITY_3X_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128,
        ),
        "F3_R192_SUBPARITY_2X_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R192,
        ),
        "F4_R256_SUBPARITY_1P5X_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R256,
        ),
        "F5_R384_PARITY_1X_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        ),
        "F6_R384_PARITY_1X_24X2X16": _make_w52f_cfg(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        ),
        "F7_R384_LOCAL_V_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        ),
        "F8_R384_SYM_LOCAL_CONTEXT_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            interior_mode=QuantumAttnInteriorMode.SYMMETRIZED_LOCAL_CONTEXT_RHO_V,
        ),
        "F9_R384_LIOUVILLE_COMM_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            interior_mode=QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V,
        ),
        "F10_MULTILAYER_L0_L1_R384_12X2X32": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            scope=QuantumReplacementScope.EMB_PLUS_L0_L1,
        ),
        "F11_SUBPARITY_R192_EXCEED_300M_800STEPS": _make_w52f_cfg(
            proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R192,
            purify_mode=PurificationMode.CUBIC_FERMI_DIRAC,
            state_io=QuantumStateIOMode.WF_IN_WF_OUT,
            warmup_steps=400,
        ),
    }
    if key not in presets:
        raise ValueError(f"Unknown Plan 070 arm: {arm_id}. Valid arms: {list(presets.keys())}")
    cfg = presets[key]
    if wte_lr_override is not None and wte_lr_override > 0.0:
        cfg.wte_lr = float(wte_lr_override)
    if stiefel_lr_override is not None and stiefel_lr_override > 0.0:
        cfg.stiefel_lr = float(stiefel_lr_override)
    if warmup_steps_override is not None and warmup_steps_override >= 0:
        cfg.interface_warmup_steps = int(warmup_steps_override)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 52F (Plan 070) Sub-Parity, Commutator & Multi-Layer Graft Runner")
    parser.add_argument("--arm", type=str, default="F5_R384_PARITY_1X_12X2X32")
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
        cfg = resolve_plan070_arm_config(arm_id, wte_lr_ov, stiefel_lr_ov, warmup_ov)
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
