#!/usr/bin/env python3
r"""Verification Runner for `EXPERIMENT_PLAN_072.md` (`Wave 52H`).

Evaluates **Axes T, U, and V** (`7` arms `H72_1..H72_7`) on the frozen Pretrained `300M` Classical Teacher:
- **`H72_1_R384_PARITY_LOCAL_V_800STEPS`**: `R384` (`1.00x` parity) + `ATTENDED_RHO_ON_LOCAL_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`800` steps)
- **`H72_2_R384_PARITY_LIOUVILLE_800STEPS`**: `R384` (`1.00x` parity) + `LIOUVILLE_COMMUTATOR_RHO_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`800` steps)
- **`H72_3_R128_SUBPARITY_3X_800STEPS`**: `R128` (`0.34x` = `3x` smaller) + `ATTENDED_RHO_ON_LOCAL_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`800` steps)
- **`H72_4_R64_SUBPARITY_6X_800STEPS`**: `R64` (`0.17x` = `6x` smaller) + `ATTENDED_RHO_ON_LOCAL_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`800` steps)
- **`H72_5_R768_LOCAL_V_BERRY_WF_500S`**: `R768` + `ATTENDED_RHO_ON_LOCAL_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`500` steps)
- **`H72_6_R768_LIOUVILLE_BERRY_WF_500S`**: `R768` + `LIOUVILLE_COMMUTATOR_RHO_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`500` steps)
- **`H72_7_MULTILAYER_L0_L1_R768_800S`**: `EMB_PLUS_L0_L1` (`R768` x 2) + `ATTENDED_RHO_ON_LOCAL_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`800` steps)
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "examples"))

import torch

from qdlblocks.layers.quantum_frontend_graft import (  # noqa: E402
    DEFAULT_TEACHER_300M_CKPT_PATH,
    QuantumAttnInteriorMode,
    QuantumFrontEndGraftConfig,
    QuantumQKPhaseMode,
    QuantumRecoveryScheduleMode,
    QuantumReplacementScope,
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

_ex75 = import_module("75_verify_plan065_frozen_trunk_quantum_frontend_tournament")
run_single_graft_arm = _ex75.run_single_graft_arm

DEFAULT_OUT_JSON: Path = _REPO_ROOT / ".cache" / "w52h_plan072_compression_frontier_and_local_v.json"
DEFAULT_WARMUP_STEPS: int = 300


def _make_w52h_cfg(
    rank_mode: QuantumStiefelProjMode = QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
    interior_mode: QuantumAttnInteriorMode = QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
    scope: QuantumReplacementScope = QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
    io_mode: QuantumStateIOMode = QuantumStateIOMode.WF_IN_WF_OUT,
    qk_mode: QuantumQKPhaseMode = QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
) -> QuantumFrontEndGraftConfig:
    """Build a calibrated `QuantumFrontEndGraftConfig` for Wave 52H (`<= 8` args)."""
    return QuantumFrontEndGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        state_io_mode=io_mode,
        attn_interior_mode=interior_mode,
        replacement_scope=scope,
        purification_mode=PurificationMode.QUADRATIC_MCWEENY,
        channel_mode=QuantumChannelMode.UNITARY_U16,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
        interface_warmup_steps=warmup_steps,
        interface_align_weight=0.5,
        wte_lr=0.004,
        stiefel_lr=0.012,
        stiefel_proj_mode=rank_mode,
        self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        qk_phase_mode=qk_mode,
    )


ARM_SPECS: dict[str, tuple[Path, QuantumFrontEndGraftConfig]] = {
    "H72_1_R384_PARITY_LOCAL_V_800STEPS": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w52h_cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        ),
    ),
    "H72_2_R384_PARITY_LIOUVILLE_800STEPS": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w52h_cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            interior_mode=QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V,
        ),
    ),
    "H72_3_R128_SUBPARITY_3X_800STEPS": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w52h_cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128,
            interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        ),
    ),
    "H72_4_R64_SUBPARITY_6X_800STEPS": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w52h_cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
            interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        ),
    ),
    "H72_5_R768_LOCAL_V_BERRY_WF_500S": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w52h_cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        ),
    ),
    "H72_6_R768_LIOUVILLE_BERRY_WF_500S": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w52h_cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            interior_mode=QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V,
        ),
    ),
    "H72_7_MULTILAYER_L0_L1_R768_800S": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _make_w52h_cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            scope=QuantumReplacementScope.EMB_PLUS_L0_L1,
        ),
    ),
}


def resolve_plan072_arm_config(arm_id: str) -> QuantumFrontEndGraftConfig:
    """Resolve `arm_id` (`H72_1..H72_7`) to its `QuantumFrontEndGraftConfig`."""
    key = arm_id.strip().upper()
    if key not in ARM_SPECS:
        raise ValueError(f"Unknown Plan 072 arm: {arm_id}. Valid: {list(ARM_SPECS.keys())}")
    return ARM_SPECS[key][1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 072 Compression Frontier & Local-V Synergy Runner")
    parser.add_argument("--arms", type=str, default="H72_1_R384_PARITY_LOCAL_V_800STEPS")
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
        res = run_single_graft_arm(
            arm_id=arm_id,
            graft_cfg=cfg,
            teacher_ckpt=ckpt_path,
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
        res["config"]["qk_phase_mode"] = cfg.qk_phase_mode.value
        res["config"]["wte_lr"] = cfg.wte_lr
        res["config"]["stiefel_lr"] = cfg.stiefel_lr
        results[arm_id] = res
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
