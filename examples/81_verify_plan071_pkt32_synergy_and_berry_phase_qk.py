#!/usr/bin/env python3
r"""Verification Runner for `EXPERIMENT_PLAN_071.md` (`Wave 52G`).

Evaluates **Axes Q, R, and S** (`8` arms `G71_1..G71_8`) on the frozen Pretrained `300M` Classical Teacher:
- **Phase 1 (`500` steps = `4.10M` tokens, `G71_1..G71_6`)**:
  - `G71_1_PKT32_PURE_WF_IO`: `(V, 12, 2, 32)` + `R768` + `WF_IN_WF_OUT`
  - `G71_2_PKT32_CUBIC_FD`: `(V, 12, 2, 32)` + `R768` + `CUBIC_FERMI_DIRAC` ($3\rho^2 - 2\rho^3$)
  - `G71_3_PKT32_KRAUS_RANK4`: `(V, 12, 2, 32)` + `R768` + `CPTP_KRAUS_RANK_K` ($K=4$)
  - `G71_4_PKT32_BERRY_PHASE_QK`: `(V, 12, 2, 32)` + `R768` + `BERRY_PHASE_COMPLEX_QK`
  - `G71_5_PKT32_SYNERGY_FULL`: `(V, 12, 2, 32)` + `R768` + `WF_IN_WF_OUT` + `CUBIC_FERMI_DIRAC` + `BERRY_PHASE_COMPLEX_QK`
  - `G71_6_PKT32_SYNERGY_R384_PARITY`: `(V, 12, 2, 32)` + `R384` (1:1 Parity, `2.36M` params) + `WF_IN_WF_OUT` + `CUBIC_FERMI_DIRAC` + `BERRY_PHASE_COMPLEX_QK`
- **Phase 2 (`1,200` steps = `9.83M` tokens = `3.28%` of `300M`, `G71_7..G71_8`)**:
  - `G71_7_CHAMPION_R768_1200STEPS`: `1,200`-step extended run of `(V, 12, 2, 32)` at $r=768$.
  - `G71_8_CHAMPION_R384_PARITY_1200STEPS`: `1,200`-step extended run of `(V, 12, 2, 32)` at exact 1:1 parameter parity ($r=384$, `2.36M` params).
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

DEFAULT_OUT_JSON: Path = _REPO_ROOT / ".cache" / "w52g_plan071_pkt32_synergy_and_berry_qk.json"
DEFAULT_WARMUP_STEPS: int = 300
DEFAULT_EXTENDED_WARMUP_STEPS: int = 450


def _cfg(
    rank_mode: QuantumStiefelProjMode = QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
    io_mode: QuantumStateIOMode = QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
    pur_mode: PurificationMode = PurificationMode.QUADRATIC_MCWEENY,
    chan_mode: QuantumChannelMode = QuantumChannelMode.UNITARY_U16,
    qk_mode: QuantumQKPhaseMode = QuantumQKPhaseMode.STANDARD_REAL_DOT,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
) -> QuantumFrontEndGraftConfig:
    return QuantumFrontEndGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        state_io_mode=io_mode,
        attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
        replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        purification_mode=pur_mode,
        channel_mode=chan_mode,
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
    "G71_1_PKT32_PURE_WF_IO": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _cfg(io_mode=QuantumStateIOMode.WF_IN_WF_OUT),
    ),
    "G71_2_PKT32_CUBIC_FD": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _cfg(pur_mode=PurificationMode.CUBIC_FERMI_DIRAC),
    ),
    "G71_3_PKT32_KRAUS_RANK4": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _cfg(chan_mode=QuantumChannelMode.CPTP_KRAUS_RANK_K),
    ),
    "G71_4_PKT32_BERRY_PHASE_QK": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _cfg(qk_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK),
    ),
    "G71_5_PKT32_SYNERGY_FULL": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _cfg(
            io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            pur_mode=PurificationMode.CUBIC_FERMI_DIRAC,
            qk_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        ),
    ),
    "G71_6_PKT32_SYNERGY_R384_PARITY": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            pur_mode=PurificationMode.CUBIC_FERMI_DIRAC,
            qk_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        ),
    ),
    "G71_7_CHAMPION_R768_1200STEPS": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _cfg(
            io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            pur_mode=PurificationMode.CUBIC_FERMI_DIRAC,
            qk_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
            warmup_steps=DEFAULT_EXTENDED_WARMUP_STEPS,
        ),
    ),
    "G71_8_CHAMPION_R384_PARITY_1200STEPS": (
        DEFAULT_TEACHER_300M_CKPT_PATH,
        _cfg(
            rank_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            pur_mode=PurificationMode.CUBIC_FERMI_DIRAC,
            qk_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
            warmup_steps=DEFAULT_EXTENDED_WARMUP_STEPS,
        ),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 071 Pkt-32 Synergy & Berry-Phase QK Tournament")
    parser.add_argument("--arms", type=str, default="G71_1_PKT32_PURE_WF_IO,G71_2_PKT32_CUBIC_FD,G71_3_PKT32_KRAUS_RANK4")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
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
