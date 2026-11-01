"""Verification unit tests (`H1..H4`) for `EXPERIMENT_PLAN_069.md` (Wave 52E Interior Physics & Shapes SVD Graft)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

from qdlblocks.layers.quantum_frontend_graft import (
    DEFAULT_TEACHER_300M_CKPT_PATH,
    FrozenTrunkQuantumFrontEndGPT,
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
    SVDStiefelWavepacketStatMechBlock,
    build_frozen_trunk_quantum_frontend_optimizers,
)
from qdlblocks.layers.quantum_statmech_channels import PurificationMode, QuantumChannelMode

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
TARGET_FILES: tuple[Path, ...] = (
    REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_frontend_graft.py",
    REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_svd_stiefel_graft.py",
    REPO_ROOT / "examples" / "79_verify_plan069_interior_physics_and_shapes_svd_graft.py",
    REPO_ROOT / "tests" / "test_plan069_interior_physics_and_shapes_svd_graft.py",
)
MAX_SLOC_LIMIT: int = 1000
MAX_FUNC_ARGS_LIMIT: int = 8


def _compute_sloc_without_docstrings(py_path: Path) -> int:
    source = py_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first_stmt = node.body[0]
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                for ln in range(first_stmt.lineno, (first_stmt.end_lineno or first_stmt.lineno) + 1):
                    docstring_lines.add(ln)
    sloc = 0
    for idx, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or idx in docstring_lines:
            continue
        sloc += 1
    return sloc


def _complex_stiefel_orthogonality_error(u_re: torch.Tensor, u_im: torch.Tensor) -> float:
    """Compute max_h ||U_h^dagger U_h - I||_F / sqrt(d_h) for complex unitary/Kraus operator pair."""
    ur, ui = u_re.float(), u_im.float()
    d_cols = ur.size(-1)
    eye = torch.eye(d_cols, device=ur.device, dtype=torch.float32)
    gram_re = torch.matmul(ur.transpose(-2, -1), ur) + torch.matmul(ui.transpose(-2, -1), ui)
    gram_im = torch.matmul(ur.transpose(-2, -1), ui) - torch.matmul(ui.transpose(-2, -1), ur)
    err_sq = (gram_re - eye).pow(2).sum(dim=(-2, -1)) + gram_im.pow(2).sum(dim=(-2, -1))
    return float((err_sq.sqrt() / math.sqrt(float(d_cols))).max().item())


def test_h1_calibrated_256qubit_and_1x2x256_svd_stiefel_verification() -> None:
    """Gate H1: Verify SHAPE_256x2x2 (256 qubits) and SHAPE_1x2x256 under BIORTHOGONAL_SVD_STIEFEL_R768."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for wf_shape in (QuantumWavefunctionShape.SHAPE_256x2x2, QuantumWavefunctionShape.SHAPE_1x2x256):
        cfg = QuantumFrontEndGraftConfig(
            wf_shape=wf_shape,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=100,
            interface_align_weight=0.5,
            stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        )
        model = FrozenTrunkQuantumFrontEndGPT.from_teacher_checkpoint(DEFAULT_TEACHER_300M_CKPT_PATH, cfg, device)
        assert isinstance(model.quantum_block0, SVDStiefelWavepacketStatMechBlock)
        trunk_snap = {k: v.detach().clone() for k, v in model.classical.named_parameters()}
        opt = build_frozen_trunk_quantum_frontend_optimizers(model)

        idx = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
        targets = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
        for _ in range(2):
            opt.zero_grad()
            _, loss = model(idx, targets=targets)
            assert loss is not None and torch.isfinite(loss)
            loss.backward()
            opt.step(idx, lr_mult=1.0)

        max_trunk_diff = max(
            float((p.detach() - trunk_snap[k]).abs().max().item()) for k, p in model.classical.named_parameters()
        )
        assert max_trunk_diff == 0.0, f"Trunk mutated for {wf_shape}: {max_trunk_diff}"
        tel = model.compute_quantum_graft_telemetry()
        assert 0.0 < tel["q_l0_purity"] <= 1.0001
        assert tel["q_l0_eff_rank"] >= 0.9999
        assert tel["q_max_stiefel_err"] < 1e-4
        assert tel["q_wte_sphere_err"] < 1e-4
    print("PASS: test_h1_calibrated_256qubit_and_1x2x256_svd_stiefel_verification")


def test_h2_purification_modes_and_cptp_kraus_physicality_under_svd_stiefel() -> None:
    """Gate H2: Verify all 3 PurificationModes and both QuantumChannelModes under SVD-Stiefel."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    combos = (
        (PurificationMode.QUADRATIC_MCWEENY, QuantumChannelMode.UNITARY_U16),
        (PurificationMode.CUBIC_FERMI_DIRAC, QuantumChannelMode.UNITARY_U16),
        (PurificationMode.LEARNABLE_TEMP_MCWEENY, QuantumChannelMode.UNITARY_U16),
        (PurificationMode.QUADRATIC_MCWEENY, QuantumChannelMode.CPTP_KRAUS_RANK_K),
    )
    idx = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
    targets = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
    for pur_mode, chan_mode in combos:
        cfg = QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            purification_mode=pur_mode,
            channel_mode=chan_mode,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        )
        model = FrozenTrunkQuantumFrontEndGPT.from_teacher_checkpoint(DEFAULT_TEACHER_300M_CKPT_PATH, cfg, device)
        assert isinstance(model.quantum_block0, SVDStiefelWavepacketStatMechBlock)
        opt = build_frozen_trunk_quantum_frontend_optimizers(model)
        opt.zero_grad()
        _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        u_re, u_im = model.quantum_block0.complex_unitary_pair()
        assert u_re.grad is not None and torch.isfinite(u_re.grad).all() and u_re.grad.abs().max().item() > 0.0
        assert u_im.grad is not None and torch.isfinite(u_im.grad).all() and u_im.grad.abs().max().item() > 0.0
        if pur_mode == PurificationMode.LEARNABLE_TEMP_MCWEENY:
            pl = model.quantum_block0.core_block.purify_logits
            assert pl is not None and pl.grad is not None and pl.grad.abs().max().item() > 0.0
        opt.step(idx, lr_mult=1.0)
        u_err = _complex_stiefel_orthogonality_error(u_re, u_im)
        assert u_err < 1e-4, f"Complex unitary/Kraus orthogonality error {u_err} >= 1e-4 for {(pur_mode, chan_mode)}"
    print("PASS: test_h2_purification_modes_and_cptp_kraus_physicality_under_svd_stiefel")


def test_h3_interior_coupling_and_state_io_modes_under_svd_stiefel() -> None:
    """Gate H3: Verify all interior coupling and state I/O modes under SVD-Stiefel have zero dead parameters."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    modes = (
        (QuantumStateIOMode.WF_IN_WF_OUT, QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V),
        (QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO, QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V),
        (QuantumStateIOMode.WF_IN_WF_OUT, QuantumAttnInteriorMode.KV_CROSS_OUTER_ON_Q),
        (QuantumStateIOMode.WF_IN_DENSITY_OBS_OUT, QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V),
    )
    idx = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
    targets = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
    for sio, aim in modes:
        cfg = QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=sio,
            attn_interior_mode=aim,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        )
        model = FrozenTrunkQuantumFrontEndGPT.from_teacher_checkpoint(DEFAULT_TEACHER_300M_CKPT_PATH, cfg, device)
        assert isinstance(model.quantum_block0, SVDStiefelWavepacketStatMechBlock)
        _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        for p in model.active_quantum_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"Missing/non-finite grad for {(sio, aim)}"
        if sio == QuantumStateIOMode.WF_IN_DENSITY_OBS_OUT:
            dg = model.quantum_block0.core_block.diag_obs_gamma
            assert dg.grad is not None and dg.grad.abs().max().item() > 0.0
    print("PASS: test_h3_interior_coupling_and_state_io_modes_under_svd_stiefel")


def test_h4_best_practices_and_sloc_gate() -> None:
    """Gate H4: Verify <= 1000 SLOC, <= 8 function arguments, and zero semicolon packing across Plan 069 files."""
    for py_path in TARGET_FILES:
        if not py_path.exists():
            continue
        sloc = _compute_sloc_without_docstrings(py_path)
        assert sloc <= MAX_SLOC_LIMIT, f"{py_path.name} SLOC {sloc} > {MAX_SLOC_LIMIT}"
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arg_names = [a.arg for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)]
                if arg_names and arg_names[0] in ("self", "cls"):
                    arg_names = arg_names[1:]
                n_args = len(arg_names) + (1 if node.args.vararg else 0) + (1 if node.args.kwarg else 0)
                assert n_args <= MAX_FUNC_ARGS_LIMIT, f"{py_path.name}:{node.name} has {n_args} args > {MAX_FUNC_ARGS_LIMIT}"
    print("PASS: test_h4_best_practices_and_sloc_gate")


if __name__ == "__main__":
    test_h1_calibrated_256qubit_and_1x2x256_svd_stiefel_verification()
    test_h2_purification_modes_and_cptp_kraus_physicality_under_svd_stiefel()
    test_h3_interior_coupling_and_state_io_modes_under_svd_stiefel()
    test_h4_best_practices_and_sloc_gate()
    print("ALL PLAN 069 GATES (H1..H4) PASSED.")
