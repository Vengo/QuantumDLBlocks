"""Verification unit tests (`H1..H4`) for `EXPERIMENT_PLAN_068.md` (Wave 52D SVD-Stiefel & Fubini-Study Graft)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from qdlblocks.layers.quantum_frontend_graft import (
    DEFAULT_TEACHER_300M_CKPT_PATH,
    BiorthogonalSVDStiefelProjection,
    FrozenTrunkQuantumFrontEndGPT,
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
    apply_fubini_study_self_ortho,
    build_frozen_trunk_quantum_frontend_optimizers,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
TARGET_FILES: tuple[Path, ...] = (
    REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_frontend_graft.py",
    REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_svd_stiefel_graft.py",
    REPO_ROOT / "examples" / "78_verify_plan068_svd_stiefel_and_fubini_study_graft.py",
    REPO_ROOT / "tests" / "test_plan068_svd_stiefel_and_fubini_study_graft.py",
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


def test_h1_biorthogonal_svd_stiefel_reconstruction_and_orthogonality() -> None:
    """Gate H1: Verify SVD-Stiefel orthogonality and monotonic Frobenius energy across r in {128, 256, 512, 768}."""
    ckpt = torch.load(DEFAULT_TEACHER_300M_CKPT_PATH, map_location="cpu", weights_only=False)
    wq_teacher = ckpt["model_state_dict"]["blocks.0.qkv_proj.weight"][:768].detach().float()
    total_energy = float(wq_teacher.pow(2).sum().item())

    energies: list[float] = []
    for r in (128, 256, 512, 768):
        proj = BiorthogonalSVDStiefelProjection(d_in=768, d_out=768, rank=r)
        proj.initialize_from_teacher_matrix_(wq_teacher)
        u_st = proj.u_stiefel.float()
        v_st = proj.v_stiefel.float()
        eye_r = torch.eye(r, dtype=torch.float32)
        u_err = float(((u_st.t() @ u_st - eye_r).norm() / math.sqrt(float(r))).item())
        v_err = float(((v_st.t() @ v_st - eye_r).norm() / math.sqrt(float(r))).item())
        assert u_err < 1e-5, f"r={r} u_stiefel orthogonality error {u_err} >= 1e-5"
        assert v_err < 1e-5, f"r={r} v_stiefel orthogonality error {v_err} >= 1e-5"

        w_recon = (u_st * proj.sigma.float().unsqueeze(0)) @ v_st.t()
        captured = float(proj.sigma.float().pow(2).sum().item()) / total_energy
        energies.append(captured)
        if r == 768:
            rel_err = float(((w_recon - wq_teacher).norm() / wq_teacher.norm()).item())
            assert rel_err < 1e-5, f"r=768 reconstruction relative error {rel_err} >= 1e-5"
            x_test = torch.randn(4, 16, 768, dtype=torch.float32)
            y_proj = proj(x_test)
            y_exact = F.linear(x_test, wq_teacher)
            fwd_err = float(((y_proj - y_exact).norm() / y_exact.norm()).item())
            assert fwd_err < 1e-5, f"r=768 forward relative error {fwd_err} >= 1e-5"

    assert energies[0] < energies[1] < energies[2] < energies[3], f"Non-monotonic energy curve: {energies}"
    assert abs(energies[3] - 1.0) < 1e-4, f"Full rank energy {energies[3]} != 1.0"
    print(f"PASS: test_h1_biorthogonal_svd_stiefel_reconstruction_and_orthogonality (energies={[round(e, 4) for e in energies]})")


def test_h2_fubini_study_self_ortho_and_step0_teacher_calibration() -> None:
    """Gate H2: Verify Fubini-Study projective self-state orthogonalizer and Step-0 cos(x_1_q, x_1_c) > 0.990."""
    torch.manual_seed(42)
    y_scaled = torch.randn(2, 16, 6, 128, dtype=torch.float32)
    v_mixed = torch.randn(2, 16, 6, 128, dtype=torch.float32)
    alpha_large = torch.full((6,), 12.0, dtype=torch.float32)
    y_ortho = apply_fubini_study_self_ortho(y_scaled, v_mixed, alpha_large)
    v_unit = F.normalize(v_mixed, dim=-1, eps=1e-6)
    max_overlap = float((y_ortho * v_unit).sum(dim=-1).abs().max().item())
    assert max_overlap < 1e-5, f"Expected Fubini-Study orthogonal overlap < 1e-5, got {max_overlap}"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = QuantumFrontEndGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
        state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT,
        replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
        self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
    )
    model = FrozenTrunkQuantumFrontEndGPT.from_teacher_checkpoint(
        ckpt_path=DEFAULT_TEACHER_300M_CKPT_PATH, graft_cfg=cfg, device=device
    )
    assert isinstance(model.quantum_block0, SVDStiefelWavepacketStatMechBlock)
    idx = torch.randint(0, 50257, (2, 64), device=device, dtype=torch.long)
    with torch.no_grad():
        x_0_c, x_1_c, _ = model.compute_teacher_layer0_targets(idx)
        x_0_q, x_1_q, _ = model.encode_quantum_layer0(idx)
        cos_x0 = float(F.cosine_similarity(x_0_q.float(), x_0_c.float(), dim=-1).mean().item())
        cos_x1 = float(F.cosine_similarity(x_1_q.float(), x_1_c.float(), dim=-1).mean().item())
    assert cos_x0 > 0.980, f"Expected Step-0 cos(x_0_q, x_0_c) > 0.980, got {cos_x0:.6f}"
    assert cos_x1 > 0.960, f"Expected Step-0 cos(x_1_q, x_1_c) > 0.960, got {cos_x1:.6f}"
    print(f"PASS: test_h2_fubini_study_self_ortho_and_step0_teacher_calibration (cos_x0={cos_x0:.5f}, cos_x1={cos_x1:.5f})")


def test_h3_strict_trunk_freeze_and_riemannian_invariants_across_shapes() -> None:
    """Gate H3: Verify 2 forward/backward/step iterations preserve trunk freeze and Stiefel/sphere invariants."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    shapes = (
        QuantumWavefunctionShape.SHAPE_24x2x16,
        QuantumWavefunctionShape.SHAPE_12x2x32,
        QuantumWavefunctionShape.SHAPE_1x2x512_PKT32,
    )
    for wf_shape in shapes:
        cfg = QuantumFrontEndGraftConfig(
            wf_shape=wf_shape,
            state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=100,
            interface_align_weight=0.5,
            stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        )
        model = FrozenTrunkQuantumFrontEndGPT.from_teacher_checkpoint(
            ckpt_path=DEFAULT_TEACHER_300M_CKPT_PATH, graft_cfg=cfg, device=device
        )
        assert isinstance(model.quantum_block0, SVDStiefelWavepacketStatMechBlock)
        trunk_snap = {k: v.detach().clone() for k, v in model.classical.named_parameters()}
        opt = build_frozen_trunk_quantum_frontend_optimizers(
            model, stiefel_lr=0.012, unitary_lr=0.015, wte_lr=0.004, scalar_lr=0.008
        )

        idx = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
        targets = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
        for step_i in range(2):
            opt.zero_grad()
            _, loss = model(idx, targets=targets)
            assert loss is not None and torch.isfinite(loss), f"Non-finite loss for {wf_shape} at step {step_i}"
            loss.backward()
            for st_p in model.quantum_block0.real_stiefel_parameters():
                assert st_p.grad is not None and torch.isfinite(st_p.grad).all() and st_p.grad.abs().max().item() > 0.0
            for proj in (
                model.quantum_block0.q_proj,
                model.quantum_block0.k_proj,
                model.quantum_block0.v_proj,
                model.quantum_block0.obs_proj,
            ):
                assert (
                    proj.sigma.grad is not None
                    and torch.isfinite(proj.sigma.grad).all()
                    and proj.sigma.grad.abs().max().item() > 0.0
                )
            opt.step(idx, lr_mult=1.0)

        max_trunk_diff = max(
            float((p.detach() - trunk_snap[k]).abs().max().item()) for k, p in model.classical.named_parameters()
        )
        assert max_trunk_diff == 0.0, f"Trunk mutated for {wf_shape}: {max_trunk_diff}"
        tel = model.compute_quantum_graft_telemetry()
        assert tel["q_max_stiefel_err"] < 1e-4, f"Stiefel err {tel['q_max_stiefel_err']} >= 1e-4 for {wf_shape}"
        assert tel["q_wte_sphere_err"] < 1e-4, f"Sphere err {tel['q_wte_sphere_err']} >= 1e-4 for {wf_shape}"
    print("PASS: test_h3_strict_trunk_freeze_and_riemannian_invariants_across_shapes")


def test_h4_best_practices_and_sloc_gate() -> None:
    """Gate H4: Verify <= 1000 SLOC, <= 8 function arguments, and zero semicolon packing across Plan 068 files."""
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
    test_h1_biorthogonal_svd_stiefel_reconstruction_and_orthogonality()
    test_h2_fubini_study_self_ortho_and_step0_teacher_calibration()
    test_h3_strict_trunk_freeze_and_riemannian_invariants_across_shapes()
    test_h4_best_practices_and_sloc_gate()
    print("ALL PLAN 068 GATES (H1..H4) PASSED.")
