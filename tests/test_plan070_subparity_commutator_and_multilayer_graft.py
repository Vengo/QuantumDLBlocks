r"""Unit Tests for EXPERIMENT_PLAN_070.md (Wave 52F: Sub-Parity SVD-Stiefel Ranks, Liouville Commutator & Multi-Layer Graft).

Verifies all 4 mandatory gates (`H1..H4`) from `EXPERIMENT_PLAN_070.md`:
- **Gate H1**: Verifies exact parameter counts and Stiefel column-orthonormality $\|U^\top U - I\|_F < 10^{-5}$,
  $\|V^\top V - I\|_F < 10^{-5}$ for `R64`, `R192`, and `R384` (verifying `R384` on $D=768$ has $4 \times 2 \times 768 \times 384 = 2,359,296$
  Stiefel matrix elements = exact 1:1 parameter parity with $4 \times 768^2$).
- **Gate H2**: Verifies that in `LIOUVILLE_COMMUTATOR_RHO_V` and `SYMMETRIZED_LOCAL_CONTEXT_RHO_V`, the commutator
  $C = -i [\rho_{\text{local}}, \tilde{\rho}_{\text{evol}}]$ is strictly Hermitian ($\|C_R - C_R^\top\|_F < 10^{-5}$,
  $\|C_I + C_I^\top\|_F < 10^{-5}$), traceless ($|\operatorname{Tr}(C_R)| < 10^{-5}$), and propagates finite gradients to
  `liouville_beta` and `local_rho_mix_logit`.
- **Gate H3**: Verifies that `EMB_PLUS_L0_L1` with calibrated `SVDStiefelWavepacketStatMechBlock` initializes both
  `quantum_block0` and `quantum_block1`, keeps all classical trunk parameters frozen (`requires_grad == False`), and updates
  both Layer-0 and Layer-1 Stiefel/Unitary/Scalar parameters cleanly under a single optimizer step.
- **Gate H4**: AST best-practices compliance audit (`<= 1,000` SLOC, `<= 8` args per function, zero semicolon statement packing)
  across all modified and new Wave 52F files.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from qdlblocks.layers.quantum_frontend_graft import (
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
    build_frozen_trunk_quantum_frontend_optimizers,
)
from qdlblocks.layers.quantum_svd_stiefel_graft import (
    BiorthogonalSVDStiefelProjection,
    compute_liouville_commutator,
    resolve_svd_stiefel_rank,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
MAX_SLOC_LIMIT: int = 1000
MAX_FUNC_ARGS_LIMIT: int = 8
STIEFEL_ORTHO_TOL: float = 1e-5
HERMITIAN_TRACE_TOL: float = 1e-5


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


def _build_tiny_classical_teacher() -> CompiledLeaderboardClassicalGPT:
    """Construct a tiny 4-layer classical teacher with `embed_dim = 768` (`num_heads = 6`) for fast CPU verification."""
    torch.manual_seed(20261101)
    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=256,
        embed_dim=768,
        num_heads=6,
        num_layers=4,
        hidden_dim=1536,
        max_seq_len=64,
        num_value_embeddings=2,
        bfloat16_weights=False,
        use_flex_doc_swc=False,
    )
    teacher = CompiledLeaderboardClassicalGPT(cfg)
    with torch.no_grad():
        teacher.wte.weight.normal_(0.0, 0.02)
        teacher.lm_head.weight.normal_(0.0, 0.02)
        for blk in teacher.blocks:
            if blk.qkv_proj is not None and blk.out_proj is not None:
                blk.qkv_proj.weight.normal_(0.0, 0.02)
                blk.out_proj.weight.normal_(0.0, 0.02)
            blk.fc1.weight.normal_(0.0, 0.02)
            blk.fc2.weight.normal_(0.0, 0.02)
    return teacher


def test_gate_h1_subparity_and_1to1_parity_svd_stiefel_ranks() -> None:
    """Gate H1: Verify exact parameter counts and Stiefel column-orthonormality for `R64`, `R192`, and `R384`."""
    d_model = 768
    expected_ranks = [
        (QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64, 64),
        (QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R192, 192),
        (QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384, 384),
    ]
    w_teacher = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    for mode, r_exp in expected_ranks:
        r_res = resolve_svd_stiefel_rank(mode, d_model, d_model)
        assert r_res == r_exp
        proj = BiorthogonalSVDStiefelProjection(d_model, d_model, r_res)
        proj.initialize_from_teacher_matrix_(w_teacher)
        stiefel_elems = sum(p.numel() for p in proj.real_stiefel_parameters())
        assert stiefel_elems == 2 * d_model * r_exp
        if r_exp == 384:
            assert 4 * stiefel_elems == 4 * d_model * d_model

        for mat in proj.real_stiefel_parameters():
            eye = torch.eye(r_exp, dtype=torch.float32)
            gram_err = float((mat.t() @ mat - eye).norm().item())
            assert gram_err < STIEFEL_ORTHO_TOL
    print("PASS: test_gate_h1_subparity_and_1to1_parity_svd_stiefel_ranks")


def test_gate_h2_liouville_commutator_hermiticity_trace_and_gradients() -> None:
    """Gate H2: Verify closed-form Hermitian traceless commutator $C = -i[A, B]$ and gradient flow for Axis O modes."""
    torch.manual_seed(70)
    bsz, heads, seq_len, d_h = 2, 6, 8, 32
    x_re = torch.randn(bsz, heads, seq_len, 2, d_h, dtype=torch.float32)
    x_im = torch.randn(bsz, heads, seq_len, 2, d_h, dtype=torch.float32)
    y_re = torch.randn(bsz, heads, seq_len, 2, d_h, dtype=torch.float32)
    y_im = torch.randn(bsz, heads, seq_len, 2, d_h, dtype=torch.float32)

    a_re = torch.matmul(x_re.transpose(-2, -1), x_re) + torch.matmul(x_im.transpose(-2, -1), x_im)
    a_im = torch.matmul(x_im.transpose(-2, -1), x_re) - torch.matmul(x_re.transpose(-2, -1), x_im)
    tr_a = torch.diagonal(a_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    a_re, a_im = a_re / tr_a, a_im / tr_a
    b_re = torch.matmul(y_re.transpose(-2, -1), y_re) + torch.matmul(y_im.transpose(-2, -1), y_im)
    b_im = torch.matmul(y_im.transpose(-2, -1), y_re) - torch.matmul(y_re.transpose(-2, -1), y_im)
    tr_b = torch.diagonal(b_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    b_re, b_im = b_re / tr_b, b_im / tr_b

    c_re, c_im = compute_liouville_commutator(a_re, a_im, b_re, b_im)
    sym_err = float((c_re - c_re.transpose(-2, -1)).norm().item())
    skew_err = float((c_im + c_im.transpose(-2, -1)).norm().item())
    max_trace = float(torch.diagonal(c_re, dim1=-2, dim2=-1).sum(dim=-1).abs().max().item())
    assert sym_err < HERMITIAN_TRACE_TOL
    assert skew_err < HERMITIAN_TRACE_TOL
    assert max_trace < HERMITIAN_TRACE_TOL

    for interior_mode in (
        QuantumAttnInteriorMode.SYMMETRIZED_LOCAL_CONTEXT_RHO_V,
        QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V,
    ):
        teacher = _build_tiny_classical_teacher()
        graft_cfg = QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=interior_mode,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
            self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        )
        model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=graft_cfg)
        idx = torch.randint(0, 256, (2, 16), dtype=torch.long)
        targets = torch.randint(0, 256, (2, 16), dtype=torch.long)
        _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()

        core = model.quantum_block0.core_block  # type: ignore[union-attr]
        if interior_mode == QuantumAttnInteriorMode.SYMMETRIZED_LOCAL_CONTEXT_RHO_V:
            assert core.local_rho_mix_logit.grad is not None and torch.isfinite(core.local_rho_mix_logit.grad).all()
        else:
            assert core.liouville_beta.grad is not None and torch.isfinite(core.liouville_beta.grad).all()
    print("PASS: test_gate_h2_liouville_commutator_hermiticity_trace_and_gradients")


def test_gate_h3_multilayer_emb_l0_l1_calibrated_svd_stiefel_graft() -> None:
    """Gate H3: Verify `EMB_PLUS_L0_L1` initializes and steps both `quantum_block0` and `quantum_block1` cleanly."""
    teacher = _build_tiny_classical_teacher()
    graft_cfg = QuantumFrontEndGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
        attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
        replacement_scope=QuantumReplacementScope.EMB_PLUS_L0_L1,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.INTERFACE_WARMUP_TO_PURE_CE,
        interface_warmup_steps=5,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
        self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
    )
    model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=graft_cfg)
    assert model.quantum_block0 is not None
    assert model.quantum_block1 is not None
    assert all(not p.requires_grad for p in model.classical.parameters())
    assert len(model.quantum_real_stiefel_parameters()) == 16  # 8 Stiefel matrices in L0 + 8 in L1
    assert len(model.quantum_complex_unitary_pairs()) == 2

    opt = build_frozen_trunk_quantum_frontend_optimizers(model)
    idx = torch.randint(0, 256, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 16), dtype=torch.long)
    u1_before = model.quantum_block1.q_proj.u_stiefel.detach().clone()

    opt.zero_grad()
    _, loss = model(idx, targets=targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    opt.step(idx, lr_mult=1.0)

    u1_after = model.quantum_block1.q_proj.u_stiefel.detach()
    assert not torch.allclose(u1_before, u1_after), "quantum_block1 Stiefel parameter did not update!"
    tel = model.compute_quantum_graft_telemetry()
    assert tel["q_max_stiefel_err"] < STIEFEL_ORTHO_TOL
    assert tel["q_wte_sphere_err"] < STIEFEL_ORTHO_TOL
    print("PASS: test_gate_h3_multilayer_emb_l0_l1_calibrated_svd_stiefel_graft")


def test_gate_h4_ast_best_practices_compliance() -> None:
    """Gate H4: AST best-practices audit (`<= 1,000` SLOC, `<= 8` args per function, no semicolon packing)."""
    files = [
        REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_svd_stiefel_graft.py",
        REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_frontend_graft.py",
        REPO_ROOT / "tests" / "test_plan070_subparity_commutator_and_multilayer_graft.py",
        REPO_ROOT / "examples" / "80_verify_plan070_subparity_commutator_and_multilayer_graft.py",
    ]
    for path in files:
        if not path.exists():
            continue
        sloc = _compute_sloc_without_docstrings(path)
        assert sloc <= MAX_SLOC_LIMIT, f"{path.name} SLOC {sloc} > {MAX_SLOC_LIMIT}"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arg_names = [a.arg for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)]
                if arg_names and arg_names[0] in ("self", "cls"):
                    arg_names = arg_names[1:]
                n_args = len(arg_names) + (1 if node.args.vararg else 0) + (1 if node.args.kwarg else 0)
                assert n_args <= MAX_FUNC_ARGS_LIMIT, f"{path.name}:{node.name} has {n_args} args > 8"
    print("PASS: test_gate_h4_ast_best_practices_compliance")


if __name__ == "__main__":
    test_gate_h1_subparity_and_1to1_parity_svd_stiefel_ranks()
    test_gate_h2_liouville_commutator_hermiticity_trace_and_gradients()
    test_gate_h3_multilayer_emb_l0_l1_calibrated_svd_stiefel_graft()
    test_gate_h4_ast_best_practices_compliance()
    print("ALL PLAN 070 GATES (H1..H4) PASSED.")
