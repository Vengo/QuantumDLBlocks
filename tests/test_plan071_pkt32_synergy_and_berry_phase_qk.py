r"""Pytest verification suite for `EXPERIMENT_PLAN_071.md` (`Wave 52G`).

Verifies Gates `H1..H4`:
- **Gate H1**: `apply_berry_qk_phase` norm preservation ($\|Q'\|_2 = \|Q\|_2$), exact identity at $\Phi = 0$,
  and exact analytical Hermitian inner product $Q'_t \cdot K_s = \operatorname{Re}(e^{-i\Phi}\langle Q_t | K_s \rangle)$ for non-zero $\Phi$.
- **Gate H2**: Finite gradients to `berry_qk_phase` and clamping in `restrict_quantum_scalars_()` under `BERRY_PHASE_COMPLEX_QK`.
- **Gate H3**: All 6 Phase-1 arm configs (`G71_1..G71_6`) instantiate and run forward+backward+step cleanly on `(V, 12, 2, 32)`.
- **Gate H4**: AST best-practices audit (`<= 1,000` SLOC, `<= 8` args per function) across all modified and new files.
"""

from __future__ import annotations

import ast
import tokenize
from pathlib import Path

import torch

from qdlblocks.layers.quantum_frontend_graft import (
    FrozenTrunkQuantumFrontEndGPT,
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
    SVDStiefelWavepacketStatMechBlock,
    apply_berry_qk_phase,
    build_frozen_trunk_quantum_frontend_optimizers,
)
from qdlblocks.layers.quantum_statmech_channels import PurificationMode, QuantumChannelMode
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_MAX_SLOC: int = 1000
_MAX_ARGS: int = 8


def _make_tiny_teacher() -> CompiledLeaderboardClassicalGPT:
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


def _count_sloc(file_path: Path) -> int:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                expr = node.body[0]
                for ln in range(expr.lineno, (expr.end_lineno or expr.lineno) + 1):
                    doc_lines.add(ln)
    code_lines: set[int] = set()
    with file_path.open("rb") as f:
        for tok in tokenize.tokenize(f.readline):
            if tok.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
            ):
                continue
            for ln in range(tok.start[0], tok.end[0] + 1):
                if ln not in doc_lines:
                    code_lines.add(ln)
    return len(code_lines)


def test_gate_h1_berry_qk_phase_unitarity_and_hermitian_inner_product() -> None:
    """Gate H1: Verify norm preservation, Step-0 identity, and analytical Hermitian inner product of `apply_berry_qk_phase`."""
    torch.manual_seed(101)
    bsz, seq_len, n_h, v_rank, d_h = 2, 8, 6, 2, 32
    head_dim = 2 * v_rank * d_h
    q_rot = torch.randn(bsz, seq_len, n_h, head_dim, dtype=torch.float32)
    k_rot = torch.randn(bsz, seq_len, n_h, head_dim, dtype=torch.float32)

    zero_phi = torch.zeros(n_h, v_rank, d_h, dtype=torch.float32)
    q_zero = apply_berry_qk_phase(q_rot, zero_phi)
    assert torch.equal(q_zero, q_rot), "At Phi=0, apply_berry_qk_phase must be bit-identical to q_rot"

    phi = torch.randn(n_h, v_rank, d_h, dtype=torch.float32)
    q_phi = apply_berry_qk_phase(q_rot, phi)

    norm_before = q_rot.norm(dim=-1)
    norm_after = q_phi.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5, rtol=1e-5)

    dot_sdpa = (q_phi * k_rot).sum(dim=-1)
    qc = q_rot.view(bsz, seq_len, n_h, v_rank, 2, d_h)
    kc = k_rot.view(bsz, seq_len, n_h, v_rank, 2, d_h)
    qr, qi = qc[..., 0, :], qc[..., 1, :]
    kr, ki = kc[..., 0, :], kc[..., 1, :]
    re_qk = qr * kr + qi * ki
    im_qk = qr * ki - qi * kr
    phi_b = phi.view(1, 1, n_h, v_rank, d_h)
    analytical = (torch.cos(phi_b) * re_qk + torch.sin(phi_b) * im_qk).sum(dim=(-2, -1))
    assert torch.allclose(dot_sdpa, analytical, atol=1e-5, rtol=1e-5)


def test_gate_h2_berry_phase_gradients_and_clamping() -> None:
    """Gate H2: Verify finite gradients to `berry_qk_phase` and clamping in `restrict_quantum_scalars_()`."""
    teacher = _make_tiny_teacher()
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    cfg = QuantumFrontEndGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
        attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
        replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
        interface_warmup_steps=10,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
        qk_phase_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
    )
    model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg)
    assert isinstance(model.quantum_block0, SVDStiefelWavepacketStatMechBlock)
    assert model.quantum_block0.berry_qk_phase.requires_grad
    assert any(p is model.quantum_block0.berry_qk_phase for p in model.quantum_scalar_parameters())

    with torch.no_grad():
        model.quantum_block0.berry_qk_phase.uniform_(-0.2, 0.2)
    _, loss = model(idx, targets=targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.quantum_block0.berry_qk_phase.grad is not None
    assert torch.isfinite(model.quantum_block0.berry_qk_phase.grad).all()
    assert float(model.quantum_block0.berry_qk_phase.grad.abs().max().item()) > 0.0

    with torch.no_grad():
        model.quantum_block0.berry_qk_phase.fill_(10.0)
    model.restrict_quantum_scalars_()
    assert float(model.quantum_block0.berry_qk_phase.max().item()) <= 3.1416 + 1e-5


def test_gate_h3_all_phase1_arm_configs_forward_backward_step() -> None:
    """Gate H3: Verify all 6 Phase-1 arm configs (`G71_1..G71_6`) run forward+backward+step cleanly on `(V, 12, 2, 32)`."""
    specs = [
        (
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumStateIOMode.WF_IN_WF_OUT,
            PurificationMode.QUADRATIC_MCWEENY,
            QuantumChannelMode.UNITARY_U16,
            QuantumQKPhaseMode.STANDARD_REAL_DOT,
        ),
        (
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            PurificationMode.CUBIC_FERMI_DIRAC,
            QuantumChannelMode.UNITARY_U16,
            QuantumQKPhaseMode.STANDARD_REAL_DOT,
        ),
        (
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            PurificationMode.QUADRATIC_MCWEENY,
            QuantumChannelMode.CPTP_KRAUS_RANK_K,
            QuantumQKPhaseMode.STANDARD_REAL_DOT,
        ),
        (
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            PurificationMode.QUADRATIC_MCWEENY,
            QuantumChannelMode.UNITARY_U16,
            QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        ),
        (
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumStateIOMode.WF_IN_WF_OUT,
            PurificationMode.CUBIC_FERMI_DIRAC,
            QuantumChannelMode.UNITARY_U16,
            QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        ),
        (
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            QuantumStateIOMode.WF_IN_WF_OUT,
            PurificationMode.CUBIC_FERMI_DIRAC,
            QuantumChannelMode.UNITARY_U16,
            QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        ),
    ]
    teacher = _make_tiny_teacher()
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    for rank_mode, io_mode, pur_mode, chan_mode, qk_mode in specs:
        cfg = QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
            state_io_mode=io_mode,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
            purification_mode=pur_mode,
            channel_mode=chan_mode,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
            recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
            interface_warmup_steps=10,
            stiefel_proj_mode=rank_mode,
            self_ortho_mode=QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO,
            qk_phase_mode=qk_mode,
        )
        model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg)
        opts = build_frozen_trunk_quantum_frontend_optimizers(model)
        opts.zero_grad()
        _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        opts.step(idx, lr_mult=1.0)
        tel = model.compute_quantum_graft_telemetry()
        assert tel["q_max_stiefel_err"] < 1e-4
        assert tel["q_wte_sphere_err"] < 1e-4


def test_gate_h4_ast_best_practices_compliance() -> None:
    """Gate H4: Verify SLOC <= 1,000 and <= 8 args per function across modified and new files."""
    files = [
        _REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_svd_stiefel_graft.py",
        _REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_frontend_graft.py",
        _REPO_ROOT / "tests" / "test_plan071_pkt32_synergy_and_berry_phase_qk.py",
        _REPO_ROOT / "examples" / "81_verify_plan071_pkt32_synergy_and_berry_phase_qk.py",
    ]
    for path in files:
        if not path.exists():
            continue
        sloc = _count_sloc(path)
        assert sloc <= _MAX_SLOC, f"{path.name} SLOC {sloc} > {_MAX_SLOC}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arg_names = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                non_self = [a for a in arg_names if a not in ("self", "cls")]
                assert len(non_self) <= _MAX_ARGS, f"{path.name}:{node.name} has {len(non_self)} args > {_MAX_ARGS}"


if __name__ == "__main__":
    test_gate_h1_berry_qk_phase_unitarity_and_hermitian_inner_product()
    print("[PASS] Gate H1: Berry QK Phase Unitarity & Hermitian Inner Product")
    test_gate_h2_berry_phase_gradients_and_clamping()
    print("[PASS] Gate H2: Berry Phase Gradients & Clamping")
    test_gate_h3_all_phase1_arm_configs_forward_backward_step()
    print("[PASS] Gate H3: All 6 Phase-1 Arm Configs Forward/Backward/Step")
    test_gate_h4_ast_best_practices_compliance()
    print("[PASS] Gate H4: AST Best-Practices Compliance")
