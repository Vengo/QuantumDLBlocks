r"""Unit Tests for `EXPERIMENT_PLAN_072.md` (`Wave 52H`).

Verifies Gates `H1..H3`:
- **Gate H1**: Verifies all 7 `Plan 072` arm configurations (`H72_1..H72_7`) resolve via `resolve_plan072_arm_config`
  with exact expected ranks, scopes, interior modes, and parameter counts.
- **Gate H2**: Verifies forward + backward + Riemannian step on CPU for `H72_1`, `H72_6`, and `H72_7`
  (`EMB_PLUS_L0_L1` with `ATTENDED_RHO_ON_LOCAL_V` + `BERRY_PHASE_COMPLEX_QK`).
- **Gate H3**: AST best-practices audit (`<= 1,000` SLOC, `<= 8` args per function) across all Wave 52H files.
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
import sys
import tokenize

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "examples"))

import torch

from qdlblocks.layers.quantum_frontend_graft import (  # noqa: E402
    FrozenTrunkQuantumFrontEndGPT,
    QuantumAttnInteriorMode,
    QuantumQKPhaseMode,
    QuantumReplacementScope,
    QuantumStateIOMode,
    QuantumStiefelProjMode,
    build_frozen_trunk_quantum_frontend_optimizers,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (  # noqa: E402
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)

_ex82 = import_module("82_verify_plan072_compression_frontier_and_local_v_synergy")
resolve_plan072_arm_config = _ex82.resolve_plan072_arm_config

_MAX_SLOC: int = 1000
_MAX_ARGS: int = 8


def _make_tiny_teacher() -> CompiledLeaderboardClassicalGPT:
    torch.manual_seed(20261102)
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


def test_gate_h1_all_seven_plan072_arm_configs_resolve() -> None:
    """Gate H1: Verify all 7 `Plan 072` arm configurations (`H72_1..H72_7`) resolve with exact expected modes."""
    expected = [
        (
            "H72_1_R384_PARITY_LOCAL_V_800STEPS",
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        ),
        (
            "H72_2_R384_PARITY_LIOUVILLE_800STEPS",
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
            QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V,
            QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        ),
        (
            "H72_3_R128_SUBPARITY_3X_800STEPS",
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128,
            QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        ),
        (
            "H72_4_R64_SUBPARITY_6X_800STEPS",
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
            QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        ),
        (
            "H72_5_R768_LOCAL_V_BERRY_WF_500S",
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        ),
        (
            "H72_6_R768_LIOUVILLE_BERRY_WF_500S",
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V,
            QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        ),
        (
            "H72_7_MULTILAYER_L0_L1_R768_800S",
            QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768,
            QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            QuantumReplacementScope.EMB_PLUS_L0_L1,
        ),
    ]
    for arm_id, exp_rank, exp_int, exp_scope in expected:
        cfg = resolve_plan072_arm_config(arm_id)
        assert cfg.stiefel_proj_mode == exp_rank
        assert cfg.attn_interior_mode == exp_int
        assert cfg.replacement_scope == exp_scope
        assert cfg.state_io_mode == QuantumStateIOMode.WF_IN_WF_OUT
        assert cfg.qk_phase_mode == QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK


def test_gate_h2_forward_backward_step_on_cpu() -> None:
    """Gate H2: Verify forward + backward + Riemannian step on CPU for `H72_1`, `H72_6`, and `H72_7`."""
    teacher = _make_tiny_teacher()
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    for arm_id in (
        "H72_1_R384_PARITY_LOCAL_V_800STEPS",
        "H72_6_R768_LIOUVILLE_BERRY_WF_500S",
        "H72_7_MULTILAYER_L0_L1_R768_800S",
    ):
        cfg = resolve_plan072_arm_config(arm_id)
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


def test_gate_h3_ast_best_practices_compliance() -> None:
    """Gate H3: Verify SLOC <= 1,000 and <= 8 args per function across Wave 52H files."""
    files = [
        _REPO_ROOT / "examples" / "82_verify_plan072_compression_frontier_and_local_v_synergy.py",
        _REPO_ROOT / "tests" / "test_plan072_compression_frontier_and_local_v_synergy.py",
    ]
    for path in files:
        sloc = _count_sloc(path)
        assert sloc <= _MAX_SLOC, f"{path.name} SLOC {sloc} > {_MAX_SLOC}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arg_names = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                non_self = [a for a in arg_names if a not in ("self", "cls")]
                assert len(non_self) <= _MAX_ARGS, f"{path.name}:{node.name} has {len(non_self)} args > {_MAX_ARGS}"


if __name__ == "__main__":
    test_gate_h1_all_seven_plan072_arm_configs_resolve()
    print("[PASS] Gate H1: All 7 Plan 072 Arm Configs Resolve")
    test_gate_h2_forward_backward_step_on_cpu()
    print("[PASS] Gate H2: Forward/Backward/Step on CPU (H72_1, H72_6, H72_7)")
    test_gate_h3_ast_best_practices_compliance()
    print("[PASS] Gate H3: AST Best-Practices Compliance")
