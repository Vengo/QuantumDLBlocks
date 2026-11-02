r"""Unit Tests for `EXPERIMENT_PLAN_074.md` (`Wave 53B` — Document-Boundary Sliding-Window Quantum Wavepacket Attention & Depth-Scaled Full-Backbone Optimization).

Verifies Gates `I1..I4`:
- **Gate I1**: Document-Boundary (`<BOS> = 50256`) + Long/Short/Final Sliding-Window Causal Mask (`build_doc_swc_sdpa_masks`)
  exactness and per-layer window assignment (`is_long_window_layer`, `layer_idx`).
- **Gate I2**: Depth-Scaled $1/\sqrt{\max(1, N_{\text{sub}}/2)}$ Geodesic Trust-Region LR Scaling (`compute_depth_lr_scale`)
  and `10%` linear warmup + `90%` cosine decay (`compute_quantum_backbone_lr_multiplier`).
- **Gate I3**: End-to-end 12-layer forward, backward, and Riemannian step with `TEACHER_DOC_SWC_MASK` and `DEPTH_SQRT_WARMUP_COSINE`
  for `EMB_PLUS_ALL_12_ATTN` and `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`.
- **Gate I4**: `docs/SCRIPT_BEST_PRACTICES.md` AST compliance (`<= 1,000` SLOC, `<= 8` args per function, zero semicolon packing).
"""

from __future__ import annotations

import ast
from importlib import import_module
import math
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
    QuantumStiefelProjMode,
    QuantumWavefunctionShape,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (  # noqa: E402
    DEFAULT_BOS_TOKEN_ID,
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)
from qdlblocks.models.quantum_full_backbone_graft_gpt import (  # noqa: E402
    QuantumBackboneDepthScope,
    QuantumDepthLRMode,
    QuantumDocSWCMode,
    QuantumFullBackboneGraftConfig,
    QuantumFullBackboneGraftGPT,
    QuantumMLPChannelMode,
    build_doc_swc_sdpa_masks,
    build_quantum_full_backbone_optimizers,
    compute_depth_lr_scale,
    compute_quantum_backbone_lr_multiplier,
)

_ex84 = import_module("84_verify_plan074_doc_swc_and_depth_scaled_full_backbone")
resolve_plan074_arm_config = _ex84.resolve_plan074_arm_config

_MAX_SLOC: int = 1000
_MAX_ARGS: int = 8


def _make_12layer_tiny_teacher() -> CompiledLeaderboardClassicalGPT:
    torch.manual_seed(2026074)
    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=128,
        embed_dim=768,
        num_heads=6,
        num_layers=12,
        hidden_dim=384,
        max_seq_len=32,
        num_value_embeddings=3,
        skip_attn_layers=(7,),
        long_window_layers=(0, 4, 11),
        bfloat16_weights=False,
        use_flex_doc_swc=False,
    )
    teacher = CompiledLeaderboardClassicalGPT(cfg)
    with torch.no_grad():
        teacher.wte.weight.normal_(0.0, 0.02)
        teacher.lm_head.weight.normal_(0.0, 0.02)
        assert teacher.value_embeds is not None
        for ve in teacher.value_embeds:
            ve.weight.normal_(0.0, 0.02)
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


def test_gate_i1_doc_swc_mask_and_window_exactness() -> None:
    """Gate I1: Verify `build_doc_swc_sdpa_masks` document isolation (`<BOS>=50256`) and window horizons."""
    idx = torch.tensor(
        [
            [DEFAULT_BOS_TOKEN_ID, 10, 11, 12, DEFAULT_BOS_TOKEN_ID, 20, 21, 22],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    m_long, m_short, m_final = build_doc_swc_sdpa_masks(idx, long_win=6, short_win=3, final_win=8)
    assert m_long.shape == (2, 1, 8, 8) and m_short.shape == (2, 1, 8, 8) and m_final.shape == (2, 1, 8, 8)
    # Batch 0 has a <BOS> at position 4: positions 4..7 must NOT attend to positions 0..3
    assert not m_long[0, 0, 4:, :4].any(), "Document boundary isolation violated in m_long"
    assert not m_short[0, 0, 4:, :4].any(), "Document boundary isolation violated in m_short"
    assert not m_final[0, 0, 4:, :4].any(), "Document boundary isolation violated in m_final"
    # Causality: upper triangle q < k must be False
    for q in range(8):
        for k in range(q + 1, 8):
            assert not bool(m_long[1, 0, q, k].item())
            assert not bool(m_short[1, 0, q, k].item())
    # Short window W=3 on single-document batch 1: q=5 can attend to k=3,4,5 but NOT k=0,1,2
    assert not m_short[1, 0, 5, :3].any()
    assert m_short[1, 0, 5, 3:6].all()
    # Long window W=6 on batch 1: q=6 can attend to k=1..6 but NOT k=0
    assert not bool(m_long[1, 0, 6, 0].item())
    assert m_long[1, 0, 6, 1:7].all()


def test_gate_i2_depth_scaled_lr_and_warmup_cosine_schedule() -> None:
    """Gate I2: Verify `compute_depth_lr_scale`, `compute_quantum_backbone_lr_multiplier`, and all 8 `J74_1..J74_8` arm configs."""
    for n_sub in (3, 6, 11, 23):
        expected = 1.0 / math.sqrt(max(1.0, float(n_sub) / 2.0))
        assert abs(compute_depth_lr_scale(n_sub) - expected) < 1e-9

    m_start = compute_quantum_backbone_lr_multiplier(1, 800, QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE)
    m_peak = compute_quantum_backbone_lr_multiplier(80, 800, QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE)
    m_end = compute_quantum_backbone_lr_multiplier(800, 800, QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE)
    assert abs(m_start - 0.10) < 1e-6, f"Expected warmup start 0.10, got {m_start}"
    assert abs(m_peak - 1.00) < 1e-6, f"Expected warmup peak 1.00, got {m_peak}"
    assert abs(m_end - 0.10) < 1e-6, f"Expected cosine floor 0.10, got {m_end}"

    for i in range(1, 9):
        matched = [k for k in _ex84.ARM_SPECS if k.startswith(f"J74_{i}_")]
        assert len(matched) == 1
        cfg = resolve_plan074_arm_config(matched[0])
        assert cfg.doc_swc_mode == QuantumDocSWCMode.TEACHER_DOC_SWC_MASK
        assert cfg.depth_lr_mode == QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE


def test_gate_i3_end_to_end_12layer_doc_swc_forward_backward_step() -> None:
    """Gate I3: Verify 2 steps on CPU with `TEACHER_DOC_SWC_MASK` and `DEPTH_SQRT_WARMUP_COSINE`."""
    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    for scope, mlp_mode in (
        (QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN, QuantumMLPChannelMode.FROZEN_CLASSICAL_MLP),
        (QuantumBackboneDepthScope.EMB_PLUS_ALL_12_FULL_BLOCKS_MZ, QuantumMLPChannelMode.SVD_STIEFEL_MACH_ZEHNDER),
    ):
        teacher = _make_12layer_tiny_teacher()
        cfg = QuantumFullBackboneGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
            depth_scope=scope,
            mlp_channel_mode=mlp_mode,
            doc_swc_mode=QuantumDocSWCMode.TEACHER_DOC_SWC_MASK,
            depth_lr_mode=QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE,
            stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
            mlp_stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
            interface_warmup_steps=10,
        )
        model = QuantumFullBackboneGraftGPT(teacher, graft_cfg=cfg)
        opts = build_quantum_full_backbone_optimizers(model)
        for step in (1, 2):
            lr_mult = compute_quantum_backbone_lr_multiplier(step, 10, cfg.depth_lr_mode)
            opts.zero_grad()
            _, loss = model(idx, targets=targets)
            assert loss is not None and torch.isfinite(loss)
            loss.backward()
            for l_key, svd_blk in model.quantum_attn_blocks.items():
                assert svd_blk.q_proj.u_stiefel.grad is not None and svd_blk.q_proj.u_stiefel.grad.abs().max().item() > 0.0, (
                    f"Missing grad on layer {l_key}"
                )
            opts.step(idx, lr_mult=lr_mult)
        tel = model.compute_quantum_graft_telemetry()
        assert tel["q_max_stiefel_err"] < 1e-4, f"Stiefel error {tel['q_max_stiefel_err']} >= 1e-4"
        assert tel["q_wte_sphere_err"] < 1e-4, f"Sphere error {tel['q_wte_sphere_err']} >= 1e-4"


def test_gate_i4_ast_best_practices_compliance() -> None:
    """Gate I4: Verify SLOC <= 1,000, <= 8 args per function, and zero semicolon statement packing."""
    files = [
        _REPO_ROOT / "src" / "qdlblocks" / "models" / "quantum_full_backbone_graft_gpt.py",
        _REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_svd_stiefel_graft.py",
        _REPO_ROOT / "examples" / "84_verify_plan074_doc_swc_and_depth_scaled_full_backbone.py",
        _REPO_ROOT / "tests" / "test_plan074_doc_swc_and_depth_scaled_full_backbone.py",
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
        with path.open("rb") as f:
            for tok in tokenize.tokenize(f.readline):
                assert not (tok.type == tokenize.OP and tok.string == ";"), f"Semicolon found in {path.name}:{tok.start[0]}"


if __name__ == "__main__":
    test_gate_i1_doc_swc_mask_and_window_exactness()
    print("[PASS] Gate I1: Document-Boundary + Long/Short/Final Sliding-Window Mask Exactness")
    test_gate_i2_depth_scaled_lr_and_warmup_cosine_schedule()
    print("[PASS] Gate I2: Depth-Scaled 1/sqrt(N_sub/2) Geodesic LR & Warmup-Cosine Schedule")
    test_gate_i3_end_to_end_12layer_doc_swc_forward_backward_step()
    print("[PASS] Gate I3: End-to-End 12-Layer Doc-SWC Forward, Backward & Riemannian Step")
    test_gate_i4_ast_best_practices_compliance()
    print("[PASS] Gate I4: AST Best-Practices Compliance")
