r"""Unit Tests for `EXPERIMENT_PLAN_073.md` (`Wave 53A` — Full-Backbone Quantum Wavepacket Replacement).

Verifies Gates `H1..H4`:
- **Gate H1**: Multi-Layer & Full-Block Parameter Isolation, `use_partial_key_offset` per-layer calibration,
  polar-factorized unit-RMS sphere `q_value_embeds` Step-0 reconstruction, and zero classical trunk execution
  during `forward_features(idx)` under `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`.
- **Gate H2**: `SVDStiefelMachZehnderInterferometer` Step-0 calibration fidelity (`cos > 0.999`) and strictly
  non-zero quadrature gradients (`mz_theta`, `mz_gamma`, `up_proj`, `down_proj`).
- **Gate H3**: End-to-end 12-layer forward, backward, Riemannian Stiefel/Unitary/Sphere step, and telemetry
  (`q_max_stiefel_err < 1e-4`, `q_wte_sphere_err < 1e-4`) across `EMB_PLUS_ALL_12_ATTN` and `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`.
- **Gate H4**: `docs/SCRIPT_BEST_PRACTICES.md` AST compliance (`<= 1,000` SLOC, `<= 8` args per function, zero semicolon packing).
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
import torch.nn.functional as F

from qdlblocks.layers.quantum_frontend_graft import (  # noqa: E402
    QuantumStiefelProjMode,
    QuantumWavefunctionShape,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (  # noqa: E402
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)
from qdlblocks.models.quantum_full_backbone_graft_gpt import (  # noqa: E402
    QuantumBackboneDepthScope,
    QuantumFullBackboneGraftConfig,
    QuantumFullBackboneGraftGPT,
    QuantumKetAmplitudeMode,
    QuantumMLPChannelMode,
    SVDStiefelMachZehnderInterferometer,
    build_quantum_full_backbone_optimizers,
)

_ex83 = import_module("83_verify_plan073_full_backbone_quantum_replacement")
resolve_plan073_arm_config = _ex83.resolve_plan073_arm_config

_MAX_SLOC: int = 1000
_MAX_ARGS: int = 8


def _make_12layer_tiny_teacher() -> CompiledLeaderboardClassicalGPT:
    torch.manual_seed(2026073)
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


def test_gate_h1_multilayer_and_full_block_isolation() -> None:
    """Gate H1: Verify all 8 arm configs, per-layer `use_partial_key_offset`, polar `q_value_embeds`, and standalone execution."""
    expected_arms = [
        ("I73_1_DEPTH3_L0_TO_L2_ATTN_R384", QuantumBackboneDepthScope.EMB_PLUS_L0_TO_L2_ATTN, QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384),
        ("I73_2_DEPTH6_ENCODER_L0_TO_L5_ATTN_R384", QuantumBackboneDepthScope.EMB_PLUS_L0_TO_L5_ATTN, QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384),
        ("I73_3_ALL12_ATTN_R768_FULL_RANK", QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN, QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768),
        ("I73_4_ALL12_ATTN_R384_1TO1_PARITY", QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN, QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384),
        ("I73_5_ALL12_ATTN_R192_2X_COMPRESSED", QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN, QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R192),
        ("I73_6_ALL12_ATTN_R128_3X_COMPRESSED", QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN, QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128),
        ("I73_7_ALL12_FULL_BLOCKS_MZ_R768", QuantumBackboneDepthScope.EMB_PLUS_ALL_12_FULL_BLOCKS_MZ, QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R768),
        ("I73_8_ALL12_FULL_BLOCKS_MZ_R384_COMPRESSED", QuantumBackboneDepthScope.EMB_PLUS_ALL_12_FULL_BLOCKS_MZ, QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384),
    ]
    for arm_id, exp_scope, exp_rank in expected_arms:
        cfg = resolve_plan073_arm_config(arm_id)
        assert cfg.depth_scope == exp_scope
        assert cfg.stiefel_proj_mode == exp_rank
        assert cfg.ket_amplitude_mode == QuantumKetAmplitudeMode.ENTROPY_PRESERVING_COHERENT_KET

    teacher = _make_12layer_tiny_teacher()
    cfg_full = QuantumFullBackboneGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_FULL_BLOCKS_MZ,
        mlp_channel_mode=QuantumMLPChannelMode.SVD_STIEFEL_MACH_ZEHNDER,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128,
        mlp_stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R128,
    )
    model = QuantumFullBackboneGraftGPT(teacher, graft_cfg=cfg_full)
    assert model.replaced_attn_layers == (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11)
    assert model.replaced_mlp_layers == tuple(range(12))
    assert all(not p.requires_grad for p in model.classical.parameters())

    for l_idx in model.replaced_attn_layers:
        svd_blk = model.quantum_attn_blocks[str(l_idx)]
        expected_shift = l_idx in (0, 4, 11)
        assert svd_blk.use_partial_key_offset == expected_shift, f"Layer {l_idx} use_partial_key_offset mismatch"

    idx = torch.randint(0, 128, (2, 8))
    assert model.q_value_embeds is not None and teacher.value_embeds is not None
    for b_idx in range(3):
        ve_q = model._lookup_value_embedding_768(b_idx, idx, torch.float32)
        ve_c = teacher.value_embeds[b_idx](idx).float()
        assert torch.allclose(ve_q, ve_c, atol=1e-5), f"Bank {b_idx} Step-0 polar reconstruction mismatch"
        rms_err = (model.q_value_embeds[b_idx].float().pow(2).mean(dim=-1).sqrt() - 1.0).abs().max().item()
        assert rms_err < 1e-5

    def _poison_forward(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Classical block should not be called in EMB_PLUS_ALL_12_FULL_BLOCKS_MZ!")

    for blk in model.classical.blocks:
        blk.forward = _poison_forward  # type: ignore[method-assign]
    out_feat = model.forward_features(idx)
    assert out_feat.shape == (2, 8, 768) and torch.isfinite(out_feat).all()


def test_gate_h2_mach_zehnder_calibration_and_quadrature_gradients() -> None:
    """Gate H2: Verify `SVDStiefelMachZehnderInterferometer` Step-0 cosine > 0.999 and non-zero quadrature gradients."""
    teacher = _make_12layer_tiny_teacher()
    t_blk0 = teacher.blocks[0]
    mz = SVDStiefelMachZehnderInterferometer(
        d_model=768,
        d_hidden=384,
        proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384,
        eps=teacher.cfg.eps,
    )
    mz.calibrate_from_teacher_block_(t_blk0)
    x_mid = torch.randn(2, 8, 768, requires_grad=True)

    with torch.no_grad():
        h_mlp = F.rms_norm(x_mid, (768,), eps=teacher.cfg.eps)
        y_teacher = x_mid + t_blk0.fc2(F.relu(t_blk0.fc1(h_mlp)).square())
    y_mz = mz(x_mid)
    cos_sim = F.cosine_similarity(y_mz.reshape(-1, 768), y_teacher.reshape(-1, 768), dim=-1).mean().item()
    assert cos_sim > 0.999, f"Expected Step-0 MZ calibration cosine > 0.999, got {cos_sim:.6f}"

    loss = y_mz.square().mean()
    loss.backward()
    for name, p in (
        ("up_u", mz.up_proj.u_stiefel),
        ("up_v", mz.up_proj.v_stiefel),
        ("down_u", mz.down_proj.u_stiefel),
        ("down_v", mz.down_proj.v_stiefel),
        ("mz_theta", mz.mz_theta),
        ("mz_gamma", mz.mz_gamma),
    ):
        assert p.grad is not None and p.grad.abs().max().item() > 0.0, f"Zero gradient on {name}"


def test_gate_h3_end_to_end_12layer_forward_backward_and_riemannian_step() -> None:
    """Gate H3: Run 2 steps on CPU for `EMB_PLUS_ALL_12_ATTN` and `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`."""
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
            stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
            mlp_stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
            interface_warmup_steps=10,
        )
        model = QuantumFullBackboneGraftGPT(teacher, graft_cfg=cfg)
        opts = build_quantum_full_backbone_optimizers(model)
        for _ in range(2):
            opts.zero_grad()
            _, loss = model(idx, targets=targets)
            assert loss is not None and torch.isfinite(loss)
            loss.backward()
            for l_key, svd_blk in model.quantum_attn_blocks.items():
                assert svd_blk.q_proj.u_stiefel.grad is not None and svd_blk.q_proj.u_stiefel.grad.abs().max().item() > 0.0, (
                    f"Missing grad on layer {l_key}"
                )
            opts.step(idx, lr_mult=1.0)
        tel = model.compute_quantum_graft_telemetry()
        assert tel["q_max_stiefel_err"] < 1e-4, f"Stiefel error {tel['q_max_stiefel_err']} >= 1e-4"
        assert tel["q_wte_sphere_err"] < 1e-4, f"Sphere error {tel['q_wte_sphere_err']} >= 1e-4"


def test_gate_h4_ast_best_practices_compliance() -> None:
    """Gate H4: Verify SLOC <= 1,000, <= 8 args per function, and zero semicolon statement packing."""
    files = [
        _REPO_ROOT / "src" / "qdlblocks" / "models" / "quantum_full_backbone_graft_gpt.py",
        _REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_svd_stiefel_graft.py",
        _REPO_ROOT / "examples" / "83_verify_plan073_full_backbone_quantum_replacement.py",
        _REPO_ROOT / "tests" / "test_plan073_full_backbone_quantum_replacement.py",
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
    test_gate_h1_multilayer_and_full_block_isolation()
    print("[PASS] Gate H1: Multi-Layer & Full-Block Parameter Isolation and Standalone Execution")
    test_gate_h2_mach_zehnder_calibration_and_quadrature_gradients()
    print("[PASS] Gate H2: Mach-Zehnder Interferometer Calibration & Non-Zero Quadrature Gradients")
    test_gate_h3_end_to_end_12layer_forward_backward_and_riemannian_step()
    print("[PASS] Gate H3: End-to-End 12-Layer Forward, Backward, Riemannian Step & Telemetry")
    test_gate_h4_ast_best_practices_compliance()
    print("[PASS] Gate H4: AST Best-Practices Compliance")
