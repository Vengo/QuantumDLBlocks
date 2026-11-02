r"""Unit Tests for `EXPERIMENT_PLAN_075.md` (`Wave 54` — Fused Single-Pass Hermitian/Born Quantum Wavepacket Attention, $d_{\text{pkt}} = 64$ Granularity, & Riemannian Tangent SNR Trust-Region Gating with Momentum Warmup).

Verifies Gates `J1..J4`:
- **Gate J1**: Fused Single-Pass Kernel (`FUSED_SINGLE_PASS_RHO_KET`) numerical equivalence (`< 1e-5`) vs. `EXPANDED_SDPA_CHUNKS`
  across both $d_{\text{pkt}} = 32$ (`17` chunks) and $d_{\text{pkt}} = 64$ (`65` chunks) with `TEACHER_DOC_SWC_MASK`, plus
  `FUSED_HERMITIAN_BORN_RHO_KET` replacing external `apply_berry_qk_phase` with non-zero gradients on `berry_qk_phase` and `qk_born_beta`.
- **Gate J2**: Wavepacket granularity $d_{\text{pkt}} = 64$ (`QuantumWavefunctionShape.SHAPE_6x2x64`), `apply_causal_wavepacket_dispersion`
  on `(B, T, 12, 64)` (`wp_complex_modes = 6`), and $64 \times 64$ complex outer-product density matrix trace $\operatorname{Tr}(\rho_{t,h}) = 1.0 \pm 10^{-5}$.
- **Gate J3**: Riemannian Momentum Warmup schedules (`compute_riemannian_momentum_beta`) and 2-step end-to-end 12-layer training under
  `StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED` + `QuantumWavefunctionShape.SHAPE_6x2x64` with manifold errors `< 1e-4`.
- **Gate J4**: `docs/SCRIPT_BEST_PRACTICES.md` AST compliance (`<= 1,000` SLOC, `<= 8` args per function, zero semicolon statement packing).
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

from qdlblocks.kernels.fused_quantum_wavepacket_attn import (  # noqa: E402
    QuantumAttnKernelMode,
    fused_quantum_wavepacket_rho_ket_attention,
)
from qdlblocks.layers.quantum_frontend_graft import (  # noqa: E402
    QuantumQKPhaseMode,
    QuantumStiefelProjMode,
    QuantumWavefunctionShape,
    apply_berry_qk_phase,
    resolve_wavefunction_shape_spec,
)
from qdlblocks.layers.quantum_statmech_channels import (  # noqa: E402
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
)
from qdlblocks.layers.wavepacket_statmech_block import (  # noqa: E402
    apply_causal_wavepacket_dispersion,
    build_configurable_rho_and_ket_sdpa_chunks,
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
    build_doc_swc_sdpa_masks,
    build_quantum_full_backbone_optimizers,
    compute_quantum_backbone_lr_multiplier,
)

_ex85 = import_module("85_verify_plan075_fused_kernel_pkt64_and_momentum_warmup")
resolve_plan075_arm_config = _ex85.resolve_plan075_arm_config

_MAX_SLOC: int = 1000
_MAX_ARGS: int = 8


def _make_12layer_tiny_teacher() -> CompiledLeaderboardClassicalGPT:
    torch.manual_seed(2026075)
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


def test_gate_j1_fused_kernel_equivalence_and_native_hermitian_born_replacement() -> None:
    """Gate J1: Verify `FUSED_SINGLE_PASS_RHO_KET` equivalence (`< 1e-5`) and `FUSED_HERMITIAN_BORN_RHO_KET` replacing `apply_berry_qk_phase`."""
    torch.manual_seed(2026075)
    idx = torch.tensor(
        [
            [DEFAULT_BOS_TOKEN_ID, 10, 11, 12, DEFAULT_BOS_TOKEN_ID, 20, 21, 22],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    m_long, _, _ = build_doc_swc_sdpa_masks(idx, long_win=6, short_win=3, final_win=8)

    for wf_shape, d_h, v_rank in (
        (QuantumWavefunctionShape.SHAPE_12x2x32, 32, 2),
        (QuantumWavefunctionShape.SHAPE_6x2x64, 64, 1),
    ):
        teacher = _make_12layer_tiny_teacher()
        cfg_exp = QuantumFullBackboneGraftConfig(
            wf_shape=wf_shape,
            depth_scope=QuantumBackboneDepthScope.EMB_PLUS_L0_TO_L2_ATTN,
            doc_swc_mode=QuantumDocSWCMode.TEACHER_DOC_SWC_MASK,
            attn_kernel_mode=QuantumAttnKernelMode.EXPANDED_SDPA_CHUNKS,
            stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
            qk_phase_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        )
        m_exp = QuantumFullBackboneGraftGPT(teacher, graft_cfg=cfg_exp)
        blk_exp = m_exp.quantum_attn_blocks["0"]

        num_heads = 6
        sdpa_dim = 2 * v_rank * d_h
        q_base = torch.nn.functional.rms_norm(torch.randn(2, 8, num_heads, sdpa_dim, dtype=torch.float32), (sdpa_dim,)) * 0.25
        k_base = torch.nn.functional.rms_norm(torch.randn(2, 8, num_heads, sdpa_dim, dtype=torch.float32), (sdpa_dim,)) * 0.25
        v_wp_base = torch.randn(2, 8, blk_exp.num_pkts, d_h, dtype=torch.float32)
        v_mix_base = torch.randn(2, 8, blk_exp.num_pkts, d_h, dtype=torch.float32)

        from qdlblocks.kernels.fused_quantum_wavepacket_attn import run_coherent_ket_wavepacket_attention

        blk_exp.attn_kernel_mode = QuantumAttnKernelMode.EXPANDED_SDPA_CHUNKS.value
        q1 = q_base.clone().requires_grad_(True)
        k1 = k_base.clone().requires_grad_(True)
        vm1 = v_mix_base.clone().requires_grad_(True)
        out_exp = run_coherent_ket_wavepacket_attention(blk_exp, v_wp_base, vm1, q1, k1, attn_mask=m_long)
        out_exp.sum().backward()

        blk_exp.attn_kernel_mode = QuantumAttnKernelMode.FUSED_SINGLE_PASS_RHO_KET.value
        q2 = q_base.clone().requires_grad_(True)
        k2 = k_base.clone().requires_grad_(True)
        vm2 = v_mix_base.clone().requires_grad_(True)
        out_fus = run_coherent_ket_wavepacket_attention(blk_exp, v_wp_base, vm2, q2, k2, attn_mask=m_long)
        out_fus.sum().backward()

        assert (out_exp - out_fus).abs().max().item() < 1e-5, f"Fwd diff on d_h={d_h}"
        assert q1.grad is not None and q2.grad is not None
        assert (q1.grad - q2.grad).abs().max().item() < 1e-5, f"dQ diff on d_h={d_h}"
        assert k1.grad is not None and k2.grad is not None
        assert (k1.grad - k2.grad).abs().max().item() < 1e-5, f"dK diff on d_h={d_h}"
        assert vm1.grad is not None and vm2.grad is not None
        assert (vm1.grad - vm2.grad).abs().max().item() < 1e-5, f"dV diff on d_h={d_h}"

        # Verify FUSED_HERMITIAN_BORN_RHO_KET with head-uniform Berry phase & born_beta=0 matches external apply_berry_qk_phase
        head_phi = torch.linspace(-0.25, 0.35, num_heads, dtype=torch.float32).view(num_heads, 1, 1).expand(num_heads, v_rank, d_h).clone()
        with torch.no_grad():
            blk_exp.berry_qk_phase.copy_(head_phi)
            blk_exp.qk_born_beta.zero_()
        q_ext = apply_berry_qk_phase(q_base.clone(), blk_exp.berry_qk_phase)
        blk_exp.attn_kernel_mode = QuantumAttnKernelMode.FUSED_SINGLE_PASS_RHO_KET.value
        out_berry_ext = run_coherent_ket_wavepacket_attention(blk_exp, v_wp_base, v_mix_base, q_ext, k_base, attn_mask=m_long)

        blk_exp.attn_kernel_mode = QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET.value
        out_born0 = run_coherent_ket_wavepacket_attention(blk_exp, v_wp_base, v_mix_base, q_base, k_base, attn_mask=m_long)
        assert (out_berry_ext - out_born0).abs().max().item() < 1e-5, (
            f"Native Hermitian/Born with beta=0 did not match external apply_berry_qk_phase on d_h={d_h}"
        )

        # Verify non-zero gradients on both berry_qk_phase and qk_born_beta when born_beta != 0
        with torch.no_grad():
            blk_exp.qk_born_beta.fill_(0.08)
        blk_exp.qk_born_beta.requires_grad_(True)
        blk_exp.berry_qk_phase.requires_grad_(True)
        blk_exp.berry_qk_phase.grad = None
        blk_exp.qk_born_beta.grad = None
        q3 = q_base.clone().requires_grad_(True)
        k3 = k_base.clone().requires_grad_(True)
        vm3 = v_mix_base.clone().requires_grad_(True)
        out_hb = run_coherent_ket_wavepacket_attention(blk_exp, v_wp_base, vm3, q3, k3, attn_mask=m_long)
        (out_hb.square().sum()).backward()
        assert blk_exp.berry_qk_phase.grad is not None and blk_exp.berry_qk_phase.grad.abs().max().item() > 0.0
        assert blk_exp.qk_born_beta.grad is not None and blk_exp.qk_born_beta.grad.abs().max().item() > 0.0

        if torch.cuda.is_available():
            cpu_berry_grad = blk_exp.berry_qk_phase.grad.detach().cpu().clone()
            cpu_born_grad = blk_exp.qk_born_beta.grad.detach().cpu().clone()
            dev_cuda = torch.device("cuda:0")
            m_cuda = build_doc_swc_sdpa_masks(idx.to(dev_cuda), long_win=6, short_win=3, final_win=8)[0]
            blk_cuda = m_exp.quantum_attn_blocks["0"].to(dev_cuda)
            blk_cuda.attn_kernel_mode = QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET.value
            blk_cuda.berry_qk_phase.grad = None
            blk_cuda.qk_born_beta.grad = None
            qc = q_base.to(dev_cuda).requires_grad_(True)
            kc = k_base.to(dev_cuda).requires_grad_(True)
            vmc = v_mix_base.to(dev_cuda).requires_grad_(True)
            out_cuda = run_coherent_ket_wavepacket_attention(
                blk_cuda, v_wp_base.to(dev_cuda), vmc, qc, kc, attn_mask=m_cuda
            )
            (out_cuda.square().sum()).backward()
            assert (out_cuda.cpu() - out_hb.detach()).abs().max().item() < 1e-4, f"CUDA Triton fwd diff on d_h={d_h}"
            assert qc.grad is not None and q3.grad is not None
            assert (qc.grad.cpu() - q3.grad).abs().max().item() < 1e-4, f"CUDA Triton dQ diff on d_h={d_h}"
            assert kc.grad is not None and k3.grad is not None
            assert (kc.grad.cpu() - k3.grad).abs().max().item() < 1e-4, f"CUDA Triton dK diff on d_h={d_h}"
            assert blk_cuda.berry_qk_phase.grad is not None
            assert (blk_cuda.berry_qk_phase.grad.cpu() - cpu_berry_grad).abs().max().item() < 1e-4
            assert blk_cuda.qk_born_beta.grad is not None
            assert (blk_cuda.qk_born_beta.grad.cpu() - cpu_born_grad).abs().max().item() < 1e-4


def test_gate_j2_shape_6x2x64_geometry_and_dispersion() -> None:
    """Gate J2: Verify `SHAPE_6x2x64` (`d_pkt = 64`, `pkt_count = 12`, `num_heads = 6`) geometry and multi-lag dispersion."""
    spec = resolve_wavefunction_shape_spec(QuantumWavefunctionShape.SHAPE_6x2x64)
    assert spec == (12, 64, 768, 12, 64, 6), f"Unexpected SHAPE_6x2x64 spec: {spec}"

    raw_wp = torch.randn(2, 8, 12, 64, dtype=torch.float32)
    wp_re = torch.tensor([0.15, 0.10, 0.08, 0.05, 0.15, 0.10], dtype=torch.float32, requires_grad=True)
    wp_im = torch.tensor([0.00, 0.10, -0.08, 0.05, 0.00, 0.10], dtype=torch.float32, requires_grad=True)
    disp = apply_causal_wavepacket_dispersion(raw_wp, wp_re, wp_im)
    assert disp.shape == (2, 8, 12, 64)
    disp.square().sum().backward()
    assert wp_re.grad is not None and wp_re.grad.abs().max().item() > 0.0

    # Verify 64x64 outer-product density matrix trace = 1.0 +/- 1e-5
    sdpa_v = build_configurable_rho_and_ket_sdpa_chunks(raw_wp, raw_wp, num_heads=6, v_rank=1)
    assert sdpa_v.shape == (2, 6 * 65, 8, 128)
    v_5d = sdpa_v.view(2, 6, 65, 8, 128)
    rho_re = v_5d[:, :, :64].permute(0, 1, 3, 2, 4).view(2, 6, 8, 64, 2, 64)[..., 0, :]
    tr = torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1)
    assert (tr - 1.0).abs().max().item() < 1e-5, f"Expected unit trace 1.0 on 64x64 rho, got {tr}"


def test_gate_j3_riemannian_momentum_warmup_and_tangent_snr_12layer_step() -> None:
    """Gate J3: Verify `compute_riemannian_momentum_beta` schedules and 2-step 12-layer training under `RIEMANNIAN_TANGENT_SNR_GATED` + `SHAPE_6x2x64`."""
    # 1. Verify all 4 momentum schedules
    assert abs(compute_riemannian_momentum_beta(1, 800, QuantumMomentumScheduleMode.FIXED_HIGH_MOMENTUM_090) - 0.90) < 1e-6
    assert abs(compute_riemannian_momentum_beta(1, 800, QuantumMomentumScheduleMode.ZERO_EARLY_TO_WARMUP_090) - 0.0) < 1e-6
    assert abs(compute_riemannian_momentum_beta(80, 800, QuantumMomentumScheduleMode.ZERO_EARLY_TO_WARMUP_090) - 0.0) < 1e-6
    assert abs(compute_riemannian_momentum_beta(320, 800, QuantumMomentumScheduleMode.ZERO_EARLY_TO_WARMUP_090) - 0.90) < 1e-6
    assert abs(compute_riemannian_momentum_beta(1, 800, QuantumMomentumScheduleMode.MODERATE_WARMUP_050_TO_092) - 0.50) < 1e-6
    assert abs(compute_riemannian_momentum_beta(320, 800, QuantumMomentumScheduleMode.MODERATE_WARMUP_050_TO_092) - 0.92) < 1e-6
    assert abs(compute_riemannian_momentum_beta(800, 800, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070) - 0.70) < 1e-6

    for i in range(1, 9):
        matched = [k for k in _ex85.ARM_SPECS if k.startswith(f"K75_{i}_")]
        assert len(matched) == 1
        cfg = resolve_plan075_arm_config(matched[0])
        assert cfg.doc_swc_mode == QuantumDocSWCMode.TEACHER_DOC_SWC_MASK

    # 2. End-to-end 12-layer forward/backward/step with SHAPE_6x2x64 + FUSED_HERMITIAN_BORN_RHO_KET + RIEMANNIAN_TANGENT_SNR_GATED
    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    teacher = _make_12layer_tiny_teacher()
    cfg = QuantumFullBackboneGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_6x2x64,
        depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
        doc_swc_mode=QuantumDocSWCMode.TEACHER_DOC_SWC_MASK,
        depth_lr_mode=QuantumDepthLRMode.WARMUP_COSINE_UNSCALED,
        attn_kernel_mode=QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET,
        stiefel_momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        momentum_schedule_mode=QuantumMomentumScheduleMode.MODERATE_WARMUP_050_TO_092,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
        qk_phase_mode=QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK,
        interface_warmup_steps=10,
    )
    model = QuantumFullBackboneGraftGPT(teacher, graft_cfg=cfg)
    opts = build_quantum_full_backbone_optimizers(model)
    for step in (1, 2):
        lr_mult = compute_quantum_backbone_lr_multiplier(step, 10, cfg.depth_lr_mode)
        mom_beta = compute_riemannian_momentum_beta(step, 10, cfg.momentum_schedule_mode)
        opts.zero_grad()
        _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        for l_key, svd_blk in model.quantum_attn_blocks.items():
            assert svd_blk.q_proj.u_stiefel.grad is not None and svd_blk.q_proj.u_stiefel.grad.abs().max().item() > 0.0, (
                f"Missing grad on layer {l_key}"
            )
            assert svd_blk.qk_born_beta.grad is not None and svd_blk.qk_born_beta.grad.abs().max().item() > 0.0, (
                f"Missing qk_born_beta grad on layer {l_key}"
            )
            assert svd_blk.berry_qk_phase.grad is not None and svd_blk.berry_qk_phase.grad.abs().max().item() > 0.0, (
                f"Missing berry_qk_phase grad on layer {l_key}"
            )
        opts.step(idx, lr_mult=lr_mult, momentum_beta=mom_beta)
    tel = model.compute_quantum_graft_telemetry()
    assert tel["q_max_stiefel_err"] < 1e-4, f"Stiefel error {tel['q_max_stiefel_err']} >= 1e-4"
    assert tel["q_max_unitary_err"] < 1e-4, f"Unitary error {tel['q_max_unitary_err']} >= 1e-4"
    assert tel["q_wte_sphere_err"] < 1e-4, f"Sphere error {tel['q_wte_sphere_err']} >= 1e-4"


def test_gate_j4_ast_best_practices_compliance() -> None:
    """Gate J4: Verify SLOC <= 1,000, <= 8 args per function, and zero semicolon statement packing."""
    files = [
        _REPO_ROOT / "src" / "qdlblocks" / "kernels" / "fused_quantum_wavepacket_attn.py",
        _REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_statmech_channels.py",
        _REPO_ROOT / "src" / "qdlblocks" / "layers" / "quantum_svd_stiefel_graft.py",
        _REPO_ROOT / "src" / "qdlblocks" / "models" / "quantum_full_backbone_graft_gpt.py",
        _REPO_ROOT / "examples" / "85_verify_plan075_fused_kernel_pkt64_and_momentum_warmup.py",
        _REPO_ROOT / "tests" / "test_plan075_fused_kernel_pkt64_and_momentum_warmup.py",
    ]
    for path in files:
        sloc = _count_sloc(path)
        assert sloc <= _MAX_SLOC, f"{path.name} SLOC {sloc} > {_MAX_SLOC}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_triton_jit = any("triton" in ast.unparse(d) and "jit" in ast.unparse(d) for d in node.decorator_list)
                if is_triton_jit:
                    continue
                arg_names = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                non_self = [a for a in arg_names if a not in ("self", "cls")]
                assert len(non_self) <= _MAX_ARGS, f"{path.name}:{node.name} has {len(non_self)} args > {_MAX_ARGS}"
        with path.open("rb") as f:
            for tok in tokenize.tokenize(f.readline):
                assert not (tok.type == tokenize.OP and tok.string == ";"), f"Semicolon found in {path.name}:{tok.start[0]}"


if __name__ == "__main__":
    test_gate_j1_fused_kernel_equivalence_and_native_hermitian_born_replacement()
    print("[PASS] Gate J1: Fused Single-Pass Kernel Equivalence & Native Hermitian/Born Berry Replacement")
    test_gate_j2_shape_6x2x64_geometry_and_dispersion()
    print("[PASS] Gate J2: SHAPE_6x2x64 (d_pkt=64, 64x64 Density Matrix) Geometry & Dispersion")
    test_gate_j3_riemannian_momentum_warmup_and_tangent_snr_12layer_step()
    print("[PASS] Gate J3: Riemannian Momentum Warmup Schedules & Tangent SNR 12-Layer Training Step")
    test_gate_j4_ast_best_practices_compliance()
    print("[PASS] Gate J4: AST Best-Practices Compliance")
