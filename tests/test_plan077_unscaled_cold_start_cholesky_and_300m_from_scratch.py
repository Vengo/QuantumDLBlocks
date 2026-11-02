#!/usr/bin/env python3
r"""Deterministic Verification Test Suite for `EXPERIMENT_PLAN_077.md` (`Gates M1..M4`)."""

from __future__ import annotations

from importlib import import_module
import math
from pathlib import Path
import sys

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "examples"))

import torch

from qdlblocks.kernels.fused_quantum_wavepacket_attn import (  # noqa: E402
    DEFAULT_LOW_RANK_CHOLESKY_K,
    QuantumAttnKernelMode,
    associative_low_rank_cholesky_gram_modulation,
    associative_mcweeny_unitary_ket_modulation,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (  # noqa: E402
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)
from qdlblocks.models.quantum_full_backbone_graft_gpt import (  # noqa: E402
    DEFAULT_FROM_SCRATCH_SCALAR_LR,
    DEFAULT_FROM_SCRATCH_STIEFEL_LR,
    DEFAULT_FROM_SCRATCH_UNITARY_LR,
    DEFAULT_FROM_SCRATCH_WTE_LR,
    QuantumFullBackboneGraftGPT,
    build_quantum_full_backbone_optimizers,
    compute_depth_lr_scale,
)

_ex87 = import_module("87_verify_plan077_unscaled_cold_start_cholesky_and_300m_from_scratch")
resolve_plan077_arm_config = _ex87.resolve_plan077_arm_config


def _make_tiny_teacher() -> CompiledLeaderboardClassicalGPT:
    """Create a fast CPU `CompiledLeaderboardClassicalGPT` for deterministic gate tests."""
    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=256,
        embed_dim=768,
        num_heads=6,
        num_layers=4,
        hidden_dim=256,
        max_seq_len=32,
        num_value_embeddings=1,
        skip_attn_layers=(2,),
        long_window_layers=(0, 3),
        use_flex_doc_swc=False,
        bfloat16_weights=False,
    )
    return CompiledLeaderboardClassicalGPT(cfg)


def test_m1_low_rank_cholesky_gram_mcweeny_algebraic_parity() -> None:
    """Gate M1: `associative_low_rank_cholesky_gram_modulation` matches materialized outer-product McWeeny to `< 1e-5`."""
    torch.manual_seed(7701)
    bsz, num_heads, seq_len, v_rank = 2, 4, 16, 1
    k_chol = DEFAULT_LOW_RANK_CHOLESKY_K

    for d_h in (32, 64):
        sdpa_dim = 2 * v_rank * d_h
        l_re = torch.randn(bsz, num_heads, seq_len, k_chol, d_h, requires_grad=True)
        l_im = torch.randn(bsz, num_heads, seq_len, k_chol, d_h, requires_grad=True)
        v_bar_re = torch.randn(bsz, num_heads, seq_len, v_rank, d_h, requires_grad=True)
        v_bar_im = torch.randn(bsz, num_heads, seq_len, v_rank, d_h, requires_grad=True)
        v_mixed = torch.randn(bsz, seq_len, num_heads, sdpa_dim, requires_grad=True)

        angles = torch.linspace(-0.2, 0.2, d_h)
        u_re = torch.diag(torch.cos(angles)).unsqueeze(0).expand(num_heads, -1, -1).clone().requires_grad_(True)
        u_im = torch.diag(torch.sin(angles)).unsqueeze(0).expand(num_heads, -1, -1).clone().requires_grad_(True)
        rho_scale = torch.tensor(0.25, requires_grad=True)
        obs_scale = torch.tensor(0.85, requires_grad=True)

        y_chol, tr_sq_chol = associative_low_rank_cholesky_gram_modulation(
            l_re,
            l_im,
            v_bar_re,
            v_bar_im,
            v_mixed,
            (u_re, u_im),
            (rho_scale, obs_scale, True),
        )
        loss_chol = y_chol.square().mean()
        loss_chol.backward()
        assert l_re.grad is not None and torch.isfinite(l_re.grad).all()
        assert v_mixed.grad is not None and torch.isfinite(v_mixed.grad).all()

        with torch.no_grad():
            rho0_re = torch.matmul(l_re.transpose(-2, -1), l_re) + torch.matmul(l_im.transpose(-2, -1), l_im)
            rho0_im = torch.matmul(l_im.transpose(-2, -1), l_re) - torch.matmul(l_re.transpose(-2, -1), l_im)
            y_dense, tr_sq_dense = associative_mcweeny_unitary_ket_modulation(
                rho0_re,
                rho0_im,
                v_bar_re,
                v_bar_im,
                v_mixed,
                (u_re, u_im),
                (rho_scale, obs_scale, True),
            )
            max_y_diff = float((y_chol - y_dense).abs().max().item())
            max_tr_diff = float((tr_sq_chol - tr_sq_dense).abs().max().item())
            assert max_y_diff < 1e-5, f"Cholesky Gram ket modulation mismatch at d_h={d_h}: {max_y_diff}"
            assert max_tr_diff < 1e-5, f"Cholesky Gram purity mismatch at d_h={d_h}: {max_tr_diff}"


def test_m2_unscaled_cold_start_learning_rates_vs_graft_damping() -> None:
    """Gate M2: `build_quantum_full_backbone_optimizers` sets `d_scale=1.0` on cold start and `< 1.0` on teacher graft."""
    torch.manual_seed(7702)
    teacher = _make_tiny_teacher()

    cfg_graft = resolve_plan077_arm_config("M77_1_LOW_RANK_CHOLESKY_PKT32_R384")
    model_graft = QuantumFullBackboneGraftGPT(teacher, graft_cfg=cfg_graft)
    opt_graft = build_quantum_full_backbone_optimizers(model_graft)
    expected_d_scale = compute_depth_lr_scale(len(model_graft.replaced_attn_layers))
    assert expected_d_scale < 0.90 and compute_depth_lr_scale(11) < 0.50
    assert abs(opt_graft.q_trunk_opt.param_groups[0]["lr"] - cfg_graft.stiefel_lr * expected_d_scale) < 1e-7

    cfg_scratch = resolve_plan077_arm_config("M77_3_FROM_SCRATCH_300M_QUANTUM_WAVEPACKET_GPT")
    model_scratch = QuantumFullBackboneGraftGPT(_make_tiny_teacher(), graft_cfg=cfg_scratch)
    model_scratch.from_scratch_mode = True
    model_scratch._init_standalone_full_backbone_modules()
    opt_scratch = build_quantum_full_backbone_optimizers(model_scratch)
    assert abs(opt_scratch.q_trunk_opt.param_groups[0]["lr"] - DEFAULT_FROM_SCRATCH_STIEFEL_LR) < 1e-7
    assert abs(opt_scratch.q_trunk_opt.param_groups[0]["unitary_lr"] - DEFAULT_FROM_SCRATCH_UNITARY_LR) < 1e-7
    assert abs(opt_scratch.q_wte_opt.param_groups[0]["lr"] - DEFAULT_FROM_SCRATCH_WTE_LR) < 1e-7
    assert abs(opt_scratch.base_scalar_lr - DEFAULT_FROM_SCRATCH_SCALAR_LR) < 1e-7


def test_m3_low_rank_cholesky_kernel_end_to_end_forward_backward() -> None:
    """Gate M3: `FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY` runs forward, backward, and step at `d_pkt=32` and `d_pkt=64`."""
    torch.manual_seed(7703)
    idx = torch.randint(0, 256, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 16), dtype=torch.long)

    for arm_id in ("M77_1_LOW_RANK_CHOLESKY_PKT32_R384", "M77_2_LOW_RANK_CHOLESKY_PKT64_R384"):
        teacher = _make_tiny_teacher()
        cfg = resolve_plan077_arm_config(arm_id)
        assert cfg.attn_kernel_mode == QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY
        model = QuantumFullBackboneGraftGPT(teacher, graft_cfg=cfg)
        opt = build_quantum_full_backbone_optimizers(model)
        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        opt.step(idx, lr_mult=1.0, momentum_beta=0.35)
        tel = model.compute_quantum_graft_telemetry()
        assert tel["q_max_stiefel_err"] < 1e-4
        assert tel["q_max_unitary_err"] < 1e-4
        assert tel["q_wte_sphere_err"] < 1e-4


def test_m4_cold_start_300m_arm_window_expansion_and_loss_descent() -> None:
    """Gate M4: Cold-start `M77_3` starts at `ln(V)`, expands `ws_blocks: 3->6`, and reduces loss."""
    torch.manual_seed(7704)
    cfg = resolve_plan077_arm_config("M77_3_FROM_SCRATCH_300M_QUANTUM_WAVEPACKET_GPT")
    model = QuantumFullBackboneGraftGPT(_make_tiny_teacher(), graft_cfg=cfg)
    model.from_scratch_mode = True
    model._init_standalone_full_backbone_modules()
    model.classical.lm_head.weight.requires_grad_(True)
    for l_idx, blk in enumerate(model.classical.blocks):
        if str(l_idx) not in model.quantum_mlp_blocks:
            blk.fc1.weight.requires_grad_(True)
            blk.fc2.weight.requires_grad_(True)

    assert int(model.classical.current_ws_blocks) == 3
    assert int(model.classical.long_window_buf.item()) == 384

    model.set_sliding_window_blocks(6, apply_yarn=True)
    assert int(model.classical.current_ws_blocks) == 6
    assert int(model.classical.long_window_buf.item()) == 768
    assert int(model.classical.short_window_buf.item()) == 384

    opt = build_quantum_full_backbone_optimizers(model)
    idx = torch.randint(0, 256, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 16), dtype=torch.long)

    _, loss_0 = model(idx, targets=targets)
    assert loss_0 is not None
    expected_ce0 = math.log(float(model.vocab_size))
    assert abs(float(loss_0.item()) - expected_ce0) < 1e-3

    for step in range(1, 4):
        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        opt.step(idx, lr_mult=1.0, momentum_beta=0.20)

    _, loss_after = model(idx, targets=targets)
    assert loss_after is not None
    assert float(loss_after.item()) < float(loss_0.item()) - 0.05


if __name__ == "__main__":
    test_m1_low_rank_cholesky_gram_mcweeny_algebraic_parity()
    print("[PASS] Gate M1: 4x4 Gram McWeeny algebraic parity (< 1e-5) verified at d_h=32 and d_h=64.")
    test_m2_unscaled_cold_start_learning_rates_vs_graft_damping()
    print("[PASS] Gate M2: Unscaled cold-start LRs (d_scale=1.0) vs. graft depth damping verified.")
    test_m3_low_rank_cholesky_kernel_end_to_end_forward_backward()
    print("[PASS] Gate M3: FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY end-to-end forward/backward verified.")
    test_m4_cold_start_300m_arm_window_expansion_and_loss_descent()
    print("[PASS] Gate M4: Cold-start 300M arm window expansion (ws_blocks: 3->6) & loss descent verified.")
    print("[ALL GATES M1..M4 PASSED]")
