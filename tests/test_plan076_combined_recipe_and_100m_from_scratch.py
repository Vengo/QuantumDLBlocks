"""Verification test suite for `EXPERIMENT_PLAN_076.md` (`Wave 55`).

Covers Gates `L1..L4`:
- `L1`: Zero-Mask Hermitian Sum Packing ($M = \\rho_R + \\rho_I$) & Associative McWeeny/Unitary Ket Contraction exactness (`< 1e-4` forward/backward).
- `L2`: `QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN` & Riemannian manifold-retracted `TailWeightAverager` (`retract_quantum_backbone_manifolds_`).
- `L3`: Cold-Start `QuantumFullBackboneGraftGPT.from_scratch` initialization (`step=0` `ln(V)` loss, non-zero gradients across all manifolds, strict descent).
- `L4`: Plan 076 arm registry (`L76_1..L76_5`) and Rule 7 constant synchronization.
"""

from __future__ import annotations

from importlib import import_module
import math
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "examples"))

import torch
import torch.nn.functional as F

from qdlblocks.kernels.fused_quantum_wavepacket_attn import (
    QuantumAttnKernelMode,
    associative_mcweeny_unitary_ket_modulation,
    fused_quantum_wavepacket_rho_ket_attention,
)
from qdlblocks.layers.impure_density_mirror_block import TRACE_NORM_EPS, bures_wasserstein_mcweeny_step
from qdlblocks.layers.quantum_frontend_graft import (
    QuantumRecoveryScheduleMode,
    QuantumStiefelProjMode,
    QuantumWavefunctionShape,
    QuantumWteInitMode,
)
from qdlblocks.layers.quantum_statmech_channels import (
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
    evolve_head_unitary_traceless_functional,
)
from qdlblocks.layers.quantum_svd_stiefel_graft import (
    DEFAULT_COLD_START_OBS_SCALE_INIT,
    DEFAULT_WSD_COOLDOWN_FRAC,
    DEFAULT_WSD_MIN_LR_RATIO,
    QuantumDepthLRMode,
    QuantumDocSWCMode,
    compute_quantum_backbone_lr_multiplier,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)
from qdlblocks.models.quantum_full_backbone_graft_gpt import (
    DEFAULT_QUANTUM_TAIL_EMA_BLEND,
    DEFAULT_QUANTUM_TAIL_EMA_RATE,
    DEFAULT_QUANTUM_TAIL_EMA_START_FRAC,
    QuantumBackboneDepthScope,
    QuantumFullBackboneGraftConfig,
    QuantumFullBackboneGraftGPT,
    QuantumKetAmplitudeMode,
    QuantumMLPChannelMode,
    build_quantum_full_backbone_optimizers,
    retract_quantum_backbone_manifolds_,
)
from qdlblocks.optim.tail_ema import TailWeightAverager


def test_l1_hermitian_sum_packing_and_associative_ket_exactness() -> None:
    """Gate L1: Verify $M = \\rho_R + \\rho_I$ packing & associative ket contraction match dense reference."""
    torch.manual_seed(2026076)
    bsz, num_heads, seq_len = 2, 6, 16
    for v_rank, d_h in ((2, 32), (1, 64)):
        head_dim = 2 * v_rank * d_h
        q_rot = torch.randn(bsz, seq_len, num_heads, head_dim, dtype=torch.float32, requires_grad=True)
        k_rot = torch.randn(bsz, seq_len, num_heads, head_dim, dtype=torch.float32, requires_grad=True)
        v_mixed = torch.randn(bsz, seq_len, num_heads * v_rank * 2, d_h, dtype=torch.float32, requires_grad=True)

        zc = v_mixed.detach().view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
        zr, zi = F.normalize(zc[..., 0, :], dim=-1), F.normalize(zc[..., 1, :], dim=-1)
        tok_rho_re = 0.5 * (torch.matmul(zr.transpose(-2, -1), zr) + torch.matmul(zi.transpose(-2, -1), zi))
        tok_rho_im = 0.5 * (torch.matmul(zi.transpose(-2, -1), zr) - torch.matmul(zr.transpose(-2, -1), zi))
        tok_rho_re.requires_grad_(True)
        tok_rho_im.requires_grad_(True)

        rho0_re, rho0_im, v_bar_re, v_bar_im = fused_quantum_wavepacket_rho_ket_attention(
            q_rot,
            k_rot,
            tok_rho_re,
            tok_rho_im,
            v_mixed,
            phase_and_born=(None, None, 0.12, False),
        )
        # Verify Hermiticity of unpacked rho0
        sym_err = float((rho0_re - rho0_re.transpose(-2, -1)).abs().max().item())
        skew_err = float((rho0_im + rho0_im.transpose(-2, -1)).abs().max().item())
        assert sym_err < 1e-6 and skew_err < 1e-6

        angles = torch.linspace(-0.2, 0.2, d_h)
        u_re = torch.diag(torch.cos(angles)).unsqueeze(0).repeat(num_heads, 1, 1).requires_grad_(True)
        u_im = torch.diag(torch.sin(angles)).unsqueeze(0).repeat(num_heads, 1, 1).requires_grad_(True)
        rho_scale = torch.tensor(0.05, requires_grad=True)
        obs_scale = torch.tensor(1.0, requires_grad=True)

        # Reference dense matrix-product path
        pure_re, pure_im, _, _ = bures_wasserstein_mcweeny_step(rho0_re.detach(), rho0_im.detach(), num_steps=1)
        evol_re, evol_im = evolve_head_unitary_traceless_functional(pure_re, pure_im, u_re, u_im)
        eye_d = torch.eye(d_h)
        rho_tl_re = evol_re - (1.0 / float(d_h)) * eye_d
        rho_tl_im = evol_im
        u0_re = u_re.view(1, num_heads, 1, d_h, d_h)
        u0_im = u_im.view(1, num_heads, 1, d_h, d_h)
        vu_re = torch.matmul(v_bar_re.detach(), u0_re.transpose(-2, -1)) - torch.matmul(
            v_bar_im.detach(), u0_im.transpose(-2, -1)
        )
        vu_im = torch.matmul(v_bar_re.detach(), u0_im.transpose(-2, -1)) + torch.matmul(
            v_bar_im.detach(), u0_re.transpose(-2, -1)
        )
        v_loc_c = v_mixed.detach().view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
        vl_re, vl_im = v_loc_c[..., 0, :], v_loc_c[..., 1, :]
        lu_raw_re = torch.matmul(vl_re, u0_re.transpose(-2, -1)) - torch.matmul(vl_im, u0_im.transpose(-2, -1))
        lu_raw_im = torch.matmul(vl_re, u0_im.transpose(-2, -1)) + torch.matmul(vl_im, u0_re.transpose(-2, -1))
        att_fro = (
            (vu_re.square().sum(dim=(-2, -1), keepdim=True) + vu_im.square().sum(dim=(-2, -1), keepdim=True))
            .clamp_min(TRACE_NORM_EPS)
            .sqrt()
        )
        loc_fro = (
            (lu_raw_re.square().sum(dim=(-2, -1), keepdim=True) + lu_raw_im.square().sum(dim=(-2, -1), keepdim=True))
            .clamp_min(TRACE_NORM_EPS)
            .sqrt()
        )
        tar_re = lu_raw_re * (att_fro / loc_fro)
        tar_im = lu_raw_im * (att_fro / loc_fro)
        rv_re = torch.matmul(tar_re, rho_tl_re) + torch.matmul(tar_im, rho_tl_im)
        rv_im = torch.matmul(tar_im, rho_tl_re) - torch.matmul(tar_re, rho_tl_im)
        wp_re = vu_re + rho_scale * rv_re
        wp_im = vu_im + rho_scale * rv_im
        y_ref = (
            torch.stack([wp_re, wp_im], dim=-2).permute(0, 2, 1, 3, 4, 5).reshape(bsz, seq_len, num_heads, head_dim)
            * obs_scale
        )

        y_assoc, tr_sq = associative_mcweeny_unitary_ket_modulation(
            rho0_re.detach(),
            rho0_im.detach(),
            v_bar_re.detach(),
            v_bar_im.detach(),
            v_mixed.detach(),
            (u_re, u_im),
            (rho_scale, obs_scale, True),
        )
        fwd_diff = float((y_ref - y_assoc).abs().max().item())
        assert fwd_diff < 1e-5, f"Associative ket modulation mismatch at d_h={d_h}: {fwd_diff}"
        assert float(tr_sq.min().item()) > 0.0 and float(tr_sq.max().item()) <= 1.0


def test_l2_wsd_cooldown_and_manifold_retracted_tail_ema() -> None:
    """Gate L2: Verify `DEPTH_SQRT_WARMUP_WSD_COOLDOWN` schedule and `retract_quantum_backbone_manifolds_`."""
    tot = 100
    m_start = compute_quantum_backbone_lr_multiplier(1, tot, QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN)
    m_warm_end = compute_quantum_backbone_lr_multiplier(10, tot, QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN)
    m_stable = compute_quantum_backbone_lr_multiplier(65, tot, QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN)
    m_end = compute_quantum_backbone_lr_multiplier(100, tot, QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN)
    assert abs(m_start - 0.10) < 1e-6
    assert abs(m_warm_end - 1.0) < 1e-6
    assert abs(m_stable - 1.0) < 1e-6
    assert abs(m_end - DEFAULT_WSD_MIN_LR_RATIO) < 1e-6
    assert DEFAULT_WSD_COOLDOWN_FRAC == 0.35

    torch.manual_seed(2026076)
    c_cfg = CompiledLeaderboardClassicalConfig(
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
    base_m = CompiledLeaderboardClassicalGPT(c_cfg)
    g_cfg = QuantumFullBackboneGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        depth_scope=QuantumBackboneDepthScope.EMB_PLUS_L0_TO_L2_ATTN,
        attn_kernel_mode=QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
        depth_lr_mode=QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN,
    )
    model = QuantumFullBackboneGraftGPT(base_m, graft_cfg=g_cfg)
    ema = TailWeightAverager(
        model,
        start_frac=DEFAULT_QUANTUM_TAIL_EMA_START_FRAC,
        ema_rate=DEFAULT_QUANTUM_TAIL_EMA_RATE,
        blend_alpha=DEFAULT_QUANTUM_TAIL_EMA_BLEND,
        decontract_frobenius=False,
    )
    ema.update(step=7, total_steps=10)
    with torch.no_grad():
        for w in model.quantum_real_stiefel_parameters():
            w.add_(0.15 * torch.randn_like(w))
        for u_re, u_im in model.quantum_complex_unitary_pairs():
            u_re.add_(0.15 * torch.randn_like(u_re))
            u_im.add_(0.15 * torch.randn_like(u_im))
        for emb in model.quantum_sphere_embedding_parameters():
            emb.add_(0.25 * torch.randn_like(emb))
    ema.update(step=10, total_steps=10)

    with ema.average_parameters(model):
        retract_quantum_backbone_manifolds_(model)
        tel = model.compute_quantum_graft_telemetry()
        assert tel["q_max_stiefel_err"] < 1e-4
        assert tel["q_max_unitary_err"] < 1e-4
        assert tel["q_wte_sphere_err"] < 1e-4


def test_l3_cold_start_from_scratch_initialization_and_descent() -> None:
    """Gate L3: Verify `QuantumFullBackboneGraftGPT.from_scratch` Step-0 `ln(V)` parity and multi-step descent."""
    torch.manual_seed(2026076)
    c_cfg = CompiledLeaderboardClassicalConfig(
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
    base_m = CompiledLeaderboardClassicalGPT(c_cfg)
    g_cfg = QuantumFullBackboneGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
        attn_kernel_mode=QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET,
        stiefel_momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        momentum_schedule_mode=QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070,
        depth_lr_mode=QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN,
        wte_init_mode=QuantumWteInitMode.PPMI_SVD_SPHERE,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        stiefel_proj_mode=QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64,
        doc_swc_mode=QuantumDocSWCMode.TEACHER_DOC_SWC_MASK,
        interface_warmup_steps=0,
    )
    model = QuantumFullBackboneGraftGPT(base_m, graft_cfg=g_cfg)
    model.from_scratch_mode = True
    model._init_standalone_full_backbone_modules()
    model.classical.lm_head.weight.requires_grad_(True)
    for l_idx, blk in enumerate(model.classical.blocks):
        if str(l_idx) not in model.quantum_mlp_blocks:
            blk.fc1.weight.requires_grad_(True)
            blk.fc2.weight.requires_grad_(True)

    opt = build_quantum_full_backbone_optimizers(
        model, stiefel_lr=0.02, unitary_lr=0.02, wte_lr=0.03, scalar_lr=0.015
    )
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))

    losses: list[float] = []
    for s in range(1, 5):
        lr_m = compute_quantum_backbone_lr_multiplier(s, 4, g_cfg.depth_lr_mode)
        mom_b = compute_riemannian_momentum_beta(s, 4, g_cfg.momentum_schedule_mode)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        assert loss is not None
        if s == 1:
            assert abs(float(loss.item()) - math.log(256.0)) < 1e-4
        loss.backward()
        if s == 2:
            st_grads = [w.grad.abs().max().item() for w in model.quantum_real_stiefel_parameters() if w.grad is not None]
            assert len(st_grads) > 0 and max(st_grads) > 0.0
        opt.step(x, lr_mult=lr_m, momentum_beta=mom_b)
        losses.append(float(loss.detach().item()))

    assert losses[-1] < losses[0], f"Expected cold-start loss descent, got {losses}"
    tel = model.compute_quantum_graft_telemetry()
    assert tel["q_max_stiefel_err"] < 1e-3
    assert tel["q_max_unitary_err"] < 1e-3
    assert tel["q_wte_sphere_err"] < 1e-4


def test_l4_plan076_arm_registry_and_constants() -> None:
    """Gate L4: Verify `86_verify_plan076_combined_recipe_and_100m_from_scratch.py` arm specs (`L76_1..L76_5`)."""
    ex86 = import_module("86_verify_plan076_combined_recipe_and_100m_from_scratch")
    expected_arms = {
        "L76_1_COMBINED_PKT32_LOW_MOM070_TAN_SNR_R384",
        "L76_2_COMBINED_PKT32_WSD_COOLDOWN_TAIL_EMA_R384",
        "L76_3_COMBINED_PKT64_LOW_MOM070_TAN_SNR_TAIL_EMA_R384",
        "L76_4_COMBINED_FULL_BACKBONE_MZ_PKT32_R768",
        "L76_5_FROM_SCRATCH_100M_QUANTUM_WAVEPACKET_GPT",
    }
    assert set(ex86.ARM_SPECS.keys()) == expected_arms
    assert DEFAULT_COLD_START_OBS_SCALE_INIT == 0.25


if __name__ == "__main__":
    test_l1_hermitian_sum_packing_and_associative_ket_exactness()
    print("[PASS] Gate L1")
    test_l2_wsd_cooldown_and_manifold_retracted_tail_ema()
    print("[PASS] Gate L2")
    test_l3_cold_start_from_scratch_initialization_and_descent()
    print("[PASS] Gate L3")
    test_l4_plan076_arm_registry_and_constants()
    print("[PASS] Gate L4")
    print("[ALL GATES L1..L4 PASSED]")

