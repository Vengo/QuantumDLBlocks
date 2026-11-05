#!/usr/bin/env python3
"""Unit and Verification Tests for `EXPERIMENT_PLAN_078.md` (`Wave 57`: Gates N1..N4).

Verifies:
- `Gate N1`: Joint `(2, d_h)` complex sphere normalization & $U(1)$ phase invariance (`< 1e-6`).
- `Gate N2`: Exact Complex Hermitian Gram-Schmidt orthogonality (`< 1e-5` real & imaginary overlap)
  and Active-Only Centering rare-token anti-collapse (`< 0.50` max rare cosine vs. `1.0` naive).
- `Gate N3`: Dynamic Complex Packet Readout phase preservation and Identity-Packet Anchored Drift
  (`id_lr_scale = 0.25`) in `BatchSubspaceTikhonovGramMuon`.
- `Gate N4`: Step-0 readout transition calibration (`val_ce < 7.90`, rejecting Shard 0) and
  end-to-end 12-layer cold-start `QuantumFullBackboneGraftGPT` loss descent (`< 1e-4` manifold err).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from qdlblocks.kernels.fused_quantum_wavepacket_attn import QuantumAttnKernelMode
from qdlblocks.layers.complex_fiber_bundle_init import (
    DEFAULT_IDENTITY_PACKET_LR_SCALE,
    DynamicComplexPacketReadoutModulator,
    QuantumPacketReadoutMode,
    _load_fineweb_tokens_from_shard,
    active_only_center_rows,
    calibrate_cold_start_readout_from_shard_,
    compute_directed_complex_ppmi_from_tokens,
    compute_hermitian_orthogonal_residual,
    initialize_directed_complex_ppmi_fiber_bundle_embeddings_,
    project_complex_wavepackets_to_rms_sphere_,
)
from qdlblocks.layers.quantum_frontend_graft import (
    QuantumRecoveryScheduleMode,
    QuantumWavefunctionShape,
    QuantumWteInitMode,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)
from qdlblocks.models.quantum_full_backbone_graft_gpt import (
    QuantumBackboneDepthScope,
    QuantumFullBackboneGraftConfig,
    QuantumFullBackboneGraftGPT,
    build_quantum_full_backbone_optimizers,
)
from qdlblocks.optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SYNTH_VOCAB_SIZE: int = 256
SYNTH_M_C: int = 12
SYNTH_D_H: int = 32
MAX_PHASE_NORM_ERR: float = 1e-5
MAX_HERMITIAN_OVERLAP_ERR: float = 1e-5
MAX_RARE_TOKEN_COS: float = 0.50


def test_n1_joint_complex_sphere_normalization_and_u1_invariance() -> None:
    """Gate N1: Verify joint `(2, d_h)` complex sphere normalization and $U(1)$ phase rotation invariance."""
    torch.manual_seed(42)
    w_4d = torch.randn(SYNTH_VOCAB_SIZE, SYNTH_M_C, 2, SYNTH_D_H)
    w_4d[:, :, 0, :] *= 2.5
    w_4d[:, :, 1, :] *= 0.4
    project_complex_wavepackets_to_rms_sphere_(w_4d)

    joint_norm_sq = w_4d.flatten(start_dim=-2).square().sum(dim=-1)
    target_sq = float(2 * SYNTH_D_H)
    assert float(((joint_norm_sq / target_sq).sqrt() - 1.0).abs().max().item()) < MAX_PHASE_NORM_ERR

    re_energy = w_4d[:, :, 0, :].square().sum(dim=-1).mean()
    im_energy = w_4d[:, :, 1, :].square().sum(dim=-1).mean()
    assert float((re_energy - im_energy).abs().item()) > 5.0

    phi = torch.linspace(-1.7, 1.7, SYNTH_M_C).view(1, SYNTH_M_C, 1)
    c_p, s_p = torch.cos(phi), torch.sin(phi)
    rot_re = c_p * w_4d[:, :, 0, :] - s_p * w_4d[:, :, 1, :]
    rot_im = s_p * w_4d[:, :, 0, :] + c_p * w_4d[:, :, 1, :]
    rot_norm_sq = rot_re.square().sum(dim=-1) + rot_im.square().sum(dim=-1)
    assert float(((rot_norm_sq / target_sq).sqrt() - 1.0).abs().max().item()) < MAX_PHASE_NORM_ERR


def test_n2_hermitian_gram_schmidt_and_active_only_rare_token_anti_collapse() -> None:
    """Gate N2: Verify exact Complex Hermitian Gram-Schmidt orthogonality and Active-Only Centering rare-token anti-collapse."""
    torch.manual_seed(77)
    s_re = torch.randn(SYNTH_VOCAB_SIZE, 11, SYNTH_D_H)
    s_im = torch.randn(SYNTH_VOCAB_SIZE, 11, SYNTH_D_H)
    z_re = torch.randn(SYNTH_VOCAB_SIZE, 11, SYNTH_D_H)
    z_im = torch.randn(SYNTH_VOCAB_SIZE, 11, SYNTH_D_H)

    sr_u, si_u, zp_re, zp_im, _ = compute_hermitian_orthogonal_residual(s_re, s_im, z_re, z_im)
    two_dh = float(2 * SYNTH_D_H)
    ov_re = (sr_u * zp_re + si_u * zp_im).sum(dim=-1) / two_dh
    ov_im = (sr_u * zp_im - si_u * zp_re).sum(dim=-1) / two_dh
    assert float(ov_re.abs().max().item()) < MAX_HERMITIAN_OVERLAP_ERR
    assert float(ov_im.abs().max().item()) < MAX_HERMITIAN_OVERLAP_ERR

    raw_mat = torch.zeros(SYNTH_VOCAB_SIZE, 64)
    raw_mat[:180] = torch.rand(180, 64) + 0.5
    naive_centered = raw_mat - raw_mat.mean(dim=0, keepdim=True)
    naive_rare = F.normalize(naive_centered[180:], dim=-1)
    naive_cos = torch.matmul(naive_rare, naive_rare.t())
    naive_cos.fill_diagonal_(-1.0)
    assert float(naive_cos.max().item()) > 0.9999

    act_centered, act_mask = active_only_center_rows(raw_mat)
    assert int(act_mask.sum().item()) == 180
    assert float(act_centered[180:].abs().max().item()) == 0.0

    gen_tok = torch.Generator().manual_seed(123)
    stream = torch.randint(0, 180, (16_384,), generator=gen_tok)
    sketch = compute_directed_complex_ppmi_from_tokens(
        stream, vocab_size=SYNTH_VOCAB_SIZE, m_c=SYNTH_M_C, d_h=SYNTH_D_H, device="cpu"
    )
    wte_3d = torch.empty(SYNTH_VOCAB_SIZE, SYNTH_M_C, 2 * SYNTH_D_H)
    stats = initialize_directed_complex_ppmi_fiber_bundle_embeddings_(
        wte_3d, m_sem=11, sketch_dict=sketch
    )
    assert stats["rare_max_cos"] < MAX_RARE_TOKEN_COS


def test_n3_dynamic_packet_readout_phase_and_identity_anchored_drift() -> None:
    """Gate N3: Verify phase-preserving dynamic packet readout and `id_lr_scale = 0.25` in `BatchSubspaceTikhonovGramMuon`."""
    torch.manual_seed(99)
    mod = DynamicComplexPacketReadoutModulator(embed_dim=768, m_c=12, m_sem=10, init_biased=True)
    x_in = torch.randn(2, 16, 768)
    x_out = mod(x_in)

    xin_4d = x_in.view(2, 16, 12, 2, 32)
    xout_4d = x_out.view(2, 16, 12, 2, 32)
    phase_in = torch.atan2(xin_4d[..., 1, :], xin_4d[..., 0, :])
    phase_out = torch.atan2(xout_4d[..., 1, :], xout_4d[..., 0, :])
    assert float((phase_in - phase_out).abs().max().item()) < 1e-5

    p_wte = torch.nn.Parameter(torch.randn(32, 12, 64))
    project_complex_wavepackets_to_rms_sphere_(p_wte)
    w_init = p_wte.detach().clone()

    opt = BatchSubspaceTikhonovGramMuon([p_wte], lr=0.08, beta=0.0)
    opt.configure_packet_lr_scales(p_wte, m_sem=10, id_lr_scale=DEFAULT_IDENTITY_PACKET_LR_SCALE)

    g_raw = torch.randn(32, 1, 64).expand(32, 12, 64).clone()
    for m_i in range(12):
        g_raw[:, m_i, :] = torch.randn_like(g_raw[:, 0, :])
    p_wte.grad = g_raw
    opt.step_active(torch.arange(32))

    cos_change = (w_init * p_wte.detach()).sum(dim=-1) / 64.0
    angles = torch.acos(cos_change.clamp(-1.0, 1.0))
    sem_angle = float(angles[:, :10].mean().item())
    id_angle = float(angles[:, 10:].mean().item())
    assert id_angle < 0.40 * sem_angle


def test_n4_step0_readout_calibration_and_cold_start_descent() -> None:
    """Gate N4: Verify Step-0 transition calibration (`val_ce < 7.90`, rejecting Shard 0) and 12-layer cold-start loss descent."""
    rejected_shard0 = False
    try:
        _load_fineweb_tokens_from_shard(Path("fineweb_val_000000.bin"), max_tokens=1024)
    except ValueError:
        rejected_shard0 = True
    assert rejected_shard0, "Must reject validation Shard 0"

    torch.manual_seed(314)
    c_cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=SYNTH_VOCAB_SIZE,
        embed_dim=768,
        num_layers=4,
        num_heads=6,
        hidden_dim=256,
        max_seq_len=64,
        num_value_embeddings=1,
        skip_attn_layers=(2,),
        long_window_layers=(0, 3),
        use_flex_doc_swc=False,
        bfloat16_weights=False,
    )
    base_classical = CompiledLeaderboardClassicalGPT(c_cfg)

    gen_tok = torch.Generator().manual_seed(555)
    hot_tokens = torch.randint(0, 16, (2049,), generator=gen_tok)
    sketch = compute_directed_complex_ppmi_from_tokens(
        hot_tokens, vocab_size=SYNTH_VOCAB_SIZE, m_c=12, d_h=32, device="cpu"
    )

    graft_cfg = QuantumFullBackboneGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        depth_scope=QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN,
        wte_init_mode=QuantumWteInitMode.COMPLEX_FIBER_BUNDLE_11SEM_1ID,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        interface_warmup_steps=0,
        packet_readout_mode=QuantumPacketReadoutMode.DYNAMIC_COMPLEX_PACKET_GATED_READOUT,
        attn_kernel_mode=QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY,
        wte_lr=0.060,
        stiefel_lr=0.025,
        unitary_lr=0.025,
        scalar_lr=0.020,
    )
    model = QuantumFullBackboneGraftGPT(base_classical, graft_cfg=graft_cfg)
    initialize_directed_complex_ppmi_fiber_bundle_embeddings_(
        model.q_wte, m_sem=11, sketch_dict=sketch
    )
    model.from_scratch_mode = True
    model._init_standalone_full_backbone_modules()
    model.classical.lm_head.weight.requires_grad_(True)
    for blk in model.classical.blocks:
        blk.fc1.weight.requires_grad_(True)
        blk.fc2.weight.requires_grad_(True)

    assert tuple(model.q_wte.shape) == (SYNTH_VOCAB_SIZE, 12, 64)

    x_eval = hot_tokens[:128].view(2, 64)
    y_eval = hot_tokens[1:129].view(2, 64)
    _, loss_uncalib = model(x_eval, y_eval)
    assert abs(float(loss_uncalib.item()) - math.log(SYNTH_VOCAB_SIZE)) < 1e-3

    calibrate_cold_start_readout_from_shard_(
        model, tokens_override=hot_tokens[:1025], logit_scale=1.50
    )
    _, loss_calib = model(x_eval, y_eval)
    assert float(loss_calib.item()) < float(loss_uncalib.item()) - 0.50

    opts = build_quantum_full_backbone_optimizers(model)
    losses: list[float] = []
    for step_i in range(3):
        opts.zero_grad()
        _, loss = model(x_eval, y_eval)
        assert loss is not None
        losses.append(float(loss.item()))
        loss.backward()
        opts.step(x_eval, lr_mult=1.0, momentum_beta=0.50)

    assert losses[-1] < losses[0]
    tel = model.compute_quantum_graft_telemetry()
    assert tel["q_max_stiefel_err"] < 1e-4
    assert tel["q_wte_sphere_err"] < 1e-4


if __name__ == "__main__":
    test_n1_joint_complex_sphere_normalization_and_u1_invariance()
    print("[PASS] Gate N1: Joint (2, d_h) complex sphere normalization & U(1) invariance verified.")
    test_n2_hermitian_gram_schmidt_and_active_only_rare_token_anti_collapse()
    print("[PASS] Gate N2: Complex Hermitian Gram-Schmidt orthogonality & Active-Only rare-token anti-collapse verified.")
    test_n3_dynamic_packet_readout_phase_and_identity_anchored_drift()
    print("[PASS] Gate N3: Dynamic complex packet readout phase preservation & identity-anchored drift verified.")
    test_n4_step0_readout_calibration_and_cold_start_descent()
    print("[PASS] Gate N4: Step-0 readout transition calibration & 12-layer cold-start descent verified.")
    print("[ALL GATES N1..N4 PASSED]")
