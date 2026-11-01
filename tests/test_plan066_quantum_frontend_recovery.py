"""Verification tests for EXPERIMENT PLAN 066 (Wave 52B): Teacher-Subspace Spherical Warm-Start & Sub-300M Recovery."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qdlblocks.layers.quantum_frontend_graft import (
    FrozenTrunkQuantumFrontEndGPT,
    QuantumFrontEndGraftConfig,
    QuantumRecoveryScheduleMode,
    QuantumValueEmbedCouplingMode,
    QuantumWavefunctionShape,
    QuantumWteInitMode,
    build_frozen_trunk_quantum_frontend_optimizers,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)


def _make_small_classical_teacher() -> CompiledLeaderboardClassicalGPT:
    """Construct a deterministic 4-layer classical teacher with realistic non-zero weights for unit testing."""
    torch.manual_seed(2026066)
    c_cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=64,
        embed_dim=768,
        num_heads=6,
        num_layers=4,
        hidden_dim=1536,
        max_seq_len=32,
        num_value_embeddings=2,
        skip_attn_layers=(),
        long_window_layers=(0, 3),
        use_flex_doc_swc=False,
        bfloat16_weights=False,
    )
    teacher = CompiledLeaderboardClassicalGPT(c_cfg)
    with torch.no_grad():
        teacher.wte.weight.normal_(0.0, 0.05)
        teacher.lm_head.weight.normal_(0.0, 0.02)
        if teacher.value_embeds is not None:
            for ve in teacher.value_embeds:
                ve.weight.normal_(0.0, 0.05)
        for blk in teacher.blocks:
            blk.block_lambdas.copy_(torch.tensor([1.0, 0.15], dtype=torch.float32))
            if blk.out_proj is not None:
                blk.out_proj.weight.normal_(0.0, 0.02)
            blk.fc2.weight.normal_(0.0, 0.02)
    return teacher


def test_h1_teacher_procrustes_sphere_alignment_across_shapes() -> None:
    """Gate H1: Verify exact sphere/Stiefel manifold constraints and high Step-0 x_0 cosine similarity."""
    idx = torch.randint(0, 64, (2, 16))
    ppmi_syn = torch.randn(64, 768)

    for wf_shape in (
        QuantumWavefunctionShape.SHAPE_24x2x16,
        QuantumWavefunctionShape.SHAPE_1x2x256,
        QuantumWavefunctionShape.SHAPE_1x2x512,
        QuantumWavefunctionShape.SHAPE_256x2x2,
    ):
        teacher = _make_small_classical_teacher()
        cfg_ppmi = QuantumFrontEndGraftConfig(wf_shape=wf_shape, wte_init_mode=QuantumWteInitMode.PPMI_SVD_SPHERE)
        cfg_proc = QuantumFrontEndGraftConfig(
            wf_shape=wf_shape, wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE
        )
        m_ppmi = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg_ppmi, ppmi_embeddings=ppmi_syn)
        m_proc = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg_proc, ppmi_embeddings=ppmi_syn)

        tel = m_proc.compute_quantum_graft_telemetry()
        assert tel["q_wte_sphere_err"] < 1e-5, f"Sphere err {tel['q_wte_sphere_err']} for {wf_shape}"
        assert tel["q_max_stiefel_err"] < 1e-5, f"Stiefel err {tel['q_max_stiefel_err']} for {wf_shape}"

        with torch.no_grad():
            x0_t, _, _ = m_proc.compute_teacher_layer0_targets(idx)
            x0_ppmi, _, _ = m_ppmi.encode_quantum_layer0(idx)
            x0_proc, _, _ = m_proc.encode_quantum_layer0(idx)
            cos_ppmi = float(F.cosine_similarity(x0_ppmi.float(), x0_t.float(), dim=-1).mean().item())
            cos_proc = float(F.cosine_similarity(x0_proc.float(), x0_t.float(), dim=-1).mean().item())
            assert cos_proc > cos_ppmi + 0.50, f"Expected cos_proc >> cos_ppmi for {wf_shape}: {cos_proc} vs {cos_ppmi}"
            if wf_shape != QuantumWavefunctionShape.SHAPE_1x2x256:
                assert cos_proc > 0.90, f"Expected cos_proc > 0.90 for {wf_shape}, got {cos_proc}"
            else:
                assert cos_proc > 0.80, f"Expected cos_proc > 0.80 for {wf_shape}, got {cos_proc}"


def test_h2_frozen_teacher_ve0_coupling_gradients_and_no_dead_params() -> None:
    """Gate H2: Verify ve0_lambdas gradient flow when enabled and absence when disabled."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    teacher_none = _make_small_classical_teacher()
    m_none = FrozenTrunkQuantumFrontEndGPT(
        teacher_none,
        graft_cfg=QuantumFrontEndGraftConfig(ve_coupling_mode=QuantumValueEmbedCouplingMode.NONE),
    )
    assert m_none.ve0_lambdas is None

    for wf_shape in (
        QuantumWavefunctionShape.SHAPE_24x2x16,
        QuantumWavefunctionShape.SHAPE_1x2x256,
        QuantumWavefunctionShape.SHAPE_1x2x512,
        QuantumWavefunctionShape.SHAPE_256x2x2,
    ):
        teacher = _make_small_classical_teacher()
        cfg = QuantumFrontEndGraftConfig(
            wf_shape=wf_shape,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
            ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        )
        model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg)
        assert model.ve0_lambdas is not None and model.ve0_lambdas.requires_grad
        opt = build_frozen_trunk_quantum_frontend_optimizers(model)
        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets=targets)
        assert loss is not None
        loss.backward()
        assert all(p.grad is None for p in model.classical.parameters())
        assert model.ve0_lambdas.grad is not None
        assert float(model.ve0_lambdas.grad.abs().min().item()) > 0.0


def test_h3_interface_and_final_feat_schedule_transition() -> None:
    """Gate H3: Verify two-stage interface + final-feature alignment loss during warmup and pure CE after T_align."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    teacher = _make_small_classical_teacher()

    cfg_pure = QuantumFrontEndGraftConfig(
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PURE_CE,
        interface_warmup_steps=5,
    )
    cfg_2stage = QuantumFrontEndGraftConfig(
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE,
        interface_warmup_steps=5,
        interface_align_weight=1.5,
    )
    torch.manual_seed(42)
    m_pure = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg_pure)
    torch.manual_seed(42)
    m_2stage = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg_2stage)

    m_pure.train()
    m_2stage.train()
    _, loss_pure_0 = m_pure(idx, targets=targets, return_logits=True)
    _, loss_2stage_0 = m_2stage(idx, targets=targets, return_logits=True)
    assert loss_pure_0 is not None and loss_2stage_0 is not None
    assert float(loss_2stage_0.item()) > float(loss_pure_0.item())

    m_2stage.training_step = 5
    _, loss_2stage_after = m_2stage(idx, targets=targets, return_logits=True)
    assert loss_2stage_after is not None
    assert abs(float(loss_2stage_after.item()) - float(loss_pure_0.item())) < 1e-6


def test_h4_rapid_loss_descent_with_procrustes_and_ve0_coupling() -> None:
    """Gate H4: Verify strictly lower loss and manifold preservation over 10 optimization steps."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    teacher = _make_small_classical_teacher()

    cfg = QuantumFrontEndGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE,
        interface_warmup_steps=5,
    )
    model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg)
    opt = build_frozen_trunk_quantum_frontend_optimizers(model, stiefel_lr=0.06, wte_lr=0.18, scalar_lr=0.06)

    model.eval()
    _, init_ce = model(idx, targets=targets, return_logits=True)
    model.train()
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        _, total_loss = model(idx, targets=targets)
        assert total_loss is not None
        total_loss.backward()
        opt.step(idx=idx, lr_mult=1.0)

    model.eval()
    _, final_ce = model(idx, targets=targets, return_logits=True)
    assert init_ce is not None and final_ce is not None
    assert float(final_ce.item()) < float(init_ce.item())
    tel = model.compute_quantum_graft_telemetry()
    assert tel["q_max_stiefel_err"] < 1e-5
    assert tel["q_wte_sphere_err"] < 1e-5


if __name__ == "__main__":
    for test_fn in (
        test_h1_teacher_procrustes_sphere_alignment_across_shapes,
        test_h2_frozen_teacher_ve0_coupling_gradients_and_no_dead_params,
        test_h3_interface_and_final_feat_schedule_transition,
        test_h4_rapid_loss_descent_with_procrustes_and_ve0_coupling,
    ):
        test_fn()
        print(f"PASS: {test_fn.__name__}")
