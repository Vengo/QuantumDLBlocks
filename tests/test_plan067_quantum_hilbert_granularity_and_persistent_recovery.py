"""Verification Suite for Wave 52C (`EXPERIMENT_PLAN_067.md`):
Outer-Product Hilbert Granularity ($d_{\\text{pkt}} \\in \\{8, 16, 32\\}$), Rank-3 Causal Ensemble DM I/O, and Persistent Subspace Recovery.

Executes the 4 mandatory unit verification gates (`H1..H4`) specified in `EXPERIMENT_PLAN_067.md`.
"""

from __future__ import annotations

import torch

from qdlblocks.layers.quantum_frontend_graft import (
    ConfigurableWavepacketStatMechBlock,
    DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO,
    FrozenTrunkQuantumFrontEndGPT,
    QuantumAttnInteriorMode,
    QuantumFrontEndGraftConfig,
    QuantumRecoveryScheduleMode,
    QuantumReplacementScope,
    QuantumStateIOMode,
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
    """Create a fast, deterministic 4-layer classical teacher (`D=768, H=6, V=64`) for unit tests."""
    torch.manual_seed(2026067)
    cfg = CompiledLeaderboardClassicalConfig(
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
    teacher = CompiledLeaderboardClassicalGPT(cfg)
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


def test_h1_zero_padding_outer_product_aggregation_across_pkt_dims() -> None:
    """Gate H1: Verify unit trace, Hermiticity, PSD eigenvalues, and causal masking across d_pkt in {8, 16, 32}."""
    idx_a = torch.randint(0, 64, (2, 16))
    idx_b = idx_a.clone()
    idx_b[:, 8:] = (idx_b[:, 8:] + 17) % 64

    for wf_shape in (
        QuantumWavefunctionShape.SHAPE_48x2x8,
        QuantumWavefunctionShape.SHAPE_24x2x16,
        QuantumWavefunctionShape.SHAPE_12x2x32,
        QuantumWavefunctionShape.SHAPE_1x2x512_PKT32,
    ):
        teacher = _make_small_classical_teacher()
        cfg = QuantumFrontEndGraftConfig(
            wf_shape=wf_shape,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        )
        model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg)
        model.eval()
        with torch.no_grad():
            logits_a, _ = model(idx_a, return_logits=True)
            logits_b, _ = model(idx_b, return_logits=True)
            assert logits_a is not None and logits_b is not None
            causal_diff = float((logits_a[:, :8] - logits_b[:, :8]).abs().max().item())
            assert causal_diff < 1e-5, f"Causality violation {causal_diff} for {wf_shape}"

            assert isinstance(model.quantum_block0, ConfigurableWavepacketStatMechBlock)
            assert model.quantum_block0._last_rho0_re_im is not None
            rho_re, rho_im = model.quantum_block0._last_rho0_re_im
            tr_err = float((torch.diagonal(rho_re.float(), dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
            assert tr_err < 1e-4, f"Trace error {tr_err} for {wf_shape}"
            herm_re = float((rho_re.float() - rho_re.float().transpose(-2, -1)).abs().max().item())
            herm_im = float((rho_im.float() + rho_im.float().transpose(-2, -1)).abs().max().item())
            assert max(herm_re, herm_im) < 1e-5, f"Hermiticity error {max(herm_re, herm_im)} for {wf_shape}"
            rho_c = torch.complex(
                0.5 * (rho_re.float() + rho_re.float().transpose(-2, -1)),
                0.5 * (rho_im.float() - rho_im.float().transpose(-2, -1)),
            )
            min_eig = float(torch.linalg.eigvalsh(rho_c).min().item())
            assert min_eig >= -1e-5, f"Negative eigenvalue {min_eig} for {wf_shape}"


def test_h2_causal_ensemble_dm_io_physicality_and_gradients() -> None:
    """Gate H2: Verify CAUSAL_ENSEMBLE_DM_IO boosts Step-0 effective rank and propagates gradients to dm_lag_mix/dm_res_mix."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    teacher_wf = _make_small_classical_teacher()
    m_wf = FrozenTrunkQuantumFrontEndGPT(
        teacher_wf,
        graft_cfg=QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ),
    )
    teacher_dm = _make_small_classical_teacher()
    m_dm = FrozenTrunkQuantumFrontEndGPT(
        teacher_dm,
        graft_cfg=QuantumFrontEndGraftConfig(
            wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
            state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
            wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ),
    )
    assert isinstance(m_dm.quantum_block0, ConfigurableWavepacketStatMechBlock)
    sc_ids = {id(p) for p in m_dm.quantum_scalar_parameters()}
    assert id(m_dm.quantum_block0.dm_res_mix) in sc_ids
    assert id(m_dm.quantum_block0.dm_lag_mix) in sc_ids

    _, loss_wf = m_wf(idx, targets=targets)
    _, loss_dm = m_dm(idx, targets=targets)
    assert loss_wf is not None and loss_dm is not None
    loss_dm.backward()

    tel_wf = m_wf.compute_quantum_graft_telemetry()
    tel_dm = m_dm.compute_quantum_graft_telemetry()
    assert tel_dm["q_l0_eff_rank"] > 1.5, f"Expected eff_rank > 1.5, got {tel_dm['q_l0_eff_rank']}"
    assert tel_dm["q_l0_eff_rank"] > tel_wf["q_l0_eff_rank"], (
        f"Expected CAUSAL_ENSEMBLE_DM_IO eff_rank ({tel_dm['q_l0_eff_rank']}) > WF_IN_WF_OUT ({tel_wf['q_l0_eff_rank']})"
    )
    assert m_dm.quantum_block0.dm_res_mix.grad is not None and float(m_dm.quantum_block0.dm_res_mix.grad.abs().item()) > 0.0
    assert m_dm.quantum_block0.dm_lag_mix.grad is not None and float(m_dm.quantum_block0.dm_lag_mix.grad.abs().item()) > 0.0


def test_h3_persistent_cosine_feat_and_ce_positive_floor() -> None:
    """Gate H3: Verify PERSISTENT_COSINE_FEAT_AND_CE keeps positive floor weight active for t >= interface_warmup_steps."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    teacher = _make_small_classical_teacher()

    cfg = QuantumFrontEndGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_24x2x16,
        state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
        interface_warmup_steps=10,
        interface_align_weight=1.5,
    )
    model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg)
    model.train()

    model.training_step = 25
    _, loss_persistent = model(idx, targets=targets)
    assert loss_persistent is not None

    model.graft_cfg.recovery_schedule_mode = QuantumRecoveryScheduleMode.PURE_CE
    _, loss_pure_ce = model(idx, targets=targets)
    assert loss_pure_ce is not None
    diff = float((loss_persistent - loss_pure_ce).item())
    assert diff > 1e-4, f"Expected persistent alignment floor > 0 at step 25, got diff={diff}"
    assert DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO == 0.25


def test_h4_monotonic_loss_descent_with_two_speed_riemannian_lr() -> None:
    """Gate H4: Verify 10-step loss descent with wte_lr=0.006 and stiefel_lr=0.035 while preserving manifold constraints."""
    torch.manual_seed(2026068)
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    teacher = _make_small_classical_teacher()

    cfg = QuantumFrontEndGraftConfig(
        wf_shape=QuantumWavefunctionShape.SHAPE_12x2x32,
        state_io_mode=QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO,
        attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
        replacement_scope=QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
        wte_init_mode=QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE,
        ve_coupling_mode=QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED,
        recovery_schedule_mode=QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE,
        interface_warmup_steps=20,
        interface_align_weight=1.5,
        wte_lr=0.006,
        stiefel_lr=0.035,
    )
    model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg)
    opt = build_frozen_trunk_quantum_frontend_optimizers(model)

    losses: list[float] = []
    for _ in range(10):
        opt.zero_grad()
        _, loss = model(idx, targets=targets)
        assert loss is not None and torch.isfinite(loss)
        losses.append(float(loss.detach().item()))
        loss.backward()
        opt.step(idx=idx, lr_mult=1.0)

    assert losses[-1] < losses[0], f"Expected loss descent over 10 steps: {losses[0]:.4f} -> {losses[-1]:.4f}"
    tel = model.compute_quantum_graft_telemetry()
    assert tel["q_max_stiefel_err"] < 1e-5, f"Stiefel err {tel['q_max_stiefel_err']}"
    assert tel["q_wte_sphere_err"] < 1e-5, f"Sphere err {tel['q_wte_sphere_err']}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
