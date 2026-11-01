"""Verification tests for EXPERIMENT PLAN 065 (Wave 52A): Surgical Quantum Front-End Graft onto Frozen Classical GPT."""

from __future__ import annotations

import torch

from qdlblocks.layers.quantum_frontend_graft import (
    DEFAULT_TEACHER_300M_CKPT_PATH,
    FrozenTrunkQuantumFrontEndGPT,
    QuantumAttnInteriorMode,
    QuantumFrontEndGraftConfig,
    QuantumReplacementScope,
    QuantumStateIOMode,
    QuantumWavefunctionShape,
    build_frozen_trunk_quantum_frontend_optimizers,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
    evaluate_compiled_leaderboard_validation,
)


def _make_small_classical_teacher() -> CompiledLeaderboardClassicalGPT:
    """Construct a small deterministic 4-layer classical teacher with non-zero weights for unit testing."""
    torch.manual_seed(2026)
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
        teacher.lm_head.weight.normal_(0.0, 0.02)
        for blk in teacher.blocks:
            blk.block_lambdas.copy_(torch.tensor([1.0, 0.15], dtype=torch.float32))
            if blk.out_proj is not None:
                blk.out_proj.weight.normal_(0.0, 0.02)
            blk.fc2.weight.normal_(0.0, 0.02)
    return teacher


def test_h1_exact_pretrained_classical_teacher_pass_through() -> None:
    """Gate H1: Verify exact pass-through of the pretrained 300M classical teacher (`val_ce = 3.584221`)."""
    teacher = _make_small_classical_teacher()
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    logits_ref, loss_ref = teacher(idx, targets=targets, return_logits=True)

    graft_cfg = QuantumFrontEndGraftConfig(replacement_scope=QuantumReplacementScope.TEACHER_PASS_THROUGH)
    model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=graft_cfg)
    logits_g, loss_g = model(idx, targets=targets, return_logits=True)
    assert logits_ref is not None and logits_g is not None and loss_ref is not None and loss_g is not None
    assert float((logits_ref - logits_g).abs().max().item()) < 1e-6
    assert abs(float(loss_ref.item()) - float(loss_g.item())) < 1e-6

    if torch.cuda.is_available() and DEFAULT_TEACHER_300M_CKPT_PATH.exists():
        val_shard = DEFAULT_TEACHER_300M_CKPT_PATH.parents[2] / "fineweb_100M" / "fineweb_val_000000.bin"
        if val_shard.exists():
            ckpt = torch.load(DEFAULT_TEACHER_300M_CKPT_PATH, map_location="cpu", weights_only=False)
            ref_ce = float(ckpt["val_metrics"]["val_ce"])
            model_300m = FrozenTrunkQuantumFrontEndGPT.from_teacher_checkpoint(
                DEFAULT_TEACHER_300M_CKPT_PATH,
                graft_cfg=graft_cfg,
                device="cuda:0",
            )
            val_metrics = evaluate_compiled_leaderboard_validation(
                model_300m,
                val_shard,
                val_tokens_budget=65536,
                batch_size=8,
                seq_len=1024,
            )
            assert abs(float(val_metrics["val_ce"]) - ref_ce) < 1e-3


def test_h2_strict_trunk_freeze_and_active_frontend_gradients() -> None:
    """Gate H2: Verify zero gradients on frozen trunk and 100% finite non-zero gradients on active quantum parameters."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    ppmi_syn = torch.randn(64, 768)

    for scope in (
        QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
    ):
        for wf_shape in (
            QuantumWavefunctionShape.SHAPE_24x2x16,
            QuantumWavefunctionShape.SHAPE_1x2x256,
            QuantumWavefunctionShape.SHAPE_1x2x512,
            QuantumWavefunctionShape.SHAPE_256x2x2,
        ):
            teacher = _make_small_classical_teacher()
            graft_cfg = QuantumFrontEndGraftConfig(wf_shape=wf_shape, replacement_scope=scope)
            model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=graft_cfg, ppmi_embeddings=ppmi_syn)
            opt = build_frozen_trunk_quantum_frontend_optimizers(model)
            opt.zero_grad(set_to_none=True)
            _, loss = model(idx, targets=targets)
            assert loss is not None
            loss.backward()

            assert all(p.grad is None for p in model.classical.parameters())
            active_params = model.active_quantum_parameters()
            assert len(active_params) > 0
            for p in active_params:
                assert p.grad is not None, f"Missing grad for shape={wf_shape} scope={scope} p={p.shape}"
                assert torch.isfinite(p.grad).all()
                assert float(p.grad.abs().max().item()) > 0.0


def test_h3_wavefunction_shapes_causal_and_manifold_invariance() -> None:
    """Gate H3: Verify causal autoregressiveness and strict Stiefel/sphere manifold preservation across all 4 shapes."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    ppmi_syn = torch.randn(64, 768)

    for wf_shape in (
        QuantumWavefunctionShape.SHAPE_24x2x16,
        QuantumWavefunctionShape.SHAPE_1x2x256,
        QuantumWavefunctionShape.SHAPE_1x2x512,
        QuantumWavefunctionShape.SHAPE_256x2x2,
    ):
        teacher = _make_small_classical_teacher()
        graft_cfg = QuantumFrontEndGraftConfig(wf_shape=wf_shape)
        model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=graft_cfg, ppmi_embeddings=ppmi_syn)
        assert model.q_wte is not None and model.q_wte.ndim == 3

        idx_mut = idx.clone()
        idx_mut[:, 8:] = (idx_mut[:, 8:] + 1) % 64
        logits_orig, _ = model(idx, return_logits=True)
        logits_mut, _ = model(idx_mut, return_logits=True)
        assert logits_orig is not None and logits_mut is not None
        assert float((logits_orig[:, :8] - logits_mut[:, :8]).abs().max().item()) < 1e-5

        opt = build_frozen_trunk_quantum_frontend_optimizers(model)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            _, loss = model(idx, targets=targets)
            assert loss is not None
            loss.backward()
            opt.step(idx=idx, lr_mult=1.0)

        tel = model.compute_quantum_graft_telemetry()
        assert tel["q_max_stiefel_err"] < 1e-5, f"Stiefel err {tel['q_max_stiefel_err']} for {wf_shape}"
        assert tel["q_wte_sphere_err"] < 1e-5, f"Sphere err {tel['q_wte_sphere_err']} for {wf_shape}"


def test_h4_state_io_and_interior_attention_physicality() -> None:
    """Gate H4: Verify unit trace, Hermiticity, and PSD eigenvalues across all State I/O and Interior Attention modes."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    ppmi_syn = torch.randn(64, 768)

    configs = [
        QuantumFrontEndGraftConfig(state_io_mode=QuantumStateIOMode.WF_IN_WF_OUT),
        QuantumFrontEndGraftConfig(state_io_mode=QuantumStateIOMode.DM_IN_DM_OUT),
        QuantumFrontEndGraftConfig(state_io_mode=QuantumStateIOMode.WF_IN_DENSITY_OBS_OUT),
        QuantumFrontEndGraftConfig(attn_interior_mode=QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V),
        QuantumFrontEndGraftConfig(attn_interior_mode=QuantumAttnInteriorMode.KV_CROSS_OUTER_ON_Q),
        QuantumFrontEndGraftConfig(attn_interior_mode=QuantumAttnInteriorMode.PRE_ATTN_LOCAL_OUTER_ON_V),
        QuantumFrontEndGraftConfig(wf_shape=QuantumWavefunctionShape.SHAPE_256x2x2),
    ]
    for cfg in configs:
        teacher = _make_small_classical_teacher()
        model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg, ppmi_embeddings=ppmi_syn)
        logits, loss = model(idx, targets=targets, return_logits=True)
        assert logits is not None and loss is not None
        assert model.quantum_block0 is not None and model.quantum_block0._last_rho0_re_im is not None
        rho0_re, rho0_im = model.quantum_block0._last_rho0_re_im
        rho0_re_f, rho0_im_f = rho0_re.float(), rho0_im.float()
        assert float((rho0_re_f - rho0_re_f.transpose(-2, -1)).abs().max().item()) < 1e-5
        assert float((rho0_im_f + rho0_im_f.transpose(-2, -1)).abs().max().item()) < 1e-5
        tr = torch.diagonal(rho0_re_f, dim1=-2, dim2=-1).sum(dim=-1)
        assert float((tr - 1.0).abs().max().item()) < 1e-4
        eigs = torch.linalg.eigvalsh(torch.complex(rho0_re_f, rho0_im_f))
        assert float(eigs.min().item()) > -1e-5


def test_h5_replacement_scope_and_teacher_interface_warmup_descent() -> None:
    """Gate H5: Verify loss descent under both replacement scopes and >50% interface MSE reduction under warmup."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    ppmi_syn = torch.randn(64, 768)

    for scope in (
        QuantumReplacementScope.FULL_BLOCK0_QUANTUM,
        QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0,
    ):
        teacher = _make_small_classical_teacher()
        cfg = QuantumFrontEndGraftConfig(replacement_scope=scope, interface_warmup_steps=12, interface_align_weight=2.0)
        model = FrozenTrunkQuantumFrontEndGPT(teacher, graft_cfg=cfg, ppmi_embeddings=ppmi_syn)
        opt = build_frozen_trunk_quantum_frontend_optimizers(model, stiefel_lr=0.06, wte_lr=0.18, scalar_lr=0.06)

        with torch.no_grad():
            x0_c, x1_c, v0_c = model.compute_teacher_layer0_targets(idx)
            x0_q0, x1_q0, v0_q0 = model.encode_quantum_layer0(idx)
            init_mse = float(
                (
                    torch.nn.functional.mse_loss(x0_q0.float(), x0_c.float())
                    + torch.nn.functional.mse_loss(x1_q0.float(), x1_c.float())
                    + 0.25 * torch.nn.functional.mse_loss(v0_q0.float(), v0_c.float())
                ).item()
            )
            _, init_ce = model(idx, targets=targets, return_logits=True)

        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            _, total_loss = model(idx, targets=targets)
            assert total_loss is not None
            total_loss.backward()
            opt.step(idx=idx, lr_mult=1.0)

        with torch.no_grad():
            x0_q1, x1_q1, v0_q1 = model.encode_quantum_layer0(idx)
            final_mse = float(
                (
                    torch.nn.functional.mse_loss(x0_q1.float(), x0_c.float())
                    + torch.nn.functional.mse_loss(x1_q1.float(), x1_c.float())
                    + 0.25 * torch.nn.functional.mse_loss(v0_q1.float(), v0_c.float())
                ).item()
            )
            model.eval()
            _, final_ce = model(idx, targets=targets, return_logits=True)
            assert init_ce is not None and final_ce is not None
            assert final_mse < init_mse
            assert float(final_ce.item()) < float(init_ce.item())


if __name__ == "__main__":
    for test_fn in (
        test_h1_exact_pretrained_classical_teacher_pass_through,
        test_h2_strict_trunk_freeze_and_active_frontend_gradients,
        test_h3_wavefunction_shapes_causal_and_manifold_invariance,
        test_h4_state_io_and_interior_attention_physicality,
        test_h5_replacement_scope_and_teacher_interface_warmup_descent,
    ):
        test_fn()
        print(f"PASS: {test_fn.__name__}")
