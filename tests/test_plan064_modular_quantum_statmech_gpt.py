"""Unit & Integration Tests for `EXPERIMENT_PLAN_064.md` (Wave 49: Modular Quantum Statistical-Mechanics GPT)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qdlblocks.layers.isometric_born_trace_readout import (
    MeanCenteredIsometricBornTraceCEFn,
    VocabularyReadoutMode,
    eval_mean_centered_isometric_trace_logits,
    evaluate_readout_divergence_batch,
)
from qdlblocks.layers.quantum_statmech_channels import (
    EntropyGateMode,
    PurificationMode,
    QuantumChannelMode,
    StiefelMomentumMode,
    initialize_complex_kraus_stiefel_pair,
    purify_density_statmech,
    riemannian_complex_kraus_stiefel_muon_step,
)
from qdlblocks.models.modular_quantum_statmech_gpt import (
    EmbeddingTrainMode,
    ModularQuantumStatMechConfig,
    ModularQuantumStatMechGPT,
    build_modular_quantum_statmech_optimizers,
)
from qdlblocks.models.standalone_quantum_born_gpt import (
    StandaloneQuantumBornConfig,
    StandaloneQuantumBornGPT,
    VocabularyReadoutMode as StandaloneReadoutMode,
    build_standalone_quantum_born_optimizers,
)


def test_h1_exact_refactoring_equivalence_with_standalone_v3() -> None:
    """Gate H1: Verify `ModularQuantumStatMechGPT` reproduces `StandaloneQuantumBornGPT` (`Wave 48 v3`) to `< 1e-6`."""
    vocab_size, n_layer, n_head, n_embd = 256, 2, 2, 128
    torch.manual_seed(1337)
    cfg_orig = StandaloneQuantumBornConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        readout_mode=StandaloneReadoutMode.SOFTCAP_LINEAR_CE,
    )
    m_orig = StandaloneQuantumBornGPT(cfg_orig)
    opt_orig = build_standalone_quantum_born_optimizers(m_orig)

    torch.manual_seed(1337)
    cfg_mod = ModularQuantumStatMechConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
    )
    m_mod = ModularQuantumStatMechGPT(cfg_mod)
    opt_mod = build_modular_quantum_statmech_optimizers(m_mod, momentum_mode=StiefelMomentumMode.EUCLIDEAN_AMBIENT)

    idx = torch.randint(0, vocab_size, (2, 16))
    targets = torch.randint(0, vocab_size, (2, 16))

    logits_orig, loss_orig = m_orig(idx, targets=targets, return_logits=True)
    logits_mod, loss_mod = m_mod(idx, targets=targets, return_logits=True)
    assert loss_orig is not None and loss_mod is not None
    assert logits_orig is not None and logits_mod is not None
    assert float((loss_orig - loss_mod).abs().item()) < 1e-6
    assert float((logits_orig - logits_mod).abs().max().item()) < 1e-5

    opt_orig.zero_grad(set_to_none=True)
    loss_orig.backward()
    opt_orig.step(idx=idx, targets=targets, lr_scale=1.0)

    opt_mod.zero_grad(set_to_none=True)
    loss_mod.backward()
    opt_mod.step(idx=idx, targets=targets, lr_scale=1.0)

    _, loss_orig_step1 = m_orig(idx, targets=targets)
    _, loss_mod_step1 = m_mod(idx, targets=targets)
    assert loss_orig_step1 is not None and loss_mod_step1 is not None
    assert float((loss_orig_step1 - loss_mod_step1).abs().item()) < 1e-5


def test_h2_mean_centered_isometric_born_trace_single_pass_autograd() -> None:
    """Gate H2: Verify `MeanCenteredIsometricBornTraceCEFn` matches analytical PyTorch reference gradients."""
    torch.manual_seed(2026)
    n_tok, d_model, vocab_size, n_heads, d_h = 12, 64, 48, 2, 8
    p_dim = n_heads * d_h * d_h

    h_flat = torch.randn(n_tok, d_model, dtype=torch.float32, requires_grad=True)
    p_rho_flat = torch.randn(n_tok, p_dim, dtype=torch.float32, requires_grad=True)
    w_lm_scaled = (torch.randn(vocab_size, d_model, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    p_e_vocab = (torch.randn(vocab_size, p_dim, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    trace_beta = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, vocab_size, (n_tok,))

    loss_fn = MeanCenteredIsometricBornTraceCEFn.apply(
        h_flat, p_rho_flat, w_lm_scaled, p_e_vocab, targets, trace_beta, n_heads
    )
    loss_fn.backward()

    h_ref = h_flat.detach().clone().requires_grad_(True)
    pr_ref = p_rho_flat.detach().clone().requires_grad_(True)
    w_ref = w_lm_scaled.detach().clone().requires_grad_(True)
    pe_ref = p_e_vocab.detach().clone().requires_grad_(True)
    beta_ref = trace_beta.detach().clone().requires_grad_(True)

    logits_ref = eval_mean_centered_isometric_trace_logits(
        h_ref.unsqueeze(0), pr_ref.unsqueeze(0), w_ref, pe_ref, beta_ref, num_heads=n_heads
    ).squeeze(0)
    loss_ref = F.cross_entropy(logits_ref, targets)
    loss_ref.backward()

    assert float((loss_fn - loss_ref).abs().item()) < 1e-6
    for g_a, g_b in [
        (h_flat.grad, h_ref.grad),
        (p_rho_flat.grad, pr_ref.grad),
        (w_lm_scaled.grad, w_ref.grad),
        (p_e_vocab.grad, pe_ref.grad),
        (trace_beta.grad, beta_ref.grad),
    ]:
        assert g_a is not None and g_b is not None
        cos_sim = float(F.cosine_similarity(g_a.flatten(), g_b.flatten(), dim=0).item())
        assert cos_sim > 0.9999


def test_h3_cptp_kraus_channel_stiefel_preservation() -> None:
    """Gate H3: Verify Rank-K CPTP Kraus channel preserves $\\sum_k A_{h,k}^\\dagger A_{h,k} = I_{d_h}$ and trace-1 density."""
    torch.manual_seed(77)
    n_heads, d_h, kraus_rank = 4, 8, 4
    kr_re, kr_im = initialize_complex_kraus_stiefel_pair(n_heads, d_h, kraus_rank=kraus_rank)
    mom_re = torch.zeros_like(kr_re)
    mom_im = torch.zeros_like(kr_im)

    for _ in range(5):
        grad_re = torch.randn_like(kr_re)
        grad_im = torch.randn_like(kr_im)
        riemannian_complex_kraus_stiefel_muon_step(kr_re, kr_im, grad_re, grad_im, mom_re, mom_im, lr=0.03)

    eye = torch.eye(d_h).unsqueeze(0)
    gram_re = torch.matmul(kr_re.transpose(-2, -1), kr_re) + torch.matmul(kr_im.transpose(-2, -1), kr_im)
    gram_im = torch.matmul(kr_re.transpose(-2, -1), kr_im) - torch.matmul(kr_im.transpose(-2, -1), kr_re)
    err = float(((gram_re - eye).square().sum() + gram_im.square().sum()).sqrt().item()) / (n_heads * d_h)
    assert err < 1e-5


def test_h4_purification_modes_and_tsallis_gate_and_embedding_modes() -> None:
    """Gate H4: Verify Quadratic, Cubic Fermi-Dirac, Learnable Temp purification, Detached Tsallis Gate, and Embedding modes."""
    torch.manual_seed(99)
    bsz, n_heads, seq_len, d_h = 2, 3, 8, 8
    a_re = torch.randn(bsz, n_heads, seq_len, 4, d_h)
    a_im = torch.randn(bsz, n_heads, seq_len, 4, d_h)
    rho_re = torch.matmul(a_re.transpose(-2, -1), a_re) + torch.matmul(a_im.transpose(-2, -1), a_im)
    rho_im = torch.matmul(a_re.transpose(-2, -1), a_im) - torch.matmul(a_im.transpose(-2, -1), a_re)

    logits_h3 = torch.zeros(n_heads, 3)
    for mode in (PurificationMode.QUADRATIC_MCWEENY, PurificationMode.CUBIC_FERMI_DIRAC, PurificationMode.LEARNABLE_TEMP_MCWEENY):
        out_re, _, purity, eff_rank = purify_density_statmech(rho_re, rho_im, mode=mode, purify_logits_h3=logits_h3)
        traces = torch.diagonal(out_re, dim1=-2, dim2=-1).sum(dim=-1)
        assert float((traces - 1.0).abs().max().item()) < 1e-5
        assert bool((purity >= 1.0 / d_h - 1e-4).all() and (purity <= 1.0 + 1e-4).all())
        assert bool((eff_rank >= 1.0 - 1e-4).all() and (eff_rank <= d_h + 1e-4).all())

    cfg = ModularQuantumStatMechConfig(
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=128,
        purification_mode=PurificationMode.LEARNABLE_TEMP_MCWEENY,
        channel_mode=QuantumChannelMode.CPTP_KRAUS_RANK_K,
        entropy_gate_mode=EntropyGateMode.DETACHED_TSALLIS_PURITY_GATE,
        readout_mode=VocabularyReadoutMode.MEAN_CENTERED_ISOMETRIC_TRACE,
    )
    model = ModularQuantumStatMechGPT(cfg, kraus_rank=4)
    model.configure_embedding_mode(EmbeddingTrainMode.FROZEN_RANDOM_SPHERE)
    assert not model.wte.weight.requires_grad

    opt = build_modular_quantum_statmech_optimizers(model)
    idx = torch.randint(0, 128, (2, 12))
    targets = torch.randint(0, 128, (2, 12))
    wte_before = model.wte.weight.detach().clone()

    logits, loss = model(idx, targets=targets, return_logits=True)
    assert loss is not None and logits is not None
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(idx=idx, targets=targets, lr_scale=1.0)
    assert float((model.wte.weight - wte_before).abs().max().item()) == 0.0

    div = evaluate_readout_divergence_batch(logits, targets)
    assert "val_ce" in div and "gap_top5_top1" in div and "nll_rank2_5" in div and "nll_rank6_50" in div
    tel = model.compute_manifold_and_purity_telemetry()
    assert tel["max_stiefel_err"] < 1e-5 and tel["max_unitary_err"] < 1e-5


def test_h5_riemannian_tangent_transported_momentum_and_vector_diagnostics() -> None:
    """Gate H5: Verify `RIEMANNIAN_TANGENT_TRANSPORTED` eliminates off-manifold normal momentum accumulation and leakage."""
    cfg = ModularQuantumStatMechConfig(vocab_size=128, n_layer=2, n_head=2, n_embd=128)
    idx = torch.randint(0, 128, (2, 12))
    targets = torch.randint(0, 128, (2, 12))

    torch.manual_seed(42)
    m_eucl = ModularQuantumStatMechGPT(cfg)
    opt_eucl = build_modular_quantum_statmech_optimizers(m_eucl, momentum_mode=StiefelMomentumMode.EUCLIDEAN_AMBIENT)
    for step_i in range(3):
        if step_i == 2:
            opt_eucl.trunk_opt.collect_diagnostics = True
        opt_eucl.zero_grad(set_to_none=True)
        _, loss = m_eucl(idx, targets=targets)
        assert loss is not None
        loss.backward()
        opt_eucl.step(idx=idx, targets=targets, lr_scale=1.0)
    diag_eucl = opt_eucl.trunk_opt.get_last_step_diagnostics()

    torch.manual_seed(42)
    m_riem = ModularQuantumStatMechGPT(cfg)
    opt_riem = build_modular_quantum_statmech_optimizers(
        m_riem, momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED
    )
    for step_i in range(3):
        if step_i == 2:
            opt_riem.trunk_opt.collect_diagnostics = True
        opt_riem.zero_grad(set_to_none=True)
        _, loss = m_riem(idx, targets=targets)
        assert loss is not None
        loss.backward()
        tan_norm = m_riem.clip_riemannian_tangent_grad_norm_(1.0)
        assert tan_norm > 0.0
        opt_riem.step(idx=idx, targets=targets, lr_scale=1.0)
    diag_riem = opt_riem.trunk_opt.get_last_step_diagnostics()

    assert diag_eucl["pre_mom_normal_frac"] > 0.01
    assert diag_eucl["post_mom_normal_at_w_next_frac"] > 0.01
    assert diag_riem["pre_mom_normal_frac"] < 1e-5
    assert diag_riem["post_mom_normal_at_w_next_frac"] < 1e-5
    assert diag_riem["unitary_post_mom_normal_frac"] < 1e-5
    assert diag_riem["grad_skew_tangent_frac"] > 0.0
    assert diag_riem["normal_to_tangent_leakage_ratio"] < 1e-5
    assert diag_riem["cos_utan_retract"] > 0.99


def test_h6_isometric_normalized_branches_and_column_scaled_stiefel() -> None:
    """Gate H6: Verify `ISOMETRIC_NORMALIZED_BRANCHES` preserves forward output while cutting normal gradients, and `RIEMANNIAN_COLUMN_SCALED_STIEFEL` harvests column scales."""
    idx = torch.randint(0, 128, (2, 64))
    targets = torch.randint(0, 128, (2, 64))

    torch.manual_seed(108)
    m_base = ModularQuantumStatMechGPT(
        ModularQuantumStatMechConfig(vocab_size=128, n_layer=2, n_head=2, n_embd=128, entropy_gate_mode=EntropyGateMode.NONE)
    )
    opt_base = build_modular_quantum_statmech_optimizers(
        m_base, momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED
    )
    opt_base.trunk_opt.collect_diagnostics = True
    opt_base.zero_grad(set_to_none=True)
    logits_base, loss_base = m_base(idx, targets=targets, return_logits=True)
    assert loss_base is not None and logits_base is not None
    loss_base.backward()
    opt_base.step(idx=idx, targets=targets, lr_scale=1.0)
    diag_base = opt_base.trunk_opt.get_last_step_diagnostics()

    torch.manual_seed(108)
    m_clean = ModularQuantumStatMechGPT(
        ModularQuantumStatMechConfig(
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_embd=128,
            entropy_gate_mode=EntropyGateMode.ISOMETRIC_NORMALIZED_BRANCHES,
        )
    )
    opt_clean = build_modular_quantum_statmech_optimizers(
        m_clean, momentum_mode=StiefelMomentumMode.RIEMANNIAN_COLUMN_SCALED_STIEFEL
    )
    for step_i in range(3):
        if step_i == 0:
            opt_clean.trunk_opt.collect_diagnostics = True
        opt_clean.zero_grad(set_to_none=True)
        logits_clean, loss_clean = m_clean(idx, targets=targets, return_logits=(step_i == 0))
        assert loss_clean is not None
        if step_i == 0:
            assert logits_clean is not None
            assert float((loss_base - loss_clean).abs().item()) < 1e-5
            assert float((logits_base - logits_clean).abs().max().item()) < 1e-4
        loss_clean.backward()
        opt_clean.step(idx=idx, targets=targets, lr_scale=1.0)
        if step_i == 0:
            diag_clean = opt_clean.trunk_opt.get_last_step_diagnostics()

    assert diag_clean["unitary_grad_normal_frac"] < diag_base["unitary_grad_normal_frac"] - 0.10
    assert diag_clean["unitary_grad_normal_frac"] < 0.73
    tel_clean = m_clean.compute_manifold_and_purity_telemetry()
    assert tel_clean["max_stiefel_err"] < 1e-5
    assert tel_clean["max_unitary_err"] < 1e-5
    assert tel_clean["mean_stiefel_col_scale_std"] > 1e-4


def test_h7_unnormalized_final_readout_and_final_block() -> None:
    """Gate H7: Verify `UNNORMALIZED_FINAL_READOUT_HEAD` (Level A) and `UNNORMALIZED_FINAL_BLOCK_AND_HEAD` (Level B)."""
    idx = torch.randint(0, 128, (2, 32))
    targets = torch.randint(0, 128, (2, 32))
    cfg = ModularQuantumStatMechConfig(vocab_size=128, n_layer=2, n_head=2, n_embd=128)

    for mode in (
        StiefelMomentumMode.UNNORMALIZED_FINAL_READOUT_HEAD,
        StiefelMomentumMode.UNNORMALIZED_FINAL_BLOCK_AND_HEAD,
    ):
        torch.manual_seed(2026)
        model = ModularQuantumStatMechGPT(cfg)
        opt = build_modular_quantum_statmech_optimizers(model, momentum_mode=mode)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            _, loss = model(idx, targets=targets)
            assert loss is not None
            loss.backward()
            model.clip_riemannian_tangent_grad_norm_(1.0)
            opt.step(idx=idx, targets=targets, lr_scale=1.0)
        tel = model.compute_manifold_and_purity_telemetry()
        assert tel["max_stiefel_err"] < 1e-5
        assert tel["max_unitary_err"] < 1e-5
        assert tel["max_head_sphere_rms_err"] < 1e-5
        assert tel["head_row_scale_std"] > 1e-4


def test_h8_causal_wavepacket_48x16_embedding() -> None:
    """Gate H8: Verify `(V, 48, 16)` product-sphere and causal wavepacket embeddings + strict causality."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    cfg = ModularQuantumStatMechConfig(vocab_size=64, n_layer=1, n_head=6, n_embd=768)

    for emb_mode in (EmbeddingTrainMode.WAVEPACKET_SPHERE_48X16, EmbeddingTrainMode.CAUSAL_WAVEPACKET_48X16):
        torch.manual_seed(2026)
        model = ModularQuantumStatMechGPT(cfg)
        model.configure_embedding_mode(emb_mode)
        assert model.wte.weight.shape == (64, 48, 16)

        if emb_mode == EmbeddingTrainMode.CAUSAL_WAVEPACKET_48X16:
            idx_mut = idx.clone()
            idx_mut[:, 8:] = (idx_mut[:, 8:] + 1) % 64
            logits_orig, _ = model(idx, return_logits=True)
            logits_mut, _ = model(idx_mut, return_logits=True)
            assert logits_orig is not None and logits_mut is not None
            assert float((logits_orig[:, :8] - logits_mut[:, :8]).abs().max().item()) < 1e-5

        opt = build_modular_quantum_statmech_optimizers(
            model, momentum_mode=StiefelMomentumMode.UNNORMALIZED_FINAL_READOUT_HEAD
        )
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            _, loss = model(idx, targets=targets)
            assert loss is not None
            loss.backward()
            model.clip_riemannian_tangent_grad_norm_(1.0)
            opt.step(idx=idx, targets=targets, lr_scale=1.0)

        assert model.wte.weight.shape == (64, 48, 16)
        tel = model.compute_manifold_and_purity_telemetry()
        assert tel["max_wte_sphere_rms_err"] < 1e-5
        assert tel["max_stiefel_err"] < 1e-5


def test_h9_native_wavepacket_48x16_layer_architecture() -> None:
    """Gate H9: Verify native `(B, T, 48, 16)` self-attention + wavepacket density evolution across all layers."""
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    cfg = ModularQuantumStatMechConfig(vocab_size=64, n_layer=2, n_head=6, n_embd=768)

    for emb_mode in (EmbeddingTrainMode.NATIVE_WAVEPACKET_48X16, EmbeddingTrainMode.NATIVE_CAUSAL_WAVEPACKET_48X16):
        torch.manual_seed(2026)
        model = ModularQuantumStatMechGPT(cfg)
        model.configure_embedding_mode(emb_mode)
        assert model.is_native_wavepacket_mode
        assert model.wte.weight.shape == (64, 48, 16)
        assert model.w_in_pkt is not None and model.w_in_pkt.shape == (48, 16, 16)
        assert model.w_in_mode is not None and model.w_in_mode.shape == (16, 48, 48)

        idx_mut = idx.clone()
        idx_mut[:, 8:] = (idx_mut[:, 8:] + 1) % 64
        logits_orig, _ = model(idx, return_logits=True)
        logits_mut, _ = model(idx_mut, return_logits=True)
        assert logits_orig is not None and logits_mut is not None
        assert float((logits_orig[:, :8] - logits_mut[:, :8]).abs().max().item()) < 1e-5

        opt = build_modular_quantum_statmech_optimizers(
            model, momentum_mode=StiefelMomentumMode.UNNORMALIZED_FINAL_READOUT_HEAD
        )
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            _, loss = model(idx, targets=targets)
            assert loss is not None
            loss.backward()
            model.clip_riemannian_tangent_grad_norm_(1.0)
            opt.step(idx=idx, targets=targets, lr_scale=1.0)

        tel = model.compute_manifold_and_purity_telemetry()
        assert tel["max_wte_sphere_rms_err"] < 1e-5
        assert tel["max_stiefel_err"] < 1e-5
        assert tel["max_unitary_err"] < 1e-5
        assert tel["mean_stiefel_col_scale_std"] > 1e-5
        assert tel["head_row_scale_std"] > 1e-4


def test_h10_hybrid_quantum_frontend_classical_gpt() -> None:
    """Gate H10: Verify HybridQuantumFrontEndClassicalGPT across all quantum front-end modes."""
    from qdlblocks.layers.wavepacket_statmech_block import (
        ClassicalQuantumFrontEndMode,
        HybridQuantumFrontEndClassicalGPT,
        build_hybrid_quantum_classical_optimizers,
    )
    from qdlblocks.models.compiled_leaderboard_classical_gpt import CompiledLeaderboardClassicalConfig

    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    c_cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=64,
        embed_dim=768,
        num_heads=6,
        num_layers=2,
        hidden_dim=1536,
        max_seq_len=16,
        num_value_embeddings=0,
        use_flex_doc_swc=False,
        bfloat16_weights=False,
    )

    for f_mode in (
        ClassicalQuantumFrontEndMode.PURE_CLASSICAL,
        ClassicalQuantumFrontEndMode.QUANTUM_WAVEPACKET_EMB_ONLY,
        ClassicalQuantumFrontEndMode.NATIVE_WAVEPACKET_EMB_AND_L0,
        ClassicalQuantumFrontEndMode.FLAT_STATMECH_EMB_AND_L0,
    ):
        torch.manual_seed(2026)
        model = HybridQuantumFrontEndClassicalGPT(c_cfg, frontend_mode=f_mode)
        idx_mut = idx.clone()
        idx_mut[:, 8:] = (idx_mut[:, 8:] + 1) % 64
        logits_orig, _ = model(idx, return_logits=True)
        logits_mut, _ = model(idx_mut, return_logits=True)
        assert logits_orig is not None and logits_mut is not None
        assert float((logits_orig[:, :8] - logits_mut[:, :8]).abs().max().item()) < 1e-5

        opt = build_hybrid_quantum_classical_optimizers(model)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            _, loss = model(idx, targets=targets)
            assert loss is not None
            loss.backward()
            opt.step(idx=idx, lr_mult=1.0)

        if f_mode != ClassicalQuantumFrontEndMode.PURE_CLASSICAL:
            q_tel = model.compute_quantum_frontend_telemetry()
            assert q_tel["q_max_stiefel_err"] < 1e-5
            assert q_tel["q_wte_sphere_err"] < 1e-5


def test_h11_multi_token_ensemble_density_aggregation() -> None:
    """Gate H11: Verify zero-packing $16 \\times 16$ complex outer-product attention (`D=128`) preserves exact Hermiticity, unit trace, PSD, and multi-token rank."""
    from qdlblocks.layers.wavepacket_statmech_block import (
        DEFAULT_WAVEPACKET_SDPA_HEAD_DIM,
        NativeWavepacketStatMechBlock,
        form_wavepacket_outer_product_and_v,
        wavepacket_statmech_purify_evolve_impl,
    )

    torch.manual_seed(2026)
    cfg = ModularQuantumStatMechConfig(vocab_size=64, n_layer=1, n_head=6, n_embd=768)
    blk = NativeWavepacketStatMechBlock(cfg, layer_idx=0)
    assert not hasattr(blk, "re_pack_mask") and not hasattr(blk, "im_pack_mask")
    assert blk.w_obs_pkt.shape == (48, 16, 16)
    assert blk.m_obs_mode.shape == (16, 48, 48)

    bsz, seq_len = 2, 32
    v = torch.randn(bsz, seq_len, 48, 16)
    sdpa_v = form_wavepacket_outer_product_and_v(v, num_heads=6, v_rank=4)
    assert sdpa_v.shape == (bsz, seq_len, 30, DEFAULT_WAVEPACKET_SDPA_HEAD_DIM)

    q_h = torch.randn(bsz, 6, 1, seq_len, DEFAULT_WAVEPACKET_SDPA_HEAD_DIM)
    k_h = torch.randn(bsz, 6, 1, seq_len, DEFAULT_WAVEPACKET_SDPA_HEAD_DIM)
    sdpa_q = q_h.expand(bsz, 6, 5, seq_len, DEFAULT_WAVEPACKET_SDPA_HEAD_DIM).reshape(
        bsz, 30, seq_len, DEFAULT_WAVEPACKET_SDPA_HEAD_DIM
    )
    sdpa_k = k_h.expand(bsz, 6, 5, seq_len, DEFAULT_WAVEPACKET_SDPA_HEAD_DIM).reshape(
        bsz, 30, seq_len, DEFAULT_WAVEPACKET_SDPA_HEAD_DIM
    )
    a_all = torch.nn.functional.scaled_dot_product_attention(
        sdpa_q, sdpa_k, sdpa_v.permute(0, 2, 1, 3), is_causal=True, scale=0.22
    ).view(bsz, 6, 5, seq_len, DEFAULT_WAVEPACKET_SDPA_HEAD_DIM)

    a_rho = a_all[:, :, :4].permute(0, 1, 3, 2, 4).reshape(bsz, 6, seq_len, 4, 2, 4, 16)
    rho0_re = a_rho[:, :, :, :, 0].reshape(bsz, 6, seq_len, 16, 16)
    rho0_im = a_rho[:, :, :, :, 1].reshape(bsz, 6, seq_len, 16, 16)
    assert float((rho0_re - rho0_re.transpose(-2, -1)).abs().max().item()) < 1e-6
    assert float((rho0_im + rho0_im.transpose(-2, -1)).abs().max().item()) < 1e-6
    assert float((torch.diagonal(rho0_re, dim1=-2, dim2=-1).sum(-1) - 1.0).abs().max().item()) < 1e-5
    eigs = torch.linalg.eigvalsh(torch.complex(rho0_re, rho0_im))
    assert float(eigs.min().item()) > -1e-6

    a_out_48 = a_all[:, :, 4].permute(0, 2, 1, 3).reshape(bsz, seq_len, 48, 16)
    modes = (cfg.purification_mode.value, cfg.channel_mode.value, cfg.entropy_gate_mode.value)

    y_coh, pur_coh, er_coh, _, _ = wavepacket_statmech_purify_evolve_impl(
        a_out_48, blk.head_unitary_re, blk.head_unitary_im, blk.purify_logits, blk.entropy_gate_params, modes
    )
    y_ens, pur_ens, er_ens, _, _ = wavepacket_statmech_purify_evolve_impl(
        a_all, blk.head_unitary_re, blk.head_unitary_im, blk.purify_logits, blk.entropy_gate_params, modes
    )
    assert y_coh.shape == (bsz, seq_len, 48, 16) and y_ens.shape == (bsz, seq_len, 48, 16)
    assert float(er_ens.mean().item()) > float(er_coh.mean().item()) + 1.0
    assert float(pur_ens.mean().item()) < float(pur_coh.mean().item())


def test_h12_ppmi_wavepacket_sphere_initialization() -> None:
    """Gate H12: Verify Spectral-Interleaved + Ortho-48 Balanced PPMI-SVD initialization on `(V, 48, 16)`."""
    from qdlblocks.layers.wavepacket_statmech_block import initialize_wavepacket_ppmi_sphere_embeddings_

    torch.manual_seed(2026)
    vocab_size, num_pkts, d_h = 128, 48, 16
    sigma = torch.linspace(20.0, 1.0, num_pkts * d_h)
    u_rand, _ = torch.linalg.qr(torch.randn(vocab_size, vocab_size))
    v_rand, _ = torch.linalg.qr(torch.randn(num_pkts * d_h, num_pkts * d_h))
    ppmi_syn = (u_rand[:, :128] @ torch.diag(sigma[:128]) @ v_rand[:128, :]).contiguous()

    wte_3d = torch.empty(vocab_size, num_pkts, d_h)
    used_ppmi = initialize_wavepacket_ppmi_sphere_embeddings_(wte_3d, ppmi_embeddings=ppmi_syn)
    assert used_ppmi
    rms_err = float((wte_3d.pow(2).mean(dim=-1).sqrt() - 1.0).abs().max().item())
    assert rms_err < 1e-5

    u_ref = torch.nn.functional.normalize(ppmi_syn, dim=-1)
    u_bal = torch.nn.functional.normalize(wte_3d.view(vocab_size, -1), dim=-1)
    gram_rmse_bal = float((u_bal @ u_bal.T - u_ref @ u_ref.T).pow(2).mean().sqrt().item())
    assert gram_rmse_bal < 0.05


if __name__ == "__main__":
    for test_fn in (
        test_h1_exact_refactoring_equivalence_with_standalone_v3,
        test_h2_mean_centered_isometric_born_trace_single_pass_autograd,
        test_h3_cptp_kraus_channel_stiefel_preservation,
        test_h4_purification_modes_and_tsallis_gate_and_embedding_modes,
        test_h5_riemannian_tangent_transported_momentum_and_vector_diagnostics,
        test_h6_isometric_normalized_branches_and_column_scaled_stiefel,
        test_h7_unnormalized_final_readout_and_final_block,
        test_h8_causal_wavepacket_48x16_embedding,
        test_h9_native_wavepacket_48x16_layer_architecture,
        test_h10_hybrid_quantum_frontend_classical_gpt,
        test_h11_multi_token_ensemble_density_aggregation,
        test_h12_ppmi_wavepacket_sphere_initialization,
    ):
        test_fn()
        print(f"PASS: {test_fn.__name__}")
