"""Unit tests for the small-scale FineWeb Span Reversal initialization and readout laboratory."""

from __future__ import annotations

import math

import torch

from qdlblocks.benchmarks.micro_reversal_readout_lab import (
    FineWebSpanReversalDataset,
    MicroBackboneMode,
    MicroInitMode,
    MicroReadoutMode,
    MicroReversalConfig,
    MicroReversalGPT,
    compute_gpu_countsketch_bigram_sphere_embeddings,
    compute_gpu_unigram_log_prior_and_radial,
)
from qdlblocks.benchmarks.support_preserving_countsketch_lab import (
    SupportSketchMode,
    SupportSketchSpec,
    _hermitian_gram_schmidt_complex_blend,
    compute_complex_wavefunction_split_sphere,
    compute_split_meaning_id_wavepacket_sphere,
    compute_support_gated_countsketch_sphere,
    compute_whitened_support_countsketch_sphere,
    measure_sphere_geometry_and_crowding,
    project_complex_wavepackets_to_rms_sphere_,
)


def test_gpu_unigram_and_countsketch_sphere_invariants() -> None:
    """Verify that GPU Unigram log-prior is zero-mean and CountSketch embeddings lie on the unit-RMS sphere."""
    tokens = torch.randint(0, 128, (8192,), dtype=torch.long)
    log_prior, radial = compute_gpu_unigram_log_prior_and_radial(tokens, vocab_size=128, rev_token_id=127)
    assert log_prior.shape == (128,)
    assert radial.shape == (128,)
    assert abs(float(log_prior.mean().item())) < 1e-5
    assert bool((radial >= 0.40).all() and (radial <= 2.50).all())

    emb_3d = compute_gpu_countsketch_bigram_sphere_embeddings(tokens, vocab_size=128, pkt_count=4, pkt_dim=16)
    assert emb_3d.shape == (128, 4, 16)
    pkt_rms = emb_3d.pow(2).mean(dim=-1).sqrt()
    assert float((pkt_rms - 1.0).abs().max().item()) < 1e-5


def test_support_preserving_countsketch_eliminates_rare_token_collapse() -> None:
    """Verify that I5, I6, I7 prevent unseen tokens from collapsing onto a single antipodal mean vector."""
    # Only tokens 0..63 appear in `tokens`; tokens 64..255 are unseen (count == 0)
    tokens = torch.randint(0, 64, (8192,), dtype=torch.long)
    raw_emb = compute_gpu_countsketch_bigram_sphere_embeddings(tokens, vocab_size=256, pkt_count=4, pkt_dim=16)
    raw_geom = measure_sphere_geometry_and_crowding(raw_emb, tokens, vocab_size=256)
    assert raw_geom["rare_max_cos"] > 0.95

    for mode, fn in (
        (SupportSketchMode.I5_SUPPORT_GATED_COUNTSKETCH, compute_support_gated_countsketch_sphere),
        (SupportSketchMode.I6_SPLIT_MEANING_ID_WAVEPACKET, compute_split_meaning_id_wavepacket_sphere),
        (SupportSketchMode.I7_WHITENED_SUPPORT_COUNTSKETCH, compute_whitened_support_countsketch_sphere),
    ):
        spec = SupportSketchSpec(mode=mode, vocab_size=256, pkt_count=4, pkt_dim=16)
        emb = fn(tokens, spec)
        pkt_rms = emb.pow(2).mean(dim=-1).sqrt()
        assert float((pkt_rms - 1.0).abs().max().item()) < 1e-5
        geom = measure_sphere_geometry_and_crowding(emb, tokens, vocab_size=256)
        assert geom["rare_max_cos"] < 0.55


def test_complex_wavefunction_4x2x32_and_4x2x64_hermitian_sphere() -> None:
    """Verify (V, 4, 2, 32), (V, 4, 2, 64), (V, 8, 2, 32), and (V, 16, 2, 16) complex sphere norms and Hermitian Gram-Schmidt."""
    tokens = torch.randint(0, 64, (8192,), dtype=torch.long)
    for mode, expected_mc, expected_dh in (
        (SupportSketchMode.I9_COMPLEX_WAVEFUNCTION_4x2x32, 4, 32),
        (SupportSketchMode.I10_COMPLEX_WAVEFUNCTION_4x2x64, 4, 64),
        (SupportSketchMode.I11_COMPLEX_WAVEFUNCTION_8x2x32_ID2, 8, 32),
        (SupportSketchMode.I12_COMPLEX_WAVEFUNCTION_8x2x32_ID1, 8, 32),
        (SupportSketchMode.I13_COMPLEX_WAVEFUNCTION_16x2x16_ID4, 16, 16),
        (SupportSketchMode.I14_COMPLEX_WAVEFUNCTION_16x2x16_ID2, 16, 16),
    ):
        spec = SupportSketchSpec(mode=mode, vocab_size=256, pkt_count=2 * expected_mc, pkt_dim=expected_dh)
        psi_4d = compute_complex_wavefunction_split_sphere(tokens, spec)
        assert psi_4d.shape == (256, expected_mc, 2, expected_dh)
        # Joint (2, d_h) complex norm squared must equal 2 * d_h (mean squared coordinate == 1.0)
        complex_rms = psi_4d.pow(2).mean(dim=(-2, -1)).sqrt()
        assert float((complex_rms - 1.0).abs().max().item()) < 1e-5

    # Verify exact Hermitian orthogonality <s | z_perp>_C == 0 + 0i when alpha == 0 vs alpha == 1
    sem_4d = torch.randn((32, 4, 2, 32), dtype=torch.float32)
    project_complex_wavepackets_to_rms_sphere_(sem_4d)
    act_mask = torch.ones(32, dtype=torch.float32)
    z_only = _hermitian_gram_schmidt_complex_blend(
        sem_4d, act_mask, torch.zeros((32, 4, 1, 1), dtype=torch.float32), seed=42
    )
    s_re, s_im = sem_4d[:, :, 0, :], sem_4d[:, :, 1, :]
    z_re, z_im = z_only[:, :, 0, :], z_only[:, :, 1, :]
    ip_re = ((s_re * z_re).sum(dim=-1) + (s_im * z_im).sum(dim=-1)) / 64.0
    ip_im = ((s_re * z_im).sum(dim=-1) - (s_im * z_re).sum(dim=-1)) / 64.0
    assert float(ip_re.abs().max().item()) < 1e-5
    assert float(ip_im.abs().max().item()) < 1e-5


def test_all_four_readout_modes_forward_backward() -> None:
    """Verify that Classical and Quantum Wavepacket reversal models run forward/backward across R1..R4."""
    dataset = FineWebSpanReversalDataset(
        vocab_size=128,
        span_len=7,
        train_tokens_limit=4096,
        val_tokens_limit=2048,
        device="cpu",
    )
    idx_in, targets_rev = dataset.sample_train_batch(step=1, batch_size=4)
    assert idx_in.shape == (4, 14)
    assert targets_rev.shape == (4, 7)

    for b_mode in (MicroBackboneMode.CLASSICAL, MicroBackboneMode.QUANTUM_WAVEPACKET):
        for r_mode in (
            MicroReadoutMode.R1_LINEAR_SOFTCAP,
            MicroReadoutMode.R2_LEVEL_A_RADIAL_SPHERE,
            MicroReadoutMode.R3_TIED_WAVEPACKET_EXIT_BRIDGE,
            MicroReadoutMode.R4_HERMITIAN_QUADRATIC_BORN,
        ):
            cfg = MicroReversalConfig(
                backbone_mode=b_mode,
                init_mode=MicroInitMode.GPU_COUNTSKETCH_BIGRAM_SPHERE,
                readout_mode=r_mode,
                vocab_size=128,
                span_len=7,
                embed_dim=64,
                num_heads=2,
                num_layers=1,
                mlp_hidden=128,
                pkt_count=4,
                pkt_dim=16,
            )
            model = MicroReversalGPT(cfg, seq_len=dataset.seq_len)
            model.apply_initialization_(dataset)
            logits, loss = model(idx_in, targets_rev)
            assert logits.shape == (4, 7, 128)
            assert loss is not None and math.isfinite(float(loss.item()))
            loss.backward()
            assert model.wte_3d.grad is not None


def test_step0_causal_readout_procrustes_calibration() -> None:
    """Verify Step-0 Ridge-Procrustes SVD calibration lowers Step-0 CE and keeps Identity Packet dispersion zero."""
    import torch.nn.functional as F
    from qdlblocks.benchmarks.support_preserving_countsketch_lab import (
        _encode_causal_lm_hidden,
        calibrate_step0_causal_readout_,
    )

    dataset = FineWebSpanReversalDataset(
        vocab_size=256,
        span_len=7,
        train_tokens_limit=8192,
        val_tokens_limit=2048,
        device="cpu",
    )
    # Give train_tokens a structured transition sequence over 0..253 so transitions are informative
    t_idx = torch.arange(8192, dtype=torch.long)
    dataset.train_tokens = ((t_idx * 7 + (t_idx // 64)) % 250).to(torch.long)

    spec = SupportSketchSpec(
        mode=SupportSketchMode.I16_SHARD1_100M_STEP0_CALIBRATED,
        vocab_size=256,
        pkt_count=16,
        pkt_dim=32,
    )
    m_cfg = MicroReversalConfig(
        backbone_mode=MicroBackboneMode.QUANTUM_WAVEPACKET,
        init_mode=MicroInitMode.GPU_COUNTSKETCH_BIGRAM_SPHERE,
        readout_mode=MicroReadoutMode.R3_TIED_WAVEPACKET_EXIT_BRIDGE,
        vocab_size=256,
        span_len=7,
        embed_dim=512,
        num_heads=8,
        num_layers=1,
        mlp_hidden=512,
        pkt_count=16,
        pkt_dim=32,
    )
    model = MicroReversalGPT(m_cfg, seq_len=dataset.seq_len)
    model.apply_initialization_(dataset)
    psi_4d = compute_complex_wavefunction_split_sphere(dataset.train_tokens, spec)
    with torch.no_grad():
        model.wte_3d.copy_(psi_4d.reshape(256, 16, 32))

    # Measure pre-calibration Step-0 loss on training stream
    chunk = dataset.train_tokens[: 16 * 8].reshape(16, 8)
    idx_b, y_b = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
    with torch.no_grad():
        h_before = _encode_causal_lm_hidden(model, idx_b)
        logits_before = model.readout(h_before).reshape(-1, 256).float()
        loss_before = F.cross_entropy(logits_before, y_b.reshape(-1).long())

    calibrate_step0_causal_readout_(model, dataset, n_sem_complex=7, n_cal_batches=8, batch_size=16)

    with torch.no_grad():
        h_after = _encode_causal_lm_hidden(model, idx_b)
        logits_after = model.readout(h_after).reshape(-1, 256).float()
        loss_after = F.cross_entropy(logits_after, y_b.reshape(-1).long())
        assert float(loss_after.item()) < float(loss_before.item())
        # Verify Stiefel orthogonality of exit_bridge u_stiefel and v_stiefel
        assert model.readout.exit_bridge is not None
        u_s = model.readout.exit_bridge.u_stiefel
        v_s = model.readout.exit_bridge.v_stiefel
        eye = torch.eye(512, dtype=u_s.dtype)
        assert float((u_s.T @ u_s - eye).abs().max().item()) < 1e-4
        assert float((v_s.T @ v_s - eye).abs().max().item()) < 1e-4
        # Verify semantic packets 0..6 have positive causal dispersion while identity packet 7 has strictly 0.0
        wp_re = model.wp_causal_re
        assert float(wp_re[:7].abs().min().item()) > 0.0
        assert float(wp_re[7].abs().max().item()) == 0.0


def test_shallow_longseq_wavepacket_lm_forward_backward_and_identity_invariants() -> None:
    """Verify `ShallowLongSeqWavepacketGPT` forward/backward pass and `SemanticSubspaceNGramProjector` zero-identity injection."""
    from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
        ShallowArchMode,
        ShallowLongSeqSpec,
        ShallowLongSeqWavepacketGPT,
    )

    idx_in = torch.randint(0, 256, (2, 32), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 32), dtype=torch.long)

    for mode in (
        ShallowArchMode.S1_CLASSICAL_4L_BASELINE,
        ShallowArchMode.S2_QUANTUM_WAVEPACKET_4L_PURE,
        ShallowArchMode.S3_QUANTUM_WAVEPACKET_4L_SEMANTIC_NGRAM,
    ):
        spec = ShallowLongSeqSpec(
            arch_mode=mode,
            vocab_size=256,
            embed_dim=512,
            num_heads=8,
            num_layers=2,
            mlp_hidden=512,
            seq_len=32,
            batch_size=2,
            seed=42,
        )
        model = ShallowLongSeqWavepacketGPT(spec)
        if model.sem_ngram is not None:
            in_delta, out_sem, out_id = model.sem_ngram(idx_in, torch.float32)
            assert in_delta.shape == (2, 32, 512)
            assert out_sem.shape == (2, 32, 448)
            assert out_id.shape == (2, 32, 64)
            # Input N-gram injection must be strictly 0.0 in Identity Packet 7 (last 64 dims)
            assert float(in_delta[:, :, 448:].abs().max().item()) == 0.0

        loss = model.forward_loss(idx_in, targets)
        assert math.isfinite(float(loss.item()))
        loss.backward()
        assert model.wte_3d.grad is not None
        model.clamp_identity_dispersion_and_scalars_()
        assert float(model.wp_causal_re[7].abs().item()) == 0.0
        assert float(model.wp_causal_im[7].abs().item()) == 0.0


def test_multibranch_wavepacket_readout_stiefel_frame_and_custom_autograd() -> None:
    """Verify R-branch identity frame orthogonality, exact custom autograd gradients, and K-target superposition gain."""
    import torch.nn.functional as F
    from qdlblocks.benchmarks.multibranch_wavepacket_readout_lab import (
        DEFAULT_SOFTCAP_LOGIT,
        MultiBranchMode,
        MultiBranchReadoutSpec,
        MultiBranchWavepacketGPT,
        _FactoredMultiBranchWavepacketCEFn,
        orthogonalize_identity_stiefel_frame,
        run_k_target_superposition_probe,
    )

    # 1. Verify exact Gram-Schmidt R-frame orthogonality on 64D identity sphere
    raw_id = torch.randn(2, 8, 8, 64, dtype=torch.float32)
    ortho_id = orthogonalize_identity_stiefel_frame(raw_id, target_rms_norm=8.0)
    gram = torch.matmul(ortho_id, ortho_id.transpose(-1, -2)) / 64.0  # (2, 8, 8, 8)
    eye_r = torch.eye(8, dtype=torch.float32).view(1, 1, 8, 8)
    assert float((gram - eye_r).abs().max().item()) < 1e-4

    # 2. Verify exact gradient match between _FactoredMultiBranchWavepacketCEFn and reference PyTorch autograd
    bsz, seq_len, num_r, v_sz = 2, 8, 4, 64
    hb1 = torch.randn(bsz, seq_len, 464, dtype=torch.float32, requires_grad=True)
    hs1 = torch.randn(bsz, seq_len, 448, dtype=torch.float32, requires_grad=True)
    hid1 = torch.randn(bsz, seq_len, num_r, 64, dtype=torch.float32, requires_grad=True)
    ctrl1 = torch.randn(bsz, seq_len, num_r, 3, dtype=torch.float32, requires_grad=True)
    w1 = torch.randn(v_sz, 528, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, v_sz, (bsz, seq_len), dtype=torch.long)

    loss_custom = _FactoredMultiBranchWavepacketCEFn.apply(hb1, hs1, hid1, ctrl1, w1, targets)
    loss_custom.backward()

    hb2 = hb1.detach().clone().requires_grad_(True)
    hs2 = hs1.detach().clone().requires_grad_(True)
    hid2 = hid1.detach().clone().requires_grad_(True)
    ctrl2 = ctrl1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)

    w_base_t = torch.cat([w2[:, :448], w2[:, 512:]], dim=-1).t()
    z_base = torch.matmul(hb2, w_base_t).unsqueeze(2)
    z_sem1 = torch.matmul(hs2, w2[:, :448].t()).unsqueeze(2)
    z_id = torch.matmul(hid2, w2[:, 448:512].t())
    raw_r = z_base + ctrl2[..., 0:1] * z_sem1 + ctrl2[..., 1:2] * z_id
    z_r = DEFAULT_SOFTCAP_LOGIT * torch.tanh(raw_r / DEFAULT_SOFTCAP_LOGIT)
    log_p_r = F.log_softmax(z_r, dim=-1)
    log_pi = F.log_softmax(ctrl2[..., 2], dim=-1).unsqueeze(-1)
    log_mix = torch.logsumexp(log_pi + log_p_r, dim=2)
    loss_ref = F.nll_loss(log_mix.reshape(-1, v_sz), targets.reshape(-1), reduction="mean")
    loss_ref.backward()

    assert torch.allclose(loss_custom.detach(), loss_ref.detach(), atol=1e-5)
    assert torch.allclose(hb1.grad, hb2.grad, atol=1e-4)  # type: ignore[arg-type]
    assert torch.allclose(hs1.grad, hs2.grad, atol=1e-4)  # type: ignore[arg-type]
    assert torch.allclose(hid1.grad, hid2.grad, atol=1e-4)  # type: ignore[arg-type]
    assert torch.allclose(ctrl1.grad, ctrl2.grad, atol=1e-4)  # type: ignore[arg-type]
    assert torch.allclose(w1.grad, w2.grad, atol=1e-4)  # type: ignore[arg-type]

    # 3. Verify MultiBranchWavepacketGPT forward/backward and K-target superposition probe
    spec = MultiBranchReadoutSpec(
        mode=MultiBranchMode.M3_COUPLED_SEM_ID_WAVEPACKETS_R8,
        vocab_size=512,
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        mlp_hidden=512,
        seq_len=16,
        batch_size=2,
    )
    model = MultiBranchWavepacketGPT(spec)
    idx_in = torch.randint(0, 512, (2, 16), dtype=torch.long)
    y_in = torch.randint(0, 512, (2, 16), dtype=torch.long)
    loss_mb = model.forward_loss(idx_in, y_in)
    assert math.isfinite(float(loss_mb.item()))
    loss_mb.backward()
    model.clamp_identity_dispersion_and_scalars_()

    probe = run_k_target_superposition_probe(
        torch.randn(512, 512), k_values=(1, 4, 8), r_values=(1, 4, 8)
    )
    assert probe["by_k"]["K8"]["R8"]["excess_ce_nats"] < probe["by_k"]["K8"]["R1"]["excess_ce_nats"]
    assert probe["by_k"]["K8"]["R8"]["topk_recall"] > probe["by_k"]["K8"]["R1"]["topk_recall"]


def test_born_schur_density_attention_and_highway_invariants() -> None:
    """Verify BornSchurDensityAttention, DensityMatrixHighwayBlock, Hermitian packing, and Schur PSD invariants."""
    from qdlblocks.benchmarks.born_schur_density_lab import (
        BornSchurDensityAttention,
        DensityMatrixHighwayBlock,
        run_schur_thermodynamic_cooling_probe,
    )

    cooling = run_schur_thermodynamic_cooling_probe(n_packets=4, d_state=16, num_layers=3, device="cpu")
    assert cooling["pure_schur_identity_err"] < 1e-6
    assert cooling["packed_gemm_equivalence_err"] < 1e-6
    for mode_layers in cooling["cooling_vs_thermal_death"].values():
        for entry in mode_layers:
            assert entry["hermitian_err"] < 1e-5
            assert entry["trace_err"] < 1e-4
            assert entry["min_eigenvalue"] >= -1e-5

    x_wp = torch.randn(2, 8, 4, 2, 16, requires_grad=True)
    attn = BornSchurDensityAttention(n_packets=4, d_packet=16, use_dynamic_povm_gates=True)
    y_wp = attn(x_wp)
    assert y_wp.shape == (2, 8, 4, 2, 16)
    norms = y_wp.square().sum(dim=(3, 4))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    y_wp.sum().backward()
    assert x_wp.grad is not None

    psi = torch.randn(2, 8, 4, 2, 16)
    psi = psi / torch.sqrt(psi.square().sum(dim=(3, 4), keepdim=True))
    pr, pi = psi[:, :, :, 0], psi[:, :, :, 1]
    rho_r = pr.unsqueeze(-1) * pr.unsqueeze(-2) + pi.unsqueeze(-1) * pi.unsqueeze(-2)
    rho_i = pi.unsqueeze(-1) * pr.unsqueeze(-2) - pr.unsqueeze(-1) * pi.unsqueeze(-2)
    rho_in = torch.stack([rho_r, rho_i], dim=3).requires_grad_(True)

    hwy_blk = DensityMatrixHighwayBlock(n_packets=4, d_state=16, d_probe=8)
    rho_out, coinc = hwy_blk(rho_in)
    assert rho_out.shape == (2, 8, 4, 2, 16, 16)
    assert coinc > 0.0
    rho_out.sum().backward()
    assert rho_in.grad is not None


def test_wavefunction_outer_density_input_and_cone_free_readout() -> None:
    """Verify exact global phase gauge invariance of outer-product input lift and cone-free Cholesky readout."""
    from qdlblocks.benchmarks.wavefunction_outer_density_lab import (
        NativeCholeskyDensitySpec,
        NativeCholeskyDensityWavepacketGPT,
        apply_phase_invariant_outer_density_input_lift,
        phase_lock_cholesky_modes,
    )

    torch.manual_seed(1337)
    curr_wp = torch.randn(2, 6, 4, 2, 16)
    prev_wp = torch.randn(2, 6, 4, 2, 16)
    m_mix_re, m_mix_im = torch.randn(4, 16), torch.randn(4, 16)
    m_sch_re, m_sch_im = torch.randn(4, 16), torch.randn(4, 16)
    mix_logit, sch_logit = torch.zeros(4), torch.zeros(4)

    out_a = apply_phase_invariant_outer_density_input_lift(
        curr_wp, prev_wp, m_mix_re, m_mix_im, m_sch_re, m_sch_im, (mix_logit, sch_logit, True)
    )
    # Rotate prev_wp by an arbitrary token-and-packet-varying global phase e^{i * theta_{b, t, h}}
    theta = torch.randn(2, 6, 4, 1) * 3.14159
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    pr, pi = prev_wp[..., 0, :], prev_wp[..., 1, :]
    prev_rotated = torch.stack([pr * cos_t - pi * sin_t, pr * sin_t + pi * cos_t], dim=-2)
    out_b = apply_phase_invariant_outer_density_input_lift(
        curr_wp, prev_rotated, m_mix_re, m_mix_im, m_sch_re, m_sch_im, (mix_logit, sch_logit, True)
    )
    assert torch.allclose(out_a, out_b, atol=1e-5)

    # Verify U(1)^4 row gauge phase-locking preserves rho = L^dagger L and yields positive overlap
    ref_r, ref_i = torch.randn(2, 4, 6, 4, 16), torch.randn(2, 4, 6, 4, 16)
    cand_r, cand_i = torch.randn(2, 4, 6, 4, 16), torch.randn(2, 4, 6, 4, 16)
    al_r, al_i = phase_lock_cholesky_modes(ref_r, ref_i, cand_r, cand_i)
    rho_orig_r = torch.einsum("bhtrk,bhtrm->bhtkm", cand_r, cand_r) + torch.einsum("bhtrk,bhtrm->bhtkm", cand_i, cand_i)
    rho_al_r = torch.einsum("bhtrk,bhtrm->bhtkm", al_r, al_r) + torch.einsum("bhtrk,bhtrm->bhtkm", al_i, al_i)
    assert torch.allclose(rho_orig_r, rho_al_r, atol=1e-5)
    real_ov = (ref_r * al_r + ref_i * al_i).sum(dim=-1)
    assert (real_ov >= -1e-5).all()

    # Verify uncollapsed 4-mode Cholesky highway + semantic Cholesky readout + KN synergy eval (D8/D9)
    nat_model = NativeCholeskyDensityWavepacketGPT(
        NativeCholeskyDensitySpec(
            use_outer_product_input=True,
            uncollapsed_cholesky_highway=True,
            use_semantic_cholesky_readout=True,
            enable_kn_synergy_eval=True,
            seed=42,
        )
    )
    assert nat_model.mb_head is None
    assert nat_model.num_branches == 4
    idx = torch.randint(0, 256, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 16), dtype=torch.long)
    hb, hs1, hid, ctrl, w_aug = nat_model.build_multibranch_readout_tensors(idx)
    assert hb.shape == (2, 16, 464)
    assert hs1.shape == (2, 16, 448)
    assert hid.shape == (2, 16, 4, 64)
    assert ctrl.shape == (2, 16, 4, 3)
    loss = nat_model.forward_loss(idx, targets)
    assert math.isfinite(float(loss.item()))
    loss.backward()
    assert nat_model.chol_mode_log_prior.grad is not None
