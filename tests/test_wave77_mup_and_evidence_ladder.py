"""Unit tests for Wave 77 (`EXPERIMENT_PLAN_097.md`): MuP Coordinate Check, 5-Rung Evidence Ladder, & Carbon Routing."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks import (
    FINEWEB_BOS_TOKEN_ID,
    CalibratedPBosonAssociativeCopyGate,
    MuPCoordinateCheckConfig,
    Wave77RungKind,
    apply_bos_anchor_sink_row_norm,
    build_bos_anchor_sink_doc_masks,
    build_wave77_rung_adapter,
    explicit_complementary_schur_matrix_pair,
    run_mup_width_coordinate_check,
    zero_matrix_complementary_schur_probe_action,
)


def _make_trace_unit_factors(batch: int, rank: int, d_h: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create trace-normalized complex factor pairs `(u_re, u_im)` of shape `(batch, rank, d_h)`."""
    gen = torch.Generator().manual_seed(seed)
    u_re = torch.randn((batch, rank, d_h), generator=gen, dtype=torch.float64)
    u_im = torch.randn((batch, rank, d_h), generator=gen, dtype=torch.float64)
    tr = (u_re.square() + u_im.square()).sum(dim=(-2, -1), keepdim=True)
    inv_s = torch.rsqrt(tr)
    return (u_re * inv_s).float(), (u_im * inv_s).float()


def test_complementary_schur_zero_matrix_matches_explicit_and_psd() -> None:
    """Verify `< 1e-5` identity with Carbon's explicit partial-trace matrix, PSD eigenvalues, & trace conservation."""
    d_h = 16
    bsz = 6
    rank = 3
    u_a_re, u_a_im = _make_trace_unit_factors(bsz, rank, d_h, seed=101)
    u_b_re, u_b_im = _make_trace_unit_factors(bsz, rank, d_h, seed=202)

    gen = torch.Generator().manual_seed(303)
    psi_re = torch.randn((bsz, d_h), generator=gen)
    psi_im = torch.randn((bsz, d_h), generator=gen)
    nrm = torch.rsqrt((psi_re.square() + psi_im.square()).sum(dim=-1, keepdim=True))
    psi_re, psi_im = psi_re * nrm, psi_im * nrm

    and_re, and_im, comp_re, comp_im, p_zero = zero_matrix_complementary_schur_probe_action(
        u_a_re, u_a_im, u_b_re, u_b_im, psi_re, psi_im
    )

    ua = torch.complex(u_a_re, u_a_im)
    ub = torch.complex(u_b_re, u_b_im)
    rho_a = torch.matmul(ua.transpose(-2, -1), ua.conj())
    rho_b = torch.matmul(ub.transpose(-2, -1), ub.conj())
    rho_and, rho_comp, p_exp = explicit_complementary_schur_matrix_pair(rho_a, rho_b)

    psi_c = torch.complex(psi_re, psi_im).unsqueeze(-1)
    ref_and = torch.matmul(rho_and, psi_c).squeeze(-1)
    ref_comp = torch.matmul(rho_comp, psi_c).squeeze(-1)

    assert torch.allclose(p_zero, p_exp, atol=1e-6)
    assert (and_re - ref_and.real).abs().max().item() < 1e-5
    assert (and_im - ref_and.imag).abs().max().item() < 1e-5
    assert (comp_re - ref_comp.real).abs().max().item() < 1e-5
    assert (comp_im - ref_comp.imag).abs().max().item() < 1e-5

    # Check Hermitian, unit trace, and PSD eigenvalues
    for mat in (rho_and, rho_comp):
        herm_err = (mat - mat.transpose(-2, -1).conj()).abs().max().item()
        assert herm_err < 1e-6
        tr_vals = torch.diagonal(mat, dim1=-2, dim2=-1).real.sum(dim=-1)
        assert torch.allclose(tr_vals, torch.ones_like(tr_vals), atol=1e-5)
        eigvals = torch.linalg.eigvalsh(mat)
        assert float(eigvals.amin().item()) >= -1e-5


def test_complementary_schur_degenerate_p0_and_p1_gradient_safety() -> None:
    """Verify epsilon fallback to `I / d_h` prevents NaNs/gradient spikes when `p -> 0` or `p -> 1`."""
    d_h = 16
    # Case 1: p -> 0 (orthogonal computational basis states |0> and |1>)
    u_a_re = torch.zeros((1, 1, d_h), requires_grad=True)
    u_b_re = torch.zeros((1, 1, d_h), requires_grad=True)
    with torch.no_grad():
        u_a_re[0, 0, 0] = 1.0
        u_b_re[0, 0, 1] = 1.0
    z_im = torch.zeros((1, 1, d_h))
    psi_re = torch.full((1, d_h), 0.25)
    psi_im = torch.zeros((1, d_h))

    and_re0, _, comp_re0, _, p0 = zero_matrix_complementary_schur_probe_action(
        u_a_re, z_im, u_b_re, z_im, psi_re, psi_im
    )
    assert float(p0.item()) == 0.0
    loss0 = and_re0.sum() + comp_re0.sum()
    loss0.backward()
    assert u_a_re.grad is not None and torch.isfinite(u_a_re.grad).all()

    # Case 2: p -> 1 (identical computational basis state |0> and |0>)
    v_a_re = torch.zeros((1, 1, d_h), requires_grad=True)
    v_b_re = torch.zeros((1, 1, d_h), requires_grad=True)
    with torch.no_grad():
        v_a_re[0, 0, 0] = 1.0
        v_b_re[0, 0, 0] = 1.0
    and_re1, _, comp_re1, _, p1 = zero_matrix_complementary_schur_probe_action(
        v_a_re, z_im, v_b_re, z_im, psi_re, psi_im
    )
    assert abs(float(p1.item()) - 1.0) < 1e-6
    loss1 = and_re1.sum() + comp_re1.sum()
    loss1.backward()
    assert v_a_re.grad is not None and torch.isfinite(v_a_re.grad).all()


def test_bos_anchor_sink_row_sum_and_zero_cross_doc_leakage() -> None:
    """Verify exterior masked rows right after `<BOS>` sum to `1.0` with `0.0` cross-`<BOS>` leakage."""
    idx = torch.tensor([[10, 11, 12, FINEWEB_BOS_TOKEN_ID, 21, 22, 23, 24]], dtype=torch.long)
    pack = build_bos_anchor_sink_doc_masks(idx)
    t_idx = torch.arange(8).unsqueeze(1)
    s_idx = torch.arange(8).unsqueeze(0)
    exterior = torch.where(t_idx >= 4, s_idx <= (t_idx - 4), s_idx <= t_idx)
    valid = exterior.unsqueeze(0).unsqueeze(0) & pack.doc_same.unsqueeze(1)

    raw_scores = torch.rand((1, 2, 8, 8)) + 0.2
    probs = apply_bos_anchor_sink_row_norm(raw_scores, valid, pack.doc_start_pos)

    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    cross_doc_mask = (~pack.doc_same).unsqueeze(1).expand_as(probs)
    assert float(probs[cross_doc_mask].abs().max().item()) == 0.0
    # Position 4 (`pos_in_doc == 1` after `<BOS>` at index 3) has all exterior `s <= 0` in prior doc,
    # so anchor sink must route 1.0 to `s == 3` (`<BOS>`)!
    assert torch.allclose(probs[:, :, 4, 3], torch.ones((1, 2)), atol=1e-6)


def test_calibrated_copy_gate_sharp_match_and_bos_isolation() -> None:
    """Verify exact single match produces `> +8` logits while pre-`<BOS>` false matches get `0.0`."""
    d_c = 32
    gate = CalibratedPBosonAssociativeCopyGate(copy_ket_dim=d_c, ctx_dim=64)
    gen = torch.Generator().manual_seed(77)
    vocab_re = torch.randn((128, d_c), generator=gen)
    vocab_im = torch.randn((128, d_c), generator=gen)

    # Sequence: [A=5, C_false=9, <BOS>, 14, A=5, B_true=22, 31, A=5]
    idx = torch.tensor([[5, 9, FINEWEB_BOS_TOKEN_ID, 14, 5, 22, 31, 5]], dtype=torch.long)
    mask_pack = build_bos_anchor_sink_doc_masks(idx)
    k_re = vocab_re[idx.clamp_max(127)]
    k_im = vocab_im[idx.clamp_max(127)]
    ctx = torch.zeros((1, 8, 64))

    copy_lo, copy_conf = gate(k_re, k_im, ctx, mask_pack.doc_same)
    # Query at t=7 (`A=5`): pre-<BOS> C_false is at s=1 -> must be 0.0; true B_true is at s=5 -> > 8.0 logits!
    assert float(copy_lo[0, 7, 1].item()) == 0.0
    assert float(copy_lo[0, 7, 5].item()) > 8.0
    assert float(copy_conf[0, 7].item()) > 0.80


def test_strict_causality_across_all_wave77_rungs_including_carbon_mtp() -> None:
    """Mutating future tokens `idx[:, t+1:]` must produce `0.0` change in `log_probs[:, :t+1]`."""
    device = torch.device("cpu")
    rungs_to_check = [
        Wave77RungKind.R1_CALIB_COPY_SINK_2L_D64,
        Wave77RungKind.R2_PRUNED_UNTIED_2L_D64,
        Wave77RungKind.R3_MUON_MIXER_2L_D64,
        Wave77RungKind.R5A_CARBON_COMP_SCHUR_2L_D128,
        Wave77RungKind.R5B_CARBON_COMP_SCHUR_MTP_2L_D128,
    ]
    x_a = torch.tensor(
        [[15, 42, 19, FINEWEB_BOS_TOKEN_ID, 88, 104, 88, 205, 311, 17, 99, 44]],
        dtype=torch.long,
    )
    x_b = x_a.clone()
    pivot_t = 6
    x_b[:, pivot_t + 1 :] = torch.tensor([[FINEWEB_BOS_TOKEN_ID, 7, 8, 9, 10]], dtype=torch.long)

    for rung in rungs_to_check:
        adap = build_wave77_rung_adapter(rung, device=device, seed=99)
        lp_a, _ = adap.predict_log_probs_and_span(x_a)
        lp_b, _ = adap.predict_log_probs_and_span(x_b)
        max_diff = float((lp_a[:, : pivot_t + 1] - lp_b[:, : pivot_t + 1]).abs().max().item())
        assert max_diff < 1e-5, f"Causality violation on {rung.value}: max_diff={max_diff}"

        # Also verify 1 train_step executes cleanly with finite loss
        y_dummy = torch.roll(x_a, shifts=-1, dims=1)
        tr_loss = adap.train_step(x_a, y_dummy, step=1, total_steps=10)
        assert tr_loss > 0.0 and tr_loss < 30.0


def test_mup_width_coordinate_check_cpu_smoke() -> None:
    """Run a fast 2-step CPU smoke test of `run_mup_width_coordinate_check`."""
    cfg = MuPCoordinateCheckConfig(
        widths=(32, 64),
        steps=2,
        batch_size=2,
        seq_len=32,
        num_layers=2,
        eval_tokens=64,
        seed=55,
    )
    report = run_mup_width_coordinate_check(cfg, device=torch.device("cpu"))
    assert "naive_sp_adamw" in report["records_by_mode"]
    assert "mup_stiefel_muon" in report["records_by_mode"]
    assert len(report["records_by_mode"]["mup_stiefel_muon"]) == 2


def test_multishard_confirmation_spin_smoke(tmp_path: Any, monkeypatch: Any) -> None:
    """Verify `run_wave77_multishard_confirmation_spin` + Shard-0 eval executes end-to-end."""
    import qdlblocks.benchmarks.experiment_logger as exp_log_mod
    from qdlblocks.benchmarks.architecture_xray_diagnostics import (
        TeacherReferenceProfile,
        build_empirical_zipf_decile_map,
        load_fineweb_token_slice,
    )
    import qdlblocks.benchmarks.wave77_evidence_ladder_runner as w77_run_mod

    monkeypatch.setattr(w77_run_mod, "DEFAULT_W77_CKPT_DIR", tmp_path / "ckpts")
    orig_logger_cls = exp_log_mod.StandardizedExperimentLogger
    monkeypatch.setattr(
        w77_run_mod,
        "StandardizedExperimentLogger",
        lambda spec: orig_logger_cls(
            spec,
            live_logs_dir=tmp_path / "live_logs",
            model_cards_dir=tmp_path / "model_cards",
        ),
    )

    toks = load_fineweb_token_slice("fineweb_val_000000.bin", num_tokens=129)
    x_v = toks[:-1].view(2, 64)
    y_v = toks[1:].view(2, 64)
    dec_map = build_empirical_zipf_decile_map(device=torch.device("cpu"))
    dummy_ref = TeacherReferenceProfile(
        anchor_name="smoke_ref",
        anchor_val_ce=3.5,
        token_nll=torch.full((2, 64), 3.5),
        token_entropy=torch.full((2, 64), 3.5),
    )
    spin_cfg = w77_run_mod.MultiShardConfirmConfig(
        target_tokens=256,
        batch_size=2,
        seq_len=64,
        eval_interval=2,
        official_val_tokens=256,
        wave_id="W77TEST",
        plan_id="EXPERIMENT_PLAN_097",
    )
    res = w77_run_mod.run_wave77_multishard_confirmation_spin(
        rung=Wave77RungKind.R3_MUON_MIXER_2L_D64,
        spin_cfg=spin_cfg,
        x_xray=x_v,
        y_xray=y_v,
        decile_map=dec_map,
        ref_550m=dummy_ref,
        device=torch.device("cpu"),
    )
    assert res["steps"] == 2
    assert res["official_shard0_10m_eval"]["eval_tokens"] == 256
    assert res["official_shard0_10m_eval"]["official_shard0_val_ce"] > 0.0

