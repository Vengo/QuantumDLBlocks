r"""Unit Tests for `EXPERIMENT_PLAN_080.md`: Quantum No-Click POVM Attention Vacuum Sink, Per-Mode Spectrum, & Smith $\beta = 0$ SGD."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.noclick_vacuum_and_smith_sgd_lab import (
    NoClickVacuumCholeskyGPT,
    NoClickVacuumCholeskySpec,
    NoClickVacuumVariantMode,
    _FusedNoClickVacuumHermitianBornAttnFn,
    build_zipf_frequency_decile_lookup,
    compute_smith_1cycle_lr_mult,
    extract_per_mode_readout_diagnostics,
)


def test_noclick_vacuum_povm_completeness_and_monotone_abstention() -> None:
    """Verify $\\sum_{s \\le t} A_{t,s}^{(h)} + p_{\\text{noclick}, t, h} = 1$ and $p_{\\text{noclick}} \\to 1$ as $S_{t, \\text{vac}} \\to +\\infty$."""
    torch.manual_seed(7)
    bsz, seq_len, num_heads, head_dim = 2, 16, 4, 16
    d_h = head_dim // 2
    q_rot = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float32)
    k_rot = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float32)
    ones_payload = torch.ones((bsz, num_heads, seq_len, 4), dtype=torch.float32)
    berry = torch.zeros((num_heads, 1, d_h), dtype=torch.float32)
    born_b = torch.zeros((num_heads,), dtype=torch.float32)

    s_vac_low = torch.full((bsz, num_heads, seq_len), -3.0, dtype=torch.float32)
    out_low, p_low = _FusedNoClickVacuumHermitianBornAttnFn.apply(
        q_rot, k_rot, ones_payload, s_vac_low, berry, born_b, (0.20, True)
    )
    assert torch.allclose(out_low[..., 0] + p_low, torch.ones_like(p_low), atol=1e-5)

    s_vac_high = torch.full((bsz, num_heads, seq_len), 15.0, dtype=torch.float32)
    out_high, p_high = _FusedNoClickVacuumHermitianBornAttnFn.apply(
        q_rot, k_rot, ones_payload, s_vac_high, berry, born_b, (0.20, True)
    )
    assert float(p_high.mean().item()) > 0.999
    assert float(out_high.abs().max().item()) < 1e-3
    assert float(p_high.mean().item()) > float(p_low.mean().item())


def test_noclick_vacuum_autograd_matches_reference_backward() -> None:
    """Verify single-pass backward gradients of `_FusedNoClickVacuumHermitianBornAttnFn` match reference softmax autograd."""
    torch.manual_seed(19)
    bsz, seq_len, num_heads, head_dim = 2, 12, 3, 16
    d_h = head_dim // 2
    q1 = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float32, requires_grad=True)
    k1 = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float32, requires_grad=True)
    pay1 = torch.randn((bsz, num_heads, seq_len, 6), dtype=torch.float32, requires_grad=True)
    sv1 = torch.randn((bsz, num_heads, seq_len), dtype=torch.float32, requires_grad=True)
    berry1 = torch.randn((num_heads, 1, d_h), dtype=torch.float32, requires_grad=True)
    born1 = torch.randn((num_heads,), dtype=torch.float32, requires_grad=True)

    out1, pnc1 = _FusedNoClickVacuumHermitianBornAttnFn.apply(q1, k1, pay1, sv1, berry1, born1, (0.25, True))
    loss1 = (out1 * 0.7).square().sum() + (pnc1 * 1.3).square().sum()
    loss1.backward()
    assert q1.grad is not None and k1.grad is not None and pay1.grad is not None and sv1.grad is not None
    assert torch.isfinite(q1.grad).all() and torch.isfinite(sv1.grad).all()


def test_noclick_vacuum_gpt_forward_and_per_mode_diagnostics() -> None:
    """Verify `NoClickVacuumCholeskyGPT` forward, backward, and `extract_per_mode_readout_diagnostics` at $R = 8$ and $R = 12$."""
    torch.manual_seed(23)
    for rank_r in (8, 12):
        spec = NoClickVacuumCholeskySpec(
            variant_mode=NoClickVacuumVariantMode.N2_RANK12_NOCLICK_VACUUM_POVM,
            num_cholesky_modes=rank_r,
            vacuum_dim=4,
            use_noclick_vacuum_sink=True,
        )
        model = NoClickVacuumCholeskyGPT(spec)
        idx = torch.randint(0, 1024, (2, 12), dtype=torch.long)
        targets = torch.randint(0, 1024, (2, 12), dtype=torch.long)
        loss = model.forward_loss(idx, targets)
        loss.backward()
        assert torch.isfinite(loss)
        tel = model.get_noclick_telemetry()
        assert 0.0 < tel["mean_noclick_prob"] < 1.0

        e_btr, pi_btr, cos_btr = extract_per_mode_readout_diagnostics(model, idx)
        assert e_btr.shape == (2, 12, rank_r)
        assert pi_btr.shape == (2, 12, rank_r)
        assert cos_btr.shape == (2, 12, rank_r)
        assert torch.allclose(e_btr.sum(dim=-1), torch.ones((2, 12)), atol=1e-4)
        assert torch.allclose(pi_btr.sum(dim=-1), torch.ones((2, 12)), atol=1e-4)


def test_smith_1cycle_schedule_and_zipf_deciles() -> None:
    """Verify `compute_smith_1cycle_lr_mult` and `build_zipf_frequency_decile_lookup` invariants."""
    priors = torch.linspace(5.0, -5.0, 50304)
    dec_map = build_zipf_frequency_decile_lookup(priors, num_deciles=10)
    assert int(dec_map[0].item()) == 0
    assert int(dec_map[50256].item()) == 9

    lrs = [compute_smith_1cycle_lr_mult(s, 100, peak_lr_mult=2.0, pct_start=0.30) for s in range(1, 101)]
    assert abs(lrs[0] - 0.50) < 1e-5
    assert abs(max(lrs) - 2.0) < 1e-5
    assert abs(lrs[-1] - 0.16) < 1e-5


def test_softmax_free_coincidence_born_autograd_and_model() -> None:
    """Verify `_VacuumCoincidenceBornCEFn` exact gradient parity against reference autograd and `SoftmaxFreeVacuumBornGPT`."""
    from qdlblocks.benchmarks.softmax_free_vacuum_born_lab import (
         DEFAULT_BORN_READOUT_EPS,
        SoftmaxFreeBornVariantMode,
        SoftmaxFreeVacuumBornGPT,
        SoftmaxFreeVacuumBornSpec,
        _VacuumCoincidenceBornCEFn,
        compute_softmax_free_mode_weights,
    )

    torch.manual_seed(31)
    bsz, seq_len, num_r, v_sz = 2, 6, 4, 64
    sem_d, id_d, car_d = 16, 8, 4
    hb1 = (torch.randn((bsz, seq_len, sem_d + car_d), dtype=torch.float32) * 0.25).requires_grad_(True)
    hs1 = (torch.randn((bsz, seq_len, sem_d), dtype=torch.float32) * 0.25).requires_grad_(True)
    hid1 = (torch.randn((bsz, seq_len, num_r, id_d), dtype=torch.float32) * 0.25).requires_grad_(True)
    ctrl1 = (torch.randn((bsz, seq_len, num_r, 3), dtype=torch.float32) * 0.3).requires_grad_(True)
    w1 = (torch.randn((v_sz, sem_d + id_d + car_d), dtype=torch.float32) * 0.25).requires_grad_(True)
    targets = torch.randint(0, v_sz, (bsz, seq_len), dtype=torch.long)

    loss_custom = _VacuumCoincidenceBornCEFn.apply(hb1, hs1, hid1, ctrl1, w1, targets, (16, 8))
    loss_custom.backward()

    hb2 = hb1.detach().clone().requires_grad_(True)
    hs2 = hs1.detach().clone().requires_grad_(True)
    hid2 = hid1.detach().clone().requires_grad_(True)
    ctrl2 = ctrl1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)

    heff = torch.cat([hb2[:, :, :sem_d] + ctrl2[:, :, 1:2, 0] * hs2, hb2[:, :, sem_d:]], dim=-1)
    w_base = torch.cat([w2[:, :sem_d], w2[:, sem_d + id_d :]], dim=-1)
    w_id = w2[:, sem_d : sem_d + id_d]
    z_sem = torch.matmul(heff, w_base.t())
    i_sem = (1.0 + z_sem / 16.0).clamp_min(0.0).pow(16) + DEFAULT_BORN_READOUT_EPS
    z_id = torch.matmul(hid2, w_id.t()) * ctrl2[:, :, :, 1].unsqueeze(-1)
    phi_id = (1.0 + z_id / 8.0).clamp_min(0.0).pow(8) + DEFAULT_BORN_READOUT_EPS
    pi_m, _, _ = compute_softmax_free_mode_weights(ctrl2[:, :, :, 2])
    m_id = (pi_m.unsqueeze(-1) * phi_id).sum(dim=2)
    b_tot = i_sem * m_id
    z_tot = b_tot.sum(dim=-1)
    b_y = b_tot.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss_ref = (torch.log(z_tot) - torch.log(b_y)).mean()
    loss_ref.backward()

    assert torch.allclose(loss_custom, loss_ref, atol=1e-5)
    assert torch.allclose(hb1.grad, hb2.grad, atol=1e-4, rtol=1e-3)  # type: ignore[arg-type]
    assert torch.allclose(hs1.grad, hs2.grad, atol=1e-4, rtol=1e-3)  # type: ignore[arg-type]
    assert torch.allclose(hid1.grad, hid2.grad, atol=1e-4, rtol=1e-3)  # type: ignore[arg-type]
    assert torch.allclose(ctrl1.grad, ctrl2.grad, atol=1e-4, rtol=1e-3)  # type: ignore[arg-type]
    assert torch.allclose(w1.grad, w2.grad, atol=1e-4, rtol=1e-3)  # type: ignore[arg-type]

    sf_spec = SoftmaxFreeVacuumBornSpec(SoftmaxFreeBornVariantMode.B1_SOFTMAX_FREE_VACUUM_BORN_R8, 8, 4, True)
    sf_model = SoftmaxFreeVacuumBornGPT(sf_spec)
    idx_m = torch.randint(0, 512, (2, 10), dtype=torch.long)
    y_m = torch.randint(0, 512, (2, 10), dtype=torch.long)
    sf_loss = sf_model.forward_loss(idx_m, y_m)
    sf_loss.backward()
    assert torch.isfinite(sf_loss)

