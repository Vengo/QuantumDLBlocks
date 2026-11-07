"""Unit Tests for `EXPERIMENT_PLAN_078.md` (`Gates V1..V4`): Orthogonal Vacuum Subspace & 4-Mode 512D Cholesky Readout."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import DEFAULT_SOFTCAP_LOGIT
from qdlblocks.benchmarks.vacuum_subspace_cholesky_lab import (
    DEFAULT_VACUUM_SUBSPACE_DIM,
    VacuumCholeskyExperimentSpec,
    VacuumCholeskyVariantMode,
    VacuumSubspaceCholeskyGPT,
    _VacuumFourMode512DWavepacketCEFn,
    build_layer0_multihorizon_cholesky_factor,
    vacuum_sphere_rms_norm,
)


def test_gate_v1_vacuum_sphere_rms_norm_geometry_and_abstention() -> None:
    """Gate V1: Verify exact sphere norm conservation on S^{(d + d_vac) - 1} and monotonic abstention angle cos(theta_vac)."""
    torch.manual_seed(1337)
    d_vocab, d_vac = 64, 4
    u_vocab = torch.randn(3, 8, d_vocab)
    u_vac_zero = torch.zeros(3, 8, d_vac)

    # 1. Zero vacuum reduces identically to standard F.rms_norm(u_vocab, (d_vocab,))
    h_vocab_0, h_vac_0, cos_0 = vacuum_sphere_rms_norm(u_vocab, u_vac_zero, eps=1e-6)
    ref_rms = F.rms_norm(u_vocab, (d_vocab,), eps=1e-6)
    assert torch.allclose(h_vocab_0, ref_rms, atol=1e-5)
    assert torch.allclose(h_vac_0, torch.zeros_like(h_vac_0), atol=1e-6)
    assert torch.allclose(cos_0, torch.ones_like(cos_0), atol=1e-4)

    # 2. Lifted vector [h_vocab, sqrt(d_vocab / d_vac) * h_vac] has exact squared norm d_vocab on S^{(d_vocab + d_vac) - 1}
    u_vac_mod = torch.randn(3, 8, d_vac) * 0.75
    h_vocab_1, h_vac_1, cos_1 = vacuum_sphere_rms_norm(u_vocab, u_vac_mod, eps=1e-8)
    lifted = torch.cat([h_vocab_1, math.sqrt(float(d_vocab) / float(d_vac)) * h_vac_1], dim=-1)
    lifted_norm_sq = lifted.square().sum(dim=-1)
    assert torch.allclose(lifted_norm_sq, torch.full_like(lifted_norm_sq, float(d_vocab)), atol=1e-4)

    # 3. Effective RMS norm inside H_vocab equals cos(theta_vac) in [0, 1], and decreases monotonically with vacuum mass
    vocab_rms = h_vocab_1.square().mean(dim=-1).sqrt()
    assert torch.allclose(vocab_rms, cos_1, atol=1e-5)
    _, _, cos_large_vac = vacuum_sphere_rms_norm(u_vocab, u_vac_mod * 4.0, eps=1e-8)
    assert bool((cos_large_vac < cos_1).all().item())


def test_gate_v2_single_pass_4mode_512d_autograd_gradient_parity() -> None:
    """Gate V2 (Rule 9): Verify exact forward and backward gradient parity of `_VacuumFourMode512DWavepacketCEFn`."""
    torch.manual_seed(42)
    bsz, seq_len, num_r, v_sz = 2, 6, 4, 64
    hb1 = torch.randn(bsz, seq_len, 464, dtype=torch.float32, requires_grad=True)
    hs1 = torch.randn(bsz, seq_len, num_r, 448, dtype=torch.float32, requires_grad=True)
    hid1 = torch.randn(bsz, seq_len, num_r, 64, dtype=torch.float32, requires_grad=True)
    ctrl1 = torch.randn(bsz, seq_len, num_r, 3, dtype=torch.float32, requires_grad=True)
    w1 = torch.randn(v_sz, 528, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, v_sz, (bsz, seq_len), dtype=torch.long)

    loss_custom = _VacuumFourMode512DWavepacketCEFn.apply(hb1, hs1, hid1, ctrl1, w1, targets)
    loss_custom.backward()

    hb2 = hb1.detach().clone().requires_grad_(True)
    hs2 = hs1.detach().clone().requires_grad_(True)
    hid2 = hid1.detach().clone().requires_grad_(True)
    ctrl2 = ctrl1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)

    w_base_t = torch.cat([w2[:, :448], w2[:, 512:]], dim=-1).t()
    z_base = torch.matmul(hb2, w_base_t).unsqueeze(2)
    z_sem = torch.matmul(hs2, w2[:, :448].t())
    z_id = torch.matmul(hid2, w2[:, 448:512].t())
    raw_r = z_base + ctrl2[..., 0:1] * z_sem + ctrl2[..., 1:2] * z_id
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


def test_gate_v3_layer0_multihorizon_cholesky_unit_trace_and_id_isolation() -> None:
    """Gate V3: Verify Layer-0 Cholesky factor has exact unit Frobenius norm and isolates Identity Packet 7 from neighbors."""
    torch.manual_seed(99)
    raw_wp = torch.randn(2, 8, 8, 2, 32)
    wp_re = torch.randn(8) * 0.2
    wp_im = torch.randn(8) * 0.2
    wp_re[-1] = 0.0
    wp_im[-1] = 0.0
    row_scales = torch.tensor([0.35, 0.22, 0.25], dtype=torch.float32)

    l0_re, l0_im = build_layer0_multihorizon_cholesky_factor(raw_wp, wp_re, wp_im, row_scales)
    assert l0_re.shape == (2, 8, 8, 4, 32)
    assert l0_im.shape == (2, 8, 8, 4, 32)

    # Unit Frobenius norm per (b, h, t): ||L^{(0)}_{t,h}||_F^2 == 1.0
    frob_sq = l0_re.square().sum(dim=(-2, -1)) + l0_im.square().sum(dim=(-2, -1))
    assert torch.allclose(frob_sq, torch.ones_like(frob_sq), atol=1e-4)

    # Identity Packet 7 (h = 7) has zero energy on neighbor rows r = 1, 2, 3
    id_neighbor_energy = l0_re[:, 7, :, 1:, :].square().sum() + l0_im[:, 7, :, 1:, :].square().sum()
    assert float(id_neighbor_energy.item()) == 0.0


def test_gate_v4_vacuum_subspace_cholesky_gpt_end_to_end_variants() -> None:
    """Gate V4: Verify end-to-end forward, backward, and vacuum abstention telemetry across V1, V2, and V3."""
    torch.manual_seed(1337)
    idx = torch.randint(0, 256, (2, 12), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 12), dtype=torch.long)

    for mode, use_l0, use_indep in [
        (VacuumCholeskyVariantMode.V1_VACUUM_SUBSPACE_STEP0_PURE_NEURAL, False, False),
        (VacuumCholeskyVariantMode.V2_VACUUM_PLUS_LAYER0_INPUT_CHOLESKY, True, False),
        (VacuumCholeskyVariantMode.V3_VACUUM_4MODE_INDEPENDENT_512D_CHOLESKY, True, True),
    ]:
        spec = VacuumCholeskyExperimentSpec(
            variant_mode=mode,
            vacuum_dim=DEFAULT_VACUUM_SUBSPACE_DIM,
            use_layer0_input_cholesky=use_l0,
            use_independent_4mode_512d=use_indep,
            seed=1337,
        )
        model = VacuumSubspaceCholeskyGPT(spec)
        hb, hs, hid, ctrl, _ = model.build_multibranch_readout_tensors(idx)
        assert hb.shape == (2, 12, 464)
        assert hid.shape == (2, 12, 4, 64)
        assert ctrl.shape == (2, 12, 4, 3)
        if use_indep:
            assert hs.shape == (2, 12, 4, 448)
        else:
            assert hs.shape == (2, 12, 448)

        loss = model.forward_loss(idx, targets)
        assert math.isfinite(float(loss.item()))
        loss.backward()
        assert model.id_vac_mass.grad is not None
        assert model.chol_id_vac_mass.grad is not None
        assert 0.0 < model.last_anchor_id_cos_vac < 1.0
        assert 0.0 < model.last_excited_id_cos_vac < 1.0
        model.clamp_identity_dispersion_and_scalars_()
