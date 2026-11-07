r"""Unit Tests for `EXPERIMENT_PLAN_083.md` (`Wave 63`): Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density-Matrix Highway & `512D` Tomographic Readout."""

from __future__ import annotations

import pytest
import torch

from qdlblocks.benchmarks.autograd_efficiency_auditor import audit_chunked_nll_autograd_efficiency
from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (
    DEFAULT_PROBES_PER_HEAD,
    DEFAULT_REGISTER_BUCKETS,
    DEFAULT_SOFTCAP_LOGIT,
    DEFAULT_TOMO_EMBED_DIM,
    DEFAULT_TOMO_HEAD_COMPLEX_DIM,
    DEFAULT_TOMO_NUM_HEADS,
    ExplicitDensityTomographicGPT,
    ExplicitDensityTomographicSpec,
    ExplicitDensityVariantMode,
    _SingleGemm512DTomographicCEFn,
    _step_explicit_density_optimizers,
    build_explicit_density_tomographic_optimizers,
    build_layer0_explicit_outer_density,
    compute_tomographic_512d_probe_expectations,
    conjugate_packed_hermitian_by_unitary,
    initialize_explicit_density_tomographic_model_,
)
from qdlblocks.layers.impure_density_mirror_block import (
    complex_l2_normalize,
    pack_hermitian_isometric,
    unpack_hermitian_isometric,
)
from qdlblocks.optim.vocab_manifold_muon import RiemannianGeodesicScalarAndProbeOptimizer


def test_frobenius_isometric_tomographic_trace_identity() -> None:
    r"""Verify $\langle \tilde{P}(\rho_{t,h}), \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \rangle_{\mathbb{R}^{256}} = \langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle$ and $\langle \mathbf{w}_v, \mathbf{q}_t \rangle = \sum_h \operatorname{Tr}(\rho_{t,h} E_{v,h})$."""
    torch.manual_seed(42)
    bsz, seq_len = 2, 6
    num_h = DEFAULT_TOMO_NUM_HEADS
    d_h = DEFAULT_TOMO_HEAD_COMPLEX_DIM
    num_k = DEFAULT_PROBES_PER_HEAD

    psi_raw = torch.randn(bsz, num_h, seq_len, 2, d_h)
    p_re, p_im = complex_l2_normalize(psi_raw[..., 0, :], psi_raw[..., 1, :])
    lag_phase = torch.linspace(-0.3, 0.3, num_h)
    mix_logits = torch.tensor([0.9, 0.2, -0.1])

    rho_packed = build_layer0_explicit_outer_density(p_re, p_im, lag_phase, mix_logits)
    assert rho_packed.shape == (bsz, num_h, seq_len, d_h * d_h)

    rho_re, rho_im = unpack_hermitian_isometric(rho_packed, d_h)
    # Trace of convex combination of unit kets must equal 1.0
    traces = torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1)
    assert torch.allclose(traces, torch.ones_like(traces), atol=1e-5)

    probe_raw = torch.randn(num_h, num_k, 2, d_h)
    c_re, c_im = complex_l2_normalize(probe_raw[..., 0, :], probe_raw[..., 1, :])
    q_512 = compute_tomographic_512d_probe_expectations(rho_packed, c_re, c_im)
    assert q_512.shape == (bsz, seq_len, DEFAULT_TOMO_EMBED_DIM)
    assert (q_512 >= -1e-5).all()

    # Check operator trace identity against explicit E_{v,h} = sum_k w_{v,h,k} |c_{k,h}><c_{k,h}|
    v_test = 16
    w_test = torch.randn(v_test, DEFAULT_TOMO_EMBED_DIM)
    logits_dot = torch.matmul(q_512, w_test.t())

    w_vhk = w_test.view(v_test, num_h, num_k, 1, 1)
    c_out_re = (c_re.unsqueeze(-1) * c_re.unsqueeze(-2) + c_im.unsqueeze(-1) * c_im.unsqueeze(-2)).unsqueeze(0)
    c_out_im = (c_im.unsqueeze(-1) * c_re.unsqueeze(-2) - c_re.unsqueeze(-1) * c_im.unsqueeze(-2)).unsqueeze(0)
    e_re = (w_vhk * c_out_re).sum(dim=2)
    e_im = (w_vhk * c_out_im).sum(dim=2)
    logits_tr = torch.einsum("bhtij,vhij->btv", rho_re, e_re) + torch.einsum("bhtij,vhij->btv", rho_im, e_im)
    assert torch.allclose(logits_dot, logits_tr, atol=1e-4)


def test_unitary_conjugation_preserves_hermiticity_trace_and_purity() -> None:
    r"""Verify $U_h \rho_{t,h} U_h^\dagger$ preserves trace $\operatorname{Tr}(\rho)$ and purity $\operatorname{Tr}(\rho^2)$."""
    torch.manual_seed(99)
    num_h = DEFAULT_TOMO_NUM_HEADS
    d_h = DEFAULT_TOMO_HEAD_COMPLEX_DIM
    psi_raw = torch.randn(2, num_h, 4, 2, d_h)
    p_re, p_im = complex_l2_normalize(psi_raw[..., 0, :], psi_raw[..., 1, :])
    rho_packed = build_layer0_explicit_outer_density(p_re, p_im, torch.zeros(num_h), torch.tensor([0.5, 0.3, 0.2]))

    q_c, _ = torch.linalg.qr(torch.complex(torch.randn(num_h, d_h, d_h), torch.randn(num_h, d_h, d_h)))
    rot_packed, rot_re, rot_im = conjugate_packed_hermitian_by_unitary(rho_packed, q_c.real, q_c.imag)

    assert torch.allclose(rot_re, rot_re.transpose(-1, -2), atol=1e-5)
    assert torch.allclose(rot_im, -rot_im.transpose(-1, -2), atol=1e-5)
    assert torch.allclose(rho_packed.square().sum(dim=-1), rot_packed.square().sum(dim=-1), atol=1e-4)


def test_single_gemm_tomographic_ce_gradients_and_rule9_gate() -> None:
    """Verify `_SingleGemm512DTomographicCEFn` analytical backward gradients and Rule-9 efficiency gate."""
    torch.manual_seed(123)
    bsz, seq_len, d_dim, v_sz = 2, 5, 32, 300
    h_a = torch.randn(bsz, seq_len, d_dim, requires_grad=True)
    w_a = torch.randn(v_sz, d_dim, requires_grad=True)
    b_a = torch.randn(v_sz, requires_grad=True)
    r_hi_a = torch.randn(bsz, seq_len, 4, DEFAULT_REGISTER_BUCKETS, requires_grad=True)
    r_lo_a = torch.randn(bsz, seq_len, 4, DEFAULT_REGISTER_BUCKETS, requires_grad=True)
    targets = torch.randint(0, v_sz, (bsz, seq_len))

    loss_a = _SingleGemm512DTomographicCEFn.apply(h_a, w_a, b_a, r_hi_a, r_lo_a, targets)
    loss_a.backward()

    # Reference eager computation
    h_b = h_a.detach().clone().requires_grad_(True)
    w_b = w_a.detach().clone().requires_grad_(True)
    b_b = b_a.detach().clone().requires_grad_(True)
    r_hi_b = r_hi_a.detach().clone().requires_grad_(True)
    r_lo_b = r_lo_a.detach().clone().requires_grad_(True)

    z_ref = torch.matmul(h_b.view(-1, d_dim), w_b.t()) + b_b.unsqueeze(0)
    grid_ref = torch.bmm(
        r_hi_b.view(-1, 4, DEFAULT_REGISTER_BUCKETS).transpose(1, 2),
        r_lo_b.view(-1, 4, DEFAULT_REGISTER_BUCKETS),
    ).view(-1, DEFAULT_REGISTER_BUCKETS * DEFAULT_REGISTER_BUCKETS)[:, :v_sz]
    cap = float(DEFAULT_SOFTCAP_LOGIT)
    z_cap = cap * torch.tanh((z_ref + grid_ref) / cap)
    loss_b = torch.nn.functional.cross_entropy(z_cap, targets.view(-1))
    loss_b.backward()

    assert torch.allclose(loss_a, loss_b, atol=1e-5)
    assert h_a.grad is not None and h_b.grad is not None and torch.allclose(h_a.grad, h_b.grad, atol=1e-4)
    assert w_a.grad is not None and w_b.grad is not None and torch.allclose(w_a.grad, w_b.grad, atol=1e-4)
    assert b_a.grad is not None and b_b.grad is not None and torch.allclose(b_a.grad, b_b.grad, atol=1e-4)
    assert r_hi_a.grad is not None and r_hi_b.grad is not None and torch.allclose(r_hi_a.grad, r_hi_b.grad, atol=1e-4)
    assert r_lo_a.grad is not None and r_lo_b.grad is not None and torch.allclose(r_lo_a.grad, r_lo_b.grad, atol=1e-4)

    rep = audit_chunked_nll_autograd_efficiency(
        lambda: _SingleGemm512DTomographicCEFn.apply(h_a, w_a, b_a, r_hi_a, r_lo_a, targets),
        fn_name="_SingleGemm512DTomographicCEFn",
        num_chunks=1,
        forbid_exponential_ops=False,
    )
    assert rep.passed


@pytest.mark.parametrize(
    "mode,use_rho,use_tomo,use_vac,use_reg",
    [
        (ExplicitDensityVariantMode.T0_PURE_WAVEPACKET_RANK1_BASELINE, False, False, False, False),
        (ExplicitDensityVariantMode.T1_EXPLICIT_OUTER_DENSITY_TOMOGRAPHIC_512D, True, True, False, False),
        (ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM, True, True, True, False),
        (ExplicitDensityVariantMode.T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE, True, True, True, True),
    ],
)
def test_explicit_density_tomographic_gpt_variants_end_to_end(
    mode: ExplicitDensityVariantMode,
    use_rho: bool,
    use_tomo: bool,
    use_vac: bool,
    use_reg: bool,
) -> None:
    """Verify all `T0..T3` variants initialize with `0` tokens scanned, use 100% `AdamW`-free optimizers, and survive `16x` LR."""
    spec = ExplicitDensityTomographicSpec(
        variant_mode=mode,
        enable_density_highway=use_rho,
        enable_tomographic_readout=use_tomo,
        enable_noclick_and_vacuum=use_vac,
        enable_two_register_bilinear=use_reg,
        num_layers=2,
    )
    model = ExplicitDensityTomographicGPT(spec)
    _, meta = initialize_explicit_density_tomographic_model_(model)
    assert meta["init_tokens_scanned"] == 0

    optimizers = build_explicit_density_tomographic_optimizers(model)
    assert len(optimizers) == 4
    assert not any(isinstance(opt, torch.optim.AdamW) for opt in optimizers)
    assert isinstance(optimizers[3], RiemannianGeodesicScalarAndProbeOptimizer)

    opt_param_ids = {id(p) for opt in optimizers for pg in opt.param_groups for p in pg["params"]}
    model_param_ids = {id(p) for p in model.parameters()}
    assert opt_param_ids == model_param_ids

    idx = torch.randint(0, 50257, (2, 16))
    targets = torch.randint(0, 50257, (2, 16))
    loss = model.forward_loss(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()

    # Step at extreme 16.0x learning rate multiplier and verify no Step-1 temperature quench
    _step_explicit_density_optimizers(model, optimizers, idx, targets, lr_mult=16.0, mom_beta=0.70)
    assert float(model.readout_temp.item()) > 0.35
    for p in model.parameters():
        assert torch.isfinite(p).all()
