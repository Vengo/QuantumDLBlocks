r"""Unit tests for `EXPERIMENT_PLAN_079.md` — Variable-Rank $R$-Mode Uncollapsed Cholesky & Vacuum Subspace (`Gates R1..R4`)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import DEFAULT_SOFTCAP_LOGIT
from qdlblocks.benchmarks.vacuum_subspace_cholesky_lab import _VacuumFourMode512DWavepacketCEFn, vacuum_sphere_rms_norm
from qdlblocks.benchmarks.variable_rank_cholesky_vacuum_lab import (
    VariableRankCholeskySpec,
    VariableRankVacuumCholeskyGPT,
    VariableRankVariantMode,
    build_layer0_variable_rank_cholesky_factor,
    build_variable_rank_cholesky_optimizers,
    compute_rank_adaptive_cholesky_rms_scale,
    compute_rank_invariant_excited_trace_scale,
    fused_variable_rank_cholesky_ket_attention,
)


def test_gate_r1_unrestricted_rank_unit_trace_and_identity_isolation() -> None:
    """Gate R1: Verify exact unit-trace conservation and Identity Packet 7 neighbor isolation across R in {1, 2, 4, 6, 8, 12, 16}."""
    torch.manual_seed(42)
    bsz, seq_len, n_heads, d_h = 2, 12, 8, 32
    raw_wp_a = torch.randn(bsz, seq_len, n_heads, 2, d_h)
    raw_wp_b = raw_wp_a.clone()
    raw_wp_b[:, :6, :, :, :] = torch.randn_like(raw_wp_b[:, :6, :, :, :])
    raw_wp_b[:, 6, -1, :, :] = raw_wp_a[:, 6, -1, :, :]

    wp_re = torch.tensor([0.22, 0.22, 0.14, 0.08, 0.22, 0.22, 0.14, 0.00])
    wp_im = torch.zeros(n_heads)

    q_rot = F.rms_norm(torch.randn(bsz, seq_len, n_heads, 2 * d_h), (2 * d_h,))
    k_rot = F.rms_norm(torch.randn(bsz, seq_len, n_heads, 2 * d_h), (2 * d_h,))
    v_mixed = torch.randn(bsz, seq_len, n_heads, 2 * d_h)
    v_c = v_mixed.view(bsz, seq_len, n_heads, 1, 2, d_h).permute(0, 2, 1, 3, 4, 5)
    vn_re, vn_im = v_c[..., 0, :], v_c[..., 1, :]
    nsq = (vn_re.square().sum(dim=(-2, -1), keepdim=True) + vn_im.square().sum(dim=(-2, -1), keepdim=True)).clamp_min(1e-6).rsqrt()
    r_re, r_im = vn_re * nsq, vn_im * nsq
    berry = torch.zeros(n_heads, 1, d_h)
    born_b = torch.zeros(n_heads)

    for r_modes in (1, 2, 4, 6, 8, 12, 16):
        row_scales = torch.full((max(1, r_modes - 1),), -0.9)
        l0_a_re, l0_a_im = build_layer0_variable_rank_cholesky_factor(raw_wp_a, wp_re, wp_im, row_scales, num_modes=r_modes)
        l0_b_re, l0_b_im = build_layer0_variable_rank_cholesky_factor(raw_wp_b, wp_re, wp_im, row_scales, num_modes=r_modes)

        assert l0_a_re.shape == (bsz, n_heads, seq_len, r_modes, d_h)
        frob = (l0_a_re.square().sum(dim=(-2, -1)) + l0_a_im.square().sum(dim=(-2, -1))).sqrt()
        assert torch.allclose(frob, torch.ones_like(frob), atol=1e-5)

        # Identity Packet h=7 at position t=6 must be strictly invariant to neighbor tokens t < 6 for all R!
        assert torch.allclose(l0_a_re[:, -1, 6], l0_b_re[:, -1, 6], atol=1e-6)
        assert torch.allclose(l0_a_im[:, -1, 6], l0_b_im[:, -1, 6], atol=1e-6)

        high_gates = torch.full((max(1, r_modes - 4),), -0.35)
        la_re, la_im, _, _ = fused_variable_rank_cholesky_ket_attention(
            q_rot, k_rot, (r_re, r_im), v_mixed, high_gates,
            phase_and_born=(berry, born_b, 0.125, True), k_chol=r_modes,
        )
        assert la_re.shape == (bsz, n_heads, seq_len, r_modes, d_h)
        assert la_im.shape == (bsz, n_heads, seq_len, r_modes, d_h)


def test_gate_r2_rank_invariant_anchor_trace_and_progressive_vacuum_parking() -> None:
    """Gate R2: Verify rank-invariant anchor trace share E_{t,0} and progressive vacuum parking monotonicity across R."""
    torch.manual_seed(7)
    raw_wp = torch.randn(2, 16, 8, 2, 32)
    wp_re = torch.full((8,), 0.20)
    wp_im = torch.zeros(8)

    anchor_energies: list[float] = []
    for r_modes in (2, 4, 6, 8, 12, 16):
        row_scales = torch.full((r_modes - 1,), -1.0)
        l0_re, l0_im = build_layer0_variable_rank_cholesky_factor(raw_wp, wp_re, wp_im, row_scales, num_modes=r_modes)
        # Check semantic heads 0..6 where all R-1 excited modes are active
        e0_sem = (l0_re[:, :7, -1, 0].square().sum(dim=-1) + l0_im[:, :7, -1, 0].square().sum(dim=-1)).mean().item()
        anchor_energies.append(e0_sem)

        alpha_r = compute_rank_adaptive_cholesky_rms_scale(r_modes, 64)
        assert abs(alpha_r - 8.0 * math.sqrt(float(r_modes))) < 1e-6

    # Anchor trace share E_{t,0} must stay invariant within < 1e-4 across R in {2, 4, 6, 8, 12, 16}!
    assert max(anchor_energies) - min(anchor_energies) < 1e-4

    # Verify progressive vacuum rest-mass parks higher-index modes deeper in H_vac under equal mode energy
    spec_r12 = VariableRankCholeskySpec(
        variant_mode=VariableRankVariantMode.R5_RANK12_VACUUM_CHOLESKY,
        num_cholesky_modes=12,
        vacuum_dim=4,
    )
    model_r12 = VariableRankVacuumCholeskyGPT(spec_r12)
    equal_vocab = torch.randn(1, 1, 12, 64)
    equal_vocab = F.rms_norm(equal_vocab, (64,))
    u_vac = model_r12.chol_id_vac_mass.view(1, 1, 12, 4)
    _, _, cos_per_mode = vacuum_sphere_rms_norm(equal_vocab, u_vac)
    cos_vals = cos_per_mode.view(-1).tolist()
    # Anchor mode 0 has highest cos(theta_vac), and excited modes r=1..11 monotonically decrease in cos(theta_vac)
    assert cos_vals[0] > cos_vals[1]
    for r_idx in range(1, 11):
        assert cos_vals[r_idx] >= cos_vals[r_idx + 1] - 1e-6
    assert cos_vals[1] > cos_vals[-1]


def test_gate_r3_variable_rank_single_pass_autograd_parity() -> None:
    """Gate R3: Verify _VacuumFourMode512DWavepacketCEFn autograd parity across R in {1, 2, 6, 8, 12} (Rule 9)."""
    torch.manual_seed(99)
    bsz, seq_len, sem_dim, id_dim, car_dim, v_sz = 2, 4, 32, 16, 8, 96
    cap = float(DEFAULT_SOFTCAP_LOGIT)

    for r_modes in (1, 2, 6, 8, 12):
        h_base = torch.randn(bsz, seq_len, sem_dim + car_dim, dtype=torch.float32, requires_grad=True)
        h_sem = torch.randn(bsz, seq_len, sem_dim, dtype=torch.float32, requires_grad=True)
        h_id = torch.randn(bsz, seq_len, r_modes, id_dim, dtype=torch.float32, requires_grad=True)
        ctrl = torch.randn(bsz, seq_len, r_modes, 3, dtype=torch.float32)
        ctrl[:, :, 0, 0] = 0.0
        ctrl.requires_grad_(True)
        w_aug = torch.randn(v_sz, sem_dim + id_dim + car_dim, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, v_sz, (bsz, seq_len))

        loss_custom = _VacuumFourMode512DWavepacketCEFn.apply(h_base, h_sem, h_id, ctrl, w_aug, targets)
        grads_custom = torch.autograd.grad(loss_custom, (h_base, h_sem, h_id, ctrl, w_aug))

        # Reference PyTorch autograd
        w_s, w_i, w_c = w_aug[:, :sem_dim], w_aug[:, sem_dim : sem_dim + id_dim], w_aug[:, sem_dim + id_dim :]
        z_base = torch.matmul(h_base, torch.cat([w_s, w_c], dim=-1).t()).unsqueeze(2)
        z_id = torch.matmul(h_id, w_i.t())
        z_s1 = torch.matmul(h_sem, w_s.t()).unsqueeze(2)
        raw = z_base + ctrl[..., 1:2] * z_id + (0.0 if r_modes == 1 else 1.0) * ctrl[..., 0:1] * z_s1
        logits_r = cap * torch.tanh(raw / cap)
        log_p_r = F.log_softmax(logits_r, dim=-1)
        log_pi = F.log_softmax(ctrl[..., 2], dim=-1).unsqueeze(-1)
        log_mix = torch.logsumexp(log_pi + log_p_r, dim=2)
        loss_ref = F.nll_loss(log_mix.reshape(-1, v_sz), targets.reshape(-1))
        grads_ref = torch.autograd.grad(loss_ref, (h_base, h_sem, h_id, ctrl, w_aug))

        assert abs(float(loss_custom.item()) - float(loss_ref.item())) < 1e-5
        for g_c, g_r in zip(grads_custom, grads_ref):
            assert float((g_c - g_r).abs().max().item()) < 1e-4


def test_gate_r4_end_to_end_variable_rank_forward_backward_and_step() -> None:
    """Gate R4: Verify VariableRankVacuumCholeskyGPT forward, backward, and Riemannian step across R in {1, 2, 6, 8, 12}."""
    torch.manual_seed(123)
    idx = torch.randint(0, 512, (2, 16))
    targets = torch.randint(0, 512, (2, 16))
    active_ids = torch.unique(torch.cat([idx.reshape(-1), targets.reshape(-1)]))

    for mode, r_modes, vac_d in (
        (VariableRankVariantMode.R0_RANK8_NO_VACUUM_SPHERE, 8, 0),
        (VariableRankVariantMode.R1_RANK2_VACUUM_CHOLESKY, 2, 4),
        (VariableRankVariantMode.R3_RANK6_VACUUM_CHOLESKY, 6, 4),
        (VariableRankVariantMode.R4_RANK8_VACUUM_CHOLESKY, 8, 4),
        (VariableRankVariantMode.R5_RANK12_VACUUM_CHOLESKY, 12, 4),
    ):
        spec = VariableRankCholeskySpec(
            variant_mode=mode,
            num_cholesky_modes=r_modes,
            vacuum_dim=vac_d,
            use_layer0_input_cholesky=True,
        )
        model = VariableRankVacuumCholeskyGPT(spec)
        q_trunk_opt, wte_opt, head_opt, adam_opt = build_variable_rank_cholesky_optimizers(model)

        loss0 = float(model.forward_loss(idx, targets).item())
        for _ in range(2):
            q_trunk_opt.zero_grad(set_to_none=True)
            wte_opt.zero_grad(set_to_none=True)
            head_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            loss = model.forward_loss(idx, targets)
            assert torch.isfinite(loss)
            loss.backward()
            q_trunk_opt.step()
            wte_opt.step_active(active_ids, lr_scale=1.0)
            head_opt.step_full(lr_scale=1.0, active_idx=active_ids)
            adam_opt.step()
            model.clamp_identity_dispersion_and_scalars_()
        loss1 = float(model.forward_loss(idx, targets).item())
        assert loss1 < loss0
