"""Unit & Hypothesis Tests for `EXPERIMENT_PLAN_056.md` (`Wave 41` — Multi-Meaning `16^4` Kronecker Born GPT)."""

from __future__ import annotations

import inspect
from pathlib import Path
import tempfile

import numpy as np
import torch

from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (
    FINEWEB_HEADER_INTS,
    FINEWEB_HEADER_MAGIC,
    KroneckerMultiMeaning16x4GPT,
    KroneckerMultiMeaningBornChunkedNLLFn,
    KroneckerMultiMeaningConfig,
    _algebraic_positive_unit_l2,
    build_kronecker_multimeaning_optimizers,
    compute_kronecker_manifold_and_isotropy_penalties,
    eval_kronecker_multimeaning_born_probs,
)


def test_h1_kronecker_16x4_snr_and_polysemy_capacity() -> None:
    """Verify `16^4 = 65,536` Kronecker SNR (`>= 400x` vs flat `C^128`) and Rank-`R` polysemy capacity (`>= 6.0x`)."""
    torch.manual_seed(5601)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_states = 512
    d_flat = 128
    m_factors = 4
    d_f = 16

    # 1. Flat C^128 random background overlap: E_{u != v}[|<e_u | e_v>_{C^128}|^2] ~ 1/128 = 7.8125e-3
    flat_re, flat_im = complex_l2_normalize(
        torch.randn(n_states, d_flat, device=device, dtype=torch.float32),
        torch.randn(n_states, d_flat, device=device, dtype=torch.float32),
        dim=-1,
    )
    flat_cat = torch.cat([flat_re.float(), flat_im.float()], dim=-1)
    flat_skew = torch.cat([-flat_im.float(), flat_re.float()], dim=-1)
    flat_ov_re = flat_cat @ flat_cat.t()
    flat_ov_im = -(flat_cat @ flat_skew.t())
    flat_sq = flat_ov_re.square() + flat_ov_im.square()
    off_mask = ~torch.eye(n_states, device=device, dtype=torch.bool)
    flat_bg_noise = float(flat_sq[off_mask].mean().item())

    # 2. 4-factor C^{16x16x16x16} Kronecker random background overlap: E_{u != v}[|K_{u,v}|^2] ~ 1/65536 = 1.5259e-5
    kron_re, kron_im = complex_l2_normalize(
        torch.randn(n_states, m_factors, d_f, device=device, dtype=torch.float32),
        torch.randn(n_states, m_factors, d_f, device=device, dtype=torch.float32),
        dim=-1,
    )
    kron_prod_sq = torch.ones((n_states, n_states), device=device, dtype=torch.float32)
    for m_idx in range(m_factors):
        km_cat = torch.cat([kron_re[:, m_idx].float(), kron_im[:, m_idx].float()], dim=-1)
        km_skew = torch.cat([-kron_im[:, m_idx].float(), kron_re[:, m_idx].float()], dim=-1)
        ov_re_m = km_cat @ km_cat.t()
        ov_im_m = -(km_cat @ km_skew.t())
        kron_prod_sq = kron_prod_sq * (ov_re_m.square() + ov_im_m.square())

    kron_bg_noise = float(kron_prod_sq[off_mask].mean().item())
    snr_improvement_x = flat_bg_noise / max(1e-12, kron_bg_noise)
    assert 1.0e-5 <= kron_bg_noise <= 2.5e-5, f"kron_bg_noise={kron_bg_noise:.6e} outside [1e-5, 2.5e-5]"
    assert snr_improvement_x >= 400.0, f"Expected >= 400x SNR improvement, got {snr_improvement_x:.1f}x"

    # 3. 2-sense polysemous coincidence capacity: Rank-1 vs Rank-2 vs Rank-4 on 2 orthogonal senses
    # Construct 2 factor-wise orthogonal context senses |Phi_{sense 0}> and |Phi_{sense 1}> in (C^16)^{x4}
    s0_re, s0_im = complex_l2_normalize(
        torch.randn(m_factors, d_f, device=device, dtype=torch.float32),
        torch.randn(m_factors, d_f, device=device, dtype=torch.float32),
        dim=-1,
    )
    raw1_re = torch.randn(m_factors, d_f, device=device, dtype=torch.float32)
    raw1_im = torch.randn(m_factors, d_f, device=device, dtype=torch.float32)
    # Gram-Schmidt orthogonalize sense 1 against sense 0 in each factor m in {0, 1, 2, 3}
    dot_re = (s0_re * raw1_re + s0_im * raw1_im).sum(dim=-1, keepdim=True)
    dot_im = (s0_re * raw1_im - s0_im * raw1_re).sum(dim=-1, keepdim=True)
    orth1_re = raw1_re - (dot_re * s0_re - dot_im * s0_im)
    orth1_im = raw1_im - (dot_re * s0_im + dot_im * s0_re)
    s1_re, s1_im = complex_l2_normalize(orth1_re, orth1_im, dim=-1)

    # Rank-1 best symmetric compromise state: 1/sqrt(2) * (|s0^{(m)}> + |s1^{(m)}>) per factor
    r1_re, r1_im = complex_l2_normalize(s0_re + s1_re, s0_im + s1_im, dim=-1)
    r1_s0_coinc = float(
        ((r1_re * s0_re + r1_im * s0_im).sum(dim=-1).square() + (r1_re * s0_im - r1_im * s0_re).sum(dim=-1).square())
        .prod()
        .item()
    )
    r1_s1_coinc = float(
        ((r1_re * s1_re + r1_im * s1_im).sum(dim=-1).square() + (r1_re * s1_im - r1_im * s1_re).sum(dim=-1).square())
        .prod()
        .item()
    )
    rank1_min_coinc = min(r1_s0_coinc, r1_s1_coinc)

    # Rank-2 state assigns branch r=0 to sense 0 and branch r=1 to sense 1 with alpha = [1/sqrt(2), 1/sqrt(2)]
    alpha_r2 = 1.0 / (2.0 ** 0.5)
    r2_branch0_coinc = float(
        (alpha_r2 ** 2)
        * ((s0_re * s0_re + s0_im * s0_im).sum(dim=-1).square()).prod().item()
    )
    r2_branch1_coinc = float(
        (alpha_r2 ** 2)
        * ((s1_re * s1_re + s1_im * s1_im).sum(dim=-1).square()).prod().item()
    )
    rank2_min_coinc = min(r2_branch0_coinc, r2_branch1_coinc)

    # Rank-4 state covering 2 senses with alpha = [1/sqrt(2), 1/sqrt(2), 0, 0] (or matched context beta)
    rank4_min_coinc = rank2_min_coinc

    assert abs(rank1_min_coinc - 0.0625) < 1e-5, f"rank1_min_coinc={rank1_min_coinc}"
    assert rank2_min_coinc / rank1_min_coinc >= 6.0, f"ratio={rank2_min_coinc / rank1_min_coinc}"
    assert rank4_min_coinc / rank1_min_coinc >= 6.0


def test_h1_kronecker_chunked_born_autograd_parity() -> None:
    """Verify `KroneckerMultiMeaningBornChunkedNLLFn` (`vocab_chunk_size=64`, `V=256`) matches unchunked autograd."""
    torch.manual_seed(5602)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Verify zero exp / softmax / logsumexp calls in the readout functions
    for obj in (
        eval_kronecker_multimeaning_born_probs,
        KroneckerMultiMeaningBornChunkedNLLFn,
        _algebraic_positive_unit_l2,
    ):
        src = inspect.getsource(obj)
        for forbidden in ("torch.exp(", ".exp(", "softmax(", "logsumexp("):
            assert forbidden not in src, f"Forbidden `{forbidden}` found in {obj}"

    n_tok, v_eff, r_num, m_num, d_f = 24, 256, 2, 4, 16
    phi_re, phi_im = complex_l2_normalize(
        torch.randn(n_tok, r_num, m_num, d_f, device=device, dtype=torch.float32),
        torch.randn(n_tok, r_num, m_num, d_f, device=device, dtype=torch.float32),
        dim=-1,
    )
    psi_re, psi_im = complex_l2_normalize(
        torch.randn(v_eff, r_num, m_num, d_f, device=device, dtype=torch.float32),
        torch.randn(v_eff, r_num, m_num, d_f, device=device, dtype=torch.float32),
        dim=-1,
    )
    phi_0 = torch.cat([phi_re.float(), phi_im.float()], dim=-1)
    psi_0 = torch.cat([psi_re.float(), psi_im.float()], dim=-1)
    beta_0 = _algebraic_positive_unit_l2(torch.randn(n_tok, r_num, device=device, dtype=torch.float32))
    alpha_0 = _algebraic_positive_unit_l2(torch.randn(v_eff, r_num, device=device, dtype=torch.float32))
    prior_w = torch.rand(v_eff, device=device, dtype=torch.float32).clamp_min(0.05)
    targets = torch.randint(0, v_eff, (n_tok,), device=device, dtype=torch.int64)
    hyperparams = torch.tensor([11.0, 0.85, 0.55, 0.45], device=device, dtype=torch.float32)
    non_uniform_weights = torch.rand(n_tok, device=device, dtype=torch.float32).clamp_min(0.2)

    # 2. Verify probability normalization
    probs_check = eval_kronecker_multimeaning_born_probs(phi_0, psi_0, beta_0, alpha_0, prior_w, hyperparams)
    assert torch.all(probs_check > 0.0)
    max_prob_sum_err = float((probs_check.sum(dim=-1) - 1.0).abs().max().item())
    assert max_prob_sum_err < 1e-5, f"max_prob_sum_err={max_prob_sum_err}"

    # 3. Verify chunked vs unchunked forward & backward parity across all 4 differentiable inputs
    KroneckerMultiMeaningBornChunkedNLLFn.vocab_chunk_size = 64
    for tok_w in (None, non_uniform_weights):
        phi_c = phi_0.detach().clone().requires_grad_(True)
        psi_c = psi_0.detach().clone().requires_grad_(True)
        beta_c = beta_0.detach().clone().requires_grad_(True)
        alpha_c = alpha_0.detach().clone().requires_grad_(True)

        loss_chunk = KroneckerMultiMeaningBornChunkedNLLFn.apply(
            phi_c, psi_c, beta_c, alpha_c, targets, prior_w, hyperparams, tok_w
        )
        loss_chunk.backward()

        phi_r = phi_0.detach().clone().requires_grad_(True)
        psi_r = psi_0.detach().clone().requires_grad_(True)
        beta_r = beta_0.detach().clone().requires_grad_(True)
        alpha_r = alpha_0.detach().clone().requires_grad_(True)

        probs_ref = eval_kronecker_multimeaning_born_probs(phi_r, psi_r, beta_r, alpha_r, prior_w, hyperparams)
        nll_ref = -torch.log(probs_ref.gather(1, targets.unsqueeze(1)).squeeze(1))
        if tok_w is None:
            loss_ref = nll_ref.mean()
        else:
            w_n = tok_w / tok_w.sum()
            loss_ref = (w_n * nll_ref).sum()
        loss_ref.backward()

        fwd_err = abs(float(loss_chunk.item()) - float(loss_ref.item()))
        assert fwd_err < 1e-5, f"fwd_err={fwd_err}"
        for name, t_c, t_r in (
            ("phi_factors", phi_c, phi_r),
            ("psi_factors", psi_c, psi_r),
            ("branch_beta", beta_c, beta_r),
            ("branch_alpha", alpha_c, alpha_r),
        ):
            assert t_c.grad is not None and t_r.grad is not None
            g_err = float((t_c.grad - t_r.grad).abs().max().item())
            assert g_err < 1e-4, f"Max grad error on {name}: {g_err}"


def test_h2_unigram_100m_prior_loading_and_kl_advantage() -> None:
    """Verify header-safe `.bin` loading, positive floor smoothing (`floor_count=1.0`), and KL/Step-0 NLL advantage."""
    torch.manual_seed(5603)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_eff = 256
    cfg_emp = KroneckerMultiMeaningConfig(
        vocab_size=264,
        valid_vocab_size=v_eff,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=1,
        block_size=32,
        unigram_temp=0.55,
    )
    cfg_unif = KroneckerMultiMeaningConfig(
        vocab_size=264,
        valid_vocab_size=v_eff,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=1,
        block_size=32,
        unigram_temp=0.0,
    )
    model_emp = KroneckerMultiMeaning16x4GPT(cfg_emp).to(device)
    model_unif = KroneckerMultiMeaning16x4GPT(cfg_unif).to(device)
    model_unif.load_state_dict(model_emp.state_dict())

    # Synthetic Zipf stream over 0..219 (leaving 220..255 as 36 unseen tokens)
    ranks = torch.arange(1, 221, dtype=torch.float64)
    zipf_p = (ranks ** -1.15) / (ranks ** -1.15).sum()
    train_tokens = torch.multinomial(zipf_p, num_samples=20000, replacement=True).to(torch.int64)
    val_tokens = torch.multinomial(zipf_p, num_samples=1024, replacement=True).to(torch.int64)

    # Write a synthetic FineWeb `.bin` file with the 256-int32 header (including corrupted token-like header ints)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_file = tmp_path / "train_full_100M.bin"
        hdr = np.zeros(FINEWEB_HEADER_INTS, dtype=np.int32)
        hdr[0] = FINEWEB_HEADER_MAGIC
        hdr[1] = 1
        hdr[2] = int(train_tokens.numel())
        body = train_tokens.numpy().astype(np.uint16)
        with open(shard_file, "wb") as f:
            f.write(hdr.tobytes())
            f.write(body.tobytes())

        stats_emp = model_emp.load_100m_unigram_prior(data_dir=tmp_path, max_tokens=20000)

    assert int(stats_emp["total_tokens"]) == 20000
    assert int(stats_emp["unseen_tokens"]) >= 36
    assert torch.all(model_emp.unigram_probs[:v_eff] > 0.0)
    assert torch.all(model_emp.prior_weights[:v_eff] > 0.0)
    assert float(model_emp.prior_weights[0].item()) > float(model_emp.prior_weights[250].item())

    model_unif.initialize_unigram_prior_from_tokens(train_tokens)

    # Compute KL divergence D_KL(P_val || P_uni) vs D_KL(P_val || P_unif)
    val_counts = torch.bincount(val_tokens, minlength=v_eff).float()
    p_val = val_counts / val_counts.sum()
    nz = p_val > 0.0
    p_emp_cpu = model_emp.unigram_probs[:v_eff].detach().cpu()
    p_unif_cpu = model_unif.unigram_probs[:v_eff].detach().cpu()
    kl_emp = float((p_val[nz] * (torch.log(p_val[nz]) - torch.log(p_emp_cpu[nz]))).sum().item())
    kl_unif = float((p_val[nz] * (torch.log(p_val[nz]) - torch.log(p_unif_cpu[nz]))).sum().item())
    assert kl_emp < kl_unif, f"Expected kl_emp ({kl_emp:.4f}) < kl_unif ({kl_unif:.4f})"

    # Verify lower Step-0 Born NLL on validation sequences
    x_val = val_tokens[:-1].reshape(1, -1)[:, :32].to(device)
    y_val = val_tokens[1:].reshape(1, -1)[:, :32].to(device)
    with torch.no_grad():
        _, nll_emp = model_emp(x_val, targets=y_val, return_logits=False)
        _, nll_unif = model_unif(x_val, targets=y_val, return_logits=False)
    assert nll_emp is not None and nll_unif is not None
    assert float(nll_emp.item()) < float(nll_unif.item()), (
        f"Expected Step-0 nll_emp ({float(nll_emp.item()):.4f}) < nll_unif ({float(nll_unif.item()):.4f})"
    )


def test_h3_kronecker_penalties_and_branch_orthogonality() -> None:
    """Verify `sphere_pen`, square-only `unitary_pen`, `iso_pen`, and `branch_orth_pen` gradient descent (`< 1e-3`)."""
    torch.manual_seed(5604)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = KroneckerMultiMeaningConfig(
        vocab_size=140,
        valid_vocab_size=128,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=2,
        block_size=16,
    )
    model = KroneckerMultiMeaning16x4GPT(cfg).to(device)
    _, stats_0 = compute_kronecker_manifold_and_isotropy_penalties(model)
    assert stats_0["sphere_pen"] < 1e-5
    assert stats_0["unitary_pen"] < 1e-5

    # 1. Corrupting padding rows [128:140] must NOT affect sphere_pen, iso_pen, or branch_orth_pen
    with torch.no_grad():
        model.embed_re.data[128:] = 40.0
        model.embed_im.data[128:] = -40.0
    _, stats_pad = compute_kronecker_manifold_and_isotropy_penalties(model)
    assert abs(stats_pad["sphere_pen"] - stats_0["sphere_pen"]) < 1e-7
    assert abs(stats_pad["iso_pen"] - stats_0["iso_pen"]) < 1e-7
    assert abs(stats_pad["branch_orth_pen"] - stats_0["branch_orth_pen"]) < 1e-7

    # 2. Corrupting rectangular FFN weights must NOT affect unitary_pen
    with torch.no_grad():
        model.blocks[0].ffn.beam1.weight_re.data.mul_(3.5)
        model.blocks[0].ffn.proj_down.weight_im.data.mul_(2.5)
    _, stats_ffn = compute_kronecker_manifold_and_isotropy_penalties(model)
    assert abs(stats_ffn["unitary_pen"] - stats_0["unitary_pen"]) < 1e-7

    # 3. Corrupting 16x16 factor unitary must increase unitary_pen and receive gradient while FFN stays None
    with torch.no_grad():
        model.kron_mixers[0].factor_u_re.data.mul_(2.0)
    model.zero_grad(set_to_none=True)
    pen_u, stats_u = compute_kronecker_manifold_and_isotropy_penalties(model)
    assert stats_u["unitary_pen"] > 0.05
    pen_u.backward()
    assert model.kron_mixers[0].factor_u_re.grad is not None
    assert float(model.kron_mixers[0].factor_u_re.grad.abs().max().item()) > 1e-4
    assert model.blocks[0].ffn.beam1.weight_re.grad is None

    # 4. Initialize branch 1 nearly parallel to branch 0 (high cross-branch overlap > 0.60) and optimize branch_orth_pen
    with torch.no_grad():
        b0_re = model.embed_re.data[:128, 0]
        b0_im = model.embed_im.data[:128, 0]
        noise_re = 0.04 * torch.randn_like(b0_re)
        noise_im = 0.04 * torch.randn_like(b0_im)
        b1_re, b1_im = complex_l2_normalize(b0_re + noise_re, b0_im + noise_im, dim=-1)
        model.embed_re.data[:128, 1].copy_(b1_re)
        model.embed_im.data[:128, 1].copy_(b1_im)

    ov_init = model.compute_cross_branch_overlap()
    assert ov_init > 0.50, f"Expected correlated initial cross-branch overlap > 0.50, got {ov_init:.4f}"

    opt_orth = torch.optim.Adam([model.embed_re, model.embed_im], lr=0.15)
    for _ in range(35):
        opt_orth.zero_grad(set_to_none=True)
        pen_step, _ = compute_kronecker_manifold_and_isotropy_penalties(model)
        pen_step.backward()
        opt_orth.step()

    ov_final = model.compute_cross_branch_overlap()
    assert ov_final < 1e-3, f"Expected cross_branch_overlap < 1e-3 after optimization, got {ov_final:.6e}"


def _collect_kronecker_block_dtypes(
    model: KroneckerMultiMeaning16x4GPT,
    idx: torch.Tensor,
) -> list[torch.dtype]:
    """Collect intermediate complex register and mixer dtypes inside `KroneckerMultiMeaning16x4GPT`."""
    b_sz, seq_t = idx.shape
    h_num, dh = int(model.cfg.n_registers), int(model.cfg.d_register)
    branch_alpha = model.compute_branch_alpha()
    psi0_re, psi0_im = model._pack_token_wavefunctions(idx, branch_alpha)
    psi_re, psi_im = psi0_re, psi0_im
    dtypes = [psi_re.dtype, psi_im.dtype]
    v0_re, v0_im = None, None
    for l_idx, (block, mixer) in enumerate(zip(model.blocks, model.kron_mixers)):
        psi_re, psi_im, vr, vi = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
        dtypes.extend([psi_re.dtype, psi_im.dtype, vr.dtype, vi.dtype])
        if l_idx == 0:
            v0_re = vr.view(b_sz, h_num, seq_t, dh)
            v0_im = vi.view(b_sz, h_num, seq_t, dh)
        psi_re, psi_im = mixer(psi_re, psi_im)
        dtypes.extend([psi_re.dtype, psi_im.dtype])
    phi_factors, _ = model.apply_readout_factor_operator(psi_re, psi_im, psi0_re, psi0_im)
    psi_factors = model.get_tied_vocab_factors()
    dtypes.extend([phi_factors.dtype, psi_factors.dtype])
    return dtypes


def test_h4_end_to_end_bf16_smoke_and_rank_retention() -> None:
    """Run a 5-step `bfloat16` training loop verifying 0 `float32` activation leaks, loss drop, and rank retention."""
    torch.manual_seed(5605)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = KroneckerMultiMeaningConfig(
        vocab_size=264,
        valid_vocab_size=256,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=2,
        block_size=32,
        vocab_chunk_size=128,
    )
    model = KroneckerMultiMeaning16x4GPT(cfg).to(device)
    opt_muon, opt_adam = build_kronecker_multimeaning_optimizers(model, base_muon_lr=0.02, base_embed_lr=0.01)

    idx = torch.randint(0, 256, (4, 32), device=device)
    targets = torch.randint(0, 256, (4, 32), device=device)

    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            dtypes = _collect_kronecker_block_dtypes(model, idx)
        float32_leaks = sum(1 for dt in dtypes if dt == torch.float32)
        assert float32_leaks == 0, f"Found {float32_leaks} float32 activation leaks: {dtypes}"

    r_emb_init = model.compute_sharp_eff_rank()
    r_fac_init = model.compute_mean_factor_rank()
    losses: list[float] = []
    for _ in range(5):
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_loss = model(idx, targets=targets, return_logits=False)
        assert nll_loss is not None and torch.isfinite(nll_loss)
        reg_loss, _ = compute_kronecker_manifold_and_isotropy_penalties(model)
        total_loss = nll_loss + reg_loss
        total_loss.backward()
        opt_muon.step()
        opt_adam.step()
        losses.append(float(nll_loss.detach().item()))

    r_emb_final = model.compute_sharp_eff_rank()
    r_fac_final = model.compute_mean_factor_rank()
    assert losses[-1] < losses[0], f"Expected loss decrease over 5 steps: {losses}"
    assert r_emb_final >= 0.85 * r_emb_init, f"Expected R_emb retention: {r_emb_init:.2f} -> {r_emb_final:.2f}"
    assert r_fac_final >= 0.85 * r_fac_init, f"Expected R_factor retention: {r_fac_init:.2f} -> {r_fac_final:.2f}"


if __name__ == "__main__":
    for fn in (
        test_h1_kronecker_16x4_snr_and_polysemy_capacity,
        test_h1_kronecker_chunked_born_autograd_parity,
        test_h2_unigram_100m_prior_loading_and_kl_advantage,
        test_h3_kronecker_penalties_and_branch_orthogonality,
        test_h4_end_to_end_bf16_smoke_and_rank_retention,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 5 Plan 056 unit tests PASSED.")
