"""Unit & Hypothesis Tests for `EXPERIMENT_PLAN_055.md` (`Wave 40` — Tied Holographic Born & Rare Sampling)."""

from __future__ import annotations

import inspect

import torch

from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize
from qdlblocks.models.tied_holo_born_complex_gpt import (
    RareTokenImportanceWindowSampler,
    TiedHoloBornChunkedNLLFn,
    TiedHoloBornComplexRoPEGPT,
    TiedHoloBornConfig,
    build_tied_holo_born_optimizers,
    compute_soft_manifold_and_isotropy_penalties,
    eval_tied_energy_born_probs,
    eval_tied_holo_born_probs,
)


def test_h1_tied_holo_born_self_overlap_and_prob_sum() -> None:
    """Verify unit self-overlap `|<e_v | e_v>_C|^2 == 1.0`, exact probability sum, and zero `exp`/`softmax` calls."""
    torch.manual_seed(5501)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_eff, d_c = 256, 64
    raw_re = torch.randn(v_eff, d_c, device=device, dtype=torch.float32)
    raw_im = torch.randn(v_eff, d_c, device=device, dtype=torch.float32)
    e_re, e_im = complex_l2_normalize(raw_re, raw_im, dim=-1)

    # 1. Self-overlap without legacy 1/sqrt(2) shrinkage must be 1.000000 (vs 0.25 intensity / 0.50 amplitude with 1/sqrt(2))
    e_cat = torch.cat([e_re.float(), e_im.float()], dim=-1)
    self_amp = (e_cat * e_cat).sum(dim=-1)
    self_overlap_sq = self_amp.square()
    max_self_overlap_err = float((self_overlap_sq - 1.0).abs().max().item())
    assert max_self_overlap_err < 1e-5, f"max_self_overlap_err={max_self_overlap_err}"

    # Verify that legacy 1/sqrt(2) scaling on both ket and bra halved amplitude to 0.50 (intensity 0.25)
    legacy_shrunk_amp = ((e_cat * (2.0 ** -0.5)) * (e_cat * (2.0 ** -0.5))).sum(dim=-1)
    assert float((legacy_shrunk_amp - 0.5).abs().max().item()) < 1e-5

    # 2. Verify zero exp / softmax / logsumexp calls in eval_tied_holo_born_probs and TiedHoloBornChunkedNLLFn
    for obj in (eval_tied_holo_born_probs, eval_tied_energy_born_probs, TiedHoloBornChunkedNLLFn):
        src = inspect.getsource(obj)
        for forbidden in ("torch.exp(", ".exp(", "softmax(", "logsumexp("):
            assert forbidden not in src, f"Forbidden `{forbidden}` found in {obj}"

    # 3. Verify exact probability normalization for both Holographic Born and Energy Born
    phi_re, phi_im = complex_l2_normalize(
        torch.randn(4, 16, d_c, device=device),
        torch.randn(4, 16, d_c, device=device),
        dim=-1,
    )
    prior_w = torch.rand(v_eff, device=device, dtype=torch.float32).clamp_min(0.05)
    probs_holo = eval_tied_holo_born_probs(phi_re, phi_im, e_re, e_im, prior_w, boson_p=12.0, holo_alpha=0.85)
    probs_energy = eval_tied_energy_born_probs(phi_re, phi_im, e_re, e_im, prior_w, boson_p=12.0)

    assert torch.all(probs_holo > 0.0)
    assert float((probs_holo.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5
    assert float((probs_energy.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5

    # 4. Verify sequence-side out_operator acts on (B, T, d_complex) rather than (V, d_complex)
    cfg = TiedHoloBornConfig(
        vocab_size=264,
        valid_vocab_size=v_eff,
        d_complex=d_c,
        n_registers=2,
        d_register=32,
        num_layers=2,
        block_size=16,
    )
    model = TiedHoloBornComplexRoPEGPT(cfg).to(device)
    out_re, out_im = model.apply_readout_operator(phi_re, phi_im, phi_re, phi_im)
    assert out_re.shape == (4, 16, d_c) and out_im.shape == (4, 16, d_c)


def test_h1_chunked_holo_born_autograd_parity() -> None:
    """Verify `TiedHoloBornChunkedNLLFn` (`vocab_chunk_size=64`, `V=256`) matches unchunked float32 autograd."""
    torch.manual_seed(5502)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_tok, v_eff, d_c = 32, 256, 32
    phi_re_0, phi_im_0 = complex_l2_normalize(
        torch.randn(n_tok, d_c, device=device, dtype=torch.float32),
        torch.randn(n_tok, d_c, device=device, dtype=torch.float32),
        dim=-1,
    )
    e_re_0, e_im_0 = complex_l2_normalize(
        torch.randn(v_eff, d_c, device=device, dtype=torch.float32),
        torch.randn(v_eff, d_c, device=device, dtype=torch.float32),
        dim=-1,
    )
    phi_cat_0 = torch.cat([phi_re_0.float(), phi_im_0.float()], dim=-1)
    e_cat_0 = torch.cat([e_re_0.float(), e_im_0.float()], dim=-1)
    targets = torch.randint(0, v_eff, (n_tok,), device=device, dtype=torch.int64)
    prior_w = torch.rand(v_eff, device=device, dtype=torch.float32).clamp_min(0.1)
    non_uniform_weights = torch.rand(n_tok, device=device, dtype=torch.float32).clamp_min(0.2)

    TiedHoloBornChunkedNLLFn.vocab_chunk_size = 64
    for tok_w in (None, non_uniform_weights):
        phi_c = phi_cat_0.detach().clone().requires_grad_(True)
        e_c = e_cat_0.detach().clone().requires_grad_(True)
        loss_chunk = TiedHoloBornChunkedNLLFn.apply(phi_c, e_c, targets, prior_w, 12.0, 0.85, tok_w)
        loss_chunk.backward()

        phi_ref = phi_cat_0.detach().clone().requires_grad_(True)
        e_ref = e_cat_0.detach().clone().requires_grad_(True)
        probs_ref = eval_tied_holo_born_probs(
            phi_ref[:, :d_c],
            phi_ref[:, d_c:],
            e_ref[:, :d_c],
            e_ref[:, d_c:],
            prior_w,
            boson_p=12.0,
            holo_alpha=0.85,
        )
        nll_ref = -torch.log(probs_ref.gather(1, targets.unsqueeze(1)).squeeze(1))
        if tok_w is None:
            loss_ref = nll_ref.mean()
        else:
            w_n = tok_w / tok_w.sum()
            loss_ref = (w_n * nll_ref).sum()
        loss_ref.backward()

        fwd_err = abs(float(loss_chunk.item()) - float(loss_ref.item()))
        assert fwd_err < 1e-5, f"fwd_err={fwd_err}"
        assert phi_c.grad is not None and phi_ref.grad is not None
        assert e_c.grad is not None and e_ref.grad is not None
        grad_phi_err = float((phi_c.grad - phi_ref.grad).abs().max().item())
        grad_e_err = float((e_c.grad - e_ref.grad).abs().max().item())
        assert grad_phi_err < 1e-4, f"grad_phi_err={grad_phi_err}"
        assert grad_e_err < 1e-4, f"grad_e_err={grad_e_err}"


def test_h2_soft_sphere_unitary_and_parseval_isotropy_penalties() -> None:
    """Verify `[:valid_vocab_size]` slicing, square-only unitary penalty, and monotonic `R_emb` rise under `L_iso`."""
    torch.manual_seed(5503)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TiedHoloBornConfig(
        vocab_size=264,
        valid_vocab_size=250,
        d_complex=32,
        n_registers=2,
        d_register=16,
        num_layers=2,
        block_size=16,
    )
    model = TiedHoloBornComplexRoPEGPT(cfg).to(device)
    pen_0, stats_0 = compute_soft_manifold_and_isotropy_penalties(model)
    assert stats_0["sphere_pen"] < 1e-5
    assert stats_0["unitary_pen"] < 1e-5

    # 1. Corrupting padding rows [250:264] must NOT change sphere_pen or iso_pen
    with torch.no_grad():
        model.embed_re.data[250:] = 50.0
        model.embed_im.data[250:] = -50.0
    _, stats_pad = compute_soft_manifold_and_isotropy_penalties(model)
    assert abs(stats_pad["sphere_pen"] - stats_0["sphere_pen"]) < 1e-7
    assert abs(stats_pad["iso_pen"] - stats_0["iso_pen"]) < 1e-7

    # 2. Corrupting rectangular FFN weights must NOT change unitary_pen
    with torch.no_grad():
        model.blocks[0].ffn.beam1.weight_re.data.mul_(4.0)
        model.blocks[0].ffn.proj_down.weight_im.data.mul_(3.0)
    _, stats_ffn = compute_soft_manifold_and_isotropy_penalties(model)
    assert abs(stats_ffn["unitary_pen"] - stats_0["unitary_pen"]) < 1e-7

    # 3. Corrupting square out_operator must increase unitary_pen and produce grad on out_operator only
    with torch.no_grad():
        model.out_operator.weight_re.data.mul_(2.0)
    model.zero_grad(set_to_none=True)
    pen_sq, stats_sq = compute_soft_manifold_and_isotropy_penalties(model)
    assert stats_sq["unitary_pen"] > 0.1
    pen_sq.backward()
    assert model.out_operator.weight_re.grad is not None
    assert float(model.out_operator.weight_re.grad.abs().max().item()) > 1e-4
    assert model.blocks[0].ffn.beam1.weight_re.grad is None

    # 4. 25 gradient steps on L_iso monotonically increase R_emb from a collapsed low-rank unit-sphere table
    with torch.no_grad():
        z_re = 0.12 * torch.randn((250, 32), device=device, dtype=torch.float32)
        z_im = 0.12 * torch.randn((250, 32), device=device, dtype=torch.float32)
        z_re[:, :2] *= 10.0
        z_im[:, :2] *= 10.0
        e_re_u, e_im_u = complex_l2_normalize(z_re, z_im, dim=-1)
        model.embed_re.data[:250].copy_(e_re_u)
        model.embed_im.data[:250].copy_(e_im_u)

    ranks: list[float] = [model.compute_sharp_eff_rank()]
    opt_iso = torch.optim.SGD([model.embed_re, model.embed_im], lr=2.5)
    for _ in range(25):
        opt_iso.zero_grad(set_to_none=True)
        pen_step, _ = compute_soft_manifold_and_isotropy_penalties(model)
        pen_step.backward()
        opt_iso.step()
        ranks.append(model.compute_sharp_eff_rank())

    for idx in range(len(ranks) - 1):
        assert ranks[idx + 1] > ranks[idx], f"Non-monotonic R_emb at step {idx}: {ranks}"
    assert ranks[-1] > ranks[0] * 2.5, f"Expected substantial R_emb recovery: {ranks[0]:.2f} -> {ranks[-1]:.2f}"


def test_h3_unbiased_rare_token_importance_sampler() -> None:
    """Verify Laplace smoothing on unseen tokens, `> 1.5x` rare-token exposure gain, and `< 1e-6` IS bias."""
    torch.manual_seed(5504)
    v_eff = 200
    # Synthetic Zipf stream where tokens 0..99 appear frequently and tokens 100..199 are rare or unseen
    ranks = torch.arange(1, 101, dtype=torch.float64)
    zipf_probs = (ranks ** -1.25) / (ranks ** -1.25).sum()
    heavy_tokens = torch.multinomial(zipf_probs, num_samples=15000, replacement=True)
    rare_tokens = torch.randint(100, 170, (400,), dtype=torch.int64)  # tokens 170..199 have count == 0
    ref_stream = torch.cat([heavy_tokens, rare_tokens], dim=0)

    sampler = RareTokenImportanceWindowSampler(
        reference_tokens=ref_stream,
        valid_vocab_size=v_eff,
        alpha_rare=0.65,
        pool_multiplier=8,
        laplace_eps=1.0,
    )
    # 1. Zero-count tokens (170..199) have strictly positive Laplace-smoothed probability and finite rarity
    assert int((sampler.token_counts[170:] == 0).sum().item()) == 30
    assert torch.all(sampler.unigram_probs > 0.0)
    assert torch.all(torch.isfinite(sampler.token_rarity))
    assert float(sampler.token_rarity[180].item()) > float(sampler.token_rarity[0].item())

    # 2. Construct a buffer with mixed frequent and rare windows and verify > 1.5x bottom-50% exposure gain
    freq_windows = torch.randint(0, 30, (28 * 33,), dtype=torch.int64)
    rare_windows = torch.randint(110, 195, (4 * 33,), dtype=torch.int64)
    token_buf = torch.cat([freq_windows, rare_windows], dim=0)

    unif_sampler = RareTokenImportanceWindowSampler(
        reference_tokens=ref_stream,
        valid_vocab_size=v_eff,
        alpha_rare=0.0,
        pool_multiplier=8,
        laplace_eps=1.0,
    )
    rare_hits_is = 0
    rare_hits_unif = 0
    total_sampled = 0
    for step_seed in range(1, 41):
        _, y_is, _ = sampler.sample_batch(token_buf, batch_size=8, seq_len=32, step_seed=step_seed)
        _, y_un, _ = unif_sampler.sample_batch(token_buf, batch_size=8, seq_len=32, step_seed=step_seed)
        rare_hits_is += int(sampler.rare_bottom50_mask[y_is].sum().item())
        rare_hits_unif += int(sampler.rare_bottom50_mask[y_un].sum().item())
        total_sampled += int(y_is.numel())

    exposure_ratio = float(rare_hits_is) / float(max(1, rare_hits_unif))
    assert exposure_ratio > 1.5, f"Expected > 1.5x rare token exposure gain, got {exposure_ratio:.3f}x"

    # 3. Verify exact mathematical unbiasedness: sum_{i=1}^M q_i * (1 / (M * q_i)) * L(W_i) == (1/M) sum_i L(W_i)
    m_pool = 64
    cand_y = torch.randint(0, v_eff, (m_pool, 32), dtype=torch.int64)
    window_losses = torch.randn(m_pool, dtype=torch.float32) + 6.0
    q_probs, unnorm_inv_w = sampler.compute_pool_distribution(cand_y)
    is_expectation = float((q_probs * unnorm_inv_w * window_losses).sum().item())
    unif_mean = float(window_losses.mean().item())
    assert abs(is_expectation - unif_mean) < 1e-6, f"IS bias={abs(is_expectation - unif_mean)}"


def _collect_block_dtypes(
    model: TiedHoloBornComplexRoPEGPT,
    idx: torch.Tensor,
) -> list[torch.dtype]:
    """Collect intermediate complex register dtypes inside `TiedHoloBornComplexRoPEGPT` without closures."""
    b_sz, seq_t = idx.shape
    d_c = int(model.cfg.d_complex)
    h_num, dh = int(model.cfg.n_registers), int(model.cfg.d_register)
    psi0_re = model.embed_re.index_select(0, idx.reshape(-1)).view(b_sz, seq_t, d_c)
    psi0_im = model.embed_im.index_select(0, idx.reshape(-1)).view(b_sz, seq_t, d_c)
    psi_re, psi_im = complex_l2_normalize(psi0_re, psi0_im, dim=-1)
    dtypes = [psi_re.dtype, psi_im.dtype]
    v0_re, v0_im = None, None
    for l_idx, block in enumerate(model.blocks):
        psi_re, psi_im, vr, vi = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
        dtypes.extend([psi_re.dtype, psi_im.dtype, vr.dtype, vi.dtype])
        if l_idx == 0:
            v0_re = vr.view(b_sz, h_num, seq_t, dh)
            v0_im = vi.view(b_sz, h_num, seq_t, dh)
    phi_re, phi_im = model.apply_readout_operator(psi_re, psi_im, psi0_re, psi0_im)
    e_re, e_im = model.get_tied_vocab_wavefunctions()
    dtypes.extend([phi_re.dtype, phi_im.dtype, e_re.dtype, e_im.dtype])
    return dtypes


def test_h4_end_to_end_bf16_smoke_and_rank_retention() -> None:
    """Run a 5-step `bfloat16` training loop verifying 0 `float32` activation leaks, loss drop, and `R_emb` retention."""
    torch.manual_seed(5505)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TiedHoloBornConfig(
        vocab_size=264,
        valid_vocab_size=256,
        d_complex=64,
        n_registers=2,
        d_register=32,
        num_layers=2,
        block_size=32,
        vocab_chunk_size=128,
    )
    model = TiedHoloBornComplexRoPEGPT(cfg).to(device)
    opt_muon, opt_adam = build_tied_holo_born_optimizers(model, base_muon_lr=0.02, base_embed_lr=0.01)

    idx = torch.randint(0, 256, (4, 32), device=device)
    targets = torch.randint(0, 256, (4, 32), device=device)

    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            dtypes = _collect_block_dtypes(model, idx)
        float32_leaks = sum(1 for dt in dtypes if dt == torch.float32)
        assert float32_leaks == 0, f"Found {float32_leaks} float32 activation leaks: {dtypes}"

    r_init = model.compute_sharp_eff_rank()
    losses: list[float] = []
    for _ in range(5):
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_loss = model(idx, targets=targets, return_logits=False)
        assert nll_loss is not None and torch.isfinite(nll_loss)
        reg_loss, _ = compute_soft_manifold_and_isotropy_penalties(model)
        total_loss = nll_loss + reg_loss
        total_loss.backward()
        opt_muon.step()
        opt_adam.step()
        losses.append(float(nll_loss.detach().item()))

    r_final = model.compute_sharp_eff_rank()
    assert losses[-1] < losses[0], f"Expected loss decrease over 5 steps: {losses}"
    assert r_final >= 0.85 * r_init, f"Expected rank retention: r_init={r_init:.2f}, r_final={r_final:.2f}"


if __name__ == "__main__":
    for fn in (
        test_h1_tied_holo_born_self_overlap_and_prob_sum,
        test_h1_chunked_holo_born_autograd_parity,
        test_h2_soft_sphere_unitary_and_parseval_isotropy_penalties,
        test_h3_unbiased_rare_token_importance_sampler,
        test_h4_end_to_end_bf16_smoke_and_rank_retention,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 5 Plan 055 unit tests PASSED.")
