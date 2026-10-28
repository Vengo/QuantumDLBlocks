"""Unit & Hypothesis Tests for `EXPERIMENT_PLAN_057.md` (`Wave 42` — Calibrated Kronecker Born GPT & HParam Search)."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(ROOT / "src"), str(ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize  # noqa: E402
from qdlblocks.models.calibrated_kronecker_born_gpt import (  # noqa: E402
    CalibratedKroneckerBornChunkedNLLFn,
    CalibratedKroneckerBornGPT,
    CalibratedKroneckerConfig,
    _algebraic_positive_povm_weights,
    build_calibrated_kronecker_optimizers,
    compute_calibrated_kronecker_penalties,
    eval_calibrated_kronecker_born_probs,
)
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (  # noqa: E402
    _algebraic_positive_unit_l2,
)
from scripts.kronecker_born_hyperparam_search import (  # noqa: E402
    HyperparamTrialSpec,
    SearchRunnerConfig,
    build_validation_batches,
    run_hyperparam_search_suite,
)


def test_h1_procrustes_carrier_alignment_and_step0_unigram_fit() -> None:
    """Verify Procrustes carrier fidelity (`1.0 ± 1e-5`), unitarity (`< 1e-5`), and Step-0 KL (`< 0.005` nats)."""
    torch.manual_seed(5701)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_eff = 4096
    cfg = CalibratedKroneckerConfig(
        vocab_size=4104,
        valid_vocab_size=v_eff,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=2,
        block_size=32,
        unigram_temp=0.90,
        carrier_rho_max=0.25,
    )

    # Steep Zipf token stream over 0..3600 (leaving 3601..4095 unseen) so D_KL(P_target || Uniform) > 1.8 nats
    ranks = torch.arange(1, 3601, dtype=torch.float64)
    zipf_p = (ranks ** -1.35) / (ranks ** -1.35).sum()
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(5701)
    stream_tokens = torch.multinomial(zipf_p, num_samples=50000, replacement=True, generator=gen_cpu).to(
        torch.int64
    )

    # 1. Uncalibrated Haar baseline
    model_uncal = CalibratedKroneckerBornGPT(cfg).to(device)
    assert not hasattr(model_uncal, "prior_weights"), "CalibratedKroneckerBornGPT must have zero frozen prior_weights buffer"
    model_uncal.update_unigram_probs_from_tokens(stream_tokens)
    eval_seqs = stream_tokens[:128].reshape(4, 32).to(device)
    kl_uncal = model_uncal.compute_empirical_output_kl_to_target(eval_seqs)
    assert kl_uncal > 1.5, f"Expected uncalibrated Haar KL > 1.5 nats, got {kl_uncal:.4f}"

    # 2. Trainable POVM only (rho_max = 0.0)
    model_povm = CalibratedKroneckerBornGPT(cfg).to(device)
    stats_povm = model_povm.calibrate_readout_to_random_attention_unigram(
        token_stream=stream_tokens,
        num_batches=2,
        seq_len=32,
        carrier_rho_max=0.0,
        calibrate_povm=True,
        seed=5701,
    )
    assert abs(stats_povm["carrier_alignment_fidelity"] - 1.0) < 1e-5
    assert stats_povm["unitary_frobenius_err"] < 1e-5
    assert stats_povm["step0_kl_to_target_unigram"] < 0.005, (
        f"Expected trainable_povm_only Step-0 KL < 0.005, got {stats_povm['step0_kl_to_target_unigram']:.6f}"
    )

    # 3. Hybrid Procrustes + Carrier Angle (rho_max = 0.25) + Trainable POVM
    model_hyb = CalibratedKroneckerBornGPT(cfg).to(device)
    stats_hyb = model_hyb.calibrate_readout_to_random_attention_unigram(
        token_stream=stream_tokens,
        num_batches=2,
        seq_len=32,
        carrier_rho_max=0.25,
        calibrate_povm=True,
        seed=5701,
    )
    assert abs(stats_hyb["carrier_alignment_fidelity"] - 1.0) < 1e-5, (
        f"Expected carrier_alignment_fidelity = 1.0 ± 1e-5, got {stats_hyb['carrier_alignment_fidelity']}"
    )
    assert stats_hyb["pre_procrustes_carrier_fidelity"] < 0.35
    assert stats_hyb["unitary_frobenius_err"] < 1e-5, (
        f"Expected unitary_frobenius_err < 1e-5, got {stats_hyb['unitary_frobenius_err']:.2e}"
    )
    assert stats_hyb["step0_kl_to_target_unigram"] < 0.005, (
        f"Expected hybrid Step-0 KL < 0.005, got {stats_hyb['step0_kl_to_target_unigram']:.6f}"
    )
    assert stats_hyb["mean_factor_rank"] >= 28.0, f"Expected R_factor >= 28.0, got {stats_hyb['mean_factor_rank']:.2f}"
    assert stats_hyb["sharp_eff_rank"] >= 220.0, f"Expected R_emb >= 220.0, got {stats_hyb['sharp_eff_rank']:.2f}"


def test_h1_calibrated_kronecker_chunked_autograd_parity() -> None:
    """Verify `CalibratedKroneckerBornChunkedNLLFn` (`vocab_chunk_size=64, V=256`) matches unchunked autograd across all 5 inputs."""
    torch.manual_seed(5702)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Verify zero exp / softmax / logsumexp calls in the readout functions
    for obj in (
        eval_calibrated_kronecker_born_probs,
        CalibratedKroneckerBornChunkedNLLFn,
        _algebraic_positive_povm_weights,
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
    povm_0 = _algebraic_positive_povm_weights(torch.randn(v_eff, device=device, dtype=torch.float32))
    targets = torch.randint(0, v_eff, (n_tok,), device=device, dtype=torch.int64)
    hyperparams = torch.tensor([11.0, 0.85, 0.55, 0.45], device=device, dtype=torch.float32)
    non_uniform_weights = torch.rand(n_tok, device=device, dtype=torch.float32).clamp_min(0.2)

    # 2. Verify probability normalization
    probs_check = eval_calibrated_kronecker_born_probs(phi_0, psi_0, beta_0, alpha_0, povm_0, hyperparams)
    assert torch.all(probs_check > 0.0)
    max_prob_sum_err = float((probs_check.sum(dim=-1) - 1.0).abs().max().item())
    assert max_prob_sum_err < 1e-5, f"max_prob_sum_err={max_prob_sum_err}"

    # 3. Verify chunked vs unchunked forward & backward parity across all 5 differentiable inputs
    saved_chunk = CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size
    CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = 64
    try:
        for tok_w in (None, non_uniform_weights):
            phi_c = phi_0.detach().clone().requires_grad_(True)
            psi_c = psi_0.detach().clone().requires_grad_(True)
            beta_c = beta_0.detach().clone().requires_grad_(True)
            alpha_c = alpha_0.detach().clone().requires_grad_(True)
            povm_c = povm_0.detach().clone().requires_grad_(True)

            loss_chunk = CalibratedKroneckerBornChunkedNLLFn.apply(
                phi_c, psi_c, beta_c, alpha_c, povm_c, targets, hyperparams, tok_w
            )
            loss_chunk.backward()

            phi_r = phi_0.detach().clone().requires_grad_(True)
            psi_r = psi_0.detach().clone().requires_grad_(True)
            beta_r = beta_0.detach().clone().requires_grad_(True)
            alpha_r = alpha_0.detach().clone().requires_grad_(True)
            povm_r = povm_0.detach().clone().requires_grad_(True)

            probs_ref = eval_calibrated_kronecker_born_probs(
                phi_r, psi_r, beta_r, alpha_r, povm_r, hyperparams
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
            for name, t_c, t_r in (
                ("phi_factors", phi_c, phi_r),
                ("psi_factors", psi_c, psi_r),
                ("branch_beta", beta_c, beta_r),
                ("branch_alpha", alpha_c, alpha_r),
                ("povm_weights", povm_c, povm_r),
            ):
                assert t_c.grad is not None and t_r.grad is not None
                g_err = float((t_c.grad - t_r.grad).abs().max().item())
                assert g_err < 1e-4, f"Max grad error on {name}: {g_err}"
    finally:
        CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = saved_chunk


def _collect_calibrated_block_dtypes(
    model: CalibratedKroneckerBornGPT,
    idx: torch.Tensor,
) -> list[torch.dtype]:
    """Collect intermediate complex register and mixer dtypes inside `CalibratedKroneckerBornGPT`."""
    b_sz, seq_t = idx.shape
    h_num, dh = int(model.cfg.n_registers), int(model.cfg.d_register)
    branch_alpha = model.compute_branch_alpha()
    psi0_re, psi0_im = model._pack_token_wavefunctions(idx, branch_alpha)
    psi_re, psi_im = psi0_re, psi0_im
    dtypes = [psi_re.dtype, psi_im.dtype]
    v0_re: torch.Tensor | None = None
    v0_im: torch.Tensor | None = None
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


def test_h2_trainable_povm_adaptation_and_penalties() -> None:
    """Verify `compute_calibrated_kronecker_penalties` and active `povm_scale_logits` adaptation in `bfloat16`."""
    torch.manual_seed(5703)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = CalibratedKroneckerConfig(
        vocab_size=264,
        valid_vocab_size=256,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=2,
        block_size=32,
        vocab_chunk_size=128,
    )
    model = CalibratedKroneckerBornGPT(cfg).to(device)
    ranks = torch.arange(1, 257, dtype=torch.float64)
    zipf_p = (ranks ** -1.2) / (ranks ** -1.2).sum()
    stream = torch.multinomial(zipf_p, num_samples=8192, replacement=True).to(torch.int64)
    model.calibrate_readout_to_random_attention_unigram(
        token_stream=stream, num_batches=1, seq_len=32, carrier_rho_max=0.22, calibrate_povm=True, seed=5703
    )

    _, stats_0 = compute_calibrated_kronecker_penalties(model)
    assert stats_0["sphere_pen"] < 1e-5
    assert stats_0["unitary_pen"] < 1e-5
    assert stats_0["povm_anchor_pen"] < 1e-8

    opt_muon, opt_adam = build_calibrated_kronecker_optimizers(
        model, base_muon_lr=0.024, base_embed_lr=0.015, base_povm_lr=0.010
    )
    assert len(opt_adam.param_groups) == 3, "Expected 3 AdamW groups (embed, povm_scale_logits, gates)"

    idx = stream[:128].reshape(4, 32).to(device)
    targets = stream[1:129].reshape(4, 32).to(device)

    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            dtypes = _collect_calibrated_block_dtypes(model, idx)
        float32_leaks = sum(1 for dt in dtypes if dt == torch.float32)
        assert float32_leaks == 0, f"Found {float32_leaks} float32 activation leaks: {dtypes}"

    losses: list[float] = []
    for _ in range(5):
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            _, nll_loss = model(idx, targets=targets, return_logits=False)
        assert nll_loss is not None and torch.isfinite(nll_loss)
        reg_loss, _ = compute_calibrated_kronecker_penalties(model)
        total_loss = nll_loss + reg_loss
        total_loss.backward()
        assert model.povm_scale_logits.grad is not None
        assert float(model.povm_scale_logits.grad[:256].abs().max().item()) > 1e-5
        opt_muon.step()
        opt_adam.step()
        losses.append(float(nll_loss.detach().item()))

    _, stats_fin = compute_calibrated_kronecker_penalties(model)
    assert losses[-1] < losses[0], f"Expected loss decrease over 5 steps: {losses}"
    assert stats_fin["povm_drift_std"] > 0.0, "Expected povm_scale_logits to adapt away from Step-0 init"
    assert stats_fin["povm_anchor_pen"] > 0.0


def test_h3_hyperparam_search_trial_runner_smoke() -> None:
    """Run a 2-trial micro-smoke test of `run_hyperparam_search_suite` verifying ranking and schema."""
    torch.manual_seed(5704)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_eff = 256
    ranks = np.arange(1, v_eff + 1, dtype=np.float64)
    probs = (ranks ** -1.2) / np.sum(ranks ** -1.2)
    rng = np.random.default_rng(5704)
    tokens_np = rng.choice(v_eff, size=4096, replace=True, p=probs).astype(np.uint16)
    val_np = rng.choice(v_eff, size=1024, replace=True, p=probs).astype(np.uint16)
    work_gpu = torch.from_numpy(tokens_np.view("int16")).to(device)

    cfg_runner = SearchRunnerConfig(
        steps_per_trial=3,
        batch_size=2,
        seq_len=16,
        num_layers=1,
        vocab_size=264,
        valid_vocab_size=v_eff,
        val_batches=2,
        log_to_webui=False,
    )
    val_batches = build_validation_batches(val_np, batch_size=2, seq_len=16, num_batches=2, device=device)

    trials = (
        HyperparamTrialSpec(
            trial_id="smoke_baseline",
            num_meanings=2,
            boson_p=11.0,
            kron_blend=0.55,
            coinc_gamma=0.45,
            carrier_rho_max=0.0,
            unigram_temp=0.80,
            muon_lr=0.015,
            embed_lr=0.008,
            povm_lr=0.004,
            is_baseline=True,
        ),
        HyperparamTrialSpec(
            trial_id="smoke_candidate",
            num_meanings=2,
            boson_p=13.5,
            kron_blend=0.40,
            coinc_gamma=0.50,
            carrier_rho_max=0.22,
            unigram_temp=0.90,
            muon_lr=0.030,
            embed_lr=0.020,
            povm_lr=0.010,
            is_baseline=False,
        ),
    )
    report: dict[str, Any] = run_hyperparam_search_suite(
        trials, work_gpu, tokens_np, val_batches, cfg_runner, device
    )
    assert report["num_trials"] == 2
    assert report["total_nan_inf_events"] == 0
    assert len(report["leaderboard"]) == 2
    assert report["leaderboard"][0]["rank"] == 1
    assert report["leaderboard"][1]["rank"] == 2
    assert report["leaderboard"][0]["final_val_ce"] <= report["leaderboard"][1]["final_val_ce"]
    assert "best_trial" in report and "val_ce_improvement_over_baseline_nats" in report


if __name__ == "__main__":
    for fn in (
        test_h1_procrustes_carrier_alignment_and_step0_unigram_fit,
        test_h1_calibrated_kronecker_chunked_autograd_parity,
        test_h2_trainable_povm_adaptation_and_penalties,
        test_h3_hyperparam_search_trial_runner_smoke,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 4 Plan 057 unit tests PASSED.")
