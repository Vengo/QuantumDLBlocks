"""Unit tests for Experiment Plan 038: Last-Layer Measurement Operator, Traceless Observable Readout, & Parseval / Block-Cholesky Tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p_dir in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p_dir not in sys.path:
        sys.path.insert(0, p_dir)

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.direct_chunked_ce import (
    compute_two_beam_kboson_logits,
    direct_chunked_two_beam_kboson_ce,
)
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_radical_2pow16_optimizers,
    initialize_radical_2pow16_with_ppmi_svd,
)


def _load_script_module(script_name: str):
    """Load a script module from `scripts/` by filename."""
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_synthetic_ppmi_and_stream(
    vocab_size: int,
    valid_vocab_size: int,
    embed_dim: int,
    stream_len: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize clustered PPMI embeddings and a directed token stream for CPU tests."""
    gen = torch.Generator(device="cpu").manual_seed(2038)
    num_clusters = 16
    svd_env = torch.arange(1, embed_dim + 1, dtype=torch.float32) ** -0.45
    centers = torch.randn((num_clusters, embed_dim), generator=gen)
    q_mat, _ = torch.linalg.qr(centers.t())
    centers = F.normalize(q_mat.t()[:num_clusters] * svd_env, p=2.0, dim=-1)
    cluster_of_v = torch.arange(vocab_size, dtype=torch.long) % num_clusters
    noise = 0.45 * torch.randn((vocab_size, embed_dim), generator=gen) * svd_env
    ppmi = F.normalize(centers[cluster_of_v] + noise, p=2.0, dim=-1)

    steps = torch.arange(stream_len, dtype=torch.long)
    c_seq = (steps * 3) % num_clusters
    offset_seq = ((steps // num_clusters) % 32) * num_clusters
    toks = (c_seq + offset_seq) % valid_vocab_size
    return ppmi, toks


def test_traceless_observable_chunked_ce_forward_and_backward_equivalence() -> None:
    """Verify `direct_chunked_two_beam_kboson_ce(traceless_gamma=0.55)` matches `compute_two_beam_kboson_logits` + `F.cross_entropy`."""
    gen = torch.Generator(device="cpu").manual_seed(1038)
    n_tok, v_vocab, v_active, d = 24, 256, 240, 32
    psi_s_base = l2_normalize(torch.randn((n_tok, d), generator=gen, dtype=torch.float32), dim=-1)
    psi_d_base = l2_normalize(torch.randn((n_tok, d), generator=gen, dtype=torch.float32), dim=-1)
    m_s_base = l2_normalize(torch.randn((v_vocab, d), generator=gen, dtype=torch.float32), dim=-1)
    m_d_base = l2_normalize(torch.randn((v_vocab, d), generator=gen, dtype=torch.float32), dim=-1)
    beta_base = torch.full((n_tok, 1), 28.0, dtype=torch.float32)
    k_base = torch.tensor(34.0, dtype=torch.float32)
    lam_base = torch.tensor(0.14, dtype=torch.float32)
    log_prior = torch.randn((v_vocab,), generator=gen, dtype=torch.float32) * 0.2
    targets = torch.randint(0, v_active, (n_tok,), generator=gen, dtype=torch.long)

    # 1. Direct chunked autograd pass
    psi_s_1 = psi_s_base.clone().requires_grad_(True)
    psi_d_1 = psi_d_base.clone().requires_grad_(True)
    m_s_1 = m_s_base.clone().requires_grad_(True)
    m_d_1 = m_d_base.clone().requires_grad_(True)
    beta_1 = beta_base.clone().requires_grad_(True)
    k_1 = k_base.clone().requires_grad_(True)
    loss_chunked = direct_chunked_two_beam_kboson_ce(
        psi_sharp=psi_s_1,
        psi_diff=psi_d_1,
        m_sharp=m_s_1,
        m_diff=m_d_1,
        beta_t=beta_1,
        k_copies=k_1,
        lambda_diff=lam_base,
        targets=targets,
        log_prior=log_prior,
        chunk_size=8,
        valid_vocab_size=v_active,
        quadratic_diffuse_fidelity=0.50,
        traceless_gamma=0.55,
    )
    loss_chunked.backward()

    # 2. Reference unchunked logits + F.cross_entropy pass
    psi_s_2 = psi_s_base.clone().requires_grad_(True)
    psi_d_2 = psi_d_base.clone().requires_grad_(True)
    m_s_2 = m_s_base.clone().requires_grad_(True)
    m_d_2 = m_d_base.clone().requires_grad_(True)
    beta_2 = beta_base.clone().requires_grad_(True)
    k_2 = k_base.clone().requires_grad_(True)
    logits_ref = compute_two_beam_kboson_logits(
        psi_sharp=psi_s_2,
        psi_diff=psi_d_2,
        m_sharp=m_s_2,
        m_diff=m_d_2,
        beta_t=beta_2,
        k_copies=k_2,
        lambda_diff=lam_base,
        log_prior=log_prior,
        quadratic_diffuse_fidelity=0.50,
        traceless_gamma=0.55,
        valid_vocab_size=v_active,
    )
    loss_ref = F.cross_entropy(logits_ref[:, :v_active], targets)
    loss_ref.backward()

    loss_diff = float((loss_chunked.detach() - loss_ref.detach()).abs().item())
    assert loss_diff < 1e-5, f"Forward loss mismatch: {loss_diff:.6e} >= 1e-5"

    for name, g1, g2 in (
        ("dpsi_sharp", psi_s_1.grad, psi_s_2.grad),
        ("dpsi_diff", psi_d_1.grad, psi_d_2.grad),
        ("dm_sharp", m_s_1.grad, m_s_2.grad),
        ("dm_diff", m_d_1.grad, m_d_2.grad),
        ("dbeta", beta_1.grad, beta_2.grad),
        ("dk", k_1.grad, k_2.grad),
    ):
        assert g1 is not None and g2 is not None, f"Missing gradient for {name}"
        max_g_diff = float((g1 - g2).abs().max().item())
        assert max_g_diff < 1e-4, f"Gradient mismatch on {name}: {max_g_diff:.6e} >= 1e-4"


def test_last_layer_measurement_and_traceless_model_invariants() -> None:
    """Verify parameter budget `<11.0M`, `max_self_overlap <= 1e-5`, `directed_bigram_overlap_ratio >= 1.8`, and `_stabilize_pyramid_frame`."""
    full_cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
        use_last_layer_measurement_operator=True,
        use_traceless_observable=True,
        traceless_gamma=0.55,
    )
    full_model = Radical2Pow16WaveletCholeskyGPT(full_cfg)
    total_params = sum(p.numel() for p in full_model.parameters())
    assert total_params < 11_000_000, f"Expected < 11.0M params, got {total_params:,}"

    muon_opt, _ = build_radical_2pow16_optimizers(full_model)
    muon_param_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    assert id(full_model.pyramid.up_8192) in muon_param_ids
    assert id(full_model.pyramid.up_65536) in muon_param_ids
    assert id(full_model.pyramid.kraus_generators) in muon_param_ids

    # CPU initialization + self-overlap & bigram overlap audit
    ppmi, toks = _build_synthetic_ppmi_and_stream(
        vocab_size=50304, valid_vocab_size=50257, embed_dim=64
    )
    cpu_cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_layers=2,
        embed_dim=64,
        num_heads=4,
        block_size=32,
        window_size=32,
        fine_detail_dim=16,
        mid_detail_dim=32,
        diffuse_rank=8,
        num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
        use_last_layer_measurement_operator=True,
        use_traceless_observable=True,
        traceless_gamma=0.55,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cpu_cfg)
    init_stats = initialize_radical_2pow16_with_ppmi_svd(model, ppmi, toks, toks)

    assert init_stats["max_self_overlap"] <= 1e-5, (
        f"Expected max_self_overlap <= 1e-5, got {init_stats['max_self_overlap']:.6e}"
    )
    assert init_stats["directed_bigram_overlap_ratio"] >= 1.8, (
        f"Expected directed_bigram_overlap_ratio >= 1.8, got {init_stats['directed_bigram_overlap_ratio']:.4f}"
    )

    # Verify initial wavefunctions have non-collapsed sharp_eff_rank (> 20.0 on 16-cluster d=64 synthetic PPMI)
    psi_init = model.pyramid.synthesize_initial_wavefunctions()
    r_before = model.pyramid.compute_sharp_eff_rank(psi_init)
    assert r_before >= 20.0, f"Expected initial sharp_eff_rank >= 20.0 in d=64, got {r_before:.2f}"

    # Perturb pyramid with anisotropic rank-1 spike + Cholesky drift and verify _stabilize_pyramid_frame recovers rank
    gen = torch.Generator(device="cpu").manual_seed(3038)
    with torch.no_grad():
        spike = torch.ones((1, 16), dtype=torch.float32) * 0.65
        model.pyramid.wav_65536.add_(spike)
        model.pyramid.chol_256.raw_diag.add_(torch.randn_like(model.pyramid.chol_256.raw_diag) * 0.9)
        r_perturbed = model.pyramid.compute_sharp_eff_rank()
        for _ in range(4):
            model.pyramid._stabilize_pyramid_frame(rate=0.35)
        r_stabilized = model.pyramid.compute_sharp_eff_rank()

    assert r_stabilized > r_perturbed, (
        f"Expected _stabilize_pyramid_frame to improve sharp_eff_rank ({r_stabilized:.2f} vs {r_perturbed:.2f})"
    )
    assert float(model.pyramid.chol_256.raw_diag.abs().max().item()) <= 0.35 + 1e-5


def test_parseval_tail_overlap_profiler_cli_and_json(tmp_path: Path) -> None:
    """Verify `scripts/parseval_tail_overlap_profiler.py` CLI, JSON output, and 50,304-vector sweep artifact."""
    prof_mod = _load_script_module("parseval_tail_overlap_profiler.py")
    out_json = tmp_path / "parseval_test.json"
    cfg = prof_mod.parse_args([
        "--vocab-size", "2048",
        "--dims", "16,64,256",
        "--probe-queries", "64",
        "--num-landing-steps", "4",
        "--device", "cpu",
        "--out-json", str(out_json),
    ])
    rep = prof_mod.run_parseval_profiler(cfg)
    assert out_json.exists()
    d256 = rep["dimensions"]["256"]
    welch_lb = d256["welch_rms_lower_bound"]
    r_qr = d256["regimes"]["incomplete_block_qr"]
    r_welch = d256["regimes"]["welch_frame_landing"]
    r_cone = d256["regimes"]["zipf_cone"]

    assert r_qr["intra_block_rms_off_diag_cos"] <= 1e-4
    assert r_qr["empirical_rms_off_diag_cos"] >= welch_lb * 0.98
    assert r_welch["sharp_eff_rank"] > r_cone["sharp_eff_rank"] * 5.0
    assert (
        r_welch["probe_328"]["traceless_observable_kboson_ce"]
        < r_welch["probe_328"]["uncentered_kboson_ce"]
    )

    sweep_cache = ROOT / ".cache" / "parseval_packing_50304_sweep.json"
    assert sweep_cache.exists(), "Missing .cache/parseval_packing_50304_sweep.json"
    full_sweep = json.loads(sweep_cache.read_text(encoding="utf-8"))
    assert full_sweep["config"]["vocab_size"] == 50304
    w256 = full_sweep["dimensions"]["256"]["regimes"]["welch_frame_landing"]
    assert w256["probe_328"]["traceless_observable_kboson_ce"] < 3.28
    assert w256["probe_250"]["traceless_observable_kboson_ce"] < 2.50


def test_block_cholesky_kernel_benchmark_cli_and_json(tmp_path: Path) -> None:
    """Verify `scripts/block_cholesky_kernel_benchmark.py` CLI and numerical equivalence (`< 1e-5`)."""
    bench_mod = _load_script_module("block_cholesky_kernel_benchmark.py")
    out_json = tmp_path / "block_cholesky_bench.json"
    cfg = bench_mod.parse_args([
        "--dims", "64,128,256",
        "--token-rows", "512",
        "--vocab-rows", "2048",
        "--warmup-iters", "1",
        "--bench-iters", "2",
        "--device", "cpu",
        "--out-json", str(out_json),
    ])
    rep = bench_mod.run_block_cholesky_benchmark(cfg)
    assert rep["all_numerical_equivalence_passed"] is True
    for d_key in ("64", "128", "256"):
        assert rep["dimensions"][d_key]["max_abs_diff"] < 1e-5
        assert rep["dimensions"][d_key]["condition_metrics"]["spd_cond_number"] >= 1.0
    assert out_json.exists()


def test_plan038_ast_signature_and_length_compliance() -> None:
    """Verify all Plan 038 model, layer, script, and test files pass `ast_signature_and_length_linter.py`."""
    linter_mod = _load_script_module("ast_signature_and_length_linter.py")
    target_files = [
        "src/qdlblocks/layers/wavelet_cholesky.py",
        "src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py",
        "scripts/parseval_tail_overlap_profiler.py",
        "scripts/block_cholesky_kernel_benchmark.py",
        "tests/test_plan038_last_layer_measurement_and_traceless.py",
    ]
    if (ROOT / "examples/48_verify_plan038_last_layer_measurement_and_traceless.py").exists():
        target_files.append("examples/48_verify_plan038_last_layer_measurement_and_traceless.py")
    linter_cfg = linter_mod.parse_args([*target_files, "--max-lines", "1000"])
    linter_rep = linter_mod.run_linter(linter_cfg)
    assert linter_rep["all_passed"] is True
