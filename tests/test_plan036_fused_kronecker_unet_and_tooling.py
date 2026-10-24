"""Unit tests for Experiment Plan 036: Fused Kronecker-U-Net N=2^16 Architecture & Promoted CLI Tooling."""

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


def _build_structured_ppmi_and_bigram_stream(
    vocab_size: int,
    valid_vocab_size: int,
    embed_dim: int,
    stream_len: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize clustered PPMI embeddings and a directed cluster-transition bigram token stream."""
    gen = torch.Generator(device="cpu").manual_seed(2036)
    num_clusters = 16
    svd_env = torch.arange(1, embed_dim + 1, dtype=torch.float32) ** -0.85
    centers = torch.randn((num_clusters, embed_dim), generator=gen)
    centers, _ = torch.linalg.qr(centers.t())
    centers = F.normalize(centers.t() * svd_env, p=2.0, dim=-1)
    cluster_of_v = torch.arange(vocab_size, dtype=torch.long) % num_clusters
    noise = 0.02 * torch.randn((vocab_size, embed_dim), generator=gen) * svd_env
    ppmi = F.normalize(centers[cluster_of_v] + noise, p=2.0, dim=-1)

    # Directed bigram stream: token in cluster c transitions to a token in cluster (c + 3) % num_clusters
    steps = torch.arange(stream_len, dtype=torch.long)
    c_seq = (steps * 3) % num_clusters
    offset_seq = ((steps // num_clusters) % 32) * num_clusters
    toks = (c_seq + offset_seq) % valid_vocab_size
    return ppmi, toks


def test_fused_kronecker_unet_parameter_budget_init_causality_and_grads() -> None:
    """Verify `< 12.5M` params, Step-0 self-overlap <= 0.05, bigram overlap >= 1.8x, causality, and finite grads."""
    torch.manual_seed(2036)
    full_cfg = Radical2Pow16Config(arch_mode="fused_kronecker_unet")
    full_model = Radical2Pow16WaveletCholeskyGPT(full_cfg)
    total_params = sum(p.numel() for p in full_model.parameters())
    assert total_params < 12_500_000, f"fused_kronecker_unet has {total_params:,} params (>= 12.5M)"

    small_cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        vocab_size=50304,
        valid_vocab_size=50257,
        ambient_n=65536,
        num_layers=2,
        embed_dim=64,
        num_heads=4,
        block_size=32,
        window_size=32,
        fine_detail_dim=16,
        mid_detail_dim=32,
        diffuse_rank=8,
    )
    model = Radical2Pow16WaveletCholeskyGPT(small_cfg)
    ppmi, toks = _build_structured_ppmi_and_bigram_stream(
        vocab_size=small_cfg.vocab_size,
        valid_vocab_size=small_cfg.valid_vocab_size,
        embed_dim=small_cfg.embed_dim,
    )
    init_stats = initialize_radical_2pow16_with_ppmi_svd(model, ppmi, toks, toks)
    assert init_stats["max_self_overlap"] <= 0.05
    assert init_stats["directed_bigram_overlap_ratio"] >= 1.80

    gen = torch.Generator(device="cpu").manual_seed(3036)
    x1 = torch.randint(0, small_cfg.valid_vocab_size, (2, 16), generator=gen, dtype=torch.long)
    x2 = x1.clone()
    x2[:, 8:] = (x2[:, 8:] + 137) % small_cfg.valid_vocab_size
    with torch.no_grad():
        logits1, _ = model(x1, targets=None, return_logits=True)
        logits2, _ = model(x2, targets=None, return_logits=True)
    assert logits1 is not None and logits2 is not None
    max_future_leakage = float((logits1[:, :8, :] - logits2[:, :8, :]).abs().max().item())
    assert max_future_leakage == 0.0
    assert float((logits1[:, 8:, :] - logits2[:, 8:, :]).abs().max().item()) > 0.0

    y = torch.randint(0, small_cfg.valid_vocab_size, (2, 16), generator=gen, dtype=torch.long)
    _, loss = model(x1, targets=y, return_logits=False)
    assert loss is not None and bool(torch.isfinite(loss).item())
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

    muon_opt, adam_opt = build_radical_2pow16_optimizers(model)
    assert muon_opt is not None and adam_opt is not None


def test_dyadic_spectral_tree_inspector_cli_and_json(tmp_path: Path) -> None:
    """Verify `scripts/dyadic_spectral_tree_inspector.py` multi-scale coherence, monotonicity, and JSON export."""
    insp_mod = _load_script_module("dyadic_spectral_tree_inspector.py")
    out_json = tmp_path / "spectral_tree_inspector.json"
    cfg = insp_mod.parse_args([
        "--vocab-size", "2048",
        "--ambient-n", "4096",
        "--embed-dim", "64",
        "--num-clusters", "32",
        "--top-k-siblings", "6",
        "--device", "cpu",
        "--out-json", str(out_json),
    ])
    rep = insp_mod.run_spectral_tree_inspector(cfg)
    assert rep["is_bijection"] is True
    coh = rep["multiscale_coherence"]
    assert coh["monotonic_key_scales_2_8_64_256"] is True
    assert coh["sibling_block2_cos"] >= coh["block8_cos"] >= coh["block64_cos"] >= coh["block256_cos"]
    assert coh["block8_gain_ratio"] >= 2.0
    assert coh["block64_gain_ratio"] >= 2.0
    sib = rep["siblings_and_clusters"]
    assert len(sib["top_sibling_pairs"]) == 6
    assert len(sib["coarse_cluster_summaries"]) > 0
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["is_bijection"] is True


def test_cholesky_band_sparsity_profiler_cli_and_json(tmp_path: Path) -> None:
    """Verify `scripts/cholesky_band_sparsity_profiler.py` across `fused_kronecker_unet` Cholesky blocks."""
    prof_mod = _load_script_module("cholesky_band_sparsity_profiler.py")
    out_json = tmp_path / "cholesky_profiler.json"
    cfg = prof_mod.parse_args([
        "--arch-mode", "fused_kronecker_unet",
        "--embed-dim", "64",
        "--num-layers", "2",
        "--perturb-std", "0.08",
        "--bench-iters", "2",
        "--device", "cpu",
        "--out-json", str(out_json),
    ])
    rep = prof_mod.run_cholesky_profiler(cfg)
    summary = rep["summary"]
    assert summary["all_diag_positive"] is True
    assert summary["num_cholesky_modules"] >= 15
    assert summary["mean_off_diag_frobenius_ratio"] > 0.0
    assert summary["overall_compression_vs_dense"] >= 1.0
    assert summary["max_spd_cond_number"] >= 1.0
    mod_names = {m["name"] for m in rep["modules"]}
    for expected in (
        "pyramid.chol_256",
        "pyramid.chol_2048",
        "pyramid.chol_8192",
        "pyramid.chol_65536",
        "pyramid.readout_chol_bridge",
        "pyramid.chol_pred_2048",
        "pyramid.bigram_chol_fine",
        "pyramid.bigram_chol_mid",
        "pyramid.val_chol_bridge",
    ):
        assert expected in mod_names, f"Missing expected Cholesky module {expected}"
    assert out_json.exists()


def test_plan036_ast_signature_and_length_compliance() -> None:
    """Verify all Plan 036 model, script, and test files pass `ast_signature_and_length_linter.py`."""
    linter_mod = _load_script_module("ast_signature_and_length_linter.py")
    target_files = [
        "src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py",
        "scripts/dyadic_spectral_tree_inspector.py",
        "scripts/cholesky_band_sparsity_profiler.py",
        "tests/test_plan036_fused_kronecker_unet_and_tooling.py",
    ]
    linter_cfg = linter_mod.parse_args([*target_files, "--max-lines", "1000"])
    linter_rep = linter_mod.run_linter(linter_cfg)
    assert linter_rep["all_passed"] is True
