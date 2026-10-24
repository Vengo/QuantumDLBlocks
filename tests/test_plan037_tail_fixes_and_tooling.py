"""Unit tests for Experiment Plan 037: Banked N=2^16 Subtree Lifts (Fix 2), Kraus Retraction + Lambda Annealing (Fix 3), Fast Cholesky Readout (Fix 4), & CLI Tooling."""

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

from qdlblocks.layers.wavelet_cholesky import BlockCholeskyLinear
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
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
    gen = torch.Generator(device="cpu").manual_seed(2037)
    num_clusters = 16
    svd_env = torch.arange(1, embed_dim + 1, dtype=torch.float32) ** -0.55
    centers = torch.randn((num_clusters, embed_dim), generator=gen)
    q_mat, _ = torch.linalg.qr(centers.t())
    centers = F.normalize(q_mat.t()[:num_clusters] * svd_env, p=2.0, dim=-1)
    cluster_of_v = torch.arange(vocab_size, dtype=torch.long) % num_clusters
    noise = 0.35 * torch.randn((vocab_size, embed_dim), generator=gen) * svd_env
    ppmi = F.normalize(centers[cluster_of_v] + noise, p=2.0, dim=-1)

    steps = torch.arange(stream_len, dtype=torch.long)
    c_seq = (steps * 3) % num_clusters
    offset_seq = ((steps // num_clusters) % 32) * num_clusters
    toks = (c_seq + offset_seq) % valid_vocab_size
    return ppmi, toks


def test_fix2_banked_subtree_lifts_parameter_budget_and_sharp_eff_rank() -> None:
    """Verify Fix 2 (`num_fine_up_banks=16`) keeps params `< 11.0M` and raises `sharp_eff_rank` significantly over `1` bank."""
    full_cfg_16 = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
    )
    full_model_16 = Radical2Pow16WaveletCholeskyGPT(full_cfg_16)
    total_params = sum(p.numel() for p in full_model_16.parameters())
    assert total_params < 11_000_000, f"Expected < 11.0M params, got {total_params:,}"
    assert full_model_16.pyramid.up_65536.shape == (16, 64, 256)

    ppmi, toks = _build_synthetic_ppmi_and_stream(
        vocab_size=50304, valid_vocab_size=50257, embed_dim=64
    )
    cfg_1 = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_layers=2,
        embed_dim=64,
        num_heads=4,
        block_size=32,
        window_size=32,
        fine_detail_dim=16,
        mid_detail_dim=32,
        diffuse_rank=8,
        num_fine_up_banks=1,
    )
    cfg_16 = Radical2Pow16Config(
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
    )
    model_1 = Radical2Pow16WaveletCholeskyGPT(cfg_1)
    model_16 = Radical2Pow16WaveletCholeskyGPT(cfg_16)
    stats_1 = initialize_radical_2pow16_with_ppmi_svd(model_1, ppmi, toks, toks)
    stats_16 = initialize_radical_2pow16_with_ppmi_svd(model_16, ppmi, toks, toks)

    assert stats_1["max_self_overlap"] <= 0.05
    assert stats_16["max_self_overlap"] <= 0.05
    assert stats_16["sharp_eff_rank"] >= 1.8 * stats_1["sharp_eff_rank"], (
        f"Expected 16-bank sharp_eff_rank ({stats_16['sharp_eff_rank']:.2f}) >= 1.8x "
        f"1-bank ({stats_1['sharp_eff_rank']:.2f})"
    )


def test_fix3_kraus_stiefel_retraction_and_lambda_diff_cosine_annealing() -> None:
    """Verify Fix 3 `retract_kraus_generators()` restores Stiefel orthogonality and `set_training_progress(p)` cosine-anneals `lambda_diff`."""
    cfg = Radical2Pow16Config(
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
        lambda_diff_start=0.165,
        lambda_diff_end=0.052,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg)
    gen = torch.Generator(device="cpu").manual_seed(3037)
    with torch.no_grad():
        noise = torch.randn(model.pyramid.kraus_generators.shape, generator=gen) * 0.35
        model.pyramid.kraus_generators.add_(noise)

    err_before = model.pyramid.compute_kraus_stiefel_err()
    assert err_before > 0.2, f"Expected perturbed Stiefel error > 0.2, got {err_before:.6f}"
    err_after = model.pyramid.retract_kraus_generators()
    assert err_after <= 1e-5, f"Expected retracted Stiefel error <= 1e-5, got {err_after:.6e}"

    progress_points = [0.18, 0.35, 0.55, 0.75, 0.90, 1.00]
    lam_vals: list[float] = []
    for p in progress_points:
        model.set_training_progress(p)
        lam_vals.append(float(model.current_lambda_diff().detach().item()))

    for i in range(len(lam_vals) - 1):
        assert lam_vals[i] > lam_vals[i + 1], (
            f"Expected monotonic cosine decrease from p={progress_points[i]} ({lam_vals[i]:.5f}) "
            f"to p={progress_points[i + 1]} ({lam_vals[i + 1]:.5f})"
        )
    assert lam_vals[0] > 0.14
    assert lam_vals[-1] < 0.065


def test_fix4_fast_cholesky_forward_logits_and_loss_equivalence() -> None:
    """Verify Fix 4 (`use_fast_cholesky_and_compile=True`) matches `False` forward logits and loss within `< 1e-4` on CPU."""
    base_kwargs = dict(
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
        num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True,
    )
    slow_model = Radical2Pow16WaveletCholeskyGPT(
        Radical2Pow16Config(**base_kwargs, use_fast_cholesky_and_compile=False)
    )
    fast_model = Radical2Pow16WaveletCholeskyGPT(
        Radical2Pow16Config(**base_kwargs, use_fast_cholesky_and_compile=True)
    )

    gen = torch.Generator(device="cpu").manual_seed(4037)
    with torch.no_grad():
        for mod in slow_model.modules():
            if isinstance(mod, BlockCholeskyLinear):
                mod.raw_tril.add_(torch.randn(mod.raw_tril.shape, generator=gen) * 0.12)
                mod.raw_diag.add_(torch.randn(mod.raw_diag.shape, generator=gen) * 0.08)
        slow_model.pyramid.wav_65536.normal_(0.0, 0.2, generator=gen)
        slow_model.pyramid.wav_8192.normal_(0.0, 0.2, generator=gen)
        slow_model.pyramid.wav_2048.normal_(0.0, 0.2, generator=gen)
        slow_model.pyramid.wav_256.normal_(0.0, 0.2, generator=gen)
        fast_model.load_state_dict(slow_model.state_dict(), strict=True)

    x = torch.randint(0, 50257, (2, 12), generator=gen, dtype=torch.long)
    y = torch.randint(0, 50257, (2, 12), generator=gen, dtype=torch.long)

    with torch.no_grad():
        logits_slow, loss_slow_l = slow_model(x, targets=y, return_logits=True)
        logits_fast, loss_fast_l = fast_model(x, targets=y, return_logits=True)
        _, loss_slow_ce = slow_model(x, targets=y, return_logits=False)
        _, loss_fast_ce = fast_model(x, targets=y, return_logits=False)

    assert logits_slow is not None and logits_fast is not None
    assert loss_slow_l is not None and loss_fast_l is not None
    assert loss_slow_ce is not None and loss_fast_ce is not None

    max_logit_diff = float((logits_slow - logits_fast).abs().max().item())
    loss_l_diff = float((loss_slow_l - loss_fast_l).abs().item())
    loss_ce_diff = float((loss_slow_ce - loss_fast_ce).abs().item())
    assert max_logit_diff < 1e-4, f"max_logit_diff={max_logit_diff:.6e} >= 1e-4"
    assert loss_l_diff < 1e-4, f"loss_l_diff={loss_l_diff:.6e} >= 1e-4"
    assert loss_ce_diff < 1e-4, f"loss_ce_diff={loss_ce_diff:.6e} >= 1e-4"


def test_dyadic_bigram_transition_heatmap_cli_and_json(tmp_path: Path) -> None:
    """Verify `scripts/dyadic_bigram_transition_heatmap.py` CLI and JSON output."""
    hm_mod = _load_script_module("dyadic_bigram_transition_heatmap.py")
    out_json = tmp_path / "bigram_transition_audit.json"
    cfg = hm_mod.parse_args([
        "--embed-dim", "64",
        "--num-layers", "2",
        "--num-fine-up-banks", "16",
        "--stream-len", "8192",
        "--top-k-transitions", "6",
        "--device", "cpu",
        "--out-json", str(out_json),
    ])
    rep = hm_mod.run_bigram_transition_audit(cfg)
    assert rep["step0_init_stats"]["max_self_overlap"] <= 0.05
    t_audit = rep["coarse_transition_audit"]
    assert t_audit["shape"] == [256, 256]
    assert t_audit["active_rows"] > 0
    assert 0.0 < t_audit["mean_row_entropy_nats"] < t_audit["uniform_entropy_nats"]
    assert len(t_audit["top_transitions"]) == 6

    p_audit = rep["predecessor_2048_audit"]
    assert p_audit["active_buckets_2048"] > 0
    assert p_audit["centered_directed_bigram_overlap_ratio"] > p_audit["uncentered_directed_bigram_overlap_ratio"]
    assert p_audit["centering_overlap_gain_ratio"] > 1.05
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["coarse_transition_audit"]["shape"] == [256, 256]


def test_wavelet_scale_ablation_probe_cli_and_json(tmp_path: Path) -> None:
    """Verify `scripts/wavelet_scale_ablation_probe.py` CLI and JSON output."""
    ab_mod = _load_script_module("wavelet_scale_ablation_probe.py")
    out_json = tmp_path / "wavelet_scale_ablation.json"
    cfg = ab_mod.parse_args([
        "--embed-dim", "64",
        "--num-layers", "2",
        "--sibling-probe-pairs", "1024",
        "--bench-iters", "2",
        "--device", "cpu",
        "--out-json", str(out_json),
    ])
    rep = ab_mod.run_wavelet_scale_ablation_probe(cfg)
    cmp = rep["comparison"]
    assert cmp["bank16_full_sharp_eff_rank"] > cmp["bank1_full_sharp_eff_rank"] * 1.8
    assert cmp["bank16_parseval_bg_sum"] < cmp["bank1_parseval_bg_sum"]
    assert cmp["fast_cholesky_synthesis"]["max_abs_m_sharp_diff"] < 1e-4

    b16_ab = rep["bank16_profile"]["ablations"]
    assert b16_ab["drop_65536_c0"]["sibling_separation"] == 0.0
    assert b16_ab["only_65536_c0"]["sibling_separation"] > 0.5
    assert b16_ab["cum_c3_c2_c1_c0"]["sharp_eff_rank"] > b16_ab["cum_c3_c2_c1"]["sharp_eff_rank"]
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["comparison"]["sharp_eff_rank_gain_ratio"] > 1.8


def test_plan037_ast_signature_and_length_compliance() -> None:
    """Verify all Plan 037 model, layer, script, and test files pass `ast_signature_and_length_linter.py`."""
    linter_mod = _load_script_module("ast_signature_and_length_linter.py")
    target_files = [
        "src/qdlblocks/layers/wavelet_cholesky.py",
        "src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py",
        "scripts/dyadic_bigram_transition_heatmap.py",
        "scripts/wavelet_scale_ablation_probe.py",
        "tests/test_plan037_tail_fixes_and_tooling.py",
    ]
    if (ROOT / "examples/47_verify_plan037_tail_fixes_and_fast_cholesky.py").exists():
        target_files.append("examples/47_verify_plan037_tail_fixes_and_fast_cholesky.py")
    linter_cfg = linter_mod.parse_args([*target_files, "--max-lines", "1000"])
    linter_rep = linter_mod.run_linter(linter_cfg)
    assert linter_rep["all_passed"] is True
