"""Unit and integration tests for Experiment Plan 035 (Standalone Radical N=2^16 Wavelet-Cholesky GPT)."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
_EX45_SPEC = importlib.util.spec_from_file_location(
    "plan035_runner", ROOT / "examples" / "45_verify_plan035_radical_2pow16_wavelet_cholesky.py"
)
assert _EX45_SPEC is not None and _EX45_SPEC.loader is not None
_EX45 = importlib.util.module_from_spec(_EX45_SPEC)
_EX45_SPEC.loader.exec_module(_EX45)

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_dyadic_2pow16_spectral_tree,
)


def test_h1_dyadic_2pow16_spectral_tree_and_both_architectures() -> None:
    """Verify `build_dyadic_2pow16_spectral_tree` bijection and both standalone N=2^16 architectures."""
    torch.manual_seed(2035)
    ppmi = F.normalize(torch.randn(1024, 64, dtype=torch.float32), p=2.0, dim=-1)
    perm_v_to_leaf, leaf_to_v = build_dyadic_2pow16_spectral_tree(ppmi, ambient_n=1024)
    arange_n = torch.arange(1024, dtype=torch.long)
    assert torch.equal(leaf_to_v[perm_v_to_leaf], arange_n)
    assert torch.equal(perm_v_to_leaf[leaf_to_v], arange_n)

    for mode in ("multiscale_unet", "consistent_kronecker"):
        cfg = Radical2Pow16Config(
            arch_mode=mode, vocab_size=50304, valid_vocab_size=50257,
            ambient_n=65536, num_layers=4, embed_dim=256, num_heads=4,
        )
        model = Radical2Pow16WaveletCholeskyGPT(cfg)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        assert params_m < 11.0, f"{mode} has {params_m:.2f}M params (>= 11.0M)"


def test_plan035_script_best_practices_ast_compliance() -> None:
    """Verify `wavelet_cholesky_2pow16_gpt.py` and `examples/45_*.py` parse cleanly and obey <=8 args."""
    for rel_path in (
        "src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py",
        "examples/45_verify_plan035_radical_2pow16_wavelet_cholesky.py",
    ):
        path = ROOT / rel_path
        assert path.exists(), f"Missing {rel_path}"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                n_args = len(node.args.args) + len(node.args.kwonlyargs)
                if node.args.args and node.args.args[0].arg in ("self", "cls"):
                    n_args -= 1
                assert n_args <= 8, f"{rel_path}:{node.name} has {n_args} arguments (> 8 ceiling)"


def test_plan035_hypothesis_experiments_json_results() -> None:
    """Verify `.cache/hypothesis_experiments_035.json` records H1=True, H2=True, and H3=True."""
    json_path = ROOT / ".cache" / "hypothesis_experiments_035.json"
    assert json_path.exists(), "Missing .cache/hypothesis_experiments_035.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    hyps = data["hypotheses"]

    h1 = hyps["H1_dyadic_2pow16_spectral_tree_and_compression"]
    assert h1["passed"] is True
    assert h1["is_bijection_65536"] is True
    assert h1["spectral_pair_cos"] > 0.35
    assert h1["coherence_gain_ratio"] >= 2.0
    assert h1["multiscale_unet"]["params_M"] < 11.0
    assert h1["consistent_kronecker"]["params_M"] < 11.0
    assert h1["multiscale_unet"]["max_self_overlap"] <= 0.05
    assert h1["consistent_kronecker"]["max_self_overlap"] <= 0.05
    assert h1["grad_finite"] is True

    h2 = hyps["H2_900step_head_to_head_radical_2pow16_architectures"]
    assert h2["passed"] is True
    assert h2["arch1_multiscale_unet_best_val_ce"] < h2["classical_step900_val_ce"]
    assert h2["arch2_consistent_kronecker_best_val_ce"] < h2["classical_step900_val_ce"]

    h3 = hyps["H3_3600step_29p5M_token_radical_2pow16_scaling"]
    assert h3["passed"] is True
    assert h3["radical_2pow16_3600_best_val_ce"] < 5.35
    assert h3["radical_2pow16_3600_best_val_ce"] < h3["classical_30p89M_3600_best_val_ce"]
    assert h3["radical_2pow16_params_M"] < 11.0
    assert h3["peak_vram_gb"] <= 8.0
    assert all(h3["checkpoint_radical_beats_classical"].values())
