"""Unit and integration tests for Experiment Plan 036 (Fused Kronecker-U-Net N=2^16 Wavelet-Cholesky GPT)."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tokenize

import torch

ROOT = Path(__file__).resolve().parents[1]
_EX46_SPEC = importlib.util.spec_from_file_location(
    "plan036_runner", ROOT / "examples" / "46_verify_plan036_fused_kronecker_unet_2pow16.py"
)
assert _EX46_SPEC is not None and _EX46_SPEC.loader is not None
_EX46 = importlib.util.module_from_spec(_EX46_SPEC)
_EX46_SPEC.loader.exec_module(_EX46)

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
)


def test_h1_fused_kronecker_unet_parameter_budget_and_arch_modes() -> None:
    """Verify `fused_kronecker_unet` parameter budget (<12.5M) and comparison vs Arch 1 and Arch 2."""
    torch.manual_seed(2036)
    m_fused = Radical2Pow16WaveletCholeskyGPT(Radical2Pow16Config(arch_mode="fused_kronecker_unet"))
    total_params = sum(p.numel() for p in m_fused.parameters())
    assert total_params < 12_500_000, f"fused_kronecker_unet has {total_params:,} params (>= 12.5M)"
    assert 40.78 / (total_params / 1e6) >= 3.5


def test_plan036_script_best_practices_ast_compliance() -> None:
    """Verify `wavelet_cholesky_2pow16_gpt.py` and `examples/46_*.py` obey line budget, <=8 args, and 0 semicolons."""
    for rel_path, max_lines in (
        ("src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py", 1000),
        ("examples/46_verify_plan036_fused_kronecker_unet_2pow16.py", 400),
    ):
        path = ROOT / rel_path
        assert path.exists(), f"Missing {rel_path}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) <= max_lines, f"{rel_path} has {len(lines)} lines (> {max_lines})"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_args = len(node.args.posonlyargs) + len(node.args.args) + len(node.args.kwonlyargs)
                if node.args.args and node.args.args[0].arg in ("self", "cls"):
                    n_args -= 1
                if node.args.vararg is not None:
                    n_args += 1
                if node.args.kwarg is not None:
                    n_args += 1
                assert n_args <= 8, f"{rel_path}:{node.name} has {n_args} arguments (> 8 ceiling)"
        with path.open("rb") as f:
            semicolons = [tok for tok in tokenize.tokenize(f.readline) if tok.type == tokenize.OP and tok.string == ";"]
        assert not semicolons, f"{rel_path} contains semicolon statement packing"


def test_plan036_hypothesis_experiments_json_results() -> None:
    """Verify `.cache/hypothesis_experiments_036.json` records H1=True, H2=True, and H3=True."""
    json_path = ROOT / ".cache" / "hypothesis_experiments_036.json"
    assert json_path.exists(), "Missing .cache/hypothesis_experiments_036.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    hyps = data["hypotheses"]

    h1 = hyps["H1_fused_kronecker_unet_predecessor_and_bigram_gate"]
    assert h1["passed"] is True
    assert h1["fused_total_params"] < 12_500_000
    assert h1["fused_kronecker_unet"]["compression_vs_quantum_born_40p78M"] >= 3.5
    assert h1["max_self_overlap"] <= 0.05
    assert h1["directed_bigram_overlap_ratio"] >= 1.80
    assert h1["max_future_leakage"] == 0.0
    assert h1["grad_finite"] is True

    h2 = hyps["H2_900step_fused_kronecker_unet_ablation"]
    assert h2["passed"] is True
    assert h2["arch3_fused_kronecker_unet_best_val_ce"] < h2["arch1_multiscale_unet_best_val_ce"]
    assert h2["arch3_fused_kronecker_unet_best_val_ce"] < h2["arch2_consistent_kronecker_best_val_ce"]
    assert h2["arch3_fused_kronecker_unet_best_val_ce"] < h2["quantum_born_40p78M_best_val_ce"]
    assert h2["arch3_fused_kronecker_unet_best_val_ce"] < h2["classical_53p54M_best_val_ce"]
    assert h2["arch3_peak_vram_gb"] <= 8.0

    h3 = hyps["H3_3600step_29p5M_token_fused_kronecker_unet_scaling"]
    assert h3["passed"] is True
    assert h3["fused_kronecker_unet_3600_best_val_ce"] < h3["plan035_multiscale_unet_3600_best_val_ce"]
    assert h3["fused_kronecker_unet_3600_best_val_ce"] < h3["parity_split_38p28M_quantum_born_val_ce"]
    assert h3["fused_kronecker_unet_3600_best_val_ce"] < h3["classical_30p95M_3600_best_val_ce"]
    assert h3["fused_kronecker_unet_params_M"] < 12.5
    assert h3["peak_vram_gb"] <= 8.0
    assert all(h3["checkpoint_beats_plan035"].values())
    assert all(h3["checkpoint_beats_classical"].values())

    for rkey in (
        "w21_8gb_arch3_fused_kronecker_unet_2pow16_900steps",
        "w21_8gb_fused_kronecker_unet_2pow16_3600steps",
    ):
        assert (ROOT / ".cache" / "live_logs" / f"{rkey}_results.json").exists()
        assert (ROOT / ".cache" / "model_cards" / f"{rkey}.json").exists()
