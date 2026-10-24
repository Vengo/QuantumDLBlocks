"""Unit and integration tests for Experiment Plan 037 (16-Bank N=2^16 Wavelet-Cholesky, Kraus Retraction, & Fast Cholesky)."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tokenize

import torch

ROOT = Path(__file__).resolve().parents[1]
_EX47_SPEC = importlib.util.spec_from_file_location(
    "plan037_runner", ROOT / "examples" / "47_verify_plan037_tail_fixes_and_fast_cholesky.py"
)
assert _EX47_SPEC is not None and _EX47_SPEC.loader is not None
_EX47 = importlib.util.module_from_spec(_EX47_SPEC)
_EX47_SPEC.loader.exec_module(_EX47)

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_radical_2pow16_optimizers,
)


def test_h1_tail_fixes_and_fast_cholesky_parameter_budget_and_stiefel_muon() -> None:
    """Verify `num_fine_up_banks=16` parameter budget (<11.0M) and `kraus_generators` routing to `StiefelMuon`."""
    torch.manual_seed(2037)
    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 11_000_000, f"Plan 037 16-bank model has {total_params:,} params (>= 11.0M)"
    assert 40.78 / (total_params / 1e6) >= 3.8

    muon_opt, _ = build_radical_2pow16_optimizers(model)
    muon_param_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    assert id(model.pyramid.kraus_generators) in muon_param_ids


def test_plan037_script_best_practices_ast_compliance() -> None:
    """Verify Plan 037 modules and `examples/47_*.py` obey line budget, <=8 args, and 0 semicolons."""
    for rel_path, max_lines in (
        ("src/qdlblocks/layers/wavelet_cholesky.py", 1000),
        ("src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py", 1000),
        ("examples/47_verify_plan037_tail_fixes_and_fast_cholesky.py", 1000),
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


def test_plan037_hypothesis_experiments_json_results() -> None:
    """Verify `.cache/hypothesis_experiments_037.json` records H1=True, Fix 4 2.72x/2.29x speedups, K_err<=0.05, and the R_sharp cone collapse diagnosis motivating Plan 038."""
    json_path = ROOT / ".cache" / "hypothesis_experiments_037.json"
    assert json_path.exists(), "Missing .cache/hypothesis_experiments_037.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    hyps = data["hypotheses"]

    h1 = hyps["H1_tail_fixes_and_fast_cholesky_equivalence"]
    assert h1["passed"] is True
    assert h1["total_params"] < 11_000_000
    assert h1["sharp_eff_rank_16bank"] >= 190.0
    assert h1["sharp_eff_rank_16bank"] > 2.5 * h1["sharp_eff_rank_1bank"]
    assert h1["fine_subspace_eff_rank_16bank"] >= 2.5 * h1["fine_subspace_eff_rank_1bank"]
    assert h1["max_self_overlap"] <= 0.05
    assert h1["directed_bigram_overlap_ratio"] >= 1.80
    assert h1["kraus_stiefel_err_after_retract"] <= 0.05
    assert h1["max_rel_loss_diff"] < 2e-3
    assert h1["max_future_leakage"] == 0.0
    assert h1["grad_finite"] is True

    h2 = hyps["H2_900step_with_vs_without_fix4_comparison"]
    assert h2["fixes234_with_fix4_best_val_ce"] < h2["fixes23_no_fix4_best_val_ce"]
    assert abs((h2["fixes234_with_fix4_best_val_ce"] - h2["fixes23_no_fix4_best_val_ce"]) - (-0.0322)) < 1e-3
    assert h2["fixes234_with_fix4_best_val_ce"] < h2["quantum_born_40p78M_best_val_ce"]
    assert h2["fixes234_with_fix4_best_val_ce"] < h2["classical_53p54M_best_val_ce"]
    assert h2["fix4_throughput_speedup_ratio"] >= 2.70
    assert h2["fixes234_final_kraus_stiefel_err"] <= 0.05
    assert h2["fixes234_peak_vram_gb"] <= 8.0
    # Unconstrained vocabulary detector synthesis collapsed R_sharp from 249.45 -> <25.0 (motivating Plan 038)
    assert h2["fixes234_final_sharp_eff_rank"] < 25.0

    h3 = hyps["H3_3600step_29p5M_token_fixes234_scaling"]
    assert h3["w22_fixes234_3600_best_val_ce"] < h3["plan035_w20_3600_best_val_ce"]
    assert h3["w22_fixes234_3600_best_val_ce"] < h3["parity_split_38p28M_quantum_born_val_ce"]
    assert h3["w22_fixes234_3600_best_val_ce"] < h3["classical_30p95M_3600_best_val_ce"]
    assert h3["avg_tok_per_s"] >= 26000.0
    assert h3["avg_tok_per_s"] / 11588.0 >= 2.25
    assert h3["w22_final_kraus_stiefel_err"] <= 0.05
    assert h3["w22_final_sharp_eff_rank"] < 25.0
    assert h3["w22_params_M"] < 11.0
    assert h3["peak_vram_gb"] <= 8.0
    assert h3["checkpoint_beats_plan035_w20"]["3600"] is True
    assert all(h3["checkpoint_beats_classical"].values())

    for rkey in (
        "w22_8gb_fixes23_no_fix4_900steps",
        "w22_8gb_fixes234_with_fix4_900steps",
        "w22_8gb_fixes234_with_fix4_3600steps",
    ):
        assert (ROOT / ".cache" / "live_logs" / f"{rkey}_results.json").exists()
        assert (ROOT / ".cache" / "model_cards" / f"{rkey}.json").exists()

