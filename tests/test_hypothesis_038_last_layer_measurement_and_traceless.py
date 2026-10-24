"""Unit and integration tests for Experiment Plan 038 (Last-Layer Measurement Operator & Traceless Observable Readout)."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tokenize

import torch

ROOT = Path(__file__).resolve().parents[1]
_EX48_SPEC = importlib.util.spec_from_file_location(
    "plan038_runner", ROOT / "examples" / "48_verify_plan038_last_layer_measurement_and_traceless.py"
)
assert _EX48_SPEC is not None and _EX48_SPEC.loader is not None
_EX48 = importlib.util.module_from_spec(_EX48_SPEC)
_EX48_SPEC.loader.exec_module(_EX48)

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_radical_2pow16_optimizers,
)


def test_h1_last_layer_measurement_and_traceless_parameter_budget_and_autograd() -> None:
    """Verify Plan 038 parameter budget (<11.0M), StiefelMuon routing, and Traceless Observable autograd gate."""
    torch.manual_seed(2038)
    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
        use_last_layer_measurement_operator=True,
        use_traceless_observable=True,
        traceless_gamma=0.55,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 11_000_000, f"Plan 038 model has {total_params:,} params (>= 11.0M)"

    muon_opt, _ = build_radical_2pow16_optimizers(model, muon_lr=0.026, wav_lr=0.0065, head_lr=0.0035)
    muon_param_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    assert id(model.pyramid.kraus_generators) in muon_param_ids
    assert id(model.pyramid.up_8192) in muon_param_ids
    assert id(model.pyramid.up_65536) in muon_param_ids

    ag_res = _EX48.verify_traceless_observable_autograd_gate(torch.device("cpu"))
    assert ag_res["autograd_gate_passed"] is True
    assert float(ag_res["forward_loss_abs_diff"]) < 1e-5
    assert float(ag_res["max_grad_abs_diff"]) < 1e-4


def test_plan038_script_best_practices_ast_compliance() -> None:
    """Verify Plan 038 modules and `examples/48_*.py` obey line budget, <=8 args, and 0 semicolons."""
    for rel_path, max_lines in (
        ("src/qdlblocks/layers/wavelet_cholesky.py", 1000),
        ("src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py", 1000),
        ("examples/48_verify_plan038_last_layer_measurement_and_traceless.py", 1000),
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


def test_plan038_hypothesis_experiments_json_results() -> None:
    """Verify `.cache/hypothesis_experiments_038.json` and `w23_*` artifacts record complete Plan 038 telemetry."""
    json_path = ROOT / ".cache" / "hypothesis_experiments_038.json"
    assert json_path.exists(), "Missing .cache/hypothesis_experiments_038.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    hyps = data["hypotheses"]

    h1 = hyps["H1_last_layer_measurement_and_traceless_observable_gate"]
    assert h1["passed"] is True
    assert h1["autograd_gate_passed"] is True
    assert h1["forward_loss_abs_diff"] < 1e-5
    assert h1["max_grad_abs_diff"] < 1e-4
    assert h1["total_params"] < 11_000_000
    assert h1["max_self_overlap"] <= 1e-5
    assert h1["directed_bigram_overlap_ratio"] >= 1.80
    assert h1["sharp_eff_rank"] >= 100.0
    assert h1["kraus_stiefel_err_after_retract"] <= 0.05
    assert h1["grad_finite"] is True

    h2 = hyps["H2_900step_last_layer_measurement_and_traceless_ablation"]
    assert "w23_last_layer_meas_best_val_ce" in h2
    assert "w23_traceless_obs_best_val_ce" in h2
    assert "w23_combined_best_val_ce" in h2
    assert h2["w23_combined_peak_vram_gb"] <= 8.0
    assert h2["w23_combined_final_kraus_stiefel_err"] <= 0.05

    h3 = hyps["H3_3600step_last_layer_meas_traceless_scaling"]
    assert "w23_3600_best_val_ce" in h3
    assert h3["w23_params_M"] < 11.0
    assert h3["peak_vram_gb"] <= 8.0
    assert h3["w23_final_kraus_stiefel_err"] <= 0.05

    for rkey in (
        "w23_8gb_last_layer_meas_900steps",
        "w23_8gb_traceless_obs_900steps",
        "w23_8gb_last_layer_meas_traceless_900steps",
        "w23_8gb_last_layer_meas_traceless_3600steps",
    ):
        res_file = ROOT / ".cache" / "live_logs" / f"{rkey}_results.json"
        card_file = ROOT / ".cache" / "model_cards" / f"{rkey}.json"
        assert res_file.exists(), f"Missing {res_file}"
        assert card_file.exists(), f"Missing {card_file}"
        rec = json.loads(res_file.read_text(encoding="utf-8"))
        assert rec["peak_vram_gb"] <= 8.0
        for ev in rec["evals"]:
            for req_key in ("sharp_eff_rank", "kraus_stiefel_err", "val_ce", "val_top1", "val_top10", "val_top50", "tok_per_s", "peak_vram_gb"):
                assert req_key in ev, f"Missing {req_key} in {rkey} step {ev.get('step')}"
