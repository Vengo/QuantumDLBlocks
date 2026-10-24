"""Unit and integration tests for Experiment Plan 034 (Wavelet-Cholesky & Mach-Zehnder Equi-Dimensional O(N))."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from qdlblocks.layers.wavelet_cholesky import WaveletCholeskyConfig, WaveletCholeskyMultiscaleBranch

ROOT = Path(__file__).resolve().parents[1]


def test_h1_wavelet_cholesky_multiscale_vcycle_isometry_and_lifting() -> None:
    """Verify `WaveletCholeskyMultiscaleBranch` is an exact isometry at Step 0 and under non-zero lifting."""
    torch.manual_seed(2034)
    branch = WaveletCholeskyMultiscaleBranch(
        WaveletCholeskyConfig(embed_dim=256, num_levels=3, max_block_size=32, init_nonlin_gate=0.0)
    )
    psi = F.normalize(torch.randn(2, 16, 256, dtype=torch.float32), p=2.0, dim=-1)
    coarse, details = branch.downsample_amplitudes(psi)
    e_total = coarse.pow(2).sum(dim=-1) + sum(d.pow(2).sum(dim=-1) for d in details)
    energy_err = float((e_total / psi.pow(2).sum(dim=-1) - 1.0).abs().max().item())
    recon = branch(psi)
    recon_err = float((recon - psi).abs().max().item())
    assert energy_err < 1e-5
    assert recon_err < 1e-5

    with torch.no_grad():
        for p_s, u_s in zip(branch.lift_predict, branch.lift_update):
            p_s.copy_(0.15 * torch.randn_like(p_s))
            u_s.copy_(0.15 * torch.randn_like(u_s))
    c_p, d_p = branch.downsample_amplitudes(psi)
    recon_p = branch.upsample_amplitudes(c_p, d_p)
    assert float((recon_p - psi).abs().max().item()) < 1e-5


def test_plan034_script_best_practices_ast_compliance() -> None:
    """Verify `examples/44_verify_plan034_wavelet_cholesky_and_mach_zehnder.py` obeys Rules 1 & 2 (<390 lines, <=8 args)."""
    for rel_path in (
        "examples/44_verify_plan034_wavelet_cholesky_and_mach_zehnder.py",
        "src/qdlblocks/layers/wavelet_cholesky.py",
        "src/qdlblocks/benchmarks/config_models.py",
    ):
        path = ROOT / rel_path
        assert path.exists(), f"Missing {rel_path}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) < 390, f"{rel_path} has {len(lines)} lines (must be < 390)"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                n_args = len(node.args.args) + len(node.args.kwonlyargs)
                if node.args.args and node.args.args[0].arg in ("self", "cls"):
                    n_args -= 1
                assert n_args <= 8, f"{rel_path}:{node.name} has {n_args} arguments (> 8 ceiling)"


def test_plan034_hypothesis_experiments_json_results() -> None:
    """Verify `.cache/hypothesis_experiments_034.json` records H1=True, H2=True, and H3=True."""
    json_path = ROOT / ".cache" / "hypothesis_experiments_034.json"
    assert json_path.exists(), "Missing .cache/hypothesis_experiments_034.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    hyps = data["hypotheses"]

    h1 = hyps["H1_wavelet_cholesky_isometry_and_tied_2pow16_basis"]
    assert h1["passed"] is True
    assert h1["reconstruction_max_err"] < 1e-5
    assert h1["energy_ratio_err"] < 1e-5
    assert h1["perturbed_lifting_recon_max_err"] < 1e-5
    assert h1["zero_separate_output_tables"] is True
    assert h1["tied_model_params_M"] < 20.0
    assert h1["parameter_savings_M"] > 20.0
    assert h1["step0_max_self_overlap"] <= 0.15

    h2 = hyps["H2_mach_zehnder_and_tied_2pow16_wavelet_cholesky_900steps"]
    assert h2["passed"] is True
    assert h2["mach_zehnder_wavchol_ON_best_val_ce"] < h2["uncoupled_equidim_ON_best_val_ce"]
    assert h2["throughput_speedup_vs_fe2"] >= 1.50
    assert h2["mach_zehnder_wavchol_ON_params_M"] < h2["isolate3_params_M"]
    assert h2["tied_2pow16_param_savings_M"] > 20.0

    h3 = hyps["H3_3600step_29p5M_token_wavelet_cholesky_mach_zehnder_ON"]
    assert h3["passed"] is True
    assert h3["wavchol_mach_zehnder_ON_3600_best_val_ce"] < h3["classical_3600_best_val_ce"]
    assert h3["wavchol_mach_zehnder_ON_3600_best_val_ce"] < 5.00
    assert h3["margin_vs_classical_nats"] >= 0.53
    assert h3["wavchol_mach_zehnder_ON_3600_final_top1_acc"] >= 25.0
    assert h3["min_diffuse_eff_rank"] >= 6.5
    assert h3["speedup_vs_fe2_3600"] >= 1.35
    assert h3["peak_vram_gb"] <= 8.0
    assert all(h3["checkpoint_quantum_beats_classical"].values())
