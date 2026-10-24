"""Unit and integration tests for Experiment Plan 033 (Soft Spectral Floor, O(N) Scaling, & 3,600-Step Study)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import torch

from qdlblocks.layers.hierarchical_qubit import Hierarchical16QubitPOVMReadout

ROOT = Path(__file__).resolve().parents[1]


def test_h1_soft_powerlaw_spectral_floor_identity_and_lift() -> None:
    """Verify `apply_soft_powerlaw_spectral_floor` is identity above floor and lifts decayed spectra smoothly."""
    readout = Hierarchical16QubitPOVMReadout(
        vocab_size=128, embed_dim=256, diffuse_rank=16, kraus_spectral_floor=6.5
    )
    torch.manual_seed(2033)
    u_hi, _, vt_hi = torch.linalg.svd(torch.randn(256, 16), full_matrices=False)
    s_hi = torch.arange(1, 17, dtype=torch.float32) ** -0.25
    x_hi = ((u_hi * s_hi.unsqueeze(0)) @ vt_hi).view(2, 128, 16)
    y_hi = readout.apply_soft_powerlaw_spectral_floor(x_hi)
    assert readout.last_diffuse_r_before_floor >= 6.5
    assert float((y_hi - x_hi).abs().max().item()) == 0.0

    s_dec = torch.arange(1, 17, dtype=torch.float32) ** -1.65
    x_dec = (((u_hi * s_dec.unsqueeze(0)) @ vt_hi).view(2, 128, 16)).detach().requires_grad_(True)
    y_dec = readout.apply_soft_powerlaw_spectral_floor(x_dec)
    (y_dec.pow(2).sum()).backward()
    assert x_dec.grad is not None and torch.isfinite(x_dec.grad).all()

    s_out = torch.linalg.svdvals(y_dec.detach().reshape(-1, 16).float())
    p_out = s_out / s_out.sum().clamp_min(1e-12)
    r_after = float(torch.exp(-(p_out * torch.log(p_out.clamp_min(1e-12))).sum()).item())
    fro_err = abs(y_dec.detach().norm().item() - x_dec.detach().norm().item()) / x_dec.detach().norm().item()
    assert readout.last_diffuse_r_before_floor < 6.5
    assert 6.5 <= r_after <= 11.0
    assert fro_err < 1e-4
    assert bool(torch.all(s_out[:-1] >= s_out[1:] - 1e-5).item())


def test_plan033_script_best_practices_ast_compliance() -> None:
    """Verify `examples/43_verify_plan033_scaling_and_soft_spectral_floor.py` obeys Rules 1 & 2 (<400 lines, <=8 args)."""
    for rel_path in (
        "examples/43_verify_plan033_scaling_and_soft_spectral_floor.py",
        "src/qdlblocks/benchmarks/config_models.py",
    ):
        path = ROOT / rel_path
        assert path.exists(), f"Missing {rel_path}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) < 400, f"{rel_path} has {len(lines)} lines (must be < 400)"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                n_args = len(node.args.args) + len(node.args.kwonlyargs)
                if node.args.args and node.args.args[0].arg in ("self", "cls"):
                    n_args -= 1
                assert n_args <= 8, f"{rel_path}:{node.name} has {n_args} arguments (> 8 ceiling)"


def test_plan033_hypothesis_experiments_json_results() -> None:
    """Verify `.cache/hypothesis_experiments_033.json` records H1=PASS, H2=PASS, and H3=PASS."""
    json_path = ROOT / ".cache" / "hypothesis_experiments_033.json"
    assert json_path.exists(), "Missing .cache/hypothesis_experiments_033.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    hyps = data["hypotheses"]

    h1 = hyps["H1_soft_powerlaw_spectral_floor"]
    assert h1["passed"] is True
    assert h1["identity_when_above_floor_max_diff"] == 0.0
    assert 6.5 <= h1["decayed_r_after"] <= 11.0

    h2 = hyps["H2_parity_split_and_equidim_ON_900steps"]
    assert h2["passed"] is True
    assert h2["isolate3_16x16_best_val_ce"] < 5.65
    assert h2["parity_split_16x8_best_val_ce"] < 5.65
    assert h2["throughput_speedup_ratio"] >= 1.25
    assert h2["equidim_ON_params_M"] < h2["isolate3_params_M"]
    assert h2["equidim_ON_best_val_ce"] < 5.72

    h3 = hyps["H3_3600step_29p5M_token_scaling"]
    assert h3["passed"] is True
    assert h3["quantum_3600_best_val_ce"] < h3["classical_3600_best_val_ce"]
    assert h3["quantum_min_diffuse_eff_rank"] >= 6.5
    assert h3["peak_vram_gb"] <= 8.0
    assert all(h3["checkpoint_quantum_beats_classical"].values())
