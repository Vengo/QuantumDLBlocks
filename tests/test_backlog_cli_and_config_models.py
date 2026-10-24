"""Unit & AST Best-Practices Tests for Plan 033 Config Models, Spectral Floor, and Backlog CLIs."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import full_ambient_step0_calibration_audit as step0_audit
import vram_envelope_autotuner as vram_tuner

from qdlblocks.benchmarks.config_models import (
    BackboneConfig,
    KBosonReadoutConfig,
    OptimizerConfig,
    QuantumExperimentSpec,
    QubitHierarchyConfig,
    build_quantum_born_optimizers,
    compute_wsd_lr_multiplier,
    initialize_povm_log_prior,
)
from qdlblocks.layers.hierarchical_qubit import Hierarchical16QubitPOVMReadout
from qdlblocks.models.born_gpt import QuantumBornGPT


def _count_non_self_args(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count total positional, keyword-only, and vararg/kwarg parameters excluding `self`/`cls`."""
    args = func_node.args
    names = [a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    extra = (1 if args.vararg is not None else 0) + (1 if args.kwarg is not None else 0)
    return len(names) + extra


@pytest.mark.parametrize(
    "rel_path,max_lines",
    [
        ("src/qdlblocks/benchmarks/config_models.py", 400),
        ("scripts/full_ambient_step0_calibration_audit.py", 300),
        ("scripts/vram_envelope_autotuner.py", 300),
        ("docs/SCRIPT_BEST_PRACTICES.md", 400),
    ],
)
def test_ast_file_length_and_signature_ceilings(rel_path: str, max_lines: int) -> None:
    """Verify Rules 1 & 2 (`< 400` lines / `< 300` for scripts, `<= 8` args per function) via AST."""
    path = ROOT / rel_path
    assert path.exists(), f"Missing expected file: {path}"
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    assert len(lines) < max_lines, f"{rel_path} has {len(lines)} lines, exceeding ceiling {max_lines}"

    if path.suffix == ".py":
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_args = _count_non_self_args(node)
                assert n_args <= 8, (
                    f"{rel_path}:{node.lineno} function '{node.name}' takes {n_args} arguments (> 8 ceiling)"
                )


def test_frozen_config_dataclasses_and_wsd_schedule() -> None:
    """Test frozen dataclass composition and Warmup-Stable-Decay multiplier schedule."""
    spec = QuantumExperimentSpec(
        key="test_w18_spec",
        label="Test Plan 033 Spec",
        short_label="Test-P33",
        steps=100,
        backbone=BackboneConfig(num_layers=2, embed_dim=64, seed_dim=64, num_heads=4, frame_expansion=1.0),
        hierarchy=QubitHierarchyConfig(qubit_registers=8, qubit_amplitudes=8, num_higher_states=16),
        readout=KBosonReadoutConfig(diffuse_rank=8, kraus_spectral_floor=6.5),
        optimizer=OptimizerConfig(qubit_embed_lr=0.006),
    )
    assert spec.backbone.frame_expansion == 1.0
    assert spec.readout.kraus_spectral_floor == 6.5
    assert spec.optimizer.qubit_embed_lr == 0.006

    # Warmup -> Stable -> Decay checks
    m_start = compute_wsd_lr_multiplier(0, total_steps=100, warmup_steps=10, stable_frac=0.60, min_ratio=0.10)
    m_stable = compute_wsd_lr_multiplier(40, total_steps=100, warmup_steps=10, stable_frac=0.60, min_ratio=0.10)
    m_end = compute_wsd_lr_multiplier(100, total_steps=100, warmup_steps=10, stable_frac=0.60, min_ratio=0.10)
    assert math.isclose(m_start, 0.10, rel_tol=1e-5)
    assert math.isclose(m_stable, 1.00, rel_tol=1e-5)
    assert math.isclose(m_end, 0.10, rel_tol=1e-5)


def test_initialize_povm_log_prior_and_optimizer_routing_cpu() -> None:
    """Verify Bose-saturated log-prior initialization and Gentle 2x Zero-WD Qubit/VE routing on CPU."""
    torch.manual_seed(42)
    model = QuantumBornGPT(
        vocab_size=128,
        embed_dim=64,
        seed_dim=64,
        num_heads=4,
        num_layers=2,
        block_size=32,
        use_overcomplete_two_beam=True,
        frame_expansion=1.0,
        use_purified_attn=True,
        use_hierarchical_16qubit=True,
        qubit_registers=8,
        qubit_amplitudes=8,
        num_higher_states=16,
        use_two_beam_kboson_readout=True,
        diffuse_rank=8,
        kraus_spectral_floor=6.5,
        use_povm_log_prior=True,
        num_value_embeddings=2,
    )
    tokens = torch.randint(0, 128, (2048,), dtype=torch.int64)
    tokens[:512] = 0  # heavy head token
    initialize_povm_log_prior(model, tokens, alpha=0.62, bose_saturate_top_k=8)
    prior = model.povm_log_prior.detach()
    assert abs(float(prior.mean().item())) < 1e-4
    assert float(prior[0].item()) > float(prior[-1].item())

    opt_cfg = OptimizerConfig(muon_lr=0.018, body_adam_lr=0.003, qubit_embed_lr=0.006, head_lr=0.012)
    muon_opt, adam_opt, m_lr, a_lr = build_quantum_born_optimizers(model, opt_cfg)
    assert muon_opt is not None
    assert math.isclose(m_lr, 0.018)
    assert math.isclose(a_lr, 0.003)

    # Locate the Gentle 2x Zero-WD Qubit/Value-Embedding group (`lr=0.006, weight_decay=0.0`)
    embed_groups = [g for g in adam_opt.param_groups if math.isclose(g["lr"], 0.006)]
    assert len(embed_groups) == 1
    assert embed_groups[0]["weight_decay"] == 0.0
    group_param_ids = {id(p) for p in embed_groups[0]["params"]}
    assert id(model.hierarchical_qubit_embed.unigram_qubits) in group_param_ids
    assert id(model.hierarchical_qubit_embed.higher_states) in group_param_ids
    for ve in model.value_embeds:
        assert id(ve.weight) in group_param_ids


def test_soft_powerlaw_spectral_floor_identity_and_floor_lifting() -> None:
    """Verify H1: `apply_soft_powerlaw_spectral_floor` is exact identity when R_diff >= 6.5 and lifts decayed spectra."""
    torch.manual_seed(2033)
    readout = Hierarchical16QubitPOVMReadout(
        vocab_size=128,
        embed_dim=64,
        num_registers=8,
        register_dim=8,
        num_higher_states=16,
        diffuse_rank=16,
        kraus_spectral_floor=6.5,
    )

    # 1. Healthy high-rank activation (R_diff >= 6.5) -> bit-for-bit identity
    healthy_p = torch.randn(128, 16, dtype=torch.float32)
    out_healthy = readout.apply_soft_powerlaw_spectral_floor(healthy_p, r_diff_floor=6.5)
    assert readout.last_diffuse_r_before_floor >= 6.5
    assert float((out_healthy - healthy_p).abs().max().item()) == 0.0

    # 2. Decayed spectrum (R_diff in [4.5, 6.2] < 6.5) -> lifted to >= 6.5 while preserving Frobenius norm and finite grads
    u_orth, _ = torch.linalg.qr(torch.randn(128, 16))
    v_orth, _ = torch.linalg.qr(torch.randn(16, 16))
    idx = torch.arange(1, 17, dtype=torch.float32)
    decayed_s = 1.65 * (idx**-0.72)
    decayed_s[0] = 10.0
    raw_decayed = (u_orth * decayed_s.unsqueeze(0)) @ v_orth.t()
    raw_decayed.requires_grad_(True)

    floored_p = readout.apply_soft_powerlaw_spectral_floor(raw_decayed, r_diff_floor=6.5)
    assert 4.5 <= readout.last_diffuse_r_before_floor < 6.2
    assert readout.last_diffuse_r_after_floor >= 6.5
    assert readout.last_diffuse_r_after_floor <= 11.0

    # Frobenius norm preserved within 1e-4
    norm_before = float(raw_decayed.detach().norm(p="fro").item())
    norm_after = float(floored_p.detach().norm(p="fro").item())
    assert math.isclose(norm_before, norm_after, rel_tol=1e-4)

    # Backward pass finite gradients
    loss = floored_p.square().sum()
    loss.backward()
    assert raw_decayed.grad is not None
    assert bool(torch.isfinite(raw_decayed.grad).all().item())


def test_full_ambient_step0_calibration_audit_cli_cpu(tmp_path: Path) -> None:
    """Test `scripts/full_ambient_step0_calibration_audit.py` on CPU in < 5s."""
    out_file = tmp_path / "step0_audit.json"
    cfg = step0_audit.parse_args(
        [
            "--embed-dim",
            "64",
            "--qubit-registers",
            "8",
            "--qubit-amplitudes",
            "8",
            "--num-higher-states",
            "16",
            "--device",
            "cpu",
            "--out-json",
            str(out_file),
        ]
    )
    report = step0_audit.run_step0_audit_table(cfg)
    assert out_file.exists()
    assert len(report["rows"]) == 3
    janti_row = report["rows"][2]
    legacy_row = report["rows"][1]
    assert janti_row["mode"] == "Full-Ambient J_anti"
    assert janti_row["self_register_overlap_max"] < 1e-5
    assert legacy_row["self_register_overlap_max"] > 0.25
    assert janti_row["step0_val_ce"] < legacy_row["step0_val_ce"]


def test_vram_envelope_autotuner_cli_analytical_cpu(tmp_path: Path) -> None:
    """Test `scripts/vram_envelope_autotuner.py --analytical-only` on CPU in < 1s."""
    out_file = tmp_path / "vram_autotune.json"
    cfg, out_path = vram_tuner.parse_args(
        [
            "--max-vram-gb",
            "8.0",
            "--num-layers",
            "4",
            "--embed-dim",
            "256",
            "--num-heads",
            "4",
            "--seq-len",
            "512",
            "--analytical-only",
            "--out-json",
            str(out_file),
        ]
    )
    res = vram_tuner.autotune_vram_envelope(cfg, out_json=out_path)
    assert out_file.exists()
    assert res["num_feasible"] > 0
    rec = res["recommended"]
    assert rec["fits_budget"] is True
    assert rec["est_peak_vram_gb"] <= 8.0 * 0.94
    assert rec["micro_batch_size"] >= 1
    assert rec["chunk_tokens"] >= 512
