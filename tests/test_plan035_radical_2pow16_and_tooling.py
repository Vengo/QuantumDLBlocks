"""Unit tests for Experiment Plan 035: Standalone Radical N=2^16 Wavelet-Cholesky Architectures & Promoted CLI Tooling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p_dir in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p_dir not in sys.path:
        sys.path.insert(0, p_dir)

from vocab_dark_ancilla_utilization_audit import _mean_intra_block_cosine

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_dyadic_2pow16_spectral_tree,
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


def test_build_dyadic_2pow16_spectral_tree_bijection_and_coherence() -> None:
    """Verify exact bijection and >= 2.0x intra-block cosine coherence gain on structured clusters."""
    dark_mod = _load_script_module("vocab_dark_ancilla_utilization_audit.py")
    cfg = dark_mod.DarkAncillaAuditConfig(vocab_size=2048, ambient_n=4096, embed_dim=64, device="cpu")
    res = dark_mod.audit_dyadic_spectral_tree(cfg)
    perm_v_to_leaf, leaf_to_v = res["perm_v_to_leaf"], res["leaf_to_v"]
    arange_n = torch.arange(cfg.ambient_n, dtype=torch.long)
    assert res["is_bijection"] is True
    assert torch.equal(leaf_to_v[perm_v_to_leaf], arange_n)
    assert torch.equal(perm_v_to_leaf[leaf_to_v], arange_n)
    assert res["spectral_block8_cos"] > 0.35
    assert res["block8_gain_ratio"] >= 2.0
    assert res["block64_gain_ratio"] >= 2.0


def test_radical_2pow16_gpt_both_architectures_cpu() -> None:
    """Verify parameter compression (<11M), causal masking, finite gradients, and Step-0 self-overlap <0.05."""
    torch.manual_seed(2035)
    for mode in ("multiscale_unet", "consistent_kronecker"):
        full_model = Radical2Pow16WaveletCholeskyGPT(Radical2Pow16Config(arch_mode=mode))
        total_params = sum(p.numel() for p in full_model.parameters())
        assert total_params < 11_000_000, f"{mode} has {total_params:,} params (>= 11M)"

        small_cfg = Radical2Pow16Config(
            arch_mode=mode, vocab_size=50304, valid_vocab_size=50257, ambient_n=65536,
            num_layers=2, embed_dim=64, num_heads=4, block_size=32, window_size=32,
            fine_detail_dim=16, mid_detail_dim=32, diffuse_rank=8,
        )
        model = Radical2Pow16WaveletCholeskyGPT(small_cfg)
        gen = torch.Generator(device="cpu").manual_seed(3035)
        ppmi = F.normalize(torch.randn((50304, 64), generator=gen), p=2.0, dim=-1)
        toks = torch.randint(0, 50257, (8192,), generator=gen, dtype=torch.long)
        init_stats = initialize_radical_2pow16_with_ppmi_svd(model, ppmi, toks, toks)
        assert init_stats["max_self_overlap"] < 0.05

        x1 = torch.randint(0, 50257, (2, 16), generator=gen, dtype=torch.long)
        x2 = x1.clone()
        x2[:, 8:] = (x2[:, 8:] + 137) % 50257
        with torch.no_grad():
            logits1, _ = model(x1, targets=None, return_logits=True)
            logits2, _ = model(x2, targets=None, return_logits=True)
        assert logits1 is not None and logits2 is not None
        assert float((logits1[:, :8, :] - logits2[:, :8, :]).abs().max().item()) == 0.0
        assert float((logits1[:, 8:, :] - logits2[:, 8:, :]).abs().max().item()) > 0.0

        y = torch.randint(0, 50257, (2, 16), generator=gen, dtype=torch.long)
        _, loss = model(x1, targets=y, return_logits=False)
        assert loss is not None and torch.isfinite(loss).item()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {mode}:{name}"

        muon_opt, adam_opt = build_radical_2pow16_optimizers(model)
        assert muon_opt is not None and adam_opt is not None


def test_promoted_plan035_cli_scripts_and_ast_compliance(tmp_path: Path) -> None:
    """Verify vocab_dark_ancilla_utilization_audit.py, mera_entanglement_cone_visualizer.py, and AST rules."""
    dark_mod = _load_script_module("vocab_dark_ancilla_utilization_audit.py")
    dark_json = tmp_path / "dark_ancilla_audit.json"
    dark_cfg = dark_mod.parse_args(
        ["--vocab-size", "1024", "--ambient-n", "2048", "--embed-dim", "64", "--device", "cpu", "--out-json", str(dark_json)]
    )
    dark_rep = dark_mod.run_dark_ancilla_audit(dark_cfg)
    assert dark_rep["dyadic_spectral_tree"]["is_bijection"] is True
    assert dark_rep["dyadic_spectral_tree"]["block8_gain_ratio"] >= 2.0
    assert dark_rep["dyadic_spectral_tree"]["block64_gain_ratio"] >= 2.0
    assert dark_rep["parameter_compression"]["multiscale_unet"]["under_11m_target"] is True
    assert dark_rep["parameter_compression"]["consistent_kronecker"]["under_11m_target"] is True
    assert dark_rep["dark_ancilla_utilization"]["dark_ancilla_mean_norm"] > 0.0
    assert dark_rep["dark_ancilla_utilization"]["dark_ancilla_energy_fraction"] > 0.0
    assert dark_json.exists()

    cone_mod = _load_script_module("mera_entanglement_cone_visualizer.py")
    cone_json = tmp_path / "entanglement_cone.json"
    cone_cfg = cone_mod.parse_args(
        ["--embed-dim", "64", "--seq-len", "32", "--num-layers", "2", "--device", "cpu", "--out-json", str(cone_json)]
    )
    cone_rep = cone_mod.run_entanglement_cone_visualizer(cone_cfg)
    for mode in ("multiscale_unet", "consistent_kronecker"):
        lc = cone_rep["causal_light_cones"][mode]
        assert lc["strictly_causal_past_zero"] is True
        assert lc["max_past_perturbation"] == 0.0
        assert lc["downstream_cone_active"] is True
        assert lc["min_future_perturbation"] > 0.0
    ke = cone_rep["kronecker_coarse_fine_entanglement"]
    assert ke["coarse_to_fine_coupling"] > 0.0
    assert ke["fine_to_coarse_coupling"] > 0.0
    assert ke["beamsplitter_energy_err"] < 1e-5
    assert cone_json.exists()

    linter_mod = _load_script_module("ast_signature_and_length_linter.py")
    linter_cfg = linter_mod.parse_args([
        "scripts/vocab_dark_ancilla_utilization_audit.py",
        "scripts/mera_entanglement_cone_visualizer.py",
        "tests/test_plan035_radical_2pow16_and_tooling.py",
        "--max-lines",
        "240",
    ])
    linter_rep = linter_mod.run_linter(linter_cfg)
    assert linter_rep["all_passed"] is True
    assert all(r["line_count"] < 240 for r in linter_rep["rows"])
