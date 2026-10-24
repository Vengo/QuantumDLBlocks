"""Unit tests for Experiment Plan 034: Multi-Level Wavelet-Cholesky, Mach-Zehnder O(N), Tied-2^16 Basis, and Promoted CLI Tooling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.layers.overcomplete_two_beam import OvercompleteTwoBeamInterference
from qdlblocks.layers.wavelet_cholesky import (
    BlockCholeskyLinear,
    WaveletCholeskyConfig,
    WaveletCholeskyMultiscaleBranch,
    apply_register_cholesky_coupling,
)
from qdlblocks.models.born_gpt import QuantumBornGPT


def _load_script_module(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_wavelet_cholesky_exact_reconstruction_and_spd() -> None:
    """Verify exact reconstruction (Step-0 & non-zero lifting), finite gradients, and SPD Cholesky."""
    torch.manual_seed(2034)
    cfg = WaveletCholeskyConfig(embed_dim=128, num_levels=3, max_block_size=16, init_nonlin_gate=0.0)
    branch = WaveletCholeskyMultiscaleBranch(cfg)

    psi = F.normalize(torch.randn(4, 16, 128), p=2.0, dim=-1)
    coarse_0, details_0 = branch.downsample_amplitudes(psi)
    recon_0 = branch.upsample_amplitudes(coarse_0, details_0)
    vcycle_0 = branch(psi)

    assert float((recon_0 - psi).abs().max().item()) < 1e-6
    assert float((vcycle_0 - psi).abs().max().item()) < 1e-6
    energy_ratio_0 = float((recon_0.pow(2).sum(dim=-1) / psi.pow(2).sum(dim=-1)).mean().item())
    assert abs(energy_ratio_0 - 1.0) < 1e-6

    with torch.no_grad():
        for s in range(cfg.num_levels):
            branch.lift_predict[s].normal_(0.0, 0.35)
            branch.lift_update[s].normal_(0.0, 0.35)

    coarse_nz, details_nz = branch.downsample_amplitudes(psi)
    recon_nz = branch.upsample_amplitudes(coarse_nz, details_nz)
    vcycle_nz = branch(psi)
    assert float((recon_nz - psi).abs().max().item()) < 1e-5
    assert float((vcycle_nz - psi).abs().max().item()) < 1e-5

    psi_grad = psi.clone().detach().requires_grad_(True)
    out = branch(psi_grad)
    loss = (out.pow(2).sum(dim=-1)).mean()
    loss.backward()
    assert psi_grad.grad is not None and torch.isfinite(psi_grad.grad).all()
    for p in branch.parameters():
        if p is not branch.branch_scale:
            assert p.grad is not None and torch.isfinite(p.grad).all()

    chol = BlockCholeskyLinear(dim=64, max_block_size=16)
    with torch.no_grad():
        chol.raw_tril.normal_(0.0, 0.25)
        chol.raw_diag.normal_(0.0, 0.20)
    l_mat = chol.cholesky_factor()
    c_spd = l_mat @ l_mat.transpose(-1, -2)
    eigvals = torch.linalg.eigvalsh(c_spd)
    metrics = chol.condition_metrics()
    assert float(eigvals.min().item()) > 0.0
    assert metrics["min_diag"] > 0.0
    assert metrics["spd_cond_number"] >= 1.0

    # Also verify register Cholesky coupling helper
    base_q = torch.randn(32, 8, 8)
    tril_q = torch.zeros(8, 8, 8)
    diag_q = torch.zeros(8, 8)
    pkts = torch.zeros(16, 8, 8)
    pkt_idx = torch.arange(32) % 16
    coupled_q = apply_register_cholesky_coupling(base_q, tril_q, diag_q, pkts, pkt_idx)
    assert float((coupled_q - base_q).abs().max().item()) < 1e-6


def test_mach_zehnder_and_wavelet_cholesky_interference() -> None:
    """Verify SO(2) Mach-Zehnder beamsplitter energy conservation and unit-norm two-beam outputs."""
    torch.manual_seed(2034)
    layer = OvercompleteTwoBeamInterference(
        embed_dim=64,
        frame_expansion=1.0,
        init_steps=10,
        use_stereographic_lift=True,
        use_cosine_phase_beam_gate=True,
        use_mach_zehnder_beamsplitter=True,
        use_wavelet_cholesky_branch=True,
    )
    assert layer.beamsplitter_theta is not None
    assert layer.wavelet_cholesky_branch is not None

    psi = F.normalize(torch.randn(2, 12, 64), p=2.0, dim=-1).requires_grad_(True)
    u1 = F.normalize(layer.W_beam1.detach(), p=2.0, dim=-1)
    u2 = F.normalize(layer.W_beam2.detach(), p=2.0, dim=-1)
    z1 = F.linear(psi.detach(), u1) * (64.0**0.5)
    z2 = F.linear(psi.detach(), u2) * (64.0**0.5)
    theta = layer.beamsplitter_theta.detach()
    z1_rot = torch.cos(theta) * z1 - torch.sin(theta) * z2
    z2_rot = torch.sin(theta) * z1 + torch.cos(theta) * z2
    energy_before = z1.pow(2) + z2.pow(2)
    energy_after = z1_rot.pow(2) + z2_rot.pow(2)
    assert float((energy_before - energy_after).abs().max().item()) < 1e-5

    psi_main, psi_beam2 = layer(psi, return_second_beam=True)
    assert float((psi_main.norm(dim=-1) - 1.0).abs().max().item()) < 1e-5
    assert float((psi_beam2.norm(dim=-1) - 1.0).abs().max().item()) < 1e-5

    loss = (psi_main * psi_beam2).sum() + psi_main.sum()
    loss.backward()
    assert layer.beamsplitter_theta.grad is not None and torch.isfinite(layer.beamsplitter_theta.grad).all()
    assert layer.wavelet_cholesky_branch.branch_scale.grad is not None


def test_quantum_born_gpt_tied_wavelet_cholesky_basis_cpu() -> None:
    """Verify QuantumBornGPT with use_wavelet_cholesky_basis=True eliminates untied tables and backpropagates."""
    torch.manual_seed(2034)
    common_kwargs = dict(
        vocab_size=4096,
        embed_dim=64,
        seed_dim=64,
        num_heads=4,
        num_layers=2,
        block_size=32,
        readout_mode="asymmetric",
        use_overcomplete_two_beam=True,
        frame_expansion=1.0,
        use_purified_attn=True,
        use_gated_spherical_residual=True,
        use_dynamic_povm_beta=True,
        use_povm_log_prior=True,
        use_ortho_decouple=False,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=4,
        num_higher_states=16,
        use_two_beam_kboson_readout=True,
        diffuse_rank=8,
        kraus_spectral_floor=6.5,
        use_mach_zehnder_beamsplitter=True,
        use_wavelet_cholesky_branch=True,
    )
    untied_model = QuantumBornGPT(**common_kwargs, use_wavelet_cholesky_basis=False)
    tied_model = QuantumBornGPT(**common_kwargs, use_wavelet_cholesky_basis=True)

    assert tied_model.povm_head is None
    assert tied_model.hierarchical_povm_readout is not None
    hr = tied_model.hierarchical_povm_readout
    assert hr.povm_odd_qubits is None

    untied_params = sum(p.numel() for p in untied_model.parameters())
    tied_params = sum(p.numel() for p in tied_model.parameters())
    assert tied_params < untied_params

    x = torch.randint(0, 4096, (2, 16), dtype=torch.long)
    y = torch.randint(0, 4096, (2, 16), dtype=torch.long)
    _, loss = tied_model(x, targets=y, direct_chunked_ce=True)
    assert loss is not None and torch.isfinite(loss).item()
    loss.backward()

    assert tied_model.hierarchical_qubit_embed is not None
    u_q = tied_model.hierarchical_qubit_embed.unigram_qubits
    for param in (
        u_q,
        hr.wavelet_packets_256,
        hr.wavelet_packets_1024,
        hr.wavelet_packets_4096,
        hr.basis_chol_tril,
    ):
        assert param is not None
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().max().item()) > 0.0


def test_promoted_plan034_cli_scripts(tmp_path: Path) -> None:
    """Verify ast_signature_and_length_linter.py and wavelet_cholesky_spectrum_probe.py on CPU."""
    linter_mod = _load_script_module("ast_signature_and_length_linter.py")
    linter_json = tmp_path / "linter_report.json"
    linter_cfg = linter_mod.parse_args(["--out-json", str(linter_json)])
    linter_rep = linter_mod.run_linter(linter_cfg)
    assert linter_rep["all_passed"] is True
    assert linter_json.exists()
    assert linter_mod.main(["--out-json", str(linter_json)]) == 0

    probe_mod = _load_script_module("wavelet_cholesky_spectrum_probe.py")
    probe_json = tmp_path / "wavelet_probe.json"
    probe_cfg = probe_mod.parse_args(
        ["--embed-dim", "128", "--vocab-size", "512", "--qubit-amplitudes", "8", "--device", "cpu", "--out-json", str(probe_json)]
    )
    probe_rep = probe_mod.run_wavelet_cholesky_probe(probe_cfg)
    wav = probe_rep["wavelet_reconstruction_and_spectrum"]
    tied = probe_rep["tied_2pow16_basis"]
    assert wav["step0_max_abs_err"] < 1e-5
    assert wav["nonzero_lift_max_abs_err"] < 1e-5
    assert abs(wav["energy_fraction_sum"] - 1.0) < 1e-5
    assert wav["min_cholesky_diag"] > 0.0
    assert tied["param_savings"] > 0
    assert tied["povm_head_is_none"] is True
    assert tied["povm_odd_qubits_is_none"] is True
    assert tied["step0_max_self_overlap"] <= 0.15
    assert probe_json.exists()
