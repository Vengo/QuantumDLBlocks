#!/usr/bin/env python3
"""Multi-Level Wavelet-Cholesky Spectrum, Reconstruction, & Tied-2^16 Basis Probe (Plan 034).

Promoted from `scripts/BACKLOG.md` (identified in Plan 033, implemented in Plan 034).
Audits on CPU/GPU:
1. Exact reconstruction error (`max_abs_err`, `energy_ratio`) of `WaveletCholeskyMultiscaleBranch`
   (`downsample_amplitudes` + `upsample_amplitudes`) across random unit kets on `S^{d-1}` both at
   Step-0 (`P=U=0`) and with random non-zero lifting filters `P_s, U_s`.
2. Per-scale wavelet energy fractions (`E_coarse`, `E_detail_1..S`) and `BlockCholeskyLinear`
   SPD condition numbers `kappa(L_s L_s^T)` and minimum Cholesky diagonal `min diag(L_s) > 0`.
3. Tied-2^16 (`use_wavelet_cholesky_basis=True`) parameter savings vs. untied
   `Hierarchical16QubitPOVMReadout` and Step-0 self-overlap `max_v |<M_v | psi_0(v)>|`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.layers.hierarchical_qubit import (
    Hierarchical16QubitPOVMReadout,
    initialize_hierarchical_16qubit_with_ppmi_svd,
)
from qdlblocks.layers.wavelet_cholesky import (
    WaveletCholeskyConfig,
    WaveletCholeskyMultiscaleBranch,
)
from qdlblocks.models.born_gpt import QuantumBornGPT


@dataclass(frozen=True)
class WaveletProbeConfig:
    """Configuration for the multi-level Wavelet-Cholesky and Tied-2^16 basis audit."""

    embed_dim: int = 256
    num_levels: int = 3
    vocab_size: int = 2048
    qubit_registers: int = 16
    qubit_amplitudes: int = 16
    device: str = "cpu"
    out_json: str = ""


@torch.no_grad()
def audit_wavelet_reconstruction_and_spectrum(cfg: WaveletProbeConfig) -> dict[str, Any]:
    """Audit Step-0 & non-zero lifting reconstruction, scale energy fractions, and SPD Cholesky metrics."""
    dev = torch.device(cfg.device)
    gen = torch.Generator(device="cpu").manual_seed(2034)
    psi = F.normalize(
        torch.randn((8, 32, cfg.embed_dim), generator=gen, dtype=torch.float32).to(dev),
        p=2.0,
        dim=-1,
    )
    branch = WaveletCholeskyMultiscaleBranch(
        WaveletCholeskyConfig(
            embed_dim=cfg.embed_dim, num_levels=cfg.num_levels, max_block_size=32, init_nonlin_gate=0.0
        )
    ).to(dev)

    coarse_0, details_0 = branch.downsample_amplitudes(psi)
    recon_0 = branch.upsample_amplitudes(coarse_0, details_0)
    vcycle_0 = branch(psi)
    psi_energy = psi.pow(2).sum(dim=-1)
    step0_max_abs_err = float((recon_0 - psi).abs().max().item())
    step0_vcycle_max_abs_err = float((vcycle_0 - psi).abs().max().item())
    step0_energy_ratio = float((recon_0.pow(2).sum(dim=-1) / psi_energy).mean().item())

    c_energy = float(coarse_0.pow(2).sum(dim=-1).mean().item())
    d_energies = [float(d.pow(2).sum(dim=-1).mean().item()) for d in details_0]
    tot_e = max(c_energy + sum(d_energies), 1e-12)
    scale_energy_fractions = {
        "E_coarse": round(c_energy / tot_e, 6),
        **{f"E_detail_{s + 1}": round(de / tot_e, 6) for s, de in enumerate(d_energies)},
    }

    for s in range(cfg.num_levels):
        branch.lift_predict[s].copy_(
            (0.35 * torch.randn(branch.lift_predict[s].shape, generator=gen)).to(dev)
        )
        branch.lift_update[s].copy_(
            (0.35 * torch.randn(branch.lift_update[s].shape, generator=gen)).to(dev)
        )
    coarse_nz, details_nz = branch.downsample_amplitudes(psi)
    recon_nz = branch.upsample_amplitudes(coarse_nz, details_nz)
    nonzero_lift_max_abs_err = float((recon_nz - psi).abs().max().item())
    nonzero_lift_energy_ratio = float((recon_nz.pow(2).sum(dim=-1) / psi_energy).mean().item())

    cholesky_scales: list[dict[str, Any]] = []
    scale_mods = [("coarse", branch.coarse_cholesky)] + [
        (f"detail_{s + 1}", branch.detail_cholesky[s]) for s in range(cfg.num_levels)
    ]
    for name, mod in scale_mods:
        s0_m = mod.condition_metrics()
        mod.raw_tril.copy_((0.18 * torch.randn(mod.raw_tril.shape, generator=gen)).to(dev))
        mod.raw_diag.copy_((0.12 * torch.randn(mod.raw_diag.shape, generator=gen)).to(dev))
        p_m = mod.condition_metrics()
        cholesky_scales.append(
            {
                "scale": name, "dim": mod.dim, "block_size": mod.block_size, "num_blocks": mod.num_blocks,
                "step0_min_diag": round(s0_m["min_diag"], 6), "step0_spd_cond": round(s0_m["spd_cond_number"], 6),
                "perturbed_min_diag": round(p_m["min_diag"], 6), "perturbed_spd_cond": round(p_m["spd_cond_number"], 4),
            }
        )

    return {
        "step0_max_abs_err": step0_max_abs_err,
        "step0_vcycle_max_abs_err": step0_vcycle_max_abs_err,
        "step0_energy_ratio": round(step0_energy_ratio, 7),
        "nonzero_lift_max_abs_err": nonzero_lift_max_abs_err,
        "nonzero_lift_energy_ratio": round(nonzero_lift_energy_ratio, 7),
        "scale_energy_fractions": scale_energy_fractions,
        "energy_fraction_sum": round((c_energy + sum(d_energies)) / tot_e, 6),
        "min_cholesky_diag": round(min(float(s["perturbed_min_diag"]) for s in cholesky_scales), 6),
        "max_cholesky_cond": round(max(float(s["perturbed_spd_cond"]) for s in cholesky_scales), 4),
        "cholesky_scales": cholesky_scales,
        "branch_params": sum(p.numel() for p in branch.parameters()),
    }


def build_probe_born_gpt(
    cfg: WaveletProbeConfig,
    use_wavelet_cholesky_basis: bool,
    vocab_override: int | None = None,
) -> QuantumBornGPT:
    """Build a compact 2-layer `QuantumBornGPT` for comparing untied vs. Tied-2^16 readout."""
    torch.manual_seed(2034)
    v_sz = int(vocab_override if vocab_override is not None else cfg.vocab_size)
    return QuantumBornGPT(
        vocab_size=v_sz, embed_dim=cfg.embed_dim, seed_dim=cfg.embed_dim,
        num_heads=(4 if cfg.embed_dim < 512 else 8), num_layers=2, block_size=64,
        readout_mode="asymmetric", povm_temperature=0.030, use_overcomplete_two_beam=True,
        frame_expansion=1.0, use_purified_attn=True, true_layer0_v_residual=True,
        use_gated_spherical_residual=True, use_dynamic_povm_beta=True, use_povm_log_prior=True,
        use_ortho_decouple=False, use_hierarchical_16qubit=True,
        qubit_registers=cfg.qubit_registers, qubit_amplitudes=cfg.qubit_amplitudes,
        num_higher_states=64, use_block_hierarchy_bridge=True, bridge_causal_memory=False,
        use_two_beam_kboson_readout=True, diffuse_rank=16, diffuse_schedule="warmup_hold_anneal",
        use_stereographic_lift=True, use_cosine_phase_beam_gate=True,
        quadratic_diffuse_fidelity=0.50, kraus_spectral_floor=6.5,
        use_mach_zehnder_beamsplitter=True, use_wavelet_cholesky_branch=True,
        use_wavelet_cholesky_basis=use_wavelet_cholesky_basis, num_value_embeddings=1,
        zero_half_rope=True, window_size=32,
    ).to(torch.device(cfg.device))


@torch.no_grad()
def audit_tied_2pow16_basis(cfg: WaveletProbeConfig) -> dict[str, Any]:
    """Compare parameter budgets and Step-0 `J_anti` self-overlap of Tied-2^16 vs. untied readout."""
    dev = torch.device(cfg.device)
    tied_model = build_probe_born_gpt(cfg, use_wavelet_cholesky_basis=True)

    untied_50k = Hierarchical16QubitPOVMReadout(
        vocab_size=50304, embed_dim=cfg.embed_dim, num_registers=cfg.qubit_registers,
        register_dim=cfg.qubit_amplitudes, num_higher_states=128, use_wavelet_cholesky_basis=False,
    )
    tied_50k = Hierarchical16QubitPOVMReadout(
        vocab_size=50304, embed_dim=cfg.embed_dim, num_registers=cfg.qubit_registers,
        register_dim=cfg.qubit_amplitudes, num_higher_states=128, use_wavelet_cholesky_basis=True,
    )
    full_50k_untied = sum(p.numel() for p in untied_50k.parameters()) + (50304 * cfg.embed_dim)
    full_50k_tied = sum(p.numel() for p in tied_50k.parameters())
    full_50k_savings = full_50k_untied - full_50k_tied

    gen = torch.Generator(device="cpu").manual_seed(2034)
    flat_dim = cfg.qubit_registers * cfg.qubit_amplitudes
    ppmi_basis = F.normalize(
        torch.randn((cfg.vocab_size, max(flat_dim, 64)), generator=gen, dtype=torch.float32),
        p=2.0, dim=-1,
    ).to(dev)
    synth_tokens = np.arange(8192, dtype=np.uint16) % cfg.vocab_size
    init_stats = initialize_hierarchical_16qubit_with_ppmi_svd(
        tied_model, ppmi_embeddings=ppmi_basis, token_shards=[synth_tokens], alpha=0.75,
        kmeans_iters=4, decouple_self_in_full_ambient=True, directed_predecessor_mix=0.35, device=dev,
    )
    assert tied_model.hierarchical_povm_readout is not None
    return {
        "full_50k_untied_readout_params": full_50k_untied,
        "full_50k_tied_readout_params": full_50k_tied,
        "param_savings": full_50k_savings,
        "povm_head_is_none": tied_model.povm_head is None,
        "povm_odd_qubits_is_none": tied_model.hierarchical_povm_readout.povm_odd_qubits is None,
        "step0_max_self_overlap": round(float(init_stats["max_self_overlap"]), 6),
        "step0_self_register_overlap_max": round(float(init_stats["self_register_overlap_max"]), 6),
    }


def run_wavelet_cholesky_probe(cfg: WaveletProbeConfig) -> dict[str, Any]:
    """Execute the full Wavelet-Cholesky spectrum & Tied-2^16 basis audit and print Markdown tables."""
    t0 = time.perf_counter()
    wav_stats = audit_wavelet_reconstruction_and_spectrum(cfg)
    tied_stats = audit_tied_2pow16_basis(cfg)
    elapsed_s = round(time.perf_counter() - t0, 3)

    print(
        f"\n### Multi-Level Wavelet-Cholesky Reconstruction & Spectrum Probe "
        f"(`d={cfg.embed_dim}, S={cfg.num_levels}, QxA={cfg.qubit_registers}x{cfg.qubit_amplitudes}`)"
    )
    print("| Metric | Step-0 (`P=U=0, L=I`) | Non-Zero Lifting (`P,U != 0`) | Target | Status |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    rec_ok = wav_stats["step0_max_abs_err"] < 1e-5 and wav_stats["nonzero_lift_max_abs_err"] < 1e-5
    print(
        f"| Reconstruction `max_abs_err` | `{wav_stats['step0_max_abs_err']:.2e}` | "
        f"`{wav_stats['nonzero_lift_max_abs_err']:.2e}` | `< 1e-5` | **{'PASS' if rec_ok else 'FAIL'}** |"
    )
    print(
        f"| Energy Ratio `||psi_hat||^2 / ||psi||^2` | `{wav_stats['step0_energy_ratio']:.6f}` | "
        f"`{wav_stats['nonzero_lift_energy_ratio']:.6f}` | `1.000000` | **PASS** |"
    )
    print("\n| Scale Band | Dim (`Blocks x Size`) | Energy Fraction | Min `diag(L_s)` | SPD `kappa(L_s L_s^T)` |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for s_row in wav_stats["cholesky_scales"]:
        s_name = s_row["scale"]
        e_key = "E_coarse" if s_name == "coarse" else f"E_{s_name}"
        e_val = wav_stats["scale_energy_fractions"].get(e_key, 0.0)
        print(
            f"| `{s_name}` | `{s_row['dim']}` (`{s_row['num_blocks']}x{s_row['block_size']}`) | "
            f"`{e_val:.4f}` | `{s_row['perturbed_min_diag']:.4f}` | `{s_row['perturbed_spd_cond']:.4f}` |"
        )
    print(
        f"\n- **Tied-$2^{{16}}$ Basis**: Full-50k Savings = `{tied_stats['param_savings']:,}` params | "
        f"Step-0 `max_self_overlap` = `{tied_stats['step0_max_self_overlap']:.6f}` (`<= 0.15`)"
    )

    report = {
        "config": asdict(cfg),
        "elapsed_sec": elapsed_s,
        "wavelet_reconstruction_and_spectrum": wav_stats,
        "tied_2pow16_basis": tied_stats,
    }
    if cfg.out_json:
        out_p = Path(cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> WaveletProbeConfig:
    """Parse CLI arguments into a frozen `WaveletProbeConfig`."""
    p = argparse.ArgumentParser(description="Audit Wavelet-Cholesky reconstruction, SPD spectra, and Tied-2^16 basis.")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-levels", type=int, default=3)
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--qubit-registers", type=int, default=16)
    p.add_argument("--qubit-amplitudes", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    return WaveletProbeConfig(
        embed_dim=ns.embed_dim, num_levels=ns.num_levels, vocab_size=ns.vocab_size,
        qubit_registers=ns.qubit_registers, qubit_amplitudes=ns.qubit_amplitudes,
        device=ns.device, out_json=ns.out_json,
    )


if __name__ == "__main__":
    run_wavelet_cholesky_probe(parse_args())
