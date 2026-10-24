#!/usr/bin/env python3
"""MERA Causal Entanglement Light-Cone & Coarse<->Fine Kronecker Coupling Visualizer (Plan 035).

Promoted from `scripts/BACKLOG.md` (identified in Plan 034, implemented in Plan 035).
Audits on CPU for both `arch_mode="multiscale_unet"` (`WaveletCholeskyUNetBlock`) and
`arch_mode="consistent_kronecker"` (`ConsistentKroneckerWaveletBlock`):
1. Causal light-cone sensitivity: perturbs token amplitude at `t_0 = 8` and verifies
   `||Delta psi(t)||_2 == 0.0` for `t < t_0` vs. `> 0.0` for `t >= t_0` across layers and wavelet scales.
2. Coarse<->Fine Kronecker entanglement: measures cross-covariance and mutual coupling between
   Coarse-8-Bit (`:d/2`) and Fine-8-Bit (`d/2:`) registers in `ConsistentKroneckerWaveletBlock`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p_dir in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p_dir not in sys.path:
        sys.path.insert(0, p_dir)

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.isometric import UnitaryRoPE
from qdlblocks.layers.wavelet_cholesky import WaveletCholeskyConfig, WaveletCholeskyMultiscaleBranch
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    ConsistentKroneckerWaveletBlock,
    Radical2Pow16Config,
    WaveletCholeskyUNetBlock,
)


@dataclass(frozen=True)
class EntanglementConeConfig:
    """Configuration for the MERA causal entanglement cone and Kronecker subband coupling audit."""

    embed_dim: int = 64
    seq_len: int = 32
    num_layers: int = 2
    device: str = "cpu"
    out_json: str = ""


@torch.no_grad()
def audit_causal_light_cone(cfg: EntanglementConeConfig, arch_mode: str) -> dict[str, Any]:
    """Perturb token amplitude at `t0 = 8` and measure layerwise/scalewise light-cone propagation."""
    dev, d, seq_t = torch.device(cfg.device), cfg.embed_dim, cfg.seq_len
    t0 = min(8, max(1, seq_t // 4))
    torch.manual_seed(2035)
    r_cfg = Radical2Pow16Config(
        arch_mode=arch_mode, vocab_size=1024, valid_vocab_size=1000, ambient_n=1024,
        num_layers=cfg.num_layers, embed_dim=d, num_heads=4, block_size=max(64, seq_t), window_size=max(32, seq_t),
    )
    block_cls = WaveletCholeskyUNetBlock if arch_mode == "multiscale_unet" else ConsistentKroneckerWaveletBlock
    blocks = torch.nn.ModuleList([block_cls(r_cfg, i) for i in range(cfg.num_layers)]).to(dev)
    rope = UnitaryRoPE(head_dim=d // 4, max_seq_len=max(64, seq_t * 2), zero_half_freq=True).to(dev)
    wav_probe = WaveletCholeskyMultiscaleBranch(WaveletCholeskyConfig(embed_dim=d, num_levels=3, max_block_size=16)).to(dev)

    gen = torch.Generator(device="cpu").manual_seed(4035)
    psi_base = l2_normalize(torch.randn((2, seq_t, d), generator=gen).to(dev), dim=-1)
    psi_pert = psi_base.clone()
    delta_vec = l2_normalize(torch.randn((2, d), generator=gen).to(dev), dim=-1)
    psi_pert[:, t0, :] = l2_normalize(psi_pert[:, t0, :] + 0.85 * delta_vec, dim=-1)

    cos, sin = rope.get_cos_sin(seq_t)
    v0_b, v0_p = None, None
    cur_b, cur_p = psi_base, psi_pert
    layer_profiles: list[dict[str, Any]] = []

    for l_idx, blk in enumerate(blocks):
        cur_b, _, v0_b = blk(cur_b, cos=cos, sin=sin, v_0=v0_b, doc_ids=None)
        cur_p, _, v0_p = blk(cur_p, cos=cos, sin=sin, v_0=v0_p, doc_ids=None)
        diff = cur_p - cur_b
        norms_t = diff.norm(dim=-1).mean(dim=0)
        past_max = float(norms_t[:t0].max().item()) if t0 > 0 else 0.0
        at_t0 = float(norms_t[t0].item())
        future_mean = float(norms_t[t0 + 1 :].mean().item()) if (t0 + 1) < seq_t else at_t0

        c_d, det_d = wav_probe.downsample_amplitudes(diff[:, t0:, :])
        c_e = float(c_d.pow(2).sum(dim=-1).mean().item())
        d_es = [float(x.pow(2).sum(dim=-1).mean().item()) for x in det_d]
        tot_e = max(c_e + sum(d_es), 1e-12)
        layer_profiles.append({
            "layer": l_idx,
            "past_max_perturbation": past_max,
            "t0_perturbation": round(at_t0, 6),
            "future_mean_perturbation": round(future_mean, 6),
            "scale_energy_fractions": {
                "coarse": round(c_e / tot_e, 5),
                **{f"detail_{s + 1}": round(de / tot_e, 5) for s, de in enumerate(d_es)},
            },
        })

    max_past_all = max(float(lp["past_max_perturbation"]) for lp in layer_profiles)
    min_future_all = min(float(lp["future_mean_perturbation"]) for lp in layer_profiles)
    return {
        "arch_mode": arch_mode,
        "perturbation_t0": t0,
        "max_past_perturbation": max_past_all,
        "strictly_causal_past_zero": bool(max_past_all == 0.0),
        "min_future_perturbation": round(min_future_all, 6),
        "downstream_cone_active": bool(min_future_all > 0.0),
        "layers": layer_profiles,
    }


@torch.no_grad()
def audit_kronecker_coarse_fine_entanglement(cfg: EntanglementConeConfig) -> dict[str, Any]:
    """Measure Coarse-8-Bit (`:d/2`) <-> Fine-8-Bit (`d/2:`) coupling in `ConsistentKroneckerWaveletBlock`."""
    dev, d, seq_t = torch.device(cfg.device), cfg.embed_dim, cfg.seq_len
    half_d = d // 2
    torch.manual_seed(2035)
    r_cfg = Radical2Pow16Config(
        arch_mode="consistent_kronecker", vocab_size=1024, valid_vocab_size=1000, ambient_n=1024,
        num_layers=1, embed_dim=d, num_heads=4, block_size=max(64, seq_t), window_size=max(32, seq_t),
    )
    blk = ConsistentKroneckerWaveletBlock(r_cfg, layer_idx=0).to(dev)
    rope = UnitaryRoPE(head_dim=d // 4, max_seq_len=max(64, seq_t * 2), zero_half_freq=True).to(dev)
    cos, sin = rope.get_cos_sin(seq_t)

    gen = torch.Generator(device="cpu").manual_seed(5035)
    psi_0 = l2_normalize(torch.randn((4, seq_t, d), generator=gen).to(dev), dim=-1)
    _, kron_0, _ = blk(psi_0, cos=cos, sin=sin, v_0=None, doc_ids=None)

    psi_c_pert = psi_0.clone()
    psi_c_pert[..., :half_d] = l2_normalize(psi_c_pert[..., :half_d] + 0.65 * torch.randn((4, seq_t, half_d), generator=gen).to(dev), dim=-1)
    psi_c_pert = l2_normalize(psi_c_pert, dim=-1)
    _, kron_c_pert, _ = blk(psi_c_pert, cos=cos, sin=sin, v_0=None, doc_ids=None)
    coarse_to_fine_resp = float((kron_c_pert[..., half_d:] - kron_0[..., half_d:]).norm(dim=-1).mean().item())

    psi_f_pert = psi_0.clone()
    psi_f_pert[..., half_d:] = l2_normalize(psi_f_pert[..., half_d:] + 0.65 * torch.randn((4, seq_t, half_d), generator=gen).to(dev), dim=-1)
    psi_f_pert = l2_normalize(psi_f_pert, dim=-1)
    _, kron_f_pert, _ = blk(psi_f_pert, cos=cos, sin=sin, v_0=None, doc_ids=None)
    fine_to_coarse_resp = float((kron_f_pert[..., :half_d] - kron_0[..., :half_d]).norm(dim=-1).mean().item())

    flat_c = kron_0[..., :half_d].reshape(-1, half_d)
    flat_f = kron_0[..., half_d:].reshape(-1, half_d)
    flat_c = flat_c - flat_c.mean(dim=0, keepdim=True)
    flat_f = flat_f - flat_f.mean(dim=0, keepdim=True)
    cov_cf = (flat_c.t() @ flat_f) / max(1, flat_c.size(0) - 1)
    cov_cc = (flat_c.t() @ flat_c) / max(1, flat_c.size(0) - 1)
    cov_ff = (flat_f.t() @ flat_f) / max(1, flat_f.size(0) - 1)
    cross_cov_ratio = float(cov_cf.norm().item() / max((cov_cc.norm().item() * cov_ff.norm().item()) ** 0.5, 1e-12))

    c_sub, f_sub = psi_0[..., :half_d], psi_0[..., half_d:]
    th0 = blk.scale_beamsplitter_thetas[0]
    rot_c = torch.cos(th0) * c_sub + torch.sin(th0) * f_sub
    rot_f = -torch.sin(th0) * c_sub + torch.cos(th0) * f_sub
    bs_energy_err = float(((rot_c.pow(2) + rot_f.pow(2)) - (c_sub.pow(2) + f_sub.pow(2))).abs().max().item())
    return {
        "coarse_to_fine_coupling": round(coarse_to_fine_resp, 6),
        "fine_to_coarse_coupling": round(fine_to_coarse_resp, 6),
        "normalized_cross_covariance": round(cross_cov_ratio, 6),
        "beamsplitter_energy_err": bs_energy_err,
        "thetas_rad": [round(float(t.item()), 4) for t in blk.scale_beamsplitter_thetas],
    }


def run_entanglement_cone_visualizer(cfg: EntanglementConeConfig) -> dict[str, Any]:
    """Run causal light-cone and Coarse<->Fine Kronecker entanglement audits and print Markdown summary."""
    t0 = time.perf_counter()
    unet_cone = audit_causal_light_cone(cfg, "multiscale_unet")
    kron_cone = audit_causal_light_cone(cfg, "consistent_kronecker")
    kron_ent = audit_kronecker_coarse_fine_entanglement(cfg)
    report = {
        "config": asdict(cfg),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "causal_light_cones": {"multiscale_unet": unet_cone, "consistent_kronecker": kron_cone},
        "kronecker_coarse_fine_entanglement": kron_ent,
    }
    print(f"\n### MERA Causal Entanglement Light-Cone & Kronecker Coupling (`d={cfg.embed_dim}, T={cfg.seq_len}, L={cfg.num_layers}`)")
    for mode, c_res in report["causal_light_cones"].items():
        print(
            f"- **`{mode}` Light-Cone (`t0={c_res['perturbation_t0']}`)**: "
            f"`max_{{t < t0}} ||Delta psi||_2 = {c_res['max_past_perturbation']:.2e}` (Strictly Causal: **{c_res['strictly_causal_past_zero']}**) | "
            f"`mean_{{t >= t0}} ||Delta psi||_2 = {c_res['min_future_perturbation']:.4f}`"
        )
    print(
        f"- **Coarse<->Fine Kronecker Entanglement**: `C->F = {kron_ent['coarse_to_fine_coupling']:.4f}`, "
        f"`F->C = {kron_ent['fine_to_coarse_coupling']:.4f}`, Cross-Cov Ratio = `{kron_ent['normalized_cross_covariance']:.4f}`, "
        f"Beamsplitter Energy Err = `{kron_ent['beamsplitter_energy_err']:.2e}`"
    )
    if cfg.out_json:
        Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> EntanglementConeConfig:
    """Parse CLI arguments into a frozen `EntanglementConeConfig`."""
    p = argparse.ArgumentParser(description="Visualize MERA causal entanglement light-cones and Kronecker subband coupling.")
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    return EntanglementConeConfig(ns.embed_dim, ns.seq_len, ns.num_layers, ns.device, ns.out_json)


if __name__ == "__main__":
    run_entanglement_cone_visualizer(parse_args())
