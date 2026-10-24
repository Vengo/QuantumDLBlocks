#!/usr/bin/env python3
"""16-Bit Dyadic Spectral Tree, Parameter Compression, & Dark Ancilla Utilization Audit (Plan 035)."""

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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.functional import l2_normalize
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    Tied2Pow16WaveletCholeskyPyramid,
    build_dyadic_2pow16_spectral_tree,
)


@dataclass(frozen=True)
class DarkAncillaAuditConfig:
    """Configuration for the 16-bit dyadic spectral tree and dark ancilla utilization audit."""

    vocab_size: int = 2048
    ambient_n: int = 4096
    embed_dim: int = 64
    device: str = "cpu"
    out_json: str = ""


def _mean_intra_block_cosine(vecs: torch.Tensor, block_size: int) -> float:
    """Compute mean pairwise cosine similarity within contiguous blocks of `block_size`."""
    n_total, d = vecs.shape
    num_blocks = n_total // block_size
    blocks = l2_normalize(vecs[: num_blocks * block_size].view(num_blocks, block_size, d), dim=-1)
    gram = torch.bmm(blocks, blocks.transpose(1, 2))
    eye = torch.eye(block_size, device=vecs.device, dtype=vecs.dtype).unsqueeze(0)
    return float(((gram * (1.0 - eye)).sum(dim=(1, 2)) / max(1, block_size * (block_size - 1))).mean().item())


@torch.no_grad()
def audit_dyadic_spectral_tree(cfg: DarkAncillaAuditConfig) -> dict[str, Any]:
    """Audit bijection validity and intra-block cosine coherence of `build_dyadic_2pow16_spectral_tree`."""
    dev, gen = torch.device(cfg.device), torch.Generator(device="cpu").manual_seed(2035)
    svd_envelope = torch.arange(1, cfg.embed_dim + 1, dtype=torch.float32) ** -0.85
    centers = F.normalize(torch.randn((16, cfg.embed_dim), generator=gen) * svd_envelope, p=2.0, dim=-1)
    assign = torch.arange(cfg.vocab_size) % 16
    ppmi = F.normalize(centers[assign] + 0.08 * torch.randn((cfg.vocab_size, cfg.embed_dim), generator=gen) * svd_envelope, p=2.0, dim=-1).to(dev)
    perm_v_to_leaf, leaf_to_v = build_dyadic_2pow16_spectral_tree(ppmi, ambient_n=cfg.ambient_n)
    arange_n = torch.arange(cfg.ambient_n, device=dev, dtype=torch.long)
    is_bij = bool(perm_v_to_leaf.shape == (cfg.ambient_n,) and torch.equal(leaf_to_v[perm_v_to_leaf], arange_n) and torch.equal(perm_v_to_leaf[leaf_to_v], arange_n))
    ordered_lex = ppmi.index_select(0, leaf_to_v[leaf_to_v < cfg.vocab_size])
    raw_8, spec_8 = _mean_intra_block_cosine(ppmi, 8), _mean_intra_block_cosine(ordered_lex, 8)
    raw_64, spec_64 = _mean_intra_block_cosine(ppmi, 64), _mean_intra_block_cosine(ordered_lex, 64)
    return {
        "is_bijection": is_bij, "raw_block8_cos": round(raw_8, 6), "spectral_block8_cos": round(spec_8, 6),
        "block8_gain_ratio": round(spec_8 / max(abs(raw_8), 1e-4), 4), "raw_block64_cos": round(raw_64, 6),
        "spectral_block64_cos": round(spec_64, 6), "block64_gain_ratio": round(spec_64 / max(abs(raw_64), 1e-4), 4),
        "ppmi_embeddings": ppmi, "perm_v_to_leaf": perm_v_to_leaf, "leaf_to_v": leaf_to_v,
    }


@torch.no_grad()
def audit_parameter_compression() -> dict[str, Any]:
    """Audit parameter breakdown of `Radical2Pow16WaveletCholeskyGPT` at full `V=50304, N=65536, d=256`."""
    results: dict[str, Any] = {}
    for mode in ("multiscale_unet", "consistent_kronecker"):
        model = Radical2Pow16WaveletCholeskyGPT(Radical2Pow16Config(arch_mode=mode))
        groups = {"pyramid_wavelet": 0, "pyramid_cholesky_kraus": 0, "backbone_blocks": 0, "readout_gates": 0}
        for name, p in model.named_parameters():
            k = "pyramid_wavelet" if name.startswith(("pyramid.wav_", "pyramid.up_")) else ("pyramid_cholesky_kraus" if name.startswith("pyramid.") else ("backbone_blocks" if name.startswith("blocks.") else "readout_gates"))
            groups[k] += p.numel()
        total = sum(groups.values())
        results[mode] = {
            "total_params": total, "under_11m_target": bool(total < 11_000_000),
            "compression_vs_born_gpt_40_8m": round(40_780_000 / max(1, total), 3),
            "compression_vs_classical_30_9m": round(30_890_000 / max(1, total), 3), "groups": groups,
        }
    return results


@torch.no_grad()
def audit_dark_ancilla_utilization(
    cfg: DarkAncillaAuditConfig, ppmi: torch.Tensor, perm_v_to_leaf: torch.Tensor, leaf_to_v: torch.Tensor
) -> dict[str, Any]:
    """Measure norm and Cholesky coupling energy of dark ancilla leaves vs. active lexical leaves."""
    dev, d = torch.device(cfg.device), cfg.embed_dim
    d_mid, d_fine = max(16, d // 2), max(8, d // 4)
    r_cfg = Radical2Pow16Config(
        vocab_size=cfg.ambient_n, valid_vocab_size=cfg.vocab_size, ambient_n=cfg.ambient_n,
        num_layers=1, embed_dim=d, mid_detail_dim=d_mid, fine_detail_dim=d_fine,
    )
    pyr = Tied2Pow16WaveletCholeskyPyramid(r_cfg).to(dev)
    pyr.perm_v_to_leaf.copy_(perm_v_to_leaf)
    pyr.leaf_to_v.copy_(leaf_to_v)
    gen = torch.Generator(device="cpu").manual_seed(3035)
    target_v = torch.zeros((cfg.ambient_n, d), device=dev, dtype=torch.float32)
    target_v[: cfg.vocab_size] = ppmi
    dark_count = cfg.ambient_n - cfg.vocab_size
    if dark_count > 0:
        target_v[cfg.vocab_size :] = 0.22 * F.normalize(torch.randn((dark_count, d), generator=gen).to(dev), p=2.0, dim=-1)
    y_leaf = target_v.index_select(0, leaf_to_v)
    n_c, n_2k, n_8k = max(1, cfg.ambient_n >> 8), max(1, cfg.ambient_n >> 5), max(1, cfg.ambient_n >> 3)
    w_c = y_leaf.view(n_c, 256, d).mean(dim=1)
    r_2k = y_leaf.view(n_2k, 32, d).mean(dim=1) - w_c.repeat_interleave(8, dim=0)
    r_8k = y_leaf.view(n_8k, 8, d).mean(dim=1) - (w_c.repeat_interleave(32, dim=0) + r_2k.repeat_interleave(4, dim=0))
    pyr.wav_256[:n_c].copy_(w_c)
    pyr.wav_2048[:n_2k].copy_(r_2k)
    pyr.wav_8192[:n_8k].copy_(r_8k[:, :d_mid])
    pyr.wav_65536[: cfg.ambient_n].copy_((y_leaf - (w_c.repeat_interleave(256, dim=0) + r_2k.repeat_interleave(32, dim=0)))[:, :d_fine])
    p, g = pyr.perm_v_to_leaf, F.softplus(pyr.scale_gains)
    base_256 = pyr.chol_256(pyr.wav_256)
    trans_256 = base_256 + torch.tanh(pyr.transition_gain) * (pyr.coarse_transition @ base_256)
    raw_synth = (
        g[0] * trans_256.index_select(0, p >> 8) + g[1] * pyr.chol_2048(pyr.wav_2048).index_select(0, p >> 5)
        + g[2] * F.linear(pyr.chol_8192(pyr.wav_8192), pyr.up_8192.t()).index_select(0, p >> 3)
        + g[3] * F.linear(pyr.chol_65536(pyr.wav_65536), pyr.up_65536.t()).index_select(0, p)
    )
    bridged = pyr.readout_chol_bridge(pyr.apply_symplectic_swap(raw_synth))
    norms, chol_e = raw_synth.norm(dim=-1), bridged.pow(2).sum(dim=-1)
    lex_e, dark_e = chol_e[: cfg.vocab_size], chol_e[cfg.vocab_size :]
    return {
        "lexical_count": cfg.vocab_size, "dark_ancilla_count": dark_count, "full_scale_dark_ancilla_count": 65536 - 50257,
        "lexical_mean_norm": round(float(norms[: cfg.vocab_size].mean().item()), 6),
        "dark_ancilla_mean_norm": round(float(norms[cfg.vocab_size :].mean().item()), 6),
        "lexical_mean_cholesky_energy": round(float(lex_e.mean().item()), 6),
        "dark_ancilla_mean_cholesky_energy": round(float(dark_e.mean().item()), 6),
        "dark_ancilla_energy_fraction": round(float(dark_e.sum().item()) / max(float(chol_e.sum().item()), 1e-12), 6),
    }


def run_dark_ancilla_audit(cfg: DarkAncillaAuditConfig) -> dict[str, Any]:
    """Execute the full spectral tree, parameter compression, and dark ancilla audit."""
    t0 = time.perf_counter()
    tree_res, param_res = audit_dyadic_spectral_tree(cfg), audit_parameter_compression()
    dark_res = audit_dark_ancilla_utilization(cfg, tree_res.pop("ppmi_embeddings"), tree_res.pop("perm_v_to_leaf"), tree_res.pop("leaf_to_v"))
    report = {"config": asdict(cfg), "elapsed_sec": round(time.perf_counter() - t0, 3), "dyadic_spectral_tree": tree_res, "parameter_compression": param_res, "dark_ancilla_utilization": dark_res}
    print(f"\n### 16-Bit Dyadic Spectral Tree & Dark Ancilla Audit (`V={cfg.vocab_size}, N={cfg.ambient_n}, d={cfg.embed_dim}`)")
    print(f"- **Bijection**: `{tree_res['is_bijection']}` | `8`-Leaf Gain = `{tree_res['block8_gain_ratio']:.2f}x` | `64`-Leaf Gain = `{tree_res['block64_gain_ratio']:.2f}x`")
    for mode, m_info in param_res.items():
        print(f"- **`{mode}`**: `{m_info['total_params']:,}` params (`< 11M`: **{m_info['under_11m_target']}**, `{m_info['compression_vs_born_gpt_40_8m']:.2f}x` vs BornGPT)")
    print(f"- **Dark Ancilla**: Lexical Norm = `{dark_res['lexical_mean_norm']:.4f}`, Dark Norm = `{dark_res['dark_ancilla_mean_norm']:.4f}`, Energy Fraction = `{dark_res['dark_ancilla_energy_fraction']:.4f}`")
    if cfg.out_json:
        Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> DarkAncillaAuditConfig:
    """Parse CLI arguments into a frozen `DarkAncillaAuditConfig`."""
    p = argparse.ArgumentParser(description="Audit 16-bit dyadic spectral tree, model compression, and dark ancilla modes.")
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--ambient-n", type=int, default=4096)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    return DarkAncillaAuditConfig(ns.vocab_size, ns.ambient_n, ns.embed_dim, ns.device, ns.out_json)


if __name__ == "__main__":
    run_dark_ancilla_audit(parse_args())
