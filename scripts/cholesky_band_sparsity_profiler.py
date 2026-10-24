#!/usr/bin/env python3
"""Block-Cholesky Band Sparsity, SPD Conditioning, & Throughput Profiler (Plan 036).

Promoted from `scripts/BACKLOG.md` (identified in Plan 035, implemented in Plan 036).
Audits every `BlockCholeskyLinear` operator in `Radical2Pow16WaveletCholeskyGPT`:
1. Strict diagonal positivity (`min_diag > 0.0`) and SPD condition number `kappa(L_s L_s^T)`.
2. Off-diagonal Frobenius coupling ratio `||L_s - diag(L_s)||_F / ||L_s||_F`.
3. Parameter compression ratio `d^2 / (num_blocks * b^2)` vs. dense `O(d^2)` matrices.
4. Per-module forward latency (`ms / forward`) on CPU/GPU.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.layers.wavelet_cholesky import BlockCholeskyLinear
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
)


@dataclass(frozen=True)
class CholeskyProfilerConfig:
    """Configuration for the BlockCholeskyLinear band-sparsity and SPD conditioning profiler."""

    arch_mode: str = "fused_kronecker_unet"
    embed_dim: int = 256
    num_layers: int = 4
    perturb_std: float = 0.08
    bench_iters: int = 5
    checkpoint_path: str = ""
    device: str = "cpu"
    out_json: str = ""


@torch.no_grad()
def apply_cholesky_perturbation(model: Radical2Pow16WaveletCholeskyGPT, perturb_std: float) -> None:
    """Apply deterministic Gaussian perturbation to `raw_tril` and `raw_diag` when `perturb_std > 0`."""
    if perturb_std <= 0.0:
        return
    gen = torch.Generator(device="cpu").manual_seed(3036)
    for mod in model.modules():
        if isinstance(mod, BlockCholeskyLinear):
            dev = mod.raw_tril.device
            tril_noise = torch.randn(mod.raw_tril.shape, generator=gen, dtype=torch.float32).to(dev)
            diag_noise = torch.randn(mod.raw_diag.shape, generator=gen, dtype=torch.float32).to(dev)
            mod.raw_tril.add_(tril_noise * float(perturb_std))
            mod.raw_diag.add_(diag_noise * float(perturb_std * 0.5))


@torch.no_grad()
def profile_single_cholesky_module(
    name: str,
    mod: BlockCholeskyLinear,
    bench_iters: int,
    device: torch.device,
) -> dict[str, Any]:
    """Profile condition number, off-diagonal Frobenius ratio, compression, and latency of one `BlockCholeskyLinear`."""
    cond = mod.condition_metrics()
    l_mat = mod.cholesky_factor()
    off_diag = torch.tril(l_mat, diagonal=-1)
    total_frob = max(float(l_mat.norm().item()), 1e-12)
    off_diag_ratio = float(off_diag.norm().item()) / total_frob

    d, nb, b = int(mod.dim), int(mod.num_blocks), int(mod.block_size)
    block_params = nb * b * b
    tril_active_params = nb * (b * (b + 1) // 2)
    dense_params = d * d
    compression_ratio = float(dense_params) / float(max(1, block_params))
    tril_compression_ratio = float(dense_params) / float(max(1, tril_active_params))

    n_iters = max(1, int(bench_iters))
    x_probe = torch.randn((2, 64, d), device=device, dtype=torch.float32)
    _ = mod(x_probe)
    t_start = time.perf_counter()
    for _ in range(n_iters):
        _ = mod(x_probe)
    forward_ms = (time.perf_counter() - t_start) * 1000.0 / float(n_iters)

    return {
        "name": name,
        "dim": d,
        "num_blocks": nb,
        "block_size": b,
        "min_diag": round(float(cond["min_diag"]), 6),
        "min_chol_sv": round(float(cond["min_chol_sv"]), 6),
        "max_chol_sv": round(float(cond["max_chol_sv"]), 6),
        "spd_cond_number": round(float(cond["spd_cond_number"]), 6),
        "off_diag_frobenius_ratio": round(off_diag_ratio, 6),
        "compression_vs_dense_ratio": round(compression_ratio, 4),
        "tril_compression_vs_dense_ratio": round(tril_compression_ratio, 4),
        "block_params": block_params,
        "dense_equivalent_params": dense_params,
        "forward_ms": round(forward_ms, 5),
    }


@torch.no_grad()
def run_cholesky_profiler(cfg: CholeskyProfilerConfig) -> dict[str, Any]:
    """Instantiate `Radical2Pow16WaveletCholeskyGPT`, profile all `BlockCholeskyLinear` modules, and report."""
    t0 = time.perf_counter()
    dev = torch.device(cfg.device)
    d = cfg.embed_dim
    r_cfg = Radical2Pow16Config(
        arch_mode=cfg.arch_mode,  # type: ignore[arg-type]
        embed_dim=d,
        num_layers=cfg.num_layers,
        num_heads=max(1, min(4, d // 16)),
        mid_detail_dim=max(32, d // 2),
        fine_detail_dim=max(32, d // 4),
    )
    model = Radical2Pow16WaveletCholeskyGPT(r_cfg).to(dev)
    if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
        state = torch.load(cfg.checkpoint_path, map_location=dev, weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    else:
        apply_cholesky_perturbation(model, cfg.perturb_std)

    modules_rows = [
        profile_single_cholesky_module(name, mod, cfg.bench_iters, dev)
        for name, mod in model.named_modules()
        if isinstance(mod, BlockCholeskyLinear)
    ]

    all_diag_pos = all(float(r["min_diag"]) > 0.0 for r in modules_rows)
    max_cond = max((float(r["spd_cond_number"]) for r in modules_rows), default=1.0)
    mean_cond = sum(float(r["spd_cond_number"]) for r in modules_rows) / max(1, len(modules_rows))
    mean_off_diag = sum(float(r["off_diag_frobenius_ratio"]) for r in modules_rows) / max(1, len(modules_rows))
    tot_chol_p = sum(int(r["block_params"]) for r in modules_rows)
    tot_dense_p = sum(int(r["dense_equivalent_params"]) for r in modules_rows)
    overall_comp = float(tot_dense_p) / float(max(1, tot_chol_p))
    mean_ms = sum(float(r["forward_ms"]) for r in modules_rows) / max(1, len(modules_rows))

    summary = {
        "arch_mode": cfg.arch_mode,
        "total_model_params": sum(p.numel() for p in model.parameters()),
        "num_cholesky_modules": len(modules_rows),
        "all_diag_positive": bool(all_diag_pos),
        "max_spd_cond_number": round(max_cond, 6),
        "mean_spd_cond_number": round(mean_cond, 6),
        "mean_off_diag_frobenius_ratio": round(mean_off_diag, 6),
        "total_cholesky_block_params": tot_chol_p,
        "total_dense_equivalent_params": tot_dense_p,
        "overall_compression_vs_dense": round(overall_comp, 4),
        "mean_forward_ms": round(mean_ms, 5),
    }
    report = {
        "config": asdict(cfg),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "summary": summary,
        "modules": modules_rows,
    }

    print(
        f"\n### Block-Cholesky Band Sparsity & Conditioning Profile "
        f"(`arch={cfg.arch_mode}, d={d}, L={cfg.num_layers}, modules={len(modules_rows)}`)"
    )
    print(
        f"- **All Diag > 0**: `{all_diag_pos}` | **Max SPD Cond**: `{max_cond:.3f}` | "
        f"**Mean Off-Diag Frobenius**: `{mean_off_diag:.4f}` | **Overall Compression**: `{overall_comp:.2f}x`"
    )
    print("| Module | Dim (`nb x b`) | `min_diag` | `kappa(L L^T)` | Off-Diag Frob | Compression | Latency (`ms`) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in modules_rows[:12]:
        print(
            f"| `{r['name']}` | `{r['dim']}` (`{r['num_blocks']}x{r['block_size']}`) | "
            f"`{r['min_diag']:.4f}` | `{r['spd_cond_number']:.3f}` | "
            f"`{r['off_diag_frobenius_ratio']:.4f}` | `{r['compression_vs_dense_ratio']:.2f}x` | "
            f"`{r['forward_ms']:.4f}` |"
        )
    if cfg.out_json:
        out_p = Path(cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> CholeskyProfilerConfig:
    """Parse CLI arguments into a frozen `CholeskyProfilerConfig`."""
    p = argparse.ArgumentParser(description="Profile BlockCholeskyLinear band sparsity, conditioning, and latency.")
    p.add_argument("--arch-mode", type=str, default="fused_kronecker_unet")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--perturb-std", type=float, default=0.08)
    p.add_argument("--bench-iters", type=int, default=5)
    p.add_argument("--checkpoint-path", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    return CholeskyProfilerConfig(
        arch_mode=ns.arch_mode,
        embed_dim=ns.embed_dim,
        num_layers=ns.num_layers,
        perturb_std=ns.perturb_std,
        bench_iters=ns.bench_iters,
        checkpoint_path=ns.checkpoint_path,
        device=ns.device,
        out_json=ns.out_json,
    )


if __name__ == "__main__":
    run_cholesky_profiler(parse_args())
