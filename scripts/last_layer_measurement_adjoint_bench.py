#!/usr/bin/env python3
"""Last-Layer Measurement Operator Adjoint Benchmark (Plan 039).

Promoted from `scripts/BACKLOG.md` (identified in Plan 038, implemented in Plan 039).
Benchmarks:
1. Right-hand Class/Vocab Detector Synthesis `m_v = M(psi_v^(0))` on `(K, d)` vs.
   Left-hand Last-Layer Measurement Operator `psi_tilde_t = M^*(psi_t)` on `(B * T, d)`
   across `K in {100, 1000, 50304}`.
2. Exact bilinear adjoint equivalence `|<psi_t | O psi_v^(0)> - <O^T psi_t | psi_v^(0)>| < 1e-5`,
   exact input-state self-loop projection `<psi_tilde_t | psi_{x_t}^(0)> <= 1e-6`,
   analytical FLOP ratio, memory traffic ratio, and wall-clock speedup.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.layers.wavelet_cholesky import BlockCholeskyLinear

DEFAULT_NUM_CLASSES: tuple[int, ...] = (100, 1000, 50304)


@dataclass(frozen=True)
class LastLayerAdjointBenchConfig:
    """Configuration for benchmarking Last-Layer Measurement Operator vs. Class Detector Synthesis."""

    batch_tokens: int = 2048
    embed_dim: int = 128
    num_classes: tuple[int, ...] = DEFAULT_NUM_CLASSES
    kraus_rank: int = 8
    max_block_size: int = 32
    warmup_iters: int = 2
    bench_iters: int = 8
    seed: int = 2039
    device: str = "cpu"
    json_out: str = ""


def build_symplectic_swap_matrix(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construct the exact `(dim, dim)` skew-involutive symplectic swap matrix `J_anti`."""
    if dim % 2 != 0:
        raise ValueError(f"embed_dim={dim} must be even for symplectic swap J_anti")
    j_mat = torch.zeros((dim, dim), device=device, dtype=dtype)
    even_idx = torch.arange(0, dim, 2, device=device)
    odd_idx = even_idx + 1
    j_mat[even_idx, odd_idx] = 1.0
    j_mat[odd_idx, even_idx] = -1.0
    return j_mat


@torch.no_grad()
def build_measurement_operators(
    cfg: LastLayerAdjointBenchConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Construct `(O_sharp, O_diff, condition_metrics)` in `R^{d x d}` from Cholesky bridge + Kraus bath."""
    d = int(cfg.embed_dim)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    chol_bridge = BlockCholeskyLinear(
        dim=d,
        max_block_size=min(int(cfg.max_block_size), d),
        fast_spd_bmm=True,
    ).to(device)
    chol_bridge.raw_tril.copy_(torch.randn(chol_bridge.raw_tril.shape, generator=gen).to(device) * 0.14)
    chol_bridge.raw_diag.copy_(torch.randn(chol_bridge.raw_diag.shape, generator=gen).to(device) * 0.10)

    c_blocks = chol_bridge.spd_matrix().float()
    c_full = torch.block_diag(*[c_blocks[b] for b in range(c_blocks.size(0))])
    j_anti = build_symplectic_swap_matrix(d, device=device, dtype=torch.float32)
    o_sharp = c_full @ j_anti

    raw_k = torch.randn((int(cfg.kraus_rank), d, d), generator=gen).to(device)
    u_k, _, vh_k = torch.linalg.svd(raw_k, full_matrices=False)
    kraus_gens = torch.bmm(u_k, vh_k)
    w_k = F.softmax(torch.linspace(0.6, -0.6, int(cfg.kraus_rank), device=device), dim=0)
    k_mean = torch.einsum("r,rij->ij", torch.sqrt(w_k + 1e-8), kraus_gens)
    o_diff = k_mean @ o_sharp
    return o_sharp, o_diff, chol_bridge.condition_metrics()


@torch.no_grad()
def run_right_hand_detector_synthesis(
    psi_classes: torch.Tensor,
    o_sharp: torch.Tensor,
    o_diff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize `(K, d)` sharp and diffuse detectors `m_v = O psi_v^(0)` on the right-hand class table."""
    m_sharp_raw = F.linear(psi_classes, o_sharp)
    m_diff_raw = F.linear(psi_classes, o_diff)
    ov_self = (m_sharp_raw * psi_classes).sum(dim=-1, keepdim=True)
    m_sharp_orth = F.normalize(m_sharp_raw - ov_self * psi_classes, p=2.0, dim=-1)
    m_diff_orth = F.normalize(m_diff_raw - (m_diff_raw * psi_classes).sum(dim=-1, keepdim=True) * psi_classes, p=2.0, dim=-1)
    return m_sharp_orth, m_diff_orth


@torch.no_grad()
def run_left_hand_last_layer_measurement(
    psi_tokens: torch.Tensor,
    psi_xt: torch.Tensor,
    o_sharp: torch.Tensor,
    o_diff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply adjoint measurement `O^T` and self-loop projection directly to `(B * T, d)` token states."""
    psi_sharp_raw = F.linear(psi_tokens, o_sharp.t())
    psi_diff_raw = F.linear(psi_tokens, o_diff.t())
    ov_s = (psi_sharp_raw * psi_xt).sum(dim=-1, keepdim=True)
    ov_d = (psi_diff_raw * psi_xt).sum(dim=-1, keepdim=True)
    psi_tilde_sharp = F.normalize(psi_sharp_raw - ov_s * psi_xt, p=2.0, dim=-1)
    psi_tilde_diff = F.normalize(psi_diff_raw - ov_d * psi_xt, p=2.0, dim=-1)
    return psi_tilde_sharp, psi_tilde_diff


@torch.no_grad()
def benchmark_single_class_count(
    num_classes: int,
    cfg: LastLayerAdjointBenchConfig,
    o_sharp: torch.Tensor,
    o_diff: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark right-hand `(K, d)` detector synthesis vs. left-hand `(B * T, d)` measurement at class count `K`."""
    bt = int(cfg.batch_tokens)
    d = int(cfg.embed_dim)
    k_cls = int(num_classes)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + k_cls * 19)

    psi_classes = F.normalize(torch.randn((k_cls, d), generator=gen, dtype=torch.float32).to(device), p=2.0, dim=-1)
    psi_tokens = F.normalize(torch.randn((bt, d), generator=gen, dtype=torch.float32).to(device), p=2.0, dim=-1)
    input_idx = torch.randint(0, k_cls, (bt,), generator=gen, dtype=torch.long).to(device)
    psi_xt = psi_classes.index_select(0, input_idx)

    # 1. Exact bilinear adjoint equivalence check on a probe slice
    n_probe_t = min(bt, 128)
    n_probe_k = min(k_cls, 256)
    p_tok = psi_tokens[:n_probe_t]
    p_cls = psi_classes[:n_probe_k]
    rhs_sharp = p_tok @ F.linear(p_cls, o_sharp).t()
    lhs_sharp = F.linear(p_tok, o_sharp.t()) @ p_cls.t()
    rhs_diff = p_tok @ F.linear(p_cls, o_diff).t()
    lhs_diff = F.linear(p_tok, o_diff.t()) @ p_cls.t()
    adjoint_sharp_diff = float((rhs_sharp - lhs_sharp).abs().max().item())
    adjoint_diff_diff = float((rhs_diff - lhs_diff).abs().max().item())
    adjoint_max_abs_diff = max(adjoint_sharp_diff, adjoint_diff_diff)

    # 2. Input-state self-loop projection verification
    unproj_tok = F.normalize(F.linear(psi_tokens, o_sharp.t()), p=2.0, dim=-1)
    unproj_self_overlap = float((unproj_tok * psi_xt).sum(dim=-1).abs().max().item())
    psi_tilde_s, psi_tilde_d = run_left_hand_last_layer_measurement(psi_tokens, psi_xt, o_sharp, o_diff)
    max_self_loop_sharp = float((psi_tilde_s * psi_xt).sum(dim=-1).abs().max().item())
    max_self_loop_diff = float((psi_tilde_d * psi_xt).sum(dim=-1).abs().max().item())
    max_abs_self_loop_overlap = max(max_self_loop_sharp, max_self_loop_diff)

    # 3. Warmup and timed operator execution (transform stage before inner-product readout)
    for _ in range(max(1, int(cfg.warmup_iters))):
        _ = run_right_hand_detector_synthesis(psi_classes, o_sharp, o_diff)
        _ = run_left_hand_last_layer_measurement(psi_tokens, psi_xt, o_sharp, o_diff)

    t0 = time.perf_counter()
    for _ in range(max(1, int(cfg.bench_iters))):
        _ = run_right_hand_detector_synthesis(psi_classes, o_sharp, o_diff)
    right_hand_ms = ((time.perf_counter() - t0) * 1000.0) / float(max(1, int(cfg.bench_iters)))

    t1 = time.perf_counter()
    for _ in range(max(1, int(cfg.bench_iters))):
        _ = run_left_hand_last_layer_measurement(psi_tokens, psi_xt, o_sharp, o_diff)
    left_hand_ms = ((time.perf_counter() - t1) * 1000.0) / float(max(1, int(cfg.bench_iters)))

    # 4. Analytical FLOPs and memory traffic for the measurement transformation
    right_hand_flops = int(4 * k_cls * d * d + 6 * k_cls * d)
    left_hand_flops = int(4 * bt * d * d + 6 * bt * d)
    flop_ratio = float(right_hand_flops) / float(max(1, left_hand_flops))
    right_hand_bytes = int(4 * k_cls * d * 4)
    left_hand_bytes = int(4 * bt * d * 4)
    speedup = right_hand_ms / max(left_hand_ms, 1e-6)

    return {
        "num_classes": k_cls,
        "batch_tokens": bt,
        "embed_dim": d,
        "kraus_rank": int(cfg.kraus_rank),
        "adjoint_max_abs_diff": adjoint_max_abs_diff,
        "adjoint_equivalence_passed": bool(adjoint_max_abs_diff < 1e-5),
        "unprojected_max_self_overlap": round(unproj_self_overlap, 6),
        "max_abs_self_loop_overlap": max_abs_self_loop_overlap,
        "self_loop_zero_passed": bool(max_abs_self_loop_overlap <= 1e-6),
        "right_hand_detector_flops": right_hand_flops,
        "left_hand_measurement_flops": left_hand_flops,
        "measurement_flop_ratio": round(flop_ratio, 4),
        "right_hand_detector_bytes": right_hand_bytes,
        "left_hand_measurement_bytes": left_hand_bytes,
        "right_hand_detector_ms": round(right_hand_ms, 4),
        "left_hand_measurement_ms": round(left_hand_ms, 4),
        "wall_clock_speedup_ratio": round(speedup, 3),
    }


def run_last_layer_adjoint_benchmark(cfg: LastLayerAdjointBenchConfig) -> dict[str, Any]:
    """Run the full Last-Layer Measurement Adjoint benchmark across all configured class counts `K`."""
    device = torch.device(cfg.device)
    o_sharp, o_diff, cond_metrics = build_measurement_operators(cfg, device)
    rows = [
        benchmark_single_class_count(int(k), cfg, o_sharp, o_diff, device)
        for k in cfg.num_classes
    ]
    all_passed = all(
        bool(r["adjoint_equivalence_passed"]) and bool(r["self_loop_zero_passed"])
        for r in rows
    )

    print(
        f"\n### Last-Layer Measurement Operator Adjoint Benchmark "
        f"(`B*T={cfg.batch_tokens}`, `d={cfg.embed_dim}`, `kraus_rank={cfg.kraus_rank}`, `device={cfg.device}`)"
    )
    print(
        "| Classes `K` | Adjoint Max Diff | Self-Loop `|<psi_tilde|psi_x>|` | "
        "RHS Detector FLOPs | LHS Meas FLOPs | FLOP Ratio (`RHS/LHS`) | "
        "RHS Detector (`ms`) | LHS Meas (`ms`) | Wall-Clock Speedup |"
    )
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        print(
            f"| `{r['num_classes']}` | `{r['adjoint_max_abs_diff']:.2e}` | `{r['max_abs_self_loop_overlap']:.2e}` | "
            f"`{r['right_hand_detector_flops']:,}` | `{r['left_hand_measurement_flops']:,}` | "
            f"`{r['measurement_flop_ratio']:.3f}x` | `{r['right_hand_detector_ms']:.3f}` | "
            f"`{r['left_hand_measurement_ms']:.3f}` | **`{r['wall_clock_speedup_ratio']:.2f}x`** |"
        )

    report = {
        "config": asdict(cfg),
        "cholesky_condition_metrics": cond_metrics,
        "all_checks_passed": all_passed,
        "classes": {str(r["num_classes"]): r for r in rows},
        "rows": rows,
    }
    if cfg.json_out:
        out_p = Path(cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> LastLayerAdjointBenchConfig:
    """Parse CLI arguments into a frozen `LastLayerAdjointBenchConfig`."""
    p = argparse.ArgumentParser(
        description="Benchmark Right-Hand Class Detector Synthesis vs. Left-Hand Last-Layer Measurement Operator."
    )
    p.add_argument("--batch-tokens", type=int, default=2048)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-classes", type=str, default="100,1000,50304")
    p.add_argument("--kraus-rank", type=int, default=8)
    p.add_argument("--max-block-size", type=int, default=32)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--bench-iters", type=int, default=8)
    p.add_argument("--seed", type=int, default=2039)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-out", "--out-json", dest="json_out", type=str, default="")
    ns = p.parse_args(argv)
    cls_tuple = tuple(int(x.strip()) for x in ns.num_classes.split(",") if x.strip())
    return LastLayerAdjointBenchConfig(
        batch_tokens=ns.batch_tokens,
        embed_dim=ns.embed_dim,
        num_classes=cls_tuple,
        kraus_rank=ns.kraus_rank,
        max_block_size=ns.max_block_size,
        warmup_iters=ns.warmup_iters,
        bench_iters=ns.bench_iters,
        seed=ns.seed,
        device=ns.device,
        json_out=ns.json_out,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    rep = run_last_layer_adjoint_benchmark(parse_args(argv))
    return 0 if bool(rep["all_checks_passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
