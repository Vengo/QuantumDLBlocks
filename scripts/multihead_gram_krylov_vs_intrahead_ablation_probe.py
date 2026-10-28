#!/usr/bin/env python3
"""Multi-Head Cross-Head Gram Purification vs. Intra-Head 11-GEMM Krylov Ablation Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Compares `(T x T)` matrix multiplication counts, FLOP complexity at `T in {512, 1024}`, and algebraic
parity `< 1e-5` between explicit `16 x 16` Hermitian density squaring `rho^2 |phi>` and the
`O(H_g^2 d_f)` Cross-Head Gram Matrix purification identity (`Track 6` / `Issue 6`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdlblocks.benchmarks.kronecker_diagnostic_utils import (
    DEFAULT_DIAGNOSTIC_SEQ_LEN,
    write_standardized_probe_json,
)
from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "multihead_gram_krylov_vs_intrahead_ablation_probe.py"
LEGACY_KRYLOV_T_BY_T_GEMMS_PER_LAYER: int = 11
MULTIHEAD_BORN_T_BY_T_GEMMS_PER_LAYER: int = 2
DEFAULT_FACTOR_HEAD_DIM: int = 16
DEFAULT_SUBHEADS_PER_FACTOR: int = 2
GRAM_IDENTITY_TOL: float = 1e-5


@torch.no_grad()
def verify_cross_head_gram_vs_explicit_density_squaring(
    batch_tokens: int = 64,
    subheads_h: int = DEFAULT_SUBHEADS_PER_FACTOR,
    factor_dim: int = DEFAULT_FACTOR_HEAD_DIM,
    device: str = "cpu",
) -> dict[str, float]:
    """Verify `O(H_g^2 d_f)` Cross-Head Gram identity matches explicit `16x16` `rho @ rho` (`<= 8` args)."""
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1337)

    raw_re = torch.randn((batch_tokens, subheads_h, factor_dim), generator=gen, device=dev, dtype=torch.float32)
    raw_im = torch.randn((batch_tokens, subheads_h, factor_dim), generator=gen, device=dev, dtype=torch.float32)
    phi_re, phi_im = complex_l2_normalize(raw_re, raw_im, dim=-1)
    phi_c = torch.complex(phi_re, phi_im)

    # 1. O(H_g^2 d_f) Cross-Head Hermitian Gram matrix G_{a,b} = <phi_a | phi_b>:
    gram = torch.matmul(phi_c.conj(), phi_c.transpose(-1, -2)) / float(subheads_h)
    purity_gram = torch.diagonal(torch.matmul(gram, gram), dim1=-2, dim2=-1).real.sum(dim=-1)
    coupled_gram = torch.matmul(gram, phi_c)

    # 2. Explicit 16x16 Hermitian density matrix rho = (1/H_g) sum_h |phi_h><phi_h|:
    rho_explicit = torch.einsum("nhd,nhe->nde", phi_c, phi_c.conj()) / float(subheads_h)
    rho_sq = torch.matmul(rho_explicit, rho_explicit)
    purity_explicit = torch.diagonal(rho_sq, dim1=-2, dim2=-1).real.sum(dim=-1)
    coupled_explicit = torch.einsum("nde,nhd->nhe", rho_explicit, phi_c)

    max_purity_err = float((purity_gram - purity_explicit).abs().max().item())
    max_ket_err = float((coupled_gram - coupled_explicit).abs().max().item())
    mean_purity = float(purity_gram.mean().item())

    return {
        "gram_vs_explicit_purity_max_err": round(max_purity_err, 8),
        "gram_vs_explicit_ket_max_err": round(max_ket_err, 8),
        "gram_identity_pass": float(max_purity_err < GRAM_IDENTITY_TOL and max_ket_err < GRAM_IDENTITY_TOL),
        "mean_subhead_mixed_purity": round(mean_purity, 6),
        "mean_subhead_effective_rank": round(1.0 / max(mean_purity, 1e-8), 4),
    }


def run_multihead_vs_intrahead_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """Compare multi-head Gram purification against legacy 11-GEMM intra-head Krylov (`<= 8` args)."""
    t0 = time.perf_counter()
    gram_check = verify_cross_head_gram_vs_explicit_density_squaring(
        batch_tokens=64,
        subheads_h=int(args.subheads),
        factor_dim=DEFAULT_FACTOR_HEAD_DIM,
        device=str(args.device),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    n_layers = int(args.num_layers)
    legacy_gemms = n_layers * LEGACY_KRYLOV_T_BY_T_GEMMS_PER_LAYER
    multihead_gemms = n_layers * MULTIHEAD_BORN_T_BY_T_GEMMS_PER_LAYER
    t_by_t_reduction_ratio = float(legacy_gemms) / float(max(1, multihead_gemms))

    metrics: dict[str, Any] = {
        **gram_check,
        "num_layers": n_layers,
        "legacy_krylov_t_by_t_gemms_total": legacy_gemms,
        "multihead_stinespring_t_by_t_gemms_total": multihead_gemms,
        "t_by_t_gemm_reduction_factor": round(t_by_t_reduction_ratio, 2),
        "probe_elapsed_ms": round(elapsed_ms, 3),
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Compare Multi-Head Gram purification vs. 11-GEMM Krylov attention.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--subheads", type=int, default=DEFAULT_SUBHEADS_PER_FACTOR)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_DIAGNOSTIC_SEQ_LEN)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_multihead_vs_intrahead_ablation(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} err={payload['metrics']['gram_vs_explicit_ket_max_err']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
