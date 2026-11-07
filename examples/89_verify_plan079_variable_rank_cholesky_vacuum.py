#!/usr/bin/env python3
r"""Verification example for `EXPERIMENT_PLAN_079.md` (Variable-Rank $R$-Mode Uncollapsed Cholesky & Vacuum Subspace)."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.variable_rank_cholesky_vacuum_lab import (
    VariableRankCholeskySpec,
    VariableRankVacuumCholeskyGPT,
    VariableRankVariantMode,
    build_layer0_variable_rank_cholesky_factor,
    compute_rank_adaptive_cholesky_rms_scale,
    compute_rank_invariant_excited_trace_scale,
)


def main() -> int:
    """Run quick smoke verification of `VariableRankVacuumCholeskyGPT` across R in {1, 2, 4, 6, 8, 12}."""
    torch.manual_seed(1337)
    raw_wp = torch.randn(2, 8, 8, 2, 32)
    wp_re = torch.full((8,), 0.20)
    wp_im = torch.zeros(8)

    for r_modes in (1, 2, 4, 6, 8, 12, 16):
        scales = torch.zeros(max(1, r_modes - 1))
        l0_re, l0_im = build_layer0_variable_rank_cholesky_factor(raw_wp, wp_re, wp_im, scales, num_modes=r_modes)
        frob = (l0_re.square().sum(dim=(-2, -1)) + l0_im.square().sum(dim=(-2, -1))).sqrt()
        assert l0_re.shape == (2, 8, 8, r_modes, 32)
        assert torch.allclose(frob, torch.ones_like(frob), atol=1e-4)

    spec_r8 = VariableRankCholeskySpec(
        variant_mode=VariableRankVariantMode.R4_RANK8_VACUUM_CHOLESKY,
        num_cholesky_modes=8,
        vacuum_dim=4,
    )
    model = VariableRankVacuumCholeskyGPT(spec_r8)
    idx = torch.randint(0, 1024, (2, 16))
    targets = torch.randint(0, 1024, (2, 16))
    loss = model.forward_loss(idx, targets)
    loss.backward()
    print(
        f"[Plan 079 Smoke] R=8 loss={loss.item():.4f}, "
        f"alpha_chol(8)={compute_rank_adaptive_cholesky_rms_scale(8):.3f}, "
        f"gamma_trace(8)={compute_rank_invariant_excited_trace_scale(8):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
