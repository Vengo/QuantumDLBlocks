#!/usr/bin/env python3
r"""Verification Script for `EXPERIMENT_PLAN_080.md`: Quantum No-Click POVM Attention Vacuum Sink, Per-Mode Spectrum, & Leslie Smith $\beta = 0$ SGD."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.noclick_vacuum_and_smith_sgd_lab import (  # noqa: E402
    NoClickVacuumCholeskyGPT,
    NoClickVacuumCholeskySpec,
    NoClickVacuumVariantMode,
    _FusedNoClickVacuumHermitianBornAttnFn,
    compute_smith_1cycle_lr_mult,
    extract_per_mode_readout_diagnostics,
)


def main() -> None:
    torch.manual_seed(42)
    bsz, seq_len, num_heads, head_dim = 2, 16, 4, 16
    d_h = head_dim // 2
    q_rot = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float32, requires_grad=True)
    k_rot = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float32, requires_grad=True)
    payload = torch.ones((bsz, num_heads, seq_len, 8), dtype=torch.float32, requires_grad=True)
    s_vac = torch.zeros((bsz, num_heads, seq_len), dtype=torch.float32, requires_grad=True)
    berry = torch.zeros((num_heads, 1, d_h), dtype=torch.float32, requires_grad=True)
    born_b = torch.zeros((num_heads,), dtype=torch.float32, requires_grad=True)

    out_pay, p_nc = _FusedNoClickVacuumHermitianBornAttnFn.apply(
        q_rot, k_rot, payload, s_vac, berry, born_b, (0.25, True)
    )
    # Since payload is all 1s, out_pay[..., 0] = sum_{s <= t} A_{t,s}, so out_pay[..., 0] + p_nc == 1.0!
    povm_completeness_err = float((out_pay[..., 0] + p_nc - 1.0).abs().max().item())
    assert povm_completeness_err < 1e-5, f"POVM completeness violated: {povm_completeness_err}"

    loss = out_pay.square().mean() + 0.5 * p_nc.mean()
    loss.backward()
    assert q_rot.grad is not None and torch.isfinite(q_rot.grad).all()
    assert s_vac.grad is not None and torch.isfinite(s_vac.grad).all()

    spec12 = NoClickVacuumCholeskySpec(
        variant_mode=NoClickVacuumVariantMode.N2_RANK12_NOCLICK_VACUUM_POVM,
        num_cholesky_modes=12,
        vacuum_dim=4,
        use_noclick_vacuum_sink=True,
    )
    model12 = NoClickVacuumCholeskyGPT(spec12)
    idx = torch.randint(0, 2048, (2, 16), dtype=torch.long)
    e_btr, pi_btr, cos_btr = extract_per_mode_readout_diagnostics(model12, idx)
    assert e_btr.shape == (2, 16, 12) and cos_btr.shape == (2, 16, 12)

    lr_start = compute_smith_1cycle_lr_mult(1, 300, peak_lr_mult=2.4)
    lr_peak = compute_smith_1cycle_lr_mult(90, 300, peak_lr_mult=2.4)
    lr_end = compute_smith_1cycle_lr_mult(300, 300, peak_lr_mult=2.4)
    assert lr_peak > lr_start > lr_end

    report = {
        "plan": "EXPERIMENT_PLAN_080",
        "povm_completeness_max_err": povm_completeness_err,
        "mean_noclick_prob": float(p_nc.detach().mean().item()),
        "r12_energy_sum_err": float((e_btr.sum(dim=-1) - 1.0).abs().max().item()),
        "smith_1cycle_lrs": [round(lr_start, 4), round(lr_peak, 4), round(lr_end, 4)],
        "all_checks_passed": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
