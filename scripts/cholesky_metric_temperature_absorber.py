#!/usr/bin/env python3
"""Cholesky SPD Metric Temperature Absorber (`scripts/cholesky_metric_temperature_absorber.py`).

Promoted from `scripts/TOOLING_BACKLOG.md` (Wave 28 / Plan 043).
Demonstrates and proves how the analytical Parseval inverse temperature `beta_0(d) = c * sqrt(d)`
can be absorbed directly into the Symmetric Positive-Definite (SPD) `BlockCholeskyLinear` readout
operator `A = L @ L^T` so that the readout kernel itself requires zero scalar temperature
multiplier (`beta = 1.0`):
1. Mode A (Readout Bridge Absorption `L <- beta_0(d)^{1/2} L`):
   Scales `L` on the query readout bridge by `beta_0(d)^{1/2}`, yielding `A_abs = beta_0(d) A`.
2. Mode B (Symmetric Query-Detector Absorption `L_q <- beta_0(d)^{1/4} L_q`, `L_m <- beta_0(d)^{1/4} L_m`):
   Scales `L` symmetrically on both the query state and detector side by `beta_0(d)^{1/4}`,
   yielding `A_q,abs = beta_0(d)^{1/2} A_q` and `A_m,abs = beta_0(d)^{1/2} A_m`.
Verifies exact numerical equivalence (`max_logit_abs_err < 1e-5`, `max_grad_abs_err < 1e-5`)
across `d in {256, 384, 512}`.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.layers.wavelet_cholesky import BlockCholeskyLinear  # noqa: E402

_INV_SOFTPLUS_ONE = math.log(math.e - 1.0)


@dataclass(frozen=True)
class CholeskyAbsorberConfig:
    """Configuration for the Cholesky SPD metric temperature absorption audit."""

    dims: tuple[int, ...] = (256, 384, 512)
    parseval_c: float = 1.48
    vocab_size: int = 1024
    num_tokens: int = 32
    max_block_size: int = 32
    traceless_gamma: float = 0.55
    seed: int = 2043
    device: str = "cpu"
    json_out: str = ".cache/cholesky_metric_temperature_absorber.json"


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of `F.softplus(x)` for positive tensor `y > 0`."""
    y_clamped = y.clamp_min(1e-12)
    return y_clamped + torch.log(-torch.expm1(-y_clamped))


@torch.no_grad()
def absorb_scalar_into_cholesky(
    chol_module: BlockCholeskyLinear,
    cholesky_scale: float,
) -> BlockCholeskyLinear:
    """Return a cloned `BlockCholeskyLinear` with `L_absorbed = cholesky_scale * L`.

    Because `A = L @ L^T`, scaling `L <- alpha * L` scales the SPD operator `A <- alpha^2 * A`
    while preserving the condition number `kappa(A)` identically.
    """
    alpha = float(cholesky_scale)
    if alpha <= 0.0:
        raise ValueError(f"cholesky_scale must be strictly positive, got {alpha}")

    cloned = copy.deepcopy(chol_module)
    cur_pos_diag = F.softplus(cloned.raw_diag.double() + _INV_SOFTPLUS_ONE)
    target_pos_diag = alpha * cur_pos_diag
    new_raw_diag = inv_softplus(target_pos_diag) - _INV_SOFTPLUS_ONE

    cloned.raw_tril.copy_((alpha * cloned.raw_tril.double()).to(cloned.raw_tril.dtype))
    cloned.raw_diag.copy_(new_raw_diag.to(cloned.raw_diag.dtype))
    return cloned


def compute_convex_traceless_logits(
    scores: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """Compute Born-Boltzmann convex-traceless logits `beta * (c_hat + gamma * (relu(c_hat)^2 - mean))`."""
    c_hat = scores - scores.mean(dim=-1, keepdim=True)
    pos_sq = F.relu(c_hat).square()
    obs = c_hat + float(gamma) * (pos_sq - pos_sq.mean(dim=-1, keepdim=True))
    return float(beta) * obs


def audit_dim_absorption(
    embed_dim: int,
    cfg: CholeskyAbsorberConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Audit both one-sided (`beta_0^{1/2}`) and symmetric two-sided (`beta_0^{1/4}`) absorption at width `d`."""
    d = int(embed_dim)
    v_sz = int(cfg.vocab_size)
    n_tok = int(cfg.num_tokens)
    beta_0 = float(cfg.parseval_c) * math.sqrt(float(d))
    gamma = float(cfg.traceless_gamma)

    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + d * 37)
    psi_base = F.normalize(torch.randn((n_tok, d), generator=gen, dtype=torch.float32).to(device), p=2.0, dim=-1)
    m_base = F.normalize(torch.randn((v_sz, d), generator=gen, dtype=torch.float32).to(device), p=2.0, dim=-1)
    targets = (torch.arange(n_tok, device=device, dtype=torch.long) * 31 + 7) % v_sz

    chol_q = BlockCholeskyLinear(dim=d, max_block_size=cfg.max_block_size).to(device)
    chol_m = BlockCholeskyLinear(dim=d, max_block_size=cfg.max_block_size).to(device)
    with torch.no_grad():
        chol_q.raw_tril.copy_(0.08 * torch.randn(chol_q.raw_tril.shape, generator=gen).to(device))
        chol_q.raw_diag.copy_(0.12 * torch.randn(chol_q.raw_diag.shape, generator=gen).to(device))
        chol_m.raw_tril.copy_(0.06 * torch.randn(chol_m.raw_tril.shape, generator=gen).to(device))
        chol_m.raw_diag.copy_(0.10 * torch.randn(chol_m.raw_diag.shape, generator=gen).to(device))

    # 1. Explicit beta_0(d) vs. Mode A Bridge Absorption (L_q <- beta_0^{1/2} L_q, beta = 1.0)
    chol_q_half = absorb_scalar_into_cholesky(chol_q, math.sqrt(beta_0))
    l_factor_err_half = float(
        (chol_q_half.cholesky_factor() - math.sqrt(beta_0) * chol_q.cholesky_factor()).abs().max().item()
    )

    psi_exp = psi_base.clone().detach().requires_grad_(True)
    scores_exp = chol_q(psi_exp) @ m_base.t()
    logits_exp = compute_convex_traceless_logits(scores_exp, beta=beta_0, gamma=gamma)
    ce_exp = F.cross_entropy(logits_exp, targets)
    ce_exp.backward()

    psi_bridge = psi_base.clone().detach().requires_grad_(True)
    scores_bridge = chol_q_half(psi_bridge) @ m_base.t()
    # When scores are pre-scaled by beta_0, gamma_absorbed = gamma / beta_0 yields identical quadratic peak scale
    logits_bridge = compute_convex_traceless_logits(scores_bridge, beta=1.0, gamma=gamma / beta_0)
    ce_bridge = F.cross_entropy(logits_bridge, targets)
    ce_bridge.backward()

    assert psi_exp.grad is not None and psi_bridge.grad is not None
    bridge_logit_err = float((logits_exp.detach() - logits_bridge.detach()).abs().max().item())
    bridge_ce_err = float((ce_exp.detach() - ce_bridge.detach()).abs().item())
    bridge_grad_err = float((psi_exp.grad - psi_bridge.grad).abs().max().item())

    # 2. Explicit beta_0(d) vs. Mode B Symmetric Query-Detector Absorption (L_q, L_m <- beta_0^{1/4} L, beta = 1.0)
    chol_q_fourth = absorb_scalar_into_cholesky(chol_q, beta_0**0.25)
    chol_m_fourth = absorb_scalar_into_cholesky(chol_m, beta_0**0.25)

    psi_sym_exp = psi_base.clone().detach().requires_grad_(True)
    scores_sym_exp = chol_q(psi_sym_exp) @ chol_m(m_base).t()
    logits_sym_exp = compute_convex_traceless_logits(scores_sym_exp, beta=beta_0, gamma=gamma)
    ce_sym_exp = F.cross_entropy(logits_sym_exp, targets)
    ce_sym_exp.backward()

    psi_sym_abs = psi_base.clone().detach().requires_grad_(True)
    scores_sym_abs = chol_q_fourth(psi_sym_abs) @ chol_m_fourth(m_base).t()
    logits_sym_abs = compute_convex_traceless_logits(scores_sym_abs, beta=1.0, gamma=gamma / beta_0)
    ce_sym_abs = F.cross_entropy(logits_sym_abs, targets)
    ce_sym_abs.backward()

    assert psi_sym_exp.grad is not None and psi_sym_abs.grad is not None
    sym_logit_err = float((logits_sym_exp.detach() - logits_sym_abs.detach()).abs().max().item())
    sym_ce_err = float((ce_sym_exp.detach() - ce_sym_abs.detach()).abs().item())
    sym_grad_err = float((psi_sym_exp.grad - psi_sym_abs.grad).abs().max().item())

    cond_base = chol_q.condition_metrics()["spd_cond_number"]
    cond_absorbed = chol_q_half.condition_metrics()["spd_cond_number"]
    cond_rel_err = abs(cond_absorbed - cond_base) / max(cond_base, 1e-12)

    return {
        "embed_dim": d,
        "beta_0": beta_0,
        "bridge_chol_scale_sqrt_beta0": math.sqrt(beta_0),
        "symmetric_chol_scale_fourth_root_beta0": beta_0**0.25,
        "absorbed_kernel_beta": 1.0,
        "cholesky_factor_max_abs_err": l_factor_err_half,
        "bridge_mode_logit_max_abs_err": bridge_logit_err,
        "bridge_mode_ce_abs_err": bridge_ce_err,
        "bridge_mode_grad_max_abs_err": bridge_grad_err,
        "symmetric_mode_logit_max_abs_err": sym_logit_err,
        "symmetric_mode_ce_abs_err": sym_ce_err,
        "symmetric_mode_grad_max_abs_err": sym_grad_err,
        "max_logit_abs_err": max(bridge_logit_err, sym_logit_err),
        "max_grad_abs_err": max(bridge_grad_err, sym_grad_err),
        "spd_cond_number_base": cond_base,
        "spd_cond_number_absorbed": cond_absorbed,
        "spd_cond_number_rel_err": cond_rel_err,
    }


def run_absorber_audit(cfg: CholeskyAbsorberConfig) -> dict[str, Any]:
    """Run Cholesky SPD metric temperature absorption audit across all widths in `cfg.dims`."""
    device = torch.device(cfg.device)
    by_dim: dict[str, Any] = {}
    for d in cfg.dims:
        by_dim[f"d_{d}"] = audit_dim_absorption(d, cfg, device)

    max_logit_err = max(v["max_logit_abs_err"] for v in by_dim.values())
    max_grad_err = max(v["max_grad_abs_err"] for v in by_dim.values())
    max_chol_err = max(v["cholesky_factor_max_abs_err"] for v in by_dim.values())
    max_cond_rel_err = max(v["spd_cond_number_rel_err"] for v in by_dim.values())

    gates = {
        "max_logit_abs_err_lt_1e5": max_logit_err < 1e-5,
        "max_grad_abs_err_lt_1e5": max_grad_err < 1e-5,
        "max_cholesky_factor_err_lt_1e6": max_chol_err < 1e-6,
        "spd_condition_number_invariant_lt_1e5": max_cond_rel_err < 1e-5,
    }
    return {
        "script": "cholesky_metric_temperature_absorber",
        "config": asdict(cfg),
        "summary_metrics": {
            "max_logit_abs_err": max_logit_err,
            "max_grad_abs_err": max_grad_err,
            "max_cholesky_factor_err": max_chol_err,
            "max_spd_cond_number_rel_err": max_cond_rel_err,
            "absorbed_kernel_beta": 1.0,
        },
        "by_dim": by_dim,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    """Format a Markdown summary table for the Cholesky metric temperature absorber audit."""
    sm = report["summary_metrics"]
    lines = [
        "# Cholesky SPD Metric Temperature Absorber Audit (`beta = 1.0` Readout Kernel)",
        "",
        "| Width `d` | `beta_0(d)` | Bridge Scale `beta_0^{1/2}` | Sym Scale `beta_0^{1/4}` | Kernel `beta` | Bridge Logit/Grad Err | Sym Logit/Grad Err | Cond `#` Rel Err |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in report["by_dim"].values():
        lines.append(
            f"| `{row['embed_dim']}` | `{row['beta_0']:.2f}` | `{row['bridge_chol_scale_sqrt_beta0']:.4f}` | "
            f"`{row['symmetric_chol_scale_fourth_root_beta0']:.4f}` | `{row['absorbed_kernel_beta']:.1f}` | "
            f"`{row['bridge_mode_logit_max_abs_err']:.2e}` / `{row['bridge_mode_grad_max_abs_err']:.2e}` | "
            f"`{row['symmetric_mode_logit_max_abs_err']:.2e}` / `{row['symmetric_mode_grad_max_abs_err']:.2e}` | "
            f"`{row['spd_cond_number_rel_err']:.2e}` |"
        )
    lines.extend([
        "",
        "## Acceptance Gates",
        f"- `max_logit_abs_err = {sm['max_logit_abs_err']:.2e} < 1e-5`: `{report['gates']['max_logit_abs_err_lt_1e5']}`",
        f"- `max_grad_abs_err = {sm['max_grad_abs_err']:.2e} < 1e-5`: `{report['gates']['max_grad_abs_err_lt_1e5']}`",
        f"- `max_cholesky_factor_err = {sm['max_cholesky_factor_err']:.2e} < 1e-6`: `{report['gates']['max_cholesky_factor_err_lt_1e6']}`",
        f"- `max_spd_cond_number_rel_err = {sm['max_spd_cond_number_rel_err']:.2e} < 1e-5`: `{report['gates']['spd_condition_number_invariant_lt_1e5']}`",
        f"- `overall_pass`: `{report['overall_pass']}`",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> CholeskyAbsorberConfig:
    """Parse CLI arguments into a frozen `CholeskyAbsorberConfig`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=str, default="256,384,512")
    parser.add_argument("--embed-dim", type=int, default=0)
    parser.add_argument("--parseval-c", type=float, default=1.48)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=".cache/cholesky_metric_temperature_absorber.json")
    args = parser.parse_args(argv)
    if int(args.embed_dim) > 0:
        dims = (int(args.embed_dim),)
    else:
        dims = tuple(int(x.strip()) for x in args.dims.split(",") if x.strip())
    return CholeskyAbsorberConfig(
        dims=dims,
        parseval_c=float(args.parseval_c),
        vocab_size=int(args.vocab_size),
        num_tokens=int(args.num_tokens),
        device=str(args.device),
        json_out=str(args.json_out),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    cfg = parse_args(argv)
    report = run_absorber_audit(cfg)
    out_path = Path(cfg.json_out) if Path(cfg.json_out).is_absolute() else (ROOT / cfg.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_markdown_report(report))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
