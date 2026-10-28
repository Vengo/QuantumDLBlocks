#!/usr/bin/env python3
"""Complex Stiefel Orthogonality & Large-Batch Muon Scaling Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Audits Joint-Complex Polar Express vs. Separate-Real Polar Express (`Track 1` / `Issue 1`):
1. Complex unitarity error `||U U^H - I||_F / sqrt(d)` (`< 0.05` vs `> 0.40` for separate real),
2. Global `U(1)` phase equivariance `Polar_C(e^{i theta} G) == e^{i theta} Polar_C(G)`,
3. Large-batch learning-rate scaling `eta_muon(B_tok) = eta_0 * (B_tok / 8192)^0.50`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdlblocks.benchmarks.kronecker_diagnostic_utils import write_standardized_probe_json
from qdlblocks.optim.muon import _POLAR_EXPRESS_COEFFS_5, zeropower_via_polarexpress5

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "complex_stiefel_orthogonality_and_large_batch_probe.py"
DEFAULT_MATRIX_DIM: int = 16
DEFAULT_BATCH_TOKENS_CSV: str = "8192,16384,32768,65536"
BASE_REFERENCE_BATCH_TOKENS: int = 8192
BASE_MUON_LR: float = 0.030
BASE_EMBED_LR: float = 0.020
MUON_BATCH_EXPONENT: float = 0.50
EMBED_BATCH_EXPONENT: float = 0.35
POLAR_SAFETY_SCALE: float = 1.01
POLAR_NORM_EPS: float = 1e-7
PHASE_ROTATION_RAD: float = 0.61803398875


@torch.no_grad()
def _joint_complex_polar_express_2d(
    g_re: torch.Tensor,
    g_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run 5-step quintic Polar Express on joint complex matrix `G = G_re + i G_im` (`<= 8` args)."""
    xr, xi = g_re.float(), g_im.float()
    frob = torch.sqrt(xr.square().sum() + xi.square().sum())
    scale = frob * POLAR_SAFETY_SCALE + POLAR_NORM_EPS
    xr, xi = xr / scale, xi / scale

    for a_c, b_c, c_c in _POLAR_EXPRESS_COEFFS_5:
        ar = xr @ xr.t() + xi @ xi.t()
        ai = xi @ xr.t() - xr @ xi.t()
        a2r = ar @ ar - ai @ ai
        a2i = ar @ ai + ai @ ar
        br = b_c * ar + c_c * a2r
        bi = b_c * ai + c_c * a2i
        xr_next = a_c * xr + (br @ xr - bi @ xi)
        xi_next = a_c * xi + (br @ xi + bi @ xr)
        xr, xi = xr_next, xi_next
    return xr, xi


@torch.no_grad()
def compare_joint_complex_vs_separate_real_polar(
    dim: int = DEFAULT_MATRIX_DIM,
    device: str = "cpu",
) -> dict[str, float]:
    """Compare complex unitarity and U(1) phase equivariance of Joint-Complex vs. Separate-Real Polar (`<= 8` args)."""
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1337)
    d = int(dim)
    g_re = torch.randn((d, d), generator=gen, device=dev, dtype=torch.float32)
    g_im = torch.randn((d, d), generator=gen, device=dev, dtype=torch.float32)
    eye_c = torch.eye(d, device=dev, dtype=torch.complex64)

    # Separate real Polar Express (Legacy Issue 1):
    u_sep_re = zeropower_via_polarexpress5(g_re)
    u_sep_im = zeropower_via_polarexpress5(g_im)
    u_sep_c = torch.complex(u_sep_re, u_sep_im)
    sep_unitary_err = float(torch.linalg.matrix_norm(u_sep_c @ u_sep_c.conj().t() - eye_c).item() / math.sqrt(float(d)))

    # Joint Complex Polar Express (Wave 44 Track 1 Fix):
    u_jnt_re, u_jnt_im = _joint_complex_polar_express_2d(g_re, g_im)
    u_jnt_c = torch.complex(u_jnt_re, u_jnt_im)
    jnt_unitary_err = float(torch.linalg.matrix_norm(u_jnt_c @ u_jnt_c.conj().t() - eye_c).item() / math.sqrt(float(d)))

    # U(1) phase equivariance test: Polar(e^{i theta} G) vs e^{i theta} Polar(G)
    cos_t, sin_t = math.cos(PHASE_ROTATION_RAD), math.sin(PHASE_ROTATION_RAD)
    rot_g_re = cos_t * g_re - sin_t * g_im
    rot_g_im = sin_t * g_re + cos_t * g_im

    sep_rot_c = torch.complex(zeropower_via_polarexpress5(rot_g_re), zeropower_via_polarexpress5(rot_g_im))
    jnt_rot_re, jnt_rot_im = _joint_complex_polar_express_2d(rot_g_re, rot_g_im)
    jnt_rot_c = torch.complex(jnt_rot_re, jnt_rot_im)
    phase_c = complex(cos_t, sin_t)

    sep_phase_err = float(torch.linalg.matrix_norm(sep_rot_c - phase_c * u_sep_c).item() / math.sqrt(float(d)))
    jnt_phase_err = float(torch.linalg.matrix_norm(jnt_rot_c - phase_c * u_jnt_c).item() / math.sqrt(float(d)))

    return {
        "separate_real_unitary_err": round(sep_unitary_err, 6),
        "joint_complex_unitary_err": round(jnt_unitary_err, 6),
        "unitary_err_reduction_ratio": round(sep_unitary_err / max(jnt_unitary_err, 1e-8), 2),
        "separate_real_u1_phase_err": round(sep_phase_err, 6),
        "joint_complex_u1_phase_err": round(jnt_phase_err, 8),
    }


def run_stiefel_and_large_batch_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run Joint-Complex Polar Express audit and large-batch LR scaling table (`<= 8` args)."""
    polar_stats = compare_joint_complex_vs_separate_real_polar(dim=int(args.dim), device=str(args.device))
    batch_tokens_list = [int(x.strip()) for x in str(args.batch_tokens).split(",") if x.strip()]
    scaling_rows: list[dict[str, float]] = []
    for b_tok in batch_tokens_list:
        ratio = max(1.0, float(b_tok) / float(BASE_REFERENCE_BATCH_TOKENS))
        muon_scale = ratio ** MUON_BATCH_EXPONENT
        embed_scale = ratio ** EMBED_BATCH_EXPONENT
        scaling_rows.append({
            "batch_tokens": float(b_tok),
            "muon_lr_scale": round(muon_scale, 4),
            "recommended_muon_lr": round(BASE_MUON_LR * muon_scale, 5),
            "recommended_embed_lr": round(BASE_EMBED_LR * embed_scale, 5),
        })

    metrics: dict[str, Any] = {
        **polar_stats,
        "max_tested_batch_tokens": max(batch_tokens_list) if batch_tokens_list else BASE_REFERENCE_BATCH_TOKENS,
        "max_muon_lr_scale": scaling_rows[-1]["muon_lr_scale"] if scaling_rows else 1.0,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"large_batch_schedule": scaling_rows},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Audit Joint-Complex StiefelMuon orthogonality and batch scaling.")
    parser.add_argument("--dim", type=int, default=DEFAULT_MATRIX_DIM)
    parser.add_argument("--batch-tokens", type=str, default=DEFAULT_BATCH_TOKENS_CSV)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_stiefel_and_large_batch_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} joint_err={payload['metrics']['joint_complex_unitary_err']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
