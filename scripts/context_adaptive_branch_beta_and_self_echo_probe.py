#!/usr/bin/env python3
"""Context-Adaptive Branch Beta Router & Carrier-Orthogonal Self-Echo Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Audits the Track 3 Carrier-Orthogonal Complex Self-Echo projection (`ov_re + i ov_im` strictly in `H_perp`
so `<c_ref | op>` is untouched) and Track 4 Context-Selective Unitary Inner-Product Polysemy Router `beta_{t,r}`.
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

from qdlblocks.benchmarks.kronecker_diagnostic_utils import (
    DEFAULT_DIAGNOSTIC_SEQ_LEN,
    DEFAULT_FINEWEB_VAL_SHARD_PATH,
    DEFAULT_NGRAM_CALIBRATION_CACHE_PATH,
    DiagnosticModelSpec,
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)
from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize
from qdlblocks.models.calibrated_kronecker_born_gpt import _build_canonical_carrier_ket
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import _project_c16_ket_orthogonal_to_carrier

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "context_adaptive_branch_beta_and_self_echo_probe.py"
DEFAULT_ECHO_SYNTH_SAMPLES: int = 64
DEFAULT_FACTOR_DIM: int = 16
PHASE_TEST_RADIANS: float = 0.7853981633974483  # pi / 4
ORTHO_NORM_EPS: float = 1e-8


@torch.no_grad()
def verify_carrier_orthogonal_complex_self_echo(
    num_samples: int = DEFAULT_ECHO_SYNTH_SAMPLES,
    factor_dim: int = DEFAULT_FACTOR_DIM,
    device: str = "cpu",
) -> dict[str, float]:
    """Compare legacy real-only ambient self-echo vs. Track 3 carrier-orthogonal complex self-echo (`<= 8` args)."""
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1337)
    c_re, c_im = _build_canonical_carrier_ket(factor_dim, dev)

    p0_re, p0_im = complex_l2_normalize(
        0.5 * c_re.unsqueeze(0) + torch.randn((num_samples, factor_dim), generator=gen, device=dev),
        0.5 * c_im.unsqueeze(0) + torch.randn((num_samples, factor_dim), generator=gen, device=dev),
        dim=-1,
    )
    op_re, op_im = complex_l2_normalize(
        0.6 * c_re.unsqueeze(0) + 0.4 * p0_re + 0.3 * torch.randn((num_samples, factor_dim), generator=gen, device=dev),
        0.6 * c_im.unsqueeze(0) + 0.4 * p0_im + 0.3 * torch.randn((num_samples, factor_dim), generator=gen, device=dev),
        dim=-1,
    )

    # Legacy Wave 43 real-only ambient subtraction:
    ov_re_leg = (p0_re * op_re + p0_im * op_im).sum(dim=-1, keepdim=True)
    leg_re = op_re - ov_re_leg * p0_re
    leg_im = op_im - ov_re_leg * p0_im
    leg_c_re = (leg_re * c_re + leg_im * c_im).sum(dim=-1)
    orig_c_re = (op_re * c_re + op_im * c_im).sum(dim=-1)
    legacy_carrier_perturbation = float((leg_c_re - orig_c_re).abs().mean().item())

    # Track 3 Carrier-Orthogonal Complex Self-Echo in H_perp:
    p_perp_re, p_perp_im = _project_c16_ket_orthogonal_to_carrier(p0_re, p0_im)
    p_hat_re, p_hat_im = complex_l2_normalize(p_perp_re, p_perp_im, dim=-1, eps=ORTHO_NORM_EPS)
    ov_re = (p_hat_re * op_re + p_hat_im * op_im).sum(dim=-1, keepdim=True)
    ov_im = (p_hat_re * op_im - p_hat_im * op_re).sum(dim=-1, keepdim=True)
    new_re = op_re - (ov_re * p_hat_re - ov_im * p_hat_im)
    new_im = op_im - (ov_re * p_hat_im + ov_im * p_hat_re)

    new_c_re = (new_re * c_re + new_im * c_im).sum(dim=-1)
    ortho_carrier_perturbation = float((new_c_re - orig_c_re).abs().max().item())

    res_ov_re = (p_hat_re * new_re + p_hat_im * new_im).sum(dim=-1)
    res_ov_im = (p_hat_re * new_im - p_hat_im * new_re).sum(dim=-1)
    residual_self_echo_mag = float(torch.sqrt(res_ov_re.square() + res_ov_im.square()).max().item())

    cos_p, sin_p = math.cos(PHASE_TEST_RADIANS), math.sin(PHASE_TEST_RADIANS)
    p0_rot_re = cos_p * p_hat_re - sin_p * p_hat_im
    p0_rot_im = sin_p * p_hat_re + cos_p * p_hat_im
    ov_rot_re = (p0_rot_re * op_re + p0_rot_im * op_im).sum(dim=-1, keepdim=True)
    ov_rot_im = (p0_rot_re * op_im - p0_rot_im * op_re).sum(dim=-1, keepdim=True)
    rot_re = op_re - (ov_rot_re * p0_rot_re - ov_rot_im * p0_rot_im)
    rot_im = op_im - (ov_rot_re * p0_rot_im + ov_rot_im * p0_rot_re)
    u1_phase_invariance_err = float(torch.sqrt((new_re - rot_re).square() + (new_im - rot_im).square()).max().item())

    return {
        "legacy_carrier_perturbation": round(legacy_carrier_perturbation, 6),
        "ortho_carrier_perturbation": round(ortho_carrier_perturbation, 8),
        "residual_self_echo_in_h_perp": round(residual_self_echo_mag, 8),
        "u1_phase_invariance_err": round(u1_phase_invariance_err, 8),
    }


@torch.no_grad()
def run_branch_beta_and_self_echo_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Execute context-adaptive branch beta and carrier-orthogonal self-echo probe (`<= 8` args)."""
    spec = DiagnosticModelSpec(
        arch=str(args.arch),
        checkpoint_path=args.checkpoint,
        calibration_cache_path=str(args.calibration_cache),
        n_layer=int(args.num_layers),
        meaning_rank=int(args.meaning_rank),
        seq_len=int(args.seq_len),
        device=str(args.device),
    )
    model, metadata = load_or_build_calibrated_kronecker_model(spec)
    x_batch, _ = load_kronecker_probe_batch(
        val_shard_path=str(args.val_shard),
        batch_size=2,
        seq_len=min(32, int(args.seq_len)),
        device=str(args.device),
    )
    _, _, branch_beta, _ = model.extract_readout_states(x_batch)
    beta_f = branch_beta.detach().float()
    beta_std = float(beta_f[:, 0].std().item())
    beta_mean = [round(float(v), 6) for v in beta_f.mean(dim=0).tolist()]
    echo_stats = verify_carrier_orthogonal_complex_self_echo(
        num_samples=DEFAULT_ECHO_SYNTH_SAMPLES,
        factor_dim=int(model.cfg.factor_dim),
        device=str(args.device),
    )

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "branch_beta0_temporal_std": round(beta_std, 6),
        "mean_branch_beta": beta_mean,
        **echo_stats,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Audit context-adaptive branch beta router and complex self-echo.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--meaning-rank", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_DIAGNOSTIC_SEQ_LEN)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_branch_beta_and_self_echo_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} u1_err={payload['metrics']['u1_phase_invariance_err']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
