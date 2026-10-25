#!/usr/bin/env python3
"""Profile Per-Layer Pre-Attention 1D Wavefunction Conv Receptive Fields & Coincidence Warmup (`Wave 33` / `Plan 048`).

Promoted from `scripts/TOOLING_BACKLOG.md` as the Build-Before-Launch diagnostic for `EXPERIMENT_PLAN_048.md`:
1. Audits per-layer pre-attention temporal receptive-field expansion across lags `Delta_t in {0, 1, 2, 4, 8, 12, 16}`
   in stacked `PreAttnMultiRateWaveletConvBlock` layers vs. post-attention channel U-Net blocks, and verifies
   strict causality (`max_future_leakage < 1e-6`).
2. Audits Progress-Ramped Coincidence Warmup (`coincidence_gamma(tau) = 0.22 * tau`,
   `subspace_var_penalty(tau) = 0.14 * tau`) across `tau in [0, 1]` (`0` to `3,600` steps / `29.49M` tokens),
   quantifying elimination of early-step unconverged fine-leaf CE penalty at `tau = 0` and runner-up
   impostor suppression at `tau = 1`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.multirate_wavefunction_conv import (
    DEFAULT_LEVEL_TEMPORAL_STRIDES,
    MultiRateWavefunctionConvBank,
    WaveletScaleState,
)
from qdlblocks.layers.subspace_coincidence_born import (
    DEFAULT_COINCIDENCE_GAMMA,
    DEFAULT_SUBSPACE_VAR_PENALTY,
    compute_multi_subspace_coincidence_observable,
    subspace_l2_normalize,
)
from qdlblocks.models.level_separated_wavelet_conv_gpt import (
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    PreAttnMultiRateWaveletConvBlock,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_048.md (Wave 33 Constants & Calibration Ledger)
# ==============================================================================
DEFAULT_PROFILER_EMBED_DIM: int = 256
DEFAULT_PROFILER_NUM_LAYERS: int = 4
DEFAULT_PROFILER_SEQ_LEN: int = 32
DEFAULT_PROFILER_QUERY_T: int = 24
DEFAULT_PROFILER_LAGS: tuple[int, ...] = (0, 1, 2, 4, 8, 12, 16)
DEFAULT_TAU_GRID: tuple[float, ...] = (0.0, 0.0833, 0.25, 0.50, 0.75, 1.0)
CAUSALITY_LEAKAGE_MAX_ATOL: float = 1e-6
MIN_LAG8_TO_LAG0_RATIO_GATE: float = 0.02
MIN_TERMINAL_RUNNERUP_SUPPRESSION_NATS: float = 0.80


@dataclass(frozen=True)
class ReceptiveFieldProfilerConfig:
    """Configuration for `multirate_wavefunction_conv_receptive_field_profiler.py` (`<= 8` args)."""

    embed_dim: int = DEFAULT_PROFILER_EMBED_DIM
    num_layers: int = DEFAULT_PROFILER_NUM_LAYERS
    seq_len: int = DEFAULT_PROFILER_SEQ_LEN
    query_t: int = DEFAULT_PROFILER_QUERY_T
    device: str = "cpu"


def audit_pre_attn_temporal_receptive_field(cfg: ReceptiveFieldProfilerConfig) -> dict[str, Any]:
    """Measure pre-attention temporal impulse response across lags `0..16` and verify strict causality."""
    torch.manual_seed(48)
    device = torch.device(cfg.device)
    d = int(cfg.embed_dim)
    seq_len = int(cfg.seq_len)
    q_t = min(seq_len - 2, max(16, int(cfg.query_t)))

    banks = [
        MultiRateWavefunctionConvBank(
            embed_dim=d,
            temporal_strides=DEFAULT_LEVEL_TEMPORAL_STRIDES,
            fast_spd_bmm=True,
        ).to(device)
        for _ in range(cfg.num_layers)
    ]

    x_in = torch.randn((1, seq_len, d), device=device, dtype=torch.float32, requires_grad=True)
    psi_cur = l2_normalize(x_in, dim=-1)
    layer_pre_attn_states: list[torch.Tensor] = []

    for l_idx, bank in enumerate(banks):
        sep_in = WaveletScaleState.from_interleaved(psi_cur).normalize_levels()
        conv_state = bank.forward_levels(sep_in)
        psi_conv = conv_state.to_interleaved_product_of_spheres()
        gate = 0.24 / math.sqrt(float(l_idx + 1))
        psi_pre_attn = l2_normalize(psi_cur + math.tanh(gate) * psi_conv, dim=-1)
        layer_pre_attn_states.append(psi_pre_attn)
        psi_cur = psi_pre_attn

    per_layer_lag_norms: dict[str, dict[str, float]] = {}
    max_future_leakage = 0.0

    raw_probe = torch.randn((d,), device=device, dtype=torch.float32)
    for l_idx, psi_pre in enumerate(layer_pre_attn_states):
        p_unit = psi_pre[0, q_t].detach()
        probe_dir = F.normalize(raw_probe - (raw_probe * p_unit).sum() * p_unit, dim=-1)
        scalar_out = (psi_pre[0, q_t] * probe_dir).sum()
        grad_x = torch.autograd.grad(scalar_out, x_in, retain_graph=True)[0][0]

        future_norms = torch.linalg.vector_norm(grad_x[q_t + 1 :], dim=-1)
        if future_norms.numel() > 0:
            max_future_leakage = max(max_future_leakage, float(future_norms.max().item()))

        lag_dict: dict[str, float] = {}
        for lag in DEFAULT_PROFILER_LAGS:
            src_t = q_t - lag
            val = float(torch.linalg.vector_norm(grad_x[src_t]).item()) if src_t >= 0 else 0.0
            lag_dict[f"lag_{lag}"] = round(val, 6)
        per_layer_lag_norms[f"layer_{l_idx}"] = lag_dict

    last_layer = per_layer_lag_norms[f"layer_{cfg.num_layers - 1}"]
    lag0 = max(1e-8, last_layer["lag_0"])
    lag8_ratio = last_layer["lag_8"] / lag0
    lag16_ratio = last_layer["lag_16"] / lag0

    model_cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=1024,
        valid_vocab_size=1000,
        embed_dim=d,
        num_layers=cfg.num_layers,
        num_heads=max(2, d // 64),
        block_size=seq_len,
    )
    model = LevelSeparatedWaveletConvGPT(model_cfg).to(device)
    all_blocks_are_pre_attn = all(isinstance(b, PreAttnMultiRateWaveletConvBlock) for b in model.blocks)
    has_legacy_post_attn_unet = any(hasattr(b, "wavelet_branch") for b in model.blocks)

    return {
        "query_position_t": q_t,
        "temporal_strides": list(DEFAULT_LEVEL_TEMPORAL_STRIDES),
        "all_blocks_are_pre_attn_conv": bool(all_blocks_are_pre_attn),
        "has_legacy_post_attn_channel_unet": bool(has_legacy_post_attn_unet),
        "max_future_leakage": max_future_leakage,
        "strict_causality_passed": bool(max_future_leakage < CAUSALITY_LEAKAGE_MAX_ATOL),
        "final_layer_lag8_to_lag0_ratio": round(lag8_ratio, 6),
        "final_layer_lag16_to_lag0_ratio": round(lag16_ratio, 6),
        "receptive_field_expansion_passed": bool(
            all_blocks_are_pre_attn
            and not has_legacy_post_attn_unet
            and lag8_ratio >= MIN_LAG8_TO_LAG0_RATIO_GATE
        ),
        "per_layer_pre_attn_impulse_norms": per_layer_lag_norms,
    }


def audit_progress_ramped_coincidence_warmup(cfg: ReceptiveFieldProfilerConfig) -> dict[str, Any]:
    """Audit progress-ramped coincidence warmup across `tau in [0, 1]` under fine-leaf convergence."""
    torch.manual_seed(148)
    d = int(cfg.embed_dim)
    m_sub = 4
    d_sub = d // m_sub
    num_classes = 256
    beta_0 = 1.48 * math.sqrt(float(d))

    det_sub = F.normalize(torch.randn(num_classes, m_sub, d_sub), dim=-1)
    true_y = 0
    runnerup_y = 1

    # Simulate query state where coarse/mid/sub subspaces (0, 1, 2) align strongly with target y=0,
    # while runner-up y=1 has a single-subspace impostor spike in subspace 0, and fine subspace 3
    # converges with training progress tau in [0, 1].
    noise_fine = F.normalize(torch.randn(d_sub), dim=-1)
    schedule_trajectory: list[dict[str, float]] = []

    for tau in DEFAULT_TAU_GRID:
        q_sub = torch.zeros(1, m_sub, d_sub)
        q_sub[0, 0] = F.normalize(0.82 * det_sub[true_y, 0] + 0.18 * torch.randn(d_sub), dim=-1)
        q_sub[0, 1] = F.normalize(0.78 * det_sub[true_y, 1] + 0.22 * torch.randn(d_sub), dim=-1)
        q_sub[0, 2] = F.normalize(0.74 * det_sub[true_y, 2] + 0.26 * torch.randn(d_sub), dim=-1)
        fine_align = 0.05 + 0.75 * tau
        q_sub[0, 3] = F.normalize(fine_align * det_sub[true_y, 3] + (1.0 - fine_align) * noise_fine, dim=-1)

        cos_m = torch.einsum("nmd,vmd->nvm", q_sub, det_sub)
        # Inject single-subspace runner-up impostor spike in subspace 0 on class 1
        cos_m[0, runnerup_y, 0] = 0.92
        cos_m[0, runnerup_y, 1] = -0.05
        cos_m[0, runnerup_y, 2] = -0.04
        cos_m[0, runnerup_y, 3] = -0.06

        gamma_warm = DEFAULT_COINCIDENCE_GAMMA * tau
        var_warm = DEFAULT_SUBSPACE_VAR_PENALTY * tau

        obs_warm = compute_multi_subspace_coincidence_observable(
            cos_m,
            traceless_gamma=0.25,
            coincidence_gamma=gamma_warm,
            subspace_var_penalty=var_warm,
        )["observable"]
        obs_unwarmed = compute_multi_subspace_coincidence_observable(
            cos_m,
            traceless_gamma=0.25,
            coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
            subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        )["observable"]
        obs_m1 = compute_multi_subspace_coincidence_observable(
            cos_m,
            traceless_gamma=0.25,
            coincidence_gamma=0.0,
            subspace_var_penalty=0.0,
        )["observable"]

        ce_warm = float(F.cross_entropy(beta_0 * obs_warm, torch.tensor([true_y])).item())
        ce_unwarmed = float(F.cross_entropy(beta_0 * obs_unwarmed, torch.tensor([true_y])).item())
        ce_m1 = float(F.cross_entropy(beta_0 * obs_m1, torch.tensor([true_y])).item())
        margin_warm = float((beta_0 * (obs_warm[0, true_y] - obs_warm[0, runnerup_y])).item())
        margin_m1 = float((beta_0 * (obs_m1[0, true_y] - obs_m1[0, runnerup_y])).item())

        schedule_trajectory.append(
            {
                "tau": round(tau, 4),
                "coincidence_gamma_tau": round(gamma_warm, 5),
                "subspace_var_penalty_tau": round(var_warm, 5),
                "ce_m1_no_coinc": round(ce_m1, 5),
                "ce_unwarmed_coinc": round(ce_unwarmed, 5),
                "ce_progress_warmed_coinc": round(ce_warm, 5),
                "runnerup_margin_gain_nats": round(margin_warm - margin_m1, 5),
            }
        )

    tau0_entry = schedule_trajectory[0]
    tau1_entry = schedule_trajectory[-1]
    early_tau0_penalty = abs(tau0_entry["ce_progress_warmed_coinc"] - tau0_entry["ce_m1_no_coinc"])
    terminal_margin_gain = tau1_entry["runnerup_margin_gain_nats"]

    # Also verify LevelSeparatedWaveletConvGPT schedule methods across tau in [0, 1]
    m_cfg = LevelSeparatedWaveletConvConfig(embed_dim=d, num_layers=2, num_heads=4, use_coincidence_warmup=True)
    model = LevelSeparatedWaveletConvGPT(m_cfg)
    model.set_training_progress(0.0)
    spec_t0 = model.build_coincidence_spec()
    model.set_training_progress(0.5)
    spec_t05 = model.build_coincidence_spec()
    model.set_training_progress(1.0)
    spec_t1 = model.build_coincidence_spec()

    # Check subspace energy invariance
    x_test = torch.randn(8, d)
    x_bal = subspace_l2_normalize(x_test, num_subspaces=m_sub, partition_mode="strided")
    max_norm_err = float((torch.linalg.vector_norm(x_bal, dim=-1) - 1.0).abs().max().item())

    return {
        "early_tau0_ce_penalty_nats": round(early_tau0_penalty, 8),
        "terminal_tau1_runnerup_margin_gain_nats": round(terminal_margin_gain, 5),
        "spec_gamma_tau0": round(spec_t0.coincidence_gamma, 6),
        "spec_gamma_tau05": round(spec_t05.coincidence_gamma, 6),
        "spec_gamma_tau1": round(spec_t1.coincidence_gamma, 6),
        "max_unit_norm_err": max_norm_err,
        "coincidence_warmup_passed": bool(
            early_tau0_penalty <= 1e-5
            and abs(spec_t0.coincidence_gamma) <= 1e-7
            and abs(spec_t05.coincidence_gamma - 0.5 * DEFAULT_COINCIDENCE_GAMMA) <= 1e-6
            and abs(spec_t1.coincidence_gamma - DEFAULT_COINCIDENCE_GAMMA) <= 1e-6
            and terminal_margin_gain >= MIN_TERMINAL_RUNNERUP_SUPPRESSION_NATS
        ),
        "schedule_trajectory": schedule_trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Pre-Attention MultiRateWavefunctionConvBank Receptive Fields & Coincidence Warmup"
    )
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_PROFILER_EMBED_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_PROFILER_NUM_LAYERS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_PROFILER_SEQ_LEN)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(REPO_ROOT / ".cache" / "multirate_wavefunction_conv_receptive_field_profiler.json"),
    )
    args = parser.parse_args()

    cfg = ReceptiveFieldProfilerConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        device=args.device,
    )
    rf_report = audit_pre_attn_temporal_receptive_field(cfg)
    warmup_report = audit_progress_ramped_coincidence_warmup(cfg)

    all_passed = bool(
        rf_report["strict_causality_passed"]
        and rf_report["receptive_field_expansion_passed"]
        and warmup_report["coincidence_warmup_passed"]
    )
    report = {
        "plan_id": "EXPERIMENT_PLAN_048",
        "wave_id": "W33",
        "all_passed": all_passed,
        "H1_pre_attn_receptive_field_and_causality_passed": bool(
            rf_report["strict_causality_passed"] and rf_report["receptive_field_expansion_passed"]
        ),
        "H1_coincidence_warmup_schedule_passed": bool(warmup_report["coincidence_warmup_passed"]),
        "pre_attention_receptive_field": rf_report,
        "progress_ramped_coincidence_warmup": warmup_report,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[Plan 048 Receptive-Field & Warmup Profiler] all_passed={all_passed} | "
        f"max_future_leak={rf_report['max_future_leakage']:.2e} | "
        f"lag8/lag0={rf_report['final_layer_lag8_to_lag0_ratio']:.4f} | "
        f"tau0_penalty={warmup_report['early_tau0_ce_penalty_nats']:.2e} | "
        f"tau1_margin_gain={warmup_report['terminal_tau1_runnerup_margin_gain_nats']:.4f} nats"
    )


if __name__ == "__main__":
    main()
