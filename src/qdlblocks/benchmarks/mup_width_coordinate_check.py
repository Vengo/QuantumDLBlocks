"""Automated MuP Width Coordinate Check for Quantum Baton Mixer Models (`Wave 77` / `EXPERIMENT_PLAN_097.md`).

Provenance & Mathematical Rationale
-----------------------------------
Promoted from the `Wave 76` `Nice-to-Have` backlog (`scripts/TOOLING_BACKLOG.md`) as **Step 1** of
`Wave 77` (`EXPERIMENT_PLAN_097.md`, Hypothesis H1), obeying Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`:

- **When to Use It**:
  Before scaling Hilbert dimension $d$ (`64 -> 128 -> 256 -> 512`) or launching multi-tens-of-millions
  token runs, execute a 20-step coordinate check across widths on real FineWeb tokens to verify that
  feature updates $\\text{RMS}(\\Delta \\mathbf{h}^{(\\ell)})$ and pre-softcap readout logits
  $\\text{Std}(\\mathbf{x}_{\\text{final}} W_{\\text{lm}}^\\top)$ remain $\\Theta(1)$ rather than
  blowing up or vanishing with width $D = 2d$.

- **Why `MUP_STIEFEL_MUON` Achieves Width Invariance While `NAIVE_SP_ADAMW` Drifts**:
  1. Under `NAIVE_SP_ADAMW` (constant entrywise LR $\\eta$ across widths), an `AdamW` step on
     `lm_head` produces $\\Delta \\text{logit} \\sim \\Theta(D \\cdot \\eta)$ (`8x` blowup from
     $d=64 \\to 512$), and trunk matrix updates have spectral norm $\\Theta(\\eta \\sqrt{D})$.
  2. Under `MUP_STIEFEL_MUON`, `PolarExpress` produces orthogonal updates with unit spectral norm
     $\\|U\\|_{\\text{op}} = 1$ (so $\\text{RMS}(\\Delta(Wx)) = \\Theta(\\eta_{\\text{muon}})$ for
     unit-RMS inputs independent of $D$), while scaling `lm_head` `Adam` LR as $\\eta_{\\text{head}}(D)
     = \\eta_0 \\cdot (D_0 / D)$ keeps $\\Delta \\text{logit} = \\Theta(1)$ across all widths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .architecture_mechanistic_probes import evaluate_quick_ce_and_top1
from .architecture_xray_diagnostics import VALID_FINEWEB_VOCAB_SIZE, load_fineweb_token_slice
from .wave77_baton_ladder_models import (
    NAIVE_SP_FIXED_ADAM_LR,
    QuantumBatonMixerLM,
    Wave77DiagnosticAdapter,
    Wave77MixerConfig,
    build_wave77_mup_optimizers,
)

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_097.md Section 3)
# ==============================================================================
DEFAULT_COORD_CHECK_WIDTHS: tuple[int, ...] = (64, 128, 256, 512)
DEFAULT_COORD_CHECK_STEPS: int = 20
DEFAULT_COORD_CHECK_BATCH_SIZE: int = 16
DEFAULT_COORD_CHECK_SEQ_LEN: int = 256
DEFAULT_COORD_CHECK_NUM_LAYERS: int = 2
COORD_STABILITY_MAX_SPREAD_PASS: float = 1.65


class MuPParameterizationMode(str, Enum):
    """Parameterization & optimizer scaling regime compared in the coordinate check (Rule 3)."""

    NAIVE_SP_ADAMW = "naive_sp_adamw"
    MUP_STIEFEL_MUON = "mup_stiefel_muon"


@dataclass(frozen=True)
class MuPCoordinateCheckConfig:
    """Configuration for `run_mup_width_coordinate_check` (`<= 8` fields, Rule 2)."""

    widths: tuple[int, ...] = DEFAULT_COORD_CHECK_WIDTHS
    steps: int = DEFAULT_COORD_CHECK_STEPS
    batch_size: int = DEFAULT_COORD_CHECK_BATCH_SIZE
    seq_len: int = DEFAULT_COORD_CHECK_SEQ_LEN
    num_layers: int = DEFAULT_COORD_CHECK_NUM_LAYERS
    eval_tokens: int = 4096
    seed: int = 2027


@torch.no_grad()
def _capture_probe_coordinates(
    model: QuantumBatonMixerLM,
    x_probe: Tensor,
) -> dict[str, Any]:
    """Capture layerwise hidden tensors, activation RMS, and pre-softcap `lm_head` logit std (`<= 8` args)."""
    was_train = model.training
    model.eval()
    amp_on = x_probe.is_cuda
    with torch.amp.autocast(
        device_type=x_probe.device.type, dtype=torch.bfloat16, enabled=amp_on
    ):
        x_final, copy_lo, _, _, layer_acts = model.encode_trunk_features(x_probe)
        raw_lm_logits = F.linear(x_final, model.lm_head.weight).float()
        full_soft_logits = model.compute_logits(x_probe, x_final, copy_lo).float()
    if was_train:
        model.train()

    layer_rms = [
        float(act.detach().float().pow(2).mean().sqrt().item()) for act in layer_acts
    ]
    det_acts = [act.detach().float().cpu() for act in layer_acts]
    raw_std = float(raw_lm_logits.std(dim=-1).mean().item())
    raw_abs_max = float(raw_lm_logits.abs().amax(dim=-1).mean().item())
    soft_span = float(
        (full_soft_logits.amax(dim=-1) - full_soft_logits.median(dim=-1).values).mean().item()
    )
    return {
        "layer_acts_cpu": det_acts,
        "layer_rms": layer_rms,
        "mean_layer_rms": float(sum(layer_rms) / max(1, len(layer_rms))),
        "raw_lm_logit_std": raw_std,
        "raw_lm_logit_abs_max": raw_abs_max,
        "softcapped_logit_span": soft_span,
    }


def _run_single_width_mode_check(
    d_complex: int,
    mode: MuPParameterizationMode,
    cfg: MuPCoordinateCheckConfig,
    train_tokens: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Run `cfg.steps` updates at width `d_complex` under `mode` and measure coordinate drift (`<= 8` args)."""
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    m_cfg = Wave77MixerConfig(
        vocab_size=VALID_FINEWEB_VOCAB_SIZE,
        complex_dim=int(d_complex),
        num_layers=cfg.num_layers,
        mixer_mode="quantum_interference_mlp",
    )
    model = QuantumBatonMixerLM(m_cfg).to(device)
    suite = "adamw_sp" if mode == MuPParameterizationMode.NAIVE_SP_ADAMW else "mup_stiefel_muon"
    opts = build_wave77_mup_optimizers(model, suite_mode=suite, warmup_steps=max(4, cfg.steps // 4))

    adapter = Wave77DiagnosticAdapter(
        key=f"coord_{mode.value}_d{d_complex}",
        label=f"{mode.value} (d={d_complex}, D={2 * d_complex})",
        model=model,
        optimizers=opts,
        device=device,
        embed_dim=model.real_dim,
        num_layers=cfg.num_layers,
    )

    x_probe = x_val[:2].to(device)
    init_weights: dict[str, Tensor] = {
        name: p.detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if p.ndim == 2 and ("qkv_proj" in name or "mlp_up" in name)
    }
    snap0 = _capture_probe_coordinates(model, x_probe)
    step0_ce, _ = evaluate_quick_ce_and_top1(adapter, x_val, y_val, micro_batch=4)

    tok_per_step = cfg.batch_size * cfg.seq_len
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        off = (step - 1) * tok_per_step
        sl = train_tokens[off : off + tok_per_step + 1]
        xb = sl[:-1].view(cfg.batch_size, cfg.seq_len).to(device)
        yb = sl[1:].view(cfg.batch_size, cfg.seq_len).to(device)

        # Constant LR schedule during coordinate check so update scaling reflects pure parameterization
        model.train()
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        amp_on = xb.is_cuda
        with torch.amp.autocast(device_type=xb.device.type, dtype=torch.bfloat16, enabled=amp_on):
            _, loss = model(xb, targets=yb, ingest_online=True, return_logits=False)
        assert loss is not None
        loss.backward()
        for opt in opts:
            opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1.0e-6, time.perf_counter() - t0)

    snap_end = _capture_probe_coordinates(model, x_probe)
    end_ce, end_top1 = evaluate_quick_ce_and_top1(adapter, x_val, y_val, micro_batch=4)

    delta_act_rms_per_layer: list[float] = []
    for a0, aT in zip(snap0["layer_acts_cpu"], snap_end["layer_acts_cpu"]):
        rms_diff = float((aT - a0).pow(2).mean().sqrt().item())
        delta_act_rms_per_layer.append(round(rms_diff, 5))

    rel_updates: list[float] = []
    for name, p in model.named_parameters():
        if name in init_weights:
            w0 = init_weights[name]
            wT = p.detach().float().cpu()
            rel = float(torch.norm(wT - w0, p="fro").item()) / max(
                1.0e-6, float(torch.norm(w0, p="fro").item())
            )
            rel_updates.append(rel)

    p_info = adapter.parameter_breakdown()
    head_lr_used = (
        NAIVE_SP_FIXED_ADAM_LR
        if mode == MuPParameterizationMode.NAIVE_SP_ADAMW
        else float(opts[1].param_groups[0]["lr"])
    )
    return {
        "mode": mode.value,
        "complex_dim": int(d_complex),
        "real_dim": int(2 * d_complex),
        "total_params": p_info["total_params"],
        "trunk_params": p_info["trunk_params"],
        "head_lr": round(head_lr_used, 6),
        "tok_per_s": round(float(cfg.steps * tok_per_step) / elapsed, 1),
        "step0_val_ce": round(step0_ce, 5),
        "stepN_val_ce": round(end_ce, 5),
        "delta_val_ce": round(end_ce - step0_ce, 5),
        "stepN_top1_pct": round(end_top1 * 100.0, 3),
        "step0_mean_layer_rms": round(snap0["mean_layer_rms"], 5),
        "stepN_mean_layer_rms": round(snap_end["mean_layer_rms"], 5),
        "stepN_delta_act_rms_mean": round(
            float(sum(delta_act_rms_per_layer)) / max(1, len(delta_act_rms_per_layer)), 5
        ),
        "stepN_delta_act_rms_by_layer": delta_act_rms_per_layer,
        "stepN_raw_lm_logit_std": round(snap_end["raw_lm_logit_std"], 5),
        "stepN_raw_lm_logit_abs_max": round(snap_end["raw_lm_logit_abs_max"], 5),
        "stepN_softcapped_logit_span": round(snap_end["softcapped_logit_span"], 5),
        "mean_rel_trunk_fro_update": round(
            float(sum(rel_updates)) / max(1, len(rel_updates)), 5
        ),
    }


def _summarize_mode_width_spread(records: list[dict[str, Any]]) -> dict[str, float | bool]:
    """Compute cross-width max/min drift ratios for coordinate-stability gating (`<= 8` args)."""
    logit_stds = [max(1.0e-6, float(r["stepN_raw_lm_logit_std"])) for r in records]
    delta_acts = [max(1.0e-6, float(r["stepN_delta_act_rms_mean"])) for r in records]
    act_rms = [max(1.0e-6, float(r["stepN_mean_layer_rms"])) for r in records]

    logit_spread = max(logit_stds) / min(logit_stds)
    delta_act_spread = max(delta_acts) / min(delta_acts)
    act_rms_spread = max(act_rms) / min(act_rms)
    return {
        "raw_lm_logit_std_max_min_ratio": round(logit_spread, 4),
        "delta_act_rms_max_min_ratio": round(delta_act_spread, 4),
        "layer_act_rms_max_min_ratio": round(act_rms_spread, 4),
        "passes_mup_coordinate_gate": bool(
            logit_spread <= COORD_STABILITY_MAX_SPREAD_PASS
            and act_rms_spread <= COORD_STABILITY_MAX_SPREAD_PASS
        ),
    }


def run_mup_width_coordinate_check(
    cfg: MuPCoordinateCheckConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Sweep `cfg.widths` under `NAIVE_SP_ADAMW` vs. `MUP_STIEFEL_MUON` on real FineWeb tokens."""
    t_all = time.perf_counter()
    tok_needed = cfg.steps * cfg.batch_size * cfg.seq_len + cfg.seq_len + 8
    train_tokens = load_fineweb_token_slice("fineweb_train_000001.bin", num_tokens=tok_needed)

    n_val_seq = max(2, cfg.eval_tokens // cfg.seq_len)
    val_tokens = load_fineweb_token_slice(
        "fineweb_val_000000.bin", num_tokens=n_val_seq * cfg.seq_len + 1
    )
    x_val = val_tokens[:-1].view(n_val_seq, cfg.seq_len)
    y_val = val_tokens[1:].view(n_val_seq, cfg.seq_len)

    by_mode: dict[str, list[dict[str, Any]]] = {}
    spread_summary: dict[str, dict[str, float | bool]] = {}

    for mode in (MuPParameterizationMode.NAIVE_SP_ADAMW, MuPParameterizationMode.MUP_STIEFEL_MUON):
        mode_rows: list[dict[str, Any]] = []
        for d_w in cfg.widths:
            row = _run_single_width_mode_check(
                d_complex=int(d_w),
                mode=mode,
                cfg=cfg,
                train_tokens=train_tokens,
                x_val=x_val,
                y_val=y_val,
                device=device,
            )
            mode_rows.append(row)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        by_mode[mode.value] = mode_rows
        spread_summary[mode.value] = _summarize_mode_width_spread(mode_rows)

    return {
        "wave_id": "W77",
        "plan_id": "EXPERIMENT_PLAN_097",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "widths_complex": list(cfg.widths),
        "widths_real": [2 * int(w) for w in cfg.widths],
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "wall_sec": round(time.perf_counter() - t_all, 2),
        "spread_summary": spread_summary,
        "records_by_mode": by_mode,
    }
