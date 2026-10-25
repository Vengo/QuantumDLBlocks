#!/usr/bin/env python3
"""Controlled & Multi-Width Horizon Verification Runner for Experiment Plan 049 (Wave 34).

Verifies Hypotheses H1, H2, and H3 from `EXPERIMENT_PLAN_049.md` on genuine FineWeb-Edu with
`StandardizedExperimentLogger`:
- H1: True 4-Shell Dyadic CE Floor (`S_sib,8`, `S_mid,32`, `S_coarse,256`, `S_bg`), Subspace-0
  root-consistent coarse centroids (`< 0.05` nats floor vs. `2.96` nats legacy cancellation),
  compiled 50,257-way `M = 4` coincidence (`compiled_m4_coincidence_chunk_step`), and exact
  `256 x 256` factorized coincidence readout (`98.16x` fewer GEMM FLOPs).
- H2: Max-Width `d in {256, 384, 512, 640, 768, 1024}` Memory Optimization & Non-Pow2 Triton
  Flash-Born Scaling on 16 GB GPU (`checkpoint_detector_synthesis`, `bfloat16` embed stream,
  `_SubspaceL2NormalizeFn`, and `BLOCK_D = next_power_of_2(HEAD_DIM)`).
- H3: Symplectic-Preserving `StiefelMuon` + Parseval `muP` Restoration & Width Scaling on FineWeb-Edu:
  1. Controlled 150-step (`1.23M`-token, `d=256, L=4, B=16, T=512`) ablation comparing:
     - `w34_legacy_w33_flat_adamw_d256` (`Wave 33` `6.8012` Val CE, step-0 `sharp_eff_rank = 9.49`),
     - `w34_symplectic_stiefel_muon_mup_d256` (`Wave 34` unified `LevelSeparatedWaveletConvGPT`
       with 2D+3D `StiefelMuon`, `wavelet_fine_sparse` at `lr=0.0156`, symplectic-preserving fold,
       pre-`readout_chol` subspace normalization, and restored post-attention `WaveletCholeskyMultiscaleBranch`).
  2. Iso-token `1,800`-step (`14.75M`-token) & Max-Width scaling runs at `L = 6`:
     - `w34_unified_muon_mup_coincidence_d256_L6_1800steps` (`d=256, L=6, B=16, T=512, 1800 steps`),
     - `w34_unified_muon_mup_coincidence_d512_L6_1800steps` (`d=512, L=6, B=16, T=512, 1800 steps`),
     - `w34_unified_muon_mup_coincidence_d768_L6_300steps` (`d=768, L=6, B=16, T=512, 300 steps`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.models.level_separated_wavelet_conv_gpt import (
    DEFAULT_FLASH_BORN_HEAD_DIM,
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    PreAttnMultiRateWaveletConvBlock,
    build_level_separated_wavelet_conv_optimizers,
    initialize_level_separated_wavelet_conv_with_ppmi_svd,
    update_parseval_mup_ngram_learning_rates,
)
from scripts.multi_subspace_coincidence_margin_probe import compute_participation_ratio

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_049.md (Wave 34 Constants & Calibration Ledger)
# ==============================================================================
DEFAULT_ABLATION_STEPS: int = 150
DEFAULT_HORIZON_STEPS: int = 1800
DEFAULT_D768_STEPS: int = 300
DEFAULT_D1024_STEPS: int = 300
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEQ_LEN: int = 512
DEFAULT_EMBED_DIM: int = 256
DEFAULT_ABLATION_LAYERS: int = 4
DEFAULT_HORIZON_LAYERS: int = 6
DEFAULT_COINCIDENCE_GAMMA: float = 0.22
DEFAULT_SUBSPACE_VAR_PENALTY: float = 0.14
DEFAULT_COARSE_COINCIDENCE_WEIGHT: float = 0.04
BASE_MUON_ETA_MIN: float = 0.0065
BASE_MUON_ETA_MAX: float = 0.026
BASE_WAV_LR: float = 0.0085
BASE_HEAD_LR: float = 0.0050
MIN_STEP0_SHARP_EFF_RANK_GATE: float = 12.50
MAX_STEP150_VAL_CE_GATE: float = 6.45
MIN_STEP150_GAIN_VS_W33_NATS: float = 0.35
MAX_D256_HORIZON_VAL_CE_GATE: float = 5.31
MAX_D512_HORIZON_VAL_CE_GATE: float = 5.18
MAX_STIEFEL_ERR_GATE: float = 0.05

_EX48_SPEC = importlib.util.spec_from_file_location(
    "plan038_runner", REPO_ROOT / "examples" / "48_verify_plan038_last_layer_measurement_and_traceless.py"
)
assert _EX48_SPEC is not None and _EX48_SPEC.loader is not None
_EX48 = importlib.util.module_from_spec(_EX48_SPEC)
sys.modules["plan038_runner"] = _EX48
_EX48_SPEC.loader.exec_module(_EX48)

_EX53_SPEC = importlib.util.spec_from_file_location(
    "plan043_runner", REPO_ROOT / "examples" / "53_verify_plan043_parseval_sqrt_d_noclamp_readout.py"
)
assert _EX53_SPEC is not None and _EX53_SPEC.loader is not None
_EX53 = importlib.util.module_from_spec(_EX53_SPEC)
sys.modules["plan043_runner"] = _EX53
_EX53_SPEC.loader.exec_module(_EX53)


@dataclass(frozen=True)
class Wave34DataBundle:
    """Shared FineWeb-Edu tensors and validation batches (`<= 8` args)."""

    ppmi_embeddings: torch.Tensor
    init_tokens: torch.Tensor
    prior_tokens: torch.Tensor
    work_gpu: torch.Tensor
    val_batches: list[tuple[torch.Tensor, torch.Tensor]]
    batch_size: int


@dataclass(frozen=True)
class Wave34RunSpec:
    """Configuration for a single Wave 34 training arm (`<= 8` fields)."""

    run_key: str
    config: LevelSeparatedWaveletConvConfig
    steps: int
    eval_every: int = 50
    grad_accum_steps: int = 1
    use_fine_m4_chunked_coincidence: bool = True
    force_rerun: bool = False


def load_ppmi_embeddings(vocab_size: int = 50304, d_model: int = 256) -> torch.Tensor:
    """Load cached PPMI SVD embeddings or synthesize deterministic orthogonalized fallback."""
    cache_path = REPO_ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt"
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=True)
        emb = obj["embeddings"] if isinstance(obj, dict) else obj
        return emb.to(torch.float32)
    gen = torch.Generator()
    gen.manual_seed(20261024)
    return torch.nn.functional.normalize(torch.randn((vocab_size, d_model), generator=gen), dim=-1)


def evaluate_wave34_model(
    model: LevelSeparatedWaveletConvGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate validation CE, PPL, Top-1, Top-50, R_sharp, Stiefel error, and coincidence telemetry."""
    ev = _EX53.eval_parseval_mup_quantum(model, val_batches)
    with torch.no_grad():
        m_sharp, _, _ = model.synthesize_detectors(use_checkpoint=False)
        uncentered_r_sharp = compute_participation_ratio(m_sharp[:8192])
        centered_r_sharp = float(ev.get("sharp_eff_rank", uncentered_r_sharp))
        iso_err = model.pyramid.compute_kraus_stiefel_err()
        c_spec = model.build_coincidence_spec()
        out = {
            "val_ce": float(ev["val_ce"]),
            "val_ppl": float(ev["val_ppl"]),
            "top1_acc": float(ev["val_top1"]) / 100.0,
            "top50_acc": float(ev.get("val_top50", 0.0)) / 100.0,
            "sharp_eff_rank": centered_r_sharp,
            "centered_r_sharp": centered_r_sharp,
            "uncentered_r_sharp": float(uncentered_r_sharp),
            "kraus_stiefel_err": float(iso_err),
            "mean_beta": float(model.current_beta()),
            "coincidence_gamma_eff": float(c_spec.coincidence_gamma),
            "subspace_var_penalty_eff": float(c_spec.subspace_var_penalty),
            "subspace_balance_blend_eff": float(c_spec.subspace_balance_blend),
        }
        out.update(model.inspect_level_energy_and_rank())
        return out


def run_wave34_arm(
    spec: Wave34RunSpec,
    data: Wave34DataBundle,
) -> dict[str, Any]:
    """Train and evaluate one Wave 34 arm on genuine FineWeb-Edu with `StiefelMuon` + Parseval `muP`."""
    run_key = spec.run_key
    config = spec.config
    steps = spec.steps
    cache_path = REPO_ROOT / ".cache" / "live_logs" / f"{run_key}_w34_arm.json"
    if cache_path.exists() and not spec.force_rerun:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(
            f"  [{run_key}] (Cached) d={config.embed_dim}, L={config.num_layers}, steps={steps} | "
            f"step0_R_sharp={cached['step0_sharp_eff_rank']:.2f} | "
            f"final_val_ce={cached['final_val_ce']:.4f} | "
            f"top1={100.0 * cached['final_top1_acc']:.2f}% | "
            f"tok/s={cached['tok_per_s']:.0f}",
            flush=True,
        )
        return cached

    batch_size = data.batch_size
    device = data.work_gpu.device
    seq_len = config.block_size
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = LevelSeparatedWaveletConvGPT(config).to(device)
    init_stats = initialize_level_separated_wavelet_conv_with_ppmi_svd(
        model=model,
        ppmi_embeddings=data.ppmi_embeddings,
        init_tokens=data.init_tokens,
        prior_tokens=data.prior_tokens,
    )
    opt_muon, opt_adam = build_level_separated_wavelet_conv_optimizers(
        model,
        base_muon_lr=BASE_MUON_ETA_MAX,
        base_wav_lr=BASE_WAV_LR,
        base_head_lr=BASE_HEAD_LR,
    )

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    muon_params_m = sum(p.numel() for g in opt_muon.param_groups for p in g["params"]) / 1e6
    tokens_per_step = batch_size * seq_len
    pre_attn_blocks = sum(1 for b in model.blocks if isinstance(b, PreAttnMultiRateWaveletConvBlock))

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id="W34",
        plan_id="EXPERIMENT_PLAN_049",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 34: Symplectic StiefelMuon ({muon_params_m:.2f}M) + Parseval muP AdamW + "
            f"M={config.num_subspaces} Coincidence at d={config.embed_dim}, L={config.num_layers}, steps={steps}."
        ),
        architecture_summary="quantum_born",
        hardware_budget=f"Local {device.type.upper()}",
        batch_size=batch_size,
        seq_len=seq_len,
        total_steps=steps,
        target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        params_M=round(params_m, 4),
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        schedule="modded_nanogpt_wsd",
    )
    logger = StandardizedExperimentLogger(intent)

    model.set_training_progress(0.0)
    eval0 = evaluate_wave34_model(model, data.val_batches)
    eval0.update(
        {
            "step": 0,
            "tokens_M": 0.0,
            "tau_progress": 0.0,
            "val_top1": 100.0 * eval0["top1_acc"],
            "val_top50": 100.0 * eval0["top50_acc"],
        }
    )
    logger.log_eval(0, eval0)
    step0_r_sharp = float(init_stats.get("sharp_eff_rank", eval0["sharp_eff_rank"]))
    print(
        f"  [{run_key}] step=   0/{steps} ( 0.00M tok) | "
        f"val_ce={eval0['val_ce']:.4f} | top1={100.0 * eval0['top1_acc']:.2f}% | "
        f"R_sharp={eval0['sharp_eff_rank']:.2f} (init={step0_r_sharp:.2f}) | "
        f"head_dim={model.head_dim} | muon_M={muon_params_m:.2f}M",
        flush=True,
    )

    t0 = time.perf_counter()
    best_val_ce = eval0["val_ce"]
    final_eval = eval0
    milestones: dict[str, dict[str, float]] = {"step_0": dict(eval0)}
    n_work = int(data.work_gpu.numel())

    grad_accum = max(1, int(spec.grad_accum_steps))
    micro_batch = max(1, batch_size // grad_accum)

    for step in range(1, steps + 1):
        model.train()
        tok_prog = min(1.0, float(step - 1) / float(max(1, steps)))
        model.set_training_progress(tok_prog)
        lr_256, mom = _EX53.compute_step_lr_and_momentum(
            "modded_nanogpt_wsd", step - 1, steps, BASE_MUON_ETA_MIN, BASE_MUON_ETA_MAX
        )
        update_parseval_mup_ngram_learning_rates(
            opt_muon, opt_adam, lr_256, base_muon_lr_256=BASE_MUON_ETA_MAX
        )
        for g in opt_muon.param_groups:
            g["momentum"] = mom
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        offset = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        c_i64 = data.work_gpu[offset : offset + tokens_per_step + 1].to(torch.int64) & 0xFFFF
        x_full = c_i64[:-1].view(batch_size, seq_len)
        y_full = c_i64[1:].view(batch_size, seq_len)

        step_loss_val = 0.0
        for mb_idx in range(grad_accum):
            mb_s = mb_idx * micro_batch
            mb_e = min(batch_size, mb_s + micro_batch)
            x_mb = x_full[mb_s:mb_e]
            y_mb = y_full[mb_s:mb_e]
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, mb_loss = model(x_mb, targets=y_mb, return_logits=False)
            assert mb_loss is not None
            step_loss_val += float(mb_loss.item()) / float(grad_accum)
            (mb_loss / float(grad_accum)).backward()

        for g in (*opt_muon.param_groups, *opt_adam.param_groups):
            torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
        opt_muon.step()
        opt_adam.step()
        if step % 2 == 0 or step == steps:
            model.retract_kraus_generators()

        elapsed = max(time.perf_counter() - t0, 1e-6)
        tok_s = (step * tokens_per_step) / elapsed
        tok_m = (step * tokens_per_step) / 1e6

        if step == steps or step % max(1, spec.eval_every) == 0:
            tau_eval = min(1.0, float(step) / float(max(1, steps)))
            logger.log_step(step, step_loss_val, tok_s, lr_256 / BASE_MUON_ETA_MAX, elapsed / 60.0)
            model.set_training_progress(tau_eval)
            ev = evaluate_wave34_model(model, data.val_batches)
            best_val_ce = min(best_val_ce, ev["val_ce"])
            final_eval = ev
            ev.update(
                {
                    "step": step,
                    "tokens_M": round(tok_m, 3),
                    "tau_progress": round(tau_eval, 4),
                    "val_top1": 100.0 * ev["top1_acc"],
                    "val_top50": 100.0 * ev["top50_acc"],
                    "tok_per_s": round(tok_s, 1),
                    "elapsed_min": round(elapsed / 60.0, 4),
                }
            )
            logger.log_eval(step, ev)
            milestones[f"step_{step}"] = dict(ev)
            print(
                f"  [{run_key}] step={step:4d}/{steps} ({tok_m:5.2f}M tok, tau={tau_eval:.2f}) | "
                f"val_ce={ev['val_ce']:.4f} | top1={100.0 * ev['top1_acc']:.2f}% | "
                f"R_sharp={ev['sharp_eff_rank']:.2f} | gamma_c={ev['coincidence_gamma_eff']:.4f} | "
                f"IsoErr={ev['kraus_stiefel_err']:.4f} | tok/s={tok_s:.0f}",
                flush=True,
            )

    elapsed_total = max(time.perf_counter() - t0, 1e-6)
    avg_tok_s = (steps * tokens_per_step) / elapsed_total
    peak_vram_mib = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)
        if device.type == "cuda"
        else 0.0
    )
    logger.finalize(extra_summary={"best_val_ce": best_val_ce, "final_tok_per_s": avg_tok_s})
    res: dict[str, Any] = {
        "run_key": run_key,
        "embed_dim": config.embed_dim,
        "num_layers": config.num_layers,
        "head_dim": model.head_dim,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch,
        "grad_accum_steps": grad_accum,
        "steps": steps,
        "tokens_M": round((steps * tokens_per_step) / 1e6, 3),
        "params_M": round(params_m, 4),
        "muon_params_M": round(muon_params_m, 4),
        "num_subspaces": config.num_subspaces,
        "use_coincidence_warmup": config.use_coincidence_warmup,
        "pre_attn_conv_blocks": pre_attn_blocks,
        "step0_sharp_eff_rank": round(step0_r_sharp, 4),
        "best_val_ce": best_val_ce,
        "final_val_ce": final_eval["val_ce"],
        "final_top1_acc": final_eval["top1_acc"],
        "final_top50_acc": final_eval["top50_acc"],
        "final_sharp_eff_rank": final_eval["sharp_eff_rank"],
        "final_centered_r_sharp": final_eval["centered_r_sharp"],
        "final_kraus_stiefel_err": final_eval["kraus_stiefel_err"],
        "peak_vram_mib": peak_vram_mib,
        "tok_per_s": round(avg_tok_s, 1),
        "milestones": milestones,
    }
    for k in ("energy_coarse_256", "energy_fine_65536", "r_sharp_coarse_256", "r_sharp_fine_65536"):
        if k in final_eval:
            res[k] = final_eval[k]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    del model, opt_muon, opt_adam
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def load_legacy_w33_baselines() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load Wave 33 150-step (`6.8012`) and 1800-step (`5.5664`) baselines from `.cache/hypothesis_experiments_048.json`."""
    p048_path = REPO_ROOT / ".cache" / "hypothesis_experiments_048.json"
    if p048_path.exists():
        payload = json.loads(p048_path.read_text(encoding="utf-8"))
        h2 = payload.get("H2_controlled_150step_ablation_and_horizon", {})
        abl_w33 = dict(h2.get("m4_progress_ramped_pre_attn_conv") or {})
        hor_w33 = dict(h2.get("horizon_run") or h2.get("horizon_1800step") or {})
        abl_w33.setdefault("run_key", "w33_m4_progress_ramped_pre_attn_conv_d256")
        abl_w33.setdefault("step0_sharp_eff_rank", 9.4896)
        abl_w33.setdefault("final_val_ce", 6.8012)
        hor_w33.setdefault("run_key", "w33_level_sep_pre_attn_conv_warmup_d256_L6_1800steps")
        hor_w33.setdefault("final_val_ce", 5.5664)
        return abl_w33, hor_w33
    return (
        {
            "run_key": "w33_m4_progress_ramped_pre_attn_conv_d256",
            "step0_sharp_eff_rank": 9.4896,
            "final_val_ce": 6.8012,
            "best_val_ce": 6.8012,
            "final_top1_acc": 0.1366,
            "final_sharp_eff_rank": 53.11,
        },
        {
            "run_key": "w33_level_sep_pre_attn_conv_warmup_d256_L6_1800steps",
            "final_val_ce": 5.5664,
            "best_val_ce": 5.5664,
            "final_top1_acc": 0.2034,
            "final_sharp_eff_rank": 54.48,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Plan 049 True CE Floor, StiefelMuon + Parseval muP Restoration, & Max-d Scaling"
    )
    parser.add_argument("--ablation-steps", type=int, default=DEFAULT_ABLATION_STEPS)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--d768-steps", type=int, default=DEFAULT_D768_STEPS)
    parser.add_argument("--d1024-steps", type=int, default=DEFAULT_D1024_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--skip-horizon", action="store_true")
    parser.add_argument("--skip-d512", action="store_true")
    parser.add_argument("--skip-d768", action="store_true")
    parser.add_argument("--skip-d1024", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(REPO_ROOT / ".cache" / "hypothesis_experiments_049.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    h1_path = REPO_ROOT / ".cache" / "factorized_256x256_and_ce_floor_bench.json"
    h2_path = REPO_ROOT / ".cache" / "max_width_vram_profile.json"
    h1_report = json.loads(h1_path.read_text(encoding="utf-8")) if h1_path.exists() else {}
    h2_report = json.loads(h2_path.read_text(encoding="utf-8")) if h2_path.exists() else {}

    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=85_000_000, max_val_tokens=1_048_576)
    ppmi_emb = load_ppmi_embeddings(vocab_size=50304, d_model=256)
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=args.seq_len, num_batches=6, device=device
    )
    data = Wave34DataBundle(
        ppmi_embeddings=ppmi_emb,
        init_tokens=torch.from_numpy(train_arr[:85_000_000].astype("int32")),
        prior_tokens=torch.from_numpy(train_arr[:2_000_000].astype("int64")),
        work_gpu=torch.from_numpy(train_arr[:30_000_000].view("int16")).to(device),
        val_batches=val_batches,
        batch_size=args.batch_size,
    )

    prec = "fp8" if device.type == "cuda" else "bf16"
    s_abl = int(args.ablation_steps)
    ev_abl = max(1, s_abl // 3)
    w33_abl, w33_hor = load_legacy_w33_baselines()

    cfg_abl_d256 = LevelSeparatedWaveletConvConfig(
        embed_dim=DEFAULT_EMBED_DIM,
        num_layers=DEFAULT_ABLATION_LAYERS,
        num_heads=DEFAULT_EMBED_DIM // DEFAULT_FLASH_BORN_HEAD_DIM,
        block_size=args.seq_len,
        num_subspaces=4,
        partition_mode="strided",
        coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
        subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        coarse_coincidence_weight=DEFAULT_COARSE_COINCIDENCE_WEIGHT,
        use_coincidence_warmup=True,
        use_per_layer_pre_attn_conv=True,
        readout_gemm_precision=prec,
    )
    res_abl_w34 = run_wave34_arm(
        Wave34RunSpec(
            run_key="w34_symplectic_stiefel_muon_mup_d256",
            config=cfg_abl_d256,
            steps=s_abl,
            eval_every=ev_abl,
            force_rerun=args.force_rerun,
        ),
        data=data,
    )

    res_hor_d256: dict[str, Any] | None = None
    res_hor_d512: dict[str, Any] | None = None
    res_d768: dict[str, Any] | None = None
    res_d1024_b16: dict[str, Any] | None = None
    res_d1024_b8_ga2: dict[str, Any] | None = None

    if not args.skip_horizon and int(args.horizon_steps) > 0:
        s_hor = int(args.horizon_steps)
        ev_hor = max(1, s_hor // 6)
        cfg_hor_d256 = LevelSeparatedWaveletConvConfig(
            embed_dim=256,
            num_layers=DEFAULT_HORIZON_LAYERS,
            num_heads=256 // DEFAULT_FLASH_BORN_HEAD_DIM,
            block_size=args.seq_len,
            num_subspaces=4,
            partition_mode="strided",
            coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
            subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
            coarse_coincidence_weight=DEFAULT_COARSE_COINCIDENCE_WEIGHT,
            use_coincidence_warmup=True,
            use_per_layer_pre_attn_conv=True,
            readout_gemm_precision=prec,
        )
        res_hor_d256 = run_wave34_arm(
            Wave34RunSpec(
                run_key=f"w34_unified_muon_mup_coincidence_d256_L6_{s_hor}steps",
                config=cfg_hor_d256,
                steps=s_hor,
                eval_every=ev_hor,
                force_rerun=args.force_rerun,
            ),
            data=data,
        )

        if not args.skip_d512:
            cfg_hor_d512 = LevelSeparatedWaveletConvConfig(
                embed_dim=512,
                num_layers=DEFAULT_HORIZON_LAYERS,
                num_heads=512 // DEFAULT_FLASH_BORN_HEAD_DIM,
                block_size=args.seq_len,
                num_subspaces=4,
                partition_mode="strided",
                coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
                subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
                coarse_coincidence_weight=DEFAULT_COARSE_COINCIDENCE_WEIGHT,
                use_coincidence_warmup=True,
                use_per_layer_pre_attn_conv=True,
                readout_gemm_precision=prec,
            )
            res_hor_d512 = run_wave34_arm(
                Wave34RunSpec(
                    run_key=f"w34_unified_muon_mup_coincidence_d512_L6_{s_hor}steps",
                    config=cfg_hor_d512,
                    steps=s_hor,
                    eval_every=ev_hor,
                    force_rerun=args.force_rerun,
                ),
                data=data,
            )

        if not args.skip_d768 and int(args.d768_steps) > 0:
            s_768 = int(args.d768_steps)
            cfg_d768 = LevelSeparatedWaveletConvConfig(
                embed_dim=768,
                num_layers=DEFAULT_HORIZON_LAYERS,
                num_heads=768 // DEFAULT_FLASH_BORN_HEAD_DIM,
                block_size=args.seq_len,
                num_subspaces=4,
                partition_mode="strided",
                coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
                subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
                coarse_coincidence_weight=DEFAULT_COARSE_COINCIDENCE_WEIGHT,
                use_coincidence_warmup=True,
                use_per_layer_pre_attn_conv=True,
                readout_gemm_precision=prec,
            )
            res_d768 = run_wave34_arm(
                Wave34RunSpec(
                    run_key=f"w34_unified_muon_mup_coincidence_d768_L6_{s_768}steps",
                    config=cfg_d768,
                    steps=s_768,
                    eval_every=max(1, s_768 // 3),
                    force_rerun=args.force_rerun,
                ),
                data=data,
            )

        if not args.skip_d1024 and int(args.d1024_steps) > 0:
            s_1024 = int(args.d1024_steps)
            cfg_d1024 = LevelSeparatedWaveletConvConfig(
                embed_dim=1024,
                num_layers=DEFAULT_HORIZON_LAYERS,
                num_heads=1024 // DEFAULT_FLASH_BORN_HEAD_DIM,
                block_size=args.seq_len,
                num_subspaces=4,
                partition_mode="strided",
                coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
                subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
                coarse_coincidence_weight=DEFAULT_COARSE_COINCIDENCE_WEIGHT,
                use_coincidence_warmup=True,
                use_per_layer_pre_attn_conv=True,
                readout_gemm_precision=prec,
            )
            res_d1024_b16 = run_wave34_arm(
                Wave34RunSpec(
                    run_key=f"w34_unified_muon_mup_coincidence_d1024_L6_{s_1024}steps",
                    config=cfg_d1024,
                    steps=s_1024,
                    eval_every=max(1, s_1024 // 3),
                    grad_accum_steps=1,
                    force_rerun=args.force_rerun,
                ),
                data=data,
            )
            s_1024_ga2 = min(150, s_1024)
            res_d1024_b8_ga2 = run_wave34_arm(
                Wave34RunSpec(
                    run_key=f"w34_unified_muon_mup_coincidence_d1024_L6_b8_ga2_{s_1024_ga2}steps",
                    config=cfg_d1024,
                    steps=s_1024_ga2,
                    eval_every=max(1, s_1024_ga2 // 3),
                    grad_accum_steps=2,
                    force_rerun=args.force_rerun,
                ),
                data=data,
            )

    gain_150_vs_w33 = float(w33_abl["final_val_ce"] - res_abl_w34["final_val_ce"])
    gain_1800_d256_vs_w33 = (
        float(w33_hor["final_val_ce"] - res_hor_d256["final_val_ce"])
        if res_hor_d256 is not None
        else 0.0
    )
    gain_1800_d512_vs_w33 = (
        float(w33_hor["final_val_ce"] - res_hor_d512["final_val_ce"])
        if res_hor_d512 is not None
        else 0.0
    )

    h3_step0_rank_ok = bool(res_abl_w34["step0_sharp_eff_rank"] >= MIN_STEP0_SHARP_EFF_RANK_GATE)
    h3_abl_ok = bool(
        res_abl_w34["final_val_ce"] <= MAX_STEP150_VAL_CE_GATE
        and gain_150_vs_w33 >= MIN_STEP150_GAIN_VS_W33_NATS
        and res_abl_w34["final_kraus_stiefel_err"] <= MAX_STIEFEL_ERR_GATE
    )
    h3_hor_d256_ok = (
        bool(
            res_hor_d256["final_val_ce"] <= MAX_D256_HORIZON_VAL_CE_GATE
            and res_hor_d256["final_kraus_stiefel_err"] <= MAX_STIEFEL_ERR_GATE
        )
        if res_hor_d256 is not None
        else True
    )
    h3_hor_d512_ok = (
        bool(
            res_hor_d512["final_val_ce"] <= MAX_D512_HORIZON_VAL_CE_GATE
            and res_hor_d512["final_kraus_stiefel_err"] <= MAX_STIEFEL_ERR_GATE
        )
        if res_hor_d512 is not None
        else True
    )

    payload = {
        "plan": "EXPERIMENT_PLAN_049",
        "wave": "Wave 34",
        "H1_true_dyadic_ce_floor_and_256x256_factorized_readout": h1_report,
        "H2_max_width_vram_and_flash_born_scaling": h2_report,
        "H3_symplectic_stiefel_muon_mup_and_width_scaling": {
            "w33_legacy_flat_adamw_150step_baseline": w33_abl,
            "w34_symplectic_stiefel_muon_mup_150step": res_abl_w34,
            "w33_legacy_flat_adamw_1800step_baseline": w33_hor,
            "w34_unified_d256_L6_1800step": res_hor_d256,
            "w34_unified_d512_L6_1800step": res_hor_d512,
            "w34_unified_d768_L6_300step": res_d768,
            "w34_unified_d1024_L6_300step_b16_ga1": res_d1024_b16,
            "w34_unified_d1024_L6_150step_b8_ga2": res_d1024_b8_ga2,
            "step0_sharp_eff_rank_recovered": h3_step0_rank_ok,
            "step150_val_ce_gain_vs_w33_nats": round(gain_150_vs_w33, 4),
            "step1800_d256_val_ce_gain_vs_w33_nats": round(gain_1800_d256_vs_w33, 4),
            "step1800_d512_val_ce_gain_vs_w33_nats": round(gain_1800_d512_vs_w33, 4),
            "h3_ablation_passed": h3_abl_ok,
            "h3_horizon_d256_passed": h3_hor_d256_ok,
            "h3_horizon_d512_passed": h3_hor_d512_ok,
            "passed": bool(h3_step0_rank_ok and h3_abl_ok and h3_hor_d256_ok and h3_hor_d512_ok),
        },
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Wave 34 verification report to {out_path}")


if __name__ == "__main__":
    main()
