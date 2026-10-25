#!/usr/bin/env python3
"""Controlled verification runner for Experiment Plan 047 (Wave 32).

Compares three controlled arms on genuine FineWeb-Edu under identical token streams,
optimizer schedules, and StandardizedExperimentLogger telemetry:
1. `w32_m1_single_sphere_baseline_d256` (M=1 global L2 sphere baseline),
2. `w32_m4_strided_subspace_coincidence_d256` (M=4 strided round-robin x[..., m::M] + coincidence readout),
3. `w32_m4_level_separated_wavelet_conv_d256` (M=4 explicit WaveletScaleState level-separated tensors +
   pre-attention MultiRateWavefunctionConvBank 1D strided wavefunction convolutions + coincidence readout).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    build_level_separated_wavelet_conv_optimizers,
    initialize_level_separated_wavelet_conv_with_ppmi_svd,
)
from qdlblocks.models.parseval_mup_subspace_coincidence_gpt import (
    ParsevalMuPSubspaceCoincidenceConfig,
    ParsevalMuPSubspaceCoincidenceGPT,
    build_subspace_coincidence_optimizers,
    initialize_subspace_coincidence_with_ppmi_svd,
)
from scripts.multi_subspace_coincidence_margin_probe import compute_participation_ratio
from scripts.subspace_specialization_mutual_information_probe import (
    SubspaceSpecializationProbeConfig,
    audit_cross_level_norm_cannibalization,
    audit_svd_spectrum_and_mutual_information,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_047.md
# ==============================================================================
DEFAULT_STEPS: int = 150
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEQ_LEN: int = 512
DEFAULT_EMBED_DIM: int = 256
DEFAULT_NUM_LAYERS: int = 4
DEFAULT_COINCIDENCE_GAMMA: float = 0.22
DEFAULT_SUBSPACE_VAR_PENALTY: float = 0.14
DEFAULT_COARSE_COINCIDENCE_WEIGHT: float = 0.16
BASE_MUON_ETA_MIN: float = 0.0065
BASE_MUON_ETA_MAX: float = 0.026
BASE_WAV_LR: float = 0.0085
BASE_HEAD_LR: float = 0.0050
REFERENCE_PROGRESS_TOKENS: float = 29_491_200.0
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
class VerificationDataBundle:
    """Shared FineWeb-Edu tensors and validation batches (`<= 8` args)."""

    ppmi_embeddings: torch.Tensor
    init_tokens: torch.Tensor
    prior_tokens: torch.Tensor
    work_gpu: torch.Tensor
    val_batches: list[tuple[torch.Tensor, torch.Tensor]]
    steps: int
    batch_size: int


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


def evaluate_wave32_model(
    model: ParsevalMuPSubspaceCoincidenceGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate validation CE, PPL, Top-1, Top-50, R_sharp, Stiefel error, and per-level energy."""
    ev = _EX53.eval_parseval_mup_quantum(model, val_batches)
    with torch.no_grad():
        m_sharp, _, _ = model.synthesize_detectors()
        r_sharp = compute_participation_ratio(m_sharp[:8192])
        iso_err = model.pyramid.compute_kraus_stiefel_err()
        out = {
            "val_ce": float(ev["val_ce"]),
            "val_ppl": float(ev["val_ppl"]),
            "top1_acc": float(ev["val_top1"]) / 100.0,
            "top50_acc": float(ev.get("val_top50", 0.0)) / 100.0,
            "sharp_eff_rank": float(r_sharp),
            "centered_r_sharp": float(ev.get("sharp_eff_rank", r_sharp)),
            "kraus_stiefel_err": float(iso_err),
            "mean_beta": float(model.current_beta()),
        }
        if isinstance(model, LevelSeparatedWaveletConvGPT):
            lvl_diag = model.inspect_level_energy_and_rank()
            out.update(lvl_diag)
        return out


def run_single_arm(
    run_key: str,
    config: ParsevalMuPSubspaceCoincidenceConfig,
    data: VerificationDataBundle,
    use_level_separated_conv: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one controlled Wave 32 arm on genuine FineWeb-Edu."""
    steps = data.steps
    batch_size = data.batch_size
    device = data.work_gpu.device
    seq_len = config.block_size
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    if use_level_separated_conv:
        assert isinstance(config, LevelSeparatedWaveletConvConfig)
        model: ParsevalMuPSubspaceCoincidenceGPT = LevelSeparatedWaveletConvGPT(config).to(device)
        initialize_level_separated_wavelet_conv_with_ppmi_svd(
            model=model,
            ppmi_embeddings=data.ppmi_embeddings,
            init_tokens=data.init_tokens,
            prior_tokens=data.prior_tokens,
        )
        opt_muon, opt_adam = build_level_separated_wavelet_conv_optimizers(
            model, base_muon_lr=BASE_MUON_ETA_MAX, base_wav_lr=BASE_WAV_LR, base_head_lr=BASE_HEAD_LR
        )
    else:
        model = ParsevalMuPSubspaceCoincidenceGPT(config).to(device)
        initialize_subspace_coincidence_with_ppmi_svd(
            model=model,
            ppmi_embeddings=data.ppmi_embeddings,
            init_tokens=data.init_tokens,
            prior_tokens=data.prior_tokens,
        )
        opt_muon, opt_adam = build_subspace_coincidence_optimizers(
            model, base_muon_lr=BASE_MUON_ETA_MAX, base_wav_lr=BASE_WAV_LR, base_head_lr=BASE_HEAD_LR
        )

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    tokens_per_step = batch_size * seq_len

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id="W32",
        plan_id="EXPERIMENT_PLAN_047",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 32 controlled verification: M={config.num_subspaces}, mode={config.partition_mode}, "
            f"level_sep_conv={use_level_separated_conv} at d={config.embed_dim}, L={config.num_layers}."
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

    eval0 = evaluate_wave32_model(model, data.val_batches)
    eval0.update(
        {
            "step": 0,
            "tokens_M": 0.0,
            "val_top1": 100.0 * eval0["top1_acc"],
            "val_top50": 100.0 * eval0["top50_acc"],
        }
    )
    logger.log_eval(0, eval0)

    t0 = time.perf_counter()
    best_val_ce = eval0["val_ce"]
    final_eval = eval0
    n_work = int(data.work_gpu.numel())

    for step in range(1, steps + 1):
        model.train()
        tok_prog = min(1.0, float((step - 1) * tokens_per_step) / REFERENCE_PROGRESS_TOKENS)
        model.set_training_progress(tok_prog)
        lr_256, mom = _EX53.compute_step_lr_and_momentum(
            "modded_nanogpt_wsd", step - 1, steps, BASE_MUON_ETA_MIN, BASE_MUON_ETA_MAX
        )
        _EX53.apply_scaled_group_lrs(opt_muon, opt_adam, lr_256, config.embed_dim, mom)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        offset = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        c_i64 = data.work_gpu[offset : offset + tokens_per_step + 1].to(torch.int64) & 0xFFFF
        x = c_i64[:-1].view(batch_size, seq_len)
        y = c_i64[1:].view(batch_size, seq_len)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, targets=y, return_logits=False)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adam.step()
        if step % 4 == 0 or step == steps:
            model.retract_kraus_generators()

        elapsed = max(time.perf_counter() - t0, 1e-6)
        tok_s = (step * tokens_per_step) / elapsed
        tok_m = (step * tokens_per_step) / 1e6

        if step == steps or step % max(1, steps // 3) == 0:
            logger.log_step(step, float(loss.item()), tok_s, lr_256 / BASE_MUON_ETA_MAX, elapsed / 60.0)
            model.set_training_progress(min(1.0, float(step * tokens_per_step) / REFERENCE_PROGRESS_TOKENS))
            ev = evaluate_wave32_model(model, data.val_batches)
            best_val_ce = min(best_val_ce, ev["val_ce"])
            final_eval = ev
            ev.update(
                {
                    "step": step,
                    "tokens_M": round(tok_m, 3),
                    "val_top1": 100.0 * ev["top1_acc"],
                    "val_top50": 100.0 * ev["top50_acc"],
                    "tok_per_s": round(tok_s, 1),
                    "elapsed_min": round(elapsed / 60.0, 4),
                }
            )
            logger.log_eval(step, ev)
            print(
                f"  [{run_key}] step={step:4d}/{steps} ({tok_m:5.2f}M tok) | "
                f"val_ce={ev['val_ce']:.4f} | top1={100.0 * ev['top1_acc']:.2f}% | "
                f"R_sharp={ev['sharp_eff_rank']:.2f} | IsoErr={ev['kraus_stiefel_err']:.4f} | tok/s={tok_s:.0f}",
                flush=True,
            )

    elapsed_total = max(time.perf_counter() - t0, 1e-6)
    avg_tok_s = (steps * tokens_per_step) / elapsed_total
    logger.finalize(extra_summary={"best_val_ce": best_val_ce, "final_tok_per_s": avg_tok_s})
    res = {
        "run_key": run_key,
        "num_subspaces": config.num_subspaces,
        "partition_mode": config.partition_mode,
        "use_level_separated_conv": use_level_separated_conv,
        "best_val_ce": best_val_ce,
        "final_val_ce": final_eval["val_ce"],
        "final_top1_acc": final_eval["top1_acc"],
        "final_top50_acc": final_eval["top50_acc"],
        "final_sharp_eff_rank": final_eval["sharp_eff_rank"],
        "final_centered_r_sharp": final_eval["centered_r_sharp"],
        "final_kraus_stiefel_err": final_eval["kraus_stiefel_err"],
        "tok_per_s": avg_tok_s,
    }
    for k in ("energy_coarse_256", "energy_fine_65536", "r_sharp_coarse_256", "r_sharp_fine_65536"):
        if k in final_eval:
            res[k] = final_eval[k]
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Plan 047 Level-Separated Wavelet & 1D Conv GPT")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(REPO_ROOT / ".cache" / "hypothesis_experiments_047.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    p_cfg = SubspaceSpecializationProbeConfig(embed_dim=args.embed_dim, num_subspaces=4, device="cpu")
    cannib_report = audit_cross_level_norm_cannibalization(p_cfg)
    svd_report = audit_svd_spectrum_and_mutual_information(p_cfg)

    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=8_000_000, max_val_tokens=524_288)
    ppmi_emb = load_ppmi_embeddings(vocab_size=50304, d_model=256)
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=args.seq_len, num_batches=6, device=device
    )
    data = VerificationDataBundle(
        ppmi_embeddings=ppmi_emb,
        init_tokens=torch.from_numpy(train_arr[:2_000_000].astype("int32")),
        prior_tokens=torch.from_numpy(train_arr[:1_000_000].astype("int64")),
        work_gpu=torch.from_numpy(train_arr[:6_000_000].view("int16")).to(device),
        val_batches=val_batches,
        steps=args.steps,
        batch_size=args.batch_size,
    )

    prec = "fp8" if device.type == "cuda" else "bf16"
    cfg_m1 = ParsevalMuPSubspaceCoincidenceConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=max(2, args.embed_dim // 64),
        block_size=args.seq_len,
        num_subspaces=1,
        partition_mode="contiguous",
        coincidence_gamma=0.0,
        subspace_var_penalty=0.0,
        coarse_coincidence_weight=0.0,
        readout_gemm_precision=prec,
    )
    cfg_m4_strided = ParsevalMuPSubspaceCoincidenceConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=max(2, args.embed_dim // 64),
        block_size=args.seq_len,
        num_subspaces=4,
        partition_mode="strided",
        coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
        subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        coarse_coincidence_weight=DEFAULT_COARSE_COINCIDENCE_WEIGHT,
        readout_gemm_precision=prec,
    )
    cfg_m4_level_sep = LevelSeparatedWaveletConvConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=max(2, args.embed_dim // 64),
        block_size=args.seq_len,
        num_subspaces=4,
        partition_mode="strided",
        coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
        subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        coarse_coincidence_weight=DEFAULT_COARSE_COINCIDENCE_WEIGHT,
        readout_gemm_precision=prec,
    )

    res_m1 = run_single_arm(
        run_key=f"w32_m1_single_sphere_baseline_d{args.embed_dim}",
        config=cfg_m1,
        data=data,
        use_level_separated_conv=False,
    )
    res_m4_strided = run_single_arm(
        run_key=f"w32_m4_strided_subspace_coincidence_d{args.embed_dim}",
        config=cfg_m4_strided,
        data=data,
        use_level_separated_conv=False,
    )
    res_m4_level_sep = run_single_arm(
        run_key=f"w32_m4_level_separated_wavelet_conv_d{args.embed_dim}",
        config=cfg_m4_level_sep,
        data=data,
        use_level_separated_conv=True,
    )

    summary = {
        "plan_id": "EXPERIMENT_PLAN_047",
        "wave_id": "W32",
        "H1_level_separated_wavelet_and_strided_balance": {
            "passed": bool(
                cannib_report["explicit_level_separated_fine_energy_fraction"] >= 0.249
                and svd_report["skew_reduction_factor"] >= 5.0
            ),
            "cross_level_norm_cannibalization": cannib_report,
            "svd_spectrum_and_mutual_information": svd_report,
        },
        "H2_multirate_wavefunction_conv_and_coincidence_readout": {
            "passed": bool(
                res_m4_level_sep["final_kraus_stiefel_err"] <= MAX_STIEFEL_ERR_GATE
                and abs(res_m4_level_sep.get("energy_fine_65536", 0.25) - 0.25) < 1e-3
            ),
            "m1_single_sphere_baseline": res_m1,
            "m4_strided_subspace_coincidence": res_m4_strided,
            "m4_level_separated_wavelet_conv": res_m4_level_sep,
        },
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Plan 047 verification report to {out_path}")


if __name__ == "__main__":
    main()
