#!/usr/bin/env python3
"""Controlled verification runner for Experiment Plan 046 (Wave 31, Lever 3).

Compares Single-Sphere Parseval Born (M=1) against Product-of-Spheres Multi-Subspace
Coincidence Born (M=4) under identical FineWeb-Edu data streams, optimizer schedules,
and StandardizedExperimentLogger telemetry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
from qdlblocks.models.parseval_mup_subspace_coincidence_gpt import (
    ParsevalMuPSubspaceCoincidenceConfig,
    ParsevalMuPSubspaceCoincidenceGPT,
    build_subspace_coincidence_optimizers,
    initialize_subspace_coincidence_with_ppmi_svd,
)
from scripts.multi_subspace_coincidence_margin_probe import (
    compute_participation_ratio,
    run_probe_grid,
)

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


def evaluate_subspace_model(
    model: ParsevalMuPSubspaceCoincidenceGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate validation CE, PPL, Top-1, Top-5, R_sharp, and Stiefel error on genuine FineWeb."""
    ev = _EX53.eval_parseval_mup_quantum(model, val_batches)
    with torch.no_grad():
        m_sharp, _, _ = model.synthesize_detectors()
        r_sharp = compute_participation_ratio(m_sharp[:8192])
        iso_err = model.pyramid.compute_kraus_stiefel_err()
    return {
        "val_ce": float(ev["val_ce"]),
        "val_ppl": float(ev["val_ppl"]),
        "top1_acc": float(ev["val_top1"]) / 100.0,
        "top50_acc": float(ev.get("val_top50", 0.0)) / 100.0,
        "sharp_eff_rank": float(r_sharp),
        "centered_r_sharp": float(ev.get("sharp_eff_rank", r_sharp)),
        "kraus_stiefel_err": float(iso_err),
        "mean_beta": float(model.current_beta()),
    }


def run_single_arm(
    run_key: str,
    config: ParsevalMuPSubspaceCoincidenceConfig,
    ppmi_embeddings: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
    work_gpu: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    steps_and_batch: tuple[int, int],
) -> dict[str, Any]:
    """Train and evaluate one controlled arm (M=1 baseline vs M=4 coincidence) on FineWeb-Edu."""
    steps, batch_size = steps_and_batch
    device = work_gpu.device
    seq_len = config.block_size
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    model = ParsevalMuPSubspaceCoincidenceGPT(config).to(device)
    initialize_subspace_coincidence_with_ppmi_svd(
        model=model,
        ppmi_embeddings=ppmi_embeddings,
        init_tokens=init_tokens,
        prior_tokens=prior_tokens,
    )
    eta_base = 0.0065
    eta_max = 0.026
    opt_muon, opt_adam = build_subspace_coincidence_optimizers(
        model, base_muon_lr=eta_max, base_wav_lr=0.0085, base_head_lr=0.005
    )
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    tokens_per_step = batch_size * seq_len

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id="W31",
        plan_id="EXPERIMENT_PLAN_046",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 31 Lever 3 controlled verification: M={config.num_subspaces} Product-of-Spheres "
            f"(coinc_gamma={config.coincidence_gamma}, var_pen={config.subspace_var_penalty}) "
            f"at d={config.embed_dim}, L={config.num_layers}."
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

    eval0 = evaluate_subspace_model(model, val_batches)
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
    n_work = int(work_gpu.numel())

    for step in range(1, steps + 1):
        model.train()
        tok_prog = min(1.0, float((step - 1) * tokens_per_step) / 29_491_200.0)
        model.set_training_progress(tok_prog)
        lr_256, mom = _EX53.compute_step_lr_and_momentum(
            "modded_nanogpt_wsd", step - 1, steps, eta_base, eta_max
        )
        _EX53.apply_scaled_group_lrs(opt_muon, opt_adam, lr_256, config.embed_dim, mom)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        offset = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        c_i64 = work_gpu[offset : offset + tokens_per_step + 1].to(torch.int64) & 0xFFFF
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
            logger.log_step(step, float(loss.item()), tok_s, lr_256 / eta_max, elapsed / 60.0)
            model.set_training_progress(min(1.0, float(step * tokens_per_step) / 29_491_200.0))
            ev = evaluate_subspace_model(model, val_batches)
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
    return {
        "run_key": run_key,
        "num_subspaces": config.num_subspaces,
        "best_val_ce": best_val_ce,
        "final_val_ce": final_eval["val_ce"],
        "final_top1_acc": final_eval["top1_acc"],
        "final_top50_acc": final_eval["top50_acc"],
        "final_sharp_eff_rank": final_eval["sharp_eff_rank"],
        "final_centered_r_sharp": final_eval["centered_r_sharp"],
        "final_kraus_stiefel_err": final_eval["kraus_stiefel_err"],
        "tok_per_s": avg_tok_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Plan 046 Multi-Subspace Coincidence Born")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(REPO_ROOT / ".cache" / "hypothesis_experiments_046.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    probe_report = run_probe_grid(
        dims=[256, 384, 512],
        subspaces=[1, 2, 4, 8],
        vocab_size=16384,
        device="cpu",
    )

    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=8_000_000, max_val_tokens=524_288)
    ppmi_emb = load_ppmi_embeddings(vocab_size=50304, d_model=256)
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=args.seq_len, num_batches=6, device=device
    )
    init_tokens = torch.from_numpy(train_arr[:2_000_000].astype("int32"))
    prior_tokens = torch.from_numpy(train_arr[:1_000_000].astype("int64"))
    work_gpu = torch.from_numpy(train_arr[:6_000_000].view("int16")).to(device)

    cfg_m1 = ParsevalMuPSubspaceCoincidenceConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=max(2, args.embed_dim // 64),
        block_size=args.seq_len,
        num_subspaces=1,
        coincidence_gamma=0.0,
        subspace_var_penalty=0.0,
        coarse_coincidence_weight=0.0,
        readout_gemm_precision="fp8" if device.type == "cuda" else "bf16",
    )
    cfg_m4 = ParsevalMuPSubspaceCoincidenceConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=max(2, args.embed_dim // 64),
        block_size=args.seq_len,
        num_subspaces=4,
        coincidence_gamma=0.22,
        subspace_var_penalty=0.14,
        coarse_coincidence_weight=0.16,
        readout_gemm_precision="fp8" if device.type == "cuda" else "bf16",
    )

    res_m1 = run_single_arm(
        run_key=f"w31_single_sphere_m1_d{args.embed_dim}_L{args.num_layers}",
        config=cfg_m1,
        ppmi_embeddings=ppmi_emb,
        init_tokens=init_tokens,
        prior_tokens=prior_tokens,
        work_gpu=work_gpu,
        val_batches=val_batches,
        steps_and_batch=(args.steps, args.batch_size),
    )
    res_m4 = run_single_arm(
        run_key=f"w31_subspace_coincidence_m4_d{args.embed_dim}_L{args.num_layers}",
        config=cfg_m4,
        ppmi_embeddings=ppmi_emb,
        init_tokens=init_tokens,
        prior_tokens=prior_tokens,
        work_gpu=work_gpu,
        val_batches=val_batches,
        steps_and_batch=(args.steps, args.batch_size),
    )

    summary = {
        "plan_id": "EXPERIMENT_PLAN_046",
        "wave_id": "W31",
        "H1_product_of_spheres_rank_preservation": probe_report["H1_product_of_spheres_rank_preservation"],
        "H2_coincidence_runnerup_suppression": probe_report["H2_coincidence_runnerup_suppression"],
        "H3_end_to_end_training_verification": {
            "passed": bool(
                res_m4["final_kraus_stiefel_err"] <= 0.05
                and res_m4["final_sharp_eff_rank"] >= res_m1["final_sharp_eff_rank"]
            ),
            "m1_baseline": res_m1,
            "m4_subspace_coincidence": res_m4,
        },
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Plan 046 verification report to {out_path}")


if __name__ == "__main__":
    main()
