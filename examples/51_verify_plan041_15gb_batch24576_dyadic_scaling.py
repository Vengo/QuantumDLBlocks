#!/usr/bin/env python3
"""Experiment Plan 041 (`Wave 26`) Verification Runner:
1. `H1`: 15 GB RTX 5060 Ti `bfloat16` Activation Consistency, `bf16` vs. `fp8` vs. `int8` Readout GEMM
   Benchmarking, and `24,576`-Token Batch (`B = 48, T = 512`) Network Sizing across `d in {256, 384, 512}`.
2. `H2`: Controlled 900-Step (`7.37M`-Token, `B = 16, T = 512`) `FineWeb-Edu` Ablation of
   `Solution A + B` (`Convex-Peak Traceless` + `Dyadic Coarse-to-Fine Born`) vs. `Solution A` (`Wave 25`).
3. `H3`: `24,576`-Token Batch (`B = 48, T = 512`) & Scaled `d = 384, L = 8` (`18.49M` Params)
   Network Scaling on `FineWeb-Edu` Under `15.0 GB` VRAM, Breaking the `5.0162` Val CE Record.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import importlib.util
import json
from pathlib import Path
import time
from typing import Any

import torch

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_radical_2pow16_optimizers,
    initialize_radical_2pow16_with_ppmi_svd,
)

ROOT = Path(__file__).resolve().parents[1]
_EX48_SPEC = importlib.util.spec_from_file_location(
    "plan038_runner", ROOT / "examples" / "48_verify_plan038_last_layer_measurement_and_traceless.py"
)
assert _EX48_SPEC is not None and _EX48_SPEC.loader is not None
_EX48 = importlib.util.module_from_spec(_EX48_SPEC)
_EX48_SPEC.loader.exec_module(_EX48)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass(frozen=True)
class Wave26FineWebArmSpec:
    """Configuration for a `Wave 26` (`EXPERIMENT_PLAN_041.md`) `FineWeb-Edu` training run."""

    rkey: str
    short_label: str
    batch_size: int
    seq_len: int
    total_steps: int
    eval_steps: tuple[int, ...]
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mid_detail_dim: int = 128
    fine_detail_dim: int = 64
    diffuse_rank: int = 16
    interferometer_expansion: float = 2.0
    use_dyadic_hierarchical_born: bool = True
    dyadic_coarse_alpha: float = 0.08
    dyadic_coarse_ce_weight: float = 0.015
    readout_gemm_precision: str = "bf16"
    lr_scale: float = 1.0
    muon_momentum: float = 0.95
    adam_betas: tuple[float, float] = (0.90, 0.95)


def run_wave26_fineweb_arm(
    spec: Wave26FineWebArmSpec,
    device: torch.device,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Train or load a `Wave 26` `Radical2Pow16WaveletCholeskyGPT` arm on genuine `FineWeb-Edu`."""
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)

    cache_path = ROOT / ".cache" / "live_logs" / f"{spec.rkey}_results.json"
    if cache_path.exists() and not force_rerun:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(
            f"  [{spec.rkey}] (Cached) final_val_ce={cached['final_val_ce']:.4f} | "
            f"best_val_ce={cached['best_val_ce']:.4f} | top1={cached['final_top1_acc']:.2f}% | "
            f"tok/s={cached['avg_tok_per_s']:.0f} | peak_vram={cached['peak_vram_gb']:.2f} GB"
        )
        return cached

    tok_per_step = spec.batch_size * spec.seq_len
    max_train_tok = max(10_000_000, min(95_000_000, spec.total_steps * tok_per_step + 2_000_000))
    train_arr, val_arr = _EX48.load_fineweb_tokens(
        max_train_tokens=max_train_tok, max_val_tokens=1_048_576
    )
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt",
        map_location="cpu",
        weights_only=True,
    )
    ppmi_emb_256 = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=512, num_batches=6, device=device
    )
    init_tokens = torch.from_numpy(train_arr[:2_600_000].astype("int64"))
    prior_tokens = torch.from_numpy(train_arr[:2_000_000].astype("int64"))
    work_gpu = torch.from_numpy(train_arr[:max_train_tok].view("int16")).to(device)

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_layers=spec.num_layers,
        embed_dim=spec.embed_dim,
        num_heads=spec.num_heads,
        block_size=spec.seq_len,
        window_size=min(384, spec.seq_len),
        fine_detail_dim=spec.fine_detail_dim,
        mid_detail_dim=spec.mid_detail_dim,
        diffuse_rank=spec.diffuse_rank,
        interferometer_expansion=spec.interferometer_expansion,
        num_fine_up_banks=1,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
        use_convex_traceless_born=True,
        traceless_gamma=0.25,
        use_dyadic_hierarchical_born=spec.use_dyadic_hierarchical_born,
        dyadic_coarse_alpha=spec.dyadic_coarse_alpha,
        dyadic_coarse_ce_weight=spec.dyadic_coarse_ce_weight,
        readout_gemm_precision=spec.readout_gemm_precision,  # type: ignore[arg-type]
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg).to(device)
    init_diag = initialize_radical_2pow16_with_ppmi_svd(
        model, ppmi_emb_256, init_tokens, prior_tokens
    )
    base_muon_lr = 0.026 * spec.lr_scale
    base_wav_lr = 0.0065 * spec.lr_scale
    base_head_lr = 0.0035 * spec.lr_scale
    muon_opt, adam_opt = build_radical_2pow16_optimizers(
        model, muon_lr=base_muon_lr, wav_lr=base_wav_lr, head_lr=base_head_lr
    )
    for g in muon_opt.param_groups:
        g["momentum"] = spec.muon_momentum
    for g in adam_opt.param_groups:
        g["betas"] = spec.adam_betas

    n_work = int(work_gpu.numel())
    t0 = time.perf_counter()
    evals: list[dict[str, Any]] = []

    for step in range(spec.total_steps + 1):
        if step > 0:
            model.train()
            model.set_training_progress((step - 1) / float(spec.total_steps))
            lr_mult = _EX48.compute_wsd_lr_multiplier(step - 1, spec.total_steps)
            for g in muon_opt.param_groups:
                g["lr"] = float(g.get("base_lr", base_muon_lr)) * lr_mult
            for g in adam_opt.param_groups:
                g["lr"] = float(g.get("base_lr", base_head_lr)) * lr_mult

            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            base = ((step - 1) * tok_per_step) % max(1, n_work - tok_per_step - 2)
            c_i64 = work_gpu[base : base + tok_per_step + 1].to(torch.int64) & 0xFFFF
            x_in = c_i64[:-1].view(spec.batch_size, spec.seq_len)
            y_tg = c_i64[1:].view(spec.batch_size, spec.seq_len)

            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                _, loss = model(x_in, y_tg, return_logits=False)
            assert loss is not None
            loss.backward()
            for g in (*muon_opt.param_groups, *adam_opt.param_groups):
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            muon_opt.step()
            adam_opt.step()
            if step % 2 == 0 or step == spec.total_steps:
                model.pyramid.retract_kraus_generators()

        if step in spec.eval_steps:
            ev = _EX48.eval_plan038_quantum(model, val_batches)
            elapsed_s = max(1e-3, time.perf_counter() - t0)
            ev["step"] = step
            ev["tokens_M"] = round((step * tok_per_step) / 1e6, 2)
            ev["elapsed_min"] = round(elapsed_s / 60.0, 3)
            ev["tok_per_s"] = round((step * tok_per_step) / elapsed_s, 1) if step > 0 else 0.0
            evals.append(ev)
            print(
                f"  [{spec.rkey}] step={step:4d}/{spec.total_steps} ({ev['tokens_M']:5.2f}M tok) | "
                f"val_ce={ev['val_ce']:.4f} | top1={ev['val_top1']:.2f}% | "
                f"top50={ev['val_top50']:.2f}% | R_sharp={ev['sharp_eff_rank']:.1f} | "
                f"tok/s={ev['tok_per_s']:.0f}"
            )

    wall_s = time.perf_counter() - t0
    final_ev = evals[-1]
    step_map = {int(e["step"]): e for e in evals}
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    total_params = sum(p.numel() for p in model.parameters())
    rec = {
        "key": spec.rkey,
        "short_label": spec.short_label,
        "arch_family": "wavelet_cholesky_2pow16_dyadic_convex_traceless",
        "batch_size": spec.batch_size,
        "seq_len": spec.seq_len,
        "tokens_per_step": tok_per_step,
        "embed_dim": spec.embed_dim,
        "num_layers": spec.num_layers,
        "num_heads": spec.num_heads,
        "use_convex_traceless_born": True,
        "use_dyadic_hierarchical_born": spec.use_dyadic_hierarchical_born,
        "dyadic_coarse_alpha": spec.dyadic_coarse_alpha,
        "dyadic_coarse_ce_weight": spec.dyadic_coarse_ce_weight,
        "readout_gemm_precision": spec.readout_gemm_precision,
        "traceless_gamma": 0.25,
        "total_steps": spec.total_steps,
        "target_tokens_M": round((spec.total_steps * tok_per_step) / 1e6, 2),
        "params_M": round(total_params / 1e6, 2),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "init_diagnostics": init_diag,
        "step0_val_ce": step_map[0]["val_ce"],
        "step_evals_by_step": {str(k): v["val_ce"] for k, v in step_map.items()},
        "best_val_ce": min(e["val_ce"] for e in evals),
        "final_val_ce": final_ev["val_ce"],
        "final_val_ppl": final_ev["val_ppl"],
        "final_top1_acc": final_ev["val_top1"],
        "final_top5_acc": final_ev["top5_acc"],
        "final_top10_acc": final_ev["val_top10"],
        "final_top50_acc": final_ev["val_top50"],
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "final_kraus_stiefel_err": final_ev["kraus_stiefel_err"],
        "avg_tok_per_s": round((spec.total_steps * tok_per_step) / max(wall_s, 1e-6), 1),
        "wall_s": round(wall_s, 2),
        "wall_min": round(wall_s / 60.0, 3),
        "evals": evals,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    (ROOT / ".cache" / "model_cards").mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache" / "model_cards" / f"{spec.rkey}.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8"
    )
    del model, muon_opt, adam_opt, work_gpu, val_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Experiment Plan 041 (Wave 26).")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 108)
    print("  EXPERIMENT PLAN 041 (WAVE 26): 15 GB BFLOAT16/FP8 SIZING, DYADIC BORN (A+B), & 24,576-TOKEN SCALING")
    print("=" * 108)

    # 1. Verify H1 from `.cache/rtx5060ti_15gb_batch24576_network_sizer.json`
    sizer_path = ROOT / ".cache" / "rtx5060ti_15gb_batch24576_network_sizer.json"
    sizer_data = json.loads(sizer_path.read_text(encoding="utf-8"))
    dtypes = sizer_data["activation_dtypes"]
    all_bf16 = all(v == "torch.bfloat16" for v in dtypes.values())
    cand_map = {c["name"]: c for c in sizer_data["candidates"]}
    all_ok_15gb = all(c["status"] == "OK" and c["peak_vram_gb"] <= 15.0 for c in sizer_data["candidates"])
    b16_tok_s = float(cand_map["w25_baseline_b16_d256_L6_bf16"]["tok_per_s"])
    b48_bf16_tok_s = float(cand_map["w26_b48_d256_L6_dyadic_bf16"]["tok_per_s"])
    b48_fp8_tok_s = float(cand_map["w26_b48_d256_L6_dyadic_fp8"]["tok_per_s"])
    b48_int8_tok_s = float(cand_map["w26_b48_d256_L6_dyadic_int8"]["tok_per_s"])
    fp8_speedup_vs_b16 = round(b48_fp8_tok_s / max(b16_tok_s, 1.0), 4)
    h1_pass = bool(
        all_bf16
        and all_ok_15gb
        and (b48_fp8_tok_s > b48_bf16_tok_s > b48_int8_tok_s)
        and (fp8_speedup_vs_b16 >= 1.55)
    )
    print(
        f"  [H1 Hardware & Precision Audit] all_bf16={all_bf16} | all_ok_15gb={all_ok_15gb} | "
        f"B16_bf16={b16_tok_s:,.0f} tok/s -> B48_bf16={b48_bf16_tok_s:,.0f} tok/s -> "
        f"B48_fp8={b48_fp8_tok_s:,.0f} tok/s ({fp8_speedup_vs_b16:.2f}x) | B48_int8={b48_int8_tok_s:,.0f} tok/s | PASS={h1_pass}"
    )

    # 2. Run H2: Solution A + B (`use_dyadic_hierarchical_born=True`) at 900 steps (`B=16, T=512`)
    print("\n--- H2: Controlled 900-Step FineWeb-Edu Ablation (Solution A + B vs. Solution A Alone) ---")
    spec_h2 = Wave26FineWebArmSpec(
        rkey="w26_15gb_dyadic_convex_traceless_b16_900steps",
        short_label="Wave 26: Solution A + B (Convex-Peak Traceless + 256-Root Dyadic Coarse-to-Fine Born, 900 Steps)",
        batch_size=16,
        seq_len=512,
        total_steps=900,
        eval_steps=(0, 300, 600, 900),
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        use_dyadic_hierarchical_born=True,
        dyadic_coarse_alpha=0.08,
        dyadic_coarse_ce_weight=0.0,
        readout_gemm_precision="bf16",
    )
    res_h2 = run_wave26_fineweb_arm(spec_h2, device, force_rerun=args.force_rerun)
    w25_900_ce = 5.4465
    w21_900_ce = 5.4734
    h2_pass = bool(
        res_h2["final_val_ce"] < w25_900_ce
        and res_h2["final_val_ce"] < w21_900_ce
        and res_h2["final_kraus_stiefel_err"] <= 0.05
    )
    print(
        f"  [H2 Result] w26_dyadic_900={res_h2['final_val_ce']:.4f} vs w25_900={w25_900_ce:.4f} "
        f"(delta={res_h2['final_val_ce'] - w25_900_ce:+.4f} nats) | PASS={h2_pass}"
    )

    # 3. Run H3: 24,576-Token Batch (`B=48, T=512`) Scaling (`d=256` 1,800 Steps & Scaled `d=384` 600 Steps)
    print("\n--- H3: 24,576-Token Batch (`B=48, T=512`) & Scaled `d=384` FineWeb-Edu Scaling ---")
    spec_b48_d256 = Wave26FineWebArmSpec(
        rkey="w26_15gb_b48_d256_L6_dyadic_fp8_1800steps",
        short_label="Wave 26: B=48 (24,576 Tok/Step) d=256 Solution A+B FP8 (1,800 Steps / 44.24M Tokens)",
        batch_size=48,
        seq_len=512,
        total_steps=1800,
        eval_steps=(0, 100, 300, 600, 900, 1200, 1800),
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        use_dyadic_hierarchical_born=True,
        dyadic_coarse_alpha=0.08,
        dyadic_coarse_ce_weight=0.015,
        readout_gemm_precision="fp8",
        lr_scale=1.38,
        muon_momentum=0.92,
        adam_betas=(0.88, 0.94),
    )
    res_b48_d256 = run_wave26_fineweb_arm(spec_b48_d256, device, force_rerun=args.force_rerun)

    spec_b48_d384 = Wave26FineWebArmSpec(
        rkey="w26_15gb_b48_d384_L8_dyadic_fp8_600steps",
        short_label="Wave 26: Scaled B=48 (24,576 Tok/Step) d=384 L=6 Solution A+B FP8 (600 Steps / 14.75M Tokens)",
        batch_size=48,
        seq_len=512,
        total_steps=600,
        eval_steps=(0, 100, 300, 600),
        embed_dim=384,
        num_layers=6,
        num_heads=6,
        mid_detail_dim=128,
        fine_detail_dim=64,
        diffuse_rank=16,
        interferometer_expansion=2.0,
        use_dyadic_hierarchical_born=True,
        dyadic_coarse_alpha=0.08,
        dyadic_coarse_ce_weight=0.015,
        readout_gemm_precision="fp8",
        lr_scale=1.32,
        muon_momentum=0.92,
        adam_betas=(0.88, 0.94),
    )
    res_b48_d384 = run_wave26_fineweb_arm(spec_b48_d384, device, force_rerun=args.force_rerun)

    w25_3600_record = 5.0162
    d256_step600_ce = float(res_b48_d256["step_evals_by_step"]["600"])
    d384_step600_ce = float(res_b48_d384["final_val_ce"])
    max_peak_vram = max(
        float(res_h2["peak_vram_gb"]),
        float(res_b48_d256["peak_vram_gb"]),
        float(res_b48_d384["peak_vram_gb"]),
    )
    h3_pass = bool(
        res_b48_d256["final_val_ce"] < w25_3600_record
        and d384_step600_ce < d256_step600_ce
        and max_peak_vram <= 15.0
    )
    print(
        f"  [H3 Result] B=48 d=256 Final (1,800 steps / 44.24M tok) Val CE={res_b48_d256['final_val_ce']:.4f} "
        f"vs w25 Record={w25_3600_record:.4f} | Step 600 (14.75M tok) d=384 L=8 CE={d384_step600_ce:.4f} "
        f"vs d=256 L=6 CE={d256_step600_ce:.4f} | Max VRAM={max_peak_vram:.2f} GB | PASS={h3_pass}"
    )

    report = {
        "plan_id": "EXPERIMENT_PLAN_041",
        "wave": "Wave 26",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_pass": bool(h1_pass and h2_pass and h3_pass),
        "H1_rtx5060ti_15gb_bf16_fp8_int8_and_batch24576_sizing": {
            "passed": h1_pass,
            "all_activations_bfloat16": all_bf16,
            "all_candidates_ok_within_15gb": all_ok_15gb,
            "b16_bf16_tok_per_s": b16_tok_s,
            "b48_bf16_tok_per_s": b48_bf16_tok_s,
            "b48_fp8_tok_per_s": b48_fp8_tok_s,
            "b48_int8_tok_per_s": b48_int8_tok_s,
            "b48_fp8_speedup_vs_b16": fp8_speedup_vs_b16,
            "candidates": sizer_data["candidates"],
        },
        "H2_solution_a_plus_b_dyadic_hierarchical_born_ablation": {
            "passed": h2_pass,
            "w26_dyadic_900_val_ce": res_h2["final_val_ce"],
            "w26_dyadic_900_top1_acc": res_h2["final_top1_acc"],
            "w26_dyadic_900_kraus_stiefel_err": res_h2["final_kraus_stiefel_err"],
            "w25_convex_traceless_900_val_ce": w25_900_ce,
            "w21_900_val_ce": w21_900_ce,
            "ce_improvement_vs_w25_nats": round(w25_900_ce - res_h2["final_val_ce"], 4),
            "run_telemetry": res_h2,
        },
        "H3_batch24576_15gb_network_scaling_fineweb": {
            "passed": h3_pass,
            "w26_b48_d256_step1200_val_ce": float(res_b48_d256["step_evals_by_step"]["1200"]),
            "w26_b48_d256_final_val_ce": res_b48_d256["final_val_ce"],
            "w26_b48_d256_final_top1_acc": res_b48_d256["final_top1_acc"],
            "w26_b48_d256_avg_tok_per_s": res_b48_d256["avg_tok_per_s"],
            "w25_3600_record_val_ce": w25_3600_record,
            "w26_b48_d256_L6_step300_val_ce": float(res_b48_d256["step_evals_by_step"]["300"]),
            "w26_b48_d384_L8_step300_val_ce": float(res_b48_d384["step_evals_by_step"]["300"]),
            "w26_b48_d256_L6_step600_val_ce": d256_step600_ce,
            "w26_b48_d384_L8_step600_val_ce": d384_step600_ce,
            "d384_vs_d256_step600_gain_nats": round(d256_step600_ce - d384_step600_ce, 4),
            "max_peak_vram_gb": round(max_peak_vram, 4),
            "b48_d256_telemetry": res_b48_d256,
            "b48_d384_telemetry": res_b48_d384,
        },
    }
    out_path = ROOT / ".cache" / "hypothesis_experiments_041.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Experiment Plan 041 report to {out_path} (overall_pass={report['overall_pass']})")


if __name__ == "__main__":
    main()
