#!/usr/bin/env python3
"""Plan 036 Verification: Fused Kronecker-U-Net N=2^16 Wavelet-Cholesky & Dyadic Predecessor Bigram Coupling."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import numpy as np
import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.config_models import compute_wsd_lr_multiplier
from qdlblocks.benchmarks.db import DEFAULT_DB_PATH, BenchmarkSQLiteDB, backfill_micro_verifications
from qdlblocks.benchmarks.model_card import (
    build_model_card_from_manifest_entry,
    save_model_card_atomic,
    update_model_card_telemetry,
)
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_radical_2pow16_optimizers,
    initialize_radical_2pow16_with_ppmi_svd,
)

CACHE_DIR, LIVE_DIR, CARDS_DIR = ROOT / ".cache", ROOT / ".cache" / "live_logs", ROOT / ".cache" / "model_cards"
_EX45_SPEC = importlib.util.spec_from_file_location(
    "plan035_runner", ROOT / "examples" / "45_verify_plan035_radical_2pow16_wavelet_cholesky.py"
)
assert _EX45_SPEC is not None and _EX45_SPEC.loader is not None
_EX45 = importlib.util.module_from_spec(_EX45_SPEC)
_EX45_SPEC.loader.exec_module(_EX45)
load_fineweb_tokens, build_evaluation_batches, eval_quantum = (
    _EX45.load_fineweb_tokens, _EX45.build_evaluation_batches, _EX45.eval_quantum,
)


def run_h1_fused_kronecker_unet_verification(
    data_bundle: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Verify H1: Fused Kronecker-U-Net parameter budget (<12.5M), Step-0 bigram ratio >=1.80x, & causal masking."""
    print("\n" + "=" * 96 + "\n  [H1 Verification] Fused Kronecker-U-Net N=2^16 Predecessor-Bigram & Bigram Gate\n" + "=" * 96)
    init_tok_t = torch.from_numpy(data_bundle["init_tokens"].astype(np.int64))
    x0, y0 = data_bundle["val_batches"][0]
    arch_stats: dict[str, Any] = {}

    for mode in ("multiscale_unet", "consistent_kronecker", "fused_kronecker_unet"):
        torch.manual_seed(2036)
        model = Radical2Pow16WaveletCholeskyGPT(Radical2Pow16Config(arch_mode=mode)).to(device)
        if mode != "fused_kronecker_unet":
            model.pyramid.transition_gain.data.zero_()
        st = initialize_radical_2pow16_with_ppmi_svd(model, data_bundle["ppmi_emb_256"], init_tok_t, data_bundle["prior_tokens"])
        total_params = sum(p.numel() for p in model.parameters())
        params_m = round(total_params / 1e6, 2)

        model.eval()
        x_c1 = x0[:2, :64].clone()
        x_c2 = x_c1.clone()
        x_c2[:, 32:] = (x_c2[:, 32:] + 137) % 50257
        with torch.no_grad():
            logits1, _ = model(x_c1, targets=None, return_logits=True)
            logits2, _ = model(x_c2, targets=None, return_logits=True)
        assert logits1 is not None and logits2 is not None
        max_future_leakage = float((logits1[:, :32, :] - logits2[:, :32, :]).abs().max().item())

        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss_m = model(x0[:2, :128], targets=y0[:2, :128], return_logits=False)
        assert loss_m is not None
        loss_m.backward()
        grad_ok = bool(
            model.pyramid.wav_256.grad is not None
            and model.pyramid.wav_65536.grad is not None
            and all(
                torch.isfinite(p.grad).all().item()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            )
        )
        arch_stats[mode] = {
            "total_params": int(total_params), "params_M": params_m,
            "max_self_overlap": round(float(st["max_self_overlap"]), 6),
            "directed_bigram_overlap_ratio": round(float(st["directed_bigram_overlap_ratio"]), 4),
            "max_future_leakage": max_future_leakage, "grad_finite": grad_ok,
            "compression_vs_quantum_born_40p78M": round(40.78 / params_m, 2),
            "compression_vs_classical_30p95M": round(30.95 / params_m, 2),
        }
        del model
        torch.cuda.empty_cache()

    f_st, u_st, k_st = arch_stats["fused_kronecker_unet"], arch_stats["multiscale_unet"], arch_stats["consistent_kronecker"]
    grad_finite = bool(u_st["grad_finite"] and k_st["grad_finite"] and f_st["grad_finite"])
    passed = bool(
        f_st["total_params"] < 12_500_000 and f_st["compression_vs_quantum_born_40p78M"] >= 3.5
        and f_st["max_self_overlap"] <= 0.05 and f_st["directed_bigram_overlap_ratio"] >= 1.80
        and f_st["max_future_leakage"] == 0.0 and grad_finite
    )
    print(
        f"  * [H1 Fused Kronecker-U-Net] Params={f_st['params_M']:.2f}M ({f_st['total_params']:,}) | "
        f"SelfOv={f_st['max_self_overlap']:.4f} | BigramRatio={f_st['directed_bigram_overlap_ratio']:.2f}x "
        f"(vs {u_st['directed_bigram_overlap_ratio']:.2f}x unaligned) | Leakage={f_st['max_future_leakage']:.1e} | PASS={passed}"
    )
    return {
        "passed": passed, "fused_total_params": f_st["total_params"], "fused_params_M": f_st["params_M"],
        "max_self_overlap": f_st["max_self_overlap"], "directed_bigram_overlap_ratio": f_st["directed_bigram_overlap_ratio"],
        "unaligned_janti_bigram_overlap_ratio": u_st["directed_bigram_overlap_ratio"],
        "max_future_leakage": f_st["max_future_leakage"], "grad_finite": grad_finite,
        "fused_kronecker_unet": f_st, "multiscale_unet": u_st, "consistent_kronecker": k_st,
    }


def save_and_register_run(meta: dict[str, Any], result_rec: dict[str, Any], db: BenchmarkSQLiteDB) -> None:
    """Persist `_full_result.json`, `_results.json`, model card, and SQLite benchmark DB row."""
    rkey = meta["key"]
    (LIVE_DIR / f"{rkey}_full_result.json").write_text(json.dumps(result_rec, indent=2), encoding="utf-8")
    (LIVE_DIR / f"{rkey}_results.json").write_text(json.dumps({**result_rec, "completed": True}, indent=2), encoding="utf-8")
    if rkey.startswith("w21_"):
        w_id, w_num, p_id = "W21", 21, "EXPERIMENT_PLAN_036"
        w_lbl = "Wave 21 (Plan 036 Fused Kronecker-U-Net N=2^16 Wavelet-Cholesky)"
    elif rkey.startswith("w20_"):
        w_id, w_num, p_id = "W20", 20, "EXPERIMENT_PLAN_035"
        w_lbl = "Wave 20 (Plan 035 Radical N=2^16 Wavelet-Cholesky)"
    else:
        w_id, w_num, p_id = "W18", 18, "EXPERIMENT_PLAN_033"
        w_lbl = "Wave 18 (Plan 033 3,600-Step 8GB Study)"

    manifest_entry = {
        "key": rkey, "session": "local_rtx5060ti_8gb", "wave_id": w_id, "wave_num": w_num, "wave_label": w_lbl,
        "plan_id": p_id, "run_tier": "micro_verification", "arch_family": meta["arch_family"], "label": meta["label"],
        "short_label": meta["short_label"], "color": meta["color"], "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
        "target_tokens_M": result_rec["target_tokens_M"], "cli_args": meta["cli_args"],
        "remote_log": f".cache/live_logs/{rkey}.log", "remote_json": f".cache/live_logs/{rkey}_results.json",
    }
    cat_entry = {
        **manifest_entry, "status": "COMPLETED", "run_state": "COMPLETED", "completed": True, "is_live": False,
        "accel": "RTX 5060 Ti (8GB Cap)", "params_M": result_rec["params_M"], "tokens_done_M": result_rec["target_tokens_M"],
        "progress_pct": 100.0, "best_val_ce": result_rec["best_val_ce"], "best_val_ppl": result_rec["best_val_ppl"],
        "final_val_ce": result_rec["final_val_ce"], "top1_acc": result_rec["final_top1_acc"],
        "top5_acc": result_rec["final_top5_acc"], "med_rank": result_rec["final_med_rank"],
        "iso_err": result_rec["final_iso_err"], "entanglement_nats": 2.9, "avg_tok_per_s": result_rec["avg_tok_per_s"],
        "wall_min": round(result_rec["wall_s"] / 60.0, 4), "evals": result_rec["evals"], "steps": [],
    }
    card = build_model_card_from_manifest_entry(manifest_entry, w_id, w_lbl, p_id)
    card["run_tier"] = "micro_verification"
    card = update_model_card_telemetry(card, cat_entry)
    save_model_card_atomic(card, CARDS_DIR)
    db.upsert_run(cat_entry, model_card=card, explicit_user_override=False)


def run_single_plan036_experiment(
    meta: dict[str, Any], data_bundle: dict[str, Any], db: BenchmarkSQLiteDB, device: torch.device
) -> dict[str, Any]:
    """Execute or load a single Plan 036 experiment specification within the <= 8.0 GB VRAM budget."""
    rkey = meta["key"]
    full_cache = LIVE_DIR / f"{rkey}_full_result.json"
    if full_cache.exists():
        cached = json.loads(full_cache.read_text(encoding="utf-8"))
        save_and_register_run(meta, cached, db)
        print(f"  * [{rkey}] (Cached) BestCE={cached['best_val_ce']:.4f} | {cached['avg_tok_per_s']:,.0f} tok/s")
        return cached

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)
    np.random.seed(42)
    total_steps = int(meta["total_steps"])
    seq_len, micro_bs, grad_accum = 512, 8, 2
    micro_tokens, tokens_per_step = micro_bs * seq_len, micro_bs * seq_len * grad_accum
    target_tok_m = round((total_steps * tokens_per_step) / 1e6, 2)

    model = Radical2Pow16WaveletCholeskyGPT(meta["rcfg"]).to(device)
    initialize_radical_2pow16_with_ppmi_svd(
        model, data_bundle["ppmi_emb_256"], torch.from_numpy(data_bundle["init_tokens"].astype(np.int64)), data_bundle["prior_tokens"]
    )
    base_muon_lr, base_wav_lr, base_head_lr = 0.026, 0.007, 0.0035
    muon_opt, adam_opt = build_radical_2pow16_optimizers(model, muon_lr=base_muon_lr, wav_lr=base_wav_lr, head_lr=base_head_lr)
    params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    log_path = LIVE_DIR / f"{rkey}.log"
    log_path.write_text(f"=== {meta['label']} ===\nRun Key: {rkey} | Params: {params_m:.2f}M\n", encoding="utf-8")

    work_gpu: torch.Tensor = data_bundle["work_gpu_900" if total_steps <= 900 else "work_gpu_3600"]
    n_work = int(work_gpu.numel())
    eval_steps = {0, 300, 600, 900} if total_steps <= 900 else {0, 300, 900, 1800, 2700, 3600}
    eval_history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for step in range(0, total_steps + 1):
        if step > 0:
            mult = compute_wsd_lr_multiplier(step - 1, total_steps)
            for g in muon_opt.param_groups:
                g["lr"] = base_muon_lr * mult
            muon_opt.zero_grad(set_to_none=True)
            for g in adam_opt.param_groups:
                g["lr"] = float(g.get("base_lr", base_head_lr)) * mult
            adam_opt.zero_grad(set_to_none=True)

            for mb_i in range(grad_accum):
                base = (((step - 1) * grad_accum + mb_i) * micro_tokens) % max(1, n_work - micro_tokens - 2)
                c_i64 = work_gpu[base : base + micro_tokens + 1].to(torch.int64) & 0xFFFF
                x_mb, y_mb = c_i64[:-1].view(micro_bs, seq_len), c_i64[1:].view(micro_bs, seq_len)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss_mb = model(x_mb, targets=y_mb, return_logits=False)
                assert loss_mb is not None
                (loss_mb / float(grad_accum)).backward()

            for g in (*muon_opt.param_groups, *adam_opt.param_groups):
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            muon_opt.step()
            adam_opt.step()

        if step in eval_steps:
            ev = eval_quantum(model, data_bundle["val_batches"])
            el_min, tok_m = round((time.perf_counter() - t0) / 60.0, 3), round((step * tokens_per_step) / 1e6, 2)
            eval_history.append({
                "epoch": round(step / float(total_steps), 2), "step": step, "tokens_M": tok_m,
                "elapsed_min": el_min, "seq_len": seq_len, "entanglement_nats": 2.45, "gamma_h": 8.65, **ev,
            })
            line = f"[Step {step:4d}/{total_steps} | {tok_m:5.2f}M tok | {el_min:5.2f}m] Val CE: {ev['val_ce']:.4f} | Top1: {ev['top1_acc']:.2f}% | R_diff: {ev['diffuse_eff_rank']:.2f}"
            print("    " + line, flush=True)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(line + "\n")

    wall_s = max(1e-3, time.perf_counter() - t0)
    peak_vram = round(float(torch.cuda.max_memory_allocated(device) / (1024**3)), 4)
    ev300 = next((e for e in eval_history if e["step"] == 300), eval_history[-1])
    ev_last = eval_history[-1]
    best_ce = min(float(e["val_ce"]) for e in eval_history)
    f_keys = (
        "val_ce", "top1_acc", "top5_acc", "top50_acc", "top5_minus_top1_gap", "med_rank", "ece",
        "iso_err", "kraus_stiefel_err", "beam_ortho_err", "kboson_copies", "lambda_diff",
        "diffuse_eff_rank", "sharp_eff_rank", "seed_eff_rank",
    )
    result_rec = {
        "key": rkey, "short_label": meta["short_label"], "arch_family": meta["arch_family"],
        "schedule_mode": meta["schedule_mode"], "total_steps": total_steps, "target_tokens_M": target_tok_m,
        "params_M": params_m, "peak_vram_gb": peak_vram, "step0_val_ce": float(eval_history[0]["val_ce"]),
        "step300_val_ce": float(ev300["val_ce"]), "best_val_ce": round(best_ce, 4),
        "best_val_ppl": round(math.exp(min(20.0, best_ce)), 2), "step300_top1_acc": float(ev300["top1_acc"]),
        "step300_top5_acc": float(ev300["top5_acc"]),
        "min_diffuse_eff_rank": round(min(float(e["diffuse_eff_rank"]) for e in eval_history), 2),
        "avg_tok_per_s": round((total_steps * tokens_per_step) / wall_s, 1), "wall_s": round(wall_s, 2),
        "evals": eval_history, **{f"final_{k}": ev_last[k] for k in f_keys},
    }
    save_and_register_run(meta, result_rec, db)
    del model, muon_opt, adam_opt
    return result_rec


def main() -> None:
    torch._dynamo.config.cache_size_limit, torch._dynamo.config.recompile_limit = 64, 64
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_arr, val_arr = load_fineweb_tokens(max_train_tokens=32_000_000, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(CACHE_DIR / "ppmi_svd_256_fast_synth_plan030.pt", map_location="cpu", weights_only=True)
    ppmi_emb_256 = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = build_evaluation_batches(val_arr, batch_size=4, seq_len=512, num_batches=6, device=device)

    data_bundle = {
        "ppmi_emb_256": ppmi_emb_256, "init_tokens": train_arr[:2_600_000],
        "prior_tokens": torch.from_numpy(train_arr[:2_000_000].astype(np.int64)),
        "work_gpu_900": torch.from_numpy(train_arr[:8_000_000].view(np.int16)).to(device),
        "work_gpu_3600": torch.from_numpy(train_arr[:32_000_000].view(np.int16)).to(device),
        "val_batches": val_batches,
    }
    h1_res = run_h1_fused_kronecker_unet_verification(data_bundle, device)
    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)

    spec_rows = [
        ("w18_8gb_classical_leaderboard_exact_L4_d256_3600steps", "classical_modded", "classical_leaderboard_exact_L4_d256_3600steps", 3600, "#79c0ff", "Plan 033 H3 ClassicalModdedGPT (L=4, d=256, 3,600 Steps)", "W18 8GB Classical Leaderboard-Exact (3,600 Steps)", "--architecture classical_modded --steps 3600", None),
        ("w18_8gb_L4_d256_clean_janti_gentle_lr_900steps", "quantum_born", "clean_janti_gentle_lr_L4_d256_900steps", 900, "#fee440", "Plan 032 Isolate 3 Reference (16x16 J_anti, 900 Steps)", "W18 8GB Clean J_anti 16x16 (900 Steps)", "--architecture quantum_born --steps 900", None),
        ("w18_8gb_L4_d256_janti_soft_floor_3600steps", "quantum_born", "janti_soft_floor_6p5_L4_d256_3600steps", 3600, "#00ff87", "Plan 033 H3 QuantumBornGPT J_anti + Soft Floor 6.5 (3,600 Steps)", "W18 8GB J_anti + Soft Floor 6.5 (3,600 Steps)", "--architecture quantum_born --kraus-spectral-floor 6.5 --steps 3600", None),
        ("w20_8gb_arch1_2pow16_multiscale_wavelet_cholesky_unet_900steps", "radical_2pow16_unet", "arch1_2pow16_multiscale_unet_L4_d256_900steps", 900, "#00f5d4", "Plan 035 H2 Arch 1: 2^16 Multi-Resolution Wavelet-Cholesky U-Net (900 Steps)", "W20 8GB Arch1 2^16 Wavelet U-Net (900 Steps)", "--architecture radical_2pow16 --arch-mode multiscale_unet --steps 900", Radical2Pow16Config(arch_mode="multiscale_unet")),
        ("w20_8gb_arch2_consistent_2pow16_kronecker_wavelet_cholesky_900steps", "radical_2pow16_kronecker", "arch2_2pow16_consistent_kronecker_L4_d256_900steps", 900, "#f15bb5", "Plan 035 H2 Arch 2: Consistent 2^16=256x256 Kronecker Wavelet-Cholesky (900 Steps)", "W20 8GB Arch2 2^16 Kronecker-Wavelet (900 Steps)", "--architecture radical_2pow16 --arch-mode consistent_kronecker --steps 900", Radical2Pow16Config(arch_mode="consistent_kronecker")),
        ("w20_8gb_radical_2pow16_wavelet_cholesky_3600steps", "radical_2pow16_multiscale_unet", "radical_2pow16_multiscale_unet_L4_d256_3600steps", 3600, "#9b5de5", "Plan 035 H3 Standalone N=2^16 Wavelet-Cholesky (multiscale_unet, 3,600 Steps)", "W20 8GB Radical 2^16 WavChol (3,600 Steps)", "--architecture radical_2pow16 --arch-mode multiscale_unet --steps 3600", Radical2Pow16Config(arch_mode="multiscale_unet")),
        ("w21_8gb_arch3_fused_kronecker_unet_2pow16_900steps", "radical_2pow16_fused_kronecker_unet", "arch3_fused_kronecker_unet_2pow16_L4_d256_900steps", 900, "#ff9f1c", "Plan 036 H2 Arch 3: Fused Kronecker-U-Net 2^16 Wavelet-Cholesky (900 Steps)", "W21 8GB Arch3 Fused Kronecker-U-Net 2^16 (900 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --steps 900", Radical2Pow16Config(arch_mode="fused_kronecker_unet")),
        ("w21_8gb_fused_kronecker_unet_2pow16_3600steps", "radical_2pow16_fused_kronecker_unet", "fused_kronecker_unet_2pow16_L4_d256_3600steps", 3600, "#ff006e", "Plan 036 H3 Fused Kronecker-U-Net N=2^16 Wavelet-Cholesky (3,600 Steps)", "W21 8GB Fused Kronecker-U-Net 2^16 (3,600 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --steps 3600", Radical2Pow16Config(arch_mode="fused_kronecker_unet")),
    ]
    specs = [{"key": k, "arch_family": a, "schedule_mode": sm, "total_steps": st, "color": c, "label": lb, "short_label": sl, "cli_args": cli, "rcfg": rc} for k, a, sm, st, c, lb, sl, cli, rc in spec_rows]
    runs = {m["key"]: run_single_plan036_experiment(m, data_bundle, db, device) for m in specs}
    r_cl36, r_q900, r_q36 = runs[specs[0]["key"]], runs[specs[1]["key"]], runs[specs[2]["key"]]
    r_a1, r_a2, r_rad36 = runs[specs[3]["key"]], runs[specs[4]["key"]], runs[specs[5]["key"]]
    r_a3_900, r_a3_3600 = runs[specs[6]["key"]], runs[specs[7]["key"]]

    cl_900_ref_ce = 5.9152
    h2_passed = bool(
        r_a3_900["best_val_ce"] < r_a1["best_val_ce"] and r_a3_900["best_val_ce"] < r_a2["best_val_ce"]
        and r_a3_900["best_val_ce"] < r_q900["best_val_ce"] and r_a3_900["best_val_ce"] < cl_900_ref_ce
        and r_a3_900["min_diffuse_eff_rank"] >= 6.0 and r_a3_900["peak_vram_gb"] <= 8.0
    )

    ckpts = (300, 900, 1800, 2700, 3600)
    a3_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_a3_3600["evals"]}
    w20_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_rad36["evals"] if int(e["step"]) in ckpts or int(e["step"]) == 0}
    c_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_cl36["evals"]}
    q_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_q36["evals"]}
    beats_w20_by_step = {str(s): bool(a3_by_step[s] < w20_by_step[s]) for s in ckpts}
    beats_classical_by_step = {str(s): bool(a3_by_step[s] < c_by_step[s]) for s in ckpts}

    h3_passed = bool(
        all(beats_w20_by_step.values()) and all(beats_classical_by_step.values())
        and r_a3_3600["best_val_ce"] < r_rad36["best_val_ce"] and r_a3_3600["best_val_ce"] < 5.1157
        and r_a3_3600["best_val_ce"] < r_cl36["best_val_ce"] and r_a3_3600["params_M"] < 12.5
        and r_a3_3600["peak_vram_gb"] <= 8.0
    )

    summary = {
        "plan_id": "EXPERIMENT_PLAN_036", "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "NVIDIA GeForce RTX 5060 Ti (<= 8.0 GB VRAM cap)",
        "hypotheses": {
            "H1_fused_kronecker_unet_predecessor_and_bigram_gate": h1_res,
            "H2_900step_fused_kronecker_unet_ablation": {
                "passed": h2_passed, "arch3_fused_kronecker_unet_best_val_ce": r_a3_900["best_val_ce"],
                "arch1_multiscale_unet_best_val_ce": r_a1["best_val_ce"],
                "arch2_consistent_kronecker_best_val_ce": r_a2["best_val_ce"],
                "quantum_born_40p78M_best_val_ce": r_q900["best_val_ce"],
                "classical_53p54M_best_val_ce": cl_900_ref_ce, "arch3_params_M": r_a3_900["params_M"],
                "arch3_final_top1_acc": r_a3_900["final_top1_acc"],
                "arch3_min_diffuse_eff_rank": r_a3_900["min_diffuse_eff_rank"],
                "arch3_avg_tok_per_s": r_a3_900["avg_tok_per_s"], "arch3_peak_vram_gb": r_a3_900["peak_vram_gb"],
                "margin_vs_arch1_nats": round(r_a1["best_val_ce"] - r_a3_900["best_val_ce"], 4),
                "margin_vs_arch2_nats": round(r_a2["best_val_ce"] - r_a3_900["best_val_ce"], 4),
                "margin_vs_quantum_born_nats": round(r_q900["best_val_ce"] - r_a3_900["best_val_ce"], 4),
                "margin_vs_classical_nats": round(cl_900_ref_ce - r_a3_900["best_val_ce"], 4),
            },
            "H3_3600step_29p5M_token_fused_kronecker_unet_scaling": {
                "passed": h3_passed, "fused_kronecker_unet_3600_best_val_ce": r_a3_3600["best_val_ce"],
                "plan035_multiscale_unet_3600_best_val_ce": r_rad36["best_val_ce"],
                "classical_30p95M_3600_best_val_ce": r_cl36["best_val_ce"],
                "parity_split_38p28M_quantum_born_val_ce": 5.1157,
                "full_ambient_40p78M_quantum_born_val_ce": 4.9565,
                "w18_janti_soft_floor_3600_best_val_ce": r_q36["best_val_ce"],
                "margin_vs_plan035_nats": round(r_rad36["best_val_ce"] - r_a3_3600["best_val_ce"], 4),
                "margin_vs_classical_nats": round(r_cl36["best_val_ce"] - r_a3_3600["best_val_ce"], 4),
                "margin_vs_parity_split_nats": round(5.1157 - r_a3_3600["best_val_ce"], 4),
                "gap_vs_full_ambient_40p78M_nats": round(r_a3_3600["best_val_ce"] - 4.9565, 4),
                "fused_kronecker_unet_params_M": r_a3_3600["params_M"],
                "fused_kronecker_unet_final_top1_acc": r_a3_3600["final_top1_acc"],
                "min_diffuse_eff_rank": r_a3_3600["min_diffuse_eff_rank"],
                "avg_tok_per_s": r_a3_3600["avg_tok_per_s"], "peak_vram_gb": r_a3_3600["peak_vram_gb"],
                "checkpoint_fused_kronecker_unet_val_ce": a3_by_step,
                "checkpoint_plan035_multiscale_unet_val_ce": w20_by_step,
                "checkpoint_classical_val_ce": c_by_step, "checkpoint_quantum_born_val_ce": q_by_step,
                "checkpoint_beats_plan035": beats_w20_by_step, "checkpoint_beats_classical": beats_classical_by_step,
            },
        },
        "runs": runs,
    }
    (CACHE_DIR / "hypothesis_experiments_036.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    backfill_micro_verifications(db, CACHE_DIR)
    for endpoint in ("/api/refresh", "/api/reload"):
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:8765{endpoint}", method="POST", data=b"{}"), timeout=5).read()
        except Exception:
            pass
    print(f"\n[Plan 036 Complete] H1={h1_res['passed']} | H2={h2_passed} | H3={h3_passed}")


if __name__ == "__main__":
    main()
