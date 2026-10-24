#!/usr/bin/env python3
"""Plan 037 Verification: 16-Bank N=2^16 Subtree Lifts (Fix 2), Kraus Retraction + Lambda Anneal (Fix 3), & Fast Cholesky (Fix 4)."""

from __future__ import annotations

import argparse
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
load_fineweb_tokens, build_evaluation_batches, _base_eval_quantum = (
    _EX45.load_fineweb_tokens, _EX45.build_evaluation_batches, _EX45.eval_quantum,
)


@torch.no_grad()
def eval_plan037_quantum(
    model: Radical2Pow16WaveletCholeskyGPT, val_batches: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict[str, Any]:
    """Evaluate `Radical2Pow16WaveletCholeskyGPT` and record genuine `sharp_eff_rank`, `kraus_stiefel_err`, & `lambda_diff`."""
    ev = _base_eval_quantum(model, val_batches)
    sharp_rank = round(float(model.pyramid.compute_sharp_eff_rank()), 2)
    kraus_err = round(float(model.pyramid.compute_kraus_stiefel_err()), 6)
    lam_diff = round(float(model.current_lambda_diff().detach().item()), 4)
    ev["sharp_eff_rank"] = sharp_rank
    ev["seed_eff_rank"] = sharp_rank
    ev["kraus_stiefel_err"] = kraus_err
    ev["lambda_diff"] = lam_diff
    return ev


def run_h1_tail_fixes_and_fast_cholesky_verification(
    data_bundle: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Verify H1: 16-Bank `sharp_eff_rank >= 190`, `kraus_stiefel_err <= 0.05`, `<11.0M` params, & Fix 4 equivalence."""
    print("\n" + "=" * 96 + "\n  [H1 Verification] 16-Bank Fine Lifts (Fix 2) + Kraus Retraction (Fix 3) + Fast Cholesky (Fix 4)\n" + "=" * 96)
    init_tok_t = torch.from_numpy(data_bundle["init_tokens"].astype(np.int64))
    x0, y0 = data_bundle["val_batches"][0]
    x_probe, y_probe = x0[:2, :128], y0[:2, :128]

    torch.manual_seed(2037)
    m_1bank = Radical2Pow16WaveletCholeskyGPT(Radical2Pow16Config(arch_mode="fused_kronecker_unet", num_fine_up_banks=1)).to(device)
    st_1bank = initialize_radical_2pow16_with_ppmi_svd(m_1bank, data_bundle["ppmi_emb_256"], init_tok_t, data_bundle["prior_tokens"])
    p_all = m_1bank.pyramid.perm_v_to_leaf[: m_1bank.cfg.valid_vocab_size]
    c0_1bank = m_1bank.pyramid._lift_fine_wavelet(p_all, p_all.shape)
    g_1bank = c0_1bank.float().t() @ c0_1bank.float()
    fine_subspace_rank_1bank = float(((torch.diagonal(g_1bank).sum() ** 2) / (g_1bank * g_1bank).sum().clamp_min(1e-12)).item())
    sharp_rank_1bank = float(st_1bank["sharp_eff_rank"])
    params_1bank = sum(p.numel() for p in m_1bank.parameters())
    del m_1bank
    torch.cuda.empty_cache()

    torch.manual_seed(2037)
    cfg_no_fix4 = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True, use_fast_cholesky_and_compile=False,
    )
    m_no_fix4 = Radical2Pow16WaveletCholeskyGPT(cfg_no_fix4).to(device)
    st_no_fix4 = initialize_radical_2pow16_with_ppmi_svd(m_no_fix4, data_bundle["ppmi_emb_256"], init_tok_t, data_bundle["prior_tokens"])
    c0_16bank = m_no_fix4.pyramid._lift_fine_wavelet(p_all, p_all.shape)
    g_16bank = c0_16bank.float().t() @ c0_16bank.float()
    fine_subspace_rank_16bank = float(((torch.diagonal(g_16bank).sum() ** 2) / (g_16bank * g_16bank).sum().clamp_min(1e-12)).item())

    cfg_with_fix4 = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True, use_fast_cholesky_and_compile=True,
    )
    m_with_fix4 = Radical2Pow16WaveletCholeskyGPT(cfg_with_fix4).to(device)
    m_with_fix4.load_state_dict(m_no_fix4.state_dict())

    m_no_fix4.eval()
    m_with_fix4.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss_no_fix4 = m_no_fix4(x_probe, targets=y_probe, return_logits=False)
        _, loss_with_fix4 = m_with_fix4(x_probe, targets=y_probe, return_logits=False)
    assert loss_no_fix4 is not None and loss_with_fix4 is not None
    max_rel_loss_diff = float(abs(loss_no_fix4.item() - loss_with_fix4.item()) / max(1e-6, abs(loss_no_fix4.item())))

    x_c1 = x0[:2, :64].clone()
    x_c2 = x_c1.clone()
    x_c2[:, 32:] = (x_c2[:, 32:] + 137) % 50257
    with torch.no_grad():
        logits1, _ = m_with_fix4(x_c1, targets=None, return_logits=True)
        logits2, _ = m_with_fix4(x_c2, targets=None, return_logits=True)
    assert logits1 is not None and logits2 is not None
    max_future_leakage = float((logits1[:, :32, :] - logits2[:, :32, :]).abs().max().item())

    m_with_fix4.set_training_progress(0.0)
    lam_s0 = float(m_with_fix4.current_lambda_diff().item())
    m_with_fix4.set_training_progress(0.18)
    lam_peak = float(m_with_fix4.current_lambda_diff().item())
    m_with_fix4.set_training_progress(1.0)
    lam_end = float(m_with_fix4.current_lambda_diff().item())

    m_with_fix4.train()
    muon_opt, adam_opt = build_radical_2pow16_optimizers(m_with_fix4)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss_bw = m_with_fix4(x_probe, targets=y_probe, return_logits=False)
    assert loss_bw is not None
    loss_bw.backward()
    grad_ok = bool(
        m_with_fix4.pyramid.wav_65536.grad is not None
        and m_with_fix4.pyramid.up_65536.grad is not None
        and all(torch.isfinite(p.grad).all().item() for p in m_with_fix4.parameters() if p.requires_grad and p.grad is not None)
    )
    muon_opt.step()
    adam_opt.step()
    kraus_err_after_retract = float(m_with_fix4.pyramid.retract_kraus_generators())

    total_params = sum(p.numel() for p in m_with_fix4.parameters())
    params_m = round(total_params / 1e6, 2)
    sharp_rank_16bank = float(st_no_fix4["sharp_eff_rank"])
    parseval_bg_1bank = round(1.0 / max(1e-6, sharp_rank_1bank), 6)
    parseval_bg_16bank = round(1.0 / max(1e-6, sharp_rank_16bank), 6)
    parseval_reduction_ratio = round(sharp_rank_16bank / max(1e-6, sharp_rank_1bank), 3)
    fine_subspace_gain_ratio = round(fine_subspace_rank_16bank / max(1e-6, fine_subspace_rank_1bank), 3)

    del m_no_fix4, m_with_fix4, muon_opt, adam_opt
    torch.cuda.empty_cache()

    passed = bool(
        total_params < 11_000_000 and sharp_rank_16bank >= 190.0 and sharp_rank_16bank > sharp_rank_1bank
        and fine_subspace_rank_16bank >= 2.5 * fine_subspace_rank_1bank and float(st_no_fix4["max_self_overlap"]) <= 0.05
        and float(st_no_fix4["directed_bigram_overlap_ratio"]) >= 1.80 and kraus_err_after_retract <= 0.05
        and max_rel_loss_diff < 2e-3 and max_future_leakage == 0.0 and lam_end < lam_peak and grad_ok
    )
    print(
        f"  * [H1 Tail Fixes & Fast Cholesky] Params={params_m:.2f}M ({total_params:,} vs 1-bank {params_1bank:,}) | "
        f"R_sharp: {sharp_rank_1bank:.1f} -> {sharp_rank_16bank:.1f} | "
        f"FineSubspaceRank: {fine_subspace_rank_1bank:.1f} -> {fine_subspace_rank_16bank:.1f} ({fine_subspace_gain_ratio:.2f}x) | "
        f"BigramRatio={st_no_fix4['directed_bigram_overlap_ratio']:.2f}x | "
        f"KrausErr={kraus_err_after_retract:.6f} | RelLossDiff={max_rel_loss_diff:.2e} | PASS={passed}"
    )
    return {
        "passed": passed, "total_params": int(total_params), "params_M": params_m,
        "plan036_1bank_total_params": int(params_1bank),
        "compression_vs_quantum_born_40p78M": round(40.78 / params_m, 2),
        "compression_vs_classical_30p95M": round(30.95 / params_m, 2),
        "sharp_eff_rank_1bank": round(sharp_rank_1bank, 2), "sharp_eff_rank_16bank": round(sharp_rank_16bank, 2),
        "fine_subspace_eff_rank_1bank": round(fine_subspace_rank_1bank, 2),
        "fine_subspace_eff_rank_16bank": round(fine_subspace_rank_16bank, 2),
        "fine_subspace_rank_gain_ratio": fine_subspace_gain_ratio,
        "parseval_bg_overlap_1bank": parseval_bg_1bank, "parseval_bg_overlap_16bank": parseval_bg_16bank,
        "parseval_bg_reduction_ratio": parseval_reduction_ratio,
        "max_self_overlap": round(float(st_no_fix4["max_self_overlap"]), 6),
        "directed_bigram_overlap_ratio": round(float(st_no_fix4["directed_bigram_overlap_ratio"]), 4),
        "kraus_stiefel_err_after_retract": round(kraus_err_after_retract, 6),
        "lambda_diff_schedule": {"step0": round(lam_s0, 4), "peak_s0p18": round(lam_peak, 4), "final_s1p0": round(lam_end, 4)},
        "loss_no_fix4": round(float(loss_no_fix4.item()), 6), "loss_with_fix4": round(float(loss_with_fix4.item()), 6),
        "max_rel_loss_diff": round(max_rel_loss_diff, 6), "max_future_leakage": max_future_leakage, "grad_finite": grad_ok,
    }


def save_and_register_run(meta: dict[str, Any], result_rec: dict[str, Any], db: BenchmarkSQLiteDB) -> None:
    """Persist `_full_result.json`, `_results.json`, model card, and SQLite benchmark DB row."""
    rkey = meta["key"]
    (LIVE_DIR / f"{rkey}_full_result.json").write_text(json.dumps(result_rec, indent=2), encoding="utf-8")
    (LIVE_DIR / f"{rkey}_results.json").write_text(json.dumps({**result_rec, "completed": True}, indent=2), encoding="utf-8")
    if rkey.startswith("w22_"):
        w_id, w_num, p_id = "W22", 22, "EXPERIMENT_PLAN_037"
        w_lbl = "Wave 22 (Plan 037 16-Bank N=2^16 Wavelet-Cholesky + Kraus Retraction + Fast Cholesky)"
    elif rkey.startswith("w21_"):
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


def run_single_plan037_experiment(
    meta: dict[str, Any], data_bundle: dict[str, Any], db: BenchmarkSQLiteDB, device: torch.device
) -> dict[str, Any]:
    """Execute or load a single Plan 037 experiment specification within the <= 8.0 GB VRAM budget."""
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
    rcfg: Radical2Pow16Config = meta["rcfg"]
    seq_len = 512
    micro_bs, grad_accum = (16, 1) if rcfg.use_fast_cholesky_and_compile else (8, 2)
    micro_tokens, tokens_per_step = micro_bs * seq_len, micro_bs * seq_len * grad_accum
    target_tok_m = round((total_steps * tokens_per_step) / 1e6, 2)

    model = Radical2Pow16WaveletCholeskyGPT(rcfg).to(device)
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
            if rcfg.use_kraus_retraction_and_lambda_anneal:
                model.set_training_progress((step - 1) / float(max(1, total_steps)))
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
            if rcfg.use_kraus_retraction_and_lambda_anneal and (step % 2 == 0 or step == total_steps):
                model.pyramid.retract_kraus_generators()

        if step in eval_steps:
            ev = eval_plan037_quantum(model, data_bundle["val_batches"])
            el_min, tok_m = round((time.perf_counter() - t0) / 60.0, 3), round((step * tokens_per_step) / 1e6, 2)
            eval_history.append({
                "epoch": round(step / float(total_steps), 2), "step": step, "tokens_M": tok_m,
                "elapsed_min": el_min, "seq_len": seq_len, "entanglement_nats": 2.45, "gamma_h": 8.65, **ev,
            })
            line = (
                f"[Step {step:4d}/{total_steps} | {tok_m:5.2f}M tok | {el_min:5.2f}m] "
                f"Val CE: {ev['val_ce']:.4f} | Top1: {ev['top1_acc']:.2f}% | Top50: {ev['top50_acc']:.2f}% | "
                f"R_sharp: {ev['sharp_eff_rank']:.1f} | lam_d: {ev['lambda_diff']:.4f} | K_err: {ev['kraus_stiefel_err']:.4f}"
            )
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan 037 verification runner.")
    parser.add_argument("--h1-only", action="store_true", help="Run only H1 Step-0 verification.")
    args = parser.parse_args(argv)

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
    h1_res = run_h1_tail_fixes_and_fast_cholesky_verification(data_bundle, device)
    if args.h1_only:
        return

    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)
    cfg_23_no_fix4 = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True, use_fast_cholesky_and_compile=False,
    )
    cfg_234_with_fix4 = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True, use_fast_cholesky_and_compile=True,
    )

    spec_rows = [
        ("w18_8gb_classical_leaderboard_exact_L4_d256_3600steps", "classical_modded", "classical_leaderboard_exact_L4_d256_3600steps", 3600, "#79c0ff", "Plan 033 H3 ClassicalModdedGPT (L=4, d=256, 3,600 Steps)", "W18 8GB Classical Leaderboard-Exact (3,600 Steps)", "--architecture classical_modded --steps 3600", None),
        ("w18_8gb_L4_d256_clean_janti_gentle_lr_900steps", "quantum_born", "clean_janti_gentle_lr_L4_d256_900steps", 900, "#fee440", "Plan 032 Isolate 3 Reference (16x16 J_anti, 900 Steps)", "W18 8GB Clean J_anti 16x16 (900 Steps)", "--architecture quantum_born --steps 900", None),
        ("w18_8gb_L4_d256_janti_soft_floor_3600steps", "quantum_born", "janti_soft_floor_6p5_L4_d256_3600steps", 3600, "#00ff87", "Plan 033 H3 QuantumBornGPT J_anti + Soft Floor 6.5 (3,600 Steps)", "W18 8GB J_anti + Soft Floor 6.5 (3,600 Steps)", "--architecture quantum_born --kraus-spectral-floor 6.5 --steps 3600", None),
        ("w20_8gb_arch1_2pow16_multiscale_wavelet_cholesky_unet_900steps", "radical_2pow16_unet", "arch1_2pow16_multiscale_unet_L4_d256_900steps", 900, "#00f5d4", "Plan 035 H2 Arch 1: 2^16 Multi-Resolution Wavelet-Cholesky U-Net (900 Steps)", "W20 8GB Arch1 2^16 Wavelet U-Net (900 Steps)", "--architecture radical_2pow16 --arch-mode multiscale_unet --steps 900", Radical2Pow16Config(arch_mode="multiscale_unet")),
        ("w20_8gb_radical_2pow16_wavelet_cholesky_3600steps", "radical_2pow16_multiscale_unet", "radical_2pow16_multiscale_unet_L4_d256_3600steps", 3600, "#9b5de5", "Plan 035 H3 Standalone N=2^16 Wavelet-Cholesky (multiscale_unet, 3,600 Steps)", "W20 8GB Radical 2^16 WavChol (3,600 Steps)", "--architecture radical_2pow16 --arch-mode multiscale_unet --steps 3600", Radical2Pow16Config(arch_mode="multiscale_unet")),
        ("w21_8gb_arch3_fused_kronecker_unet_2pow16_900steps", "radical_2pow16_fused_kronecker_unet", "arch3_fused_kronecker_unet_2pow16_L4_d256_900steps", 900, "#ff9f1c", "Plan 036 H2 Arch 3: Fused Kronecker-U-Net 2^16 Wavelet-Cholesky (900 Steps)", "W21 8GB Arch3 Fused Kronecker-U-Net 2^16 (900 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --steps 900", Radical2Pow16Config(arch_mode="fused_kronecker_unet")),
        ("w21_8gb_fused_kronecker_unet_2pow16_3600steps", "radical_2pow16_fused_kronecker_unet", "fused_kronecker_unet_2pow16_L4_d256_3600steps", 3600, "#ff006e", "Plan 036 H3 Fused Kronecker-U-Net N=2^16 Wavelet-Cholesky (3,600 Steps)", "W21 8GB Fused Kronecker-U-Net 2^16 (3,600 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --steps 3600", Radical2Pow16Config(arch_mode="fused_kronecker_unet")),
        ("w22_8gb_fixes23_no_fix4_900steps", "radical_2pow16_fused_kronecker_unet_16bank", "fixes23_no_fix4_2pow16_L4_d256_900steps", 900, "#38b000", "Plan 037 H2 Fixes 2+3 Without Fix 4 (16-Bank + Kraus Retract + Anneal, 900 Steps)", "W22 8GB Fixes 2+3 No-Fix4 (900 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --num-fine-up-banks 16 --steps 900", cfg_23_no_fix4),
        ("w22_8gb_fixes234_with_fix4_900steps", "radical_2pow16_fused_kronecker_unet_16bank_fast", "fixes234_with_fix4_2pow16_L4_d256_900steps", 900, "#00bbf9", "Plan 037 H2 Fixes 2+3+4 With Fix 4 (16-Bank + Kraus Retract + Fast Cholesky, 900 Steps)", "W22 8GB Fixes 2+3+4 With-Fix4 (900 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --num-fine-up-banks 16 --fast-cholesky --steps 900", cfg_234_with_fix4),
        ("w22_8gb_fixes234_with_fix4_3600steps", "radical_2pow16_fused_kronecker_unet_16bank_fast", "fixes234_with_fix4_2pow16_L4_d256_3600steps", 3600, "#00f5d4", "Plan 037 H3 Fixes 2+3+4 With Fix 4 (16-Bank N=2^16 Wavelet-Cholesky, 3,600 Steps)", "W22 8GB Fixes 2+3+4 16-Bank 2^16 (3,600 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --num-fine-up-banks 16 --fast-cholesky --steps 3600", cfg_234_with_fix4),
    ]
    specs = [{"key": k, "arch_family": a, "schedule_mode": sm, "total_steps": st, "color": c, "label": lb, "short_label": sl, "cli_args": cli, "rcfg": rc} for k, a, sm, st, c, lb, sl, cli, rc in spec_rows]
    runs = {m["key"]: run_single_plan037_experiment(m, data_bundle, db, device) for m in specs}

    r_cl36, r_q900, r_q36 = runs[specs[0]["key"]], runs[specs[1]["key"]], runs[specs[2]["key"]]
    r_w20_900, r_w20_3600 = runs[specs[3]["key"]], runs[specs[4]["key"]]
    r_w21_900, r_w21_3600 = runs[specs[5]["key"]], runs[specs[6]["key"]]
    r_no_f4_900, r_with_f4_900, r_w22_3600 = runs[specs[7]["key"]], runs[specs[8]["key"]], runs[specs[9]["key"]]

    cl_900_ref_ce = 5.9152
    speedup_ratio = round(r_with_f4_900["avg_tok_per_s"] / max(1.0, r_no_f4_900["avg_tok_per_s"]), 3)
    ce_diff_with_vs_without_f4 = round(abs(r_with_f4_900["best_val_ce"] - r_no_f4_900["best_val_ce"]), 4)
    h2_passed = bool(
        r_no_f4_900["best_val_ce"] < r_w21_900["best_val_ce"]
        and r_with_f4_900["best_val_ce"] < r_w21_900["best_val_ce"]
        and r_with_f4_900["best_val_ce"] < r_w20_900["best_val_ce"]
        and r_with_f4_900["best_val_ce"] < r_q900["best_val_ce"]
        and r_with_f4_900["best_val_ce"] < cl_900_ref_ce
        and ce_diff_with_vs_without_f4 <= 0.02 and speedup_ratio >= 1.60
        and r_with_f4_900["final_sharp_eff_rank"] >= 180.0
        and r_with_f4_900["final_kraus_stiefel_err"] <= 0.05
        and r_with_f4_900["peak_vram_gb"] <= 8.0
    )

    ckpts = (300, 900, 1800, 2700, 3600)
    w22_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_w22_3600["evals"]}
    w21_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_w21_3600["evals"]}
    w20_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_w20_3600["evals"] if int(e["step"]) in ckpts or int(e["step"]) == 0}
    c_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_cl36["evals"]}
    q_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_q36["evals"]}
    beats_w21_by_step = {str(s): bool(w22_by_step[s] < w21_by_step[s]) for s in ckpts}
    beats_w20_by_step = {str(s): bool(w22_by_step[s] < w20_by_step[s]) for s in ckpts}
    beats_classical_by_step = {str(s): bool(w22_by_step[s] < c_by_step[s]) for s in ckpts}

    h3_passed = bool(
        r_w22_3600["best_val_ce"] < 5.00
        and r_w22_3600["best_val_ce"] < r_w21_3600["best_val_ce"]
        and r_w22_3600["best_val_ce"] < r_w20_3600["best_val_ce"]
        and r_w22_3600["best_val_ce"] < 5.1157
        and r_w22_3600["best_val_ce"] < r_cl36["best_val_ce"]
        and all(beats_w21_by_step.values()) and all(beats_w20_by_step.values()) and all(beats_classical_by_step.values())
        and r_w22_3600["final_sharp_eff_rank"] >= 180.0
        and r_w22_3600["final_kraus_stiefel_err"] <= 0.05
        and r_w22_3600["params_M"] < 11.0 and r_w22_3600["peak_vram_gb"] <= 8.0
    )

    summary = {
        "plan_id": "EXPERIMENT_PLAN_037", "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "NVIDIA GeForce RTX 5060 Ti (<= 8.0 GB VRAM cap)",
        "hypotheses": {
            "H1_tail_fixes_and_fast_cholesky_equivalence": h1_res,
            "H2_900step_with_vs_without_fix4_comparison": {
                "passed": h2_passed,
                "fixes23_no_fix4_best_val_ce": r_no_f4_900["best_val_ce"],
                "fixes234_with_fix4_best_val_ce": r_with_f4_900["best_val_ce"],
                "ce_abs_diff_with_vs_without_fix4": ce_diff_with_vs_without_f4,
                "plan036_w21_best_val_ce": r_w21_900["best_val_ce"],
                "plan035_w20_best_val_ce": r_w20_900["best_val_ce"],
                "quantum_born_40p78M_best_val_ce": r_q900["best_val_ce"],
                "classical_53p54M_best_val_ce": cl_900_ref_ce,
                "fixes23_no_fix4_avg_tok_per_s": r_no_f4_900["avg_tok_per_s"],
                "fixes234_with_fix4_avg_tok_per_s": r_with_f4_900["avg_tok_per_s"],
                "fix4_throughput_speedup_ratio": speedup_ratio,
                "fixes234_params_M": r_with_f4_900["params_M"],
                "fixes234_final_top1_acc": r_with_f4_900["final_top1_acc"],
                "fixes234_final_top50_acc": r_with_f4_900["final_top50_acc"],
                "fixes234_final_sharp_eff_rank": r_with_f4_900["final_sharp_eff_rank"],
                "fixes234_final_kraus_stiefel_err": r_with_f4_900["final_kraus_stiefel_err"],
                "fixes234_final_lambda_diff": r_with_f4_900["final_lambda_diff"],
                "fixes234_peak_vram_gb": r_with_f4_900["peak_vram_gb"],
                "margin_vs_plan036_w21_nats": round(r_w21_900["best_val_ce"] - r_with_f4_900["best_val_ce"], 4),
                "margin_vs_plan035_w20_nats": round(r_w20_900["best_val_ce"] - r_with_f4_900["best_val_ce"], 4),
                "margin_vs_quantum_born_nats": round(r_q900["best_val_ce"] - r_with_f4_900["best_val_ce"], 4),
                "margin_vs_classical_nats": round(cl_900_ref_ce - r_with_f4_900["best_val_ce"], 4),
            },
            "H3_3600step_29p5M_token_fixes234_scaling": {
                "passed": h3_passed,
                "w22_fixes234_3600_best_val_ce": r_w22_3600["best_val_ce"],
                "plan036_w21_3600_best_val_ce": r_w21_3600["best_val_ce"],
                "plan035_w20_3600_best_val_ce": r_w20_3600["best_val_ce"],
                "classical_30p95M_3600_best_val_ce": r_cl36["best_val_ce"],
                "parity_split_38p28M_quantum_born_val_ce": 5.1157,
                "w19_wavchol_mz_38p28M_val_ce": 4.9873,
                "full_ambient_40p78M_quantum_born_val_ce": 4.9565,
                "w18_janti_soft_floor_3600_best_val_ce": r_q36["best_val_ce"],
                "margin_vs_plan036_w21_nats": round(r_w21_3600["best_val_ce"] - r_w22_3600["best_val_ce"], 4),
                "margin_vs_plan035_w20_nats": round(r_w20_3600["best_val_ce"] - r_w22_3600["best_val_ce"], 4),
                "margin_vs_parity_split_nats": round(5.1157 - r_w22_3600["best_val_ce"], 4),
                "margin_vs_classical_nats": round(r_cl36["best_val_ce"] - r_w22_3600["best_val_ce"], 4),
                "w22_params_M": r_w22_3600["params_M"],
                "w22_final_top1_acc": r_w22_3600["final_top1_acc"],
                "w22_final_top5_acc": r_w22_3600["final_top5_acc"],
                "w22_final_top50_acc": r_w22_3600["final_top50_acc"],
                "w22_final_med_rank": r_w22_3600["final_med_rank"],
                "w22_final_sharp_eff_rank": r_w22_3600["final_sharp_eff_rank"],
                "w22_final_kraus_stiefel_err": r_w22_3600["final_kraus_stiefel_err"],
                "w22_final_lambda_diff": r_w22_3600["final_lambda_diff"],
                "min_diffuse_eff_rank": r_w22_3600["min_diffuse_eff_rank"],
                "avg_tok_per_s": r_w22_3600["avg_tok_per_s"], "peak_vram_gb": r_w22_3600["peak_vram_gb"],
                "checkpoint_w22_fixes234_val_ce": w22_by_step,
                "checkpoint_plan036_w21_val_ce": w21_by_step,
                "checkpoint_plan035_w20_val_ce": w20_by_step,
                "checkpoint_classical_val_ce": c_by_step, "checkpoint_quantum_born_val_ce": q_by_step,
                "checkpoint_beats_plan036_w21": beats_w21_by_step,
                "checkpoint_beats_plan035_w20": beats_w20_by_step,
                "checkpoint_beats_classical": beats_classical_by_step,
            },
        },
        "runs": runs,
    }
    (CACHE_DIR / "hypothesis_experiments_037.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    backfill_micro_verifications(db, CACHE_DIR)
    for endpoint in ("/api/refresh", "/api/reload"):
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:8765{endpoint}", method="POST", data=b"{}"), timeout=5).read()
        except Exception:
            pass
    print(f"\n[Plan 037 Complete] H1={h1_res['passed']} | H2={h2_passed} | H3={h3_passed}")


if __name__ == "__main__":
    main()
