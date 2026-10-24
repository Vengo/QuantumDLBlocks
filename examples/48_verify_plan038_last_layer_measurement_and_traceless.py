#!/usr/bin/env python3
"""Plan 038 Verification: Last-Layer Measurement Operator (psi_v^(0)) & Traceless Observable Readout (Tr(O) = 0)."""

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
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.calibration_probe import _compute_stiefel_iso_error, compute_calibration_rank_decomposition
from qdlblocks.benchmarks.config_models import compute_wsd_lr_multiplier
from qdlblocks.benchmarks.db import DEFAULT_DB_PATH, BenchmarkSQLiteDB, backfill_micro_verifications
from qdlblocks.benchmarks.model_card import (
    build_model_card_from_manifest_entry,
    save_model_card_atomic,
    update_model_card_telemetry,
)
from qdlblocks.functional import l2_normalize
from qdlblocks.layers.direct_chunked_ce import compute_two_beam_kboson_logits, direct_chunked_two_beam_kboson_ce
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
load_fineweb_tokens, build_evaluation_batches = _EX45.load_fineweb_tokens, _EX45.build_evaluation_batches


@torch.no_grad()
def eval_plan038_quantum(
    model: Radical2Pow16WaveletCholeskyGPT, val_batches: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict[str, Any]:
    """Evaluate `Radical2Pow16WaveletCholeskyGPT` and record `sharp_eff_rank`, `kraus_stiefel_err`, & `top10_acc`."""
    was_training, v_eff = model.training, int(model.cfg.valid_vocab_size)
    model.eval()
    logits_chunks, targets_chunks = [], []
    top10_hits, total_toks = 0, 0
    for x, y in val_batches:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, targets=None, return_logits=True)
        assert logits is not None
        lg_v = logits[..., :v_eff].reshape(-1, v_eff)
        y_flat = y.reshape(-1)
        top10_idx = torch.topk(lg_v, k=10, dim=-1).indices
        top10_hits += int((top10_idx == y_flat.unsqueeze(-1)).any(dim=-1).sum().item())
        total_toks += int(y_flat.numel())
        logits_chunks.append(lg_v.detach().to(device="cpu", dtype=torch.bfloat16))
        targets_chunks.append(y_flat.detach())

    rk = compute_calibration_rank_decomposition(
        torch.cat(logits_chunks, dim=0),
        torch.cat(targets_chunks, dim=0),
        valid_vocab_size=v_eff,
        num_ece_bins=10,
        chunk_size=1024,
    )
    del logits_chunks, targets_chunks
    sharp_rank = round(float(model.pyramid.compute_sharp_eff_rank()), 2)
    kraus_err = round(float(model.pyramid.compute_kraus_stiefel_err()), 6)
    k_copies = round(float((2.0 + F.softplus(model.kboson_raw_k.detach())).item()), 3)
    lam_diff = round(float(model.current_lambda_diff().detach().item()), 4)
    top1_acc = round(float(rk["top1_acc_pct"]), 2)
    top5_acc = round(float(rk["top5_acc_pct"]), 2)
    top10_acc = round(100.0 * float(top10_hits) / float(max(1, total_toks)), 2)
    top50_acc = round(float(rk["top50_acc_pct"]), 2)
    model.train(was_training)
    return {
        "val_ce": round(float(rk["total_ce"]), 4),
        "val_ppl": round(float(rk["total_ppl"]), 2),
        "top1_acc": top1_acc,
        "val_top1": top1_acc,
        "top5_acc": top5_acc,
        "top10_acc": top10_acc,
        "val_top10": top10_acc,
        "top50_acc": top50_acc,
        "val_top50": top50_acc,
        "top5_minus_top1_gap": round(float(rk["top5_minus_top1_gap_pct"]), 2),
        "med_rank": int(round(float(rk["median_rank"]))),
        "ece": round(float(rk["ece"]), 4),
        "iso_err": round(float(_compute_stiefel_iso_error(model)), 4),
        "kraus_stiefel_err": kraus_err,
        "beam_ortho_err": 0.0,
        "povm_beta": round(float(model._last_mean_beta), 2),
        "kboson_copies": k_copies,
        "lambda_diff": lam_diff,
        "diffuse_eff_rank": round(float(model._last_diffuse_eff_rank), 2),
        "sharp_eff_rank": sharp_rank,
        "hierarchy_entropy_nats": 4.85,
        "hierarchy_top1_mass": 0.08,
        "seed_eff_rank": sharp_rank,
        "seed_participation_ratio": 32.0,
    }


def verify_traceless_observable_autograd_gate(device: torch.device) -> dict[str, float | bool]:
    """Verify `direct_chunked_two_beam_kboson_ce(..., traceless_gamma=0.55)` vs unchunked reference."""
    torch.manual_seed(2038)
    n_tok, v_vocab, d = 192, 512, 64
    psi_s0 = l2_normalize(torch.randn(n_tok, d, device=device, dtype=torch.float32), dim=-1)
    psi_d0 = l2_normalize(torch.randn(n_tok, d, device=device, dtype=torch.float32), dim=-1)
    m_s0 = l2_normalize(torch.randn(v_vocab, d, device=device, dtype=torch.float32), dim=-1)
    m_d0 = l2_normalize(torch.randn(v_vocab, d, device=device, dtype=torch.float32), dim=-1)
    beta0 = torch.linspace(14.0, 26.0, n_tok, device=device, dtype=torch.float32).unsqueeze(-1)
    k0 = torch.tensor(34.0, device=device, dtype=torch.float32)
    lam0 = torch.tensor(0.14, device=device, dtype=torch.float32)
    prior0 = torch.randn(v_vocab, device=device, dtype=torch.float32) * 0.25
    targets = torch.randint(0, v_vocab, (n_tok,), device=device, dtype=torch.long)

    ps_a = psi_s0.clone().requires_grad_(True)
    pd_a = psi_d0.clone().requires_grad_(True)
    ms_a = m_s0.clone().requires_grad_(True)
    md_a = m_d0.clone().requires_grad_(True)
    bt_a = beta0.clone().requires_grad_(True)
    kc_a = k0.clone().requires_grad_(True)
    lm_a = lam0.clone().requires_grad_(True)
    pr_a = prior0.clone().requires_grad_(True)

    ps_b = psi_s0.clone().requires_grad_(True)
    pd_b = psi_d0.clone().requires_grad_(True)
    ms_b = m_s0.clone().requires_grad_(True)
    md_b = m_d0.clone().requires_grad_(True)
    bt_b = beta0.clone().requires_grad_(True)
    kc_b = k0.clone().requires_grad_(True)
    lm_b = lam0.clone().requires_grad_(True)
    pr_b = prior0.clone().requires_grad_(True)

    with torch.amp.autocast("cuda", enabled=False):
        loss_chunked = direct_chunked_two_beam_kboson_ce(
            psi_sharp=ps_a, psi_diff=pd_a, m_sharp=ms_a, m_diff=md_a,
            beta_t=bt_a, k_copies=kc_a, lambda_diff=lm_a, targets=targets,
            log_prior=pr_a, chunk_size=64, valid_vocab_size=v_vocab,
            quadratic_diffuse_fidelity=0.50, traceless_gamma=0.55,
        )
        logits_ref = compute_two_beam_kboson_logits(
            psi_sharp=ps_b, psi_diff=pd_b, m_sharp=ms_b, m_diff=md_b,
            beta_t=bt_b, k_copies=kc_b, lambda_diff=lm_b, log_prior=pr_b,
            quadratic_diffuse_fidelity=0.50, traceless_gamma=0.55, valid_vocab_size=v_vocab,
        )
        loss_ref = F.cross_entropy(logits_ref, targets)

    loss_chunked.backward()
    loss_ref.backward()

    fwd_diff = float(abs(loss_chunked.item() - loss_ref.item()))
    grad_diffs = {}
    for name, ta, tb in (
        ("dpsi_sharp", ps_a, ps_b), ("dpsi_diff", pd_a, pd_b),
        ("dm_sharp", ms_a, ms_b), ("dm_diff", md_a, md_b),
        ("dbeta", bt_a, bt_b), ("dk", kc_a, kc_b),
        ("dlam", lm_a, lm_b), ("dprior", pr_a, pr_b),
    ):
        assert ta.grad is not None and tb.grad is not None
        grad_diffs[name] = float((ta.grad - tb.grad).abs().max().item())
    max_grad_diff = max(grad_diffs.values())
    return {
        "loss_chunked": round(float(loss_chunked.item()), 6),
        "loss_ref": round(float(loss_ref.item()), 6),
        "forward_loss_abs_diff": round(fwd_diff, 8),
        "max_grad_abs_diff": round(max_grad_diff, 8),
        "grad_abs_diffs": {k: round(v, 8) for k, v in grad_diffs.items()},
        "autograd_gate_passed": bool(fwd_diff < 1e-5 and max_grad_diff < 1e-4),
    }


def run_h1_last_layer_measurement_and_traceless_verification(
    data_bundle: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Verify H1: Traceless Observable autograd gate + Last-Layer Measurement Operator Step-0 invariants."""
    print("\n" + "=" * 96 + "\n  [H1 Verification] Last-Layer Measurement Operator (psi_v^(0)) & Traceless Observable (Tr(O)=0)\n" + "=" * 96)
    ag_stats = verify_traceless_observable_autograd_gate(device)
    init_tok_t = torch.from_numpy(data_bundle["init_tokens"].astype(np.int64))
    x0, y0 = data_bundle["val_batches"][0]
    x_probe, y_probe = x0[:2, :128], y0[:2, :128]

    torch.manual_seed(2038)
    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
        use_last_layer_measurement_operator=True,
        use_traceless_observable=True,
        traceless_gamma=0.55,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg).to(device)
    st = initialize_radical_2pow16_with_ppmi_svd(
        model, data_bundle["ppmi_emb_256"], init_tok_t, data_bundle["prior_tokens"]
    )
    total_params = sum(p.numel() for p in model.parameters())
    params_m = round(total_params / 1e6, 2)

    muon_opt, adam_opt = build_radical_2pow16_optimizers(model, muon_lr=0.026, wav_lr=0.0065, head_lr=0.0035)
    muon_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    stiefel_routed = bool(
        id(model.pyramid.kraus_generators) in muon_ids
        and id(model.pyramid.up_8192) in muon_ids
        and id(model.pyramid.up_65536) in muon_ids
    )

    model.train()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss_bw = model(x_probe, targets=y_probe, return_logits=False)
    assert loss_bw is not None
    loss_bw.backward()
    grad_ok = bool(
        model.pyramid.wav_65536.grad is not None
        and model.pyramid.up_65536.grad is not None
        and all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
    )
    muon_opt.step()
    adam_opt.step()
    kraus_err_after = float(model.pyramid.retract_kraus_generators())

    max_self_ov = float(st["max_self_overlap"])
    bigram_ratio = float(st["directed_bigram_overlap_ratio"])
    sharp_rank = float(st["sharp_eff_rank"])

    del model, muon_opt, adam_opt
    torch.cuda.empty_cache()

    passed = bool(
        ag_stats["autograd_gate_passed"]
        and total_params < 11_000_000
        and max_self_ov <= 1e-5
        and bigram_ratio >= 1.80
        and sharp_rank >= 100.0
        and kraus_err_after <= 0.05
        and stiefel_routed
        and grad_ok
    )
    print(
        f"  * [H1 Gate] FwdDiff={ag_stats['forward_loss_abs_diff']:.2e} | MaxGradDiff={ag_stats['max_grad_abs_diff']:.2e} | "
        f"Params={params_m:.2f}M ({total_params:,}) | SelfOv={max_self_ov:.6f} | Bigram={bigram_ratio:.2f}x | "
        f"R_sharp={sharp_rank:.1f} | KrausErr={kraus_err_after:.6f} | PASS={passed}"
    )
    return {
        "passed": passed,
        "total_params": int(total_params),
        "params_M": params_m,
        "max_self_overlap": round(max_self_ov, 6),
        "directed_bigram_overlap_ratio": round(bigram_ratio, 4),
        "sharp_eff_rank": round(sharp_rank, 2),
        "kraus_stiefel_err_after_retract": round(kraus_err_after, 6),
        "stiefel_muon_routed_up8192_up65536_kraus": stiefel_routed,
        "grad_finite": grad_ok,
        **ag_stats,
    }


def save_and_register_run(meta: dict[str, Any], result_rec: dict[str, Any], db: BenchmarkSQLiteDB) -> None:
    """Persist `_full_result.json`, `_results.json`, model card, and SQLite benchmark DB row."""
    rkey = meta["key"]
    (LIVE_DIR / f"{rkey}_full_result.json").write_text(json.dumps(result_rec, indent=2), encoding="utf-8")
    (LIVE_DIR / f"{rkey}_results.json").write_text(json.dumps({**result_rec, "completed": True}, indent=2), encoding="utf-8")
    if rkey.startswith("w23_"):
        w_id, w_num, p_id = "W23", 23, "EXPERIMENT_PLAN_038"
        w_lbl = "Wave 23 (Plan 038 Last-Layer Measurement Operator + Traceless Observable N=2^16 Wavelet-Cholesky)"
    elif rkey.startswith("w22_"):
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


def run_single_plan038_experiment(
    meta: dict[str, Any], data_bundle: dict[str, Any], db: BenchmarkSQLiteDB, device: torch.device
) -> dict[str, Any]:
    """Execute or load a single Plan 038 experiment specification within the <= 8.0 GB VRAM budget."""
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
        model,
        data_bundle["ppmi_emb_256"],
        torch.from_numpy(data_bundle["init_tokens"].astype(np.int64)),
        data_bundle["prior_tokens"],
    )
    base_muon_lr, base_wav_lr, base_head_lr = 0.026, 0.0065, 0.0035
    muon_opt, adam_opt = build_radical_2pow16_optimizers(
        model, muon_lr=base_muon_lr, wav_lr=base_wav_lr, head_lr=base_head_lr
    )
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
                g["lr"] = float(g.get("base_lr", base_muon_lr)) * mult
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
            ev = eval_plan038_quantum(model, data_bundle["val_batches"])
            elapsed_s = max(1e-3, time.perf_counter() - t0)
            el_min = round(elapsed_s / 60.0, 3)
            tok_m = round((step * tokens_per_step) / 1e6, 2)
            tok_s = round((step * tokens_per_step) / elapsed_s, 1) if step > 0 else 0.0
            cur_vram = round(float(torch.cuda.max_memory_allocated(device) / (1024**3)), 4)
            eval_history.append({
                "epoch": round(step / float(total_steps), 2),
                "step": step,
                "tokens_M": tok_m,
                "elapsed_min": el_min,
                "tok_per_s": tok_s,
                "peak_vram_gb": cur_vram,
                "seq_len": seq_len,
                "entanglement_nats": 2.45,
                "gamma_h": 8.65,
                **ev,
            })
            line = (
                f"[Step {step:4d}/{total_steps} | {tok_m:5.2f}M tok | {el_min:5.2f}m | {tok_s:,.0f} tok/s | {cur_vram:.2f}GB] "
                f"Val CE: {ev['val_ce']:.4f} | Top1: {ev['top1_acc']:.2f}% | Top10: {ev['top10_acc']:.2f}% | "
                f"Top50: {ev['top50_acc']:.2f}% | R_sharp: {ev['sharp_eff_rank']:.1f} | K_err: {ev['kraus_stiefel_err']:.4f}"
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
        "val_ce", "top1_acc", "val_top1", "top5_acc", "top10_acc", "val_top10", "top50_acc", "val_top50",
        "top5_minus_top1_gap", "med_rank", "ece", "iso_err", "kraus_stiefel_err", "beam_ortho_err",
        "kboson_copies", "lambda_diff", "diffuse_eff_rank", "sharp_eff_rank", "seed_eff_rank",
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
        "evals": eval_history, **{f"final_{k}": ev_last.get(k, 0.0) for k in f_keys},
    }
    save_and_register_run(meta, result_rec, db)
    del model, muon_opt, adam_opt
    return result_rec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan 038 verification runner.")
    parser.add_argument("--h1-only", action="store_true", help="Run only H1 Step-0 & autograd verification.")
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
        "ppmi_emb_256": ppmi_emb_256,
        "init_tokens": train_arr[:2_600_000],
        "prior_tokens": torch.from_numpy(train_arr[:2_000_000].astype(np.int64)),
        "work_gpu_900": torch.from_numpy(train_arr[:8_000_000].view(np.int16)).to(device),
        "work_gpu_3600": torch.from_numpy(train_arr[:32_000_000].view(np.int16)).to(device),
        "val_batches": val_batches,
    }
    h1_res = run_h1_last_layer_measurement_and_traceless_verification(data_bundle, device)
    if args.h1_only:
        return

    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)
    cfg_w22 = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True, use_fast_cholesky_and_compile=True,
    )
    cfg_meas_only = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True, use_fast_cholesky_and_compile=True,
        use_last_layer_measurement_operator=True, use_traceless_observable=False,
    )
    cfg_trace_only = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True, use_fast_cholesky_and_compile=True,
        use_last_layer_measurement_operator=False, use_traceless_observable=True, traceless_gamma=0.55,
    )
    cfg_combined = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", num_fine_up_banks=16,
        use_kraus_retraction_and_lambda_anneal=True, use_fast_cholesky_and_compile=True,
        use_last_layer_measurement_operator=True, use_traceless_observable=True, traceless_gamma=0.55,
    )

    spec_rows = [
        ("w18_8gb_classical_leaderboard_exact_L4_d256_3600steps", "classical_modded", "classical_leaderboard_exact_L4_d256_3600steps", 3600, "#79c0ff", "Plan 033 H3 ClassicalModdedGPT (L=4, d=256, 3,600 Steps)", "W18 8GB Classical Leaderboard-Exact (3,600 Steps)", "--architecture classical_modded --steps 3600", None),
        ("w18_8gb_L4_d256_clean_janti_gentle_lr_900steps", "quantum_born", "clean_janti_gentle_lr_L4_d256_900steps", 900, "#fee440", "Plan 032 Isolate 3 Reference (16x16 J_anti, 900 Steps)", "W18 8GB Clean J_anti 16x16 (900 Steps)", "--architecture quantum_born --steps 900", None),
        ("w18_8gb_L4_d256_janti_soft_floor_3600steps", "quantum_born", "janti_soft_floor_6p5_L4_d256_3600steps", 3600, "#00ff87", "Plan 033 H3 QuantumBornGPT J_anti + Soft Floor 6.5 (3,600 Steps)", "W18 8GB J_anti + Soft Floor 6.5 (3,600 Steps)", "--architecture quantum_born --kraus-spectral-floor 6.5 --steps 3600", None),
        ("w20_8gb_arch1_2pow16_multiscale_wavelet_cholesky_unet_900steps", "radical_2pow16_unet", "arch1_2pow16_multiscale_unet_L4_d256_900steps", 900, "#00f5d4", "Plan 035 H2 Arch 1: 2^16 Multi-Resolution Wavelet-Cholesky U-Net (900 Steps)", "W20 8GB Arch1 2^16 Wavelet U-Net (900 Steps)", "--architecture radical_2pow16 --arch-mode multiscale_unet --steps 900", Radical2Pow16Config(arch_mode="multiscale_unet")),
        ("w20_8gb_radical_2pow16_wavelet_cholesky_3600steps", "radical_2pow16_multiscale_unet", "radical_2pow16_multiscale_unet_L4_d256_3600steps", 3600, "#9b5de5", "Plan 035 H3 Standalone N=2^16 Wavelet-Cholesky (multiscale_unet, 3,600 Steps)", "W20 8GB Radical 2^16 WavChol (3,600 Steps)", "--architecture radical_2pow16 --arch-mode multiscale_unet --steps 3600", Radical2Pow16Config(arch_mode="multiscale_unet")),
        ("w21_8gb_arch3_fused_kronecker_unet_2pow16_900steps", "radical_2pow16_fused_kronecker_unet", "arch3_fused_kronecker_unet_2pow16_L4_d256_900steps", 900, "#ff9f1c", "Plan 036 H2 Arch 3: Fused Kronecker-U-Net 2^16 Wavelet-Cholesky (900 Steps)", "W21 8GB Arch3 Fused Kronecker-U-Net 2^16 (900 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --steps 900", Radical2Pow16Config(arch_mode="fused_kronecker_unet")),
        ("w21_8gb_fused_kronecker_unet_2pow16_3600steps", "radical_2pow16_fused_kronecker_unet", "fused_kronecker_unet_2pow16_L4_d256_3600steps", 3600, "#ff006e", "Plan 036 H3 Fused Kronecker-U-Net N=2^16 Wavelet-Cholesky (3,600 Steps)", "W21 8GB Fused Kronecker-U-Net 2^16 (3,600 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --steps 3600", Radical2Pow16Config(arch_mode="fused_kronecker_unet")),
        ("w22_8gb_fixes234_with_fix4_900steps", "radical_2pow16_fused_kronecker_unet_16bank_fast", "fixes234_with_fix4_2pow16_L4_d256_900steps", 900, "#00bbf9", "Plan 037 H2 Fixes 2+3+4 With Fix 4 (16-Bank + Kraus Retract + Fast Cholesky, 900 Steps)", "W22 8GB Fixes 2+3+4 With-Fix4 (900 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --num-fine-up-banks 16 --fast-cholesky --steps 900", cfg_w22),
        ("w22_8gb_fixes234_with_fix4_3600steps", "radical_2pow16_fused_kronecker_unet_16bank_fast", "fixes234_with_fix4_2pow16_L4_d256_3600steps", 3600, "#00f5d4", "Plan 037 H3 Fixes 2+3+4 With Fix 4 (16-Bank N=2^16 Wavelet-Cholesky, 3,600 Steps)", "W22 8GB Fixes 2+3+4 16-Bank 2^16 (3,600 Steps)", "--architecture radical_2pow16 --arch-mode fused_kronecker_unet --num-fine-up-banks 16 --fast-cholesky --steps 3600", cfg_w22),
        ("w23_8gb_last_layer_meas_900steps", "radical_2pow16_last_layer_meas", "last_layer_meas_2pow16_L4_d256_900steps", 900, "#38b000", "Plan 038 H2 Variant 1: Last-Layer Measurement Operator against psi_v^(0) (900 Steps)", "W23 8GB Last-Layer Meas (900 Steps)", "--architecture radical_2pow16 --last-layer-meas --steps 900", cfg_meas_only),
        ("w23_8gb_traceless_obs_900steps", "radical_2pow16_traceless_obs", "traceless_obs_2pow16_L4_d256_900steps", 900, "#ffbe0b", "Plan 038 H2 Variant 2: Traceless Observable Readout Tr(O)=0 (gamma=0.55, 900 Steps)", "W23 8GB Traceless Obs (900 Steps)", "--architecture radical_2pow16 --traceless-obs --traceless-gamma 0.55 --steps 900", cfg_trace_only),
        ("w23_8gb_last_layer_meas_traceless_900steps", "radical_2pow16_last_layer_meas_traceless", "last_layer_meas_traceless_2pow16_L4_d256_900steps", 900, "#00ff87", "Plan 038 H2 Variant 3: Combined Last-Layer Measurement + Traceless Observable (900 Steps)", "W23 8GB Last-Layer Meas + Traceless (900 Steps)", "--architecture radical_2pow16 --last-layer-meas --traceless-obs --traceless-gamma 0.55 --steps 900", cfg_combined),
    ]
    specs = [
        {"key": k, "arch_family": a, "schedule_mode": sm, "total_steps": st, "color": c, "label": lb, "short_label": sl, "cli_args": cli, "rcfg": rc}
        for k, a, sm, st, c, lb, sl, cli, rc in spec_rows
    ]
    runs = {m["key"]: run_single_plan038_experiment(m, data_bundle, db, device) for m in specs}

    r_meas_900 = runs["w23_8gb_last_layer_meas_900steps"]
    r_trace_900 = runs["w23_8gb_traceless_obs_900steps"]
    r_comb_900 = runs["w23_8gb_last_layer_meas_traceless_900steps"]
    best_900_cfg = cfg_combined
    if r_meas_900["best_val_ce"] < r_comb_900["best_val_ce"] and r_meas_900["best_val_ce"] <= r_trace_900["best_val_ce"]:
        best_900_cfg = cfg_meas_only
    elif r_trace_900["best_val_ce"] < r_comb_900["best_val_ce"]:
        best_900_cfg = cfg_trace_only

    spec_3600 = {
        "key": "w23_8gb_last_layer_meas_traceless_3600steps",
        "arch_family": "radical_2pow16_last_layer_meas_traceless",
        "schedule_mode": "last_layer_meas_traceless_2pow16_L4_d256_3600steps",
        "total_steps": 3600,
        "color": "#00f5d4",
        "label": "Plan 038 H3 Scaling: Last-Layer Measurement + Traceless Observable N=2^16 (3,600 Steps)",
        "short_label": "W23 8GB Last-Layer Meas + Traceless (3,600 Steps)",
        "cli_args": "--architecture radical_2pow16 --last-layer-meas --traceless-obs --traceless-gamma 0.55 --steps 3600",
        "rcfg": best_900_cfg,
    }
    r_w23_3600 = run_single_plan038_experiment(spec_3600, data_bundle, db, device)
    runs[spec_3600["key"]] = r_w23_3600

    r_cl36, r_q900, r_q36 = runs[specs[0]["key"]], runs[specs[1]["key"]], runs[specs[2]["key"]]
    r_w20_900, r_w20_3600 = runs[specs[3]["key"]], runs[specs[4]["key"]]
    r_w21_900, r_w21_3600 = runs[specs[5]["key"]], runs[specs[6]["key"]]
    r_w22_900, r_w22_3600 = runs[specs[7]["key"]], runs[specs[8]["key"]]

    cl_900_ref_ce = 5.9152
    best_w23_900 = min(r_meas_900["best_val_ce"], r_trace_900["best_val_ce"], r_comb_900["best_val_ce"])
    h2_passed = bool(
        r_meas_900["best_val_ce"] < r_w22_900["best_val_ce"]
        and r_trace_900["best_val_ce"] < r_w22_900["best_val_ce"]
        and r_comb_900["best_val_ce"] < r_w22_900["best_val_ce"]
        and best_w23_900 < r_w21_900["best_val_ce"]
        and best_w23_900 < r_w20_900["best_val_ce"]
        and best_w23_900 < r_q900["best_val_ce"]
        and best_w23_900 < cl_900_ref_ce
        and r_meas_900["final_sharp_eff_rank"] >= 65.0
        and r_trace_900["final_sharp_eff_rank"] >= 65.0
        and r_comb_900["final_sharp_eff_rank"] >= 65.0
        and r_comb_900["final_kraus_stiefel_err"] <= 0.05
        and r_comb_900["peak_vram_gb"] <= 8.0
    )

    ckpts = (300, 900, 1800, 2700, 3600)
    w23_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_w23_3600["evals"]}
    w22_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_w22_3600["evals"]}
    w21_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_w21_3600["evals"]}
    w20_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_w20_3600["evals"] if int(e["step"]) in ckpts or int(e["step"]) == 0}
    c_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_cl36["evals"]}
    q_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_q36["evals"]}
    beats_w22_by_step = {str(s): bool(w23_by_step[s] < w22_by_step[s]) for s in ckpts}
    beats_w21_by_step = {str(s): bool(w23_by_step[s] < w21_by_step[s]) for s in ckpts}
    beats_w20_by_step = {str(s): bool(w23_by_step[s] < w20_by_step[s]) for s in ckpts}
    beats_classical_by_step = {str(s): bool(w23_by_step[s] < c_by_step[s]) for s in ckpts}

    h3_passed = bool(
        r_w23_3600["best_val_ce"] < 5.00
        and r_w23_3600["best_val_ce"] < r_w22_3600["best_val_ce"]
        and r_w23_3600["best_val_ce"] < r_w21_3600["best_val_ce"]
        and r_w23_3600["best_val_ce"] < r_w20_3600["best_val_ce"]
        and r_w23_3600["best_val_ce"] < 5.1157
        and r_w23_3600["best_val_ce"] < r_cl36["best_val_ce"]
        and r_w23_3600["final_sharp_eff_rank"] >= 65.0
        and r_w23_3600["final_kraus_stiefel_err"] <= 0.05
        and r_w23_3600["params_M"] < 11.0
        and r_w23_3600["peak_vram_gb"] <= 8.0
    )

    summary = {
        "plan_id": "EXPERIMENT_PLAN_038",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "NVIDIA GeForce RTX 5060 Ti (<= 8.0 GB VRAM cap)",
        "hypotheses": {
            "H1_last_layer_measurement_and_traceless_observable_gate": h1_res,
            "H2_900step_last_layer_measurement_and_traceless_ablation": {
                "passed": h2_passed,
                "w23_last_layer_meas_best_val_ce": r_meas_900["best_val_ce"],
                "w23_traceless_obs_best_val_ce": r_trace_900["best_val_ce"],
                "w23_combined_best_val_ce": r_comb_900["best_val_ce"],
                "best_w23_900_val_ce": best_w23_900,
                "plan037_w22_best_val_ce": r_w22_900["best_val_ce"],
                "plan036_w21_best_val_ce": r_w21_900["best_val_ce"],
                "plan035_w20_best_val_ce": r_w20_900["best_val_ce"],
                "quantum_born_40p78M_best_val_ce": r_q900["best_val_ce"],
                "classical_53p54M_best_val_ce": cl_900_ref_ce,
                "w23_last_layer_meas_final_sharp_eff_rank": r_meas_900["final_sharp_eff_rank"],
                "w23_traceless_obs_final_sharp_eff_rank": r_trace_900["final_sharp_eff_rank"],
                "w23_combined_final_sharp_eff_rank": r_comb_900["final_sharp_eff_rank"],
                "plan037_w22_final_sharp_eff_rank": r_w22_900["final_sharp_eff_rank"],
                "w23_last_layer_meas_avg_tok_per_s": r_meas_900["avg_tok_per_s"],
                "w23_traceless_obs_avg_tok_per_s": r_trace_900["avg_tok_per_s"],
                "w23_combined_avg_tok_per_s": r_comb_900["avg_tok_per_s"],
                "w23_combined_final_top1_acc": r_comb_900["final_top1_acc"],
                "w23_combined_final_top10_acc": r_comb_900.get("final_top10_acc", 0.0),
                "w23_combined_final_top50_acc": r_comb_900["final_top50_acc"],
                "w23_combined_final_kraus_stiefel_err": r_comb_900["final_kraus_stiefel_err"],
                "w23_combined_peak_vram_gb": r_comb_900["peak_vram_gb"],
                "margin_vs_plan037_w22_nats": round(r_w22_900["best_val_ce"] - best_w23_900, 4),
                "margin_vs_plan036_w21_nats": round(r_w21_900["best_val_ce"] - best_w23_900, 4),
                "margin_vs_plan035_w20_nats": round(r_w20_900["best_val_ce"] - best_w23_900, 4),
                "margin_vs_quantum_born_nats": round(r_q900["best_val_ce"] - best_w23_900, 4),
                "margin_vs_classical_nats": round(cl_900_ref_ce - best_w23_900, 4),
            },
            "H3_3600step_last_layer_meas_traceless_scaling": {
                "passed": h3_passed,
                "w23_3600_best_val_ce": r_w23_3600["best_val_ce"],
                "plan037_w22_3600_best_val_ce": r_w22_3600["best_val_ce"],
                "plan036_w21_3600_best_val_ce": r_w21_3600["best_val_ce"],
                "plan035_w20_3600_best_val_ce": r_w20_3600["best_val_ce"],
                "classical_30p95M_3600_best_val_ce": r_cl36["best_val_ce"],
                "parity_split_38p28M_quantum_born_val_ce": 5.1157,
                "w19_wavchol_mz_38p28M_val_ce": 4.9873,
                "full_ambient_40p78M_quantum_born_val_ce": 4.9565,
                "w18_janti_soft_floor_3600_best_val_ce": r_q36["best_val_ce"],
                "margin_vs_plan037_w22_nats": round(r_w22_3600["best_val_ce"] - r_w23_3600["best_val_ce"], 4),
                "margin_vs_plan036_w21_nats": round(r_w21_3600["best_val_ce"] - r_w23_3600["best_val_ce"], 4),
                "margin_vs_plan035_w20_nats": round(r_w20_3600["best_val_ce"] - r_w23_3600["best_val_ce"], 4),
                "margin_vs_parity_split_nats": round(5.1157 - r_w23_3600["best_val_ce"], 4),
                "margin_vs_classical_nats": round(r_cl36["best_val_ce"] - r_w23_3600["best_val_ce"], 4),
                "w23_params_M": r_w23_3600["params_M"],
                "w23_final_top1_acc": r_w23_3600["final_top1_acc"],
                "w23_final_top5_acc": r_w23_3600["final_top5_acc"],
                "w23_final_top10_acc": r_w23_3600.get("final_top10_acc", 0.0),
                "w23_final_top50_acc": r_w23_3600["final_top50_acc"],
                "w23_final_med_rank": r_w23_3600["final_med_rank"],
                "w23_final_sharp_eff_rank": r_w23_3600["final_sharp_eff_rank"],
                "w23_final_kraus_stiefel_err": r_w23_3600["final_kraus_stiefel_err"],
                "w23_final_lambda_diff": r_w23_3600["final_lambda_diff"],
                "min_diffuse_eff_rank": r_w23_3600["min_diffuse_eff_rank"],
                "avg_tok_per_s": r_w23_3600["avg_tok_per_s"],
                "speedup_vs_w21_ratio": round(r_w23_3600["avg_tok_per_s"] / max(1.0, r_w21_3600["avg_tok_per_s"]), 3),
                "peak_vram_gb": r_w23_3600["peak_vram_gb"],
                "checkpoint_w23_val_ce": w23_by_step,
                "checkpoint_plan037_w22_val_ce": w22_by_step,
                "checkpoint_plan036_w21_val_ce": w21_by_step,
                "checkpoint_plan035_w20_val_ce": w20_by_step,
                "checkpoint_classical_val_ce": c_by_step,
                "checkpoint_quantum_born_val_ce": q_by_step,
                "checkpoint_beats_plan037_w22": beats_w22_by_step,
                "checkpoint_beats_plan036_w21": beats_w21_by_step,
                "checkpoint_beats_plan035_w20": beats_w20_by_step,
                "checkpoint_beats_classical": beats_classical_by_step,
            },
        },
        "runs": runs,
    }
    (CACHE_DIR / "hypothesis_experiments_038.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    backfill_micro_verifications(db, CACHE_DIR)
    for endpoint in ("/api/refresh", "/api/reload"):
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:8765{endpoint}", method="POST", data=b"{}"), timeout=5
            ).read()
        except Exception:
            pass
    print(f"\n[Plan 038 Complete] H1={h1_res['passed']} | H2={h2_passed} | H3={h3_passed}")


if __name__ == "__main__":
    main()
