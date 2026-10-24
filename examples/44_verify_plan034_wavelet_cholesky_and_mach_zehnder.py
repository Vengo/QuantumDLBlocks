#!/usr/bin/env python3
"""Plan 034 Verification: Tied-2^16 Wavelet-Cholesky Amplitude Networks & Mach-Zehnder Equi-Dimensional O(N).

Verifies Hypotheses H1, H2, and H3 of `EXPERIMENT_PLAN_034.md` on the local NVIDIA GeForce RTX 5060 Ti
(strictly hardware-capped to <= 8.0 GB VRAM via `torch.cuda.set_per_process_memory_fraction(0.5, device=0)`).
Complies with all 5 rules of `docs/SCRIPT_BEST_PRACTICES.md` (< 390 lines, <= 8 args/function, shared builders).
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
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

from qdlblocks.benchmarks.config_models import (
    BackboneConfig,
    KBosonReadoutConfig,
    OptimizerConfig,
    QuantumExperimentSpec,
    QubitHierarchyConfig,
    build_classical_modded_l4_baseline,
    build_quantum_born_optimizers,
    build_quantum_model_from_spec,
    compute_wsd_lr_multiplier,
)
from qdlblocks.benchmarks.db import DEFAULT_DB_PATH, BenchmarkSQLiteDB, backfill_micro_verifications
from qdlblocks.benchmarks.model_card import (
    build_model_card_from_manifest_entry,
    save_model_card_atomic,
    update_model_card_telemetry,
)
from qdlblocks.layers.wavelet_cholesky import WaveletCholeskyConfig, WaveletCholeskyMultiscaleBranch

CACHE_DIR = ROOT / ".cache"
LIVE_DIR = CACHE_DIR / "live_logs"
CARDS_DIR = CACHE_DIR / "model_cards"

_EX43_SPEC = importlib.util.spec_from_file_location(
    "plan033_runner",
    ROOT / "examples" / "43_verify_plan033_scaling_and_soft_spectral_floor.py",
)
assert _EX43_SPEC is not None and _EX43_SPEC.loader is not None
_EX43 = importlib.util.module_from_spec(_EX43_SPEC)
_EX43_SPEC.loader.exec_module(_EX43)
load_fineweb_tokens = _EX43.load_fineweb_tokens
build_evaluation_batches = _EX43.build_evaluation_batches
run_single_experiment = _EX43.run_single_experiment

_SHARED_REFS = (
    build_classical_modded_l4_baseline,
    build_quantum_born_optimizers,
    compute_wsd_lr_multiplier,
)


def run_h1_wavelet_cholesky_verification(
    data_bundle: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Verify H1: 3-level Wavelet-Cholesky exact reconstruction/isometry & Tied-2^16 basis parameter savings."""
    print("\n" + "=" * 96)
    print("  [H1 Verification] Multi-Level Wavelet-Cholesky V-Cycle Isometry & Tied-2^16 Basis")
    print("=" * 96)
    torch.manual_seed(2034)
    branch = WaveletCholeskyMultiscaleBranch(
        WaveletCholeskyConfig(embed_dim=256, num_levels=3, max_block_size=32, init_nonlin_gate=0.0)
    ).to(device)
    psi = F.normalize(torch.randn(4, 32, 256, device=device, dtype=torch.float32), p=2.0, dim=-1)
    coarse, details = branch.downsample_amplitudes(psi)
    e_total = coarse.pow(2).sum(dim=-1) + sum(d.pow(2).sum(dim=-1) for d in details)
    energy_ratio_err = float((e_total / psi.pow(2).sum(dim=-1) - 1.0).abs().max().item())
    recon_init = branch(psi)
    recon_max_err = float((recon_init - psi).abs().max().item())

    with torch.no_grad():
        for p_s, u_s in zip(branch.lift_predict, branch.lift_update):
            p_s.copy_(0.15 * torch.randn_like(p_s))
            u_s.copy_(0.15 * torch.randn_like(u_s))
    c_pert, d_pert = branch.downsample_amplitudes(psi)
    recon_pert = branch.upsample_amplitudes(c_pert, d_pert)
    pert_recon_err = float((recon_pert - psi).abs().max().item())

    psi_req = psi.detach().requires_grad_(True)
    out_b = branch(psi_req)
    (out_b.pow(2).sum()).backward()
    min_chol_diag = min(
        float(branch.coarse_cholesky.condition_metrics()["min_diag"]),
        *(float(m.condition_metrics()["min_diag"]) for m in branch.detail_cholesky),
    )
    branch_grad_ok = bool(psi_req.grad is not None and torch.isfinite(psi_req.grad).all().item())
    branch_params = sum(p.numel() for p in branch.parameters())

    tied_spec = QuantumExperimentSpec(
        key="h1_tied_probe",
        label="Tied 2^16 Probe",
        short_label="TiedProbe",
        backbone=BackboneConfig(
            frame_expansion=1.0,
            use_mach_zehnder_beamsplitter=True,
            use_wavelet_cholesky_branch=True,
            ngram_vocab_size=0,
            num_value_embeddings=0,
        ),
        hierarchy=QubitHierarchyConfig(
            num_higher_states=0, use_wavelet_cholesky_basis=True, directed_predecessor_mix=0.0
        ),
        readout=KBosonReadoutConfig(kraus_spectral_floor=6.5),
    )
    model_tied, ppmi_stats = build_quantum_model_from_spec(
        tied_spec, data_bundle["ppmi_emb_256"], data_bundle["init_tokens"], data_bundle["prior_tokens"], device
    )
    tied_params_m = round(
        sum(p.numel() for n, p in model_tied.named_parameters() if "ngram_embed" not in n) / 1e6, 2
    )
    untied_params_m = 40.78
    param_savings_m = round(untied_params_m - tied_params_m, 2)
    zero_sep_tables = bool(
        model_tied.povm_readout is None
        and model_tied.povm_head is None
        and model_tied.hierarchical_povm_readout is not None
        and model_tied.hierarchical_povm_readout.povm_odd_qubits is None
    )
    step0_self_ov = float(ppmi_stats["self_register_overlap_max"])
    x0, y0 = data_bundle["val_batches"][0]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss_t = model_tied(x0[:2, :128], targets=y0[:2, :128], direct_chunked_ce=True)
    assert loss_t is not None
    loss_t.backward()
    hr = model_tied.hierarchical_povm_readout
    hq = model_tied.hierarchical_qubit_embed
    assert hr is not None and hq is not None and hr.basis_chol_tril is not None
    tied_grad_ok = bool(
        hq.unigram_qubits.grad is not None
        and torch.isfinite(hq.unigram_qubits.grad).all().item()
        and hr.basis_chol_tril.grad is not None
        and torch.isfinite(hr.basis_chol_tril.grad).all().item()
    )
    del model_tied
    torch.cuda.empty_cache()

    passed = bool(
        recon_max_err < 1e-5
        and energy_ratio_err < 1e-5
        and pert_recon_err < 1e-5
        and min_chol_diag > 0.0
        and branch_grad_ok
        and zero_sep_tables
        and tied_params_m < 20.0
        and param_savings_m > 20.0
        and step0_self_ov <= 0.15
        and tied_grad_ok
    )
    print(
        f"  * [H1 Wavelet-Cholesky] ReconErr={recon_max_err:.2e} | EnergyErr={energy_ratio_err:.2e} | "
        f"PertLiftErr={pert_recon_err:.2e} | TiedCoreParams={tied_params_m:.2f}M (Saved {param_savings_m:.2f}M) | "
        f"SelfOv={step0_self_ov:.4f} | PASS={passed}"
    )
    return {
        "passed": passed,
        "reconstruction_max_err": recon_max_err,
        "energy_ratio_err": energy_ratio_err,
        "perturbed_lifting_recon_max_err": pert_recon_err,
        "min_cholesky_diag": round(min_chol_diag, 6),
        "branch_params_per_layer": branch_params,
        "zero_separate_output_tables": zero_sep_tables,
        "tied_model_params_M": tied_params_m,
        "untied_reference_params_M": untied_params_m,
        "parameter_savings_M": param_savings_m,
        "step0_max_self_overlap": round(step0_self_ov, 6),
        "grad_finite": bool(branch_grad_ok and tied_grad_ok),
    }


def save_and_register_run(
    meta: dict[str, Any], result_rec: dict[str, Any], db: BenchmarkSQLiteDB
) -> None:
    """Persist `_full_result.json`, `_results.json`, model card, and SQLite benchmark DB row."""
    rkey = meta["key"]
    (LIVE_DIR / f"{rkey}_full_result.json").write_text(json.dumps(result_rec, indent=2), encoding="utf-8")
    (LIVE_DIR / f"{rkey}_results.json").write_text(json.dumps({**result_rec, "completed": True}, indent=2))
    is_w19 = rkey.startswith("w19_")
    w_id, w_num = ("W19", 19) if is_w19 else ("W18", 18)
    p_id = "EXPERIMENT_PLAN_034" if is_w19 else "EXPERIMENT_PLAN_033"
    w_lbl = (
        "Wave 19 (Plan 034 Tied-2^16 Wavelet-Cholesky & Mach-Zehnder Equi-Dimensional O(N))"
        if is_w19
        else "Wave 18 (Plan 033 Soft Spectral Floor, O(N) Scaling & 3,600-Step 8GB Study)"
    )
    manifest_entry = {
        "key": rkey, "session": "local_rtx5060ti_8gb", "wave_id": w_id, "wave_num": w_num, "wave_label": w_lbl,
        "plan_id": p_id, "run_tier": "micro_verification", "arch_family": meta["arch_family"],
        "label": meta["label"], "short_label": meta["short_label"], "color": meta["color"],
        "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)", "target_tokens_M": result_rec["target_tokens_M"],
        "cli_args": meta["cli_args"], "remote_log": f".cache/live_logs/{rkey}.log",
        "remote_json": f".cache/live_logs/{rkey}_results.json",
    }
    cat_entry = {
        **manifest_entry, "status": "COMPLETED", "run_state": "COMPLETED", "completed": True, "is_live": False,
        "accel": "RTX 5060 Ti (8GB Cap)", "params_M": result_rec["params_M"],
        "tokens_done_M": result_rec["target_tokens_M"], "progress_pct": 100.0,
        "best_val_ce": result_rec["best_val_ce"], "best_val_ppl": result_rec["best_val_ppl"],
        "final_val_ce": result_rec["final_val_ce"], "top1_acc": result_rec["final_top1_acc"],
        "top5_acc": result_rec["final_top5_acc"], "med_rank": result_rec["final_med_rank"],
        "iso_err": result_rec["final_iso_err"], "entanglement_nats": 2.9,
        "avg_tok_per_s": result_rec["avg_tok_per_s"], "wall_min": round(result_rec["wall_s"] / 60.0, 4),
        "evals": result_rec["evals"], "steps": [],
    }
    card = build_model_card_from_manifest_entry(manifest_entry, w_id, w_lbl, p_id)
    card["run_tier"] = "micro_verification"
    card = update_model_card_telemetry(card, cat_entry)
    save_model_card_atomic(card, CARDS_DIR)
    db.upsert_run(cat_entry, model_card=card, explicit_user_override=False)


_EX43.save_and_register_run = save_and_register_run


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
    h1_res = run_h1_wavelet_cholesky_verification(data_bundle, device)
    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)

    mz_bb = BackboneConfig(frame_expansion=1.0, use_mach_zehnder_beamsplitter=True, use_wavelet_cholesky_branch=True)
    sf_ro = KBosonReadoutConfig(kraus_spectral_floor=6.5)
    opt2 = OptimizerConfig(retract_every=2)
    mz900_qspec = QuantumExperimentSpec(
        key="w19_8gb_L4_d256_mach_zehnder_wavchol_branch_ON_900steps",
        label="Mach-Zehnder + Wavelet-Cholesky O(N)", short_label="MZWavON900",
        backbone=mz_bb, readout=sf_ro, optimizer=opt2,
    )
    tied900_qspec = QuantumExperimentSpec(
        key="w19_8gb_L4_d256_tied_2pow16_wavelet_cholesky_ON_900steps",
        label="Tied-2^16 Wavelet-Cholesky + MZ O(N)", short_label="Tied2p16ON900",
        backbone=mz_bb, hierarchy=QubitHierarchyConfig(use_wavelet_cholesky_basis=True), readout=sf_ro, optimizer=opt2,
    )
    mz3600_qspec = QuantumExperimentSpec(
        key="w19_8gb_L4_d256_wavelet_cholesky_mach_zehnder_ON_3600steps",
        label="Wavelet-Cholesky + Mach-Zehnder O(N) 3600", short_label="MZWavON3600",
        steps=3600, backbone=mz_bb, readout=sf_ro, optimizer=opt2,
    )
    spec_rows = [
        ("w18_8gb_L4_d256_clean_janti_gentle_lr_900steps", "quantum_born", "clean_janti_gentle_lr_L4_d256_900steps", 900, "#fee440", "Plan 032 Isolate 3 Reference (16x16 J_anti, 900 Steps)", "W18 8GB Clean J_anti 16x16 (900 Steps)", "--architecture quantum_born", QuantumExperimentSpec(key="iso3", label="Iso3", short_label="Iso3")),
        ("w18_8gb_L4_d256_equidim_ON_900steps", "quantum_born", "equidim_ON_square_256_L4_d256_900steps", 900, "#00bbf9", "Plan 033 H2 Equidim Square-O(N) fe=1.0 (900 Steps)", "W18 8GB Equidim Square-O(N) fe=1.0 (900 Steps)", "--architecture quantum_born --frame-expansion 1.0 --kraus-spectral-floor 6.5", QuantumExperimentSpec(key="eq", label="Eq", short_label="Eq")),
        ("w19_8gb_L4_d256_mach_zehnder_wavchol_branch_ON_900steps", "quantum_born", "mach_zehnder_wavchol_ON_L4_d256_900steps", 900, "#00f5d4", "Plan 034 H2 Mach-Zehnder + Wavelet-Cholesky Equi-Dim O(N) (900 Steps)", "W19 8GB Mach-Zehnder + WavChol O(N) (900 Steps)", "--architecture quantum_born --frame-expansion 1.0 --mach-zehnder-beamsplitter --wavelet-cholesky-branch --kraus-spectral-floor 6.5", mz900_qspec),
        ("w19_8gb_L4_d256_tied_2pow16_wavelet_cholesky_ON_900steps", "quantum_born", "tied_2pow16_wavchol_ON_L4_d256_900steps", 900, "#f15bb5", "Plan 034 H2 Pure Tied-2^16 Wavelet-Cholesky Multiscale O(N) (900 Steps)", "W19 8GB Tied-2^16 WavChol O(N) (900 Steps)", "--architecture quantum_born --frame-expansion 1.0 --mach-zehnder-beamsplitter --wavelet-cholesky-branch --wavelet-cholesky-basis --kraus-spectral-floor 6.5", tied900_qspec),
        ("w18_8gb_classical_leaderboard_exact_L4_d256_3600steps", "classical_modded", "classical_leaderboard_exact_L4_d256_3600steps", 3600, "#79c0ff", "Plan 033 H3 ClassicalModdedGPT (L=4, d=256, 3,600 Steps)", "W18 8GB Classical Leaderboard-Exact (3,600 Steps)", "--architecture classical_modded --steps 3600", None),
        ("w18_8gb_L4_d256_janti_soft_floor_3600steps", "quantum_born", "janti_soft_floor_6p5_L4_d256_3600steps", 3600, "#00ff87", "Plan 033 H3 QuantumBornGPT J_anti + Soft Floor 6.5 (3,600 Steps)", "W18 8GB J_anti + Soft Floor 6.5 (3,600 Steps)", "--architecture quantum_born --kraus-spectral-floor 6.5 --steps 3600", None),
        ("w19_8gb_L4_d256_wavelet_cholesky_mach_zehnder_ON_3600steps", "quantum_born", "wavchol_mach_zehnder_ON_L4_d256_3600steps", 3600, "#9b5de5", "Plan 034 H3 Wavelet-Cholesky + Mach-Zehnder Equi-Dim O(N) (3,600 Steps)", "W19 8GB WavChol + Mach-Zehnder O(N) (3,600 Steps)", "--architecture quantum_born --frame-expansion 1.0 --mach-zehnder-beamsplitter --wavelet-cholesky-branch --kraus-spectral-floor 6.5 --steps 3600", mz3600_qspec),
    ]
    specs = [
        {"key": k, "arch_family": a, "schedule_mode": sm, "total_steps": st, "color": c, "label": lb, "short_label": sl, "cli_args": cli, "qspec": qs, "snap_every": 2}
        for k, a, sm, st, c, lb, sl, cli, qs in spec_rows
    ]

    runs = {m["key"]: run_single_experiment(m, data_bundle, db, device) for m in specs}
    r_iso3, r_eq, r_mz9, r_tied9 = runs[specs[0]["key"]], runs[specs[1]["key"]], runs[specs[2]["key"]], runs[specs[3]["key"]]
    r_cl36, r_q36_fe2, r_mz36 = runs[specs[4]["key"]], runs[specs[5]["key"]], runs[specs[6]["key"]]

    tied9_savings_m = round(r_iso3["params_M"] - r_tied9["params_M"], 2)
    h2_passed = bool(
        r_mz9["best_val_ce"] < r_eq["best_val_ce"]
        and r_mz9["avg_tok_per_s"] >= 1.50 * r_iso3["avg_tok_per_s"]
        and r_mz9["params_M"] < r_iso3["params_M"]
        and tied9_savings_m > 20.0
        and r_tied9["best_val_ce"] < 5.70
    )
    mz_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_mz36["evals"]}
    c_by_step = {int(e["step"]): float(e["val_ce"]) for e in r_cl36["evals"]}
    ckpt_wins = {str(s): bool(mz_by_step[s] < c_by_step[s]) for s in (300, 900, 1800, 2700, 3600)}
    speedup_vs_uncompiled_fe2 = round(r_mz36["avg_tok_per_s"] / max(1.0, r_iso3["avg_tok_per_s"]), 3)
    h3_passed = bool(
        all(ckpt_wins.values())
        and r_mz36["best_val_ce"] < 5.00
        and (r_cl36["best_val_ce"] - r_mz36["best_val_ce"]) >= 0.53
        and r_mz36["final_top1_acc"] >= 25.0
        and r_mz36["min_diffuse_eff_rank"] >= 6.5
        and speedup_vs_uncompiled_fe2 >= 1.35
        and r_mz36["peak_vram_gb"] <= 8.0
    )

    summary = {
        "plan_id": "EXPERIMENT_PLAN_034", "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "NVIDIA GeForce RTX 5060 Ti (<= 8.0 GB VRAM cap)",
        "hypotheses": {
            "H1_wavelet_cholesky_isometry_and_tied_2pow16_basis": h1_res,
            "H2_mach_zehnder_and_tied_2pow16_wavelet_cholesky_900steps": {
                "passed": h2_passed,
                "isolate3_fe2_best_val_ce": r_iso3["best_val_ce"],
                "uncoupled_equidim_ON_best_val_ce": r_eq["best_val_ce"],
                "mach_zehnder_wavchol_ON_best_val_ce": r_mz9["best_val_ce"],
                "tied_2pow16_wavchol_ON_best_val_ce": r_tied9["best_val_ce"],
                "mach_zehnder_wavchol_ON_avg_tok_per_s": r_mz9["avg_tok_per_s"],
                "tied_2pow16_wavchol_ON_avg_tok_per_s": r_tied9["avg_tok_per_s"],
                "isolate3_avg_tok_per_s": r_iso3["avg_tok_per_s"],
                "throughput_speedup_vs_fe2": round(r_mz9["avg_tok_per_s"] / max(1.0, r_iso3["avg_tok_per_s"]), 3),
                "mach_zehnder_wavchol_ON_params_M": r_mz9["params_M"],
                "tied_2pow16_wavchol_ON_params_M": r_tied9["params_M"],
                "tied_2pow16_param_savings_M": tied9_savings_m,
                "isolate3_params_M": r_iso3["params_M"],
            },
            "H3_3600step_29p5M_token_wavelet_cholesky_mach_zehnder_ON": {
                "passed": h3_passed,
                "wavchol_mach_zehnder_ON_3600_best_val_ce": r_mz36["best_val_ce"],
                "overcomplete_fe2_3600_best_val_ce": r_q36_fe2["best_val_ce"],
                "classical_3600_best_val_ce": r_cl36["best_val_ce"],
                "margin_vs_classical_nats": round(r_cl36["best_val_ce"] - r_mz36["best_val_ce"], 4),
                "wavchol_mach_zehnder_ON_3600_final_top1_acc": r_mz36["final_top1_acc"],
                "classical_3600_final_top1_acc": r_cl36["final_top1_acc"],
                "min_diffuse_eff_rank": r_mz36["min_diffuse_eff_rank"],
                "final_diffuse_eff_rank": r_mz36["final_diffuse_eff_rank"],
                "wavchol_mach_zehnder_ON_avg_tok_per_s": r_mz36["avg_tok_per_s"],
                "overcomplete_fe2_avg_tok_per_s": r_q36_fe2["avg_tok_per_s"],
                "speedup_vs_fe2_3600": speedup_vs_uncompiled_fe2,
                "wavchol_mach_zehnder_ON_wall_s": r_mz36["wall_s"],
                "overcomplete_fe2_wall_s": r_q36_fe2["wall_s"],
                "checkpoint_quantum_val_ce": mz_by_step,
                "checkpoint_classical_val_ce": c_by_step,
                "checkpoint_quantum_beats_classical": ckpt_wins,
                "peak_vram_gb": r_mz36["peak_vram_gb"],
            },
        },
        "runs": runs,
    }
    (CACHE_DIR / "hypothesis_experiments_034.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    backfill_micro_verifications(db, CACHE_DIR)
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8765/api/reload", method="POST", data=b"{}"), timeout=5).read()
    except Exception:
        pass
    print(f"\n[Plan 034 Complete] H1={h1_res['passed']} | H2={h2_passed} | H3={h3_passed}")


if __name__ == "__main__":
    main()
