"""Experiment Plan 040 (`Wave 25`) Verification Runner:
1. `H1`: 900-Step (`7.37M` Tokens) `FineWeb-Edu` Scaling of `Convex-Peak Traceless Born`
   with and without `Last-Layer Measurement Operator`.
2. `H2`: 3,600-Step (`29.49M` Tokens) `FineWeb-Edu` Scaling of `Convex-Peak Traceless Born`
   breaking `w23` (`5.1479`), `w22` (`5.0688`), and `w21` (`5.0211`).
3. `H3`: Genuine `ILSVRC ImageNet-1K` (`1,281,167` Train / `50,000` Val / `100,000` Test,
   `1,000` Official WordNet Classes) 8-Arm Quantum Building-Block & Readout Ablation Study.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_radical_2pow16_optimizers,
    initialize_radical_2pow16_with_ppmi_svd,
)
from qdlblocks.models.wavelet_cholesky_vision import (
    VisionDatasetBundle,
    VisionWaveletCholeskyConfig,
    WaveletCholeskyVisionClassifier,
    build_vision_wavelet_optimizers,
    load_or_download_imagenet1k_32x32,
)

ROOT = Path(__file__).resolve().parents[1]
_EX48_SPEC = importlib.util.spec_from_file_location(
    "plan038_runner", ROOT / "examples" / "48_verify_plan038_last_layer_measurement_and_traceless.py"
)
assert _EX48_SPEC is not None and _EX48_SPEC.loader is not None
_EX48 = importlib.util.module_from_spec(_EX48_SPEC)
_EX48_SPEC.loader.exec_module(_EX48)

_EX49_SPEC = importlib.util.spec_from_file_location(
    "plan039_runner", ROOT / "examples" / "49_verify_plan039_cifar100_and_1000class_fast_loop.py"
)
assert _EX49_SPEC is not None and _EX49_SPEC.loader is not None
_EX49 = importlib.util.module_from_spec(_EX49_SPEC)
_EX49_SPEC.loader.exec_module(_EX49)

IMAGENET1K_ABLATION_ARMS: tuple[dict[str, Any], ...] = (
    {
        "arm_key": "imagenet1k_classical_softmax",
        "short_label": "Classical Conv/Patch + SwiGLU + Softmax Head",
        "readout_mode": "classical_softmax",
        "use_wavelet_cholesky_branch": False,
        "use_mach_zehnder_interferometer": False,
        "use_kraus_diffuse_bath": False,
    },
    {
        "arm_key": "imagenet1k_naive_kboson_born",
        "short_label": "Quantum Backbone + Legacy Concave k-Boson Born Readout",
        "readout_mode": "naive_kboson_born",
        "use_wavelet_cholesky_branch": True,
        "use_mach_zehnder_interferometer": True,
        "use_kraus_diffuse_bath": True,
    },
    {
        "arm_key": "imagenet1k_w23_traceless_kboson",
        "short_label": "Quantum Backbone + Wave 23 Concave Traceless k-Boson Readout",
        "readout_mode": "last_layer_meas_traceless_kboson",
        "use_wavelet_cholesky_branch": True,
        "use_mach_zehnder_interferometer": True,
        "use_kraus_diffuse_bath": True,
    },
    {
        "arm_key": "imagenet1k_ablate_no_wavelet_cholesky",
        "short_label": "Ablate Multi-Scale Wavelet-Cholesky Branch (Single-Scale Linear)",
        "readout_mode": "convex_traceless_born",
        "use_wavelet_cholesky_branch": False,
        "use_mach_zehnder_interferometer": True,
        "use_kraus_diffuse_bath": True,
    },
    {
        "arm_key": "imagenet1k_ablate_no_mach_zehnder",
        "short_label": "Ablate Mach-Zehnder Two-Beam Spatial Phase Interference",
        "readout_mode": "convex_traceless_born",
        "use_wavelet_cholesky_branch": True,
        "use_mach_zehnder_interferometer": False,
        "use_kraus_diffuse_bath": True,
    },
    {
        "arm_key": "imagenet1k_ablate_no_kraus_bath",
        "short_label": "Ablate R=8 Stiefel Kraus Diffuse Measurement Bath (lambda_diff=0)",
        "readout_mode": "convex_traceless_born",
        "use_wavelet_cholesky_branch": True,
        "use_mach_zehnder_interferometer": True,
        "use_kraus_diffuse_bath": False,
    },
    {
        "arm_key": "imagenet1k_full_convex_traceless_born",
        "short_label": "Full Quantum Architecture + Solution A (Convex-Peak Traceless Born)",
        "readout_mode": "convex_traceless_born",
        "use_wavelet_cholesky_branch": True,
        "use_mach_zehnder_interferometer": True,
        "use_kraus_diffuse_bath": True,
    },
    {
        "arm_key": "imagenet1k_full_dyadic_hierarchical_born",
        "short_label": "Full Quantum Architecture + Solution B (Dyadic Tree Conditional Born)",
        "readout_mode": "dyadic_hierarchical_born",
        "use_wavelet_cholesky_branch": True,
        "use_mach_zehnder_interferometer": True,
        "use_kraus_diffuse_bath": True,
    },
)


def run_fineweb_convex_traceless_run(
    rkey: str,
    short_label: str,
    total_steps: int,
    eval_steps: tuple[int, ...],
    use_last_layer_meas: bool,
    device: torch.device,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Train a `Radical2Pow16WaveletCholeskyGPT` arm on genuine FineWeb-Edu and return full telemetry."""
    cache_path = ROOT / ".cache" / "live_logs" / f"{rkey}_results.json"
    if cache_path.exists() and not force_rerun:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(
            f"  [{rkey}] (Cached) final_val_ce={cached['final_val_ce']:.4f} | "
            f"best_val_ce={cached['best_val_ce']:.4f} | top1={cached['final_top1_acc']:.2f}% | "
            f"tok/s={cached['avg_tok_per_s']:.0f}"
        )
        return cached

    max_train_tok = 32_000_000 if total_steps > 900 else 10_000_000
    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=max_train_tok, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt",
        map_location="cpu",
        weights_only=True,
    )
    ppmi_emb_256 = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(val_arr, batch_size=4, seq_len=512, num_batches=6, device=device)
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
        num_fine_up_banks=1,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
        use_last_layer_measurement_operator=use_last_layer_meas,
        use_convex_traceless_born=True,
        traceless_gamma=0.25,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg).to(device)
    init_diag = initialize_radical_2pow16_with_ppmi_svd(
        model, ppmi_emb_256, init_tokens, prior_tokens
    )
    base_muon_lr, base_wav_lr, base_head_lr = 0.026, 0.0065, 0.0035
    muon_opt, adam_opt = build_radical_2pow16_optimizers(
        model, muon_lr=base_muon_lr, wav_lr=base_wav_lr, head_lr=base_head_lr
    )

    b_sz, seq_len = 16, 512
    tok_per_step = b_sz * seq_len
    n_work = int(work_gpu.numel())
    t0 = time.perf_counter()
    evals: list[dict[str, Any]] = []

    for step in range(total_steps + 1):
        if step > 0:
            model.train()
            model.set_training_progress((step - 1) / float(total_steps))
            lr_mult = _EX48.compute_wsd_lr_multiplier(step - 1, total_steps)
            for g in muon_opt.param_groups:
                g["lr"] = float(g.get("base_lr", base_muon_lr)) * lr_mult
            for g in adam_opt.param_groups:
                g["lr"] = float(g.get("base_lr", base_head_lr)) * lr_mult

            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            base = ((step - 1) * tok_per_step) % max(1, n_work - tok_per_step - 2)
            c_i64 = work_gpu[base : base + tok_per_step + 1].to(torch.int64) & 0xFFFF
            x_in, y_tg = c_i64[:-1].view(b_sz, seq_len), c_i64[1:].view(b_sz, seq_len)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x_in, y_tg, return_logits=False)
            assert loss is not None
            loss.backward()
            for g in (*muon_opt.param_groups, *adam_opt.param_groups):
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            muon_opt.step()
            adam_opt.step()
            if step % 2 == 0 or step == total_steps:
                model.pyramid.retract_kraus_generators()

        if step in eval_steps:
            ev = _EX48.eval_plan038_quantum(model, val_batches)
            elapsed_s = max(1e-3, time.perf_counter() - t0)
            ev["step"] = step
            ev["tokens_M"] = round((step * tok_per_step) / 1e6, 2)
            ev["elapsed_min"] = round(elapsed_s / 60.0, 3)
            ev["tok_per_s"] = round((step * tok_per_step) / elapsed_s, 1) if step > 0 else 0.0
            evals.append(ev)
            print(
                f"  [{rkey}] step={step:4d}/{total_steps} | "
                f"val_ce={ev['val_ce']:.4f} | top1={ev['val_top1']:.2f}% | "
                f"top50={ev['val_top50']:.2f}% | R_sharp={ev['sharp_eff_rank']:.1f} | "
                f"tok/s={ev['tok_per_s']:.0f}"
            )

    wall_s = time.perf_counter() - t0
    final_ev = evals[-1]
    step_map = {int(e["step"]): e for e in evals}
    peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / (1024**3) if device.type == "cuda" else 0.0
    total_params = sum(p.numel() for p in model.parameters())
    rec = {
        "key": rkey,
        "short_label": short_label,
        "arch_family": "wavelet_cholesky_2pow16_convex_traceless",
        "use_last_layer_measurement_operator": use_last_layer_meas,
        "use_convex_traceless_born": True,
        "traceless_gamma": 0.25,
        "total_steps": total_steps,
        "target_tokens_M": round((total_steps * tok_per_step) / 1e6, 2),
        "params_M": round(total_params / 1e6, 2),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "init_diagnostics": init_diag,
        "step0_val_ce": step_map[0]["val_ce"],
        "step300_val_ce": step_map.get(300, final_ev)["val_ce"],
        "step900_val_ce": step_map.get(900, final_ev)["val_ce"],
        "step3600_val_ce": step_map.get(3600, final_ev)["val_ce"],
        "best_val_ce": min(e["val_ce"] for e in evals),
        "final_val_ce": final_ev["val_ce"],
        "final_val_ppl": final_ev["val_ppl"],
        "final_top1_acc": final_ev["val_top1"],
        "final_top5_acc": final_ev["top5_acc"],
        "final_top10_acc": final_ev["val_top10"],
        "final_top50_acc": final_ev["val_top50"],
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "final_kraus_stiefel_err": final_ev["kraus_stiefel_err"],
        "avg_tok_per_s": round((total_steps * tok_per_step) / max(wall_s, 1e-6), 1),
        "wall_s": round(wall_s, 2),
        "wall_min": round(wall_s / 60.0, 3),
        "evals": evals,
    }
    cache_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    (ROOT / ".cache" / "model_cards" / f"{rkey}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    del model, muon_opt, adam_opt, work_gpu, val_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def train_and_evaluate_imagenet1k_arm(
    dataset: VisionDatasetBundle,
    arm_spec: dict[str, Any],
    epochs: int = 14,
    batch_size: int = 1000,
    seed: int = 2040,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a single `WaveletCholeskyVisionClassifier` arm on genuine `ILSVRC ImageNet-1K`."""
    rkey = f"w25_{arm_spec['arm_key']}"
    cache_path = ROOT / ".cache" / "live_logs" / f"{rkey}_results.json"
    if cache_path.exists() and not force_rerun:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(
            f"  [ilsvrc_imagenet1k | {arm_spec['arm_key']:38s}] (Cached) "
            f"Val CE={cached['final_test_ce']:.4f} (Best={cached['best_test_ce']:.4f}) | "
            f"Top1={cached['final_test_top1_acc']:5.2f}% | Top5={cached['final_test_top5_acc']:5.2f}% | "
            f"Coarse={cached['final_test_coarse_acc']:5.2f}% | Tail>10={cached['mean_tail_gt_10_prob']:.4f}"
        )
        return cached

    torch.manual_seed(seed)
    dev = dataset.train_x.device
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(dev)

    readout_mode = str(arm_spec["readout_mode"])
    cfg = VisionWaveletCholeskyConfig(
        num_classes=dataset.num_classes,
        num_coarse_classes=dataset.num_coarse_classes,
        dyadic_dim=dataset.dyadic_dim,
        d_model=128,
        num_layers=3,
        patch_size=4,
        kraus_rank=8,
        cholesky_block_size=16,
        readout_mode=readout_mode,
        init_beta=6.8,
        init_kboson=12.0,
        init_lambda_diff=0.18,
        traceless_gamma=0.45,
        dropout=0.05,
        use_wavelet_cholesky_branch=bool(arm_spec["use_wavelet_cholesky_branch"]),
        use_mach_zehnder_interferometer=bool(arm_spec["use_mach_zehnder_interferometer"]),
        use_kraus_diffuse_bath=bool(arm_spec["use_kraus_diffuse_bath"]),
    )
    model = WaveletCholeskyVisionClassifier(cfg, dataset.class_to_coarse).to(dev)
    muon_opt, adam_opt = build_vision_wavelet_optimizers(model, muon_lr=0.022, adam_lr=0.0032)

    n_train = dataset.train_x.size(0)
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = epochs * steps_per_epoch
    t0 = time.perf_counter()

    step = 0
    best_eval: dict[str, float] | None = None
    eval_history: list[dict[str, float]] = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            step += 1
            idx = perm[start : start + batch_size]
            xb = _EX49._augment_vision_batch(dataset.train_x[idx])
            yb = dataset.train_y[idx]
            cyb = dataset.train_coarse_y[idx]

            prog = step / float(max(1, total_steps))
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * prog))
            for g in muon_opt.param_groups:
                g["lr"] = 0.022 * max(0.10, lr_mult)
            for g in adam_opt.param_groups:
                g["lr"] = 0.0032 * max(0.10, lr_mult)

            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                log_probs, diag = model.compute_log_probs_and_diagnostics(xb)
                loss = F.nll_loss(log_probs, yb)
                if "coarse_logits" in diag and cfg.use_wavelet_cholesky_branch and cfg.use_kraus_diffuse_bath:
                    loss = loss + 0.06 * F.cross_entropy(diag["coarse_logits"], cyb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon_opt.step()
            adam_opt.step()
            model.pyramid.retract_kraus_generators()

        if ep in (1, 5, 10, epochs):
            ev = _EX49.evaluate_vision_model(model, dataset, batch_size=2000)
            ev["epoch"] = ep
            ev["step"] = step
            ev["elapsed_s"] = round(time.perf_counter() - t0, 2)
            eval_history.append(ev)
            if best_eval is None or ev["test_ce"] < best_eval["test_ce"]:
                best_eval = dict(ev)

    wall_s = time.perf_counter() - t0
    assert best_eval is not None
    final_eval = eval_history[-1]
    peak_vram_gb = float(torch.cuda.max_memory_allocated(dev)) / (1024**3) if dev.type == "cuda" else 0.0
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  [ilsvrc_imagenet1k | {arm_spec['arm_key']:38s}] "
        f"Val CE={final_eval['test_ce']:.4f} (Best={best_eval['test_ce']:.4f}) | "
        f"Top1={final_eval['test_top1_acc']:5.2f}% | Top5={final_eval['test_top5_acc']:5.2f}% | "
        f"Coarse={final_eval['test_coarse_acc']:5.2f}% | Tail>10={final_eval['mean_tail_gt_10_prob']:.4f} | "
        f"Wall={wall_s:.2f}s"
    )
    res = {
        "key": rkey,
        "arm_key": arm_spec["arm_key"],
        "short_label": arm_spec["short_label"],
        "dataset": dataset.name,
        "total_imagenet_train_rows": dataset.total_train_rows,
        "total_imagenet_val_rows": dataset.total_val_rows,
        "total_imagenet_test_rows": dataset.total_test_rows,
        "active_train_samples": int(n_train),
        "active_val_samples": int(dataset.test_x.size(0)),
        "readout_mode": readout_mode,
        "use_wavelet_cholesky_branch": bool(arm_spec["use_wavelet_cholesky_branch"]),
        "use_mach_zehnder_interferometer": bool(arm_spec["use_mach_zehnder_interferometer"]),
        "use_kraus_diffuse_bath": bool(arm_spec["use_kraus_diffuse_bath"]),
        "num_classes": dataset.num_classes,
        "num_coarse_classes": dataset.num_coarse_classes,
        "dyadic_dim": dataset.dyadic_dim,
        "epochs": epochs,
        "total_steps": total_steps,
        "total_params": total_params,
        "params_M": round(total_params / 1e6, 3),
        "wall_s": round(wall_s, 2),
        "img_per_s": round((n_train * epochs) / max(wall_s, 1e-6), 1),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "final_test_ce": final_eval["test_ce"],
        "best_test_ce": best_eval["test_ce"],
        "final_test_top1_acc": final_eval["test_top1_acc"],
        "best_test_top1_acc": max(e["test_top1_acc"] for e in eval_history),
        "final_test_top5_acc": final_eval["test_top5_acc"],
        "final_test_coarse_acc": final_eval["test_coarse_acc"],
        "mean_target_prob": final_eval["mean_target_prob"],
        "mean_ranks_2_10_prob": final_eval["mean_ranks_2_10_prob"],
        "mean_tail_gt_10_prob": final_eval["mean_tail_gt_10_prob"],
        "mean_target_runnerup_margin_nats": final_eval["mean_target_runnerup_margin_nats"],
        "max_prob_sum_abs_err": final_eval["max_prob_sum_abs_err"],
        "sharp_eff_rank": final_eval["sharp_eff_rank"],
        "kraus_stiefel_err": final_eval["kraus_stiefel_err"],
        "evals": eval_history,
    }
    cache_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    (ROOT / ".cache" / "model_cards" / f"{rkey}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    del model, muon_opt, adam_opt
    return res


def run_plan040_experiments(
    epochs_imagenet: int = 14,
    imagenet_train_samples: int = 60000,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Execute H1 (900-Step FineWeb-Edu), H2 (3,600-Step FineWeb-Edu), and H3 (ILSVRC ImageNet-1K Ablations)."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== [H1] 900-Step (7.37M Tokens) FineWeb-Edu Convex-Peak Traceless Born Ablation ===")
    fw_900_conv = run_fineweb_convex_traceless_run(
        rkey="w25_8gb_convex_traceless_born_900steps",
        short_label="W25 8GB Convex-Peak Traceless Born (900 Steps / 7.37M Tok)",
        total_steps=900,
        eval_steps=(0, 300, 600, 900),
        use_last_layer_meas=False,
        device=device,
        force_rerun=force_rerun,
    )
    fw_900_llm = run_fineweb_convex_traceless_run(
        rkey="w25_8gb_last_layer_meas_convex_traceless_900steps",
        short_label="W25 8GB Last-Layer Meas + Convex-Peak Traceless Born (900 Steps / 7.37M Tok)",
        total_steps=900,
        eval_steps=(0, 300, 600, 900),
        use_last_layer_meas=True,
        device=device,
        force_rerun=force_rerun,
    )

    print("\n=== [H2] 3,600-Step (29.49M Tokens) FineWeb-Edu Convex-Peak Traceless Born Scaling Run ===")
    fw_3600_conv = run_fineweb_convex_traceless_run(
        rkey="w25_8gb_convex_traceless_born_3600steps",
        short_label="W25 8GB Convex-Peak Traceless Born (3,600 Steps / 29.49M Tok)",
        total_steps=3600,
        eval_steps=(0, 300, 900, 1800, 2700, 3600),
        use_last_layer_meas=False,
        device=device,
        force_rerun=force_rerun,
    )

    print("\n=== [H3] Genuine ILSVRC ImageNet-1K (1,281,167 Train / 50,000 Val / 100,000 Test) Ablation Study ===")
    imagenet1k = load_or_download_imagenet1k_32x32(
        ROOT / ".cache" / "datasets" / "imagenet1k_32x32",
        device=device,
        max_train_samples=imagenet_train_samples,
    )
    print(
        f"  Loaded ILSVRC ImageNet-1K: total_train={imagenet1k.total_train_rows:,}, "
        f"total_val={imagenet1k.total_val_rows:,}, total_test={imagenet1k.total_test_rows:,} | "
        f"active_train={imagenet1k.train_x.size(0):,}, active_val={imagenet1k.test_x.size(0):,}"
    )
    h3_runs: dict[str, dict[str, Any]] = {}
    for arm_spec in IMAGENET1K_ABLATION_ARMS:
        res = train_and_evaluate_imagenet1k_arm(
            imagenet1k,
            arm_spec,
            epochs=epochs_imagenet,
            batch_size=1000,
            force_rerun=force_rerun,
        )
        h3_runs[arm_spec["arm_key"]] = res

    best_900_ce = min(fw_900_conv["final_val_ce"], fw_900_llm["final_val_ce"])
    h1_summary = {
        "passed": bool(
            best_900_ce < 5.4734
            and fw_900_conv["peak_vram_gb"] <= 8.0
            and fw_900_llm["peak_vram_gb"] <= 8.0
        ),
        "w25_convex_traceless_900_val_ce": fw_900_conv["final_val_ce"],
        "w25_convex_traceless_900_top1_acc": fw_900_conv["final_top1_acc"],
        "w25_convex_traceless_900_top10_acc": fw_900_conv["final_top10_acc"],
        "w25_convex_traceless_900_top50_acc": fw_900_conv["final_top50_acc"],
        "w25_convex_traceless_900_tok_per_s": fw_900_conv["avg_tok_per_s"],
        "w25_last_layer_meas_convex_traceless_900_val_ce": fw_900_llm["final_val_ce"],
        "w25_last_layer_meas_convex_traceless_900_top1_acc": fw_900_llm["final_top1_acc"],
        "w25_last_layer_meas_convex_traceless_900_top10_acc": fw_900_llm["final_top10_acc"],
        "w25_last_layer_meas_convex_traceless_900_top50_acc": fw_900_llm["final_top50_acc"],
        "w25_last_layer_meas_convex_traceless_900_tok_per_s": fw_900_llm["avg_tok_per_s"],
        "w23_900_val_ce": 5.6818,
        "w22_900_val_ce": 5.5613,
        "w21_900_val_ce": 5.4734,
        "w20_900_val_ce": 5.5253,
        "quantum_born_93p97M_900_val_ce": 5.6092,
        "classical_53p54M_900_val_ce": 5.9152,
        "gain_over_w23_nats": round(5.6818 - best_900_ce, 4),
        "gain_over_w22_nats": round(5.5613 - best_900_ce, 4),
        "gain_over_w21_nats": round(5.4734 - best_900_ce, 4),
        "gain_over_classical_nats": round(5.9152 - best_900_ce, 4),
    }

    h2_summary = {
        "passed": bool(
            fw_3600_conv["final_val_ce"] < 5.0211
            and fw_3600_conv["peak_vram_gb"] <= 8.0
            and fw_3600_conv["final_kraus_stiefel_err"] <= 0.05
        ),
        "w25_convex_traceless_3600_val_ce": fw_3600_conv["final_val_ce"],
        "w25_convex_traceless_3600_val_ppl": fw_3600_conv["final_val_ppl"],
        "w25_convex_traceless_3600_top1_acc": fw_3600_conv["final_top1_acc"],
        "w25_convex_traceless_3600_top10_acc": fw_3600_conv["final_top10_acc"],
        "w25_convex_traceless_3600_top50_acc": fw_3600_conv["final_top50_acc"],
        "w25_convex_traceless_3600_tok_per_s": fw_3600_conv["avg_tok_per_s"],
        "w25_convex_traceless_3600_wall_min": fw_3600_conv["wall_min"],
        "w25_convex_traceless_3600_peak_vram_gb": fw_3600_conv["peak_vram_gb"],
        "w23_3600_val_ce": 5.1479,
        "w22_3600_val_ce": 5.0688,
        "w21_3600_val_ce": 5.0211,
        "w20_3600_val_ce": 5.0858,
        "parity_split_38p28M_3600_val_ce": 5.1157,
        "classical_30p95M_3600_val_ce": 5.5202,
        "gain_over_w23_nats": round(5.1479 - fw_3600_conv["final_val_ce"], 4),
        "gain_over_w22_nats": round(5.0688 - fw_3600_conv["final_val_ce"], 4),
        "gain_over_w21_nats": round(5.0211 - fw_3600_conv["final_val_ce"], 4),
        "gain_over_parity_split_nats": round(5.1157 - fw_3600_conv["final_val_ce"], 4),
        "gain_over_classical_nats": round(5.5202 - fw_3600_conv["final_val_ce"], 4),
    }

    in_cls = h3_runs["imagenet1k_classical_softmax"]
    in_naive = h3_runs["imagenet1k_naive_kboson_born"]
    in_w23 = h3_runs["imagenet1k_w23_traceless_kboson"]
    in_no_wav = h3_runs["imagenet1k_ablate_no_wavelet_cholesky"]
    in_no_mz = h3_runs["imagenet1k_ablate_no_mach_zehnder"]
    in_no_kr = h3_runs["imagenet1k_ablate_no_kraus_bath"]
    in_conv = h3_runs["imagenet1k_full_convex_traceless_born"]
    in_hier = h3_runs["imagenet1k_full_dyadic_hierarchical_born"]

    best_full_ce = min(in_conv["final_test_ce"], in_hier["final_test_ce"])
    best_full_top1 = max(in_conv["final_test_top1_acc"], in_hier["final_test_top1_acc"])

    h3_summary = {
        "passed": bool(
            imagenet1k.total_train_rows == 1281167
            and imagenet1k.total_val_rows == 50000
            and imagenet1k.total_test_rows == 100000
            and (in_naive["final_test_ce"] - in_cls["final_test_ce"] >= 0.25)
            and (in_naive["final_test_ce"] - best_full_ce >= 0.35)
            and (best_full_ce <= in_cls["final_test_ce"])
            and (best_full_top1 >= in_cls["final_test_top1_acc"])
            and (best_full_ce < in_no_wav["final_test_ce"])
            and (best_full_ce < in_no_mz["final_test_ce"])
            and (best_full_ce < in_no_kr["final_test_ce"])
        ),
        "total_imagenet_train_rows": imagenet1k.total_train_rows,
        "total_imagenet_val_rows": imagenet1k.total_val_rows,
        "total_imagenet_test_rows": imagenet1k.total_test_rows,
        "classical_softmax_val_ce": in_cls["final_test_ce"],
        "naive_kboson_born_val_ce": in_naive["final_test_ce"],
        "w23_traceless_kboson_val_ce": in_w23["final_test_ce"],
        "ablate_no_wavelet_cholesky_val_ce": in_no_wav["final_test_ce"],
        "ablate_no_mach_zehnder_val_ce": in_no_mz["final_test_ce"],
        "ablate_no_kraus_bath_val_ce": in_no_kr["final_test_ce"],
        "full_convex_traceless_born_val_ce": in_conv["final_test_ce"],
        "full_dyadic_hierarchical_born_val_ce": in_hier["final_test_ce"],
        "classical_softmax_top1_acc": in_cls["final_test_top1_acc"],
        "naive_kboson_born_top1_acc": in_naive["final_test_top1_acc"],
        "w23_traceless_kboson_top1_acc": in_w23["final_test_top1_acc"],
        "ablate_no_wavelet_cholesky_top1_acc": in_no_wav["final_test_top1_acc"],
        "ablate_no_mach_zehnder_top1_acc": in_no_mz["final_test_top1_acc"],
        "ablate_no_kraus_bath_top1_acc": in_no_kr["final_test_top1_acc"],
        "full_convex_traceless_born_top1_acc": in_conv["final_test_top1_acc"],
        "full_dyadic_hierarchical_born_top1_acc": in_hier["final_test_top1_acc"],
        "wavelet_cholesky_contribution_ce_nats": round(in_no_wav["final_test_ce"] - best_full_ce, 4),
        "mach_zehnder_contribution_ce_nats": round(in_no_mz["final_test_ce"] - best_full_ce, 4),
        "kraus_bath_contribution_ce_nats": round(in_no_kr["final_test_ce"] - best_full_ce, 4),
        "convex_traceless_gain_over_naive_ce_nats": round(in_naive["final_test_ce"] - best_full_ce, 4),
        "convex_traceless_gain_over_classical_ce_nats": round(in_cls["final_test_ce"] - best_full_ce, 4),
    }

    payload = {
        "plan_id": "EXPERIMENT_PLAN_040",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "NVIDIA GeForce RTX 5060 Ti (<= 8.0 GB VRAM cap)",
        "hypotheses": {
            "H1_fineweb_900step_convex_traceless_ablation": h1_summary,
            "H2_fineweb_3600step_convex_traceless_scaling": h2_summary,
            "H3_ilsvrc_imagenet1k_quantum_building_block_ablations": h3_summary,
        },
        "fineweb_900step_runs": {
            "w25_8gb_convex_traceless_born_900steps": fw_900_conv,
            "w25_8gb_last_layer_meas_convex_traceless_900steps": fw_900_llm,
        },
        "fineweb_3600step_run": fw_3600_conv,
        "imagenet1k_ablation_runs": h3_runs,
    }
    out_path = ROOT / ".cache" / "hypothesis_experiments_040.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved complete Plan 040 telemetry to {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Experiment Plan 040 (Wave 25).")
    parser.add_argument("--epochs-imagenet", type=int, default=14)
    parser.add_argument("--imagenet-train-samples", type=int, default=60000)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    run_plan040_experiments(
        epochs_imagenet=args.epochs_imagenet,
        imagenet_train_samples=args.imagenet_train_samples,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
