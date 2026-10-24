"""Experiment Plan 039 (`Wave 24`) Verification Runner:
Fast-Loop Standard Multi-Class Vision Benchmarks (`CIFAR-100` $K=100, N=2^7$ &
`ImageNet-1K`-Scale `1,000`-Class $K=1000, N=2^{10}$) + `FineWeb-Edu` ($V=50,257$) Transfer Probe.

Reproduces and resolves the "High Top-1 Accuracy, Bad Cross-Entropy" paradox of legacy concave
$k$-boson Born readouts in a `< 15s` per-run GPU loop under the strict `<= 8.0 GB` VRAM cap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from qdlblocks.layers.direct_chunked_ce import (
    compute_two_beam_kboson_logits,
    direct_chunked_two_beam_kboson_ce,
)
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
    build_1000class_finegrained_dataset,
    build_vision_wavelet_optimizers,
    load_or_download_cifar100,
)

ROOT = Path(__file__).resolve().parents[1]
_EX48_SPEC = importlib.util.spec_from_file_location(
    "plan038_runner", ROOT / "examples" / "48_verify_plan038_last_layer_measurement_and_traceless.py"
)
assert _EX48_SPEC is not None and _EX48_SPEC.loader is not None
_EX48 = importlib.util.module_from_spec(_EX48_SPEC)
_EX48_SPEC.loader.exec_module(_EX48)

READOUT_MODES = (
    "classical_softmax",
    "naive_kboson_born",
    "last_layer_meas_traceless_kboson",
    "convex_traceless_born",
    "dyadic_hierarchical_born",
)


def _augment_vision_batch(x: torch.Tensor) -> torch.Tensor:
    """Fast GPU tensor augmentation: random horizontal flip + 4px reflect-pad crop."""
    b_sz = x.size(0)
    flip_mask = torch.rand((b_sz, 1, 1, 1), device=x.device) < 0.5
    x_flipped = torch.where(flip_mask, torch.flip(x, dims=[-1]), x)
    x_pad = F.pad(x_flipped, (4, 4, 4, 4), mode="reflect")
    dy = int(torch.randint(0, 9, (1,), device=x.device).item())
    dx = int(torch.randint(0, 9, (1,), device=x.device).item())
    return x_pad[:, :, dy : dy + 32, dx : dx + 32]


def _compute_sharp_eff_rank(psi_fine: torch.Tensor) -> float:
    """Compute participation ratio `(Tr G)^2 / ||G||_F^2` of the `(K, d)` class wavefunctions."""
    with torch.no_grad():
        m_c = psi_fine.detach().float()
        m_c = m_c - m_c.mean(dim=0, keepdim=True)
        gram = m_c.t() @ m_c
        tr_g = float(torch.trace(gram).item())
        frob_sq = float((gram * gram).sum().item())
        return float((tr_g * tr_g) / max(frob_sq, 1e-12))


@torch.no_grad()
def evaluate_vision_model(
    model: WaveletCholeskyVisionClassifier,
    dataset: VisionDatasetBundle,
    batch_size: int = 1000,
) -> dict[str, float]:
    """Evaluate `model` on `dataset.test_x` and return CE, Top-1/5, Coarse Acc, and tail-mass decomposition."""
    model.eval()
    n_test = dataset.test_x.size(0)
    total_nll = 0.0
    total_top1 = 0
    total_top5 = 0
    total_coarse = 0
    total_target_prob = 0.0
    total_ranks_2_10_prob = 0.0
    total_tail_gt_10_prob = 0.0
    total_margin = 0.0
    max_prob_sum_err = 0.0

    c_map = dataset.class_to_coarse
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        xb = dataset.test_x[start:end]
        yb = dataset.test_y[start:end]
        cyb = dataset.test_coarse_y[start:end]
        b_len = end - start

        with torch.amp.autocast(device_type=xb.device.type, dtype=torch.bfloat16, enabled=xb.is_cuda):
            log_probs, _ = model.compute_log_probs_and_diagnostics(xb)

        log_p = log_probs.float()
        probs = log_p.exp()
        prob_sum_err = float((probs.sum(dim=-1) - 1.0).abs().max().item())
        max_prob_sum_err = max(max_prob_sum_err, prob_sum_err)

        nll_batch = F.nll_loss(log_p, yb, reduction="sum")
        total_nll += float(nll_batch.item())

        top5_vals, top5_idx = torch.topk(log_p, k=min(5, log_p.size(-1)), dim=-1)
        pred_top1 = top5_idx[:, 0]
        total_top1 += int((pred_top1 == yb).sum().item())
        total_top5 += int((top5_idx == yb.unsqueeze(1)).any(dim=1).sum().item())
        total_coarse += int((c_map[pred_top1] == cyb).sum().item())

        # Target probability & runner-up margin
        target_log_p = log_p.gather(1, yb.unsqueeze(1)).squeeze(1)
        target_p = probs.gather(1, yb.unsqueeze(1)).squeeze(1)
        masked_other = log_p.scatter(1, yb.unsqueeze(1), -1e9)
        runnerup_log_p = masked_other.max(dim=-1).values
        total_margin += float((target_log_p - runnerup_log_p).sum().item())

        # Sorted probability mass buckets (Rank 1, Ranks 2-10, Ranks > 10)
        sorted_p, _ = torch.sort(probs, dim=-1, descending=True)
        total_target_prob += float(target_p.sum().item())
        total_ranks_2_10_prob += float(sorted_p[:, 1:10].sum().item())
        total_tail_gt_10_prob += float(sorted_p[:, 10:].sum().item())

    _, psi_fine = model.pyramid.synthesize_class_wavefunctions()
    return {
        "test_ce": round(total_nll / n_test, 4),
        "test_ppl": round(math.exp(min(20.0, total_nll / n_test)), 2),
        "test_top1_acc": round(100.0 * total_top1 / n_test, 2),
        "test_top5_acc": round(100.0 * total_top5 / n_test, 2),
        "test_coarse_acc": round(100.0 * total_coarse / n_test, 2),
        "mean_target_prob": round(total_target_prob / n_test, 4),
        "mean_ranks_2_10_prob": round(total_ranks_2_10_prob / n_test, 4),
        "mean_tail_gt_10_prob": round(total_tail_gt_10_prob / n_test, 4),
        "mean_target_runnerup_margin_nats": round(total_margin / n_test, 4),
        "max_prob_sum_abs_err": float(f"{max_prob_sum_err:.2e}"),
        "sharp_eff_rank": round(_compute_sharp_eff_rank(psi_fine), 2),
        "kraus_stiefel_err": round(model.pyramid.kraus_stiefel_error(), 6),
    }


def train_and_evaluate_vision_arm(
    dataset: VisionDatasetBundle,
    readout_mode: str,
    epochs: int = 18,
    batch_size: int = 512,
    seed: int = 2039,
) -> dict[str, Any]:
    """Train a single `WaveletCholeskyVisionClassifier` readout arm in `< 15s` and return full telemetry."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    dev = dataset.train_x.device
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
        init_beta=10.5 if dataset.num_classes == 100 else 12.5,
        init_kboson=12.0,
        init_lambda_diff=0.14,
        traceless_gamma=0.75 if dataset.num_classes == 100 else 0.90,
        dropout=0.05,
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
            xb = _augment_vision_batch(dataset.train_x[idx])
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
                if readout_mode == "dyadic_hierarchical_born" and "coarse_logits" in diag:
                    loss = loss + 0.04 * F.cross_entropy(diag["coarse_logits"], cyb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon_opt.step()
            adam_opt.step()
            model.pyramid.retract_kraus_generators()

        if ep in (1, 6, 12, epochs):
            ev = evaluate_vision_model(model, dataset)
            ev["epoch"] = ep
            ev["step"] = step
            ev["elapsed_s"] = round(time.perf_counter() - t0, 2)
            eval_history.append(ev)
            if best_eval is None or ev["test_ce"] < best_eval["test_ce"]:
                best_eval = dict(ev)

    wall_s = time.perf_counter() - t0
    assert best_eval is not None
    final_eval = eval_history[-1]
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated(dev)) / (1024**3)
        if dev.type == "cuda"
        else 0.0
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  [{dataset.name} | {readout_mode:32s}] "
        f"Test CE={final_eval['test_ce']:.4f} (Best={best_eval['test_ce']:.4f}) | "
        f"Top1={final_eval['test_top1_acc']:5.2f}% | Top5={final_eval['test_top5_acc']:5.2f}% | "
        f"Coarse={final_eval['test_coarse_acc']:5.2f}% | "
        f"Tail>10={final_eval['mean_tail_gt_10_prob']:.4f} | "
        f"Margin={final_eval['mean_target_runnerup_margin_nats']:+.3f} | "
        f"Wall={wall_s:.2f}s"
    )
    return {
        "dataset": dataset.name,
        "readout_mode": readout_mode,
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


def verify_convex_traceless_born_autograd_gate(device: torch.device) -> dict[str, Any]:
    """Verify `direct_chunked_two_beam_kboson_ce(..., convex_traceless=True)` matches PyTorch autograd."""
    torch.manual_seed(2039)
    n_tok, d, v_vocab, v_act = 48, 32, 128, 100
    psi_s = F.normalize(torch.randn(n_tok, d, device=device, dtype=torch.float32), dim=-1).requires_grad_(True)
    psi_d = F.normalize(torch.randn(n_tok, d, device=device, dtype=torch.float32), dim=-1).requires_grad_(True)
    m_s = F.normalize(torch.randn(v_vocab, d, device=device, dtype=torch.float32), dim=-1).requires_grad_(True)
    m_d = F.normalize(torch.randn(v_vocab, d, device=device, dtype=torch.float32), dim=-1).requires_grad_(True)
    beta_t = (torch.rand(n_tok, 1, device=device, dtype=torch.float32) * 10.0 + 12.0).requires_grad_(True)
    k_cop = torch.tensor(28.0, device=device, dtype=torch.float32, requires_grad=True)
    lam_d = torch.tensor(0.14, device=device, dtype=torch.float32, requires_grad=True)
    prior = (torch.randn(v_vocab, device=device, dtype=torch.float32) * 0.2).requires_grad_(True)
    targets = torch.randint(0, v_act, (n_tok,), device=device, dtype=torch.long)

    loss_chunk = direct_chunked_two_beam_kboson_ce(
        psi_sharp=psi_s, psi_diff=psi_d, m_sharp=m_s, m_diff=m_d,
        beta_t=beta_t, k_copies=k_cop, lambda_diff=lam_d, targets=targets,
        log_prior=prior, chunk_size=16, valid_vocab_size=v_act,
        quadratic_diffuse_fidelity=0.5, traceless_gamma=0.75, convex_traceless=True,
    )
    grads_chunk = torch.autograd.grad(loss_chunk, (psi_s, psi_d, m_s, m_d, beta_t, k_cop, lam_d, prior))

    logits_ref = compute_two_beam_kboson_logits(
        psi_sharp=psi_s, psi_diff=psi_d, m_sharp=m_s, m_diff=m_d,
        beta_t=beta_t, k_copies=k_cop, lambda_diff=lam_d,
        log_prior=prior, quadratic_diffuse_fidelity=0.5,
        traceless_gamma=0.75, convex_traceless=True, valid_vocab_size=v_act,
    )
    loss_ref = F.cross_entropy(logits_ref[:, :v_act], targets)
    grads_ref = torch.autograd.grad(loss_ref, (psi_s, psi_d, m_s, m_d, beta_t, k_cop, lam_d, prior))

    fwd_diff = float((loss_chunk - loss_ref).abs().item())
    names = ("dpsi_sharp", "dpsi_diff", "dm_sharp", "dm_diff", "dbeta", "dk", "dlam", "dprior")
    grad_diffs = {name: float((gc - gr).abs().max().item()) for name, gc, gr in zip(names, grads_chunk, grads_ref)}
    max_g_diff = max(grad_diffs.values())
    return {
        "autograd_gate_passed": bool(fwd_diff < 1e-5 and max_g_diff < 1e-4),
        "forward_loss_abs_diff": float(f"{fwd_diff:.2e}"),
        "max_grad_abs_diff": float(f"{max_g_diff:.2e}"),
        "grad_abs_diffs": {k: float(f"{v:.2e}") for k, v in grad_diffs.items()},
    }


def run_fineweb_300step_convex_traceless_probe(
    device: torch.device,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Run a 300-step (`2.46M` tokens) FineWeb-Edu transfer probe with `use_convex_traceless_born=True`."""
    rkey = "w24_8gb_convex_traceless_born_300steps"
    cache_path = ROOT / ".cache" / "live_logs" / f"{rkey}_results.json"
    if cache_path.exists() and not force_rerun:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(
            f"  [{rkey}] (Cached) step300_val_ce={cached['step300_val_ce']:.4f} | "
            f"top1={cached['final_top1_acc']:.2f}% | tok/s={cached['avg_tok_per_s']:.0f}"
        )
        return cached

    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=8_000_000, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt",
        map_location="cpu",
        weights_only=True,
    )
    ppmi_emb_256 = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(val_arr, batch_size=4, seq_len=512, num_batches=6, device=device)
    init_tokens = torch.from_numpy(train_arr[:2_600_000].astype("int64"))
    prior_tokens = torch.from_numpy(train_arr[:2_000_000].astype("int64"))
    work_gpu = torch.from_numpy(train_arr[:8_000_000].view("int16")).to(device)

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

    total_steps = 300
    schedule_steps = 900
    eval_steps = (0, 100, 200, 300)
    b_sz, seq_len = 16, 512
    tok_per_step = b_sz * seq_len
    n_work = int(work_gpu.numel())
    t0 = time.perf_counter()
    evals: list[dict[str, Any]] = []

    for step in range(total_steps + 1):
        if step > 0:
            model.train()
            model.set_training_progress((step - 1) / float(schedule_steps))
            lr_mult = _EX48.compute_wsd_lr_multiplier(step - 1, schedule_steps)
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
                f"R_sharp={ev['sharp_eff_rank']:.1f} | tok/s={ev['tok_per_s']:.0f}"
            )

    wall_s = time.perf_counter() - t0
    final_ev = evals[-1]
    peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / (1024**3) if device.type == "cuda" else 0.0
    total_params = sum(p.numel() for p in model.parameters())
    rec = {
        "key": rkey,
        "short_label": "W24 8GB Convex-Peak Traceless Born (300 Steps / 2.46M Tok)",
        "arch_family": "wavelet_cholesky_2pow16_convex_traceless",
        "total_steps": total_steps,
        "target_tokens_M": round((total_steps * tok_per_step) / 1e6, 2),
        "params_M": round(total_params / 1e6, 2),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "init_diagnostics": init_diag,
        "step0_val_ce": evals[0]["val_ce"],
        "step300_val_ce": final_ev["val_ce"],
        "best_val_ce": min(e["val_ce"] for e in evals),
        "final_val_ce": final_ev["val_ce"],
        "final_top1_acc": final_ev["val_top1"],
        "final_top5_acc": final_ev["top5_acc"],
        "final_top10_acc": final_ev["val_top10"],
        "final_top50_acc": final_ev["val_top50"],
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "final_kraus_stiefel_err": final_ev["kraus_stiefel_err"],
        "avg_tok_per_s": round((total_steps * tok_per_step) / max(wall_s, 1e-6), 1),
        "wall_s": round(wall_s, 2),
        "evals": evals,
    }
    cache_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    (ROOT / ".cache" / "model_cards" / f"{rkey}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def run_plan039_experiments(epochs_vision: int = 18, force_rerun: bool = False) -> dict[str, Any]:
    """Execute H1 (CIFAR-100), H2 (1,000-Class), and H3 (FineWeb-Edu 300-Step Transfer) under <=8.0 GB VRAM."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== [Plan 039 / Wave 24] Loading Official CIFAR-100 & Constructing 1,000-Class Benchmark ===")
    cifar100 = load_or_download_cifar100(ROOT / ".cache" / "datasets", device=device)
    bench1000 = build_1000class_finegrained_dataset(cifar100)

    print("\n=== [H1] Running Genuine CIFAR-100 (K=100, K_coarse=20, N=128) Readout Comparison ===")
    h1_runs: dict[str, dict[str, Any]] = {}
    for mode in READOUT_MODES:
        rkey = f"w24_cifar100_{mode}"
        cpath = ROOT / ".cache" / "live_logs" / f"{rkey}_results.json"
        if cpath.exists() and not force_rerun:
            res = json.loads(cpath.read_text(encoding="utf-8"))
            print(
                f"  [cifar100_k100 | {mode:32s}] (Cached) "
                f"Test CE={res['final_test_ce']:.4f} | Top1={res['final_test_top1_acc']:5.2f}% | "
                f"Tail>10={res['mean_tail_gt_10_prob']:.4f}"
            )
        else:
            res = train_and_evaluate_vision_arm(cifar100, mode, epochs=epochs_vision)
            cpath.write_text(json.dumps(res, indent=2), encoding="utf-8")
            (ROOT / ".cache" / "model_cards" / f"{rkey}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        h1_runs[mode] = res

    print("\n=== [H2] Running ImageNet-1K-Scale 1,000-Class Visual Benchmark (K=1000, K_coarse=32, N=1024) ===")
    h2_runs: dict[str, dict[str, Any]] = {}
    for mode in READOUT_MODES:
        rkey = f"w24_1000class_{mode}"
        cpath = ROOT / ".cache" / "live_logs" / f"{rkey}_results.json"
        if cpath.exists() and not force_rerun:
            res = json.loads(cpath.read_text(encoding="utf-8"))
            print(
                f"  [imagenet1k_scale_cifar1000 | {mode:32s}] (Cached) "
                f"Test CE={res['final_test_ce']:.4f} | Top1={res['final_test_top1_acc']:5.2f}% | "
                f"Tail>10={res['mean_tail_gt_10_prob']:.4f}"
            )
        else:
            res = train_and_evaluate_vision_arm(bench1000, mode, epochs=epochs_vision)
            cpath.write_text(json.dumps(res, indent=2), encoding="utf-8")
            (ROOT / ".cache" / "model_cards" / f"{rkey}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        h2_runs[mode] = res

    print("\n=== [H3] Verifying Convex-Peak Traceless Born Autograd & 300-Step FineWeb-Edu Transfer ===")
    ag_res = verify_convex_traceless_born_autograd_gate(device)
    fw_300 = run_fineweb_300step_convex_traceless_probe(device, force_rerun=force_rerun)

    c100_cls = h1_runs["classical_softmax"]
    c100_naive = h1_runs["naive_kboson_born"]
    c100_w23 = h1_runs["last_layer_meas_traceless_kboson"]
    c100_conv = h1_runs["convex_traceless_born"]
    c100_hier = h1_runs["dyadic_hierarchical_born"]

    c1k_cls = h2_runs["classical_softmax"]
    c1k_naive = h2_runs["naive_kboson_born"]
    c1k_w23 = h2_runs["last_layer_meas_traceless_kboson"]
    c1k_conv = h2_runs["convex_traceless_born"]
    c1k_hier = h2_runs["dyadic_hierarchical_born"]

    h1_summary = {
        "passed": bool(
            (c100_naive["final_test_ce"] - c100_cls["final_test_ce"] >= 0.20)
            and (c100_naive["final_test_ce"] - c100_conv["final_test_ce"] >= 0.25)
            and (c100_conv["final_test_ce"] <= c100_cls["final_test_ce"])
            and (c100_hier["final_test_ce"] <= c100_cls["final_test_ce"])
            and (max(c100_conv["final_test_top1_acc"], c100_hier["final_test_top1_acc"]) >= c100_cls["final_test_top1_acc"])
        ),
        "classical_softmax_test_ce": c100_cls["final_test_ce"],
        "naive_kboson_born_test_ce": c100_naive["final_test_ce"],
        "last_layer_meas_traceless_kboson_test_ce": c100_w23["final_test_ce"],
        "convex_traceless_born_test_ce": c100_conv["final_test_ce"],
        "dyadic_hierarchical_born_test_ce": c100_hier["final_test_ce"],
        "classical_softmax_top1_acc": c100_cls["final_test_top1_acc"],
        "naive_kboson_born_top1_acc": c100_naive["final_test_top1_acc"],
        "last_layer_meas_traceless_kboson_top1_acc": c100_w23["final_test_top1_acc"],
        "convex_traceless_born_top1_acc": c100_conv["final_test_top1_acc"],
        "dyadic_hierarchical_born_top1_acc": c100_hier["final_test_top1_acc"],
        "naive_vs_classical_ce_penalty_nats": round(c100_naive["final_test_ce"] - c100_cls["final_test_ce"], 4),
        "convex_gain_over_naive_ce_nats": round(c100_naive["final_test_ce"] - c100_conv["final_test_ce"], 4),
        "hierarchical_gain_over_naive_ce_nats": round(c100_naive["final_test_ce"] - c100_hier["final_test_ce"], 4),
        "convex_gain_over_classical_ce_nats": round(c100_cls["final_test_ce"] - c100_conv["final_test_ce"], 4),
        "hierarchical_gain_over_classical_ce_nats": round(c100_cls["final_test_ce"] - c100_hier["final_test_ce"], 4),
        "naive_tail_gt_10_prob": c100_naive["mean_tail_gt_10_prob"],
        "convex_tail_gt_10_prob": c100_conv["mean_tail_gt_10_prob"],
        "hierarchical_tail_gt_10_prob": c100_hier["mean_tail_gt_10_prob"],
    }

    h2_summary = {
        "passed": bool(
            (c1k_naive["final_test_ce"] - c1k_cls["final_test_ce"] >= 0.40)
            and (c1k_naive["final_test_ce"] - c1k_conv["final_test_ce"] >= 0.50)
            and (c1k_conv["final_test_ce"] <= c1k_cls["final_test_ce"])
            and (c1k_hier["final_test_ce"] <= c1k_cls["final_test_ce"])
            and (max(c1k_conv["final_test_top1_acc"], c1k_hier["final_test_top1_acc"]) >= c1k_cls["final_test_top1_acc"])
        ),
        "classical_softmax_test_ce": c1k_cls["final_test_ce"],
        "naive_kboson_born_test_ce": c1k_naive["final_test_ce"],
        "last_layer_meas_traceless_kboson_test_ce": c1k_w23["final_test_ce"],
        "convex_traceless_born_test_ce": c1k_conv["final_test_ce"],
        "dyadic_hierarchical_born_test_ce": c1k_hier["final_test_ce"],
        "classical_softmax_top1_acc": c1k_cls["final_test_top1_acc"],
        "naive_kboson_born_top1_acc": c1k_naive["final_test_top1_acc"],
        "last_layer_meas_traceless_kboson_top1_acc": c1k_w23["final_test_top1_acc"],
        "convex_traceless_born_top1_acc": c1k_conv["final_test_top1_acc"],
        "dyadic_hierarchical_born_top1_acc": c1k_hier["final_test_top1_acc"],
        "naive_vs_classical_ce_penalty_nats": round(c1k_naive["final_test_ce"] - c1k_cls["final_test_ce"], 4),
        "convex_gain_over_naive_ce_nats": round(c1k_naive["final_test_ce"] - c1k_conv["final_test_ce"], 4),
        "hierarchical_gain_over_naive_ce_nats": round(c1k_naive["final_test_ce"] - c1k_hier["final_test_ce"], 4),
        "convex_gain_over_classical_ce_nats": round(c1k_cls["final_test_ce"] - c1k_conv["final_test_ce"], 4),
        "hierarchical_gain_over_classical_ce_nats": round(c1k_cls["final_test_ce"] - c1k_hier["final_test_ce"], 4),
        "naive_tail_gt_10_prob": c1k_naive["mean_tail_gt_10_prob"],
        "convex_tail_gt_10_prob": c1k_conv["mean_tail_gt_10_prob"],
        "hierarchical_tail_gt_10_prob": c1k_hier["mean_tail_gt_10_prob"],
    }

    h3_summary = {
        "passed": bool(
            ag_res["autograd_gate_passed"]
            and fw_300["step300_val_ce"] < 6.2375
            and fw_300["peak_vram_gb"] <= 8.0
        ),
        **ag_res,
        "w24_step300_val_ce": fw_300["step300_val_ce"],
        "w24_step300_top1_acc": fw_300["final_top1_acc"],
        "w24_step300_top10_acc": fw_300["final_top10_acc"],
        "w24_step300_top50_acc": fw_300["final_top50_acc"],
        "w24_avg_tok_per_s": fw_300["avg_tok_per_s"],
        "w24_peak_vram_gb": fw_300["peak_vram_gb"],
        "w23_step300_val_ce": 6.2989,
        "w22_step300_val_ce": 6.2375,
        "w21_step300_val_ce": 6.1668,
        "w20_step300_val_ce": 6.1868,
        "classical_53p61M_step300_val_ce": 6.5447,
        "gain_over_w23_nats": round(6.2989 - fw_300["step300_val_ce"], 4),
        "gain_over_w22_nats": round(6.2375 - fw_300["step300_val_ce"], 4),
        "gain_over_classical_nats": round(6.5447 - fw_300["step300_val_ce"], 4),
    }

    payload = {
        "plan_id": "EXPERIMENT_PLAN_039",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "NVIDIA GeForce RTX 5060 Ti (<= 8.0 GB VRAM cap)",
        "hypotheses": {
            "H1_cifar100_k100_paradox_and_resolution": h1_summary,
            "H2_imagenet1k_scale_k1000_paradox_and_resolution": h2_summary,
            "H3_fineweb_50257_convex_traceless_transfer": h3_summary,
        },
        "cifar100_runs": h1_runs,
        "bench1000_runs": h2_runs,
        "fineweb_300step_run": fw_300,
    }
    out_path = ROOT / ".cache" / "hypothesis_experiments_039.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved complete Plan 039 telemetry to {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Experiment Plan 039 (Wave 24).")
    parser.add_argument("--epochs-vision", type=int, default=18)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    run_plan039_experiments(epochs_vision=args.epochs_vision, force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()
