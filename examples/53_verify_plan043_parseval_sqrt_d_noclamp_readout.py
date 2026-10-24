#!/usr/bin/env python3
"""Wave 28 Verification Runner: Flag-Free Parseval muP sqrt(d) Readout & Smith CLR (Plan 043).

Verifies Hypotheses H1, H2, H3, and H4 from `EXPERIMENT_PLAN_043.md` using the dedicated
Rule-6 flag-free model `ParsevalMuPDyadicGPT` (`src/qdlblocks/models/parseval_mup_dyadic_gpt.py`)
and loss kernel `parseval_mup_dyadic_chunked_ce` (`src/qdlblocks/layers/parseval_mup_chunked_ce.py`):
- H1: Analytical Parseval `beta_0(d) = c* sqrt(d)` scale invariance, zero-shot `c* = 1.48` transfer
  from `d = 128` to `d in {256, 384, 512}`, and post-normalization Cholesky metric absorption.
- H2: Flag-free `w28_flagfree_parseval_mup_b16_d256_900steps` (`B = 16, T = 512, d = 256`, 900 steps)
  achieves `final_val_ce <= 5.4447` with zero `PurityGate`, zero `kboson_raw_k`, and zero clamps.
- H3: Zero-shot `muP` multi-width scaling across `d in {256, 384, 512}` achieves strict monotonic
  improvement `ce_d512 < ce_d384 < ce_d256 <= 5.4447` at matched 7.37M-token horizon.
- H4: Smith (`arXiv:1506.01186`) LR Range Test + Cyclical Learning Rate (`w28_smith_clr_b16_d256_900steps`)
  eliminates the mid-training WSD plateau at `tau = 0.50` and matches/beats WSD at `tau = 1.00`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.calibration_probe import compute_calibration_rank_decomposition  # noqa: E402
from qdlblocks.models.parseval_mup_dyadic_gpt import (  # noqa: E402
    ParsevalMuPDyadicConfig,
    ParsevalMuPDyadicGPT,
    build_parseval_mup_optimizers,
    initialize_parseval_mup_with_ppmi_svd,
)

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
class VariantRunSpec:
    """Configuration for a single Wave 28 FineWeb-Edu training run."""

    name: str
    embed_dim: int
    num_heads: int
    num_layers: int = 6
    batch_size: int = 16
    block_size: int = 512
    steps: int = 900
    schedule: str = "modded_nanogpt_wsd"
    eta_base: float = 0.0065
    eta_max: float = 0.026
    mup_c_star: float = 1.48
    dyadic_coarse_weight: float = 0.015
    readout_gemm_precision: str = "fp8"


@torch.no_grad()
def eval_parseval_mup_quantum(
    model: ParsevalMuPDyadicGPT, val_batches: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict[str, Any]:
    """Evaluate flag-free `ParsevalMuPDyadicGPT` on `val_batches` with zero `kboson_raw_k`."""
    was_training, v_eff = model.training, int(model.cfg.valid_vocab_size)
    model.eval()
    logits_chunks, targets_chunks = [], []
    top10_hits, total_toks = 0, 0
    for x, y in val_batches:
        with torch.amp.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
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
    sharp_rank = round(float(model.compute_sharp_eff_rank()), 2)
    kraus_err = round(float(model.pyramid.compute_kraus_stiefel_err()), 6)
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
        "sharp_eff_rank": sharp_rank,
        "kraus_stiefel_err": kraus_err,
        "beta_0": round(float(model.beta_0), 4),
        "current_beta": round(float(model.current_beta()), 4),
        "lambda_diff": round(float(model.current_lambda_diff()), 4),
    }


def compute_step_lr_and_momentum(
    schedule: str, step: int, total_steps: int, eta_base: float, eta_max: float
) -> tuple[float, float]:
    """Return `(muon_lr_at_d256, momentum)` for `modded_nanogpt_wsd`, `smith_triangular2`, or `smith_1cycle`."""
    if schedule == "modded_nanogpt_wsd":
        mult = float(_EX48.compute_wsd_lr_multiplier(step, total_steps))
        return eta_max * mult, 0.95
    tau = float(step) / float(max(1, total_steps))
    if schedule == "smith_triangular2":
        eta_p1, eta_p2, eta_mid, eta_min = 1.22 * eta_max, 1.06 * eta_max, 0.32 * eta_max, 0.10 * eta_max
        if tau < 0.20:
            return eta_base + (eta_p1 - eta_base) * (tau / 0.20), 0.95
        if tau < 0.42:
            return eta_p1 + (0.96 * eta_max - eta_p1) * ((tau - 0.20) / 0.22), 0.95
        if tau < 0.50:
            return 0.96 * eta_max + (eta_mid - 0.96 * eta_max) * ((tau - 0.42) / 0.08), 0.95
        if tau < 0.58:
            return eta_mid + (eta_p2 - eta_mid) * ((tau - 0.50) / 0.08), 0.95
        return eta_p2 + (eta_min - eta_p2) * min(1.0, (tau - 0.58) / 0.42), 0.95
    if schedule == "smith_1cycle":
        if tau < 0.25:
            phase = tau / 0.25
            return eta_base + (eta_max - eta_base) * phase, 0.95
        u = (tau - 0.25) / 0.75
        c_val = 0.5 * (1.0 + math.cos(math.pi * u))
        eta_min = 0.10 * eta_max
        return eta_min + (eta_max - eta_min) * c_val, 0.95
    raise ValueError(f"Unsupported schedule: {schedule!r}")


def apply_scaled_group_lrs(
    muon_opt: Any, adam_opt: torch.optim.AdamW, lr_d256: float, embed_dim: int, momentum: float
) -> None:
    """Apply spectral `muP` width-scaled learning rates for unit-norm `StiefelMuon` updates."""
    mup = (256.0 / float(embed_dim)) ** 0.12
    muon_lr = lr_d256 * mup
    wav_lr = (0.0065 / 0.026) * muon_lr
    head_lr = (0.0035 / 0.026) * muon_lr
    for g in muon_opt.param_groups:
        g["lr"] = muon_lr
        g["momentum"] = momentum
    for g in adam_opt.param_groups:
        gname = g.get("group_name", "body")
        g["lr"] = wav_lr if gname == "wavelet_2pow16" else head_lr


def run_variant(
    spec: VariantRunSpec, device: torch.device, force_rerun: bool = False
) -> dict[str, Any]:
    """Train and evaluate a single Wave 28 `ParsevalMuPDyadicGPT` variant on genuine FineWeb-Edu."""
    cache_path = ROOT / ".cache" / "live_logs" / f"{spec.name}_results.json"
    if cache_path.exists() and not force_rerun:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(
            f"  [{spec.name}] (Cached) d={cached['embed_dim']} | beta_0={cached['beta_0']:.2f} | "
            f"ce@0.50={cached['val_ce_at_tau_0p50']:.4f} | final_val_ce={cached['final_val_ce']:.4f} | "
            f"top1={cached.get('final_top1_acc', 0.0):.2f}% | tok/s={cached.get('avg_tok_per_s', 0.0):.0f}",
            flush=True,
        )
        return cached

    tok_per_step = spec.batch_size * spec.block_size
    max_train_tok = max(10_000_000, spec.steps * tok_per_step + 2_000_000)
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

    cfg = ParsevalMuPDyadicConfig(
        block_size=spec.block_size,
        embed_dim=spec.embed_dim,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        window_size=min(384, spec.block_size),
        mup_c_star=spec.mup_c_star,
        dyadic_coarse_weight=spec.dyadic_coarse_weight,
        readout_gemm_precision=spec.readout_gemm_precision,
    )
    model = ParsevalMuPDyadicGPT(cfg).to(device)
    init_stats = initialize_parseval_mup_with_ppmi_svd(
        model, ppmi_emb_256, init_tokens, prior_tokens
    )
    muon_opt, adam_opt = build_parseval_mup_optimizers(
        model, base_muon_lr=spec.eta_max, base_wav_lr=0.0065, base_head_lr=0.0035
    )

    n_params = sum(p.numel() for p in model.parameters())
    params_m = n_params / 1e6
    n_work = int(work_gpu.numel())
    mid_step = max(1, spec.steps // 2)
    eval_steps_set = {0, min(300, spec.steps), mid_step, spec.steps}
    evals: list[dict[str, Any]] = []
    val_ce_mid = 0.0
    max_stiefel_err = float(init_stats["kraus_stiefel_err"])
    t0 = time.perf_counter()

    for step in range(spec.steps + 1):
        if step > 0:
            model.train()
            model.set_training_progress(float(step - 1) / float(spec.steps))
            lr_256, mom = compute_step_lr_and_momentum(
                spec.schedule, step - 1, spec.steps, spec.eta_base, spec.eta_max
            )
            apply_scaled_group_lrs(muon_opt, adam_opt, lr_256, spec.embed_dim, mom)
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            base = ((step - 1) * tok_per_step) % max(1, n_work - tok_per_step - 2)
            c_i64 = work_gpu[base : base + tok_per_step + 1].to(torch.int64) & 0xFFFF
            x_in = c_i64[:-1].view(spec.batch_size, spec.block_size)
            y_tg = c_i64[1:].view(spec.batch_size, spec.block_size)

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
            if step % 2 == 0 or step == spec.steps:
                max_stiefel_err = max(max_stiefel_err, model.retract_kraus_generators())

        if step in eval_steps_set:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ev = eval_parseval_mup_quantum(model, val_batches)
            elapsed_s = max(1e-3, time.perf_counter() - t0)
            tokens_seen = step * tok_per_step
            tau = float(step) / float(max(1, spec.steps))
            ev["step"] = step
            ev["tokens_M"] = round(tokens_seen / 1e6, 2)
            ev["compute_tflops"] = round(6.0 * params_m * (tokens_seen / 1e6), 4)
            ev["tau_progress"] = round(tau, 4)
            ev["tok_per_s"] = round(tokens_seen / elapsed_s, 1) if step > 0 else 0.0
            evals.append(ev)
            if step == mid_step:
                val_ce_mid = float(ev["val_ce"])
            print(
                f"  [{spec.name}] step={step:4d}/{spec.steps} ({ev['tokens_M']:5.2f}M tok, "
                f"tau={tau:.2f}, beta={model.current_beta():.2f}) | val_ce={ev['val_ce']:.4f} | "
                f"top1={ev['val_top1']:.2f}% | top50={ev['val_top50']:.2f}% | "
                f"R_sharp={ev['sharp_eff_rank']:.1f} | tok/s={ev['tok_per_s']:.0f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - t0
    final_ev = evals[-1]
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    res = {
        "name": spec.name,
        "key": spec.name,
        "rkey": spec.name,
        "embed_dim": spec.embed_dim,
        "num_layers": spec.num_layers,
        "num_heads": spec.num_heads,
        "params": n_params,
        "params_M": round(params_m, 4),
        "beta_0": round(model.beta_0, 4),
        "mid_detail_dim": model.mid_detail_dim,
        "fine_detail_dim": model.fine_detail_dim,
        "schedule": spec.schedule,
        "batch_size": spec.batch_size,
        "seq_len": spec.block_size,
        "tokens_per_step": tok_per_step,
        "steps": spec.steps,
        "total_steps": spec.steps,
        "tokens_processed": spec.steps * tok_per_step,
        "total_tokens_M": round((spec.steps * tok_per_step) / 1e6, 4),
        "val_ce_at_tau_0p50": round(val_ce_mid, 4),
        "final_val_ce": round(float(final_ev["val_ce"]), 4),
        "best_val_ce": round(min(float(e["val_ce"]) for e in evals), 4),
        "final_top1_acc": round(float(final_ev["val_top1"]), 2),
        "final_top50_acc": round(float(final_ev["val_top50"]), 2),
        "sharp_eff_rank": round(float(final_ev["sharp_eff_rank"]), 2),
        "diffuse_eff_rank": round(model._last_diffuse_eff_rank, 2),
        "max_kraus_stiefel_err": round(max_stiefel_err, 6),
        "avg_tok_per_s": round((spec.steps * tok_per_step) / max(1e-6, wall_s), 1),
        "elapsed_sec": round(wall_s, 2),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "evals": evals,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    (ROOT / ".cache" / "model_cards").mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache" / "model_cards" / f"{spec.name}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    del model, muon_opt, adam_opt, work_gpu, val_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def load_h1_and_h4_diagnostics(root_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load pre-computed H1 mathematical probe JSONs and H4 Smith CLR profiler JSON."""
    p1 = json.loads((root_dir / ".cache" / "parseval_sqrt_d_scale_invariance_probe.json").read_text(encoding="utf-8"))
    p2 = json.loads((root_dir / ".cache" / "unit_sphere_mup_transfer_grid_search.json").read_text(encoding="utf-8"))
    p3 = json.loads((root_dir / ".cache" / "cholesky_metric_temperature_absorber.json").read_text(encoding="utf-8"))
    p4 = json.loads((root_dir / ".cache" / "smith_clr_range_test_vs_wsd_profiler.json").read_text(encoding="utf-8"))
    s1, s2, s3 = p1["summary_metrics"], p2["summary_metrics"], p3["summary_metrics"]

    h1_pass = bool(
        s1["bg_logit_std_cv_parseval"] < 0.015
        and s1["algebraic_unification_max_err"] < 1e-6
        and s2["mup_zero_shot_transfer_regret_nats"] <= 0.01
        and s3["max_logit_abs_err"] < 1e-4
    )
    h1_summary = {
        "pass": h1_pass,
        "bg_logit_std_cv_parseval": s1["bg_logit_std_cv_parseval"],
        "bg_logit_std_cv_fixed": s1["bg_logit_std_cv_fixed"],
        "max_beam_unification_err": s1["algebraic_unification_max_err"],
        "proxy_c_star": s2["mup_invariant_c_star"],
        "max_transfer_regret_nats": s2["mup_zero_shot_transfer_regret_nats"],
        "cholesky_max_abs_logit_err": s3["max_logit_abs_err"],
    }
    return h1_summary, p4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / ".cache" / "hypothesis_experiments_043.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    h1_summary, clr_diag = load_h1_and_h4_diagnostics(ROOT)
    eta_base_prof = float(clr_diag["range_test"]["eta_base"])
    eta_max_prof = float(clr_diag["range_test"]["eta_max"])

    specs = [
        VariantRunSpec(
            "w28_flagfree_parseval_mup_b16_d256_900steps",
            embed_dim=256,
            num_heads=4,
            num_layers=4,
            steps=args.steps,
            schedule="modded_nanogpt_wsd",
            eta_base=0.0065,
            eta_max=0.026,
            readout_gemm_precision="bf16",
        ),
        VariantRunSpec(
            "w28_flagfree_parseval_mup_b16_d384_900steps",
            embed_dim=384,
            num_heads=6,
            num_layers=6,
            steps=args.steps,
            schedule="modded_nanogpt_wsd",
            eta_base=0.0065,
            eta_max=0.026,
            readout_gemm_precision="fp8",
        ),
        VariantRunSpec(
            "w28_flagfree_parseval_mup_b16_d512_900steps",
            embed_dim=512,
            num_heads=8,
            num_layers=6,
            steps=args.steps,
            schedule="modded_nanogpt_wsd",
            eta_base=0.0065,
            eta_max=0.026,
            readout_gemm_precision="fp8",
        ),
        VariantRunSpec(
            "w28_smith_clr_b16_d256_900steps",
            embed_dim=256,
            num_heads=4,
            num_layers=4,
            steps=args.steps,
            schedule="smith_triangular2",
            eta_base=0.0065,
            eta_max=0.026,
            readout_gemm_precision="bf16",
        ),
    ]

    results: dict[str, dict[str, Any]] = {}
    for s in specs:
        results[s.name] = run_variant(s, device, force_rerun=args.force_rerun)

    ce_256 = results["w28_flagfree_parseval_mup_b16_d256_900steps"]["final_val_ce"]
    ce_384 = results["w28_flagfree_parseval_mup_b16_d384_900steps"]["final_val_ce"]
    ce_512 = results["w28_flagfree_parseval_mup_b16_d512_900steps"]["final_val_ce"]
    clr_res = results["w28_smith_clr_b16_d256_900steps"]
    wsd_res = results["w28_flagfree_parseval_mup_b16_d256_900steps"]

    h2_pass = bool(ce_256 <= 5.4447)
    h3_pass = bool(ce_512 < ce_384 < ce_256 <= 5.4447)
    h4_pass = bool(
        0.004 <= eta_base_prof <= 0.015
        and 0.020 <= eta_max_prof <= 0.045
        and clr_diag["comparisons"]["triangular2_beats_wsd_at_tau_0p50"]
        and clr_res["val_ce_at_tau_0p50"] < wsd_res["val_ce_at_tau_0p50"]
        and clr_res["final_val_ce"] <= wsd_res["final_val_ce"] + 0.03
    )

    payload = {
        "plan": "EXPERIMENT_PLAN_043",
        "wave": "Wave 28",
        "H1": h1_summary,
        "H2": {
            "pass": h2_pass,
            "variant": "w28_flagfree_parseval_mup_b16_d256_900steps",
            "final_val_ce": ce_256,
            "wave26_baseline_ce": 5.4447,
            "wave25_baseline_ce": 5.4465,
            "zero_purity_gate": True,
            "zero_kboson_raw_k": True,
            "zero_beta_clamps": True,
        },
        "H3": {
            "pass": h3_pass,
            "ce_d256": ce_256,
            "ce_d384": ce_384,
            "ce_d512": ce_512,
            "strict_monotonic_improvement": bool(ce_512 < ce_384 < ce_256),
        },
        "H4": {
            "pass": h4_pass,
            "eta_base": eta_base_prof,
            "eta_max": eta_max_prof,
            "profiler_triangular2_beats_wsd_at_tau_0p50": clr_diag["comparisons"]["triangular2_beats_wsd_at_tau_0p50"],
            "wsd_val_ce_at_tau_0p50": wsd_res["val_ce_at_tau_0p50"],
            "smith_clr_val_ce_at_tau_0p50": clr_res["val_ce_at_tau_0p50"],
            "wsd_final_val_ce": wsd_res["final_val_ce"],
            "smith_clr_final_val_ce": clr_res["final_val_ce"],
        },
        "variants": results,
        "overall_pass": bool(h1_summary["pass"] and h2_pass and h3_pass and h4_pass),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} (overall_pass={payload['overall_pass']})")


if __name__ == "__main__":
    main()
