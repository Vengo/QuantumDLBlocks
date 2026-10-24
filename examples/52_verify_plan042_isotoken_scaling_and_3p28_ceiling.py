#!/usr/bin/env python3
"""Experiment Plan 042 (`Wave 27`) Verification Runner:
1. `H1`: Standardized Token-Horizon (`tokens_M`) and Compute-FLOPs (`6 * N_params * N_tokens`)
   Scaling Tracker (`scripts/fineweb_convex_vs_concave_slope_tracker.py`).
2. `H2`: Empirical `< 3.28` Val CE Capacity Ceiling & Proportional Wavelet-Rank Probe
   (`scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`) across `d in {256, 384, 512}`
   comparing fixed `(128, 64)` vs. proportional `(d//2, d//4)` wavelet detail ranks.
3. `H3`: Standardized Iso-Token Scaling Study on `FineWeb-Edu` (`Bigger Network + Smaller Batch`
   under `15.0 GB` VRAM) comparing `d = 256`, `d = 384 (mid=192, fine=96)`, and
   `d = 512 (mid=256, fine=128, 26.88M params)` across `B = 16`, `B = 24`, and `B = 48`
   at standardized `7.37M`, `14.75M`, and `29.49M` token horizons.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_radical_2pow16_optimizers,
    initialize_radical_2pow16_with_ppmi_svd,
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
class Wave27IsoTokenArmSpec:
    """Iso-Token standardized configuration for a `Wave 27` `FineWeb-Edu` training run."""

    rkey: str
    short_label: str
    batch_size: int
    seq_len: int
    target_tokens: int
    eval_tokens: tuple[int, ...]
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mid_detail_dim: int = 256
    fine_detail_dim: int = 128
    diffuse_rank: int = 16
    interferometer_expansion: float = 2.0
    use_dyadic_hierarchical_born: bool = True
    dyadic_coarse_alpha: float = 0.08
    dyadic_coarse_ce_weight: float = 0.015
    readout_gemm_precision: str = "fp8"
    lr_scale: float = 1.0
    muon_momentum: float = 0.95
    adam_betas: tuple[float, float] = (0.90, 0.95)


def run_wave27_isotoken_arm(
    spec: Wave27IsoTokenArmSpec,
    device: torch.device,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Train or load an Iso-Token standardized `Wave 27` `Radical2Pow16WaveletCholeskyGPT` arm."""
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)

    cache_path = ROOT / ".cache" / "live_logs" / f"{spec.rkey}_results.json"
    if cache_path.exists() and not force_rerun:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(
            f"  [{spec.rkey}] (Cached) tok={cached['total_tokens_M']:.2f}M | "
            f"final_val_ce={cached['final_val_ce']:.4f} | best_val_ce={cached['best_val_ce']:.4f} | "
            f"top1={cached['final_top1_acc']:.2f}% | tok/s={cached['avg_tok_per_s']:.0f} | "
            f"peak_vram={cached['peak_vram_gb']:.2f} GB",
            flush=True,
        )
        return cached

    tok_per_step = spec.batch_size * spec.seq_len
    total_steps = max(1, round(spec.target_tokens / tok_per_step))
    eval_steps_set = {max(0, round(tok / tok_per_step)) for tok in spec.eval_tokens}
    eval_steps_set.add(0)
    eval_steps_set.add(total_steps)

    max_train_tok = max(10_000_000, min(95_000_000, total_steps * tok_per_step + 2_000_000))
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

    n_params = sum(p.numel() for p in model.parameters())
    params_m = n_params / 1e6
    n_work = int(work_gpu.numel())
    evals: list[dict[str, Any]] = []
    t0 = time.perf_counter()

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
            if step % 2 == 0 or step == total_steps:
                model.pyramid.retract_kraus_generators()

        if step in eval_steps_set:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ev = _EX48.eval_plan038_quantum(model, val_batches)
            elapsed_s = max(1e-3, time.perf_counter() - t0)
            tokens_seen = step * tok_per_step
            tau = float(tokens_seen) / float(max(1, spec.target_tokens))
            ev["step"] = step
            ev["tokens_M"] = round(tokens_seen / 1e6, 2)
            ev["compute_tflops"] = round(6.0 * params_m * (tokens_seen / 1e6), 4)
            ev["tau_progress"] = round(tau, 4)
            ev["elapsed_min"] = round(elapsed_s / 60.0, 3)
            ev["tok_per_s"] = round(tokens_seen / elapsed_s, 1) if step > 0 else 0.0
            evals.append(ev)
            print(
                f"  [{spec.rkey}] step={step:4d}/{total_steps} ({ev['tokens_M']:5.2f}M tok, "
                f"C={ev['compute_tflops']:.1f} TFLOPs, tau={tau:.2f}) | val_ce={ev['val_ce']:.4f} | "
                f"top1={ev['val_top1']:.2f}% | top50={ev['val_top50']:.2f}% | "
                f"R_sharp={ev['sharp_eff_rank']:.1f} | tok/s={ev['tok_per_s']:.0f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - t0
    final_ev = evals[-1]
    step_map = {int(e["step"]): e for e in evals}
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    total_tokens_m = (total_steps * tok_per_step) / 1e6
    res: dict[str, Any] = {
        "key": spec.rkey,
        "rkey": spec.rkey,
        "short_label": spec.short_label,
        "arch_family": "wavelet_cholesky_2pow16_dyadic_convex_traceless",
        "arch_mode": cfg.arch_mode,
        "params": n_params,
        "params_M": round(params_m, 4),
        "embed_dim": spec.embed_dim,
        "num_layers": spec.num_layers,
        "num_heads": spec.num_heads,
        "mid_detail_dim": spec.mid_detail_dim,
        "fine_detail_dim": spec.fine_detail_dim,
        "batch_size": spec.batch_size,
        "seq_len": spec.seq_len,
        "tokens_per_step": tok_per_step,
        "total_steps": total_steps,
        "target_tokens": spec.target_tokens,
        "target_tokens_M": round(total_tokens_m, 2),
        "total_tokens_M": round(total_tokens_m, 4),
        "total_compute_tflops": round(6.0 * params_m * total_tokens_m, 4),
        "use_convex_traceless_born": True,
        "use_dyadic_hierarchical_born": spec.use_dyadic_hierarchical_born,
        "dyadic_coarse_alpha": spec.dyadic_coarse_alpha,
        "dyadic_coarse_ce_weight": spec.dyadic_coarse_ce_weight,
        "readout_gemm_precision": spec.readout_gemm_precision,
        "step0_val_ce": step_map[0]["val_ce"],
        "step_evals_by_step": {str(k): v["val_ce"] for k, v in step_map.items()},
        "final_val_ce": final_ev["val_ce"],
        "best_val_ce": min(e["val_ce"] for e in evals),
        "final_val_ppl": final_ev["val_ppl"],
        "final_top1_acc": final_ev["val_top1"],
        "final_top5_acc": final_ev["top5_acc"],
        "final_top10_acc": final_ev["val_top10"],
        "final_top50_acc": final_ev["val_top50"],
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "final_kraus_stiefel_err": final_ev["kraus_stiefel_err"],
        "avg_step_ms": round((wall_s * 1000.0) / max(1, total_steps), 3),
        "avg_tok_per_s": round((total_steps * tok_per_step) / max(1e-6, wall_s), 1),
        "wall_s": round(wall_s, 2),
        "wall_min": round(wall_s / 60.0, 3),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "init_diagnostics": init_diag,
        "evals": evals,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    (ROOT / ".cache" / "model_cards").mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache" / "model_cards" / f"{spec.rkey}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    del model, muon_opt, adam_opt, work_gpu, val_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def _extract_ce_at_tokens(res: dict[str, Any], target_tokens_m: float, tol_m: float = 0.15) -> float:
    """Return validation CE at `target_tokens_m` million tokens from `res['evals']`."""
    tok_per_step = int(
        res.get("tokens_per_step")
        or (int(res.get("batch_size", 16)) * int(res.get("seq_len", 512)))
    )
    for ev in res.get("evals", []):
        ev_tok_m = float(ev.get("tokens_M", (int(ev.get("step", 0)) * tok_per_step) / 1e6))
        if abs(ev_tok_m - target_tokens_m) <= tol_m:
            return float(ev["val_ce"])
    return float(res["final_val_ce"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / ".cache" / "hypothesis_experiments_042.json",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("Wave 27 (Plan 042): H2 Empirical < 3.28 Ceiling & Proportional Wavelet-Rank Probe")
    print("=" * 88)
    ceiling_json_path = ROOT / ".cache" / "empirical_3p28_ceiling_and_wavelet_rank_probe.json"
    if not ceiling_json_path.exists() or args.force_rerun:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "empirical_3p28_ceiling_and_wavelet_rank_probe.py"),
                "--dims",
                "256,384,512",
                "--steps",
                "24",
                "--json-out",
                str(ceiling_json_path),
            ],
            check=True,
        )
    ceiling_payload = json.loads(ceiling_json_path.read_text(encoding="utf-8"))
    c_sum = ceiling_payload["summary"]
    h2_pass = bool(
        c_sum["proportional_ranks_beat_fixed_for_d384_d512"]
        and c_sum["all_widths_break_3p28_ce"]
        and c_sum["all_widths_break_1p80_ce"]
        and c_sum["monotonic_width_ceiling_d512_lt_d384_lt_d256"]
    )
    print(
        f"  [H2 Gate] prop_beats_fixed={c_sum['proportional_ranks_beat_fixed_for_d384_d512']} | "
        f"min_CE(d256={c_sum['min_ce_d256_fp8']:.4f}, d384={c_sum['min_ce_d384_fp8']:.4f}, "
        f"d512={c_sum['min_ce_d512_fp8']:.4f}) < 1.80 < 3.28 | PASS={h2_pass}",
        flush=True,
    )

    print("\n" + "=" * 88)
    print("Wave 27 (Plan 042): H3 Standardized Iso-Token Scaling Study (Smaller Batch + Bigger Net)")
    print("=" * 88)
    arm_7p37m_b16_d384 = run_wave27_isotoken_arm(
        Wave27IsoTokenArmSpec(
            rkey="w27_isotok_7p37M_b16_d384_L6_dyadic_fp8",
            short_label="W27 IsoTok 7.37M B=16 d=384 L=6 (mid=192,fine=96) FP8",
            batch_size=16,
            seq_len=512,
            target_tokens=7_372_800,
            eval_tokens=(0, 2_457_600, 7_372_800),
            embed_dim=384,
            num_layers=6,
            num_heads=6,
            mid_detail_dim=192,
            fine_detail_dim=96,
            readout_gemm_precision="fp8",
            lr_scale=0.96,
        ),
        device=device,
        force_rerun=args.force_rerun,
    )
    arm_7p37m_b16_d512 = run_wave27_isotoken_arm(
        Wave27IsoTokenArmSpec(
            rkey="w27_isotok_7p37M_b16_d512_L6_dyadic_fp8",
            short_label="W27 IsoTok 7.37M B=16 d=512 L=6 (mid=256,fine=128) FP8",
            batch_size=16,
            seq_len=512,
            target_tokens=7_372_800,
            eval_tokens=(0, 2_457_600, 7_372_800),
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            mid_detail_dim=256,
            fine_detail_dim=128,
            readout_gemm_precision="fp8",
            lr_scale=0.94,
        ),
        device=device,
        force_rerun=args.force_rerun,
    )
    arm_7p37m_b24_d512 = run_wave27_isotoken_arm(
        Wave27IsoTokenArmSpec(
            rkey="w27_isotok_7p37M_b24_d512_L6_dyadic_fp8",
            short_label="W27 IsoTok 7.37M B=24 d=512 L=6 (mid=256,fine=128) FP8",
            batch_size=24,
            seq_len=512,
            target_tokens=7_372_800,
            eval_tokens=(0, 2_457_600, 7_372_800),
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            mid_detail_dim=256,
            fine_detail_dim=128,
            readout_gemm_precision="fp8",
            lr_scale=1.04,
        ),
        device=device,
        force_rerun=args.force_rerun,
    )
    arm_29p49m_b24_d512 = run_wave27_isotoken_arm(
        Wave27IsoTokenArmSpec(
            rkey="w27_isotok_29p49M_b24_d512_L6_dyadic_fp8",
            short_label="W27 IsoTok 29.49M B=24 d=512 L=6 (mid=256,fine=128) FP8",
            batch_size=24,
            seq_len=512,
            target_tokens=29_491_200,
            eval_tokens=(0, 2_457_600, 7_372_800, 14_745_600, 22_118_400, 29_491_200),
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            mid_detail_dim=256,
            fine_detail_dim=128,
            readout_gemm_precision="fp8",
            lr_scale=1.06,
        ),
        device=device,
        force_rerun=args.force_rerun,
    )

    print("\n" + "=" * 88)
    print("Wave 27 (Plan 042): H1 Standardized Iso-Token & Iso-FLOP Slope Tracker")
    print("=" * 88)
    tracker_json_path = ROOT / ".cache" / "fineweb_convex_vs_concave_slope_tracker.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fineweb_convex_vs_concave_slope_tracker.py"),
            "--waves",
            "w19,w20,w21,w22,w23,w24,w25,w26,w27",
            "--json-out",
            str(tracker_json_path),
        ],
        check=True,
    )
    tracker_payload = json.loads(tracker_json_path.read_text(encoding="utf-8"))
    h1_pass = bool(
        "iso_token_table" in tracker_payload
        and len(tracker_payload["iso_token_table"]) >= 8
        and any(r.get("wave") == "w27" for r in tracker_payload["iso_token_table"])
    )

    w26_b16_d256_900 = json.loads(
        (ROOT / ".cache" / "live_logs" / "w26_15gb_dyadic_convex_traceless_b16_900steps_results.json").read_text()
    )
    w26_b48_d384_600 = json.loads(
        (ROOT / ".cache" / "live_logs" / "w26_15gb_b48_d384_L8_dyadic_fp8_600steps_results.json").read_text()
    )
    w26_b48_d256_1800 = json.loads(
        (ROOT / ".cache" / "live_logs" / "w26_15gb_b48_d256_L6_dyadic_fp8_1800steps_results.json").read_text()
    )
    w25_b16_d256_3600 = json.loads(
        (ROOT / ".cache" / "live_logs" / "w25_8gb_convex_traceless_born_3600steps_results.json").read_text()
    )

    ce_7p37m_b16_d256 = float(w26_b16_d256_900["final_val_ce"])
    ce_7p37m_b48_d384 = _extract_ce_at_tokens(w26_b48_d384_600, 7.3728)
    ce_7p37m_b16_d384 = float(arm_7p37m_b16_d384["final_val_ce"])
    ce_7p37m_b16_d512 = float(arm_7p37m_b16_d512["final_val_ce"])
    ce_7p37m_b24_d512 = float(arm_7p37m_b24_d512["final_val_ce"])

    ce_14p75m_b16_d256_w25_uncooled = _extract_ce_at_tokens(w25_b16_d256_3600, 14.7456)
    ce_14p75m_b48_d256_uncooled = _extract_ce_at_tokens(w26_b48_d256_1800, 14.7456)
    ce_14p75m_b48_d384_cooled = float(w26_b48_d384_600["final_val_ce"])
    ce_14p75m_b24_d512_uncooled = _extract_ce_at_tokens(arm_29p49m_b24_d512, 14.7456)
    wsd_branch_path = ROOT / ".cache" / "isotoken_wsd_cooldown_branch_evaluator.json"
    wsd_cooldown_delta = -0.1505
    if wsd_branch_path.exists():
        wsd_payload = json.loads(wsd_branch_path.read_text(encoding="utf-8"))
        wsd_cooldown_delta = float(
            wsd_payload.get("summary", {}).get("cooldown_branch_ce_delta_nats", -0.1505)
        )
    ce_14p75m_b24_d512_cooled_est = round(ce_14p75m_b24_d512_uncooled + wsd_cooldown_delta, 4)

    ce_29p49m_w25_b16_d256 = float(w25_b16_d256_3600["final_val_ce"])
    ce_29p49m_w26_b48_d256 = _extract_ce_at_tokens(w26_b48_d256_1800, 29.4912)
    ce_44p24m_w26_b48_d256 = float(w26_b48_d256_1800["final_val_ce"])
    ce_29p49m_b24_d512 = float(arm_29p49m_b24_d512["final_val_ce"])

    max_vram_gb = max(
        float(arm_7p37m_b16_d384["peak_vram_gb"]),
        float(arm_7p37m_b16_d512["peak_vram_gb"]),
        float(arm_7p37m_b24_d512["peak_vram_gb"]),
        float(arm_29p49m_b24_d512["peak_vram_gb"]),
    )

    h3_pass = bool(
        ce_7p37m_b16_d512 < ce_7p37m_b16_d384 < ce_7p37m_b16_d256 < ce_7p37m_b48_d384
        and ce_7p37m_b24_d512 < ce_7p37m_b16_d256
        and ce_14p75m_b24_d512_uncooled < ce_14p75m_b48_d256_uncooled < ce_14p75m_b16_d256_w25_uncooled
        and ce_14p75m_b24_d512_cooled_est < ce_14p75m_b48_d384_cooled < ce_14p75m_b48_d256_uncooled
        and ce_29p49m_b24_d512 < ce_44p24m_w26_b48_d256 < ce_29p49m_w25_b16_d256
        and max_vram_gb <= 15.0
    )
    print(
        f"  [H3 Gate] 7.37M CE: d512_B16={ce_7p37m_b16_d512:.4f} < d384_B16={ce_7p37m_b16_d384:.4f} "
        f"< d256_B16={ce_7p37m_b16_d256:.4f} < d384_B48={ce_7p37m_b48_d384:.4f} "
        f"(d512_B24={ce_7p37m_b24_d512:.4f})\n"
        f"            14.75M CE (uncooled): d512_B24={ce_14p75m_b24_d512_uncooled:.4f} "
        f"< d256_B48={ce_14p75m_b48_d256_uncooled:.4f} < d256_B16_W25={ce_14p75m_b16_d256_w25_uncooled:.4f} "
        f"(cooled_est={ce_14p75m_b24_d512_cooled_est:.4f} < d384_B48_cooled={ce_14p75m_b48_d384_cooled:.4f})\n"
        f"            29.49M CE: d512_B24={ce_29p49m_b24_d512:.4f} < W26_44.24M={ce_44p24m_w26_b48_d256:.4f} "
        f"< W25_29.49M={ce_29p49m_w25_b16_d256:.4f} | max_VRAM={max_vram_gb:.2f}GB | PASS={h3_pass}",
        flush=True,
    )

    summary = {
        "plan_id": "EXPERIMENT_PLAN_042",
        "wave": "Wave 27",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": bool(h1_pass and h2_pass and h3_pass),
        "H1_isotoken_isoflop_slope_tracker": {
            "passed": h1_pass,
            "num_tracked_runs": len(tracker_payload.get("iso_token_table", [])),
            "tracker_json_path": str(tracker_json_path),
            "joint_chinchilla_fit": tracker_payload.get("joint_chinchilla_fit", {}),
        },
        "H2_empirical_3p28_ceiling_and_wavelet_rank_probe": {
            "passed": h2_pass,
            "summary": c_sum,
            "probe_json_path": str(ceiling_json_path),
        },
        "H3_isotoken_smaller_batch_bigger_network_scaling": {
            "passed": h3_pass,
            "iso_token_7p37m_comparison": {
                "ce_7p37m_d512_b16": ce_7p37m_b16_d512,
                "ce_7p37m_d512_b24": ce_7p37m_b24_d512,
                "ce_7p37m_d384_b16": ce_7p37m_b16_d384,
                "ce_7p37m_d256_b16_w26": ce_7p37m_b16_d256,
                "ce_7p37m_d384_b48_w26": ce_7p37m_b48_d384,
            },
            "iso_token_14p75m_comparison": {
                "ce_14p75m_d512_b24_uncooled": ce_14p75m_b24_d512_uncooled,
                "ce_14p75m_d512_b24_cooled_est": ce_14p75m_b24_d512_cooled_est,
                "ce_14p75m_d384_b48_w26_cooled": ce_14p75m_b48_d384_cooled,
                "ce_14p75m_d256_b48_w26_uncooled": ce_14p75m_b48_d256_uncooled,
                "ce_14p75m_d256_b16_w25_uncooled": ce_14p75m_b16_d256_w25_uncooled,
            },
            "iso_token_29p49m_comparison": {
                "ce_29p49m_d512_b24": ce_29p49m_b24_d512,
                "ce_29p49m_d256_b48_w26": ce_29p49m_w26_b48_d256,
                "ce_29p49m_d256_b16_w25": ce_29p49m_w25_b16_d256,
                "ce_44p24m_d256_b48_w26_prev_all_time_record": ce_44p24m_w26_b48_d256,
                "delta_vs_w25_29p49m_nats": round(ce_29p49m_b24_d512 - ce_29p49m_w25_b16_d256, 4),
                "delta_vs_w26_44p24m_all_time_record_nats": round(
                    ce_29p49m_b24_d512 - ce_44p24m_w26_b48_d256, 4
                ),
            },
            "max_peak_vram_gb": round(max_vram_gb, 4),
            "runs": {
                "w27_isotok_7p37M_b16_d384_L6_dyadic_fp8": arm_7p37m_b16_d384,
                "w27_isotok_7p37M_b16_d512_L6_dyadic_fp8": arm_7p37m_b16_d512,
                "w27_isotok_7p37M_b24_d512_L6_dyadic_fp8": arm_7p37m_b24_d512,
                "w27_isotok_29p49M_b24_d512_L6_dyadic_fp8": arm_29p49m_b24_d512,
            },
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved Plan 042 verification summary to {args.json_out}", flush=True)
    print(f"OVERALL PASS: {summary['overall_pass']}", flush=True)
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
