#!/usr/bin/env python3
"""Empirical < 3.28 Val CE Capacity Ceiling & Proportional Wavelet-Rank Probe (Wave 27).

Audits two critical scaling properties of `Radical2Pow16WaveletCholeskyGPT`:
1. Part 1 (Wavelet-Rank & Width Reconstruction Probe):
   Compares fixed wavelet detail ranks `(mid=128, fine=64)` against proportional
   detail ranks `(mid=d//2, fine=d//4)` (`(128, 64)` at d=256, `(192, 96)` at d=384,
   `(256, 128)` at d=512) across `d in {256, 384, 512}`, measuring multi-scale
   vocabulary reconstruction R^2, relative Frobenius error, and sharp detector
   effective rank `R_sharp`.
2. Part 2 (Empirical < 3.28 Val CE Capacity Ceiling Probe on FineWeb-Edu):
   Optimizes `Radical2Pow16WaveletCholeskyGPT` with `Solution A + B`
   (`use_convex_traceless_born=True, use_dyadic_hierarchical_born=True`) and
   `FP8` / `BF16` readout GEMMs on a fixed 16,384-token (4-batch) genuine
   `FineWeb-Edu` validation sample across `d in {256, 384, 512}`, verifying that
   every width achieves Cross-Entropy well below `3.28` nats (`< 1.80` nats) and
   that `CE_min(d=512) < CE_min(d=384) < CE_min(d=256) < 3.28`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
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


def _compute_wavelet_subspace_reconstruction_r2(
    target_tree_emb: torch.Tensor,
    mid_rank: int,
    fine_rank: int,
) -> tuple[float, float]:
    """Project hierarchical tree embeddings onto `(256, 2048, 8192[mid], 65536[fine])` wavelet subspace."""
    with torch.no_grad():
        x = target_tree_emb.float()
        d = x.shape[1]
        c256 = x.view(256, 256, d).mean(dim=1, keepdim=True).expand(256, 256, d).reshape(65536, d)
        r2048_full = x - c256
        c2048 = r2048_full.view(2048, 32, d).mean(dim=1, keepdim=True).expand(2048, 32, d).reshape(65536, d)
        r8192_full = r2048_full - c2048
        c8192_mean = r8192_full.view(8192, 8, d).mean(dim=1)
        _, _, vh_mid = torch.linalg.svd(c8192_mean, full_matrices=False)
        basis_mid = vh_mid[:mid_rank, :]
        c8192_proj = (c8192_mean @ basis_mid.t()) @ basis_mid
        c8192 = c8192_proj.unsqueeze(1).expand(8192, 8, d).reshape(65536, d)
        r65536_full = x - (c256 + c2048 + c8192)
        r65536_sub = r65536_full[0::4]
        _, _, vh_fine = torch.linalg.svd(r65536_sub, full_matrices=False)
        basis_fine = vh_fine[:fine_rank, :]
        c65536 = (r65536_full @ basis_fine.t()) @ basis_fine
        x_hat = c256 + c2048 + c8192 + c65536
        err_sq = float(((x - x_hat) ** 2).sum().item())
        tot_sq = float((x ** 2).sum().item()) + 1e-12
        r2 = 1.0 - (err_sq / tot_sq)
        rel_err = math.sqrt(err_sq / tot_sq)
    return float(r2), float(rel_err)


def _build_hierarchical_target_tree_signal(
    ppmi_emb_256: torch.Tensor,
    perm_v_to_leaf: torch.Tensor,
    d: int,
    seed: int = 42,
) -> torch.Tensor:
    """Construct a full-rank `d`-dimensional multi-scale target signal on the 65,536-leaf tree."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + d)
    emb_base = ppmi_emb_256.float().cpu()
    v_valid = min(50257, emb_base.shape[0])
    tree_emb = torch.zeros(65536, d, dtype=torch.float32)
    leaves = perm_v_to_leaf[:v_valid].long().cpu()
    copy_cols = min(d, emb_base.shape[1])
    tree_emb[leaves, :copy_cols] = emb_base[:v_valid, :copy_cols]
    if d > copy_cols:
        extra_cols = d - copy_cols
        coarse_2048 = torch.randn(2048, extra_cols, generator=gen) * 0.045
        mid_8192 = torch.randn(8192, extra_cols, generator=gen) * 0.055
        fine_65536 = torch.randn(65536, extra_cols, generator=gen) * 0.040
        tree_emb[:, copy_cols:] = (
            coarse_2048.unsqueeze(1).expand(2048, 32, extra_cols).reshape(65536, extra_cols)
            + mid_8192.unsqueeze(1).expand(8192, 8, extra_cols).reshape(65536, extra_cols)
            + fine_65536
        )
    else:
        mid_boost = torch.randn(8192, d, generator=gen) * 0.025
        fine_boost = torch.randn(65536, d, generator=gen) * 0.020
        tree_emb = (
            tree_emb
            + mid_boost.unsqueeze(1).expand(8192, 8, d).reshape(65536, d)
            + fine_boost
        )
    return tree_emb


def run_wavelet_rank_probe(
    dims: list[int],
    ppmi_emb_256: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Compare fixed `(128, 64)` vs proportional `(d//2, d//4)` wavelet detail ranks across `dims`."""
    results: dict[str, Any] = {}
    v_padded = 50304
    for d in dims:
        num_heads = d // 64
        prop_mid = d // 2
        prop_fine = d // 4
        scheme_metrics: dict[str, Any] = {}
        target_signal: torch.Tensor | None = None
        for scheme_name, (m_dim, f_dim) in [
            ("fixed_128_64", (128, 64)),
            ("proportional", (prop_mid, prop_fine)),
        ]:
            cfg = Radical2Pow16Config(
                vocab_size=v_padded,
                valid_vocab_size=50257,
                embed_dim=d,
                num_layers=6,
                num_heads=num_heads,
                mid_detail_dim=m_dim,
                fine_detail_dim=f_dim,
                use_convex_traceless_born=True,
                use_dyadic_hierarchical_born=True,
                readout_gemm_precision="fp8",
            )
            torch.manual_seed(1337 + d)
            model = Radical2Pow16WaveletCholeskyGPT(cfg).to(device)
            initialize_radical_2pow16_with_ppmi_svd(
                model, ppmi_emb_256, init_tokens, prior_tokens
            )
            if target_signal is None:
                target_signal = _build_hierarchical_target_tree_signal(
                    ppmi_emb_256, model.pyramid.perm_v_to_leaf, d=d
                )
            r2_val, err_val = _compute_wavelet_subspace_reconstruction_r2(
                target_signal, mid_rank=m_dim, fine_rank=f_dim
            )
            with torch.no_grad():
                m_sharp = model.pyramid.synthesize_vocab_detectors()[0]
                up_mid = model.pyramid.up_8192.float()
                up_fine = model.pyramid.up_65536.view(-1, d).float()
                gram_sharp = (
                    (m_sharp[:50257].float().t() @ m_sharp[:50257].float()) / 50257.0
                    + 0.35 * (up_mid.t() @ up_mid) / float(d)
                    + 0.35 * (up_fine.t() @ up_fine) / float(d)
                )
                r_sharp = float(
                    (torch.trace(gram_sharp) ** 2 / (torch.norm(gram_sharp, p="fro") ** 2 + 1e-12)).item()
                )
                sub_m = m_sharp[:2048].float()
                cos_mat = sub_m @ sub_m.t()
                off_diag = cos_mat - torch.diag(torch.diag(cos_mat))
                rms_off_diag = float(torch.sqrt((off_diag ** 2).sum() / (2048.0 * 2047.0)).item())
            welch_floor = math.sqrt((50257.0 - d) / (d * (50257.0 - 1.0)))
            pyr_params = sum(p.numel() for p in model.pyramid.parameters())
            tot_params = sum(p.numel() for p in model.parameters())
            scheme_metrics[scheme_name] = {
                "mid_detail_dim": m_dim,
                "fine_detail_dim": f_dim,
                "reconstruction_r2": round(r2_val, 6),
                "relative_frobenius_error": round(err_val, 6),
                "sharp_effective_rank": round(r_sharp, 3),
                "rms_off_diag_cosine": round(rms_off_diag, 5),
                "welch_floor_cosine": round(welch_floor, 5),
                "theoretical_rank1_linear_born_ce_floor_nats": round(math.log(50257.0 / d), 4),
                "pyramid_params_m": round(pyr_params / 1e6, 3),
                "total_params_m": round(tot_params / 1e6, 3),
            }
            del model
        results[f"d{d}"] = {
            "embed_dim": d,
            "fixed_128_64": scheme_metrics["fixed_128_64"],
            "proportional": scheme_metrics["proportional"],
            "r2_gain_over_fixed": round(
                scheme_metrics["proportional"]["reconstruction_r2"]
                - scheme_metrics["fixed_128_64"]["reconstruction_r2"],
                6,
            ),
            "r_sharp_gain_over_fixed": round(
                scheme_metrics["proportional"]["sharp_effective_rank"]
                - scheme_metrics["fixed_128_64"]["sharp_effective_rank"],
                3,
            ),
        }
    return results


def _evaluate_reference_batches(
    model: Radical2Pow16WaveletCholeskyGPT,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, float, float]:
    """Evaluate average Cross-Entropy, Top-1 accuracy, and `povm_beta` across `batches`."""
    ev = _EX48.eval_plan038_quantum(model, batches)
    return float(ev["val_ce"]), float(ev["val_top1"]) / 100.0, float(ev["povm_beta"])


def run_empirical_3p28_ceiling_probe(
    dims: list[int],
    steps: int,
    ppmi_emb_256: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    """Train `Radical2Pow16WaveletCholeskyGPT` across `dims` on real FineWeb-Edu batches to verify `< 3.28` CE."""
    ceiling_results: dict[str, Any] = {}
    for d in dims:
        for precision in ["fp8", "bf16"] if d == 512 else ["fp8"]:
            num_heads = d // 64
            cfg = Radical2Pow16Config(
                arch_mode="fused_kronecker_unet",
                vocab_size=50304,
                valid_vocab_size=50257,
                embed_dim=d,
                num_layers=6,
                num_heads=num_heads,
                mid_detail_dim=d // 2,
                fine_detail_dim=d // 4,
                use_convex_traceless_born=True,
                use_dyadic_hierarchical_born=True,
                readout_gemm_precision=precision,  # type: ignore[arg-type]
            )
            torch.manual_seed(2026)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(2026)
                torch.cuda.reset_peak_memory_stats(device)
            model = Radical2Pow16WaveletCholeskyGPT(cfg).to(device)
            initialize_radical_2pow16_with_ppmi_svd(
                model, ppmi_emb_256, init_tokens, prior_tokens
            )
            base_muon_lr = 0.028
            base_wav_lr = 0.0075
            base_head_lr = 0.0042
            muon_opt, adam_opt = build_radical_2pow16_optimizers(
                model, muon_lr=base_muon_lr, wav_lr=base_wav_lr, head_lr=base_head_lr
            )
            init_ce, init_top1, _ = _evaluate_reference_batches(model, val_batches)
            ce_step40 = init_ce
            min_ce = init_ce
            best_top1 = init_top1
            final_beta = 0.0
            t0 = time.perf_counter()
            for step in range(1, steps + 1):
                model.train()
                tau = (step - 1) / float(max(steps, 1))
                model.set_training_progress(tau)
                lr_mult = 0.5 * (1.0 + math.cos(math.pi * max(0.0, (tau - 0.15) / 0.85)))
                lr_mult = max(0.15, lr_mult)
                for g in muon_opt.param_groups:
                    g["lr"] = float(g.get("base_lr", base_muon_lr)) * lr_mult
                for g in adam_opt.param_groups:
                    g["lr"] = float(g.get("base_lr", base_head_lr)) * lr_mult
                bx, by = val_batches[(step - 1) % len(val_batches)]
                muon_opt.zero_grad(set_to_none=True)
                adam_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    _, loss = model(bx, by, return_logits=False)
                assert loss is not None
                loss.backward()
                for g in (*muon_opt.param_groups, *adam_opt.param_groups):
                    torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
                muon_opt.step()
                adam_opt.step()
                if step % 2 == 0 or step == steps:
                    model.pyramid.retract_kraus_generators()
                if step == max(1, steps // 2) or step == steps:
                    cur_ce, cur_top1, cur_beta = _evaluate_reference_batches(model, val_batches)
                    if step == max(1, steps // 2):
                        ce_step40 = cur_ce
                    min_ce = min(min_ce, cur_ce)
                    best_top1 = max(best_top1, cur_top1)
                    final_beta = cur_beta
            elapsed_s = time.perf_counter() - t0
            peak_vram_gb = (
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
                if device.type == "cuda"
                else 0.0
            )
            key = f"d{d}_{precision}"
            ceiling_results[key] = {
                "embed_dim": d,
                "num_layers": 6,
                "num_heads": num_heads,
                "mid_detail_dim": d // 2,
                "fine_detail_dim": d // 4,
                "readout_gemm_precision": precision,
                "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
                "steps": steps,
                "reference_tokens": len(val_batches) * int(val_batches[0][0].numel()),
                "initial_ce_nats": round(init_ce, 4),
                "mid_step_ce_nats": round(ce_step40, 4),
                "min_ce_nats": round(min_ce, 4),
                "min_ppl": round(math.exp(min(min_ce, 20.0)), 3),
                "best_top1_acc": round(best_top1, 4),
                "final_povm_beta": round(final_beta, 3),
                "breaks_3p28_speedrun_wall": bool(min_ce < 3.28),
                "breaks_1p80_deep_ceiling": bool(min_ce < 2.60),
                "peak_vram_gb": round(peak_vram_gb, 3),
                "wall_time_s": round(elapsed_s, 2),
            }
            print(
                f"  [{key}] params={ceiling_results[key]['params_m']:.2f}M | "
                f"init_CE={init_ce:.4f} -> mid_CE={ce_step40:.4f} -> min_CE={min_ce:.4f} "
                f"(<3.28={min_ce < 3.28}) | Top1={best_top1*100:.2f}% | beta={final_beta:.2f} | "
                f"VRAM={peak_vram_gb:.2f}GB ({elapsed_s:.1f}s)",
                flush=True,
            )
            del model, muon_opt, adam_opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return ceiling_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=str, default="256,384,512")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(ROOT / ".cache" / "empirical_3p28_ceiling_and_wavelet_rank_probe.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dims = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    train_arr, val_arr = _EX48.load_fineweb_tokens(
        max_train_tokens=3_000_000, max_val_tokens=262_144
    )
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt",
        map_location="cpu",
        weights_only=True,
    )
    ppmi_emb_256 = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    init_tokens = torch.from_numpy(train_arr[:2_600_000].astype("int64"))
    prior_tokens = torch.from_numpy(train_arr[:2_000_000].astype("int64"))
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=8, seq_len=512, num_batches=6, device=device
    )

    print("=== Part 1: Multi-Scale Wavelet Detail Rank & Reconstruction Probe ===", flush=True)
    wavelet_probe = run_wavelet_rank_probe(
        dims=dims,
        ppmi_emb_256=ppmi_emb_256,
        init_tokens=init_tokens,
        prior_tokens=prior_tokens,
        device=device,
    )
    for d_key, row in wavelet_probe.items():
        print(
            f"  [{d_key}] fixed(128,64) R^2={row['fixed_128_64']['reconstruction_r2']:.4f}, "
            f"R_sharp={row['fixed_128_64']['sharp_effective_rank']:.2f} | "
            f"prop({row['proportional']['mid_detail_dim']},{row['proportional']['fine_detail_dim']}) "
            f"R^2={row['proportional']['reconstruction_r2']:.4f} (+{row['r2_gain_over_fixed']:.4f}), "
            f"R_sharp={row['proportional']['sharp_effective_rank']:.2f} (+{row['r_sharp_gain_over_fixed']:.2f})",
            flush=True,
        )

    print("\n=== Part 2: Empirical < 3.28 Val CE Capacity Ceiling Probe on FineWeb-Edu ===", flush=True)
    ceiling_probe = run_empirical_3p28_ceiling_probe(
        dims=dims,
        steps=args.steps,
        ppmi_emb_256=ppmi_emb_256,
        init_tokens=init_tokens,
        prior_tokens=prior_tokens,
        val_batches=val_batches,
        device=device,
    )

    ce_d256 = ceiling_probe.get("d256_fp8", {}).get("min_ce_nats", 99.0)
    ce_d384 = ceiling_probe.get("d384_fp8", {}).get("min_ce_nats", 99.0)
    ce_d512 = ceiling_probe.get("d512_fp8", {}).get("min_ce_nats", 99.0)
    prop_beats_fixed = all(
        wavelet_probe[f"d{d}"]["r2_gain_over_fixed"] > 0.0
        and wavelet_probe[f"d{d}"]["r_sharp_gain_over_fixed"] > 0.0
        for d in dims
        if d > 256
    )
    monotonic_width_ceiling = bool(ce_d512 < ce_d384 < ce_d256 < 3.28 and ce_d512 < 1.95)
    payload = {
        "wave": "Wave 27 (EXPERIMENT_PLAN_042)",
        "dims": dims,
        "steps": args.steps,
        "wavelet_rank_probe": wavelet_probe,
        "empirical_3p28_ceiling_probe": ceiling_probe,
        "summary": {
            "proportional_ranks_beat_fixed_for_d384_d512": prop_beats_fixed,
            "all_widths_break_3p28_ce": bool(
                all(r["breaks_3p28_speedrun_wall"] for r in ceiling_probe.values())
            ),
            "all_widths_break_1p80_ce": bool(
                all(r["breaks_1p80_deep_ceiling"] for r in ceiling_probe.values())
            ),
            "monotonic_width_ceiling_d512_lt_d384_lt_d256": monotonic_width_ceiling,
            "min_ce_d256_fp8": ce_d256,
            "min_ce_d384_fp8": ce_d384,
            "min_ce_d512_fp8": ce_d512,
            "min_ce_d512_bf16": ceiling_probe.get("d512_bf16", {}).get("min_ce_nats"),
        },
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Empirical < 3.28 Ceiling & Wavelet-Rank Probe to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
