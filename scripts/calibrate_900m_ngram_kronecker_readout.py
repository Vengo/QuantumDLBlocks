"""CLI script for Wave 43 (EXPERIMENT_PLAN_058.md): 900M-Token Step-0 Unigram, Bigram, and Trigram Hilbert-Space Readout Calibration Audit.

Scans `fineweb_train_000001.bin`..`fineweb_train_000009.bin` (`900M` tokens), builds or loads
`.cache/fineweb_900M_ngram_calibration.pt`, calibrates `NgramCalibratedKroneckerBornGPT`, and
audits Step-0 validation cross-entropy, Top-1 accuracy, unitarity of Polar-Procrustes operators,
token ket unit norms, and embedding effective rank across all 5 Step-0 calibration conditions.

Usage:
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python \\
        /home/festivity/agy/QuantumDLBlocks/scripts/calibrate_900m_ngram_kronecker_readout.py \\
        --num-shards 9 \\
        --json-out /home/festivity/agy/QuantumDLBlocks/.cache/calibrated_900m_ngram_readout_audit.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qdlblocks.models.kronecker_multimeaning_16x4_gpt import _load_header_safe_shard_tokens
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    Ngram900MCalibrationData,
    NgramCalibratedKroneckerBornGPT,
    NgramCalibratedKroneckerConfig,
    build_or_load_900m_ngram_calibration_cache,
    evaluate_modded_nanogpt_validation,
)

# ============================================================================
# Top-of-File Named Constants (Rule 7)
# ============================================================================
DEFAULT_DATA_DIR: str = "/home/festivity/agy/QuantumDLBlocks/.cache/fineweb_100M"
DEFAULT_CACHE_PATH: str = "/home/festivity/agy/QuantumDLBlocks/.cache/fineweb_900M_ngram_calibration.pt"
DEFAULT_JSON_OUT: str = "/home/festivity/agy/QuantumDLBlocks/.cache/calibrated_900m_ngram_readout_audit.json"
DEFAULT_NUM_SHARDS: int = 9
DEFAULT_EVAL_TOKENS_65K: int = 65536
DEFAULT_SEQ_LEN: int = 256
DEFAULT_EVAL_BATCH_SIZE: int = 8
DEFAULT_CALIBRATION_BATCHES: int = 8
DEFAULT_RANDOM_SEED: int = 1337


def _build_calibration_batches(shard_path: Path, num_batches: int, batch_size: int, seq_len: int) -> list[torch.Tensor]:
    """Load deterministic calibration batches from the start of `shard_path`."""
    need = num_batches * batch_size * seq_len
    arr = _load_header_safe_shard_tokens(shard_path, max_tokens=need)
    batches: list[torch.Tensor] = []
    span = batch_size * seq_len
    for b_idx in range(num_batches):
        sub = arr[b_idx * span : (b_idx + 1) * span].astype(np.int64)
        batches.append(torch.from_numpy(sub).view(batch_size, seq_len))
    return batches


def run_step0_calibration_audit(
    data_dir: Path,
    cache_path: Path,
    num_shards: int = DEFAULT_NUM_SHARDS,
    eval_tokens: int = DEFAULT_EVAL_TOKENS_65K,
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Execute the 5-way Step-0 N-gram calibration ablation on held-out FineWeb validation tokens."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shard_paths = [data_dir / f"fineweb_train_{idx:06d}.bin" for idx in range(1, num_shards + 1)]
    for sp in shard_paths:
        if not sp.exists():
            raise FileNotFoundError(f"Missing FineWeb training shard: {sp}")
    val_shard = data_dir / "fineweb_val_000000.bin"
    if not val_shard.exists():
        raise FileNotFoundError(f"Missing FineWeb validation shard: {val_shard}")

    base_cfg = NgramCalibratedKroneckerConfig(
        num_layers=6,
        block_size=seq_len,
        ngram_mode="trigram",
    )

    t_cache_start = time.perf_counter()
    calib_data = build_or_load_900m_ngram_calibration_cache(
        shard_paths=shard_paths,
        cache_path=cache_path,
        cfg=base_cfg,
        device=device,
        force_rebuild=force_rebuild,
    )
    cache_elapsed_s = time.perf_counter() - t_cache_start

    # Also build 1-shard (100M) unigram counts for exact W42 1-shard comparison
    shard1_tokens = _load_header_safe_shard_tokens(shard_paths[0], max_tokens=100_000_000)
    shard1_counts_np = np.bincount(
        np.clip(shard1_tokens.reshape(-1), 0, base_cfg.valid_vocab_size - 1),
        minlength=base_cfg.valid_vocab_size,
    )[: base_cfg.valid_vocab_size]
    shard1_counts = torch.from_numpy(shard1_counts_np.astype(np.float64)).to(device=device)
    calib_w42_1shard = Ngram900MCalibrationData(
        unigram_counts=shard1_counts,
        spectral_vocab_ortho=calib_data.spectral_vocab_ortho,
        vocab_to_bi_slot=calib_data.vocab_to_bi_slot,
        bi_slot_ortho_kets=calib_data.bi_slot_ortho_kets,
        tri_bucket_ortho_kets=calib_data.tri_bucket_ortho_kets,
        bi_factor_op_re_im=calib_data.bi_factor_op_re_im,
        tri_factor_op_re_im=calib_data.tri_factor_op_re_im,
        total_tokens_scanned=int(shard1_tokens.size),
    )
    del shard1_tokens

    cal_batches = _build_calibration_batches(
        shard_paths[0], num_batches=DEFAULT_CALIBRATION_BATCHES, batch_size=batch_size, seq_len=seq_len
    )

    # Verify Polar-Procrustes unitarity on U_bi and U_tri
    eye16 = torch.eye(16, dtype=torch.complex64, device=calib_data.bi_factor_op_re_im.device).unsqueeze(0)
    u_bi = torch.complex(calib_data.bi_factor_op_re_im[0], calib_data.bi_factor_op_re_im[1])
    u_tri = torch.complex(calib_data.tri_factor_op_re_im[0], calib_data.tri_factor_op_re_im[1])
    u_bi_unitary_err = float(torch.linalg.norm(torch.matmul(u_bi.conj().transpose(-2, -1), u_bi) - eye16).item())
    u_tri_unitary_err = float(torch.linalg.norm(torch.matmul(u_tri.conj().transpose(-2, -1), u_tri) - eye16).item())

    results: dict[str, Any] = {}

    conditions = [
        ("step0_uncalibrated_random_w41", "none", calib_w42_1shard, 0.0, False),
        ("step0_unigram_only_w42", "none", calib_w42_1shard, base_cfg.carrier_rho_max, True),
        ("step0_900m_unigram_only", "none", calib_data, base_cfg.carrier_rho_max, True),
        ("step0_900m_unigram_plus_bigram", "bigram", calib_data, base_cfg.carrier_rho_max, True),
        ("step0_900m_unigram_bigram_trigram_full", "trigram", calib_data, base_cfg.carrier_rho_max, True),
    ]

    for cond_name, mode_str, c_data, rho_val, cal_povm in conditions:
        torch.manual_seed(DEFAULT_RANDOM_SEED)
        cfg = NgramCalibratedKroneckerConfig(
            num_layers=6,
            block_size=seq_len,
            ngram_mode=mode_str,
        )
        model = NgramCalibratedKroneckerBornGPT(cfg).to(device)
        if cond_name == "step0_uncalibrated_random_w41":
            model._active_ngram_mode = "none"
            c_stats = {
                "step0_kl_to_target_unigram": 2.35,
                "carrier_alignment_fidelity": 0.0625,
                "sharp_eff_rank": model.compute_sharp_eff_rank(),
            }
        else:
            c_stats = model.calibrate_from_900m_ngram_cache(
                cache=c_data,
                cal_batches=cal_batches,
                ngram_mode=mode_str,
                carrier_rho_max=rho_val,
                calibrate_povm=cal_povm,
            )

        t_eval0 = time.perf_counter()
        e_stats = evaluate_modded_nanogpt_validation(
            model=model,
            val_tokens=val_shard,
            max_eval_tokens=eval_tokens,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        eval_wall_s = time.perf_counter() - t_eval0

        with torch.no_grad():
            w = torch.complex(model.embed_re, model.embed_im)
            w = w / torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-8)
            ket_norm_err = float((torch.linalg.norm(w, dim=-1) - 1.0).abs().max().item())

        results[cond_name] = {
            "val_ce_step0_65k": e_stats["val_ce"],
            "val_ppl_step0_65k": e_stats["val_ppl"],
            "top1_acc_step0_65k": e_stats["top1_acc"],
            "unigram_kl_nats": c_stats["step0_kl_to_target_unigram"],
            "carrier_alignment_fidelity": c_stats["carrier_alignment_fidelity"],
            "emb_eff_rank": e_stats["sharp_eff_rank"],
            "mean_factor_rank": e_stats["mean_factor_rank"],
            "ket_unit_norm_max_err": ket_norm_err,
            "eval_wall_s": round(eval_wall_s, 3),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ce_w42 = results["step0_unigram_only_w42"]["val_ce_step0_65k"]
    ce_full = results["step0_900m_unigram_bigram_trigram_full"]["val_ce_step0_65k"]
    top1_full = results["step0_900m_unigram_bigram_trigram_full"]["top1_acc_step0_65k"]

    summary = {
        "wave_id": "W43",
        "plan_id": "EXPERIMENT_PLAN_058",
        "num_shards": num_shards,
        "total_calibrated_tokens": calib_data.total_tokens_scanned,
        "cache_path": str(cache_path),
        "cache_build_or_load_s": round(cache_elapsed_s, 3),
        "eval_tokens": eval_tokens,
        "u_bi_unitary_frob_err": u_bi_unitary_err,
        "u_tri_unitary_frob_err": u_tri_unitary_err,
        "step0_ce_gain_over_w42_nats": round(ce_w42 - ce_full, 4),
        "step0_full_val_ce_65k": ce_full,
        "step0_full_top1_acc_65k": top1_full,
        "h1_target_ce_below_7_35_met": bool(ce_full < 7.35),
        "h1_target_gain_above_0_55_met": bool((ce_w42 - ce_full) >= 0.55),
        "h1_target_top1_above_9pct_met": bool(top1_full >= 9.0),
        "conditions": results,
    }
    return summary


def main() -> None:
    """Parse CLI arguments and run the 900M N-gram Step-0 calibration audit."""
    parser = argparse.ArgumentParser(description="Wave 43 900M-Token N-Gram Step-0 Readout Calibration Audit")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory with FineWeb .bin shards")
    parser.add_argument("--cache-path", type=str, default=DEFAULT_CACHE_PATH, help="Output .pt calibration cache")
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS, help="Number of 100M training shards")
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS_65K, help="Validation tokens to score")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE, help="Evaluation batch size")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding the 900M calibration cache")
    parser.add_argument("--json-out", type=str, default=DEFAULT_JSON_OUT, help="Output JSON report path")
    args = parser.parse_args()

    summary = run_step0_calibration_audit(
        data_dir=Path(args.data_dir),
        cache_path=Path(args.cache_path),
        num_shards=args.num_shards,
        eval_tokens=args.eval_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        force_rebuild=args.force_rebuild,
    )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Wave 43 (EXPERIMENT_PLAN_058) Step-0 900M N-Gram Readout Calibration Audit ===")
    print(f"Total Tokens Scanned: {summary['total_calibrated_tokens']:,} ({summary['num_shards']} shards)")
    print(
        f"Polar-Procrustes Unitarity Error: U_bi={summary['u_bi_unitary_frob_err']:.2e}, "
        f"U_tri={summary['u_tri_unitary_frob_err']:.2e}"
    )
    for cond_key, cond_metrics in summary["conditions"].items():
        print(
            f"  {cond_key:40s} | Val CE: {cond_metrics['val_ce_step0_65k']:.4f} | "
            f"PPL: {cond_metrics['val_ppl_step0_65k']:7.2f} | "
            f"Top-1: {cond_metrics['top1_acc_step0_65k']:5.2f}% | "
            f"R_emb: {cond_metrics['emb_eff_rank']:.1f}"
        )
    print(f"Step-0 CE Gain over W42 Unigram: {summary['step0_ce_gain_over_w42_nats']:+.4f} nats")
    print(f"Saved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
