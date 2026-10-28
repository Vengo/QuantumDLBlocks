#!/usr/bin/env python3
"""16x16x16x16 = 65,536 Kronecker SNR, Multi-Meaning Polysemy, & 100M Unigram Prior Profiler (`Wave 41`).

When to Use This CLI Tool (`EXPERIMENT_PLAN_056.md` Section 5.2):
- Run this diagnostic script to verify the four foundational pillars of `Wave 41`:
  1. Empirical random background overlap `E_{u != v}[|K_{u,v,r}|^2]` in `(C^16)^{x4} ~= C^{65,536}`
     (`~1/65,536 = 1.5259e-5`) vs. flat `C^128` (`1/128 = 7.8125e-3`, `~512x` SNR improvement).
  2. Polysemous `K`-sense target coincidence (`K in {1, 2, 4}` orthogonal senses per word) across
     meaning ranks `R in {1, 2, 4}`.
  3. `100M`-token FineWeb unigram distribution vs. `2M`-slice and uniform priors (`unseen_tokens`,
     `kl_divergence_to_val_nats`, and Step-0 validation Born NLL).
  4. Throughput (`tok/s`) and peak VRAM (`MiB`) of the single-`torch.bmm` chunked Kronecker Born
     readout and full `KroneckerMultiMeaning16x4GPT` forward+backward steps across `R in {1, 2, 4}`.

Example Invocation:
  /home/festivity/agy/QuantumDLBlocks/.venv/bin/python \\
      /home/festivity/agy/QuantumDLBlocks/scripts/kronecker_16x4_meaning_and_snr_profiler.py \\
      --ranks 1,2,4 --json-out .cache/kronecker_16x4_meaning_and_snr_profile.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(ROOT / "src"), str(ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize  # noqa: E402
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (  # noqa: E402
    DEFAULT_FACTOR_DIM,
    DEFAULT_KRON_BLEND,
    DEFAULT_KRON_BOSON_P,
    DEFAULT_KRON_COINC_GAMMA,
    DEFAULT_KRON_HOLO_ALPHA,
    DEFAULT_NUM_FACTORS,
    DEFAULT_UNIGRAM_FLOOR_COUNT,
    DEFAULT_UNIGRAM_TEMP,
    KroneckerMultiMeaning16x4GPT,
    KroneckerMultiMeaningBornChunkedNLLFn,
    KroneckerMultiMeaningConfig,
    _algebraic_positive_unit_l2,
    _load_header_safe_shard_tokens,
    build_kronecker_multimeaning_optimizers,
    compute_kronecker_manifold_and_isotropy_penalties,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_056.md (Wave 41)
# ==============================================================================
PADDED_VOCAB_SIZE: int = 50304
VALID_VOCAB_SIZE: int = 50257
DEFAULT_SNR_SAMPLE_STATES: int = 1024
DEFAULT_FLAT_COMPLEX_DIM: int = 128
DEFAULT_BENCH_BATCH_SIZE: int = 8
DEFAULT_BENCH_SEQ_LEN: int = 256
DEFAULT_BENCH_LAYERS: int = 4
DEFAULT_BENCH_STEPS: int = 8
SLICE_2M_TOKENS: int = 2_000_000
SLICE_100M_TOKENS: int = 100_000_000
VAL_AUDIT_TOKENS: int = 1_048_576


@dataclass(frozen=True)
class KroneckerProfilerConfig:
    """Configuration for `kronecker_16x4_meaning_and_snr_profiler.py` (`<= 8` fields)."""

    ranks: tuple[int, ...] = (1, 2, 4)
    snr_states: int = DEFAULT_SNR_SAMPLE_STATES
    batch_size: int = DEFAULT_BENCH_BATCH_SIZE
    seq_len: int = DEFAULT_BENCH_SEQ_LEN
    num_layers: int = DEFAULT_BENCH_LAYERS
    bench_steps: int = DEFAULT_BENCH_STEPS
    json_out: str = str(ROOT / ".cache" / "kronecker_16x4_meaning_and_snr_profile.json")


def profile_kronecker_16x4_snr_vs_flat(
    n_states: int,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark random impostor overlap in `(C^16)^{x4} ~= C^{65,536}` vs. flat `C^128` (`<= 8` args)."""
    torch.manual_seed(4101)
    m_num = DEFAULT_NUM_FACTORS
    d_f = DEFAULT_FACTOR_DIM
    d_flat = DEFAULT_FLAT_COMPLEX_DIM

    flat_re, flat_im = complex_l2_normalize(
        torch.randn(n_states, d_flat, device=device, dtype=torch.float32),
        torch.randn(n_states, d_flat, device=device, dtype=torch.float32),
        dim=-1,
    )
    flat_cat = torch.cat([flat_re.float(), flat_im.float()], dim=-1)
    flat_skew = torch.cat([-flat_im.float(), flat_re.float()], dim=-1)
    flat_sq = (flat_cat @ flat_cat.t()).square() + (flat_cat @ flat_skew.t()).square()
    off_mask = ~torch.eye(n_states, device=device, dtype=torch.bool)
    flat_mean_sq = float(flat_sq[off_mask].mean().item())
    flat_p99_sq = float(torch.quantile(flat_sq[off_mask], 0.99).item())

    kron_re, kron_im = complex_l2_normalize(
        torch.randn(n_states, m_num, d_f, device=device, dtype=torch.float32),
        torch.randn(n_states, m_num, d_f, device=device, dtype=torch.float32),
        dim=-1,
    )
    level_overlaps: dict[str, float] = {}
    running_prod = torch.ones((n_states, n_states), device=device, dtype=torch.float32)
    for m_idx in range(m_num):
        km_cat = torch.cat([kron_re[:, m_idx].float(), kron_im[:, m_idx].float()], dim=-1)
        km_skew = torch.cat([-kron_im[:, m_idx].float(), kron_re[:, m_idx].float()], dim=-1)
        mag_sq_m = (km_cat @ km_cat.t()).square() + (km_cat @ km_skew.t()).square()
        running_prod = running_prod * mag_sq_m
        dim_level = d_f ** (m_idx + 1)
        level_overlaps[f"level_{m_idx + 1}_dim_{dim_level}"] = float(running_prod[off_mask].mean().item())

    kron_mean_sq = float(running_prod[off_mask].mean().item())
    kron_p99_sq = float(torch.quantile(running_prod[off_mask], 0.99).item())
    snr_gain_x = flat_mean_sq / max(1e-12, kron_mean_sq)

    return {
        "sample_states": n_states,
        "flat_c128_theoretical_bg_sq": 1.0 / float(d_flat),
        "flat_c128_empirical_mean_bg_sq": kron_round(flat_mean_sq, 8),
        "flat_c128_empirical_p99_bg_sq": kron_round(flat_p99_sq, 8),
        "kronecker_16x4_theoretical_bg_sq": 1.0 / float(d_f ** m_num),
        "kronecker_16x4_empirical_mean_bg_sq": kron_round(kron_mean_sq, 8),
        "kronecker_16x4_empirical_p99_bg_sq": kron_round(kron_p99_sq, 8),
        "hierarchical_level_bg_sq": level_overlaps,
        "single_branch_snr_improvement_x": round(snr_gain_x, 2),
    }


def kron_round(val: float, digits: int = 8) -> float:
    """Round float `val` to `digits` decimal places."""
    return round(float(val), digits)


def profile_polysemy_sense_capacity(
    ranks: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    """Profile 4-factor coincidence across `K in {1, 2, 4}` orthogonal word senses for `R in ranks` (`<= 8` args)."""
    torch.manual_seed(4102)
    m_num = DEFAULT_NUM_FACTORS
    d_f = DEFAULT_FACTOR_DIM

    # Construct 4 mutually orthogonal factor kets per factor m in {0, 1, 2, 3} via Haar unitary columns
    z_re = torch.randn((m_num, d_f, d_f), device=device, dtype=torch.float32)
    z_im = torch.randn((m_num, d_f, d_f), device=device, dtype=torch.float32)
    q_c, _ = torch.linalg.qr(torch.complex(z_re, z_im))
    senses_re = q_c.real[:, :4, :].permute(1, 0, 2).contiguous()  # (4_senses, 4_factors, 16)
    senses_im = q_c.imag[:, :4, :].permute(1, 0, 2).contiguous()

    report: dict[str, Any] = {}
    for k_senses in (1, 2, 4):
        k_entry: dict[str, Any] = {}
        for r_rank in ranks:
            # A Rank-R word distributes its R branches across the k_senses orthogonal target senses
            branches_used = min(r_rank, k_senses)
            senses_per_branch = float(k_senses) / float(branches_used)
            # Each branch is an equal superposition of `senses_per_branch` orthogonal senses in each factor
            per_factor_sq_overlap = 1.0 / senses_per_branch
            four_factor_kron_sq = per_factor_sq_overlap ** m_num
            branch_weight_sq = 1.0 / float(branches_used)

            # Direct tensor verification for k_senses = 2
            if k_senses == 2:
                if r_rank == 1:
                    b0_re, b0_im = complex_l2_normalize(senses_re[0] + senses_re[1], senses_im[0] + senses_im[1], dim=-1)
                    c0 = ((b0_re * senses_re[0] + b0_im * senses_im[0]).sum(dim=-1).square()).prod().item()
                    c1 = ((b0_re * senses_re[1] + b0_im * senses_im[1]).sum(dim=-1).square()).prod().item()
                    unweighted_kron_coinc = float(min(c0, c1))
                    weighted_coinc = unweighted_kron_coinc
                else:
                    c0 = ((senses_re[0] * senses_re[0] + senses_im[0] * senses_im[0]).sum(dim=-1).square()).prod().item()
                    c1 = ((senses_re[1] * senses_re[1] + senses_im[1] * senses_im[1]).sum(dim=-1).square()).prod().item()
                    unweighted_kron_coinc = float(min(c0, c1))
                    weighted_coinc = 0.5 * unweighted_kron_coinc
            else:
                unweighted_kron_coinc = four_factor_kron_sq
                weighted_coinc = branch_weight_sq * four_factor_kron_sq

            k_entry[f"rank_{r_rank}"] = {
                "num_meanings_R": r_rank,
                "real_params_per_token": r_rank * m_num * (2 * d_f) + (r_rank if r_rank > 1 else 0),
                "branch_matched_4factor_kron_sq": round(unweighted_kron_coinc, 6),
                "alpha_weighted_coincidence_energy": round(weighted_coinc, 6),
            }
        if "rank_1" in k_entry and "rank_2" in k_entry and k_senses == 2:
            r1_val = k_entry["rank_1"]["alpha_weighted_coincidence_energy"]
            r2_val = k_entry["rank_2"]["alpha_weighted_coincidence_energy"]
            k_entry["rank2_vs_rank1_coincidence_gain_x"] = round(r2_val / max(1e-9, r1_val), 2)
        report[f"k_senses_{k_senses}"] = k_entry
    return report


def profile_unigram_100m_vs_2m_and_uniform(
    data_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Audit `100M`-token FineWeb unigram prior against `2M`-slice and uniform priors (`<= 8` args)."""
    shard1 = data_dir / "train_full_100M.bin"
    if not shard1.exists():
        shard1 = data_dir / "fineweb_train_000001.bin"
    val_path = data_dir / "val_slice.bin"

    tokens_100m = _load_header_safe_shard_tokens(shard1, max_tokens=SLICE_100M_TOKENS)
    val_tokens = _load_header_safe_shard_tokens(val_path, max_tokens=VAL_AUDIT_TOKENS)
    val_t = torch.from_numpy(val_tokens.astype(np.int64)).clamp(0, VALID_VOCAB_SIZE - 1)
    val_counts = torch.bincount(val_t, minlength=VALID_VOCAB_SIZE)[:VALID_VOCAB_SIZE].to(torch.float64)
    p_val = val_counts / val_counts.sum().clamp_min(1.0)
    nz_val = p_val > 0.0

    # Build 4 small evaluation batches on device for Step-0 Born NLL measurement
    val_seq = val_t[: 4 * (DEFAULT_BENCH_SEQ_LEN + 1)].reshape(4, DEFAULT_BENCH_SEQ_LEN + 1).to(device)
    x_val = val_seq[:, :-1].contiguous()
    y_val = val_seq[:, 1:].contiguous()

    torch.manual_seed(4103)
    base_cfg = KroneckerMultiMeaningConfig(
        vocab_size=PADDED_VOCAB_SIZE,
        valid_vocab_size=VALID_VOCAB_SIZE,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=2,
        block_size=DEFAULT_BENCH_SEQ_LEN,
        unigram_temp=DEFAULT_UNIGRAM_TEMP,
    )
    model = KroneckerMultiMeaning16x4GPT(base_cfg).to(device)
    model.eval()

    priors_report: dict[str, Any] = {}
    for key, tok_slice, temp in (
        ("uniform_prior", tokens_100m[:SLICE_2M_TOKENS], 0.0),
        ("slice_2m_prior", tokens_100m[:SLICE_2M_TOKENS], DEFAULT_UNIGRAM_TEMP),
        ("slice_100m_prior", tokens_100m, DEFAULT_UNIGRAM_TEMP),
    ):
        cfg_k = KroneckerMultiMeaningConfig(
            vocab_size=PADDED_VOCAB_SIZE,
            valid_vocab_size=VALID_VOCAB_SIZE,
            num_meanings=2,
            num_factors=4,
            factor_dim=16,
            num_layers=2,
            block_size=DEFAULT_BENCH_SEQ_LEN,
            unigram_temp=temp,
        )
        model.cfg = cfg_k
        stats = model.initialize_unigram_prior_from_tokens(
            tok_slice, floor_count=DEFAULT_UNIGRAM_FLOOR_COUNT
        )
        if key == "uniform_prior":
            p_uni_d = torch.full((VALID_VOCAB_SIZE,), 1.0 / float(VALID_VOCAB_SIZE), dtype=torch.float64)
        else:
            p_uni_d = model.unigram_probs[:VALID_VOCAB_SIZE].detach().cpu().to(torch.float64)

        kl_div = float((p_val[nz_val] * (torch.log(p_val[nz_val]) - torch.log(p_uni_d[nz_val]))).sum().item())
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            _, step0_nll = model(x_val, targets=y_val, return_logits=False)
        assert step0_nll is not None

        priors_report[key] = {
            "tokens_counted": int(tok_slice.shape[0]),
            "unigram_temp": temp,
            "observed_tokens": int(stats["observed_tokens"]),
            "unseen_tokens": int(stats["unseen_tokens"]),
            "unigram_entropy_nats": stats["unigram_entropy_nats"],
            "kl_div_to_val_nats": round(kl_div, 4),
            "step0_val_born_nll": round(float(step0_nll.item()), 4),
        }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    unseen_2m = priors_report["slice_2m_prior"]["unseen_tokens"]
    unseen_100m = priors_report["slice_100m_prior"]["unseen_tokens"]
    priors_report["unseen_token_reduction_x"] = round(float(unseen_2m) / float(max(1, unseen_100m)), 2)
    return priors_report


def benchmark_ranks_throughput_and_vram(
    cfg: KroneckerProfilerConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark readout kernel and full model throughput (`tok/s`) and peak VRAM (`MiB`) across `R in cfg.ranks`."""
    n_tok = cfg.batch_size * cfg.seq_len
    m_num = DEFAULT_NUM_FACTORS
    d_f = DEFAULT_FACTOR_DIM
    d_2 = 2 * d_f
    v_eff = VALID_VOCAB_SIZE

    rank_benchmarks: dict[str, Any] = {}
    for r_num in cfg.ranks:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # 1. Standalone chunked Kronecker Born readout benchmark (fwd + bwd)
        phi_f = torch.randn(n_tok, r_num, m_num, d_2, device=device, dtype=torch.bfloat16, requires_grad=True)
        psi_f = torch.randn(v_eff, r_num, m_num, d_2, device=device, dtype=torch.bfloat16, requires_grad=True)
        beta_f = _algebraic_positive_unit_l2(torch.zeros(n_tok, r_num, device=device)).requires_grad_(True)
        alpha_f = _algebraic_positive_unit_l2(torch.zeros(v_eff, r_num, device=device)).requires_grad_(True)
        prior_w = torch.ones(v_eff, device=device, dtype=torch.float32)
        targets = torch.randint(0, v_eff, (n_tok,), device=device, dtype=torch.int64)
        hp = torch.tensor(
            [DEFAULT_KRON_BOSON_P, DEFAULT_KRON_HOLO_ALPHA, DEFAULT_KRON_BLEND, DEFAULT_KRON_COINC_GAMMA],
            device=device,
            dtype=torch.float32,
        )

        # Warmup
        for _ in range(2):
            loss_r = KroneckerMultiMeaningBornChunkedNLLFn.apply(
                phi_f, psi_f, beta_f, alpha_f, targets, prior_w, hp, None
            )
            loss_r.backward()
            phi_f.grad = None
            psi_f.grad = None
            beta_f.grad = None
            alpha_f.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0_ro = time.perf_counter()
        for _ in range(cfg.bench_steps):
            loss_r = KroneckerMultiMeaningBornChunkedNLLFn.apply(
                phi_f, psi_f, beta_f, alpha_f, targets, prior_w, hp, None
            )
            loss_r.backward()
            phi_f.grad = None
            psi_f.grad = None
            beta_f.grad = None
            alpha_f.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ro_wall = time.perf_counter() - t0_ro
        readout_tok_s = (cfg.bench_steps * n_tok) / max(1e-6, ro_wall)
        del phi_f, psi_f, beta_f, alpha_f, prior_w, targets, hp

        # 2. Full KroneckerMultiMeaning16x4GPT training step benchmark
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        m_cfg = KroneckerMultiMeaningConfig(
            vocab_size=PADDED_VOCAB_SIZE,
            valid_vocab_size=VALID_VOCAB_SIZE,
            num_meanings=r_num,
            num_factors=m_num,
            factor_dim=d_f,
            num_layers=cfg.num_layers,
            block_size=cfg.seq_len,
        )
        model = KroneckerMultiMeaning16x4GPT(m_cfg).to(device)
        opt_muon, opt_adam = build_kronecker_multimeaning_optimizers(model)
        params_m = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)

        x_b = torch.randint(0, v_eff, (cfg.batch_size, cfg.seq_len), device=device, dtype=torch.int64)
        y_b = torch.randint(0, v_eff, (cfg.batch_size, cfg.seq_len), device=device, dtype=torch.int64)

        # Warmup
        for _ in range(2):
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, nll = model(x_b, targets=y_b, return_logits=False)
            assert nll is not None
            pen, _ = compute_kronecker_manifold_and_isotropy_penalties(model)
            (nll + pen).backward()
            opt_muon.step()
            opt_adam.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        t0_full = time.perf_counter()
        for _ in range(cfg.bench_steps):
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, nll = model(x_b, targets=y_b, return_logits=False)
            assert nll is not None
            pen, _ = compute_kronecker_manifold_and_isotropy_penalties(model)
            (nll + pen).backward()
            opt_muon.step()
            opt_adam.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        else:
            peak_mib = 0.0
        full_wall = time.perf_counter() - t0_full
        full_tok_s = (cfg.bench_steps * n_tok) / max(1e-6, full_wall)

        rank_benchmarks[f"rank_{r_num}"] = {
            "num_meanings_R": r_num,
            "d_complex_total": m_cfg.d_complex_total,
            "n_registers": m_cfg.n_registers,
            "params_M": params_m,
            "readout_only_tok_s": round(readout_tok_s, 1),
            "full_step_tok_s": round(full_tok_s, 1),
            "peak_vram_mib": round(peak_mib, 1),
            "sharp_eff_rank": round(model.compute_sharp_eff_rank(), 2),
            "mean_factor_rank": round(model.compute_mean_factor_rank(), 2),
            "cross_branch_overlap": round(model.compute_cross_branch_overlap(), 6),
        }
        del model, opt_muon, opt_adam
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return rank_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wave 41 (EXPERIMENT_PLAN_056) 16^4 Kronecker SNR, Multi-Meaning, & 100M Unigram Profiler"
    )
    parser.add_argument("--ranks", type=str, default="1,2,4")
    parser.add_argument("--snr-states", type=int, default=DEFAULT_SNR_SAMPLE_STATES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BENCH_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_BENCH_SEQ_LEN)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_BENCH_LAYERS)
    parser.add_argument("--bench-steps", type=int, default=DEFAULT_BENCH_STEPS)
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(ROOT / ".cache" / "kronecker_16x4_meaning_and_snr_profile.json"),
    )
    args = parser.parse_args()

    ranks_tuple = tuple(int(x.strip()) for x in str(args.ranks).split(",") if x.strip())
    cfg = KroneckerProfilerConfig(
        ranks=ranks_tuple,
        snr_states=int(args.snr_states),
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        num_layers=int(args.num_layers),
        bench_steps=int(args.bench_steps),
        json_out=str(args.json_out),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / ".cache" / "fineweb_100M"

    print(f"[Wave 41 Profiler] 1/4 Auditing 16^4 = 65,536 Kronecker SNR vs flat C^128 on {device}...", flush=True)
    snr_report = profile_kronecker_16x4_snr_vs_flat(cfg.snr_states, device)

    print(f"[Wave 41 Profiler] 2/4 Auditing polysemy capacity across R={cfg.ranks} on {device}...", flush=True)
    polysemy_report = profile_polysemy_sense_capacity(cfg.ranks, device)

    print(f"[Wave 41 Profiler] 3/4 Auditing 100M-token FineWeb unigram prior in {data_dir}...", flush=True)
    unigram_report = profile_unigram_100m_vs_2m_and_uniform(data_dir, device)

    print(f"[Wave 41 Profiler] 4/4 Benchmarking throughput & VRAM across R={cfg.ranks} on {device}...", flush=True)
    throughput_report = benchmark_ranks_throughput_and_vram(cfg, device)

    payload = {
        "wave_id": "W41",
        "plan_id": "EXPERIMENT_PLAN_056",
        "kronecker_16x4_snr_audit": snr_report,
        "polysemy_sense_capacity": polysemy_report,
        "unigram_100m_prior_audit": unigram_report,
        "rank_throughput_and_vram": throughput_report,
    }
    out_path = Path(cfg.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[Wave 41 Profiler] Saved profile report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
