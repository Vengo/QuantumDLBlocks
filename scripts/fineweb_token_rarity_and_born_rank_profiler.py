#!/usr/bin/env python3
"""FineWeb Zipf Token Rarity & Complex Born Rank-Preservation Profiler (`Wave 40` / `EXPERIMENT_PLAN_055.md`).

When to Use This CLI Tool:
- Run this script before launching long-horizon `FineWeb` training runs (`100M-900M` tokens) to:
  1. Audit Zipf token frequency statistics across slice sizes (`2M`, `8M`, `30M`, `100M`, `300M`) including
     unseen token counts, `< 10` count starvation, top-100 token dominance, and bottom-50% gradient share.
  2. Profile `RareTokenImportanceWindowSampler` exposure gains across vocabulary frequency deciles and
     verify exact inverse-probability importance-weight unbiasedness (`< 1e-6` analytical error).
  3. Run a head-to-head N-step (`--steps 100`) Hilbert-space rank-preservation benchmark on `FineWeb`
     comparing `legacy_boltzmann_real` (`GeminiPackedComplexRoPEGPT` with `meas_proj` + hard retractions)
     against `tied_holo_born` + `soft_penalties_parseval_iso` (`TiedHoloBornComplexRoPEGPT`).

Example Invocation:
  /home/festivity/agy/QuantumDLBlocks/.venv/bin/python \\
      /home/festivity/agy/QuantumDLBlocks/scripts/fineweb_token_rarity_and_born_rank_profiler.py \\
      --steps 100 --json-out .cache/fineweb_token_rarity_and_born_rank_profile.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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

from qdlblocks.models.gemini_packed_complex_rope_gpt import (  # noqa: E402
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
    build_gemini_complex_optimizers,
    initialize_gemini_complex_gpt_with_ppmi_svd,
)
from qdlblocks.models.tied_holo_born_complex_gpt import (  # noqa: E402
    DEFAULT_RARE_ALPHA,
    RareTokenImportanceWindowSampler,
    TiedHoloBornComplexRoPEGPT,
    TiedHoloBornConfig,
    build_tied_holo_born_optimizers,
    compute_soft_manifold_and_isotropy_penalties,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_055.md (Wave 40)
# ==============================================================================
VALID_VOCAB_SIZE: int = 50257
PADDED_VOCAB_SIZE: int = 50304
DEFAULT_PROFILE_STEPS: int = 100
DEFAULT_D_COMPLEX: int = 128
DEFAULT_N_REGISTERS: int = 4
DEFAULT_D_REGISTER: int = 32
DEFAULT_NUM_LAYERS: int = 4
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_SEQ_LEN: int = 256
HEADER_INTS: int = 256
SLICE_SIZES_M: tuple[int, ...] = (2, 8, 30, 100)


@dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for `fineweb_token_rarity_and_born_rank_profiler.py` (`<= 8` fields)."""

    steps: int = DEFAULT_PROFILE_STEPS
    d_complex: int = DEFAULT_D_COMPLEX
    num_layers: int = DEFAULT_NUM_LAYERS
    batch_size: int = DEFAULT_BATCH_SIZE
    seq_len: int = DEFAULT_SEQ_LEN
    alpha_rare: float = DEFAULT_RARE_ALPHA
    json_out: str = str(ROOT / ".cache" / "fineweb_token_rarity_and_born_rank_profile.json")
    plot_out: str = str(ROOT / "fineweb_token_distribution.png")


def _load_shard_tokens(shard_path: Path, max_tokens: int) -> np.ndarray:
    """Load up to `max_tokens` uint16 token IDs from a FineWeb `.bin` shard."""
    raw = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4)
    n_take = min(int(max_tokens), int(raw.shape[0]))
    return np.array(raw[:n_take], dtype=np.uint16, copy=True)


def _resolve_shard1_path(data_dir: Path) -> Path:
    """Resolve Shard 1 from `fineweb_train_000001.bin` or `train_full_100M.bin`."""
    for name in ("fineweb_train_000001.bin", "train_full_100M.bin"):
        cand = data_dir / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Missing FineWeb Shard 1 in {data_dir}")


def _compute_zipf_slice_entry(counts: np.ndarray, n_tok: int) -> dict[str, Any]:
    """Compute Zipf summary statistics from a `VALID_VOCAB_SIZE` unigram count array."""
    sorted_c = np.sort(counts)[::-1]
    total = float(max(1, counts.sum()))
    unseen = int((counts == 0).sum())
    lt_10 = int((counts < 10).sum())
    lt_100 = int((counts < 100).sum())
    top100_share = float(sorted_c[:100].sum() / total) * 100.0
    top1000_share = float(sorted_c[:1000].sum() / total) * 100.0
    bottom50_share = float(sorted_c[25000:].sum() / total) * 100.0

    probs = counts[counts > 0].astype(np.float64) / total
    entropy_nats = float(-(probs * np.log(probs)).sum())
    eff_perplexity = float(np.exp(entropy_nats))
    return {
        "tokens": int(n_tok),
        "unseen_tokens": unseen,
        "unseen_pct": round(100.0 * unseen / VALID_VOCAB_SIZE, 2),
        "lt_10_tokens": lt_10,
        "lt_10_pct": round(100.0 * lt_10 / VALID_VOCAB_SIZE, 2),
        "lt_100_tokens": lt_100,
        "lt_100_pct": round(100.0 * lt_100 / VALID_VOCAB_SIZE, 2),
        "top_100_token_share_pct": round(top100_share, 2),
        "top_1000_token_share_pct": round(top1000_share, 2),
        "bottom_25257_token_share_pct": round(bottom50_share, 2),
        "unigram_entropy_nats": round(entropy_nats, 4),
        "effective_vocab_perplexity": round(eff_perplexity, 1),
        "median_token_count": int(np.median(counts)),
    }


def profile_fineweb_zipf_slices(data_dir: Path) -> dict[str, Any]:
    """Compute Zipf token frequency and starvation metrics across `2M`, `8M`, `30M`, `100M`, and `300M` slices."""
    shard1 = _resolve_shard1_path(data_dir)
    tokens_100m = _load_shard_tokens(shard1, 100_000_000)
    slice_reports: dict[str, Any] = {}
    counts_100m = np.zeros(VALID_VOCAB_SIZE, dtype=np.int64)

    for m_tok in SLICE_SIZES_M:
        n_tok = min(int(tokens_100m.shape[0]), m_tok * 1_000_000)
        sub = tokens_100m[:n_tok]
        counts = np.bincount(sub, minlength=VALID_VOCAB_SIZE)[:VALID_VOCAB_SIZE].astype(np.int64)
        if m_tok == 100:
            counts_100m = counts
        slice_reports[f"{m_tok}M"] = _compute_zipf_slice_entry(counts, n_tok)

    # Accumulate across all available shards up to 300M tokens
    counts_multi = counts_100m.copy()
    total_multi_tok = int(tokens_100m.shape[0])
    for idx in (2, 3):
        sh_p = data_dir / f"fineweb_train_{idx:06d}.bin"
        if sh_p.exists():
            arr = _load_shard_tokens(sh_p, 100_000_000)
            c_sh = np.bincount(arr, minlength=VALID_VOCAB_SIZE)[:VALID_VOCAB_SIZE].astype(np.int64)
            counts_multi += c_sh
            total_multi_tok += int(arr.shape[0])
    if total_multi_tok > 100_000_000:
        slice_reports[f"{total_multi_tok // 1_000_000}M"] = _compute_zipf_slice_entry(counts_multi, total_multi_tok)

    return {
        "valid_vocab_size": VALID_VOCAB_SIZE,
        "slices": slice_reports,
        "tokens_30m": tokens_100m[:30_000_000],
    }


def profile_rare_token_sampler(
    tokens_30m: np.ndarray,
    cfg: ProfilerConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Audit `RareTokenImportanceWindowSampler` rare-token exposure gain and unbiased importance weighting."""
    ref_2m = tokens_30m[:2_000_000]
    sampler_is = RareTokenImportanceWindowSampler(
        reference_tokens=ref_2m,
        valid_vocab_size=VALID_VOCAB_SIZE,
        alpha_rare=cfg.alpha_rare,
        laplace_eps=1.0,
    )
    sampler_unif = RareTokenImportanceWindowSampler(
        reference_tokens=ref_2m,
        valid_vocab_size=VALID_VOCAB_SIZE,
        alpha_rare=0.0,
        laplace_eps=1.0,
    )

    work_gpu = torch.from_numpy(tokens_30m[:8_000_000].view(np.int16)).to(device)
    rare_mask_dev = sampler_is.rare_bottom50_mask.to(device)

    rare_hits_is = 0
    rare_hits_unif = 0
    max_window_rarity_is = 0.0
    mean_rarity_is_accum = 0.0
    mean_rarity_unif_accum = 0.0
    n_Audit_batches = 60

    rarity_dev = sampler_is.token_rarity.to(device)
    for step_idx in range(1, n_Audit_batches + 1):
        _, y_is, _ = sampler_is.sample_batch(work_gpu, cfg.batch_size, cfg.seq_len, step_seed=7000 + step_idx)
        _, y_un, _ = sampler_unif.sample_batch(work_gpu, cfg.batch_size, cfg.seq_len, step_seed=7000 + step_idx)
        rare_hits_is += int(rare_mask_dev[y_is].sum().item())
        rare_hits_unif += int(rare_mask_dev[y_un].sum().item())
        r_is = float(rarity_dev[y_is].mean().item())
        r_un = float(rarity_dev[y_un].mean().item())
        mean_rarity_is_accum += r_is
        mean_rarity_unif_accum += r_un
        max_window_rarity_is = max(max_window_rarity_is, r_is)

    # Analytical unbiasedness check on a candidate pool of M = 64 windows from FineWeb
    starts = torch.randint(0, 4_000_000, (64,), device=device)
    offs = starts.unsqueeze(1) + torch.arange(cfg.seq_len, device=device).unsqueeze(0)
    cand_y = (work_gpu[offs].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
    q_probs, unnorm_inv_w = sampler_is.compute_pool_distribution(cand_y)
    synth_losses = 5.5 + 0.8 * torch.sin(torch.arange(64, device=device, dtype=torch.float32))
    exact_is_est = float((q_probs * unnorm_inv_w * synth_losses).sum().item())
    exact_unif_est = float(synth_losses.mean().item())
    unbiased_abs_err = abs(exact_is_est - exact_unif_est)

    bottom50_gain = float(rare_hits_is) / float(max(1, rare_hits_unif))
    rarity_score_gain = (mean_rarity_is_accum / n_Audit_batches) / max(1e-8, mean_rarity_unif_accum / n_Audit_batches)
    return {
        "alpha_rare": cfg.alpha_rare,
        "pool_multiplier": 4,
        "laplace_eps": 1.0,
        "zero_count_tokens_in_2M_ref": int((sampler_is.token_counts == 0).sum().item()),
        "min_unigram_prob_laplace": float(sampler_is.unigram_probs.min().item()),
        "bottom50_rare_token_hits_is": rare_hits_is,
        "bottom50_rare_token_hits_uniform": rare_hits_unif,
        "bottom50_exposure_gain_ratio": round(bottom50_gain, 4),
        "mean_rarity_score_gain_ratio": round(rarity_score_gain, 4),
        "exact_importance_weighted_bias_err": unbiased_abs_err,
    }


def run_rank_preservation_benchmark(
    tokens_30m: np.ndarray,
    cfg: ProfilerConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Compare N-step rank trajectory: `legacy_boltzmann_real` vs `tied_holo_born + soft_penalties_parseval_iso`."""
    work_gpu = torch.from_numpy(tokens_30m[:8_000_000].view(np.int16)).to(device)
    prior_tokens = torch.from_numpy(tokens_30m[:2_000_000].astype(np.int64))
    ppmi_path = ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt"
    ppmi_obj = torch.load(ppmi_path, map_location="cpu", weights_only=True)
    ppmi_vecs = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj

    tokens_per_step = cfg.batch_size * cfg.seq_len
    h_num = max(1, cfg.d_complex // DEFAULT_D_REGISTER)

    # Arm 1: Legacy GeminiPackedComplexRoPEGPT (meas_proj + hard retraction + stiefel_penalty=0.02)
    torch.manual_seed(1337)
    legacy_cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=PADDED_VOCAB_SIZE,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=cfg.d_complex,
        n_registers=h_num,
        d_register=DEFAULT_D_REGISTER,
        num_layers=cfg.num_layers,
        block_size=cfg.seq_len,
    )
    legacy_model = GeminiPackedComplexRoPEGPT(legacy_cfg).to(device)
    initialize_gemini_complex_gpt_with_ppmi_svd(legacy_model, ppmi_vecs, prior_tokens)
    opt_m_leg, opt_a_leg = build_gemini_complex_optimizers(legacy_model)

    legacy_r0 = legacy_model.compute_sharp_eff_rank()
    t0_leg = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        opt_m_leg.zero_grad(set_to_none=True)
        opt_a_leg.zero_grad(set_to_none=True)
        off = ((step - 1) * tokens_per_step) % (int(work_gpu.numel()) - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
        x_b = seq[:-1].view(cfg.batch_size, cfg.seq_len)
        y_b = seq[1:].view(cfg.batch_size, cfg.seq_len)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss_leg = legacy_model(x_b, targets=y_b, return_logits=False)
        assert loss_leg is not None
        loss_leg.backward()
        torch.nn.utils.clip_grad_norm_(legacy_model.parameters(), max_norm=1.0)
        opt_m_leg.step()
        opt_a_leg.step()
        if step % 2 == 0:
            legacy_model.retract_complex_unitary_weights(rate=0.05)
    legacy_wall = time.perf_counter() - t0_leg
    legacy_r_final = legacy_model.compute_sharp_eff_rank()
    del legacy_model, opt_m_leg, opt_a_leg

    # Arm 2: Wave 40 TiedHoloBornComplexRoPEGPT (tied <e_v| + sequence-side out_operator + soft Parseval isotropy)
    torch.manual_seed(1337)
    holo_cfg = TiedHoloBornConfig(
        vocab_size=PADDED_VOCAB_SIZE,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=cfg.d_complex,
        n_registers=h_num,
        d_register=DEFAULT_D_REGISTER,
        num_layers=cfg.num_layers,
        block_size=cfg.seq_len,
    )
    holo_model = TiedHoloBornComplexRoPEGPT(holo_cfg).to(device)
    holo_model.initialize_isotropic_haar(prior_tokens=prior_tokens, seed=1337)
    opt_m_holo, opt_a_holo = build_tied_holo_born_optimizers(holo_model)

    holo_r0 = holo_model.compute_sharp_eff_rank()
    t0_holo = time.perf_counter()
    last_pen_stats: dict[str, float] = {}
    for step in range(1, cfg.steps + 1):
        opt_m_holo.zero_grad(set_to_none=True)
        opt_a_holo.zero_grad(set_to_none=True)
        off = ((step - 1) * tokens_per_step) % (int(work_gpu.numel()) - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
        x_b = seq[:-1].view(cfg.batch_size, cfg.seq_len)
        y_b = seq[1:].view(cfg.batch_size, cfg.seq_len)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_holo = holo_model(x_b, targets=y_b, return_logits=False)
        assert nll_holo is not None
        reg_pen, last_pen_stats = compute_soft_manifold_and_isotropy_penalties(holo_model)
        (nll_holo + reg_pen).backward()
        torch.nn.utils.clip_grad_norm_(holo_model.parameters(), max_norm=1.0)
        opt_m_holo.step()
        opt_a_holo.step()
    holo_wall = time.perf_counter() - t0_holo
    holo_r_final = holo_model.compute_sharp_eff_rank()

    return {
        "steps": cfg.steps,
        "d_complex": cfg.d_complex,
        "real_equiv_dim": 2 * cfg.d_complex,
        "legacy_boltzmann_real": {
            "step0_sharp_eff_rank": round(legacy_r0, 2),
            "final_sharp_eff_rank": round(legacy_r_final, 2),
            "wall_sec": round(legacy_wall, 2),
        },
        "tied_holo_born_soft_iso": {
            "step0_sharp_eff_rank": round(holo_r0, 2),
            "final_sharp_eff_rank": round(holo_r_final, 2),
            "embed_norm_median": round(last_pen_stats.get("embed_norm_median", 1.0), 4),
            "complex_unitary_err": round(last_pen_stats.get("complex_unitary_err", 0.0), 4),
            "wall_sec": round(holo_wall, 2),
        },
        "rank_preservation_ratio_vs_legacy": round(holo_r_final / max(1e-6, legacy_r_final), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FineWeb Zipf Rarity & Born Rank-Preservation Profiler")
    parser.add_argument("--steps", type=int, default=DEFAULT_PROFILE_STEPS)
    parser.add_argument("--d-complex", type=int, default=DEFAULT_D_COMPLEX)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--alpha-rare", type=float, default=DEFAULT_RARE_ALPHA)
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(ROOT / ".cache" / "fineweb_token_rarity_and_born_rank_profile.json"),
    )
    args = parser.parse_args()

    cfg = ProfilerConfig(
        steps=int(args.steps),
        d_complex=int(args.d_complex),
        num_layers=int(args.num_layers),
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        alpha_rare=float(args.alpha_rare),
        json_out=str(args.json_out),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / ".cache" / "fineweb_100M"

    print(f"[Wave 40 Profiler] Auditing FineWeb Zipf token frequency slices in {data_dir}...", flush=True)
    zipf_info = profile_fineweb_zipf_slices(data_dir)
    tokens_30m = zipf_info.pop("tokens_30m")

    print(f"[Wave 40 Profiler] Auditing RareTokenImportanceWindowSampler on {device}...", flush=True)
    sampler_report = profile_rare_token_sampler(tokens_30m, cfg, device)

    print(f"[Wave 40 Profiler] Running {cfg.steps}-step rank preservation benchmark on {device}...", flush=True)
    rank_report = run_rank_preservation_benchmark(tokens_30m, cfg, device)

    payload = {
        "wave_id": "W40",
        "plan_id": "EXPERIMENT_PLAN_055",
        "zipf_slice_statistics": zipf_info,
        "rare_token_sampler_audit": sampler_report,
        "rank_preservation_benchmark": rank_report,
    }
    out_path = Path(cfg.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[Wave 40 Profiler] Saved profile report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
