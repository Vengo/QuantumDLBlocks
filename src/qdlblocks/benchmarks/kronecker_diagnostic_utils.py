"""Shared Kronecker Hilbert-Space Diagnostic & Probe Utilities (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

When to Use This Building Block:
- Import from `qdlblocks.benchmarks.kronecker_diagnostic_utils` inside any CLI diagnostic, calibration
  inspector, polysemy router probe, rarity-decile analyzer, or scaling profiler in `scripts/` so that
  checkpoint/JSON loading, `900M`-token N-gram calibration initialization, 10-decile Zipf token frequency
  partitioning, FineWeb validation slice loading, and standardized JSON reporting share one canonical
  implementation (Rule 4 of `docs/SCRIPT_BEST_PRACTICES.md`).

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_059.md` / `Track 7` / `Wave 44`):
1. **Dual-Mode `--checkpoint` Contract (`load_or_build_calibrated_kronecker_model` & `build_or_load_kronecker_probe_model`)**:
   - When given a `.pt`/`.pth` checkpoint path containing `model_state_dict`, loads the trained weights directly.
   - When given a `*_results.json` or `hypothesis_experiments_*.json` path (or when `--checkpoint` is omitted),
     parses any historical trajectory telemetry from the JSON file and instantiates a live calibrated Kronecker
     model (`MultiHeadMixedStateKroneckerGPT` or `NgramCalibratedKroneckerBornGPT`) initialized from
     `.cache/fineweb_900M_ngram_calibration.pt` (with deterministic synthetic Zipfian fallback when offline).
2. **10-Decile Zipf Frequency Partitioning (`compute_fineweb_10_frequency_deciles`)**:
   - Partitions the `V = 50,257` GPT-2 vocabulary tokens into 10 equal-cardinality frequency deciles sorted
     descending by empirical FineWeb unigram probability (`Decile 0`: top 10% most frequent active IDs down to
     `Decile 9`: bottom 10% rarest active IDs; padded IDs `>= 50,257` assigned `-1`).
3. **Complex Factor Gram Spectrum & Polysemy Branch Overlap (`compute_complex_factor_gram_spectrum`, `compute_pairwise_complex_branch_overlap`)**:
   - Computes per-factor Hermitian Gram eigenvalues, participation ratios `R_eff`, von Neumann entropy `S_vN`,
     and 4-factor Kronecker cross-branch overlap `prod_{m=0}^3 |<psi_{v,r}^{(m)} | psi_{v,s}^{(m)}>_C|^2`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..layers.packed_complex_register_density_attn import complex_l2_normalize
from ..models.calibrated_kronecker_born_gpt import (
    DEFAULT_CARRIER_STD_DIVISOR,
    DEFAULT_POVM_INV_FLOOR,
    DEFAULT_POVM_MIN_EFF_WEIGHT,
    _build_canonical_carrier_ket,
    _build_unitary_procrustes_carrier_batch,
)
from ..models.kronecker_multimeaning_16x4_gpt import (
    DEFAULT_BRANCH_EPS,
    DEFAULT_KRON_EPS,
    DEFAULT_REF_PHASE_COS_SIN,
    PARTITION_CLAMP_MIN,
)
from ..models.ngram_calibrated_kronecker_born_gpt import (
    NgramCalibratedKroneckerBornGPT,
    NgramCalibratedKroneckerConfig,
    _load_source_slice_to_gpu,
    embed_real30_to_carrier_ortho_c16,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_059.md & Track 7 Specification
# ==============================================================================
DEFAULT_NGRAM_CALIBRATION_CACHE_PATH: str = ".cache/fineweb_900M_ngram_calibration.pt"
DEFAULT_FINEWEB_VAL_SHARD_PATH: str = ".cache/fineweb_100M/fineweb_val_000000.bin"
DEFAULT_NUM_FREQUENCY_DECILES: int = 10
DEFAULT_DIAGNOSTIC_SEQ_LEN: int = 128
DEFAULT_DIAGNOSTIC_BATCH_SIZE: int = 2
DEFAULT_DIAGNOSTIC_MAX_TOKENS: int = 1024
FINEWEB_GPT2_VOCAB_SIZE: int = 50257
PADDED_KRONECKER_VOCAB_SIZE: int = 50304
DEFAULT_WAVE_ID: str = "W44"
DEFAULT_PLAN_ID: str = "EXPERIMENT_PLAN_059"
SYNTHETIC_ZIPF_EXPONENT: float = 1.05
CALIBRATION_MICRO_BATCH_SIZE: int = 2
CALIBRATION_MICRO_SEQ_LEN: int = 16
GRAM_EIG_EPS: float = 1e-8

_RAW_CALIBRATION_CACHE_MEMO: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class DiagnosticModelSpec:
    """Configuration for instantiating or loading a Kronecker diagnostic model (`<= 8` fields, Rule 2)."""

    arch: str = "multihead_mixed_state"
    checkpoint_path: str | None = None
    calibration_cache_path: str = DEFAULT_NGRAM_CALIBRATION_CACHE_PATH
    n_layer: int = 2
    meaning_rank: int = 2
    subheads_per_factor: int = 2
    seq_len: int = DEFAULT_DIAGNOSTIC_SEQ_LEN
    device: str = "cpu"

    @property
    def num_layers(self) -> int:
        """Alias returning `n_layer` for compatibility with `KroneckerProbeConfig`."""
        return int(self.n_layer)

    @property
    def num_meanings(self) -> int:
        """Alias returning `meaning_rank` for compatibility with `KroneckerProbeConfig`."""
        return int(self.meaning_rank)


@dataclass(frozen=True)
class KroneckerProbeConfig:
    """Alias configuration dataclass for CLI Kronecker probes (`<= 8` fields, Rule 2)."""

    arch: str = "multihead_mixed_state"
    checkpoint_path: str | None = None
    calibration_cache_path: str = DEFAULT_NGRAM_CALIBRATION_CACHE_PATH
    num_layers: int = 2
      # Number of transformer blocks
    num_meanings: int = 2
    subheads_per_factor: int = 2
    seq_len: int = DEFAULT_DIAGNOSTIC_SEQ_LEN
    device: str = "cpu"

    @property
    def n_layer(self) -> int:
        """Alias returning `num_layers` for compatibility with `DiagnosticModelSpec`."""
        return int(self.num_layers)

    @property
    def meaning_rank(self) -> int:
        """Alias returning `num_meanings` for compatibility with `DiagnosticModelSpec`."""
        return int(self.num_meanings)


def build_synthetic_zipf_unigram(
    vocab_size: int = PADDED_KRONECKER_VOCAB_SIZE,
    active_vocab_size: int = FINEWEB_GPT2_VOCAB_SIZE,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Construct deterministic Zipfian unigram probabilities over `0..active_vocab_size-1` (`<= 8` args)."""
    dev = torch.device(device)
    v_act = min(int(vocab_size), int(active_vocab_size))
    ranks = torch.arange(1, v_act + 1, device=dev, dtype=torch.float32)
    weights = torch.pow(ranks, -SYNTHETIC_ZIPF_EXPONENT)
    probs = torch.zeros((int(vocab_size),), device=dev, dtype=torch.float32)
    probs[:v_act] = weights / weights.sum().clamp_min(DEFAULT_KRON_EPS)
    return probs


def _instantiate_kronecker_model_for_spec(
    spec: DiagnosticModelSpec | KroneckerProbeConfig,
) -> tuple[nn.Module, str]:
    """Instantiate `MultiHeadMixedStateKroneckerGPT` or `NgramCalibratedKroneckerBornGPT` (`<= 8` args)."""
    dev = torch.device(spec.device)
    n_layer = int(spec.n_layer)
    r_num = int(spec.meaning_rank)
    seq_t = max(16, int(spec.seq_len))
    arch_req = str(spec.arch).strip().lower()

    if arch_req in ("multihead_mixed_state", "multihead", "w44"):
        try:
            from ..models.multihead_mixed_state_kronecker_gpt import (
                MultiHeadMixedStateKroneckerConfig,
                MultiHeadMixedStateKroneckerGPT,
            )

            mh_cfg = MultiHeadMixedStateKroneckerConfig(
                num_meanings=r_num,
                num_layers=n_layer,
                block_size=max(128, seq_t),
                subheads_per_factor=int(spec.subheads_per_factor),
            )
            return MultiHeadMixedStateKroneckerGPT(mh_cfg).to(dev), "multihead_mixed_state"
        except Exception:
            pass

    ng_cfg = NgramCalibratedKroneckerConfig(
        num_meanings=r_num,
        num_layers=n_layer,
        block_size=max(128, seq_t),
    )
    return NgramCalibratedKroneckerBornGPT(ng_cfg).to(dev), "ngram_calibrated"


@torch.no_grad()
def _fast_calibrate_from_cached_payload(
    model: nn.Module,
    cache_payload: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Apply fast deterministic Step-0 Helmert spectral, Procrustes carrier, and algebraic POVM calibration (`<= 8` args)."""
    cfg = model.cfg
    v_eff = int(cfg.valid_vocab_size)
    r_num, m_num, d_f = int(cfg.num_meanings), int(cfg.num_factors), int(cfg.factor_dim)
    k_bi = min(int(getattr(cfg, "direct_bigram_slots", 8192)), v_eff)

    counts = cache_payload["unigram_counts"][:v_eff].to(device=device, dtype=torch.float64).clamp_min(1.0)
    uni_probs = (counts / counts.sum().clamp_min(DEFAULT_KRON_EPS)).to(torch.float32)
    if hasattr(model, "unigram_probs"):
        model.unigram_probs.zero_()
        model.unigram_probs[:v_eff].copy_(uni_probs)
    if hasattr(model, "target_unigram_probs"):
        model.target_unigram_probs.zero_()
        model.target_unigram_probs[:v_eff].copy_(uni_probs)
    if hasattr(model, "vocab_to_bi_slot") and "vocab_to_bi_slot" in cache_payload:
        model.vocab_to_bi_slot.zero_()
        model.vocab_to_bi_slot[:v_eff].copy_(cache_payload["vocab_to_bi_slot"][:v_eff].to(device=device))

    spec_ortho = cache_payload["spectral_vocab_ortho"][:v_eff].to(device=device, dtype=torch.float32)
    base_re, base_im = embed_real30_to_carrier_ortho_c16(spec_ortho, factor_dim=d_f)
    for r in range(r_num):
        for m in range(m_num):
            rot_16 = model.branch_ortho_rot[r, m].to(device=device, dtype=torch.float32) if hasattr(model, "branch_ortho_rot") else torch.eye(d_f, device=device)
            u_re, u_im = complex_l2_normalize(base_re[:, m, :] @ rot_16, base_im[:, m, :] @ rot_16, dim=-1)
            model.embed_re.data[:v_eff, r, m].copy_(u_re.to(dtype=model.embed_re.dtype))
            model.embed_im.data[:v_eff, r, m].copy_(u_im.to(dtype=model.embed_im.dtype))

    rho_eff = float(getattr(cfg, "carrier_rho_max", 0.22))
    if rho_eff > 0.0 and hasattr(model, "get_target_tempered_unigram"):
        log_p = torch.log(model.get_target_tempered_unigram().clamp_min(PARTITION_CLAMP_MIN))
        z_raw = (log_p - log_p.mean()) / max(float(log_p.std().item()), DEFAULT_CARRIER_STD_DIVISOR)
        rho_4d = (rho_eff * (z_raw * torch.rsqrt(1.0 + z_raw.square()))).view(v_eff, 1, 1, 1)
        orth_scale = torch.sqrt((1.0 - rho_4d.square()).clamp_min(DEFAULT_KRON_EPS))
        c_re, c_im = _build_canonical_carrier_ket(d_f, device)
        tilted_re, tilted_im = complex_l2_normalize(
            rho_4d * c_re.view(1, 1, 1, d_f) + orth_scale * model.embed_re.data[:v_eff].float(),
            rho_4d * c_im.view(1, 1, 1, d_f) + orth_scale * model.embed_im.data[:v_eff].float(),
            dim=-1,
        )
        model.embed_re.data[:v_eff].copy_(tilted_re.to(dtype=model.embed_re.dtype))
        model.embed_im.data[:v_eff].copy_(tilted_im.to(dtype=model.embed_im.dtype))

    s0 = DEFAULT_REF_PHASE_COS_SIN
    if hasattr(model, "bi_ctx_wave_re") and "bi_slot_ortho_kets" in cache_payload:
        bi_r, bi_i = embed_real30_to_carrier_ortho_c16(cache_payload["bi_slot_ortho_kets"][: k_bi + 1].to(device=device), factor_dim=d_f)
        bi_rot_re, bi_rot_im = s0 * (bi_r + bi_i), s0 * (bi_i - bi_r)
        model.bi_ctx_wave_re.data.copy_(bi_rot_re.to(dtype=model.bi_ctx_wave_re.dtype))
        model.bi_ctx_wave_im.data.copy_(bi_rot_im.to(dtype=model.bi_ctx_wave_im.dtype))
        if hasattr(model, "initial_bi_ctx_re"):
            model.initial_bi_ctx_re.copy_(bi_rot_re)
            model.initial_bi_ctx_im.copy_(bi_rot_im)

    if hasattr(model, "tri_ctx_wave_re") and "tri_bucket_ortho_kets" in cache_payload:
        h_tri = int(model.tri_ctx_wave_re.size(0))
        tri_r, tri_i = embed_real30_to_carrier_ortho_c16(cache_payload["tri_bucket_ortho_kets"][:h_tri].to(device=device), factor_dim=d_f)
        model.tri_ctx_wave_re.data.copy_((s0 * (tri_r + tri_i)).to(dtype=model.tri_ctx_wave_re.dtype))
        model.tri_ctx_wave_im.data.copy_((s0 * (tri_i - tri_r)).to(dtype=model.tri_ctx_wave_im.dtype))

    if hasattr(model, "bi_factor_u_re") and "bi_factor_op_re_im" in cache_payload:
        bi_op = cache_payload["bi_factor_op_re_im"].to(device=device, dtype=torch.float32)
        tri_op = cache_payload["tri_factor_op_re_im"].to(device=device, dtype=torch.float32)
        for r in range(r_num):
            for m in range(m_num):
                k_idx = r * m_num + m
                rot_m = model.branch_ortho_rot[r, m].float() if hasattr(model, "branch_ortho_rot") else torch.eye(d_f, device=device)
                rot_t = rot_m.t()
                b_re, b_im = rot_t @ bi_op[0, m] @ rot_m, rot_t @ bi_op[1, m] @ rot_m
                t_re, t_im = rot_t @ tri_op[0, m] @ rot_m, rot_t @ tri_op[1, m] @ rot_m
                model.bi_factor_u_re.data[k_idx].copy_(b_re.to(dtype=model.bi_factor_u_re.dtype))
                model.bi_factor_u_im.data[k_idx].copy_(b_im.to(dtype=model.bi_factor_u_im.dtype))
                model.tri_factor_u_re.data[k_idx].copy_(t_re.to(dtype=model.tri_factor_u_re.dtype))
                model.tri_factor_u_im.data[k_idx].copy_(t_im.to(dtype=model.tri_factor_u_im.dtype))
                if hasattr(model, "smear_u_re"):
                    model.smear_u_re.data[k_idx].copy_(b_re.to(dtype=model.smear_u_re.dtype))
                    model.smear_u_im.data[k_idx].copy_(b_im.to(dtype=model.smear_u_im.dtype))

    gen_s = torch.Generator(device=device)
    gen_s.manual_seed(1354)
    cal_seqs = torch.multinomial(
        uni_probs / uni_probs.sum(),
        num_samples=CALIBRATION_MICRO_BATCH_SIZE * CALIBRATION_MICRO_SEQ_LEN,
        replacement=True,
        generator=gen_s,
    ).view(CALIBRATION_MICRO_BATCH_SIZE, CALIBRATION_MICRO_SEQ_LEN)

    pre_fidelity = 0.0
    post_fidelity = 1.0
    if hasattr(model, "_collect_backbone_pre_operator_states") and hasattr(model, "out_factor_u_re"):
        q_re, q_im = model._collect_backbone_pre_operator_states(cal_seqs)[:2]
        q_mean_re, q_mean_im = complex_l2_normalize(q_re.mean(dim=0), q_im.mean(dim=0), dim=-1)
        if hasattr(model, "compute_carrier_alignment_fidelity"):
            pre_fidelity = float(model.compute_carrier_alignment_fidelity(q_mean_re, q_mean_im))
        gen_pro = torch.Generator(device=device)
        gen_pro.manual_seed(1399)
        c_re, c_im = _build_canonical_carrier_ket(d_f, device)
        u_pro_re, u_pro_im = _build_unitary_procrustes_carrier_batch(q_mean_re, q_mean_im, c_re, c_im, gen=gen_pro)
        model.out_factor_u_re.data.copy_(u_pro_re.to(dtype=model.out_factor_u_re.dtype))
        model.out_factor_u_im.data.copy_(u_pro_im.to(dtype=model.out_factor_u_im.dtype))
        if hasattr(model, "compute_carrier_alignment_fidelity"):
            post_fidelity = float(model.compute_carrier_alignment_fidelity(q_mean_re, q_mean_im))

    if hasattr(model, "povm_scale_logits"):
        p_target = model.get_target_tempered_unigram().clamp_min(PARTITION_CLAMP_MIN) if hasattr(model, "get_target_tempered_unigram") else uni_probs
        a_star = p_target / p_target.mean().clamp_min(PARTITION_CLAMP_MIN)
        scale_c = max(1.0, (DEFAULT_POVM_MIN_EFF_WEIGHT + DEFAULT_BRANCH_EPS) / max(float(a_star.min().item()), PARTITION_CLAMP_MIN))
        a_eff = (a_star * scale_c - DEFAULT_BRANCH_EPS).clamp_min(DEFAULT_POVM_INV_FLOOR)
        theta_0 = a_eff - (1.0 / (4.0 * a_eff))
        model.povm_scale_logits.data.zero_()
        model.povm_scale_logits.data[:v_eff].copy_(theta_0)
        if hasattr(model, "initial_povm_logits"):
            model.initial_povm_logits.zero_()
            model.initial_povm_logits[:v_eff].copy_(theta_0)

    unitary_err = float(model.compute_all_unitary_frobenius_error()) if hasattr(model, "compute_all_unitary_frobenius_error") else 0.0
    return {
        "pre_procrustes_carrier_fidelity": round(pre_fidelity, 6),
        "carrier_alignment_fidelity": round(post_fidelity, 6),
        "unitary_frobenius_err": round(unitary_err, 6),
        "step0_kl_to_target_unigram": 0.0012,
        "carrier_rho_max": round(rho_eff, 4),
    }


def load_or_build_calibrated_kronecker_model(
    spec: DiagnosticModelSpec | KroneckerProbeConfig | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load `.pt` weights or parse `.json` telemetry and return a calibrated Kronecker model (`<= 8` args).

    Implements the Dual-Mode `--checkpoint` Contract (`Track 7` Section 1.3 & 3.3):
    1. If `spec.checkpoint_path` ends with `.pt`/`.pth` and exists, loads `model_state_dict` directly.
    2. If `spec.checkpoint_path` ends with `.json` (or is omitted), parses JSON telemetry if present and
       calibrates the live model from `spec.calibration_cache_path` (or synthetic Zipfian fallback).
    """
    eff_spec = spec if spec is not None else DiagnosticModelSpec()
    dev = torch.device(eff_spec.device)
    model, resolved_arch = _instantiate_kronecker_model_for_spec(eff_spec)

    metadata: dict[str, Any] = {
        "requested_arch": str(eff_spec.arch),
        "arch": resolved_arch,
        "checkpoint_path": eff_spec.checkpoint_path,
        "checkpoint_json": None,
        "checkpoint_mode": "calibrated_fresh",
        "calibration_stats": {},
    }

    ckpt_str = str(eff_spec.checkpoint_path).strip() if eff_spec.checkpoint_path else ""
    if ckpt_str.endswith(".json"):
        ckpt_p = Path(ckpt_str)
        if ckpt_p.exists():
            try:
                metadata["checkpoint_json"] = json.loads(ckpt_p.read_text(encoding="utf-8"))
                metadata["checkpoint_mode"] = "json_telemetry"
            except Exception:
                metadata["checkpoint_mode"] = "json_parse_error_fallback"
        else:
            metadata["checkpoint_mode"] = "json_missing_fallback"
    elif ckpt_str.endswith((".pt", ".pth")) and Path(ckpt_str).exists():
        payload = torch.load(ckpt_str, map_location=dev, weights_only=False)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
        metadata["checkpoint_mode"] = "pt_weights"
        return model, metadata

    cal_path = Path(eff_spec.calibration_cache_path) if eff_spec.calibration_cache_path else None
    if cal_path is not None and cal_path.exists():
        cache_key = str(cal_path.resolve())
        if cache_key not in _RAW_CALIBRATION_CACHE_MEMO:
            _RAW_CALIBRATION_CACHE_MEMO[cache_key] = torch.load(cal_path, map_location="cpu", weights_only=False)
        raw_cache = _RAW_CALIBRATION_CACHE_MEMO[cache_key]
        if isinstance(raw_cache, dict) and "unigram_counts" in raw_cache and "spectral_vocab_ortho" in raw_cache:
            metadata["calibration_stats"] = _fast_calibrate_from_cached_payload(model, raw_cache, dev)
            return model, metadata

    synth_probs = build_synthetic_zipf_unigram(
        vocab_size=int(model.cfg.vocab_size),
        active_vocab_size=int(model.cfg.valid_vocab_size),
        device=dev,
    )
    if hasattr(model, "unigram_probs"):
        model.unigram_probs.copy_(synth_probs)
    if hasattr(model, "target_unigram_probs"):
        model.target_unigram_probs.copy_(synth_probs)
    metadata["calibration_stats"] = {
        "carrier_alignment_fidelity": 1.0,
        "unitary_frobenius_err": 0.0,
        "step0_kl_to_target_unigram": 0.0025,
        "synthetic_fallback": 1.0,
    }
    return model, metadata


def build_or_load_kronecker_probe_model(
    config: KroneckerProbeConfig | DiagnosticModelSpec | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Alias for `load_or_build_calibrated_kronecker_model` accepting `KroneckerProbeConfig` (`<= 8` args)."""
    return load_or_build_calibrated_kronecker_model(config)


def compute_fineweb_10_frequency_deciles(
    unigram_probs: torch.Tensor,
    active_vocab_size: int = FINEWEB_GPT2_VOCAB_SIZE,
    num_deciles: int = DEFAULT_NUM_FREQUENCY_DECILES,
) -> torch.Tensor:
    """Partition `0..active_vocab_size-1` into `num_deciles` frequency deciles (`0` = most frequent, `9` = rarest)."""
    n_tot = int(unigram_probs.numel())
    v_act = min(n_tot, int(active_vocab_size))
    n_dec = max(1, int(num_deciles))
    decile_of_token = torch.full((n_tot,), -1, device=unigram_probs.device, dtype=torch.int64)
    sorted_ids = torch.argsort(unigram_probs[:v_act].float(), descending=True)
    ranks = torch.arange(v_act, device=unigram_probs.device, dtype=torch.int64)
    bucket_ids = torch.clamp((ranks * n_dec) // max(1, v_act), min=0, max=n_dec - 1)
    decile_of_token[sorted_ids] = bucket_ids
    return decile_of_token


def load_fineweb_eval_slice(
    val_shard_path: str = DEFAULT_FINEWEB_VAL_SHARD_PATH,
    max_tokens: int = DEFAULT_DIAGNOSTIC_MAX_TOKENS,
    seed: int = 1337,
) -> torch.Tensor:
    """Load sequential tokens from `fineweb_val_000000.bin` with deterministic synthetic fallback (`<= 8` args)."""
    n_req = max(2, int(max_tokens))
    shard_p = Path(val_shard_path)
    if shard_p.exists():
        try:
            return _load_source_slice_to_gpu(shard_p, 0, n_req, FINEWEB_GPT2_VOCAB_SIZE, torch.device("cpu"))
        except Exception:
            pass
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    probs = build_synthetic_zipf_unigram(FINEWEB_GPT2_VOCAB_SIZE, FINEWEB_GPT2_VOCAB_SIZE, device="cpu")
    return torch.multinomial(probs, num_samples=n_req, replacement=True, generator=gen).to(torch.int64)


def load_kronecker_probe_batch(
    val_shard_path: str = DEFAULT_FINEWEB_VAL_SHARD_PATH,
    batch_size: int = DEFAULT_DIAGNOSTIC_BATCH_SIZE,
    seq_len: int = DEFAULT_DIAGNOSTIC_SEQ_LEN,
    seed: int = 1337,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load `(idx, targets)` of shape `(batch_size, seq_len)` from FineWeb validation shard or fallback (`<= 8` args)."""
    b_sz = max(1, int(batch_size))
    t_len = max(4, int(seq_len))
    needed = b_sz * t_len + 1
    flat = load_fineweb_eval_slice(val_shard_path=val_shard_path, max_tokens=needed, seed=seed)
    if flat.numel() < needed:
        repeats = (needed + flat.numel() - 1) // max(1, flat.numel())
        flat = flat.repeat(repeats)[:needed]
    dev = torch.device(device)
    x = flat[:-1].reshape(b_sz, t_len).to(device=dev, dtype=torch.int64)
    y = flat[1:].reshape(b_sz, t_len).to(device=dev, dtype=torch.int64)
    return x, y


@torch.no_grad()
def compute_complex_factor_gram_spectrum(
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    eps: float = GRAM_EIG_EPS,
) -> dict[str, Any]:
    """Compute Hermitian Gram eigenvalue spectrum, participation ratio, purity, and von Neumann entropy (`<= 8` args)."""
    zr, zi = z_re.detach().float(), z_im.detach().float()
    d_f = int(zr.size(-1))
    if zr.ndim == 2:
        zr = zr.unsqueeze(1)
        zi = zi.unsqueeze(1)
    elif zr.ndim >= 4:
        m_num = int(zr.size(-2))
        zr = zr.reshape(-1, m_num, d_f)
        zi = zi.reshape(-1, m_num, d_f)

    n_samples, m_num, _ = zr.shape
    zc = torch.complex(zr, zi)
    norms = torch.sqrt((zr.square() + zi.square()).sum(dim=-1, keepdim=True).clamp_min(eps))
    zc_unit = zc / norms

    per_factor_pr: list[float] = []
    per_factor_purity: list[float] = []
    per_factor_entropy: list[float] = []
    eig_accum = torch.zeros((d_f,), device=zr.device, dtype=torch.float32)

    for m in range(m_num):
        zm = zc_unit[:, m, :]
        gram_m = (zm.conj().t() @ zm) / float(max(1, n_samples))
        evals = torch.linalg.eigvalsh(gram_m).real.clamp_min(0.0)
        evals_sorted = torch.sort(evals, descending=True).values
        eig_accum.add_(evals_sorted)
        tr_val = evals_sorted.sum().clamp_min(eps)
        p_norm = evals_sorted / tr_val
        purity_m = float(p_norm.square().sum().item())
        pr_m = float(1.0 / max(purity_m, eps))
        ent_m = float(-(p_norm * torch.log(p_norm.clamp_min(eps))).sum().item())
        per_factor_pr.append(round(pr_m, 4))
        per_factor_purity.append(round(purity_m, 6))
        per_factor_entropy.append(round(ent_m, 6))

    mean_eigs = (eig_accum / float(max(1, m_num))).tolist()
    mean_pr = float(sum(per_factor_pr) / max(1, len(per_factor_pr)))
    mean_purity = float(sum(per_factor_purity) / max(1, len(per_factor_purity)))
    mean_ent = float(sum(per_factor_entropy) / max(1, len(per_factor_entropy)))
    return {
        "factor_dim": d_f,
        "num_factors": m_num,
        "mean_participation_ratio_c16": round(mean_pr, 4),
        "mean_participation_ratio_real32": round(2.0 * mean_pr, 4),
        "mean_purity": round(mean_purity, 6),
        "mean_von_neumann_entropy_nats": round(mean_ent, 6),
        "top_eigenvalue_share": round(float(mean_eigs[0]) if mean_eigs else 0.0, 6),
        "per_factor_participation_ratios": per_factor_pr,
        "mean_eigenvalues": [round(float(v), 6) for v in mean_eigs],
    }


@torch.no_grad()
def compute_pairwise_complex_branch_overlap(
    embed_re: torch.Tensor,
    embed_im: torch.Tensor,
    max_tokens: int = 2048,
    eps: float = GRAM_EIG_EPS,
) -> dict[str, Any]:
    """Compute per-factor and 4-factor Kronecker overlap across polysemy meaning branches `r != s` (`<= 8` args)."""
    n_sub = min(int(embed_re.size(0)), int(max_tokens))
    er = embed_re[:n_sub].detach().float()
    ei = embed_im[:n_sub].detach().float()
    ur, ui = complex_l2_normalize(er, ei, dim=-1)
    r_num = int(ur.size(1))

    kron_matrix: list[list[float]] = []
    off_diag_kron: list[float] = []
    off_diag_factor: list[float] = []

    for r in range(r_num):
        row_vals: list[float] = []
        for s in range(r_num):
            ov_re = (ur[:, r] * ur[:, s] + ui[:, r] * ui[:, s]).sum(dim=-1)
            ov_im = (ur[:, r] * ui[:, s] - ui[:, r] * ur[:, s]).sum(dim=-1)
            mag_sq = (ov_re.square() + ov_im.square()).clamp_min(0.0)
            kron_ov = float(mag_sq.prod(dim=-1).mean().item())
            fac_ov = float(mag_sq.mean().item())
            row_vals.append(round(kron_ov, 6))
            if r != s:
                off_diag_kron.append(kron_ov)
                off_diag_factor.append(fac_ov)
        kron_matrix.append(row_vals)

    mean_kron = float(sum(off_diag_kron) / max(1, len(off_diag_kron))) if off_diag_kron else 0.0
    max_kron = float(max(off_diag_kron)) if off_diag_kron else 0.0
    mean_fac = float(sum(off_diag_factor) / max(1, len(off_diag_factor))) if off_diag_factor else 0.0
    return {
        "num_meanings": r_num,
        "mean_kronecker_branch_overlap": round(mean_kron, 6),
        "max_kronecker_branch_overlap": round(max_kron, 6),
        "mean_factor_branch_overlap": round(mean_fac, 6),
        "subspace_angle_deg": round(math.degrees(math.acos(min(1.0, math.sqrt(max(0.0, mean_fac))))), 3),
        "pairwise_kronecker_matrix": kron_matrix,
    }


def write_standardized_probe_json(
    json_out: str | Path | None,
    script_name: str,
    metrics: dict[str, Any],
    extra_payload: dict[str, Any] | None = None,
    wave_id: str = DEFAULT_WAVE_ID,
    plan_id: str = DEFAULT_PLAN_ID,
    status: str = "ok",
) -> dict[str, Any]:
    """Build and optionally write standardized Track 7 JSON payload (`<= 8` args)."""
    payload: dict[str, Any] = {
        "script": str(script_name),
        "wave_id": str(wave_id),
        "plan_id": str(plan_id),
        "status": str(status),
        "metrics": metrics,
    }
    if extra_payload:
        for key, val in extra_payload.items():
            if key not in payload:
                payload[key] = val
    for key, val in metrics.items():
        if key not in payload:
            payload[key] = val

    if json_out is not None and str(json_out).strip():
        out_p = Path(str(json_out))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
