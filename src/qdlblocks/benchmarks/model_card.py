"""Declarative Run Model Cards & Zero-Downtime Dynamic Wave Registry (`qdlblocks.benchmarks.model_card`).

Implements Experiment Plan 017:
- `RunModelCard` dataclass and `validate_model_card` schema validator.
- `deterministic_run_color` high-contrast deterministic hex color assignment.
- `parse_cli_args_to_card_sections` and `build_model_card_from_manifest_entry` for automatic
  Model Card construction from `configs/wave*_fleet.json` and historical `FLEET_SPECS`.
- `update_model_card_telemetry` and `save_model_card_atomic` for merging live training/evaluation
  metrics and quantum observables into `.cache/model_cards/<run_key>.json`.
- `render_model_card_markdown` for one-click GitHub-Flavored Markdown export.
- `DynamicWaveRegistry` for zero-downtime wave/run discovery and session-deduplicated remote polling.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_MODEL_CARDS_DIR = REPO_ROOT / ".cache" / "model_cards"
DEFAULT_LIVE_LOGS_DIR = REPO_ROOT / ".cache" / "live_logs"

TPU_V5E_BF16_PEAK_TFLOPS = 197.0
BLACKWELL_RTX6000_BF16_PEAK_TFLOPS = 503.8
A100_BF16_PEAK_TFLOPS = 312.0
L4_BF16_PEAK_TFLOPS = 121.0

HIGH_CONTRAST_PALETTE = (
    "#00f5d4",
    "#ff2a85",
    "#00bbf9",
    "#fee440",
    "#9b5de5",
    "#00ff87",
    "#ff6e40",
    "#38ef7d",
    "#f15bb5",
    "#60efff",
    "#ffbe0b",
    "#80ffdb",
    "#ff0055",
    "#72efdd",
    "#ffd166",
    "#a8ff78",
)

WAVE_DEFAULT_LABELS: dict[str, str] = {
    "Baseline": "Baseline (Single-Shard Multi-Epoch Controls)",
    "W1": "Wave 1 (Plans 002–005: MTP, Seed Scaling, Tangent Stiefel, Kraus, Tiled POVM)",
    "W2": "Wave 2 (Plans 006–007: Overcomplete 2d Frame & Seed Readout Adapter)",
    "W2B": "Wave 2b (Plan 007 SeedAdapter Ablations)",
    "W3": "Wave 3 (Plans 008–010: Two-Beam Interference & Phase-Sensitive Seed Adapter)",
    "W4": "Wave 4 (Plan 011: 900M Unique FineWeb Multi-Shard Speedrun & WSD Cooldown)",
    "W5": "Wave 5 (Plan 012: 120M–211M Multi-Axis Parameter Scaling on 900M Tokens)",
    "W6": "Wave 6 (Plan 013: Dynamic Bloch-Purity Beta & Classical Modded-NanoGPT Baselines)",
    "W7": "Wave 7 (Plan 014: Early-Epoch Quantum Fixes & Classical 12L Speedrun)",
    "W8": "Wave 8 (Plan 015: True Cross-Layer Layer-0 V^(0) & Fused Triton Purified Born)",
    "W9": "Wave 9 (Plan 016: PR #360 Speedrun V2, Quantum Unification & Optimizer Ablations)",
    "W10": "Wave 10 (Plan 018: Speedrun V3 Step-Budget Scaling, Fast SDPA, SGD Ablation & Deep L=12 Quantum)",
    "W11": "Wave 11 (Plan 019: Exact Modded-NanoGPT Leaderboard Parity & Deep L=12 4d-Frame Quantum Unification)",
    "W12": "Wave 12 (Plan 020: Classical Leaderboard-Exact Unification & Deep L=12 H=8 d128 Quantum V5)",
    "W13": "Wave 13 (Plan 021: 9-Shard PPMI-SVD Co-Occurrence Init, Hashed N-Gram Embeddings & ANVIL Cat A/B/C)",
    "W14": "Wave 14 (Plan 022: 64-Qubit Sparse Fock Excitation Space, Sparse 1/2/3-Gram Qubit Encodings & 4M Density Matrix Layers)",
    "W15": "Wave 15 (Plan 023: 16-Qubit Multi-Register Amplitudes & 256-State Interacting Quantum Hierarchy)",
    "VERIFY": "Pre-Launch Micro-Verifications & Controlled Hypothesis Tests (Plans 013–023)",
}

WAVE_DEFAULT_PLANS: dict[str, str] = {
    "Baseline": "EXPERIMENT_PLAN_001",
    "W1": "EXPERIMENT_PLAN_002_005",
    "W2": "EXPERIMENT_PLAN_006",
    "W2B": "EXPERIMENT_PLAN_007",
    "W3": "EXPERIMENT_PLAN_008_010",
    "W4": "EXPERIMENT_PLAN_011",
    "W5": "EXPERIMENT_PLAN_012",
    "W6": "EXPERIMENT_PLAN_013",
    "W7": "EXPERIMENT_PLAN_014",
    "W8": "EXPERIMENT_PLAN_015",
    "W9": "EXPERIMENT_PLAN_016",
    "W10": "EXPERIMENT_PLAN_018",
    "W11": "EXPERIMENT_PLAN_019",
    "W12": "EXPERIMENT_PLAN_020",
    "W13": "EXPERIMENT_PLAN_021",
    "W14": "EXPERIMENT_PLAN_022",
    "W15": "EXPERIMENT_PLAN_023",
    "VERIFY": "EXPERIMENT_PLAN_013_023",
}

BASELINE_RUN_KEYS: frozenset[str] = frozenset(
    {"gpu_5ep", "gpu_3ep", "gpu_wsd_mtp", "tpu_3ep", "tpu_wsd_mtp"}
)


def deterministic_run_color(run_key: str) -> str:
    """Compute a deterministic, high-saturation hex color on dark backgrounds from `run_key`."""
    digest = hashlib.sha256(run_key.encode("utf-8")).digest()
    # Convert HSL with high saturation (85-98%) and bright lightness (60-72%) to RGB hex
    hue = ((digest[0] << 8) | digest[1]) % 360
    sat = 0.86 + (digest[2] % 13) / 100.0
    light = 0.60 + (digest[3] % 13) / 100.0
    c = (1.0 - abs(2.0 * light - 1.0)) * sat
    x = c * (1.0 - abs(((hue / 60.0) % 2.0) - 1.0))
    m = light - c / 2.0
    if hue < 60:
        r1, g1, b1 = c, x, 0.0
    elif hue < 120:
        r1, g1, b1 = x, c, 0.0
    elif hue < 180:
        r1, g1, b1 = 0.0, c, x
    elif hue < 240:
        r1, g1, b1 = 0.0, x, c
    elif hue < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    r = int(round((r1 + m) * 255))
    g = int(round((g1 + m) * 255))
    b = int(round((b1 + m) * 255))
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def infer_wave_id_and_num(run_key: str, raw_wave: str = "") -> tuple[str, int]:
    """Normalize a wave identifier (e.g. `wave8_phase1_900m` or `w9_...`) into `(wave_id, wave_num)`."""
    rk_low = run_key.strip().lower()
    rw_up = raw_wave.strip().upper()
    if rw_up in ("VERIFY", "MICRO") or rk_low.startswith(("verify", "micro_", "plan0")):
        m_v = re.search(r"(?:verify|plan)0*(\d+)", rk_low)
        return "VERIFY", (int(m_v.group(1)) if m_v else 0)
    if "adapter" in rk_low and rk_low.startswith("w2_") and rw_up in ("", "W2", "W2B"):
        return "W2B", 2
    for candidate in (raw_wave.strip(), run_key.strip()):
        if not candidate:
            continue
        if candidate.lower() == "baseline":
            return "Baseline", 0
        m = re.match(r"^(?:wave|w)_?(\d+)([a-z]?)", candidate, flags=re.IGNORECASE)
        if m:
            num = int(m.group(1))
            suffix = m.group(2).upper()
            return f"W{num}{suffix}", num
    if rk_low.startswith("hyp_"):
        return "W1", 1
    if rk_low in BASELINE_RUN_KEYS or rk_low.startswith(("gpu_", "tpu_")):
        return "Baseline", 0
    if raw_wave.strip():
        return rw_up, 99
    return "W1", 1


def _estimate_params_m(arch: dict[str, Any]) -> float:
    """Estimate parameter count in millions from parsed architectural parameters when log header is not yet present."""
    vocab = 50304
    d = int(arch.get("embed_dim", 1024))
    layers = int(arch.get("num_layers", 6))
    family = arch.get("family", "quantum_born")
    num_ve = int(arch.get("num_value_embeddings", 0))
    skip_layers = set(arch.get("skip_attn_layers", []))
    unet_mode = arch.get("unet_skip_mode", "matrix")

    if bool(arch.get("use_sparse_64qubit", False)):
        n_q = int(arch.get("num_qubits", 64))
        sp_k = int(arch.get("sparse_ngram_k", 16))
        ngram_params = int(arch.get("ngram_vocab_size", 262144)) * sp_k + vocab * (n_q + sp_k)
    elif bool(arch.get("use_ngram_hash", False)):
        ngram_params = int(arch.get("ngram_vocab_size", 262144)) * d
    else:
        ngram_params = 0

    if family == "classical_modded":
        hidden = int(arch.get("hidden_dim", 4 * d))
        untied = bool(arch.get("zero_init_residual_and_head", False))
        wte_params = vocab * d * (2 if untied else 1)
        ve_params = num_ve * vocab * d
        attn_layers = max(0, layers - len(skip_layers))
        attn_params = attn_layers * (4 * d * d)
        mlp_params = layers * (2 * d * hidden)
        unet_params = (layers // 2) * (d * d if unet_mode == "matrix" else (1 if unet_mode == "scalar_preblock" else d))
        total = wte_params + ve_params + ngram_params + attn_params + mlp_params + unet_params
        return round(total / 1e6, 2)

    seed_dim = int(arch.get("seed_dim", 128))
    readout_mode = arch.get("readout_mode", "tied_stiefel")
    two_beam = bool(arch.get("overcomplete_two_beam", False))
    one_beam_frame = bool(arch.get("overcomplete_frame", False))
    rho = float(arch.get("frame_expansion", 2.0))
    hidden = int(round(rho * d)) if (two_beam or one_beam_frame) else int(arch.get("hidden_dim", int(1.5 * d)))

    # Seed table + lift + readout
    if bool(arch.get("use_hierarchical_16qubit", False)):
        q_reg = int(arch.get("qubit_registers", 16))
        q_amp = int(arch.get("qubit_amplitudes", 32))
        n_hier = int(arch.get("num_higher_states", 256))
        emb_params = (
            vocab * (q_reg * q_amp) * 2
            + n_hier * (q_reg * q_amp + q_reg * n_hier + vocab)
            + q_reg * (q_amp * q_amp)
        )
    elif readout_mode == "asymmetric":
        emb_params = vocab * seed_dim + d * seed_dim + vocab * d
    elif readout_mode == "tied_vocab":
        emb_params = vocab * seed_dim + d * seed_dim + d * d
    else:
        emb_params = 2 * vocab * seed_dim + d * seed_dim + d * d

    density_layer_params = (d * d + int(arch.get("num_qubits", 64)) ** 2) if bool(arch.get("use_sparse_density_layer", False)) else 0
    ve_params = num_ve * vocab * d
    attn_layers = max(0, layers - len(skip_layers))
    attn_params = attn_layers * (4 * d * d)
    beams = 2 if (two_beam or not one_beam_frame) else 1
    mlp_params = layers * ((beams + 1) * d * hidden)
    adapter = arch.get("seed_adapter", "None")
    if adapter == "2BeamAdapter":
        adap_params = 2 * d * hidden + hidden * seed_dim
    elif adapter == "1BeamAdapter":
        adap_params = d * hidden + hidden * seed_dim
    else:
        adap_params = 0
    unet_params = (layers // 2) * (d * d if unet_mode == "matrix" else d)
    total = emb_params + ve_params + ngram_params + density_layer_params + attn_params + mlp_params + adap_params + unet_params
    return round(total / 1e6, 2)


def _build_cli_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", type=str, default="cuda")
    parser.add_argument("--architecture", type=str, default="quantum_born")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--eval-every-tokens-m", type=float, default=50.0)
    parser.add_argument("--schedule", type=str, default="cosine")
    parser.add_argument("--wsd-decay-fraction", type=float, default=0.20)
    parser.add_argument("--wsd-min-lr-ratio", type=float, default=0.10)
    parser.add_argument("--seq-curriculum", type=str, default="")
    parser.add_argument("--mtp-horizons", type=int, default=1)
    parser.add_argument("--mtp-weight", type=float, default=0.25)
    parser.add_argument("--mtp-stride", type=int, default=4)
    parser.add_argument("--seed-dim", type=int, default=None)
    parser.add_argument("--stiefel-optimizer", type=str, default="standard")
    parser.add_argument("--kraus-beams", type=int, default=1)
    parser.add_argument("--fused-povm-tile-size", type=int, default=0)
    parser.add_argument("--overcomplete-frame", action="store_true")
    parser.add_argument("--overcomplete-seed-adapter", action="store_true")
    parser.add_argument("--overcomplete-two-beam", action="store_true")
    parser.add_argument("--two-beam-seed-adapter", action="store_true")
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--frame-expansion", type=float, default=None)
    parser.add_argument("--readout-mode", type=str, default="tied_stiefel")
    parser.add_argument("--depth-scaled-residual", action="store_true")
    parser.add_argument("--dynamic-povm-beta", action="store_true")
    parser.add_argument("--ortho-decouple", action="store_true")
    parser.add_argument("--povm-log-prior", action="store_true")
    parser.add_argument("--gated-spherical-residual", action="store_true")
    parser.add_argument("--purified-born-attn", action="store_true")
    parser.add_argument("--zero-init-classical", action="store_true")
    parser.add_argument("--true-layer0-v-residual", action="store_true")
    parser.add_argument("--classical-stiefel-penalty", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-value-embeddings", type=int, default=0)
    parser.add_argument("--unet-skip-mode", type=str, default="matrix")
    parser.add_argument("--fused-qkv", "--use-fused-qkv", dest="use_fused_qkv", action="store_true")
    parser.add_argument("--skip-attn-layers", type=str, default="")
    parser.add_argument("--zero-half-rope", action="store_true")
    parser.add_argument("--doc-mask-attn", action="store_true")
    parser.add_argument("--sliding-window", type=int, default=0)
    parser.add_argument("--muon-momentum-warmup", type=int, default=0)
    parser.add_argument("--direct-chunked-ce", dest="direct_chunked_ce", action="store_true", default=False)
    parser.add_argument("--no-direct-chunked-ce", dest="direct_chunked_ce", action="store_false")
    parser.add_argument("--stage-shard-on-gpu", dest="stage_shard_on_gpu", action="store_true", default=False)
    parser.add_argument("--no-stage-shard-on-gpu", dest="stage_shard_on_gpu", action="store_false")
    parser.add_argument("--optimizer-mode", type=str, default="muon_adamw")
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--leaderboard-exact-classical", action="store_true")
    parser.add_argument("--progressive-window", action="store_true")
    parser.add_argument("--ppmi-svd-init", action="store_true")
    parser.add_argument("--ppmi-svd-shards", type=int, default=9)
    parser.add_argument("--ppmi-svd-alpha", type=float, default=0.65)
    parser.add_argument("--ngram-hash", action="store_true")
    parser.add_argument("--ngram-vocab-size", type=int, default=262144)
    parser.add_argument("--polar-express", action="store_true")
    parser.add_argument("--normuon", action="store_true")
    parser.add_argument("--twin-rail-muon", action="store_true")
    parser.add_argument("--cautious-wd", action="store_true")
    parser.add_argument("--tail-ema", action="store_true")
    parser.add_argument("--mudd-and-gates", action="store_true")
    parser.add_argument("--smear-gate", action="store_true")
    parser.add_argument("--tapered-mtp", action="store_true")
    parser.add_argument("--align-to-bos", action="store_true")
    parser.add_argument("--sampled-softmax-candidates", type=int, default=0)
    parser.add_argument("--embed-step-every", type=int, default=1)
    parser.add_argument("--fp8-linear", action="store_true")
    parser.add_argument("--sparse-64qubit", action="store_true")
    parser.add_argument("--num-qubits", type=int, default=64)
    parser.add_argument("--sparse-ngram-k", type=int, default=16)
    parser.add_argument("--sparse-density-layer", action="store_true")
    parser.add_argument("--density-matrix-rank", type=int, default=4)
    parser.add_argument("--no-fock-coupler", action="store_true")
    parser.add_argument("--hierarchical-16qubit", action="store_true")
    parser.add_argument("--qubit-registers", type=int, default=16)
    parser.add_argument("--qubit-amplitudes", type=int, default=32)
    parser.add_argument("--num-higher-states", type=int, default=256)
    parser.add_argument("--no-block-hierarchy-bridge", action="store_true")
    parser.add_argument("--no-bridge-causal-memory", "--embed-only-causal-memory", dest="no_bridge_causal_memory", action="store_true")
    parser.add_argument("--last-layer-softmax", action="store_true")
    parser.add_argument("--two-beam-kboson-readout", "--two-beam-kboson", dest="two_beam_kboson_readout", action="store_true")
    parser.add_argument("--diffuse-rank", type=int, default=16)
    parser.add_argument("--logit-softcap", type=float, default=15.0)
    parser.add_argument("--val-tokens", type=int, default=10485760)
    return parser


_CLI_PARSER = _build_cli_arg_parser()


def parse_cli_args_to_card_sections(cli_args: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Parse a `11_fineweb_100m_multiepoch_gpu_tpu.py` CLI string into `(architecture, optimization, data_and_batch)`."""
    tokens = shlex.split(cli_args or "")
    ns, _ = _CLI_PARSER.parse_known_args(tokens)

    family = str(ns.architecture)
    num_layers = int(ns.num_layers)
    embed_dim = int(ns.embed_dim)
    num_heads = int(ns.num_heads)
    head_dim = embed_dim // max(1, num_heads)

    skip_attn_layers = [
        int(tok.strip())
        for tok in str(ns.skip_attn_layers).split(",")
        for _ in (0,)
        if tok.strip() and tok.strip().lstrip("-").isdigit()
    ]

    if family == "classical_modded":
        hidden_dim = int(ns.hidden_dim) if ns.hidden_dim is not None else 4 * embed_dim
        frame_expansion = round(hidden_dim / max(1, embed_dim), 2)
        seed_dim = embed_dim
        if ns.leaderboard_exact_classical:
            readout_mode = "untied_softcap_15"
        else:
            readout_mode = "untied_softcap_30" if ns.zero_init_classical else "tied_softcap_30"
        seed_adapter = "None"
        d_ff_display = f"{hidden_dim} (ReLU^2 MLP)"
    else:
        if ns.frame_expansion is not None:
            frame_expansion = float(ns.frame_expansion)
        elif ns.overcomplete_two_beam or ns.overcomplete_frame:
            frame_expansion = 2.0
        else:
            frame_expansion = 1.5

        if ns.overcomplete_two_beam or ns.overcomplete_frame:
            hidden_dim = int(round(frame_expansion * embed_dim))
        elif ns.hidden_dim is not None:
            hidden_dim = int(ns.hidden_dim)
            frame_expansion = round(hidden_dim / max(1, embed_dim), 2)
        else:
            hidden_dim = int(round(1.5 * embed_dim))

        if ns.two_beam_kboson_readout:
            readout_mode = (
                "two_beam_kboson_povm" if ns.hierarchical_16qubit else "two_beam_kboson"
            )
        elif ns.last_layer_softmax:
            readout_mode = (
                "calibrated_hierarchical_softmax" if ns.hierarchical_16qubit else "calibrated_softmax"
            )
        else:
            readout_mode = "hierarchical_povm" if ns.hierarchical_16qubit else str(ns.readout_mode)
        if ns.seed_dim is not None:
            seed_dim = int(ns.seed_dim)
        elif ns.hierarchical_16qubit:
            seed_dim = int(ns.qubit_registers) * int(ns.qubit_amplitudes)
        elif readout_mode == "asymmetric":
            seed_dim = embed_dim
        else:
            seed_dim = 128

        if ns.two_beam_seed_adapter:
            seed_adapter = "2BeamAdapter"
        elif ns.overcomplete_seed_adapter:
            seed_adapter = "1BeamAdapter"
        else:
            seed_adapter = "None"

        if ns.sparse_density_layer:
            d_ff_display = f"Density{embed_dim}x{embed_dim} (4.19M, R={int(ns.density_matrix_rank)})"
        elif ns.overcomplete_two_beam:
            rho_int = int(frame_expansion) if frame_expansion.is_integer() else frame_expansion
            d_ff_display = f"{hidden_dim}x2 ({rho_int}d-TwoBeam)"
        elif ns.overcomplete_frame:
            d_ff_display = f"{hidden_dim} (1-Beam 2d-Frame)"
        else:
            d_ff_display = f"{hidden_dim}x2 (2-Beam Bilinear)"

    architecture: dict[str, Any] = {
        "family": family,
        "num_layers": num_layers,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
        "d_ff_display": d_ff_display,
        "frame_expansion": frame_expansion,
        "seed_dim": seed_dim,
        "params_M": 0.0,
        "readout_mode": readout_mode,
        "seed_adapter": seed_adapter,
        "zero_init_residual_and_head": bool(ns.zero_init_classical or ns.leaderboard_exact_classical),
        "depth_scaled_residual": bool(ns.depth_scaled_residual) and not bool(ns.leaderboard_exact_classical),
        "true_layer0_v_residual": bool(ns.true_layer0_v_residual) and not bool(ns.leaderboard_exact_classical),
        "num_value_embeddings": int(ns.num_value_embeddings),
        "unet_skip_mode": str(ns.unet_skip_mode),
        "use_fused_qkv": bool(ns.use_fused_qkv),
        "skip_attn_layers": skip_attn_layers,
        "zero_half_rope": bool(ns.zero_half_rope or ns.leaderboard_exact_classical),
        "doc_mask_attn": bool(ns.doc_mask_attn),
        "sliding_window": int(ns.sliding_window),
        "dynamic_povm_beta": bool(ns.dynamic_povm_beta),
        "ortho_decouple": bool(ns.ortho_decouple),
        "povm_log_prior": bool(ns.povm_log_prior),
        "gated_spherical_residual": bool(ns.gated_spherical_residual),
        "purified_born_attn": bool(ns.purified_born_attn),
        "overcomplete_two_beam": bool(ns.overcomplete_two_beam),
        "overcomplete_frame": bool(ns.overcomplete_frame or ns.overcomplete_two_beam),
        "kraus_beams": int(ns.kraus_beams),
        "mtp_horizons": int(ns.mtp_horizons),
        "ppmi_svd_init": bool(ns.ppmi_svd_init),
        "ppmi_svd_shards": int(ns.ppmi_svd_shards),
        "ppmi_svd_alpha": float(ns.ppmi_svd_alpha),
        "use_ngram_hash": bool(ns.ngram_hash),
        "ngram_vocab_size": int(ns.ngram_vocab_size),
        "use_mudd_and_gates": bool(ns.mudd_and_gates),
        "use_smear_gate": bool(ns.smear_gate),
        "fp8_linear": bool(ns.fp8_linear),
        "use_sparse_64qubit": bool(ns.sparse_64qubit),
        "num_qubits": int(ns.num_qubits),
        "sparse_ngram_k": int(ns.sparse_ngram_k),
        "use_sparse_density_layer": bool(ns.sparse_density_layer),
        "density_matrix_rank": int(ns.density_matrix_rank),
        "use_fock_coupler": not bool(ns.no_fock_coupler),
        "use_hierarchical_16qubit": bool(ns.hierarchical_16qubit),
        "qubit_registers": int(ns.qubit_registers),
        "qubit_amplitudes": int(ns.qubit_amplitudes),
        "num_higher_states": int(ns.num_higher_states),
        "use_block_hierarchy_bridge": not bool(ns.no_block_hierarchy_bridge),
        "bridge_causal_memory": not bool(ns.no_bridge_causal_memory),
        "use_last_layer_softmax": bool(ns.last_layer_softmax),
        "use_two_beam_kboson_readout": bool(ns.two_beam_kboson_readout),
        "diffuse_rank": int(ns.diffuse_rank),
        "logit_softcap": float(ns.logit_softcap),
    }

    opt_mode = str(ns.optimizer_mode)
    if ns.stiefel_optimizer == "tangent" and opt_mode == "muon_adamw":
        opt_mode = "tangent_stiefel_muon"

    zero_init = bool(ns.zero_init_classical or ns.leaderboard_exact_classical)
    muon_lr_val = (
        float(ns.muon_lr)
        if ns.muon_lr is not None
        else (0.003 if opt_mode == "pure_adamw" else (0.05 if opt_mode == "sgd_nesterov" else 0.025))
    )
    optimization: dict[str, Any] = {
        "optimizer_mode": opt_mode,
        "schedule": str(ns.schedule).lower(),
        "wsd_decay_fraction": float(ns.wsd_decay_fraction),
        "wsd_min_lr_ratio": float(ns.wsd_min_lr_ratio),
        "muon_lr": muon_lr_val,
        "muon_momentum": 0.95,
        "muon_momentum_warmup_steps": int(ns.muon_momentum_warmup),
        "batched_newton_schulz": opt_mode not in ("pure_adamw", "sgd_nesterov"),
        "polar_method": "polar_express" if ns.polar_express else "newton_schulz",
        "use_normuon": bool(ns.normuon),
        "use_twin_rail": bool(ns.twin_rail_muon),
        "cautious_wd": bool(ns.cautious_wd),
        "tail_ema": bool(ns.tail_ema),
        "tapered_mtp": bool(ns.tapered_mtp),
        "sampled_softmax_candidates": int(ns.sampled_softmax_candidates),
        "embed_step_every": int(ns.embed_step_every),
        "stiefel_penalty": float(ns.classical_stiefel_penalty) if family == "classical_modded" else 0.01,
        "adamw_lr_wte": 0.20 if (family == "classical_modded" and zero_init) else 0.004,
        "adamw_lr_head": 0.008 if ((family == "classical_modded" and zero_init) or bool(ns.last_layer_softmax)) else 0.004,
        "adamw_lr_scalar": 0.02 if (family == "classical_modded" and zero_init) else 0.004,
        "direct_chunked_ce": bool(ns.direct_chunked_ce),
        "stage_shard_on_gpu": bool(ns.stage_shard_on_gpu),
    }

    if ns.leaderboard_exact_classical:
        architecture["leaderboard_exact_classical"] = True
        architecture["logit_softcap"] = 15.0
        architecture["attn_scale"] = 0.12
        architecture["rope_base"] = 1024.0
        architecture["unet_skip_mode"] = "scalar_preblock"
        architecture["skip_attn_layers"] = skip_attn_layers if skip_attn_layers else [7]
        architecture["num_value_embeddings"] = int(ns.num_value_embeddings) if int(ns.num_value_embeddings) > 0 else 3
        architecture["use_fused_qkv"] = True
        optimization["ema_momentum"] = True
        optimization["weight_decay"] = 0.0
        optimization["muon_lr"] = float(ns.muon_lr) if "--muon-lr" in cli_args and ns.muon_lr is not None else 0.05
        optimization["adam_wte_lr"] = 0.6
        optimization["adam_scalar_lr"] = 0.04
        optimization["adam_head_lr"] = 0.008
        optimization["adamw_lr_wte"] = 0.6
        optimization["adamw_lr_scalar"] = 0.04
        optimization["adamw_lr_head"] = 0.008
        optimization["warmup_steps"] = 0

    if ns.progressive_window:
        architecture["progressive_window"] = True
        architecture["progressive_window_schedule"] = "128->1792 (long/short)"
        if int(architecture.get("sliding_window", 0)) <= 0:
            architecture["sliding_window"] = 128

    architecture["params_M"] = _estimate_params_m(architecture)
    architecture["params_M_est"] = architecture["params_M"]

    seq_len_train = int(ns.seq_len)
    batch_size = int(ns.batch_size)
    grad_accum = int(ns.grad_accum)
    tokens_per_step = batch_size * seq_len_train * grad_accum
    num_shards = int(ns.num_shards)
    epochs = int(ns.epochs)
    target_tokens_m = round(float(num_shards * epochs * 100.0), 1)

    data_and_batch: dict[str, Any] = {
        "dataset": "FineWeb10B (GPT-2 Vocab 50,304 / Active Val Vocab 50,257)",
        "num_shards": num_shards,
        "epochs": epochs,
        "target_tokens_M": target_tokens_m,
        "seq_len_train": seq_len_train,
        "seq_len_val": seq_len_train,
        "val_tokens": int(ns.val_tokens),
        "val_vocab_size": 50257,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "tokens_per_step": tokens_per_step,
        "align_to_bos": bool(ns.align_to_bos),
    }
    return architecture, optimization, data_and_batch


def _infer_legacy_cli_args(entry: dict[str, Any]) -> str:
    """Synthesize representative CLI arguments for historical Wave 1–5 `FLEET_SPECS` entries."""
    key = str(entry.get("key", ""))
    label = str(entry.get("label", ""))
    kind = str(entry.get("kind", "gpu"))
    backend = "tpu" if kind == "tpu" else "cuda"
    parts = [f"--backend {backend}"]

    if "classical" in key or "Classical" in label:
        parts.append("--architecture classical_modded")
    else:
        parts.append("--architecture quantum_born")

    if any(p in key for p in ("w4_900m_", "w5_900m_", "w6_900m_", "w7_900m_", "w8_900m_", "w9_900m_", "w10_900m_", "w11_900m_", "w12_900m_", "w13_900m_")) or "900M" in label:
        parts.append("--epochs 1 --num-shards 9 --eval-every-tokens-m 50.0")
    elif "5ep" in key or "5-Epoch" in label or "500M" in label:
        parts.append("--epochs 5 --num-shards 1")
    else:
        parts.append("--epochs 3 --num-shards 1")

    if "wsd" in key.lower() or "WSD" in label:
        parts.append("--schedule wsd")
    else:
        parts.append("--schedule cosine")

    m_seed = re.search(r"(?:D_seed=|seed|D)(\d+)", label + " " + key)
    if m_seed:
        parts.append(f"--seed-dim {m_seed.group(1)}")

    m_layers = re.search(r"\bL=(\d+)", label)
    if m_layers:
        parts.append(f"--num-layers {m_layers.group(1)}")
    elif "deep12" in key or "12L" in key:
        parts.append("--num-layers 12")
    elif "deep9" in key:
        parts.append("--num-layers 9")
    elif "tied10" in key:
        parts.append("--num-layers 10")

    m_dim = re.search(r"\bd=(\d+)", label)
    if m_dim:
        parts.append(f"--embed-dim {m_dim.group(1)}")
    elif "wide1536" in key:
        parts.append("--embed-dim 1536 --num-heads 24")

    if "4096x2" in label or "frame4d" in key:
        parts.append("--overcomplete-two-beam --frame-expansion 4.0")
    elif "3072x2" in label or "frame3d" in key:
        parts.append("--overcomplete-two-beam --frame-expansion 3.0")
    elif "2048x2" in label or "OvercompleteTwoBeam" in label or "2beam_frame" in key:
        parts.append("--overcomplete-two-beam --frame-expansion 2.0")
    elif "Overcomplete 2d Frame" in label or "frame256" in key or "frame_wsd" in key:
        parts.append("--overcomplete-frame --frame-expansion 2.0")

    if "2BeamAdapter" in label or "TwoBeamSeedAdapter" in label or "full_synergy" in key or "2beam_seed" in key:
        parts.append("--two-beam-seed-adapter")
    elif "SeedAdapter" in label or "adapter" in key:
        parts.append("--overcomplete-seed-adapter")

    if "Asymmetric" in label or "asym1024" in key:
        parts.append("--readout-mode asymmetric")
    elif "tied_vocab" in label or "tied10" in key:
        parts.append("--readout-mode tied_vocab")

    if "DepthRes" in label or "Depth-Scaled" in label:
        parts.append("--depth-scaled-residual")
    if "DynamicBeta" in label or "dynbeta" in key:
        parts.append("--dynamic-povm-beta")
    if "EarlyFix" in label or "earlyfix" in key:
        parts.append("--ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn")
    if "ZeroInit" in label or "zeroinit" in key:
        parts.append("--zero-init-classical")
    if "Layer0-V" in label or "layer0v" in key:
        parts.append("--true-layer0-v-residual")
    if "Tangent" in label:
        parts.append("--stiefel-optimizer tangent")
    if "MTP=3" in label:
        parts.append("--mtp-horizons 3")
    if "K=4" in label:
        parts.append("--kraus-beams 4")

    return " ".join(parts)


@dataclass
class RunModelCard:
    """Validated, JSON-serializable Model Card representing a single QuantumDLBlocks training run."""

    schema_version: str
    run_key: str
    wave_id: str
    wave_label: str
    plan_id: str
    label: str
    short_label: str
    color: str
    default_visible: bool
    created_at: str
    updated_at: str
    hypothesis_summary: str
    wave_num: float | int = 1
    run_tier: str = "main_benchmark"
    pareto_frontier: bool = False
    obsolete: bool = False
    obsolete_reason: str = ""
    obsolete_at: str = ""
    architecture: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    data_and_batch: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunModelCard:
        wid = str(data.get("wave_id", "W1"))
        _, default_wnum = infer_wave_id_and_num(str(data["run_key"]), wid)
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            run_key=str(data["run_key"]),
            wave_id=wid,
            wave_label=str(data.get("wave_label", "Wave 1")),
            plan_id=str(data.get("plan_id", "")),
            label=str(data.get("label", data["run_key"])),
            short_label=str(data.get("short_label", data.get("label", data["run_key"]))),
            color=str(data.get("color", deterministic_run_color(str(data["run_key"])))),
            default_visible=bool(data.get("default_visible", True)),
            created_at=str(data.get("created_at", datetime.now().isoformat(timespec="seconds"))),
            updated_at=str(data.get("updated_at", datetime.now().isoformat(timespec="seconds"))),
            hypothesis_summary=str(data.get("hypothesis_summary", "")),
            wave_num=data.get("wave_num", default_wnum),
            run_tier=str(data.get("run_tier", "main_benchmark")),
            pareto_frontier=bool(data.get("pareto_frontier", False)),
            obsolete=bool(data.get("obsolete", False)),
            obsolete_reason=str(data.get("obsolete_reason", "")),
            obsolete_at=str(data.get("obsolete_at", "")),
            architecture=dict(data.get("architecture", {})),
            optimization=dict(data.get("optimization", {})),
            data_and_batch=dict(data.get("data_and_batch", {})),
            hardware=dict(data.get("hardware", {})),
            telemetry=dict(data.get("telemetry", {})),
        )


def set_model_card_obsolete(
    card: dict[str, Any],
    obsolete: bool,
    reason: str = "",
) -> dict[str, Any]:
    """Return a copy of `card` with updated `obsolete`, `obsolete_reason`, and `obsolete_at` fields."""
    updated = json.loads(json.dumps(card))
    is_obs = bool(obsolete)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated["obsolete"] = is_obs
    updated["obsolete_reason"] = str(reason) if is_obs else ""
    updated["obsolete_at"] = now_utc if is_obs else ""
    updated["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return updated


def validate_model_card(card: dict[str, Any] | RunModelCard) -> list[str]:
    """Validate a Model Card dictionary against the Experiment Plan 017/018 v1.0 schema.

    Returns a list of human-readable error strings (empty list `[]` if valid).
    """
    d = card.to_dict() if isinstance(card, RunModelCard) else card
    errors: list[str] = []
    if not isinstance(d, dict):
        return ["Model card must be a dictionary."]

    required_top = (
        "schema_version",
        "run_key",
        "wave_id",
        "wave_label",
        "plan_id",
        "label",
        "short_label",
        "color",
        "default_visible",
        "created_at",
        "updated_at",
        "hypothesis_summary",
        "architecture",
        "optimization",
        "data_and_batch",
        "hardware",
        "telemetry",
    )
    for k in required_top:
        if k not in d:
            errors.append(f"Missing top-level field: {k!r}")

    if "obsolete" in d and not isinstance(d.get("obsolete"), bool):
        errors.append("Field 'obsolete' must be a bool.")
    if "obsolete_reason" in d and not isinstance(d.get("obsolete_reason"), str):
        errors.append("Field 'obsolete_reason' must be a str.")
    if "obsolete_at" in d and not isinstance(d.get("obsolete_at"), str):
        errors.append("Field 'obsolete_at' must be a str.")

    if not d.get("run_key") or not isinstance(d.get("run_key"), str):
        errors.append("Field 'run_key' must be a non-empty string.")
    if not isinstance(d.get("color"), str) or not str(d.get("color", "")).startswith("#"):
        errors.append("Field 'color' must be a hex color string starting with '#'.")

    arch = d.get("architecture")
    if not isinstance(arch, dict):
        errors.append("Section 'architecture' must be a dict.")
    else:
        for req_a in (
            "family",
            "num_layers",
            "embed_dim",
            "num_heads",
            "head_dim",
            "hidden_dim",
            "d_ff_display",
            "frame_expansion",
            "seed_dim",
            "params_M",
            "readout_mode",
            "seed_adapter",
            "num_value_embeddings",
            "unet_skip_mode",
            "use_fused_qkv",
            "skip_attn_layers",
            "zero_half_rope",
            "doc_mask_attn",
            "sliding_window",
        ):
            if req_a not in arch:
                errors.append(f"Missing architecture field: {req_a!r}")

    opt = d.get("optimization")
    if not isinstance(opt, dict):
        errors.append("Section 'optimization' must be a dict.")
    else:
        for req_o in (
            "optimizer_mode",
            "schedule",
            "wsd_decay_fraction",
            "wsd_min_lr_ratio",
            "muon_lr",
            "muon_momentum",
            "muon_momentum_warmup_steps",
            "direct_chunked_ce",
            "stage_shard_on_gpu",
        ):
            if req_o not in opt:
                errors.append(f"Missing optimization field: {req_o!r}")

    dab = d.get("data_and_batch")
    if not isinstance(dab, dict):
        errors.append("Section 'data_and_batch' must be a dict.")
    else:
        for req_b in (
            "dataset",
            "num_shards",
            "epochs",
            "target_tokens_M",
            "seq_len_train",
            "seq_len_val",
            "batch_size",
            "grad_accum",
            "tokens_per_step",
        ):
            if req_b not in dab:
                errors.append(f"Missing data_and_batch field: {req_b!r}")
        if dab.get("seq_len_val") != 512:
            errors.append(f"Validation sequence length 'seq_len_val' must be 512, got {dab.get('seq_len_val')!r}")

    hw = d.get("hardware")
    if not isinstance(hw, dict):
        errors.append("Section 'hardware' must be a dict.")
    else:
        for req_h in ("session", "kind", "accelerator", "peak_bf16_tflops", "remote_log", "remote_json", "cli_args"):
            if req_h not in hw:
                errors.append(f"Missing hardware field: {req_h!r}")

    tel = d.get("telemetry")
    if not isinstance(tel, dict):
        errors.append("Section 'telemetry' must be a dict.")
    else:
        for req_t in (
            "run_state",
            "completed",
            "current_step",
            "total_steps",
            "live_tokens_M",
            "progress_pct",
            "avg_tok_per_s",
            "wall_min",
            "best_val_ce",
            "milestones_val_ce",
            "quantum_observables",
        ):
            if req_t not in tel:
                errors.append(f"Missing telemetry field: {req_t!r}")

    return errors


def _infer_hardware_details(entry: dict[str, Any]) -> tuple[str, str, float, float]:
    """Infer `(kind, accelerator_name, peak_bf16_tflops, default_tps)` from a manifest or FLEET_SPECS entry."""
    session = str(entry.get("session") or entry.get("sess") or "")
    raw_kind = str(entry.get("kind", "")).lower()
    tpu_var = entry.get("tpu_variant")
    gpu_var = str(entry.get("gpu_variant", "")).upper()
    hw_str = str(entry.get("hardware") or entry.get("accel") or entry.get("hw") or "")

    if raw_kind == "tpu" or tpu_var or session.startswith("tpu"):
        kind = "tpu"
        accel = hw_str or "Cloud TPU v5e-1"
        peak = float(entry.get("peak_tflops", TPU_V5E_BF16_PEAK_TFLOPS))
        tps = float(entry.get("default_tps", 120000.0))
        return kind, accel, peak, tps

    kind = "gpu"
    if "A100" in hw_str.upper() or gpu_var == "A100" or "a100" in session.lower():
        accel = hw_str or "NVIDIA A100-SXM4 (80GB)"
        peak = float(entry.get("peak_tflops", A100_BF16_PEAK_TFLOPS))
        tps = float(entry.get("default_tps", 165000.0))
    elif "L4" in hw_str.upper() or gpu_var == "L4":
        accel = hw_str or "NVIDIA L4 (24GB)"
        peak = float(entry.get("peak_tflops", L4_BF16_PEAK_TFLOPS))
        tps = float(entry.get("default_tps", 65000.0))
    else:
        accel = hw_str or "G4 (RTX PRO 6000 Blackwell)"
        peak = float(entry.get("peak_tflops", BLACKWELL_RTX6000_BF16_PEAK_TFLOPS))
        tps = float(entry.get("default_tps", 225000.0))
    return kind, accel, peak, tps


def build_model_card_from_manifest_entry(
    entry: dict[str, Any],
    wave_id: str = "",
    wave_description: str = "",
    plan_id: str = "",
) -> dict[str, Any]:
    """Construct a complete, validated Model Card dict from a `configs/wave*_fleet.json` or `FLEET_SPECS` entry."""
    run_key = str(entry.get("key") or entry["run_key"])
    norm_wave_id, wave_num = infer_wave_id_and_num(
        run_key, wave_id or str(entry.get("wave") or entry.get("wave_id") or "")
    )

    if wave_description:
        wave_label = wave_description.split(":")[0].strip() if len(wave_description) > 72 else wave_description
        if not wave_label.lower().startswith("wave") and norm_wave_id.startswith("W"):
            wave_label = f"Wave {wave_num} ({wave_label})"
    else:
        wave_label = str(entry.get("wave_label") or WAVE_DEFAULT_LABELS.get(norm_wave_id, f"Wave {norm_wave_id.lstrip('W')}"))

    if not plan_id:
        m_plan = re.search(r"Plan\s+(\d+)", wave_description + " " + str(entry.get("label", "")), flags=re.IGNORECASE)
        if m_plan:
            plan_id = f"EXPERIMENT_PLAN_{int(m_plan.group(1)):03d}"
        else:
            plan_id = WAVE_DEFAULT_PLANS.get(norm_wave_id, f"EXPERIMENT_PLAN_{max(1, wave_num + 7):03d}")

    cli_args = str(entry.get("cli_args") or _infer_legacy_cli_args(entry))
    architecture, optimization, data_and_batch = parse_cli_args_to_card_sections(cli_args)
    if entry.get("target_tokens_M") is not None:
        data_and_batch["target_tokens_M"] = float(entry["target_tokens_M"])
    if entry.get("dataset"):
        data_and_batch["dataset"] = str(entry["dataset"])

    kind, accel, peak_tflops, _ = _infer_hardware_details(entry)
    session = str(entry.get("session") or entry.get("sess") or run_key)
    remote_log = str(entry.get("remote_log") or entry.get("log") or f"/content/{run_key}.log")
    remote_json = str(entry.get("remote_json") or entry.get("json") or f"/content/{run_key}_results.json")

    label = str(
        entry.get("label")
        or entry.get("title")
        or (
            f"{norm_wave_id} {kind.upper()} {int(data_and_batch['target_tokens_M'])}M "
            f"{architecture['family']} (L={architecture['num_layers']}, d={architecture['embed_dim']}, "
            f"D_seed={architecture['seed_dim']}, {architecture['d_ff_display']}, {optimization['schedule']})"
        )
    )
    raw_short = str(entry.get("short_label") or "")
    if not raw_short:
        raw_short = label if len(label) <= 56 else f"{label[:53]}..."
    short_label = raw_short if raw_short.startswith("[") else f"[{session}] {raw_short}"

    color = str(entry.get("color") or deterministic_run_color(run_key))
    default_visible = bool(entry.get("default_visible", wave_num >= 3 or norm_wave_id == "Baseline"))

    hyp_summary = str(entry.get("hypothesis_summary") or "")
    if not hyp_summary:
        if architecture["family"] == "classical_modded":
            hyp_summary = (
                f"Evaluates ClassicalModdedGPT (L={architecture['num_layers']}, d={architecture['embed_dim']}, "
                f"d_ff={architecture['hidden_dim']}, VE={architecture['num_value_embeddings']}, "
                f"UNet={architecture['unet_skip_mode']}, opt={optimization['optimizer_mode']}, "
                f"sched={optimization['schedule']}) on {data_and_batch['target_tokens_M']:.0f}M FineWeb tokens "
                f"(T_train={data_and_batch['seq_len_train']}, T_val=512, B={data_and_batch['tokens_per_step']:,} tok/step)."
            )
        else:
            hyp_summary = (
                f"Evaluates QuantumBornGPT (L={architecture['num_layers']}, d={architecture['embed_dim']}, "
                f"D_seed={architecture['seed_dim']}, Frame={architecture['d_ff_display']}, "
                f"Readout={architecture['readout_mode']}, Adapter={architecture['seed_adapter']}, "
                f"VE={architecture['num_value_embeddings']}, sched={optimization['schedule']}) on "
                f"{data_and_batch['target_tokens_M']:.0f}M FineWeb tokens at fixed T_val=512."
            )

    total_steps_est = int(
        entry.get("total_steps")
        or max(
            1,
            int(round((data_and_batch["target_tokens_M"] * 1e6) / max(1, data_and_batch["tokens_per_step"]))),
        )
    )
    now_iso = datetime.now().isoformat(timespec="seconds")
    is_obsolete = bool(entry.get("obsolete", False))
    obsolete_reason = str(entry.get("obsolete_reason", "")) if is_obsolete else ""
    obsolete_at = str(entry.get("obsolete_at", now_iso if is_obsolete else "")) if is_obsolete else ""

    if entry.get("run_tier"):
        run_tier = str(entry["run_tier"])
    elif norm_wave_id in ("W14", "W15") or run_key.lower().startswith(("w14_local_", "w15_local_")):
        run_tier = "main_benchmark"
    elif (
        norm_wave_id in ("VERIFY", "MICRO")
        or run_key.lower().startswith(("verify", "micro_", "plan0", "h1_", "h2_", "h3_"))
        or (0.0 < float(data_and_batch["target_tokens_M"]) < 50.0)
    ):
        run_tier = "micro_verification"
    else:
        run_tier = "main_benchmark"

    card = RunModelCard(
        schema_version="1.0",
        run_key=run_key,
        wave_id=norm_wave_id,
        wave_label=wave_label,
        plan_id=plan_id,
        label=label,
        short_label=short_label,
        color=color,
        default_visible=default_visible,
        created_at=str(entry.get("created_at") or now_iso),
        updated_at=now_iso,
        hypothesis_summary=hyp_summary,
        wave_num=wave_num,
        run_tier=run_tier,
        pareto_frontier=bool(entry.get("pareto_frontier", False)),
        obsolete=is_obsolete,
        obsolete_reason=obsolete_reason,
        obsolete_at=obsolete_at,
        architecture=architecture,
        optimization=optimization,
        data_and_batch=data_and_batch,
        hardware={
            "session": session,
            "kind": kind,
            "accelerator": accel,
            "peak_bf16_tflops": peak_tflops,
            "remote_log": remote_log,
            "remote_json": remote_json,
            "cli_args": cli_args,
        },
        telemetry={
            "run_state": "IDLE",
            "completed": False,
            "current_step": 0,
            "total_steps": total_steps_est,
            "live_tokens_M": 0.0,
            "progress_pct": 0.0,
            "avg_tok_per_s": 0.0,
            "wall_min": 0.0,
            "best_val_ce": None,
            "best_val_ppl": None,
            "best_top1": None,
            "best_top5": None,
            "milestones_val_ce": {
                "50M": None,
                "100M": None,
                "200M": None,
                "450M": None,
                "600M": None,
                "900M": None,
            },
            "quantum_observables": {
                "seed_eff_rank": None,
                "entanglement_nats": None,
                "iso_err": None,
                "povm_beta": None,
                "gamma_h": None,
            },
        },
    )
    return card.to_dict()


def update_model_card_telemetry(card: dict[str, Any], parsed_log: dict[str, Any]) -> dict[str, Any]:
    """Merge live parsed log metrics and evaluation checkpoints into a Model Card dict."""
    updated = json.loads(json.dumps(card))
    updated["updated_at"] = datetime.now().isoformat(timespec="seconds")

    arch = updated.setdefault("architecture", {})
    if parsed_log.get("params_M") and float(parsed_log["params_M"]) > 0 and float(parsed_log["params_M"]) != 70.68:
        arch["params_M"] = float(parsed_log["params_M"])

    raw_evals = parsed_log.get("evals") or parsed_log.get("eval_checkpoints") or []
    evals = [e for e in raw_evals if float(e.get("epoch", 0)) > 0 or float(e.get("tokens_M", 0)) > 0]
    milestones: dict[str, float | None] = {
        "50M": None,
        "100M": None,
        "200M": None,
        "450M": None,
        "600M": None,
        "900M": None,
    }
    milestone_targets = [
        ("50M", 50.0, 12.0),
        ("100M", 100.0, 15.0),
        ("200M", 200.0, 18.0),
        ("450M", 450.0, 22.0),
        ("600M", 600.0, 22.0),
        ("900M", 900.0, 25.0),
    ]
    for m_key, m_tok, tol in milestone_targets:
        candidates = [e for e in evals if abs(float(e.get("tokens_M", 0.0)) - m_tok) <= tol]
        if candidates:
            nearest = min(candidates, key=lambda e: abs(float(e.get("tokens_M", 0.0)) - m_tok))
            milestones[m_key] = float(nearest["val_ce"])

    last_ev = evals[-1] if evals else None
    q_tel = parsed_log.get("quantum_telemetry") or parsed_log.get("quantum_observables") or {}
    q_obs = {
        "seed_eff_rank": (
            parsed_log.get("final_r_eff")
            if parsed_log.get("final_r_eff") is not None
            else (q_tel.get("seed_eff_rank") if q_tel.get("seed_eff_rank") is not None else (last_ev.get("seed_eff_rank") if last_ev else None))
        ),
        "entanglement_nats": (
            q_tel.get("entanglement_nats")
            if q_tel.get("entanglement_nats") is not None
            else (last_ev.get("entanglement_nats") if last_ev else None)
        ),
        "iso_err": (
            parsed_log.get("final_iso_err")
            if parsed_log.get("final_iso_err") is not None
            else (q_tel.get("iso_err") if q_tel.get("iso_err") is not None else (last_ev.get("iso_err") if last_ev else None))
        ),
        "povm_beta": (
            q_tel.get("povm_beta")
            if q_tel.get("povm_beta") is not None
            else (last_ev.get("povm_beta") if last_ev else None)
        ),
        "gamma_h": (
            q_tel.get("gamma_h")
            if q_tel.get("gamma_h") is not None
            else (last_ev.get("gamma_h") if last_ev else None)
        ),
    }

    tel = updated.setdefault("telemetry", {})
    has_log_evidence = bool(evals or parsed_log.get("steps"))
    tel.update(
        {
            "run_state": str(parsed_log.get("run_state", tel.get("run_state", "IDLE"))),
            "completed": bool(parsed_log.get("completed", tel.get("completed", False) if not has_log_evidence else False)),
            "current_step": int(parsed_log.get("current_step", 0)) if has_log_evidence else int(tel.get("current_step", 0)),
            "total_steps": int(parsed_log.get("total_steps", tel.get("total_steps", 0))),
            "live_tokens_M": float(parsed_log.get("live_tokens_M", 0.0)) if has_log_evidence else float(tel.get("live_tokens_M", 0.0)),
            "progress_pct": float(parsed_log.get("progress_pct", 0.0)) if has_log_evidence else float(tel.get("progress_pct", 0.0)),
            "avg_tok_per_s": float(parsed_log.get("avg_tok_per_s", 0.0)) if has_log_evidence else float(tel.get("avg_tok_per_s", 0.0)),
            "wall_min": float(parsed_log.get("wall_min", 0.0)) if has_log_evidence else float(tel.get("wall_min", 0.0)),
            "best_val_ce": parsed_log.get("best_val_ce"),
            "best_val_ppl": parsed_log.get("best_val_ppl"),
            "best_top1": parsed_log.get("best_top1"),
            "best_top5": parsed_log.get("best_top5"),
            "milestones_val_ce": milestones if has_log_evidence else tel.get("milestones_val_ce", milestones),
            "quantum_observables": q_obs if has_log_evidence else tel.get("quantum_observables", q_obs),
        }
    )
    updated["status"] = tel["run_state"]
    return updated


def save_model_card_atomic(card: dict[str, Any], model_cards_dir: Path = DEFAULT_MODEL_CARDS_DIR) -> Path:
    """Atomically write a Model Card JSON file to `model_cards_dir / <run_key>.json` via `.tmp` rename."""
    model_cards_dir.mkdir(parents=True, exist_ok=True)
    run_key = str(card["run_key"])
    out_path = model_cards_dir / f"{run_key}.json"
    tmp_path = model_cards_dir / f".{run_key}.json.tmp"
    tmp_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def render_model_card_markdown(card: dict[str, Any]) -> str:
    """Render a `RunModelCard` dict into a 5-section GitHub-Flavored Markdown document."""
    arch = card.get("architecture", {})
    opt = card.get("optimization", {})
    dab = card.get("data_and_batch", {})
    hw = card.get("hardware", {})
    tel = card.get("telemetry", {})
    ms = tel.get("milestones_val_ce", {})
    q_obs = tel.get("quantum_observables", {})

    best_ce = tel.get("best_val_ce")
    best_ce_str = f"{best_ce:.4f} (Δ vs 3.280 Speedrun: {best_ce - 3.280:+.4f})" if isinstance(best_ce, (int, float)) else "—"
    best_ppl_str = f"{tel['best_val_ppl']:.2f}" if isinstance(tel.get("best_val_ppl"), (int, float)) else "—"
    top1_str = f"{tel['best_top1']:.2f}%" if isinstance(tel.get("best_top1"), (int, float)) else "—"
    top5_str = f"{tel['best_top5']:.2f}%" if isinstance(tel.get("best_top5"), (int, float)) else "—"

    ms_cols = ["50M", "100M", "200M", "450M", "600M", "900M"]
    ms_vals = [f"{ms[k]:.4f}" if isinstance(ms.get(k), (int, float)) else "—" for k in ms_cols]

    lines = [
        f"# 🃏 Model Card: `{card.get('run_key')}`",
        "",
        f"- **Label**: {card.get('label')}",
        f"- **Wave**: `{card.get('wave_id')}` — {card.get('wave_label')} (`{card.get('plan_id')}`)",
        f"- **Status**: `{tel.get('run_state', 'IDLE')}` (Step `{tel.get('current_step', 0)}/{tel.get('total_steps', 0)}`, `{tel.get('live_tokens_M', 0.0):.1f}M / {dab.get('target_tokens_M', 900.0):.0f}M` tok, `{tel.get('progress_pct', 0.0):.1f}%`)",
        "",
        "## 1. Hypothesis & Milestone Summary",
        f"> {card.get('hypothesis_summary', '')}",
        "",
        f"- **Best Val CE (T_val=512)**: **{best_ce_str}** | **Val PPL**: `{best_ppl_str}` | **Top-1 / Top-5**: `{top1_str}` / `{top5_str}` | **Speed**: `{tel.get('avg_tok_per_s', 0.0):,.0f} tok/s` (`{tel.get('wall_min', 0.0):.2f} min`)",
        "",
        "| " + " | ".join(f"{c} Val CE" for c in ms_cols) + " |",
        "| " + " | ".join(":---:" for _ in ms_cols) + " |",
        "| " + " | ".join(ms_vals) + " |",
        "",
        "## 2. Architecture & Geometric Primitives",
        f"- **Architecture Family**: `{arch.get('family')}` (`{arch.get('params_M')}M` params)",
        f"- **Backbone Shape**: `L={arch.get('num_layers')}`, `d={arch.get('embed_dim')}`, `H={arch.get('num_heads')}` (`d_h={arch.get('head_dim')}`), `d_ff={arch.get('d_ff_display')}` (`rho={arch.get('frame_expansion')}`)",
        f"- **Readout & Seed Space**: `D_seed={arch.get('seed_dim')}`, `readout_mode={arch.get('readout_mode')}`, `seed_adapter={arch.get('seed_adapter')}`",
        f"- **Speedrun V2 Primitives**: `VE={arch.get('num_value_embeddings')}`, `unet_skip={arch.get('unet_skip_mode')}`, `fused_qkv={arch.get('use_fused_qkv')}`, `skip_attn_layers={arch.get('skip_attn_layers')}`, `zero_half_rope={arch.get('zero_half_rope')}`, `doc_mask={arch.get('doc_mask_attn')}`, `sliding_window={arch.get('sliding_window')}`",
        f"- **Quantum EarlyFix Flags**: `dynamic_beta={arch.get('dynamic_povm_beta')}`, `ortho_decouple={arch.get('ortho_decouple')}`, `povm_log_prior={arch.get('povm_log_prior')}`, `gated_spherical_res={arch.get('gated_spherical_residual')}`, `purified_born_attn={arch.get('purified_born_attn')}`, `layer0_v_res={arch.get('true_layer0_v_residual')}`",
        "",
        "## 3. Optimizer, Schedule & Batch Configuration",
        f"- **Optimizer Mode**: `{opt.get('optimizer_mode')}` (`Muon LR={opt.get('muon_lr')}`, `momentum={opt.get('muon_momentum')}`, `warmup={opt.get('muon_momentum_warmup_steps')} steps`, `batched_ns={opt.get('batched_newton_schulz')}`)",
        f"- **Schedule**: `{opt.get('schedule')}` (`wsd_decay={opt.get('wsd_decay_fraction')}`, `min_lr_ratio={opt.get('wsd_min_lr_ratio')}`)",
        f"- **AdamW Group LRs**: `lr_wte={opt.get('adamw_lr_wte')}`, `lr_head={opt.get('adamw_lr_head')}`, `lr_scalar={opt.get('adamw_lr_scalar')}`",
        f"- **Batch & Context**: `{dab.get('dataset')}` | `Shards={dab.get('num_shards')}` × `Epochs={dab.get('epochs')}` (`{dab.get('target_tokens_M')}M` tok) | `T_train={dab.get('seq_len_train')}`, `T_val={dab.get('seq_len_val')}` | `B={dab.get('batch_size')}` × `G_acc={dab.get('grad_accum')}` = **`{dab.get('tokens_per_step'):,}` tok/step**",
        f"- **Systems Flags**: `direct_chunked_ce={opt.get('direct_chunked_ce')}`, `stage_shard_on_gpu={opt.get('stage_shard_on_gpu')}`",
        "",
        "## 4. Quantum-Geometric Observables",
        f"- **Seed Effective Rank $R_{{\\text{{eff}}}}(\\rho_{{\\text{{seed}}}})$**: `{q_obs.get('seed_eff_rank') if q_obs.get('seed_eff_rank') is not None else '—'}`",
        f"- **10-Qubit Bipartite Entanglement $S(\\rho_A)$**: `{q_obs.get('entanglement_nats') if q_obs.get('entanglement_nats') is not None else '—'}` nats",
        f"- **Stiefel / Frame Isometry Drift `IsoErr`**: `{q_obs.get('iso_err') if q_obs.get('iso_err') is not None else '—'}`",
        f"- **POVM Sharpness $\\beta$ / Head Gain $\\gamma_h$**: `{q_obs.get('povm_beta') if q_obs.get('povm_beta') is not None else '—'}` / `{q_obs.get('gamma_h') if q_obs.get('gamma_h') is not None else '—'}`",
        "",
        "## 5. Hardware & Reproducible CLI Command",
        f"- **Session**: `{hw.get('session')}` (`{hw.get('accelerator')}`, `{hw.get('peak_bf16_tflops')} BF16 TFLOPS`)",
        f"- **Remote Artifacts**: `{hw.get('remote_log')}` | `{hw.get('remote_json')}`",
        "",
        "```bash",
        f"PYTHONPATH=src python3 -u examples/11_fineweb_100m_multiepoch_gpu_tpu.py {hw.get('cli_args', '')} --out-json {hw.get('remote_json', '')}",
        "```",
    ]
    return "\n".join(lines) + "\n"


class DynamicWaveRegistry:
    """Zero-downtime registry that discovers runs across `FLEET_SPECS`, `configs/wave*.json`, and `.cache/model_cards/*.json`."""

    def __init__(
        self,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
        model_cards_dir: Path = DEFAULT_MODEL_CARDS_DIR,
        live_logs_dir: Path = DEFAULT_LIVE_LOGS_DIR,
        historical_specs: list[dict[str, Any]] | None = None,
        sessions_file: Path | None = None,
    ) -> None:
        self.configs_dir = Path(configs_dir)
        self.model_cards_dir = Path(model_cards_dir)
        self.live_logs_dir = Path(live_logs_dir)
        self.historical_specs = list(historical_specs or [])
        self.sessions_file = sessions_file

    def _spec_from_card(self, card: dict[str, Any], order_idx: int, from_manifest: bool) -> dict[str, Any]:
        hw = card.get("hardware", {})
        wave_id = str(card.get("wave_id", "W1"))
        _, wave_num = infer_wave_id_and_num(str(card["run_key"]), wave_id)
        return {
            "key": str(card["run_key"]),
            "session": str(hw.get("session", card["run_key"])),
            "kind": str(hw.get("kind", "gpu")),
            "accel": str(hw.get("accelerator", "G4 (RTX PRO 6000 Blackwell)")),
            "peak_tflops": float(hw.get("peak_bf16_tflops", BLACKWELL_RTX6000_BF16_PEAK_TFLOPS)),
            "default_tps": 225000.0 if hw.get("kind", "gpu") == "gpu" else 120000.0,
            "log": str(hw.get("remote_log", f"/content/{card['run_key']}.log")),
            "json": str(hw.get("remote_json", f"/content/{card['run_key']}_results.json")),
            "label": str(card.get("label", card["run_key"])),
            "short_label": str(card.get("short_label", card["run_key"])).split("] ", 1)[-1],
            "color": str(card.get("color", deterministic_run_color(str(card["run_key"])))),
            "wave_id": wave_id,
            "wave_num": wave_num,
            "wave_label": str(card.get("wave_label", WAVE_DEFAULT_LABELS.get(wave_id, f"Wave {wave_num}"))),
            "plan_id": str(card.get("plan_id", "")),
            "run_tier": str(card.get("run_tier", "main_benchmark")),
            "pareto_frontier": bool(card.get("pareto_frontier", False)),
            "cli_args": str(hw.get("cli_args", "")),
            "default_visible": bool(card.get("default_visible", True)),
            "order_idx": order_idx,
            "from_manifest": from_manifest,
            "model_card": card,
        }

    def discover_all_specs(self, persist_cards: bool = False) -> list[dict[str, Any]]:
        """Scan `historical_specs`, `configs/*.json`, and `.cache/model_cards/*.json` and return merged specs."""
        by_key: dict[str, dict[str, Any]] = {}
        order_counter = 0

        # 1. Load existing model cards on disk first so existing telemetry is preserved
        existing_cards_on_disk: dict[str, dict[str, Any]] = {}
        if self.model_cards_dir.exists():
            for mc_path in sorted(self.model_cards_dir.glob("*.json")):
                try:
                    mc_data = json.loads(mc_path.read_text(encoding="utf-8"))
                    if isinstance(mc_data, dict) and "run_key" in mc_data:
                        existing_cards_on_disk[str(mc_data["run_key"])] = mc_data
                except Exception:
                    continue

        # 2. Historical FLEET_SPECS
        for raw_spec in self.historical_specs:
            rkey = str(raw_spec["key"])
            order_counter += 1
            card = build_model_card_from_manifest_entry(raw_spec)
            if rkey in existing_cards_on_disk:
                if "telemetry" in existing_cards_on_disk[rkey]:
                    card["telemetry"] = existing_cards_on_disk[rkey]["telemetry"]
                for obs_k in ("obsolete", "obsolete_reason", "obsolete_at", "run_tier", "pareto_frontier"):
                    if obs_k in existing_cards_on_disk[rkey]:
                        card[obs_k] = existing_cards_on_disk[rkey][obs_k]
            spec = self._spec_from_card(card, order_counter, from_manifest=False)
            if "default_tps" in raw_spec:
                spec["default_tps"] = float(raw_spec["default_tps"])
            by_key[rkey] = spec

        # 3. Manifest files in configs/ (sorted by wave number then filename)
        if self.configs_dir.exists():
            manifest_files = sorted(
                self.configs_dir.glob("*.json"),
                key=lambda p: (infer_wave_id_and_num(p.stem, p.stem)[1], p.name),
            )
            for mf in manifest_files:
                try:
                    mdata = json.loads(mf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(mdata, dict) or not isinstance(mdata.get("runs"), list):
                    continue
                wave_raw = str(mdata.get("wave") or mf.stem)
                wave_desc = str(mdata.get("description") or "")
                plan_id = str(mdata.get("plan_id") or "")
                for run_entry in mdata["runs"]:
                    if not isinstance(run_entry, dict) or "key" not in run_entry:
                        continue
                    rkey = str(run_entry["key"])
                    order_counter += 1
                    prev_spec = by_key.get(rkey)
                    merged_entry = dict(run_entry)
                    if prev_spec:
                        for fallback_field in ("color", "short_label", "label"):
                            if fallback_field not in merged_entry and fallback_field in prev_spec:
                                merged_entry[fallback_field] = prev_spec[fallback_field]
                    card = build_model_card_from_manifest_entry(
                        merged_entry,
                        wave_id=wave_raw,
                        wave_description=wave_desc,
                        plan_id=plan_id,
                    )
                    if rkey in existing_cards_on_disk:
                        if "telemetry" in existing_cards_on_disk[rkey]:
                            card["telemetry"] = existing_cards_on_disk[rkey]["telemetry"]
                        for obs_k in ("obsolete", "obsolete_reason", "obsolete_at", "run_tier", "pareto_frontier"):
                            if obs_k in existing_cards_on_disk[rkey]:
                                card[obs_k] = existing_cards_on_disk[rkey][obs_k]
                        if bool(merged_entry.get("obsolete", False)):
                            card["obsolete"] = True
                            card["obsolete_reason"] = str(
                                merged_entry.get("obsolete_reason") or card.get("obsolete_reason") or ""
                            )
                            card["default_visible"] = False
                    spec = self._spec_from_card(card, order_counter, from_manifest=True)
                    by_key[rkey] = spec

        # 4. Standalone Model Cards in .cache/model_cards/*.json (e.g. pushed via POST /api/ingest or ad-hoc runs)
        for rkey, mc_data in existing_cards_on_disk.items():
            if rkey in BASELINE_RUN_KEYS:
                continue
            if rkey not in by_key:
                order_counter += 1
                by_key[rkey] = self._spec_from_card(mc_data, order_counter, from_manifest=True)

        specs = list(by_key.values())
        if persist_cards:
            for s in specs:
                save_model_card_atomic(s["model_card"], self.model_cards_dir)
        return specs

    def is_run_completed_locally(self, spec: dict[str, Any]) -> bool:
        """Return True if the run's local log file contains `COMPLETED` or its model card is marked completed."""
        local_log = self.live_logs_dir / Path(spec["log"]).name
        if local_log.exists():
            try:
                if "COMPLETED" in local_log.read_text(encoding="utf-8", errors="replace"):
                    return True
            except Exception:
                pass
        mc_tel = spec.get("model_card", {}).get("telemetry", {})
        return bool(mc_tel.get("completed", False))

    def select_session_poll_targets(
        self,
        specs: list[dict[str, Any]] | None = None,
        active_sessions: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Select at most ONE non-completed run per Colab session (the latest wave/order run on that session).

        Eliminates redundant `colab exec` calls when multiple waves (`W6`, `W7`, `W8`, `W9`) share a session name.
        """
        if specs is None:
            specs = self.discover_all_specs(persist_cards=False)

        if active_sessions is None and self.sessions_file is not None and self.sessions_file.exists():
            try:
                sdata = json.loads(self.sessions_file.read_text(encoding="utf-8"))
                if isinstance(sdata, dict) and sdata:
                    active_sessions = set(sdata.keys())
            except Exception:
                active_sessions = None

        all_by_session: dict[str, list[dict[str, Any]]] = {}
        for spec in specs:
            sess = str(spec["session"])
            if sess.startswith("local") or spec.get("wave_id") in ("VERIFY", "MICRO") or spec.get("run_tier") == "micro_verification":
                continue
            if active_sessions is not None and sess not in active_sessions:
                continue
            wave_num = int(spec.get("wave_num", 1))
            if not (wave_num >= 4 or spec.get("from_manifest", False)):
                continue
            all_by_session.setdefault(sess, []).append(spec)

        selected: dict[str, dict[str, Any]] = {}
        for sess, sess_specs in all_by_session.items():
            max_wave = max(int(s.get("wave_num", 1)) for s in sess_specs)
            latest_wave_incomplete = [
                s
                for s in sess_specs
                if int(s.get("wave_num", 1)) == max_wave and not self.is_run_completed_locally(s)
            ]
            if not latest_wave_incomplete:
                continue
            fallback_incomplete = [
                s
                for s in sess_specs
                if int(s.get("wave_num", 1)) >= max_wave - 1 and not self.is_run_completed_locally(s)
            ]
            fallback_incomplete.sort(
                key=lambda s: (
                    int(s.get("wave_num", 1)),
                    0 if bool(s.get("model_card", {}).get("obsolete", False)) else 1,
                    int(s.get("order_idx", 0)),
                )
            )
            primary = dict(fallback_incomplete[-1])
            primary["session_fallback_specs"] = fallback_incomplete[:-1]
            selected[sess] = primary
        return selected

    def get_remote_poll_run_keys(
        self,
        specs: list[dict[str, Any]] | None = None,
        active_sessions: set[str] | None = None,
    ) -> set[str]:
        """Return the set of `run_key`s selected for live remote session polling (at most 1 per session)."""
        targets = self.select_session_poll_targets(specs=specs, active_sessions=active_sessions)
        return {str(spec["key"]) for spec in targets.values()}

    def should_poll_remote(
        self,
        spec: dict[str, Any],
        specs: list[dict[str, Any]] | None = None,
        active_sessions: set[str] | None = None,
    ) -> bool:
        """Return True iff `spec` is the active non-completed poll target for its Colab session."""
        return str(spec["key"]) in self.get_remote_poll_run_keys(specs=specs, active_sessions=active_sessions)

    def get_wave_metadata(
        self,
        specs: list[dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return a mapping of `wave_id -> {wave_id, wave_num, wave_label, description, plan_id, run_keys}`."""
        if specs is None:
            specs = self.discover_all_specs(persist_cards=False)
        waves: dict[str, dict[str, Any]] = {}
        for s in specs:
            wid = str(s.get("wave_id", "W1"))
            if wid not in waves:
                waves[wid] = {
                    "wave_id": wid,
                    "wave_num": int(s.get("wave_num", 1)),
                    "wave_label": str(s.get("wave_label", wid)),
                    "description": str(s.get("wave_label", wid)),
                    "plan_id": str(s.get("plan_id", "")),
                    "run_keys": [],
                }
            waves[wid]["run_keys"].append(str(s["key"]))
        return waves

