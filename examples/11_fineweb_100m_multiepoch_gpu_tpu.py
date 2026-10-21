"""Example 11: Full 100M-Token Multi-Epoch FineWeb Quantum Slowrun Benchmark (GPU & TPU).

Benchmarks the 10-Qubit (`d = 1024 = 2^10`, ~70.68M parameters) Next-Gen Slowrun
`QuantumBornGPT` architecture over multiple full passes of the 100,000,000-token
FineWeb training shard (`fineweb_train_000001.bin`, `299,827,200` total tokens across
3 full epochs at `65,536` tokens/step) on both:

1. **CUDA / ROCm GPU (`--backend cuda`)**:
   - Fused Triton `flash_born_attention` kernel (64x64 Tensor Core / RDNA 4 WMMA tiles +
     atomic-free SRAM-accumulated backward `dQ` and `dK/dV` kernels, zero `[B, H, T, T]`
     materialization)
   - `torch.compile` (TorchInductor fusion across Isometric Hilbert Lift, Temporal Bigram
     Dispersion, 2-Tap Causal QK Wave Filter, Unitary RoPE, `TwoBeamQuantumInterference` MLP,
     Mach-Zehnder U-Net Layer-Skip Interferometry, and Stiefel-Tied POVM Readout)
   - Compiled 5-step Newton-Schulz `StiefelMuon` polar factor + BFloat16 Riemannian Stiefel
     retraction (`retract_every=4`) + Checkpointed BFloat16 Stiefel-POVM Cross-Entropy
   - Supports Warmup-Stable-Decay (`--schedule wsd`), Progressive Sequence-Length Curriculum
     (`--seq-curriculum 256,512,1024,2048`), Multi-Token Quantum POVM Prediction (`--mtp-horizons 3`),
     and 10-Qubit (`5+5` split) Bipartite von Neumann Entanglement Entropy telemetry.

2. **Cloud TPU (`--backend tpu`)**:
   - Pure JAX/XLA `@jax.jit(donate_argnums=(0, 1))` whole-step compilation (fusing forward,
     backward, global gradient norm clipping, 5-step BFloat16 MXU Newton-Schulz `StiefelMuon`,
     Continuous Riemannian Stiefel Penalty, and `AdamW` into a single XLA HLO executable)
   - Supports Custom Pallas TPU VMEM-tiled Flash-Born kernel (`--tpu-attn-kernel pallas`),
     chunked `lax.scan` (`--tpu-attn-kernel scan`), or rematerialized MXU Born Attention
     (`--tpu-attn-kernel remat`) + Scatter-Free Factorized MXU Stiefel-POVM Cross-Entropy.
"""

from __future__ import annotations

import argparse
from functools import partial
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "tpu"),
        default="auto",
        help="Accelerator backend to run ('auto' detects CUDA GPU or JAX TPU).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of full 100M-token epochs to train (default: 3 = 300M total tokens).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of distinct 100M-token FineWeb shards (fineweb_train_000001..N.bin) to stream per epoch (default: 1; set --num-shards 9 --epochs 1 for 900M unique tokens).",
    )
    parser.add_argument(
        "--eval-every-tokens-m",
        type=float,
        default=50.0,
        help="Validation evaluation interval in millions of tokens (default: 50.0M tokens = 762 steps at 65,536 tok/step).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/data/fineweb10B"),
        help="Directory containing train_full_100M.bin and val_slice.bin.",
    )
    parser.add_argument(
        "--schedule",
        choices=("cosine", "wsd"),
        default="cosine",
        help="Learning rate schedule: 'cosine' or 'wsd' (Warmup-Stable-Decay with 1 - sqrt(tau) cooldown).",
    )
    parser.add_argument(
        "--wsd-decay-fraction",
        type=float,
        default=0.20,
        help="Fraction of total steps spent in radical 1 - sqrt(tau) cooldown when --schedule=wsd.",
    )
    parser.add_argument(
        "--seq-curriculum",
        type=str,
        default="",
        help="Optional comma-separated progressive sequence lengths across epochs (e.g. '256,512,1024').",
    )
    parser.add_argument(
        "--mtp-horizons",
        type=int,
        default=1,
        help="Number of Multi-Token Quantum POVM Prediction horizons (1 = t+1 only; 3 = t+1, t+2, t+3).",
    )
    parser.add_argument(
        "--mtp-weight",
        type=float,
        default=0.25,
        help="Auxiliary loss weight for Multi-Token POVM Prediction horizons t+2..t+K.",
    )
    parser.add_argument(
        "--seed-dim",
        type=int,
        default=128,
        help="Seed Hilbert subspace dimension D_seed (default: 128).",
    )
    parser.add_argument(
        "--mtp-stride",
        type=int,
        default=4,
        help="Sequence subsampling stride s >= 1 for auxiliary MTP horizons when --mtp-horizons > 1 (default: 4).",
    )
    parser.add_argument(
        "--mtp-detach-readout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to detach M_seed on auxiliary MTP horizons (default: True).",
    )
    parser.add_argument(
        "--mtp-decoupled-temp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use horizon-specific inverse temperatures mtp_inv_temps (default: True).",
    )
    parser.add_argument(
        "--stiefel-optimizer",
        choices=("standard", "tangent"),
        default="standard",
        help="Stiefel manifold optimizer variant: 'standard' (StiefelMuon) or 'tangent' (TangentStiefelMuon, Plan 003).",
    )
    parser.add_argument(
        "--kraus-beams",
        type=int,
        default=1,
        help="Number of Multi-Beam Kraus Subspace POVM readout beams K >= 1 (Plan 004, default: 1).",
    )
    parser.add_argument(
        "--fused-povm-tile-size",
        type=int,
        default=0,
        help="If > 0, use Analytic-Gradient L2-Tiled POVM Cross-Entropy with this tile size (Plan 005, default: 0).",
    )
    parser.add_argument(
        "--overcomplete-frame",
        action="store_true",
        default=False,
        help="Replace block MLP with Overcomplete 2d Tight-Frame Interference and relaxed frame penalty (Plan 006).",
    )
    parser.add_argument(
        "--overcomplete-seed-adapter",
        action="store_true",
        default=False,
        help="Augment D -> D_seed seed collapse with OvercompleteSeedReadoutAdapter (Plan 007).",
    )
    parser.add_argument(
        "--overcomplete-two-beam",
        action="store_true",
        default=False,
        help="Replace block MLP with OvercompleteTwoBeamInterference 2d tight frames in both beams (Plan 009).",
    )
    parser.add_argument(
        "--two-beam-seed-adapter",
        action="store_true",
        default=False,
        help="Augment D -> D_seed seed collapse with phase-sensitive TwoBeamSeedReadoutAdapter (Plan 010).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of QuantumBornBlock circuit layers L (Plan 012, default: 6).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=1024,
        help="Ambient Hilbert space dimension d_model (Plan 012, default: 1024).",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Number of subspace projector Born attention heads H (Plan 012, default: 16).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1536,
        help="Two-beam interference MLP hidden dimension d_ff (Plan 012, default: 1536).",
    )
    parser.add_argument(
        "--frame-expansion",
        type=float,
        default=2.0,
        help="Overcomplete frame redundancy factor rho = M / d_model (Plan 012, default: 2.0).",
    )
    parser.add_argument(
        "--readout-mode",
        choices=("tied_stiefel", "tied_vocab", "symmetric_dual", "asymmetric"),
        default="tied_stiefel",
        help="POVM measurement readout mode (Plan 012, default: 'tied_stiefel').",
    )
    parser.add_argument(
        "--depth-scaled-residual",
        action="store_true",
        default=False,
        help="Enable depth-scaled spherical residuals alpha_res(L) = sqrt(6 / L) (Plan 012).",
    )
    parser.add_argument(
        "--architecture",
        choices=("quantum_born", "classical_modded"),
        default="quantum_born",
        help="Model architecture to benchmark: 'quantum_born' (QuantumBornGPT) or 'classical_modded' (ClassicalModdedGPT, Plan 013).",
    )
    parser.add_argument(
        "--dynamic-povm-beta",
        action="store_true",
        default=False,
        help="Enable ContextualBlochPurityGate per-token dynamic POVM inverse temperature beta_{b,t} (Plan 013).",
    )
    parser.add_argument(
        "--ortho-decouple",
        action="store_true",
        default=False,
        help="Initialize readout_rot and unet_gates with orthogonal Stiefel matrices to eliminate Step-0 self-repetition (Plan 014).",
    )
    parser.add_argument(
        "--povm-log-prior",
        action="store_true",
        default=False,
        help="Enable learnable rank-1 POVM log-prior bias b in R^V (Plan 014).",
    )
    parser.add_argument(
        "--gated-spherical-residual",
        action="store_true",
        default=False,
        help="Enable pre-normalized gated spherical residual updates psi <- N(psi + alpha * N(sublayer(psi))) (Plan 014).",
    )
    parser.add_argument(
        "--purified-born-attn",
        action="store_true",
        default=False,
        help="Enable coherence-thresholded degree-4 purified Born attention with non-canceling gamma_h (Plan 014).",
    )
    parser.add_argument(
        "--zero-init-classical",
        action="store_true",
        default=False,
        help="Enable modded-nanogpt zero-init residual/lm_head, Nesterov+aspect-scale Muon, and decoupled AdamW LRs for ClassicalModdedGPT (Plan 014).",
    )
    parser.add_argument(
        "--true-layer0-v-residual",
        action="store_true",
        default=False,
        help="Enable true cross-layer first-layer value residual mixing V^(0) = W_V^(0) x_in^(0) across Layers 1..L-1 (Plan 015).",
    )
    parser.add_argument(
        "--classical-stiefel-penalty",
        type=float,
        default=0.0,
        help="Continuous Riemannian Stiefel retraction penalty for ClassicalModdedGPT 2D matrices (Plan 015, default: 0.0; set 0.005 for bounded-norm).",
    )
    parser.add_argument(
        "--wsd-min-lr-ratio",
        type=float,
        default=0.10,
        help="Minimum learning-rate ratio at the end of WSD 1 - sqrt(tau) cooldown (Plan 015, default: 0.10; set 0.0 for full zero-floor cooldown).",
    )
    parser.add_argument(
        "--stage-shard-on-gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stage active 100M-token uint16 shard directly in GPU VRAM (Plan 016, default: auto-enabled on >=20GB VRAM).",
    )
    parser.add_argument(
        "--direct-chunked-ce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route vocabulary readout loss through single-pass direct chunked cross-entropy autograd (Plan 016, default: True).",
    )
    parser.add_argument(
        "--unet-skip-mode",
        choices=("matrix", "scalar"),
        default="matrix",
        help="U-Net encoder-to-decoder skip mode: 'matrix' or 'scalar' (Plan 016, default: 'matrix').",
    )
    parser.add_argument(
        "--fused-qkv",
        action="store_true",
        default=False,
        help="Enable single-GEMM fused QKV projection W_qkv in attention blocks (Plan 016).",
    )
    parser.add_argument(
        "--num-value-embeddings",
        type=int,
        default=0,
        help="Number of speedrun token-indexed Value Embedding tables ve_0..ve_{k-1} shared into first and last k layers (Plan 016, default: 0).",
    )
    parser.add_argument(
        "--skip-attn-layers",
        type=str,
        default="",
        help="Comma-separated layer indices to skip attention computation (MLP-only layers, Plan 016, e.g. '3').",
    )
    parser.add_argument(
        "--zero-half-rope",
        action="store_true",
        default=False,
        help="Enable half-truncated RoPE frequencies (zeroing upper half of head frequencies, Plan 016).",
    )
    parser.add_argument(
        "--doc-mask-attn",
        action="store_true",
        default=False,
        help="Enable causal document-boundary attention masking at EOT token 50256 (Plan 016).",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=0,
        help="Sliding-window causal attention span W (0 = full causal context, Plan 016).",
    )
    parser.add_argument(
        "--muon-momentum-warmup",
        type=int,
        default=0,
        help="Steps to linearly warm up StiefelMuon momentum from 0.85 to 0.95 (Plan 016, default: 0).",
    )
    parser.add_argument(
        "--muon-lr",
        type=float,
        default=None,
        help="Base StiefelMuon learning rate for 2D matrix parameters (Plan 018/019, default: 0.05 with --leaderboard-exact-classical else 0.025).",
    )
    parser.add_argument(
        "--leaderboard-exact-classical",
        action="store_true",
        default=False,
        help="Enable exact KellerJordan/modded-nanogpt leaderboard parity (H=6 head_dim=128, attn_scale=0.12, RoPE base=1024, un-normed VE lr=0.6, softcap=15, pre-block U-Net skip, skip_attn=7, ema_momentum=True, wd=0.0, warmup=0, Plan 019).",
    )
    parser.add_argument(
        "--progressive-window",
        action="store_true",
        default=False,
        help="Enable progressive long/short sliding-window schedule (128 -> 1792) during training while evaluating at fixed T_val=512 (Plan 019).",
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=("muon_adamw", "pure_adamw", "sgd_nesterov"),
        default="muon_adamw",
        help="Optimizer family for 2D matrix parameters (Plan 016, default: 'muon_adamw').",
    )
    parser.add_argument(
        "--tpu-attn-kernel",
        choices=("remat", "pallas", "scan"),
        default="remat",
        help="TPU Born attention kernel mode ('remat', 'pallas', or 'scan').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Micro-batch size per forward pass at T=512 (default: 128 = 65,536 tokens/step at grad_accum=1).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps per optimizer update (default: 1; tok_per_step = batch_size * grad_accum * seq_len).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Default sequence length T when --seq-curriculum is not set (default: 512).",
    )
    parser.add_argument(
        "--synthetic-tokens",
        type=int,
        default=0,
        help="If > 0 (or if FineWeb shard is absent), generate deterministic Zipfian token stream of this length.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON path for checkpoint history.",
    )
    parser.add_argument(
        "--ppmi-svd-init",
        action="store_true",
        default=False,
        help="Initialize embeddings using GPU Shifted PPMI + Randomized SVD co-occurrence sketch (Plan 021).",
    )
    parser.add_argument(
        "--ppmi-svd-shards",
        type=int,
        default=9,
        help="Number of FineWeb shards (1..9, up to 900M tokens) to include in PPMI-SVD sketch (Plan 021, default: 9).",
    )
    parser.add_argument(
        "--ppmi-svd-alpha",
        type=float,
        default=0.65,
        help="Blend weight alpha_ppmi in [0, 1] between random Gaussian identity and PPMI-SVD basis (Plan 021, default: 0.65).",
    )
    parser.add_argument(
        "--ngram-hash",
        action="store_true",
        default=False,
        help="Enable HashedNgramEmbedding (hashed bigram + trigram lookup with +-1 sign pool, Plan 021).",
    )
    parser.add_argument(
        "--ngram-vocab-size",
        type=int,
        default=262144,
        help="Number of rows in HashedNgramEmbedding table (Plan 021, default: 262144).",
    )
    parser.add_argument(
        "--polar-express",
        action="store_true",
        default=False,
        help="Use 6-step ANVIL quintic maps (zeropower_via_polarexpress5) in StiefelMuon (Plan 021).",
    )
    parser.add_argument(
        "--normuon",
        action="store_true",
        default=False,
        help="Enable NorMuon per-lane variance equalization in StiefelMuon (Plan 021).",
    )
    parser.add_argument(
        "--twin-rail-muon",
        action="store_true",
        default=False,
        help="Enable ANVIL twin-rail (fast + slow EMA) Nesterov momentum in StiefelMuon (Plan 021).",
    )
    parser.add_argument(
        "--cautious-wd",
        action="store_true",
        default=False,
        help="Enable sign-aligned cautious weight decay in StiefelMuon (Plan 021).",
    )
    parser.add_argument(
        "--tail-ema",
        action="store_true",
        default=False,
        help="Enable TailWeightAverager over the last 25%% of training steps with Frobenius/spherical decontraction (Plan 021).",
    )
    parser.add_argument(
        "--mudd-and-gates",
        action="store_true",
        default=False,
        help="Enable InputChannelGate (ve_gate, attn_gate), GatedXSA, and MUDDDynamicSkip (Plan 021).",
    )
    parser.add_argument(
        "--smear-gate",
        action="store_true",
        default=False,
        help="Enable 1-token ForwardSmearGate (even when --leaderboard-exact-classical is active, Plan 021).",
    )
    parser.add_argument(
        "--tapered-mtp",
        action="store_true",
        default=False,
        help="Linearly taper mtp_weight to 0.0 at 65%% of total steps (Plan 021).",
    )
    parser.add_argument(
        "--align-to-bos",
        action="store_true",
        default=False,
        help="Align training batch sequence starts to <|endoftext|> (50256) document boundaries (Plan 021).",
    )
    parser.add_argument(
        "--sampled-softmax-candidates",
        type=int,
        default=0,
        help="When > 0, use sampled-softmax candidate CE during first 75%% of training steps and full softmax for final 25%% (Plan 021).",
    )
    parser.add_argument(
        "--embed-step-every",
        type=int,
        default=1,
        help="Step the AdamW/Adam embedding optimizer every K steps (Plan 021, default: 1).",
    )
    parser.add_argument(
        "--fp8-linear",
        action="store_true",
        default=False,
        help="Wrap eligible linear projections in FP8LinearWrapper (Plan 021).",
    )
    parser.add_argument(
        "--sparse-64qubit",
        action="store_true",
        default=False,
        help="Enable 64-qubit truncated Fock space H_64^(<=2) (d_fock=2048) with Sparse64QubitNgramEmbedder (Plan 022).",
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=64,
        help="Number of logical qubits Q in Sparse64QubitNgramEmbedder (Plan 022, default: 64).",
    )
    parser.add_argument(
        "--sparse-ngram-k",
        type=int,
        default=16,
        help="Sparse Fock excitation fan-out K per n-gram event (Plan 022, default: 16).",
    )
    parser.add_argument(
        "--sparse-density-layer",
        action="store_true",
        default=False,
        help="Enable 4.19M-parameter Sparse64QubitDensityLayer inside each QuantumBornGPT block (Plan 022).",
    )
    parser.add_argument(
        "--density-matrix-rank",
        type=int,
        default=4,
        help="Rank R of the internal density matrix operator rho_l in Sparse64QubitDensityLayer (Plan 022, default: 4).",
    )
    parser.add_argument(
        "--no-fock-coupler",
        action="store_true",
        default=False,
        help="Disable the 64 <-> 1984 single-qubit to pair-excitation Fock coupler in Sparse64QubitDensityLayer (Plan 022).",
    )
    parser.add_argument(
        "--hierarchical-16qubit",
        action="store_true",
        default=False,
        help="Enable 50,304 x 16 x 32 Multi-Qubit Unigram Table & 256 x 16 x 32 Interacting Higher-Level Quantum Hierarchy (Plan 023).",
    )
    parser.add_argument(
        "--qubit-registers",
        type=int,
        default=16,
        help="Number of qubit sub-registers Q per token in Hierarchical16QubitEmbedder (Plan 023, default: 16).",
    )
    parser.add_argument(
        "--qubit-amplitudes",
        type=int,
        default=32,
        help="Sub-register Hilbert dimension A per qubit register in Hierarchical16QubitEmbedder (Plan 023, default: 32).",
    )
    parser.add_argument(
        "--num-higher-states",
        type=int,
        default=256,
        help="Number of interacting higher-level quantum states M in Hierarchical16QubitEmbedder (Plan 023, default: 256).",
    )
    parser.add_argument(
        "--no-block-hierarchy-bridge",
        action="store_true",
        default=False,
        help="Disable the per-block 256-state hierarchical quantum bridge inside QuantumBornGPT blocks (Plan 023).",
    )
    parser.add_argument(
        "--last-layer-softmax",
        action="store_true",
        default=False,
        help="Use Calibrated Last-Layer Softmax Readout (RMSNorm + unconstrained linear readout + softcap) instead of unit-sphere POVM (Plan 024).",
    )
    parser.add_argument(
        "--two-beam-kboson-readout",
        "--two-beam-kboson",
        dest="two_beam_kboson_readout",
        action="store_true",
        default=False,
        help="Enable 100%% Quantum Two-Beam k-Boson Mixed-State POVM Readout + Non-Isometric Kraus Semantic Bath (Plan 026).",
    )
    parser.add_argument(
        "--diffuse-rank",
        type=int,
        default=16,
        help="Subspace rank r of the Non-Isometric Kraus Semantic Bath K_diff (Plan 026, default: 16).",
    )
    parser.add_argument(
        "--logit-softcap",
        type=float,
        default=15.0,
        help="Tanh logit softcap for Calibrated Last-Layer Softmax Readout (Plan 024, default: 15.0).",
    )
    parser.add_argument(
        "--val-tokens",
        type=int,
        default=10485760,
        help="Total validation tokens to evaluate sequentially from pos=0 (default: 10485760 = 10 * 1024 * 1024, matching KellerJordan/modded-nanogpt).",
    )
    return parser.parse_args()


def _load_or_generate_shards(
    data_dir: Path,
    synthetic_tokens: int = 0,
    vocab_size: int = 50257,
) -> tuple[np.ndarray, np.ndarray]:
    train_path = data_dir / "train_full_100M.bin"
    val_path = data_dir / "val_slice.bin"
    val_target_bytes = 21_103_616  # 1024B header + (10,485,760 + 65,536) * 2B uint16 tokens
    if synthetic_tokens <= 0:
        need_val_dl = (
            (not val_path.exists())
            or (val_path.stat().st_size <= 1024)
            or (val_path.stat().st_size == 10_000_000)
        )
        need_train_dl = (not train_path.exists()) or (train_path.stat().st_size < 200_001_024)
        if need_val_dl or need_train_dl:
            import urllib.request

            data_dir.mkdir(parents=True, exist_ok=True)
            val_url = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin"
            train_url = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_train_000001.bin"
            for url, dest, n_bytes, should_dl in [
                (val_url, val_path, val_target_bytes, need_val_dl),
                (train_url, train_path, 200_001_024, need_train_dl),
            ]:
                if should_dl:
                    tmp_dest = dest.with_suffix(".bin.part")
                    try:
                        print(f"Downloading {dest.name} ({n_bytes / 1e6:.1f} MB) from HuggingFace FineWeb10B...", flush=True)
                        req = urllib.request.Request(
                            url, headers={"Range": f"bytes=0-{n_bytes - 1}", "User-Agent": "Mozilla/5.0"}
                        )
                        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_dest, "wb") as f:
                            while True:
                                chunk = resp.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                f.write(chunk)
                        tmp_dest.replace(dest)
                    except Exception:
                        if tmp_dest.exists():
                            tmp_dest.unlink(missing_ok=True)
                        if not dest.exists() or dest.stat().st_size <= 1024:
                            raise
        if train_path.exists() and val_path.exists():
            train_tokens = np.fromfile(train_path, dtype=np.uint16, offset=1024)
            val_tokens = np.fromfile(val_path, dtype=np.uint16, offset=1024)
            return train_tokens, val_tokens

    n_train = int(synthetic_tokens) if synthetic_tokens > 0 else 1048577
    n_val = max(513, min(327681, max(n_train // 2, 1025)))
    rng = np.random.default_rng(42)
    ranks = np.arange(1, 2049, dtype=np.float64)
    probs = 1.0 / (ranks ** 1.08)
    probs /= probs.sum()
    base_vocab = min(2048, vocab_size)
    train_tokens = rng.choice(base_vocab, size=n_train, replace=True, p=probs[:base_vocab]).astype(np.uint16)
    # Inject deterministic repeating n-gram patterns so multi-epoch + MTP learning is measurable
    for offset in range(0, n_train - 4, 8):
        train_tokens[offset + 1] = (int(train_tokens[offset]) * 7 + 3) % base_vocab
        train_tokens[offset + 2] = (int(train_tokens[offset + 1]) * 5 + 11) % base_vocab
    val_tokens = train_tokens[:n_val].copy()
    return train_tokens, val_tokens


def run_cuda_benchmark(
    num_epochs: int = 3,
    num_shards: int = 1,
    eval_every_tokens_m: float = 50.0,
    data_dir: Path = Path("/content/data/fineweb10B"),
    schedule: Literal["cosine", "wsd"] = "cosine",
    wsd_decay_fraction: float = 0.20,
    wsd_min_lr_ratio: float = 0.10,
    seq_curriculum_str: str = "",
    mtp_horizons: int = 1,
    mtp_weight: float = 0.25,
    seed_dim: int = 128,
    mtp_stride: int = 4,
    mtp_detach_readout: bool = True,
    mtp_decoupled_temp: bool = True,
    stiefel_optimizer: Literal["standard", "tangent"] = "standard",
    kraus_beams: int = 1,
    fused_povm_tile_size: int = 0,
    overcomplete_frame: bool = False,
    overcomplete_seed_adapter: bool = False,
    overcomplete_two_beam: bool = False,
    two_beam_seed_adapter: bool = False,
    num_layers: int = 6,
    embed_dim: int = 1024,
    num_heads: int = 16,
    hidden_dim: int = 1536,
    frame_expansion: float = 2.0,
    readout_mode: Literal["tied_stiefel", "tied_vocab", "symmetric_dual", "asymmetric"] = "tied_stiefel",
    depth_scaled_residual: bool = False,
    architecture: Literal["quantum_born", "classical_modded"] = "quantum_born",
    dynamic_povm_beta: bool = False,
    ortho_decouple: bool = False,
    povm_log_prior: bool = False,
    gated_spherical_residual: bool = False,
    purified_born_attn: bool = False,
    zero_init_classical: bool = False,
    true_layer0_v_residual: bool = False,
    classical_stiefel_penalty: float = 0.0,
    stage_shard_on_gpu: bool | None = None,
    direct_chunked_ce: bool = True,
    unet_skip_mode: Literal["matrix", "scalar"] = "matrix",
    fused_qkv: bool = False,
    num_value_embeddings: int = 0,
    skip_attn_layers_str: str = "",
    zero_half_rope: bool = False,
    doc_mask_attn: bool = False,
    sliding_window: int = 0,
    muon_momentum_warmup: int = 0,
    muon_lr: float | None = None,
    leaderboard_exact_classical: bool = False,
    progressive_window: bool = False,
    ppmi_svd_init: bool = False,
    ppmi_svd_shards: int = 9,
    ppmi_svd_alpha: float = 0.65,
    ngram_hash: bool = False,
    ngram_vocab_size: int = 262144,
    polar_express: bool = False,
    normuon: bool = False,
    twin_rail_muon: bool = False,
    cautious_wd: bool = False,
    tail_ema: bool = False,
    mudd_and_gates: bool = False,
    smear_gate: bool = False,
    tapered_mtp: bool = False,
    align_to_bos: bool = False,
    sampled_softmax_candidates: int = 0,
    embed_step_every: int = 1,
    fp8_linear: bool = False,
    sparse_64qubit: bool = False,
    num_qubits: int = 64,
    sparse_ngram_k: int = 16,
    sparse_density_layer: bool = False,
    density_matrix_rank: int = 4,
    no_fock_coupler: bool = False,
    hierarchical_16qubit: bool = False,
    qubit_registers: int = 16,
    qubit_amplitudes: int = 32,
    num_higher_states: int = 256,
    no_block_hierarchy_bridge: bool = False,
    last_layer_softmax: bool = False,
    two_beam_kboson_readout: bool = False,
    diffuse_rank: int = 16,
    logit_softcap: float = 15.0,
    optimizer_mode: Literal["muon_adamw", "pure_adamw", "sgd_nesterov"] = "muon_adamw",
    batch_size: int = 128,
    grad_accum: int = 1,
    seq_len: int = 512,
    val_tokens_budget: int = 10485760,
    synthetic_tokens: int = 0,
    out_json: Path | None = None,
) -> list[dict[str, float]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    from qdlblocks.benchmarks.fineweb_stream import (
        FINEWEB_GPT2_VOCAB_SIZE,
        FINEWEB_VAL_TOKENS,
        FineWebMultiShardStream,
        compute_modded_nanogpt_val_steps,
        compute_tapered_mtp_weight,
        find_bos_aligned_starts,
    )
    from qdlblocks.functional import (
        bipartite_entanglement_entropy,
        effective_quantum_rank,
        l2_normalize,
        purity_participation_ratio,
        zeropower_via_newtonschulz5,
        zeropower_via_newtonschulz5_batched,
        zeropower_via_polarexpress5,
        zeropower_via_polarexpress5_batched,
    )
    from qdlblocks.kernels.fused_povm_ce import fused_tiled_povm_cross_entropy
    from qdlblocks.layers.attention import compute_document_ids
    from qdlblocks.layers.cooccurrence_init import (
        initialize_classical_with_ppmi_svd,
        initialize_quantum_born_with_ppmi_svd,
    )
    from qdlblocks.layers.hierarchical_qubit import (
        initialize_hierarchical_16qubit_with_ppmi_svd,
    )
    from qdlblocks.layers.sparse_qubit_fock import (
        initialize_sparse_64qubit_with_ppmi_svd,
    )
    from qdlblocks.layers.direct_chunked_ce import (
        compute_two_beam_kboson_logits,
        direct_chunked_povm_ce,
        direct_chunked_softcap_linear_ce,
        direct_chunked_two_beam_kboson_ce,
        sampled_povm_ce,
        sampled_softcap_linear_ce,
    )
    from qdlblocks.layers.fp8_linear import convert_linear_layers_to_fp8
    from qdlblocks.layers.kraus_readout import MultiBeamKrausReadout
    from qdlblocks.layers.overcomplete_frame import OvercompleteFrameInterference
    from qdlblocks.layers.overcomplete_two_beam import OvercompleteTwoBeamInterference
    from qdlblocks.layers.parameter_scaling import default_overcomplete_target_iso_scale
    from qdlblocks.metrics import model_quantum_scale_telemetry
    from qdlblocks.models.born_gpt import QuantumBornGPT
    from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
    import qdlblocks.optim.muon as _muon_mod
    from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params
    from qdlblocks.optim.schedules import (
        ProgressiveSequenceCurriculum,
        cosine_lr_multiplier,
        wsd_lr_multiplier,
    )
    from qdlblocks.optim.tail_ema import TailWeightAverager
    from qdlblocks.optim.tangent_stiefel_muon import TangentStiefelMuon

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    _muon_mod.zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="default")
    _muon_mod.zeropower_via_newtonschulz5_batched = torch.compile(zeropower_via_newtonschulz5_batched, mode="default")
    _muon_mod.zeropower_via_polarexpress5 = torch.compile(zeropower_via_polarexpress5, mode="default")
    _muon_mod.zeropower_via_polarexpress5_batched = torch.compile(zeropower_via_polarexpress5_batched, mode="default")

    def _chunk_ce_fn(
        zc: torch.Tensor,
        M_seed: torch.Tensor,
        inv_t: torch.Tensor,
        yc: torch.Tensor,
        log_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits_c = F.linear(zc, M_seed) * inv_t.to(zc.dtype)
        if log_prior is not None:
            logits_c = logits_c + log_prior.to(logits_c.dtype)
        v_act = min(FINEWEB_GPT2_VOCAB_SIZE, int(logits_c.size(-1)))
        return F.cross_entropy(logits_c[:, :v_act].float(), yc)

    def _classical_chunk_ce_fn(
        xc: torch.Tensor,
        w_lm: torch.Tensor,
        softcap: float,
        yc: torch.Tensor,
    ) -> torch.Tensor:
        raw_c = F.linear(xc, w_lm.to(xc.dtype))
        logits_c = softcap * torch.tanh(raw_c / softcap) if softcap > 0.0 else raw_c
        v_act = min(FINEWEB_GPT2_VOCAB_SIZE, int(logits_c.size(-1)))
        return F.cross_entropy(logits_c[:, :v_act].float(), yc)

    def forward_backbone(
        m: QuantumBornGPT,
        idx: torch.Tensor,
        return_layer_states: bool = False,
        cached_m_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        B_cur, T_cur = idx.shape
        if getattr(m, "use_hierarchical_16qubit", False) and getattr(m, "hierarchical_qubit_embed", None) is not None:
            psi_0 = m.hierarchical_qubit_embed(idx)
            if getattr(m, "ngram_embed", None) is not None:
                s0 = m.ngram_embed.layer_scales[0].to(psi_0.dtype)
                psi_0 = l2_normalize(
                    psi_0 + s0 * m.ngram_embed(idx).to(psi_0.dtype),
                    dim=-1,
                    eps=m.eps,
                )
        elif getattr(m, "use_sparse_64qubit", False) and getattr(m, "sparse_qubit_embed", None) is not None:
            psi_0 = m.sparse_qubit_embed(idx)
        else:
            psi_0 = m.hilbert_lift(idx)
            if getattr(m, "ngram_embed", None) is not None:
                s0 = m.ngram_embed.layer_scales[0].to(psi_0.dtype)
                psi_0 = l2_normalize(
                    psi_0 + s0 * m.ngram_embed(idx).to(psi_0.dtype),
                    dim=-1,
                    eps=m.eps,
                )
        if m.training and m.depolarizing_jitter > 0.0:
            noise = torch.randn_like(psi_0) * (m.depolarizing_jitter / math.sqrt(m.embed_dim))
            psi_0 = l2_normalize(psi_0 + noise, dim=-1, eps=m.eps)
        if getattr(m, "smear_gate", None) is not None:
            psi_0 = m.smear_gate(psi_0, spherical=True, eps=m.eps)
        elif m.use_bigram_smear and T_cur > 1 and m.smear_embed_gate is not None:
            psi_prev = F.pad(psi_0[:, :-1, :], (0, 0, 1, 0))
            eta = (0.06 * torch.tanh(m.smear_embed_gate)).to(psi_0.dtype)
            psi_0 = l2_normalize(psi_0 + eta * psi_prev, dim=-1, eps=m.eps)

        psi = psi_0
        prev_psi = psi_0
        encoder_states: list[torch.Tensor] = []
        layer_states: list[torch.Tensor] = []
        n_half = m.num_layers // 2
        cos, sin = m.rope.get_cos_sin(T_cur)
        doc_ids = (
            compute_document_ids(idx, eot_token_id=getattr(m, "eot_token_id", 50256))
            if getattr(m, "doc_mask_attn", False)
            else None
        )
        win_sz = int(getattr(m, "window_size", 0))
        use_block_ckpt = bool(
            m.training
            and torch.is_grad_enabled()
            and psi.is_cuda
            and (B_cur * T_cur >= 65536)
            and (torch.cuda.get_device_properties(psi.device).total_memory < 24 * (1024**3))
        )
        v_0: torch.Tensor | None = None
        need_v0 = bool(
            getattr(m, "true_layer0_v_residual", False)
            or getattr(m, "value_embeds", None) is not None
        )
        for layer_idx, block in enumerate(m.blocks):
            if (
                getattr(m, "use_sparse_64qubit", False)
                and getattr(m, "sparse_qubit_embed", None) is not None
                and layer_idx > 0
            ):
                s_l = m.sparse_qubit_embed.layer_scales[layer_idx].to(psi.dtype)
                psi = l2_normalize(
                    psi + s_l * m.sparse_qubit_embed.forward_ngram_delta(idx).to(psi.dtype),
                    dim=-1,
                    eps=m.eps,
                )
            psi_before_block = psi
            ve_l = m._get_value_embed(layer_idx, idx) if hasattr(m, "_get_value_embed") else None
            if need_v0:
                if layer_idx == 0 and block.attn is not None:
                    if use_block_ckpt:
                        psi, v_0 = checkpoint(
                            block, psi, cos, sin, psi_0, None, True, ve_l, win_sz, doc_ids, use_reentrant=False
                        )
                    else:
                        psi, v_0 = block(
                            psi,
                            cos=cos,
                            sin=sin,
                            psi_0=psi_0,
                            v_0=None,
                            return_v0=True,
                            ve=ve_l,
                            window_size=win_sz,
                            doc_ids=doc_ids,
                        )
                else:
                    if use_block_ckpt:
                        psi = checkpoint(
                            block, psi, cos, sin, psi_0, v_0, False, ve_l, win_sz, doc_ids, use_reentrant=False
                        )
                    else:
                        psi = block(
                            psi,
                            cos=cos,
                            sin=sin,
                            psi_0=psi_0,
                            v_0=v_0,
                            return_v0=False,
                            ve=ve_l,
                            window_size=win_sz,
                            doc_ids=doc_ids,
                        )
            else:
                if use_block_ckpt:
                    psi = checkpoint(
                        block, psi, cos, sin, psi_0, None, False, ve_l, win_sz, doc_ids, use_reentrant=False
                    )
                else:
                    psi = block(
                        psi,
                        cos=cos,
                        sin=sin,
                        psi_0=psi_0,
                        v_0=None,
                        return_v0=False,
                        ve=ve_l,
                        window_size=win_sz,
                        doc_ids=doc_ids,
                    )
            if (
                getattr(m, "hierarchical_qubit_embed", None) is not None
                and getattr(m, "use_block_hierarchy_bridge", False)
            ):
                psi = m.hierarchical_qubit_embed.forward_block_bridge(psi, layer_idx=layer_idx)
            if getattr(m, "mudd_skips", None) is not None:
                mudd_sources = [psi_0] if layer_idx == 0 else [psi_0, prev_psi]
                psi = m.mudd_skips[layer_idx](psi, mudd_sources, spherical=True, eps=m.eps)
            prev_psi = psi_before_block
            if m.use_unet_interferometry:
                if layer_idx < n_half:
                    encoder_states.append(psi)
                elif layer_idx >= m.num_layers - n_half:
                    enc_idx = m.num_layers - 1 - layer_idx
                    skip_psi = encoder_states[enc_idx]
                    if getattr(m, "unet_skip_mode", "matrix") == "scalar" and getattr(m, "unet_lambdas", None) is not None:
                        lam_res = m.unet_lambdas[enc_idx, 0].to(psi.dtype)
                        lam_skip = m.unet_lambdas[enc_idx, 1].to(psi.dtype)
                        psi = l2_normalize(lam_res * psi + lam_skip * skip_psi, dim=-1, eps=m.eps)
                    elif m.unet_gates is not None and m.unet_scales is not None:
                        u_scale = (0.35 * torch.tanh(m.unet_scales[enc_idx])).to(psi.dtype)
                        skip_rot = F.linear(skip_psi, m.unet_gates[enc_idx].to(psi.dtype))
                        psi = l2_normalize(
                            psi + u_scale * l2_normalize(skip_rot, dim=-1, eps=m.eps),
                            dim=-1,
                            eps=m.eps,
                        )
            if return_layer_states:
                layer_states.append(psi.detach())

        if getattr(m, "use_last_layer_softmax", False):
            if (
                getattr(m, "use_hierarchical_16qubit", False)
                and getattr(m, "hierarchical_povm_readout", None) is not None
                and getattr(m, "hierarchical_qubit_embed", None) is not None
            ):
                phi_seed = m.hierarchical_povm_readout.format_readout_features(psi)
                M_seed = (
                    cached_m_seed.to(phi_seed.dtype)
                    if cached_m_seed is not None
                    else m.hierarchical_povm_readout.get_calibrated_softmax_weight(
                        m.hierarchical_qubit_embed,
                        m.povm_readout.povm_head if m.povm_readout is not None else None,
                    ).to(phi_seed.dtype)
                )
            elif m.povm_readout is not None:
                phi_seed = F.rms_norm(psi, (m.embed_dim,), eps=m.eps)
                M_seed = (
                    cached_m_seed.to(phi_seed.dtype)
                    if cached_m_seed is not None
                    else (m.povm_readout.povm_head * (1.0 / math.sqrt(float(m.embed_dim)))).to(phi_seed.dtype)
                )
            else:
                phi_seed = F.rms_norm(psi, (m.embed_dim,), eps=m.eps)
                M_seed = (
                    cached_m_seed.to(phi_seed.dtype)
                    if cached_m_seed is not None
                    else (
                        F.linear(m.wte.weight, m.W_lift.weight) * (1.0 / math.sqrt(float(m.embed_dim)))
                    ).to(phi_seed.dtype)
                )
            inv_t = torch.tensor(float(getattr(m, "logit_softcap", 15.0)), device=phi_seed.device, dtype=torch.float32)
            return phi_seed, M_seed, inv_t, layer_states

        if (
            getattr(m, "use_sparse_64qubit", False)
            and getattr(m, "sparse_povm_readout", None) is not None
            and getattr(m, "sparse_qubit_embed", None) is not None
            and m.readout_mode != "asymmetric"
        ):
            phi_seed = l2_normalize(psi[..., : m.num_qubits], dim=-1, eps=m.eps)
            M_seed = (
                cached_m_seed.to(phi_seed.dtype)
                if cached_m_seed is not None
                else m.sparse_povm_readout.get_qubit_povm_matrix(m.sparse_qubit_embed).to(phi_seed.dtype)
            )
            inv_t = torch.clamp(m.sparse_povm_readout.inv_temp, min=1.0, max=60.0)
        elif m.readout_mode == "asymmetric" and m.povm_readout is not None:
            phi_seed = l2_normalize(psi, dim=-1, eps=m.eps)
            if cached_m_seed is not None:
                M_seed = cached_m_seed.to(phi_seed.dtype)
            elif (
                getattr(m, "use_hierarchical_16qubit", False)
                and getattr(m, "hierarchical_povm_readout", None) is not None
                and getattr(m, "hierarchical_qubit_embed", None) is not None
            ):
                M_seed = m.hierarchical_povm_readout.get_coupled_povm_matrix(
                    m.hierarchical_qubit_embed,
                    m.povm_readout.povm_head,
                ).to(phi_seed.dtype)
            elif (
                getattr(m, "use_sparse_64qubit", False)
                and getattr(m, "sparse_povm_readout", None) is not None
                and getattr(m, "sparse_qubit_embed", None) is not None
                and m.povm_readout.povm_head.size(1) >= m.num_qubits
            ):
                q_out = m.sparse_povm_readout.get_qubit_povm_matrix(m.sparse_qubit_embed).to(m.povm_readout.povm_head.dtype)
                m_raw = m.povm_readout.povm_head
                m_head_q = m_raw[:, : m.num_qubits] + 0.25 * q_out
                m_coupled = torch.cat([m_head_q, m_raw[:, m.num_qubits :]], dim=1)
                M_seed = l2_normalize(m_coupled, dim=1, eps=m.eps).to(phi_seed.dtype)
            else:
                M_seed = l2_normalize(m.povm_readout.povm_head, dim=1, eps=m.eps).to(phi_seed.dtype)
            inv_t = torch.clamp(m.povm_readout.inv_temp, min=1.0, max=60.0)
        elif (
            m.readout_mode in ("tied_stiefel", "tied_vocab")
            and m.readout_rot is not None
            and m.tied_inv_temp is not None
        ):
            psi_final = l2_normalize(m.readout_rot(psi), dim=-1, eps=m.eps)
            if kraus_beams > 1:
                phi_seed = psi_final
            else:
                phi_seed = m.hilbert_lift.collapse_to_seed(psi_final)
                if m.seed_adapter is not None:
                    phi_seed = m.seed_adapter(psi_final, phi_seed)
            if cached_m_seed is not None:
                M_seed = cached_m_seed.to(phi_seed.dtype)
            elif m.povm_head_seed_delta is not None:
                M_seed = l2_normalize(m.wte.weight + m.povm_head_seed_delta, dim=1, eps=m.eps).to(phi_seed.dtype)
            else:
                M_seed = l2_normalize(m.wte.weight, dim=1, eps=m.eps).to(phi_seed.dtype)
            inv_t = torch.clamp(m.tied_inv_temp, min=1.0, max=60.0)
        else:
            psi_final = l2_normalize(psi, dim=-1, eps=m.eps)
            if kraus_beams > 1:
                phi_seed = psi_final
            else:
                phi_seed = m.hilbert_lift.collapse_to_seed(psi_final)
                if m.seed_adapter is not None:
                    phi_seed = m.seed_adapter(psi_final, phi_seed)
            M_seed = (
                cached_m_seed.to(phi_seed.dtype)
                if cached_m_seed is not None
                else l2_normalize(m.wte.weight, dim=1, eps=m.eps).to(phi_seed.dtype)
            )
            inv_t = torch.tensor(float(m._dual_inv_temp), device=phi_seed.device, dtype=torch.float32)

        if (
            getattr(m, "sparse_density_layer", None) is not None
            and getattr(m, "sparse_povm_readout", None) is not None
            and m.sparse_density_layer.density_rank > 1
        ):
            d_lay = m.sparse_density_layer
            q_sum = psi[..., : d_lay.num_qubits]
            p_w = F.softmax(d_lay.purity_router(q_sum), dim=-1)
            purity_t = (p_w.float().square()).sum(dim=-1, keepdim=True)
            min_p = 1.0 / float(d_lay.density_rank)
            norm_purity = (purity_t - min_p) / max(1e-6, 1.0 - min_p)
            gain = torch.tanh(m.sparse_povm_readout.purity_temp_gain.float())
            inv_t = torch.clamp(inv_t.float() * (1.0 + 0.25 * gain * (norm_purity - 0.5)), min=1.0, max=60.0)

        if getattr(m, "purity_gate", None) is not None:
            inv_t = m.purity_gate(psi, inv_t)  # type: ignore[misc]
        return phi_seed, M_seed, inv_t, layer_states

    is_low_vram_gpu = bool(
        torch.cuda.is_available()
        and (torch.cuda.get_device_properties(device).total_memory < 24 * (1024**3))
        and (batch_size * seq_len >= 65536)
    )
    if is_low_vram_gpu:
        compiled_backbone = lambda m, idx: forward_backbone(m, idx, False)[:3]
        compiled_classical_features = lambda m, idx: m.forward_features(idx, return_layer_states=False)[0]
    else:
        compiled_backbone = torch.compile(
            lambda m, idx: forward_backbone(m, idx, False)[:3],
            mode="default",
        )
        compiled_classical_features = torch.compile(
            lambda m, idx: m.forward_features(idx, return_layer_states=False)[0],
            mode="default",
        )
    compiled_chunk_ce = torch.compile(_chunk_ce_fn, mode="default")
    compiled_classical_chunk_ce = torch.compile(_classical_chunk_ce_fn, mode="default")

    def _kraus_chunk_ce(zc: torch.Tensor, m_seed: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
        logits_c = model.kraus_readout(zc, m_seed)  # type: ignore[misc]
        if getattr(model, "povm_log_prior", None) is not None:
            logits_c = logits_c + model.povm_log_prior.to(logits_c.dtype)  # type: ignore[union-attr]
        v_act = min(FINEWEB_GPT2_VOCAB_SIZE, int(logits_c.size(-1)))
        return F.cross_entropy(logits_c[:, :v_act].float(), yc)

    def _chunked_povm_ce(
        phi_2d: torch.Tensor,
        M_seed: torch.Tensor,
        inv_t: torch.Tensor,
        flat_y: torch.Tensor,
        log_prior: torch.Tensor | None = None,
        cur_step: int = 0,
        use_sampled: bool = False,
    ) -> torch.Tensor:
        if getattr(model, "use_last_layer_softmax", False):
            eff_cap = float(getattr(model, "logit_softcap", 15.0))
            if use_sampled and sampled_softmax_candidates > 0:
                return sampled_softcap_linear_ce(
                    phi_2d,
                    M_seed,
                    flat_y,
                    num_candidates=sampled_softmax_candidates,
                    step=cur_step,
                    softcap=eff_cap,
                    chunk_tokens=16384,
                    bias=log_prior,
                )
            return direct_chunked_softcap_linear_ce(
                phi_2d,
                M_seed,
                flat_y,
                softcap=eff_cap,
                chunk_size=16384,
                valid_vocab_size=FINEWEB_GPT2_VOCAB_SIZE,
                bias=log_prior,
            )
        if use_sampled and sampled_softmax_candidates > 0 and kraus_beams <= 1:
            return sampled_povm_ce(
                phi_2d,
                M_seed,
                inv_t,
                flat_y,
                num_candidates=sampled_softmax_candidates,
                step=cur_step,
                log_prior=log_prior,
                chunk_tokens=16384,
            )
        if fused_povm_tile_size > 0 and kraus_beams <= 1 and inv_t.ndim < 2 and log_prior is None:
            return fused_tiled_povm_cross_entropy(
                phi_2d, M_seed, inv_t, flat_y, tile_size=fused_povm_tile_size
            )
        if direct_chunked_ce and not (kraus_beams > 1 and getattr(model, "kraus_readout", None) is not None):
            return direct_chunked_povm_ce(
                phi_2d,
                M_seed,
                inv_t,
                flat_y,
                log_prior=log_prior,
                chunk_size=16384,
                valid_vocab_size=FINEWEB_GPT2_VOCAB_SIZE,
            )
        n_tok = phi_2d.size(0)
        chunk_sz = min(4096 if kraus_beams > 1 else 16384, n_tok)
        n_chunks = max(1, n_tok // chunk_sz)
        inv_t_flat = inv_t.reshape(-1, 1) if inv_t.ndim >= 2 else None
        total_loss = torch.zeros((), device=phi_2d.device, dtype=torch.float32)
        for i in range(n_chunks):
            zc = phi_2d[i * chunk_sz : (i + 1) * chunk_sz]
            yc = flat_y[i * chunk_sz : (i + 1) * chunk_sz]
            inv_t_c = inv_t_flat[i * chunk_sz : (i + 1) * chunk_sz] if inv_t_flat is not None else inv_t
            if kraus_beams > 1 and getattr(model, "kraus_readout", None) is not None:
                total_loss = total_loss + checkpoint(_kraus_chunk_ce, zc, M_seed, yc, use_reentrant=False)
            else:
                total_loss = total_loss + checkpoint(
                    compiled_chunk_ce, zc, M_seed, inv_t_c, yc, log_prior, use_reentrant=False
                )
        return total_loss / float(n_chunks)

    def compute_step_loss(
        m: QuantumBornGPT | ClassicalModdedGPT,
        idx: torch.Tensor,
        targets: torch.Tensor,
        cur_step: int = 0,
        use_sampled: bool = False,
    ) -> torch.Tensor:
        if isinstance(m, ClassicalModdedGPT):
            x_final = compiled_classical_features(m, idx)
            x_2d = x_final.reshape(-1, m.embed_dim)
            flat_y = targets.reshape(-1)
            if use_sampled and sampled_softmax_candidates > 0:
                return sampled_softcap_linear_ce(
                    x_2d,
                    m.lm_head.weight,
                    flat_y,
                    num_candidates=sampled_softmax_candidates,
                    step=cur_step,
                    softcap=m.logit_softcap,
                    chunk_tokens=16384,
                )
            if direct_chunked_ce:
                return direct_chunked_softcap_linear_ce(
                    x_2d,
                    m.lm_head.weight,
                    flat_y,
                    softcap=m.logit_softcap,
                    chunk_size=16384,
                    valid_vocab_size=FINEWEB_GPT2_VOCAB_SIZE,
                )
            n_tok = x_2d.size(0)
            chunk_sz = min(16384, n_tok)
            n_chunks = max(1, n_tok // chunk_sz)
            total_loss = torch.zeros((), device=x_2d.device, dtype=torch.float32)
            for i in range(n_chunks):
                xc = x_2d[i * chunk_sz : (i + 1) * chunk_sz]
                yc = flat_y[i * chunk_sz : (i + 1) * chunk_sz]
                total_loss = total_loss + checkpoint(
                    compiled_classical_chunk_ce,
                    xc,
                    m.lm_head.weight,
                    m.logit_softcap,
                    yc,
                    use_reentrant=False,
                )
            return total_loss / float(n_chunks)

        phi_3d, M_seed, inv_t = compiled_backbone(m, idx)
        out_dim = int(phi_3d.size(-1))
        log_prior = getattr(m, "povm_log_prior", None)
        if (
            getattr(m, "use_two_beam_kboson_readout", False)
            and getattr(m, "hierarchical_povm_readout", None) is not None
            and getattr(m, "hierarchical_qubit_embed", None) is not None
        ):
            psi_beam2 = getattr(m.blocks[-1], "last_beam2", None) if len(m.blocks) > 0 else None
            base_head = m.povm_readout.povm_head if m.povm_readout is not None else None
            psi_s, psi_d, m_s, m_d, k_eff, lam_diff = (
                m.hierarchical_povm_readout.get_two_beam_kboson_factors(
                    m.hierarchical_qubit_embed,
                    phi_3d,
                    psi_beam2=psi_beam2,
                    base_povm_head=base_head,
                )
            )
            loss_t1 = direct_chunked_two_beam_kboson_ce(
                psi_s.reshape(-1, out_dim),
                psi_d.reshape(-1, psi_d.size(-1)),
                m_s,
                m_d,
                inv_t,
                k_eff,
                lam_diff,
                targets.reshape(-1),
                log_prior=log_prior,
                chunk_size=16384,
                valid_vocab_size=FINEWEB_GPT2_VOCAB_SIZE,
            )
        else:
            loss_t1 = _chunked_povm_ce(
                phi_3d.reshape(-1, out_dim),
                M_seed,
                inv_t,
                targets.reshape(-1),
                log_prior=log_prior,
                cur_step=cur_step,
                use_sampled=use_sampled,
            )
        total_loss = loss_t1
        if m.mtp_horizons > 1 and m.mtp_rotations is not None:
            mtp_dim = int(phi_3d.size(-1))
            if kraus_beams > 1 and m.readout_mode != "asymmetric" and not getattr(m, "use_sparse_64qubit", False):
                phi_seed_3d = m.hilbert_lift.collapse_to_seed(phi_3d)
                if m.seed_adapter is not None:
                    phi_seed_3d = m.seed_adapter(phi_3d, phi_seed_3d)
            else:
                phi_seed_3d = phi_3d
            T_cur = idx.size(1)
            for h_idx, u_rot in enumerate(m.mtp_rotations, start=1):
                T_avail = T_cur - h_idx
                if T_avail <= 0:
                    break
                if m.mtp_stride > 1 and T_avail > m.mtp_stride:
                    offset = (h_idx - 1) % m.mtp_stride
                    state_slice = slice(offset, T_avail, m.mtp_stride)
                    target_slice = slice(h_idx + offset, T_cur, m.mtp_stride)
                else:
                    state_slice = slice(0, T_avail)
                    target_slice = slice(h_idx, T_cur)
                z_h = l2_normalize(
                    F.linear(phi_seed_3d[:, state_slice, :], u_rot.to(phi_seed_3d.dtype)),
                    dim=-1,
                    eps=m.eps,
                ).reshape(-1, mtp_dim)
                y_h = targets[:, target_slice].reshape(-1)
                M_aux = M_seed.detach() if m.mtp_detach_readout else M_seed
                if m.mtp_decoupled_temp and m.mtp_inv_temps is not None:
                    beta_h = torch.clamp(m.mtp_inv_temps[h_idx - 1], min=1.0, max=60.0)
                elif inv_t.ndim >= 2:
                    beta_h = inv_t[:, state_slice, :].reshape(-1, 1)
                else:
                    beta_h = inv_t
                if fused_povm_tile_size > 0 and beta_h.ndim < 2 and log_prior is None:
                    loss_h = fused_tiled_povm_cross_entropy(
                        z_h, M_aux, beta_h, y_h, tile_size=fused_povm_tile_size
                    )
                elif direct_chunked_ce:
                    loss_h = direct_chunked_povm_ce(
                        z_h,
                        M_aux,
                        beta_h,
                        y_h,
                        log_prior=log_prior,
                        chunk_size=16384,
                        valid_vocab_size=FINEWEB_GPT2_VOCAB_SIZE,
                    )
                else:
                    n_tok_h = z_h.size(0)
                    c_sz = min(16384, n_tok_h)
                    n_ch = max(1, n_tok_h // c_sz)
                    loss_h = torch.zeros((), device=z_h.device, dtype=torch.float32)
                    for ci in range(n_ch):
                        beta_hc = beta_h[ci * c_sz : (ci + 1) * c_sz] if beta_h.ndim >= 2 else beta_h
                        loss_h = loss_h + checkpoint(
                            compiled_chunk_ce,
                            z_h[ci * c_sz : (ci + 1) * c_sz],
                            M_aux,
                            beta_hc,
                            y_h[ci * c_sz : (ci + 1) * c_sz],
                            log_prior,
                            use_reentrant=False,
                        )
                    loss_h = loss_h / float(n_ch)
                total_loss = total_loss + (m.mtp_loss_weight / float(h_idx)) * loss_h
        if (overcomplete_frame or overcomplete_two_beam) and not sparse_density_layer:
            f_pen = sum(
                b.two_beam_mlp.frame_penalty()  # type: ignore[union-attr]
                for b in m.blocks
                if hasattr(b.two_beam_mlp, "frame_penalty")
            )
            total_loss = total_loss + 0.005 * f_pen
        return total_loss

    @torch.no_grad()
    def compute_isometry_error(model: nn.Module) -> float:
        errs: list[float] = []
        for name, p in model.named_parameters():
            if p.ndim == 2 and not any(
                k in name
                for k in (
                    "wte",
                    "qubit_wte",
                    "unigram_qubits",
                    "higher_states",
                    "povm_odd_qubits",
                    "vocab_higher_affinity",
                    "hierarchical_povm_readout",
                    "hier_even_proj",
                    "token_radial_scale",
                    "readout_rms_gain",
                    "qubit_readout_gain",
                    "kraus",
                    "kboson",
                    "diffuse",
                    "causal_mem",
                    "block_even_odd_mix",
                    "block_out_parity_mix",
                    "block_hier_gates",
                    "unigram_sparse_vals",
                    "ngram_sparse_vals",
                    "branch_spectral_filters",
                    "purity_router",
                    "povm_head",
                    "lm_head",
                    "purity_gate",
                    "embed",
                    "value_embeds",
                    "ngram",
                    "mudd",
                    "input_gate",
                    "output_gate",
                    "gate_proj",
                    "unet_scales",
                    "unet_lambdas",
                    "smear",
                    "beam_",
                )
            ):
                w = p.detach().float()
                r, c = w.shape
                if r >= c:
                    gram = w.t() @ w
                    eye = torch.eye(c, device=w.device, dtype=torch.float32)
                    err = torch.norm(gram - eye, p="fro") / math.sqrt(c)
                else:
                    gram = w @ w.t()
                    eye = torch.eye(r, device=w.device, dtype=torch.float32)
                    err = torch.norm(gram - eye, p="fro") / math.sqrt(r)
                errs.append(float(err.item()))
        return float(np.mean(errs)) if errs else 0.0

    val_tokens_gpu_cache: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def evaluate_validation(
        model: QuantumBornGPT | ClassicalModdedGPT,
        val_tokens: np.ndarray,
        eval_batch_size: int = 64,
        block_size: int = 512,
        num_batches: int | None = None,
    ) -> dict[str, float | list[float]]:
        """Evaluate held-out validation loss matching KellerJordan/modded-nanogpt.

        Streams `val_tokens` (`fineweb_val_000000.bin`) sequentially from `pos = 0`
        (`align_to_bos=False`) across `val_steps = val_tokens_budget // (eval_batch_size * block_size)`
        batches (`10,485,760` tokens = `10 * 2^20` by default) and computes `float32`
        cross-entropy over the exact `50,257` GPT-2 vocabulary (`0..50256`).
        """
        model.eval()
        top1_hits = 0
        top5_hits = 0
        total_acc_toks = 0
        all_ranks: list[np.ndarray] = []
        ent_layers: list[float] = []
        seed_eff_rank = 0.0
        seed_pr = 0.0
        mean_beta_eval = 0.0

        stride = max(1, int(eval_batch_size) * int(block_size))
        eff_val_budget = (
            int(num_batches) * stride
            if (num_batches is not None and int(num_batches) > 0)
            else int(val_tokens_budget)
        )
        eff_batches = compute_modded_nanogpt_val_steps(
            len(val_tokens),
            batch_size=eval_batch_size,
            seq_len=block_size,
            val_tokens_budget=eff_val_budget,
        )
        req_toks = eff_batches * stride + 1
        cache_key = id(val_tokens)
        if cache_key not in val_tokens_gpu_cache or val_tokens_gpu_cache[cache_key].numel() < req_toks:
            if len(val_tokens) >= req_toks:
                slice_np = val_tokens[:req_toks]
            else:
                pad_n = req_toks - len(val_tokens)
                tail_np = np.full((pad_n,), int(val_tokens[0]) if len(val_tokens) > 0 else 50256, dtype=np.uint16)
                slice_np = np.concatenate([val_tokens, tail_np], axis=0)
            val_tokens_gpu_cache[cache_key] = torch.from_numpy(np.ascontiguousarray(slice_np, dtype=np.uint16)).to(
                device=device, non_blocking=True
            )
        val_gpu = val_tokens_gpu_cache[cache_key]

        total_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
        cached_m_seed: torch.Tensor | None = None
        v_active = min(FINEWEB_GPT2_VOCAB_SIZE, int(getattr(model, "vocab_size", FINEWEB_GPT2_VOCAB_SIZE)))

        for b_idx in range(eff_batches):
            base = b_idx * stride
            buf_gpu = val_gpu[base : base + stride + 1].to(torch.int64)
            x = buf_gpu[:-1].view(eval_batch_size, block_size)
            y = buf_gpu[1:].view(eval_batch_size, block_size)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if isinstance(model, ClassicalModdedGPT):
                    if b_idx == 0:
                        x_final, l_states = model.forward_features(x, return_layer_states=True)
                        keep_5 = (0, 1, 2, 3, 4)
                        for st in l_states:
                            if model.embed_dim == 1024:
                                sub_psi = l2_normalize(st.reshape(-1, model.embed_dim)[:128].float(), dim=-1)
                            elif model.embed_dim > 1024:
                                sub_psi = l2_normalize(st.reshape(-1, model.embed_dim)[:128, :1024].float(), dim=-1)
                            else:
                                sub_psi = l2_normalize(
                                    F.pad(st.reshape(-1, model.embed_dim)[:128].float(), (0, 1024 - model.embed_dim)),
                                    dim=-1,
                                )
                            ent_v = bipartite_entanglement_entropy(sub_psi, num_qubits=10, keep_qubits=keep_5)
                            ent_layers.append(round(float(ent_v.mean().item()), 4))
                        x_for_rank = l2_normalize(x_final.reshape(-1, model.embed_dim)[:512].float(), dim=-1)
                        seed_eff_rank = float(effective_quantum_rank(x_for_rank).item())
                        seed_pr = float(purity_participation_ratio(x_for_rank).item())
                    else:
                        x_final = compiled_classical_features(model, x)

                    x_flat = x_final.reshape(-1, model.embed_dim)
                    flat_y = y.reshape(-1)
                    loss_b = direct_chunked_softcap_linear_ce(
                        x_flat,
                        model.lm_head.weight,
                        flat_y,
                        softcap=model.logit_softcap,
                        chunk_size=16384,
                        valid_vocab_size=v_active,
                    )
                    total_loss_gpu = total_loss_gpu + loss_b.to(torch.float64)

                    if b_idx < 2:
                        xc_diag = x_flat[:8192]
                        yc_diag = flat_y[:8192]
                        logits_diag = model.compute_logits(xc_diag)[:, :v_active]
                        if b_idx == 0:
                            mean_beta_eval = float(logits_diag.detach().float().std(dim=-1).mean().item())
                        top5 = torch.topk(logits_diag, k=5, dim=-1).indices
                        top1_hits += int((top5[:, 0] == yc_diag).sum().item())
                        top5_hits += int((top5 == yc_diag.unsqueeze(1)).any(dim=-1).sum().item())
                        total_acc_toks += int(yc_diag.numel())

                        sub_logits = logits_diag[:1024]
                        sub_y = yc_diag[:1024]
                        target_scores = sub_logits.gather(1, sub_y.unsqueeze(1))
                        ranks = (sub_logits > target_scores).sum(dim=-1) + 1
                        all_ranks.append(ranks.cpu().numpy())
                    continue

                if b_idx == 0:
                    phi_3d, M_seed, inv_t, l_states = forward_backbone(model, x, return_layer_states=True)
                    cached_m_seed = M_seed.detach()
                    mean_beta_eval = float(inv_t.detach().float().mean().item())
                    keep_5 = (0, 1, 2, 3, 4)
                    for st in l_states:
                        if model.embed_dim == 1024:
                            sub_psi = st.reshape(-1, model.embed_dim)[:128].float()
                        elif model.embed_dim > 1024:
                            sub_psi = l2_normalize(st.reshape(-1, model.embed_dim)[:128, :1024].float(), dim=-1)
                        else:
                            sub_psi = l2_normalize(
                                F.pad(st.reshape(-1, model.embed_dim)[:128].float(), (0, 1024 - model.embed_dim)),
                                dim=-1,
                            )
                        ent_v = bipartite_entanglement_entropy(sub_psi, num_qubits=10, keep_qubits=keep_5)
                        ent_layers.append(round(float(ent_v.mean().item()), 4))
                else:
                    phi_3d, M_seed, inv_t, _ = forward_backbone(
                        model, x, return_layer_states=False, cached_m_seed=cached_m_seed
                    )

                out_dim = int(phi_3d.size(-1))
                phi_flat = phi_3d.reshape(-1, out_dim)
                inv_t_flat = inv_t.reshape(-1, 1) if inv_t.ndim >= 2 else None
                if b_idx == 0:
                    if kraus_beams > 1 and model.readout_mode != "asymmetric" and not getattr(model, "use_sparse_64qubit", False):
                        phi_for_rank = model.hilbert_lift.collapse_to_seed(phi_flat[:512]).float()
                        if model.seed_adapter is not None:
                            phi_for_rank = model.seed_adapter(phi_flat[:512].float(), phi_for_rank)
                    else:
                        phi_for_rank = l2_normalize(phi_flat[:512].float(), dim=-1)
                    seed_eff_rank = float(effective_quantum_rank(phi_for_rank).item())
                    seed_pr = float(purity_participation_ratio(phi_for_rank).item())
                flat_y = y.reshape(-1)
                log_prior_eval = getattr(model, "povm_log_prior", None)
                if (
                    getattr(model, "use_two_beam_kboson_readout", False)
                    and getattr(model, "hierarchical_povm_readout", None) is not None
                    and getattr(model, "hierarchical_qubit_embed", None) is not None
                ):
                    psi_beam2_eval = getattr(model.blocks[-1], "last_beam2", None) if len(model.blocks) > 0 else None
                    base_head_eval = model.povm_readout.povm_head if model.povm_readout is not None else None
                    psi_s_e, psi_d_e, m_s_e, m_d_e, k_eff_e, lam_diff_e = (
                        model.hierarchical_povm_readout.get_two_beam_kboson_factors(
                            model.hierarchical_qubit_embed,
                            phi_3d,
                            psi_beam2=psi_beam2_eval,
                            base_povm_head=base_head_eval,
                        )
                    )
                    loss_b = direct_chunked_two_beam_kboson_ce(
                        psi_s_e.reshape(-1, out_dim),
                        psi_d_e.reshape(-1, psi_d_e.size(-1)),
                        m_s_e,
                        m_d_e,
                        inv_t,
                        k_eff_e,
                        lam_diff_e,
                        flat_y,
                        log_prior=log_prior_eval,
                        chunk_size=16384,
                        valid_vocab_size=v_active,
                    )
                    total_loss_gpu = total_loss_gpu + loss_b.to(torch.float64)
                elif getattr(model, "use_last_layer_softmax", False):
                    eff_cap = float(getattr(model, "logit_softcap", 15.0))
                    loss_b = direct_chunked_softcap_linear_ce(
                        phi_flat,
                        M_seed,
                        flat_y,
                        softcap=eff_cap,
                        chunk_size=16384,
                        valid_vocab_size=v_active,
                        bias=log_prior_eval,
                    )
                    total_loss_gpu = total_loss_gpu + loss_b.to(torch.float64)
                elif kraus_beams > 1 and getattr(model, "kraus_readout", None) is not None:
                    chunk_sz = 8192
                    b_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
                    n_c = 0
                    for start in range(0, phi_flat.size(0), chunk_sz):
                        end = min(start + chunk_sz, phi_flat.size(0))
                        zc = phi_flat[start:end]
                        yc = flat_y[start:end]
                        logits_c = model.kraus_readout(zc, M_seed)[:, :v_active]
                        if log_prior_eval is not None:
                            logits_c = logits_c + log_prior_eval[:v_active].to(logits_c.dtype)
                        b_loss_sum = b_loss_sum + F.cross_entropy(logits_c.float(), yc).to(torch.float64)
                        n_c += 1
                    total_loss_gpu = total_loss_gpu + (b_loss_sum / float(max(1, n_c)))
                else:
                    loss_b = direct_chunked_povm_ce(
                        phi_flat,
                        M_seed,
                        inv_t,
                        flat_y,
                        log_prior=log_prior_eval,
                        chunk_size=16384,
                        valid_vocab_size=v_active,
                    )
                    total_loss_gpu = total_loss_gpu + loss_b.to(torch.float64)

                if b_idx < 2:
                    zc_diag = phi_flat[:8192]
                    yc_diag = flat_y[:8192]
                    inv_t_diag = inv_t_flat[:8192] if inv_t_flat is not None else inv_t
                    if (
                        getattr(model, "use_two_beam_kboson_readout", False)
                        and getattr(model, "hierarchical_povm_readout", None) is not None
                        and getattr(model, "hierarchical_qubit_embed", None) is not None
                    ):
                        psi_s_diag = psi_s_e.reshape(-1, out_dim)[:8192]
                        psi_d_diag = psi_d_e.reshape(-1, psi_d_e.size(-1))[:8192]
                        logits_diag = compute_two_beam_kboson_logits(
                            psi_s_diag,
                            psi_d_diag,
                            m_s_e[:v_active],
                            m_d_e[:v_active],
                            inv_t_diag,
                            k_eff_e,
                            lam_diff_e,
                            log_prior=log_prior_eval[:v_active] if log_prior_eval is not None else None,
                        )
                    elif getattr(model, "use_last_layer_softmax", False):
                        eff_cap = float(getattr(model, "logit_softcap", 15.0))
                        raw_diag = F.linear(zc_diag, M_seed[:v_active])
                        if log_prior_eval is not None:
                            raw_diag = raw_diag + log_prior_eval[:v_active].to(raw_diag.dtype)
                        logits_diag = eff_cap * torch.tanh(raw_diag / eff_cap) if eff_cap > 0.0 else raw_diag
                        if b_idx == 0:
                            mean_beta_eval = float(logits_diag.detach().float().std(dim=-1).mean().item())
                    else:
                        if kraus_beams > 1 and getattr(model, "kraus_readout", None) is not None:
                            logits_diag = model.kraus_readout(zc_diag, M_seed)[:, :v_active]
                        else:
                            logits_diag = (F.linear(zc_diag, M_seed[:v_active]) * inv_t_diag.to(zc_diag.dtype))
                        if log_prior_eval is not None:
                            logits_diag = logits_diag + log_prior_eval[:v_active].to(logits_diag.dtype)

                    top5 = torch.topk(logits_diag, k=5, dim=-1).indices
                    top1_hits += int((top5[:, 0] == yc_diag).sum().item())
                    top5_hits += int((top5 == yc_diag.unsqueeze(1)).any(dim=-1).sum().item())
                    total_acc_toks += int(yc_diag.numel())

                    sub_logits = logits_diag[:1024]
                    sub_y = yc_diag[:1024]
                    target_scores = sub_logits.gather(1, sub_y.unsqueeze(1))
                    ranks = (sub_logits > target_scores).sum(dim=-1) + 1
                    all_ranks.append(ranks.cpu().numpy())

        model.train()
        val_ce = float((total_loss_gpu / float(max(1, eff_batches))).item())
        val_ppl = float(math.exp(min(val_ce, 20.0)))
        med_rank = float(np.median(np.concatenate(all_ranks))) if all_ranks else -1.0
        iso_err = compute_isometry_error(model)
        if isinstance(model, ClassicalModdedGPT):
            g_val = round(math.sqrt(float(model.embed_dim // model.num_heads)), 4)
            q_scales: dict[str, float | list[float]] = {
                "gamma_head_mean": g_val,
                "gamma_head_min": g_val,
                "gamma_head_max": g_val,
                "gamma_layer_means": [g_val] * model.num_layers,
                "povm_beta": round(mean_beta_eval, 4),
                "unet_gate_scales": [],
            }
        else:
            q_scales = model_quantum_scale_telemetry(model)
            if mean_beta_eval > 0.0:
                q_scales["povm_beta"] = round(mean_beta_eval, 4)
        ent_mean = round(float(np.mean(ent_layers)), 4) if ent_layers else 0.0
        return {
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "val_tokens_evaluated": int(eff_batches * stride),
            "top1_acc": 100.0 * top1_hits / max(1, total_acc_toks),
            "top5_acc": 100.0 * top5_hits / max(1, total_acc_toks),
            "med_rank": med_rank,
            "iso_err": iso_err,
            "entanglement_5v5_mean": ent_mean,
            "entanglement_5v5_layers": ent_layers,
            "seed_eff_rank": round(seed_eff_rank, 2),
            "seed_participation_ratio": round(seed_pr, 2),
            **q_scales,
        }

    print("=" * 90, flush=True)
    print("FULL 100M-TOKEN MULTI-EPOCH QUANTUM SLOWRUN BENCHMARK — GPU BACKEND", flush=True)
    print(
        f"Device: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | Fused Triton Flash-Born + torch.compile",
        flush=True,
    )
    print("=" * 90, flush=True)

    B = int(batch_size)
    G_acc = max(1, int(grad_accum))
    T = int(seq_len)
    tok_per_step = B * G_acc * T
    curriculum: ProgressiveSequenceCurriculum | None = None
    if seq_curriculum_str.strip():
        seq_tuple = tuple(int(x.strip()) for x in seq_curriculum_str.split(",") if x.strip())
        curriculum = ProgressiveSequenceCurriculum(tokens_per_step=tok_per_step, seq_lengths=seq_tuple)

    stream: FineWebMultiShardStream | None = None
    if num_shards > 1:
        stream = FineWebMultiShardStream(
            data_dir=data_dir,
            num_shards=num_shards,
            prefetch_async=True,
            synthetic_tokens_per_shard=synthetic_tokens,
        )
        val_tokens = stream.ensure_val_shard()
        first_shard, first_wait_ms = stream.get_shard(1)
        train_tokens = first_shard
        print(
            f"Streaming {num_shards} Train Shards ({len(first_shard):,} tokens/shard, first wait {first_wait_ms:.1f}ms) | Val Shard: {len(val_tokens):,} tokens",
            flush=True,
        )
        steps_per_shard = max(1, (len(first_shard) - max(4, mtp_horizons + 1)) // tok_per_step)
        steps_per_epoch = steps_per_shard * num_shards
        total_steps = steps_per_epoch * num_epochs
        eval_every = max(1, int(round((eval_every_tokens_m * 1e6) / tok_per_step)))
    else:
        train_tokens, val_tokens = _load_or_generate_shards(data_dir, synthetic_tokens=synthetic_tokens)
        print(f"Loaded Train Shard: {len(train_tokens):,} tokens | Val Shard: {len(val_tokens):,} tokens", flush=True)
        steps_per_shard = max(1, (len(train_tokens) - max(4, mtp_horizons + 1)) // tok_per_step)
        steps_per_epoch = steps_per_shard
        total_steps = steps_per_epoch * num_epochs
        eval_every = max(1, steps_per_epoch // 2)

    torch.manual_seed(42)
    np.random.seed(42)

    skip_attn_tuple = tuple(int(x.strip()) for x in skip_attn_layers_str.split(",") if x.strip())
    max_T = max(curriculum.seq_lengths) if curriculum is not None else T
    if architecture == "classical_modded":
        eff_hidden_dim = (
            int(round(embed_dim * frame_expansion))
            if (overcomplete_two_beam or overcomplete_frame)
            else hidden_dim
        )
        init_window_size = 128 if (progressive_window and sliding_window <= 0) else sliding_window
        eff_zero_init = bool(zero_init_classical or leaderboard_exact_classical)
        eff_softcap = 15.0 if leaderboard_exact_classical else 30.0
        model: QuantumBornGPT | ClassicalModdedGPT = ClassicalModdedGPT(
            vocab_size=50304,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=eff_hidden_dim,
            block_size=max_T,
            use_bigram_smear=not leaderboard_exact_classical,
            use_unet_interferometry=True,
            unet_skip_mode="scalar" if leaderboard_exact_classical else unet_skip_mode,
            use_fused_qkv=True if leaderboard_exact_classical else fused_qkv,
            num_value_embeddings=3 if (leaderboard_exact_classical and num_value_embeddings == 0) else num_value_embeddings,
            skip_attn_layers=skip_attn_tuple,
            zero_half_rope=True if leaderboard_exact_classical else zero_half_rope,
            window_size=init_window_size,
            doc_mask_attn=doc_mask_attn,
            value_residual_mixing=True,
            input_highway_mixing=True,
            depth_scaled_residual=False if leaderboard_exact_classical else depth_scaled_residual,
            logit_softcap=eff_softcap,
            zero_init_residual_and_head=eff_zero_init,
            true_layer0_v_residual=False if leaderboard_exact_classical else true_layer0_v_residual,
            leaderboard_exact=leaderboard_exact_classical,
            use_ngram_hash=ngram_hash,
            ngram_vocab_size=ngram_vocab_size,
            use_mudd_and_gates=mudd_and_gates,
            use_smear_gate=smear_gate,
        ).to(device)
        if ppmi_svd_init:
            n_init_sh = max(1, min(int(ppmi_svd_shards), int(num_shards)))
            init_shards = (
                [stream.get_shard(s)[0] for s in range(1, n_init_sh + 1)]
                if stream is not None
                else [train_tokens]
            )
            c_path = (
                data_dir / f"ppmi_svd_classical_{embed_dim}d_{n_init_sh}sh.pt"
                if synthetic_tokens <= 0
                else None
            )
            ppmi_stats = initialize_classical_with_ppmi_svd(
                model,
                token_shards=init_shards,
                alpha_ppmi=ppmi_svd_alpha,
                cache_path=c_path,
                device=device,
            )
            print(f"Initialized ClassicalModdedGPT with {n_init_sh}-Shard PPMI-SVD: {ppmi_stats}", flush=True)
        if leaderboard_exact_classical:
            model.wte.bfloat16()
            if model.value_embeds is not None:
                model.value_embeds.bfloat16()
            if model.ngram_embed is not None:
                model.ngram_embed.bfloat16()
        if fp8_linear:
            convert_linear_layers_to_fp8(model)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"Architecture: ClassicalModdedGPT (d={embed_dim}, H={num_heads}, L={num_layers}, d_ff={eff_hidden_dim}(ReLU^2), LeaderboardExact={leaderboard_exact_classical}, ProgWin={progressive_window}, ZeroInit={eff_zero_init}, TrueV0={model.true_layer0_v_residual}, UNetSkip={model.unet_skip_mode}, FusedQKV={model.use_fused_qkv}, ValEmbeds={model.num_value_embeddings}, SkipAttn={model.skip_attn_layers}, ZeroHalfRoPE={model.zero_half_rope}, Win={model.window_size}, DocMask={doc_mask_attn}, PPMISVD={ppmi_svd_init}, NgramHash={ngram_hash}, MUDD={mudd_and_gates}, SmearGate={smear_gate}, StiefelPen={classical_stiefel_penalty:g}, OptMode={optimizer_mode}, DirectCE={direct_chunked_ce}, Readout=untied_softcap{eff_softcap:g}, ResScale={model.residual_scale:.4f}) | Params: {n_params:,}",
            flush=True,
        )
    else:
        eff_q_embed_dim = 2048 if (sparse_64qubit and embed_dim == 1024) else embed_dim
        model = QuantumBornGPT(
            vocab_size=50304,
            embed_dim=eff_q_embed_dim,
            seed_dim=seed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            block_size=max_T,
            readout_mode=readout_mode,
            learnable_temperature=True,
            value_residual_mixing=True,
            input_highway_mixing=True,
            use_bigram_smear=True,
            use_qk_wave_filter=True,
            use_unet_interferometry=True,
            unet_skip_mode=unet_skip_mode,
            use_fused_qkv=fused_qkv,
            num_value_embeddings=num_value_embeddings,
            skip_attn_layers=skip_attn_tuple,
            zero_half_rope=zero_half_rope,
            window_size=sliding_window,
            doc_mask_attn=doc_mask_attn,
            use_two_beam_mlp=True,
            frame_expansion=frame_expansion,
            depth_scaled_residual=depth_scaled_residual,
            depolarizing_jitter=0.015,
            use_flash_born=True,
            use_overcomplete_seed_adapter=overcomplete_seed_adapter,
            use_two_beam_seed_adapter=two_beam_seed_adapter,
            use_dynamic_povm_beta=dynamic_povm_beta,
            use_ortho_decouple=ortho_decouple,
            use_povm_log_prior=povm_log_prior,
            use_gated_spherical_residual=gated_spherical_residual,
            use_purified_attn=purified_born_attn,
            true_layer0_v_residual=true_layer0_v_residual,
            use_ngram_hash=ngram_hash,
            ngram_vocab_size=ngram_vocab_size,
            use_mudd_and_gates=mudd_and_gates,
            use_smear_gate=smear_gate,
            use_sparse_64qubit=sparse_64qubit,
            num_qubits=num_qubits,
            sparse_ngram_k=sparse_ngram_k,
            use_sparse_density_layer=sparse_density_layer,
            density_matrix_rank=density_matrix_rank,
            use_fock_coupler=not no_fock_coupler,
            use_hierarchical_16qubit=hierarchical_16qubit,
            qubit_registers=qubit_registers,
            qubit_amplitudes=qubit_amplitudes,
            num_higher_states=num_higher_states,
            use_block_hierarchy_bridge=not no_block_hierarchy_bridge,
            use_last_layer_softmax=last_layer_softmax,
            use_two_beam_kboson_readout=two_beam_kboson_readout,
            diffuse_rank=diffuse_rank,
            logit_softcap=logit_softcap,
            mtp_horizons=mtp_horizons,
            mtp_loss_weight=mtp_weight,
            mtp_stride=mtp_stride,
            mtp_detach_readout=mtp_detach_readout,
            mtp_decoupled_temp=mtp_decoupled_temp,
        ).to(device)
        if kraus_beams > 1:
            model.kraus_readout = MultiBeamKrausReadout(
                vocab_size=50304,
                embed_dim=eff_q_embed_dim,
                seed_dim=seed_dim,
                num_beams=kraus_beams,
                beam_adapter_rank=16,
                adapter_init_scale=1.0,
            ).to(device)
        if overcomplete_two_beam and not sparse_density_layer:
            target_iso = default_overcomplete_target_iso_scale(frame_expansion)
            for l_idx, blk in enumerate(model.blocks):
                blk.mlp = OvercompleteTwoBeamInterference(
                    embed_dim=eff_q_embed_dim,
                    frame_expansion=frame_expansion,
                    target_iso_scale=target_iso,
                    init_steps=120,
                    seed=42 + 10 * l_idx,
                    device=device,
                ).to(device)
        elif overcomplete_frame and not sparse_density_layer:
            target_iso = default_overcomplete_target_iso_scale(frame_expansion)
            for l_idx, blk in enumerate(model.blocks):
                blk.mlp = OvercompleteFrameInterference(
                    embed_dim=eff_q_embed_dim,
                    frame_expansion=frame_expansion,
                    target_iso_scale=target_iso,
                    seed=42 + 10 * l_idx,
                    device=device,
                ).to(device)
        if ppmi_svd_init:
            n_init_sh = max(1, min(int(ppmi_svd_shards), int(num_shards)))
            init_shards = (
                [stream.get_shard(s)[0] for s in range(1, n_init_sh + 1)]
                if stream is not None
                else [train_tokens]
            )
            if hierarchical_16qubit:
                c_path = (
                    data_dir / f"ppmi_svd_hier16q_{qubit_registers}x{qubit_amplitudes}_{num_higher_states}m_{n_init_sh}sh.pt"
                    if synthetic_tokens <= 0
                    else None
                )
                ppmi_stats = initialize_hierarchical_16qubit_with_ppmi_svd(
                    model,
                    token_shards=init_shards,
                    alpha_ppmi=ppmi_svd_alpha,
                    cache_path=c_path,
                    device=device,
                )
                print(
                    f"Initialized Hierarchical-16Qubit ({qubit_registers}x{qubit_amplitudes} + {num_higher_states} states) QuantumBornGPT with {n_init_sh}-Shard PPMI-SVD: {ppmi_stats}",
                    flush=True,
                )
            elif sparse_64qubit:
                c_path = (
                    data_dir / f"ppmi_svd_sparse64q_{num_qubits}q_{eff_q_embed_dim}d_{n_init_sh}sh.pt"
                    if synthetic_tokens <= 0
                    else None
                )
                ppmi_stats = initialize_sparse_64qubit_with_ppmi_svd(
                    model,
                    token_shards=init_shards,
                    alpha_ppmi=ppmi_svd_alpha,
                    cache_path=c_path,
                    device=device,
                )
                print(f"Initialized Sparse-64Qubit QuantumBornGPT with {n_init_sh}-Shard PPMI-SVD: {ppmi_stats}", flush=True)
            else:
                c_path = (
                    data_dir / f"ppmi_svd_qborn_{seed_dim}s_{eff_q_embed_dim}d_{n_init_sh}sh.pt"
                    if synthetic_tokens <= 0
                    else None
                )
                ppmi_stats = initialize_quantum_born_with_ppmi_svd(
                    model,
                    token_shards=init_shards,
                    alpha_ppmi=ppmi_svd_alpha,
                    cache_path=c_path,
                    device=device,
                )
                print(f"Initialized QuantumBornGPT with {n_init_sh}-Shard PPMI-SVD: {ppmi_stats}", flush=True)
        if fp8_linear:
            convert_linear_layers_to_fp8(model)

        n_params = sum(p.numel() for p in model.parameters())
        frame_dim = int(round(eff_q_embed_dim * frame_expansion))
        frame_tag = f"{frame_expansion:g}d"
        if sparse_density_layer:
            mlp_label = f"Density{eff_q_embed_dim}x{eff_q_embed_dim}(4.19M,R={density_matrix_rank})"
        elif overcomplete_two_beam:
            mlp_label = f"{frame_dim}x2({frame_tag}-TwoBeam)"
        elif overcomplete_frame:
            mlp_label = f"{frame_dim}({frame_tag}-Frame)"
        else:
            mlp_label = f"{hidden_dim}x2"
        adapter_label = (
            "2BeamAdapter"
            if two_beam_seed_adapter
            else ("1BeamAdapter" if overcomplete_seed_adapter else "False")
        )
        if hierarchical_16qubit:
            q_prefix = f"{qubit_registers}x{qubit_amplitudes}-Qubit+{num_higher_states}Hier"
        elif sparse_64qubit:
            q_prefix = f"{num_qubits}-Qubit SparseFock"
        else:
            q_prefix = "10-Qubit"
        if two_beam_kboson_readout:
            eff_readout_lbl = f"two_beam_kboson_povm(r={diffuse_rank})"
        elif last_layer_softmax and hierarchical_16qubit:
            eff_readout_lbl = f"calibrated_hierarchical_softmax(cap={logit_softcap:g})"
        elif last_layer_softmax:
            eff_readout_lbl = f"calibrated_softmax(cap={logit_softcap:g})"
        else:
            eff_readout_lbl = readout_mode
        print(
            f"Architecture: {q_prefix} QuantumBornGPT (d={eff_q_embed_dim}, H={num_heads}, L={num_layers}, d_ff={mlp_label}, d_seed={seed_dim}, Readout={eff_readout_lbl}, LastLayerSoftmax={last_layer_softmax}, TwoBeamKBoson={two_beam_kboson_readout}, DiffRank={diffuse_rank}, Softcap={logit_softcap:g}, Hier16Q={hierarchical_16qubit}, HigherStates={num_higher_states}, BlockBridge={not no_block_hierarchy_bridge}, Sparse64Q={sparse_64qubit}, SparseK={sparse_ngram_k}, DensityLayer={sparse_density_layer}, Rank={density_matrix_rank}, Coupler={not no_fock_coupler}, DynBeta={dynamic_povm_beta}, OrthoDec={ortho_decouple}, LogPrior={povm_log_prior}, GatedRes={gated_spherical_residual}, PurifiedAttn={purified_born_attn}, TrueV0={true_layer0_v_residual}, UNetSkip={unet_skip_mode}, FusedQKV={fused_qkv}, ValEmbeds={num_value_embeddings}, SkipAttn={skip_attn_tuple}, ZeroHalfRoPE={zero_half_rope}, Win={sliding_window}, DocMask={doc_mask_attn}, PPMISVD={ppmi_svd_init}, NgramHash={ngram_hash}, MUDD={mudd_and_gates}, SmearGate={smear_gate}, ResScale={model.residual_scale:.4f}, SeedAdapter={adapter_label}, MTP={mtp_horizons}, stride={mtp_stride}, KrausBeams={kraus_beams}, Opt={stiefel_optimizer}/{optimizer_mode}, DirectCE={direct_chunked_ce}, FusedTile={fused_povm_tile_size}) | Params: {n_params:,}",
            flush=True,
        )
    if muon_lr is None:
        base_muon_lr = 0.05 if leaderboard_exact_classical else 0.025
    else:
        base_muon_lr = float(muon_lr)
        if leaderboard_exact_classical and abs(base_muon_lr - 0.025) < 1e-9:
            base_muon_lr = 0.05
    base_adamw_lr = 0.004
    if optimizer_mode == "pure_adamw":
        base_muon_lr = 0.003
    elif optimizer_mode == "sgd_nesterov":
        base_muon_lr = 0.05
    print(
        f"Schedule ({schedule.upper()} | MuonLR={base_muon_lr:g}): {num_epochs} Epochs x {num_shards} Shards/Epoch ({steps_per_epoch:,} steps/epoch) = {total_steps:,} steps ({total_steps * tok_per_step:,} total tokens)",
        flush=True,
    )

    muon_params, adamw_params = split_muon_adamw_params(
        model,
        adamw_name_patterns=(
            "wte",
            "qubit_wte",
            "unigram_qubits",
            "higher_states",
            "povm_odd_qubits",
            "vocab_higher_affinity",
            "hierarchical_povm_readout",
            "hier_even_proj",
            "readout_rms_gain",
            "token_radial_scale",
            "qubit_readout_gain",
            "kraus",
            "kboson",
            "diffuse",
            "causal_mem",
            "block_even_odd_mix",
            "block_out_parity_mix",
            "block_hier_gates",
            "unigram_sparse_vals",
            "ngram_sparse_vals",
            "branch_spectral_filters",
            "purity_router",
            "povm_head",
            "lm_head",
            "purity_gate",
            "highway_gate",
            "v_mix_gate",
            "embed",
            "value_embeds",
            "ngram",
            "mudd",
            "input_gate",
            "output_gate",
            "xsa_alpha",
            "gate_proj",
            "smear_lambda",
            "unet_scales",
            "unet_lambdas",
            "skip_weights",
            "block_lambdas",
            "sa_lambdas",
            "smear",
            "inv_temp",
            "beam_",
            "seed_gate",
            "povm_log_prior",
            "povm_odd_delta",
            "sparse_povm_readout",
            "coherence_tau",
            "res_gate",
        ),
    )
    eff_polar_method: Literal["newton_schulz", "polar_express"] = (
        "polar_express" if polar_express else "newton_schulz"
    )
    if optimizer_mode == "pure_adamw":
        opt_muon: torch.optim.Optimizer = torch.optim.AdamW(
            muon_params,
            lr=base_muon_lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=True,
        )
    elif optimizer_mode == "sgd_nesterov":
        opt_muon = torch.optim.SGD(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.01,
        )
    elif architecture == "classical_modded" and leaderboard_exact_classical:
        opt_muon = StiefelMuon(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            momentum_warmup_steps=max(300, muon_momentum_warmup),
            momentum_init=0.85,
            nesterov=True,
            aspect_scale=True,
            ema_momentum=True,
            weight_decay=0.01 if cautious_wd else 0.0,
            decay_mode="cautious" if cautious_wd else "none",
            stiefel_penalty=0.0,
            ns_steps=6 if polar_express else 5,
            retract_every=1_000_000,
            batched_ns=True,
            polar_method=eff_polar_method,
            use_normuon=normuon,
            use_twin_rail=twin_rail_muon,
        )
    elif architecture == "classical_modded":
        opt_muon = StiefelMuon(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            momentum_warmup_steps=muon_momentum_warmup,
            nesterov=zero_init_classical,
            aspect_scale=zero_init_classical,
            weight_decay=0.01,
            decay_mode="cautious",
            stiefel_penalty=classical_stiefel_penalty,
            ns_steps=6 if polar_express else 5,
            retract_every=4 if classical_stiefel_penalty > 0 else 1_000_000,
            batched_ns=True,
            polar_method=eff_polar_method,
            use_normuon=normuon,
            use_twin_rail=twin_rail_muon,
        )
    elif stiefel_optimizer == "tangent":
        opt_muon = TangentStiefelMuon(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            ns_steps=5,
            retract_every=1,
        )
    else:
        opt_muon = StiefelMuon(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            momentum_warmup_steps=muon_momentum_warmup,
            nesterov=True,
            ema_momentum=bool(muon_lr is not None and muon_lr >= 0.03),
            weight_decay=0.01,
            decay_mode="cautious",
            stiefel_penalty=0.005,
            ns_steps=6 if polar_express else 5,
            retract_every=4,
            batched_ns=True,
            polar_method=eff_polar_method,
            use_normuon=normuon,
            use_twin_rail=twin_rail_muon,
        )

    if architecture == "classical_modded" and leaderboard_exact_classical:
        wte_group = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and ("wte" in n or "value_embeds" in n)
        ]
        ngram_group = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "ngram" in n
        ]
        head_group = [p for n, p in model.named_parameters() if p.requires_grad and "lm_head" in n]
        muon_id_set = {id(p) for p in muon_params}
        wte_head_ids = {id(p) for p in wte_group} | {id(p) for p in ngram_group} | {id(p) for p in head_group}
        scalar_group = [
            p
            for _, p in model.named_parameters()
            if p.requires_grad and id(p) not in muon_id_set and id(p) not in wte_head_ids
        ]
        adam_groups = [
            {"params": head_group, "lr": 0.008, "base_lr": 0.008, "weight_decay": 0.0},
            {"params": wte_group, "lr": 0.60, "base_lr": 0.60, "weight_decay": 0.0},
            {"params": ngram_group, "lr": 0.04, "base_lr": 0.04, "weight_decay": 0.0},
            {"params": scalar_group, "lr": 0.04, "base_lr": 0.04, "weight_decay": 0.0},
        ]
        opt_adamw: torch.optim.Optimizer = torch.optim.Adam(
            adam_groups, betas=(0.8, 0.95), eps=1e-10, fused=True
        )
    elif architecture == "classical_modded" and zero_init_classical:
        wte_group = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and ("wte" in n or "value_embeds" in n)
        ]
        ngram_group = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "ngram" in n
        ]
        head_group = [p for n, p in model.named_parameters() if p.requires_grad and "lm_head" in n]
        muon_id_set = {id(p) for p in muon_params}
        wte_head_ids = {id(p) for p in wte_group} | {id(p) for p in ngram_group} | {id(p) for p in head_group}
        scalar_group = [
            p
            for _, p in model.named_parameters()
            if p.requires_grad and id(p) not in muon_id_set and id(p) not in wte_head_ids
        ]
        adamw_groups = [
            {"params": wte_group, "lr": 0.20, "base_lr": 0.20, "weight_decay": 0.0},
            {"params": ngram_group, "lr": 0.04, "base_lr": 0.04, "weight_decay": 0.0},
            {"params": head_group, "lr": 0.008, "base_lr": 0.008, "weight_decay": 0.0},
            {"params": scalar_group, "lr": 0.02, "base_lr": 0.02, "weight_decay": 0.0},
        ]
        opt_adamw = torch.optim.AdamW(adamw_groups, betas=(0.9, 0.95), fused=True)
    elif architecture == "classical_modded":
        opt_adamw = torch.optim.AdamW(
            [{"params": adamw_params, "lr": base_adamw_lr, "base_lr": base_adamw_lr, "weight_decay": 0.01}],
            betas=(0.9, 0.95),
            fused=True,
        )
    else:
        readout_softmax_ids: set[int] = set()
        if last_layer_softmax:
            readout_softmax_ids = {
                id(p)
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(
                    sub in n
                    for sub in (
                        "povm_head",
                        "povm_odd_qubits",
                        "vocab_higher_affinity",
                        "token_radial_scale",
                        "readout_rms_gain",
                        "qubit_readout_gain",
                        "hier_even_proj",
                        "povm_log_prior",
                    )
                )
            }
        sparse_embed_ids = {
            id(p)
            for n, p in model.named_parameters()
            if p.requires_grad
            and id(p) not in readout_softmax_ids
            and any(
                sub in n
                for sub in (
                    "qubit_wte",
                    "unigram_sparse_vals",
                    "ngram_sparse_vals",
                    "ngram_embed",
                    "povm_odd_delta",
                    "unigram_qubits",
                    "higher_states",
                    "povm_odd_qubits",
                    "vocab_higher_affinity",
                )
            )
        }
        readout_softmax_params = [p for p in adamw_params if id(p) in readout_softmax_ids]
        sparse_params = [p for p in adamw_params if p.ndim >= 2 and id(p) in sparse_embed_ids]
        excluded_ids = sparse_embed_ids | readout_softmax_ids
        adamw_2d = [p for p in adamw_params if p.ndim == 2 and id(p) not in excluded_ids]
        adamw_non2d = [p for p in adamw_params if p.ndim != 2 and id(p) not in excluded_ids]
        sparse_lr = base_adamw_lr * 2.5 if (sparse_64qubit or hierarchical_16qubit) else base_adamw_lr
        q_adamw_groups = [
            {"params": adamw_2d, "lr": base_adamw_lr, "base_lr": base_adamw_lr, "weight_decay": 0.01},
            {"params": adamw_non2d, "lr": base_adamw_lr, "base_lr": base_adamw_lr, "weight_decay": 0.0},
        ]
        if len(sparse_params) > 0:
            q_adamw_groups.append(
                {"params": sparse_params, "lr": sparse_lr, "base_lr": sparse_lr, "weight_decay": 0.0}
            )
        if len(readout_softmax_params) > 0:
            q_adamw_groups.append(
                {"params": readout_softmax_params, "lr": 0.008, "base_lr": 0.008, "weight_decay": 0.0}
            )
        opt_adamw = torch.optim.AdamW(
            q_adamw_groups,
            betas=(0.9, 0.95),
            fused=True,
        )

    averager: TailWeightAverager | None = (
        TailWeightAverager(model, decay=0.995, start_fraction=0.75) if tail_ema else None
    )

    B_init, T_init = curriculum.for_epoch(0, num_epochs) if curriculum is not None else (B, T)
    print("Compiling model & StiefelMuon kernels via TorchInductor + Triton...", flush=True)
    t_comp0 = time.perf_counter()
    x_warm = torch.randint(0, 50257, (B_init, T_init), device=device, dtype=torch.long)
    y_warm = torch.randint(0, 50257, (B_init, T_init), device=device, dtype=torch.long)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_w = compute_step_loss(model, x_warm, y_warm, cur_step=0, use_sampled=False)
    loss_w.backward()
    opt_muon.zero_grad(set_to_none=True)
    opt_adamw.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"Compilation complete in {time.perf_counter() - t_comp0:.2f}s", flush=True)

    torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, float]] = []

    eval_b_sz = max(1, min(64, B_init, 32768 // max(1, T_init)))
    val0 = evaluate_validation(model, val_tokens, eval_batch_size=eval_b_sz, block_size=T_init)
    print(
        f"[Epoch 0.00 | Step    0 |   0.00M tok] Val CE: {val0['val_ce']:.4f} | Val PPL: {val0['val_ppl']:8.1f} | "
        f"Top-1: {val0['top1_acc']:5.2f}% | Top-5: {val0['top5_acc']:5.2f}% | MedRank: {val0['med_rank']:5.0f} | "
        f"IsoErr: {val0['iso_err']:.4f} | S(rho_A): {val0['entanglement_5v5_mean']:.4f} nats | "
        f"gamma_h: {val0['gamma_head_mean']:.2f} | beta: {val0['povm_beta']:.2f}",
        flush=True,
    )
    history.append({"epoch": 0.0, "step": 0, "tokens_M": 0.0, "train_ce": float(val0["val_ce"]), "tok_per_s": 0.0, **val0})  # type: ignore[arg-type]

    warmup_steps = 0 if leaderboard_exact_classical else min(150, max(1, total_steps // 5))
    running_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    running_steps = 0
    train_t0 = time.perf_counter()
    window_t0 = time.perf_counter()
    window_tokens = 0
    save_path = out_json if out_json is not None else Path("/content/gpu_100M_results.json")

    if stage_shard_on_gpu is None:
        use_gpu_shard = bool(
            torch.cuda.is_available()
            and (torch.cuda.get_device_properties(device).total_memory >= 20 * (1024**3))
        )
    else:
        use_gpu_shard = bool(stage_shard_on_gpu)
    single_shard_gpu: torch.Tensor | None = None
    if use_gpu_shard and stream is None:
        single_shard_gpu = torch.from_numpy(train_tokens).to(device=device, dtype=torch.uint16)

    global_step = 0
    eff_embed_step_every = max(1, int(embed_step_every))
    for ep in range(num_epochs):
        B_ep, T_ep = curriculum.for_epoch(ep, num_epochs) if curriculum is not None else (B, T)
        if stream is not None and ep > 0:
            stream.prefetch_shard(1)

        for shard_idx in range(1, num_shards + 1):
            active_shard_gpu: torch.Tensor | None = None
            if stream is not None:
                if ep == 0 and shard_idx == 1:
                    active_shard = train_tokens
                    shard_wait_ms = first_wait_ms
                else:
                    active_shard, shard_wait_ms = stream.get_shard(shard_idx)
                if use_gpu_shard:
                    active_shard_gpu = torch.from_numpy(active_shard).to(device=device, dtype=torch.uint16)
                print(
                    f"  [Shard {shard_idx}/{num_shards} | Epoch {ep + 1}/{num_epochs}] "
                    f"Active ({len(active_shard):,} tokens | prefetch wait: {shard_wait_ms:.1f} ms | GPUStaged={use_gpu_shard})",
                    flush=True,
                )
                rng = np.random.default_rng(42 + ep * 1000 + shard_idx)
            else:
                active_shard = train_tokens
                active_shard_gpu = single_shard_gpu
                rng = np.random.default_rng(42 + ep)

            batch_order = (
                np.arange(steps_per_shard, dtype=np.int64)
                if leaderboard_exact_classical
                else rng.permutation(steps_per_shard)
            )

            for b_pos in range(steps_per_shard):
                global_step += 1
                if progressive_window and isinstance(model, ClassicalModdedGPT):
                    prog_w = min(
                        1792,
                        ((int(1728 * (global_step - 1) / max(1, total_steps)) // 128) + 1) * 128,
                    )
                    model.set_window_size(prog_w)

                if tapered_mtp and isinstance(model, QuantumBornGPT):
                    model.mtp_loss_weight = compute_tapered_mtp_weight(
                        global_step, total_steps, mtp_weight
                    )

                if schedule == "wsd":
                    lr_mult = wsd_lr_multiplier(
                        global_step,
                        total_steps,
                        warmup_steps=warmup_steps,
                        decay_fraction=wsd_decay_fraction,
                        min_lr_ratio=wsd_min_lr_ratio,
                    )
                else:
                    lr_mult = cosine_lr_multiplier(
                        global_step,
                        total_steps,
                        warmup_steps=warmup_steps,
                        min_lr_ratio=0.15,
                    )

                for g in opt_muon.param_groups:
                    g["lr"] = base_muon_lr * lr_mult
                for g in opt_adamw.param_groups:
                    g["lr"] = float(g.get("base_lr", base_adamw_lr)) * lr_mult

                chunk_idx = int(batch_order[b_pos])
                start_tok = chunk_idx * tok_per_step
                opt_muon.zero_grad(set_to_none=True)
                if (global_step - 1) % eff_embed_step_every == 0:
                    opt_adamw.zero_grad(set_to_none=True)

                use_sampled_now = bool(
                    sampled_softmax_candidates > 0 and global_step <= int(0.75 * total_steps)
                )
                max_valid_start = max(0, len(active_shard) - tok_per_step - 2)
                if align_to_bos and start_tok > 0:
                    look_end = min(len(active_shard) - tok_per_step - 2, start_tok + 64)
                    if look_end > start_tok:
                        if active_shard_gpu is not None:
                            w_gpu = active_shard_gpu[start_tok:look_end]
                            eot_m = (w_gpu == 50256).nonzero(as_tuple=False)
                            if eot_m.numel() > 0:
                                start_tok = min(max_valid_start, start_tok + int(eot_m[0].item()) + 1)
                        else:
                            w_np = active_shard[start_tok:look_end]
                            eot_idx = np.flatnonzero(w_np == 50256)
                            if eot_idx.size > 0:
                                start_tok = min(max_valid_start, start_tok + int(eot_idx[0]) + 1)
                start_tok = min(start_tok, max_valid_start)

                if active_shard_gpu is not None:
                    buf_gpu = active_shard_gpu[start_tok : start_tok + tok_per_step + 1].to(torch.int64)
                    if G_acc == 1:
                        x = buf_gpu[:-1].reshape(B_ep, T_ep)
                        y = buf_gpu[1:].reshape(B_ep, T_ep)
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = compute_step_loss(
                                model, x, y, cur_step=global_step, use_sampled=use_sampled_now
                            )
                        loss.backward()
                        step_loss_det = loss.detach().float()
                    else:
                        x_all_gpu = buf_gpu[:-1].reshape(G_acc, B_ep, T_ep)
                        y_all_gpu = buf_gpu[1:].reshape(G_acc, B_ep, T_ep)
                        step_loss_det = torch.zeros((), device=device, dtype=torch.float32)
                        for acc_i in range(G_acc):
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                loss_m = (
                                    compute_step_loss(
                                        model,
                                        x_all_gpu[acc_i],
                                        y_all_gpu[acc_i],
                                        cur_step=global_step,
                                        use_sampled=use_sampled_now,
                                    )
                                    / float(G_acc)
                                )
                            loss_m.backward()
                            step_loss_det = step_loss_det + loss_m.detach().float()
                else:
                    buf = active_shard[start_tok : start_tok + tok_per_step + 1].astype(np.int64)
                    if G_acc == 1:
                        x = torch.from_numpy(buf[:-1].reshape(B_ep, T_ep)).to(device, non_blocking=True)
                        y = torch.from_numpy(buf[1:].reshape(B_ep, T_ep)).to(device, non_blocking=True)
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = compute_step_loss(
                                model, x, y, cur_step=global_step, use_sampled=use_sampled_now
                            )
                        loss.backward()
                        step_loss_det = loss.detach().float()
                    else:
                        x_all = buf[:-1].reshape(G_acc, B_ep, T_ep)
                        y_all = buf[1:].reshape(G_acc, B_ep, T_ep)
                        step_loss_det = torch.zeros((), device=device, dtype=torch.float32)
                        for acc_i in range(G_acc):
                            x_m = torch.from_numpy(x_all[acc_i]).to(device, non_blocking=True)
                            y_m = torch.from_numpy(y_all[acc_i]).to(device, non_blocking=True)
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                loss_m = (
                                    compute_step_loss(
                                        model, x_m, y_m, cur_step=global_step, use_sampled=use_sampled_now
                                    )
                                    / float(G_acc)
                                )
                            loss_m.backward()
                            step_loss_det = step_loss_det + loss_m.detach().float()
                if not leaderboard_exact_classical:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_muon.step()
                if (global_step % eff_embed_step_every == 0) or (global_step == total_steps):
                    opt_adamw.step()
                if averager is not None:
                    averager.step(model, current_step=global_step, total_steps=total_steps)

                running_loss_t += step_loss_det
                running_steps += 1
                window_tokens += tok_per_step

                shard_step = b_pos + 1
                if num_shards == 1:
                    do_eval = (global_step == eval_every + (ep * steps_per_epoch)) or (
                        global_step == (ep + 1) * steps_per_epoch
                    )
                else:
                    do_eval = (
                        shard_step % eval_every == 0
                        and (steps_per_shard - shard_step) >= max(1, eval_every // 2)
                    ) or (shard_step == steps_per_shard)

                if global_step % 150 == 0 and (not do_eval) and (global_step != total_steps):
                    torch.cuda.synchronize()
                    dt_w = time.perf_counter() - window_t0
                    tps = window_tokens / max(dt_w, 1e-6)
                    ep_frac = global_step / float(steps_per_epoch)
                    tok_m = (global_step * tok_per_step) / 1e6
                    print(
                        f"  -> Step {global_step:4d}/{total_steps} (Epoch {ep_frac:.2f} | {tok_m:6.2f}M tok) | "
                        f"Train CE: {float(running_loss_t.item()) / running_steps:.4f} | Speed: {tps:,.0f} tok/s | "
                        f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                        flush=True,
                    )

                if do_eval:
                    torch.cuda.synchronize()
                    dt_w = time.perf_counter() - window_t0
                    tps = window_tokens / max(dt_w, 1e-6)
                    train_ce = float(running_loss_t.item()) / max(1, running_steps)
                    running_loss_t.zero_()
                    running_steps = 0

                    if progressive_window and isinstance(model, ClassicalModdedGPT) and global_step == total_steps:
                        model.set_window_size(1792)
                    eval_b_ep = max(1, min(64, B_ep, 32768 // max(1, T_ep)))
                    if averager is not None and averager.active:
                        with averager.average_parameters(model):
                            val_metrics = evaluate_validation(
                                model, val_tokens, eval_batch_size=eval_b_ep, block_size=T_ep
                            )
                    else:
                        val_metrics = evaluate_validation(
                            model, val_tokens, eval_batch_size=eval_b_ep, block_size=T_ep
                        )
                    if num_shards == 1:
                        ep_frac = round((global_step / float(steps_per_epoch)) * 2.0) / 2.0
                    else:
                        ep_frac = round(global_step / float(steps_per_epoch), 4)
                    tok_m = (global_step * tok_per_step) / 1e6
                    peak_gb = torch.cuda.max_memory_allocated() / 1e9
                    print(
                        f"[Epoch {ep_frac:4.2f} | Step {global_step:4d} | {tok_m:6.2f}M tok | T={T_ep}] "
                        f"Train CE: {train_ce:.4f} | Val CE: {val_metrics['val_ce']:.4f} | Val PPL: {val_metrics['val_ppl']:6.1f} | "
                        f"Top-1: {val_metrics['top1_acc']:5.2f}% | Top-5: {val_metrics['top5_acc']:5.2f}% | "
                        f"MedRank: {val_metrics['med_rank']:4.0f} | IsoErr: {val_metrics['iso_err']:.4f} | "
                        f"S(rho_A): {val_metrics['entanglement_5v5_mean']:.4f} | gamma_h: {val_metrics['gamma_head_mean']:.2f} | "
                        f"beta: {val_metrics['povm_beta']:.2f} | {tps:,.0f} tok/s | VRAM: {peak_gb:.2f} GB",
                        flush=True,
                    )
                    history.append(
                        {
                            "epoch": ep_frac,
                            "shard_idx": shard_idx,
                            "step": global_step,
                            "tokens_M": tok_m,
                            "seq_len": T_ep,
                            "lr_mult": round(lr_mult, 4),
                            "train_ce": train_ce,
                            "tok_per_s": tps,
                            "peak_vram_gb": peak_gb,
                            **val_metrics,  # type: ignore[arg-type]
                        }
                    )
                    try:
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_text(json.dumps(history, indent=2))
                    except Exception:
                        pass
                    window_t0 = time.perf_counter()
                    window_tokens = 0

    if stream is not None:
        stream.close()
    total_time = time.perf_counter() - train_t0
    print("=" * 90, flush=True)
    print(f"COMPLETED {num_epochs} FULL EPOCHS ON GPU IN {total_time:.1f}s ({total_time/60:.2f} min)", flush=True)
    print("=" * 90, flush=True)
    return history


def run_tpu_benchmark(
    num_epochs: int = 3,
    num_shards: int = 1,
    eval_every_tokens_m: float = 50.0,
    data_dir: Path = Path("/content/data/fineweb10B"),
    schedule: Literal["cosine", "wsd"] = "cosine",
    wsd_decay_fraction: float = 0.20,
    wsd_min_lr_ratio: float = 0.10,
    mtp_horizons: int = 1,
    mtp_weight: float = 0.25,
    seed_dim: int = 128,
    mtp_stride: int = 4,
    mtp_detach_readout: bool = True,
    mtp_decoupled_temp: bool = True,
    stiefel_optimizer: Literal["standard", "tangent"] = "standard",
    fused_povm_tile_size: int = 0,
    num_layers: int = 6,
    embed_dim: int = 1024,
    num_heads: int = 16,
    hidden_dim: int = 1536,
    frame_expansion: float = 2.0,
    readout_mode: Literal["tied_stiefel", "tied_vocab", "symmetric_dual", "asymmetric"] = "tied_stiefel",
    depth_scaled_residual: bool = False,
    architecture: Literal["quantum_born", "classical_modded"] = "quantum_born",
    dynamic_povm_beta: bool = False,
    ortho_decouple: bool = False,
    povm_log_prior: bool = False,
    gated_spherical_residual: bool = False,
    purified_born_attn: bool = False,
    zero_init_classical: bool = False,
    true_layer0_v_residual: bool = False,
    classical_stiefel_penalty: float = 0.0,
    muon_lr: float = 0.025,
    tpu_attn_kernel: Literal["remat", "pallas", "scan"] = "remat",
    val_tokens_budget: int = 10485760,
    synthetic_tokens: int = 0,
    out_json: Path | None = None,
) -> list[dict[str, float]]:
    if architecture == "classical_modded":
        raise ValueError("ClassicalModdedGPT is only supported on the CUDA benchmark path")
    _ = (
        dynamic_povm_beta,
        ortho_decouple,
        povm_log_prior,
        gated_spherical_residual,
        purified_born_attn,
        zero_init_classical,
        true_layer0_v_residual,
        classical_stiefel_penalty,
    )
    import jax
    import jax.numpy as jnp

    from qdlblocks.benchmarks.fineweb_stream import FineWebMultiShardStream
    from qdlblocks.optim.schedules import cosine_lr_multiplier, wsd_lr_multiplier
    from qdlblocks_jax.functional import (
        bipartite_entanglement_entropy,
        effective_quantum_rank,
        purity_participation_ratio,
    )
    from qdlblocks_jax.fused_povm_ce import fused_tiled_povm_cross_entropy_jax
    from qdlblocks_jax.pallas_flash_born import flash_born_attention_jax
    from qdlblocks_jax.parameter_scaling import compute_depth_residual_scale_jax
    from qdlblocks_jax.tangent_stiefel import (
        project_stiefel_tangent_jax,
        stiefel_quadratic_retract_jax,
    )

    def l2_norm_bf16(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
        x_bf16 = x.astype(jnp.bfloat16)
        sq = jnp.sum(x_bf16 * x_bf16, axis=axis, keepdims=True, dtype=jnp.float32)
        return x_bf16 * jax.lax.rsqrt(jnp.maximum(sq, eps * eps)).astype(jnp.bfloat16)

    def qr_ortho(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        raw = jax.random.normal(key, shape, dtype=jnp.float32)
        m, n = shape[0], shape[1]
        if m >= n:
            q, _ = jnp.linalg.qr(raw)
            return q
        q_t, _ = jnp.linalg.qr(raw.T)
        return q_t.T

    def precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[jax.Array, jax.Array]:
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (jnp.arange(0, half, dtype=jnp.float32) / half))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        return (
            jnp.cos(freqs)[None, None, :, :].astype(jnp.bfloat16),
            jnp.sin(freqs)[None, None, :, :].astype(jnp.bfloat16),
        )

    def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def newton_schulz5_bf16(G: jax.Array, steps: int = 5) -> jax.Array:
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.astype(jnp.bfloat16)
        transposed = X.shape[0] > X.shape[1]
        if transposed:
            X = X.T
        norm = jnp.sqrt(jnp.sum(X * X, dtype=jnp.float32)) + 1e-7
        X = (X.astype(jnp.float32) / norm).astype(jnp.bfloat16)
        for _ in range(steps):
            A = jnp.matmul(X, X.T, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            B_mat = b * A + c * jnp.matmul(A, A, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            X = a * X + jnp.matmul(B_mat, X, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        if transposed:
            X = X.T
        return X.astype(jnp.float32)

    def stiefel_retract_grad(W: jax.Array) -> jax.Array:
        W_bf16 = W.astype(jnp.bfloat16)
        r, c = W.shape
        if r >= c:
            gram = jnp.matmul(W_bf16.T, W_bf16, preferred_element_type=jnp.float32)
            diff = gram - jnp.eye(c, dtype=jnp.float32)
            denom = 1.0 + (jnp.sqrt(jnp.sum(diff * diff)) / math.sqrt(c))
            grad = jnp.matmul(W_bf16, (diff / denom).astype(jnp.bfloat16), preferred_element_type=jnp.float32)
        else:
            gram = jnp.matmul(W_bf16, W_bf16.T, preferred_element_type=jnp.float32)
            diff = gram - jnp.eye(r, dtype=jnp.float32)
            denom = 1.0 + (jnp.sqrt(jnp.sum(diff * diff)) / math.sqrt(r))
            grad = jnp.matmul((diff / denom).astype(jnp.bfloat16), W_bf16, preferred_element_type=jnp.float32)
        return grad

    print("=" * 90, flush=True)
    print("FULL 100M-TOKEN MULTI-EPOCH QUANTUM SLOWRUN BENCHMARK — COLAB TPU (CLOUD TPU v5e-1)", flush=True)
    print(f"JAX {jax.__version__} | Backend: {jax.default_backend()} | Devices: {jax.devices()}", flush=True)
    print("=" * 90, flush=True)

    V = 50304
    D_seed = int(seed_dim)
    D = int(embed_dim)
    H = int(num_heads)
    D_h = D // H
    L = int(num_layers)
    D_mlp = int(hidden_dim)
    out_dim = D if readout_mode == "asymmetric" else D_seed
    res_scale = compute_depth_residual_scale_jax(L, base_layers=6, enabled=depth_scaled_residual)
    res_scale_bf16 = jnp.bfloat16(res_scale)
    T = 512
    B_micro = 64
    grad_accum = 2
    B = B_micro * grad_accum
    tok_per_step = B * T

    stream: FineWebMultiShardStream | None = None
    if num_shards > 1:
        stream = FineWebMultiShardStream(
            data_dir=data_dir,
            num_shards=num_shards,
            prefetch_async=True,
            synthetic_tokens_per_shard=synthetic_tokens,
        )
        val_tokens = stream.ensure_val_shard()
        first_shard, first_wait_ms = stream.get_shard(1)
        train_tokens = first_shard
        print(
            f"Streaming {num_shards} Train Shards ({len(first_shard):,} tokens/shard, first wait {first_wait_ms:.1f}ms) | Val Shard: {len(val_tokens):,} tokens",
            flush=True,
        )
        steps_per_shard = max(1, (len(first_shard) - max(4, mtp_horizons + 1)) // tok_per_step)
        steps_per_epoch = steps_per_shard * num_shards
        total_steps = steps_per_epoch * num_epochs
        eval_every = max(1, int(round((eval_every_tokens_m * 1e6) / tok_per_step)))
    else:
        train_tokens, val_tokens = _load_or_generate_shards(data_dir, synthetic_tokens=synthetic_tokens)
        print(f"Loaded Train Shard: {len(train_tokens):,} tokens | Val Shard: {len(val_tokens):,} tokens", flush=True)
        steps_per_shard = max(1, (len(train_tokens) - 1) // tok_per_step)
        steps_per_epoch = steps_per_shard
        total_steps = steps_per_epoch * num_epochs
        eval_every = max(1, steps_per_epoch // 2)

    key = jax.random.PRNGKey(42)
    k_wte, k_lift, k_blocks, k_mtp = jax.random.split(key, 4)

    raw_wte = jax.random.normal(k_wte, (V, D_seed), dtype=jnp.float32) * (1.0 / math.sqrt(D_seed))
    layer_keys = jax.random.split(k_blocks, L)

    def init_block(k: jax.Array) -> dict[str, jax.Array]:
        kq, kk, kv, ko, ka, kb, kd = jax.random.split(k, 7)
        return {
            "w_q": qr_ortho(kq, (D, D)),
            "w_k": qr_ortho(kk, (D, D)),
            "w_v": qr_ortho(kv, (D, D)),
            "w_o": qr_ortho(ko, (D, D)),
            "w_a": qr_ortho(ka, (D, D_mlp)),
            "w_b": qr_ortho(kb, (D, D_mlp)),
            "w_down": qr_ortho(kd, (D_mlp, D)),
            "head_scale": jnp.full((1, H, 1, 1), math.sqrt(D_h), dtype=jnp.float32),
            "v_mix_logit": jnp.array(-1.1, dtype=jnp.float32),
            "q_prev_mix": jnp.zeros((1, H, 1, 1), dtype=jnp.float32),
            "k_prev_mix": jnp.zeros((1, H, 1, 1), dtype=jnp.float32),
            "highway_logit": jnp.array(-2.0, dtype=jnp.float32),
        }

    params: dict[str, jax.Array | dict[str, jax.Array]] = {
        "wte": raw_wte,
        "w_lift": qr_ortho(k_lift, (D_seed, D)),
        "smear_gate": jnp.zeros((1, 1, D), dtype=jnp.float32),
        "tied_inv_temp": jnp.array(1.0 / 0.035, dtype=jnp.float32),
        "unet_gates": jnp.stack([jnp.eye(D, dtype=jnp.float32) for _ in range(L // 2)]),
        "unet_scales": jnp.full((L // 2,), 0.18, dtype=jnp.float32),
        "blocks": jax.vmap(init_block)(layer_keys),
    }
    if readout_mode == "asymmetric":
        params["povm_head"] = jax.random.normal(k_mtp, (V, D), dtype=jnp.float32) * (1.0 / math.sqrt(D))
    elif readout_mode == "tied_stiefel":
        params["readout_rot"] = jnp.eye(D, dtype=jnp.float32)
        params["povm_delta"] = jnp.zeros((V, D_seed), dtype=jnp.float32)
    elif readout_mode == "tied_vocab":
        params["readout_rot"] = jnp.eye(D, dtype=jnp.float32)
    if mtp_horizons > 1:
        mtp_keys = jax.random.split(k_mtp, mtp_horizons - 1)
        params["mtp_rotations"] = jax.vmap(lambda km: qr_ortho(km, (out_dim, out_dim)))(mtp_keys)
        params["mtp_inv_temps"] = jnp.array(
            [1.0 / (0.035 * math.sqrt(float(h))) for h in range(2, mtp_horizons + 1)],
            dtype=jnp.float32,
        )

    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
    print(
        f"Architecture: 10-Qubit QuantumBornGPT (d={D}, H={H}, L={L}, d_ff={D_mlp}x2, d_seed={D_seed}, Readout={readout_mode}, ResScale={res_scale:.4f}, MTP={mtp_horizons}, stride={mtp_stride}, Kernel={tpu_attn_kernel}, Opt={stiefel_optimizer}, FusedTile={fused_povm_tile_size}) | Params: {n_params:,}",
        flush=True,
    )
    print(
        f"Schedule ({schedule.upper()}): {num_epochs} Full Epochs x {steps_per_epoch:,} steps/epoch = {total_steps:,} steps ({total_steps * tok_per_step:,} total tokens)",
        flush=True,
    )

    cos_rope, sin_rope = precompute_rope(T, D_h)
    causal_mask_bf16 = jnp.tril(jnp.ones((T, T), dtype=jnp.bfloat16))[None, None, :, :]

    def encode_backbone(
        p: dict,
        x_idx: jax.Array,
        rng_key: jax.Array | None,
        is_training: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        B_sz, T_sz = x_idx.shape
        psi_seed = l2_norm_bf16(p["wte"][x_idx])
        psi_0 = l2_norm_bf16(jnp.matmul(psi_seed, p["w_lift"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
        if is_training and rng_key is not None:
            noise = jax.random.normal(rng_key, psi_0.shape, dtype=jnp.bfloat16) * jnp.bfloat16(0.015 / math.sqrt(D))
            psi_0 = l2_norm_bf16(psi_0 + noise)
        psi_prev = jnp.pad(psi_0[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
        eta = (0.06 * jnp.tanh(p["smear_gate"])).astype(jnp.bfloat16)
        psi_0 = l2_norm_bf16(psi_0 + eta * psi_prev)

        @jax.checkpoint
        def apply_one_block(psi_in: jax.Array, blk: dict) -> jax.Array:
            gamma = jax.nn.sigmoid(blk["highway_logit"]).astype(jnp.bfloat16)
            x_h = l2_norm_bf16((1.0 - gamma) * psi_in + gamma * psi_0)

            q = jnp.matmul(x_h, blk["w_q"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16).reshape(B_sz, T_sz, H, D_h).swapaxes(1, 2)
            k = jnp.matmul(x_h, blk["w_k"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16).reshape(B_sz, T_sz, H, D_h).swapaxes(1, 2)
            v = jnp.matmul(x_h, blk["w_v"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16).reshape(B_sz, T_sz, H, D_h).swapaxes(1, 2)

            v_0 = jnp.matmul(psi_0, blk["w_v"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16).reshape(B_sz, T_sz, H, D_h).swapaxes(1, 2)
            alpha_v = jax.nn.sigmoid(blk["v_mix_logit"]).astype(jnp.bfloat16)
            v = (1.0 - alpha_v) * v + alpha_v * v_0

            q_prev = jnp.pad(q[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0)))
            k_prev = jnp.pad(k[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0)))
            qm = (0.06 * jnp.tanh(blk["q_prev_mix"])).astype(jnp.bfloat16)
            km = (0.06 * jnp.tanh(blk["k_prev_mix"])).astype(jnp.bfloat16)
            q = q + qm * q_prev
            k = k + km * k_prev

            q = apply_rope(q, cos_rope, sin_rope)
            k = apply_rope(k, cos_rope, sin_rope)
            q, k = l2_norm_bf16(q), l2_norm_bf16(k)

            scale = jnp.clip(blk["head_scale"], 1.0, 40.0).astype(jnp.bfloat16)
            if tpu_attn_kernel in ("pallas", "scan"):
                attn_heads = flash_born_attention_jax(
                    q,
                    k,
                    v,
                    scale=scale,
                    is_causal=True,
                    eps=1e-6,
                    block_size=64,
                    backend=tpu_attn_kernel,
                )
                attn_out = attn_heads.astype(jnp.bfloat16).swapaxes(1, 2).reshape(B_sz, T_sz, D)
            else:
                s = jnp.einsum("bhid,bhjd->bhij", q * scale, k, preferred_element_type=jnp.bfloat16)
                a_pos = jnp.maximum(s, jnp.bfloat16(0.0)) * causal_mask_bf16
                prob = a_pos * a_pos
                z_den = jnp.sum(prob, axis=-1, keepdims=True, dtype=jnp.float32) + 1e-6
                o_num = jnp.einsum("bhij,bhjd->bhid", prob, v, preferred_element_type=jnp.float32)
                attn_out = (o_num / z_den).astype(jnp.bfloat16).swapaxes(1, 2).reshape(B_sz, T_sz, D)

            attn_proj = jnp.matmul(attn_out, blk["w_o"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16)
            x_mid = l2_norm_bf16(x_h + res_scale_bf16 * attn_proj)

            a_beam = jnp.matmul(x_mid, blk["w_a"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16)
            b_beam = l2_norm_bf16(jnp.matmul(x_mid, blk["w_b"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            interfered = a_beam * b_beam * jnp.bfloat16(math.sqrt(D_mlp))
            mlp_out = l2_norm_bf16(jnp.matmul(interfered, blk["w_down"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            return l2_norm_bf16(x_mid + res_scale_bf16 * mlp_out)

        enc_blks = jax.tree_util.tree_map(lambda x: x[: L // 2], p["blocks"])
        dec_blks = jax.tree_util.tree_map(lambda x: x[L // 2 :], p["blocks"])

        def scan_enc(psi_c, blk):
            psi_next = apply_one_block(psi_c, blk)
            return psi_next, psi_next

        psi, enc_skips = jax.lax.scan(scan_enc, psi_0, enc_blks)
        rev_skips = jnp.flip(enc_skips, axis=0)
        rev_gates = jnp.flip(p["unet_gates"], axis=0)
        rev_scales = jnp.flip(p["unet_scales"], axis=0)

        def scan_dec(psi_c, dec_in):
            blk, skip_psi, u_gate, u_scale = dec_in
            psi_next = apply_one_block(psi_c, blk)
            us = (0.35 * jnp.tanh(u_scale)).astype(jnp.bfloat16)
            skip_rot = l2_norm_bf16(jnp.matmul(skip_psi, u_gate.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            psi_next = l2_norm_bf16(psi_next + us * skip_rot)
            return psi_next, psi_next

        psi, dec_outs = jax.lax.scan(scan_dec, psi, (dec_blks, rev_skips, rev_gates, rev_scales))
        all_layer_states = jnp.concatenate([enc_skips, dec_outs], axis=0)
        if readout_mode == "asymmetric":
            phi_seed = l2_norm_bf16(psi)
            M_seed = l2_norm_bf16(p["povm_head"])
        elif readout_mode in ("tied_stiefel", "tied_vocab"):
            psi_final = l2_norm_bf16(jnp.matmul(psi, p["readout_rot"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            phi_seed = l2_norm_bf16(jnp.matmul(psi_final, p["w_lift"].T.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            M_seed = l2_norm_bf16(p["wte"] + p["povm_delta"]) if "povm_delta" in p else l2_norm_bf16(p["wte"])
        else:
            psi_final = l2_norm_bf16(psi)
            phi_seed = l2_norm_bf16(jnp.matmul(psi_final, p["w_lift"].T.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            M_seed = l2_norm_bf16(p["wte"])
        return phi_seed, M_seed, p["tied_inv_temp"], all_layer_states

    def _factorized_mxu_ce(
        phi_2d: jax.Array,
        M_seed: jax.Array,
        inv_t: jax.Array,
        flat_y: jax.Array,
    ) -> jax.Array:
        if fused_povm_tile_size > 0 and (phi_2d.shape[0] % fused_povm_tile_size == 0):
            return fused_tiled_povm_cross_entropy_jax(
                phi_2d, M_seed, jnp.clip(inv_t.astype(jnp.float32), 1.0, 60.0), flat_y, tile_size=fused_povm_tile_size
            )
        N = phi_2d.shape[0]
        num_chunks = 4
        chunk_sz = max(1, N // num_chunks)
        phi_c = phi_2d[: num_chunks * chunk_sz].reshape(num_chunks, chunk_sz, out_dim)
        beta_f = jnp.clip(inv_t.astype(jnp.float32), 1.0, 60.0)
        beta_bf16 = beta_f.astype(jnp.bfloat16)

        y_used = flat_y[: num_chunks * chunk_sz]
        target_M = M_seed[y_used]
        target_logits = (
            jnp.sum(phi_2d[: num_chunks * chunk_sz].astype(jnp.float32) * target_M.astype(jnp.float32), axis=-1)
            * beta_f
        )
        mean_target = jnp.mean(target_logits)

        @jax.checkpoint
        def lse_chunk(carry, zc):
            logits = jnp.matmul(zc, M_seed.T, preferred_element_type=jnp.bfloat16) * beta_bf16
            max_l = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
            shifted = (logits - max_l).astype(jnp.float32)
            lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1)) + jnp.squeeze(max_l.astype(jnp.float32), axis=-1)
            return carry + jnp.sum(lse), None

        tot_lse, _ = jax.lax.scan(lse_chunk, jnp.zeros((), dtype=jnp.float32), phi_c)
        return (tot_lse / float(num_chunks * chunk_sz)) - mean_target

    def microbatch_loss(p: dict, x_mb: jax.Array, y_mb: jax.Array, rng_key: jax.Array) -> jax.Array:
        phi_seed_3d, M_seed, inv_t, _ = encode_backbone(p, x_mb, rng_key, True)
        loss_t1 = _factorized_mxu_ce(phi_seed_3d.reshape(-1, out_dim), M_seed, inv_t, y_mb.reshape(-1))
        total_loss = loss_t1
        if mtp_horizons > 1 and "mtp_rotations" in p:
            for h_idx in range(1, mtp_horizons):
                T_avail = T - h_idx
                if T_avail <= 0:
                    break
                if mtp_stride > 1 and T_avail > mtp_stride:
                    offset = (h_idx - 1) % mtp_stride
                    phi_sub = phi_seed_3d[:, offset:T_avail:mtp_stride, :]
                    y_h = y_mb[:, h_idx + offset : T : mtp_stride].reshape(-1)
                else:
                    phi_sub = phi_seed_3d[:, :T_avail, :]
                    y_h = y_mb[:, h_idx:].reshape(-1)
                u_h = p["mtp_rotations"][h_idx - 1].astype(jnp.bfloat16)
                z_h = l2_norm_bf16(
                    jnp.matmul(phi_sub, u_h.T, preferred_element_type=jnp.bfloat16)
                ).reshape(-1, out_dim)
                M_aux = jax.lax.stop_gradient(M_seed) if mtp_detach_readout else M_seed
                beta_h = (
                    jnp.clip(p["mtp_inv_temps"][h_idx - 1], 1.0, 60.0)
                    if (mtp_decoupled_temp and "mtp_inv_temps" in p)
                    else jnp.clip(inv_t, 1.0, 60.0)
                )
                loss_h = _factorized_mxu_ce(z_h, M_aux, beta_h, y_h)
                total_loss = total_loss + (mtp_weight / float(h_idx)) * loss_h
        return total_loss

    muon_block_keys = ("w_q", "w_k", "w_v", "w_o", "w_a", "w_b", "w_down")
    muon_top_keys = ("w_lift", "readout_rot", "unet_gates", "mtp_rotations")

    def is_muon_leaf(top_k: str, sub_k: str | None) -> bool:
        if top_k == "blocks":
            return sub_k in muon_block_keys
        return top_k in muon_top_keys

    opt_state = {
        "step": jnp.array(0, dtype=jnp.int32),
        "mom": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v_adam": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(p: dict, state: dict, x_full: jax.Array, y_full: jax.Array, lr_mult: jax.Array, rng_key: jax.Array):
        mb_keys = jax.random.split(rng_key, grad_accum)
        val_grad_fn = jax.value_and_grad(microbatch_loss)

        def accum_body(carry, inputs):
            loss_acc, grad_acc = carry
            x_mb, y_mb, k_mb = inputs
            l_mb, g_mb = val_grad_fn(p, x_mb, y_mb, k_mb)
            new_grad = jax.tree_util.tree_map(lambda a, b: a + b / float(grad_accum), grad_acc, g_mb)
            return (loss_acc + l_mb / float(grad_accum), new_grad), None

        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, p)
        (loss_val, grads), _ = jax.lax.scan(
            accum_body, (jnp.zeros((), dtype=jnp.float32), zero_grads), (x_full, y_full, mb_keys)
        )

        sq_norm = sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree_util.tree_leaves(grads))
        g_norm = jnp.sqrt(sq_norm + 1e-12)
        clip_scale = jnp.minimum(1.0, 1.0 / g_norm)
        grads = jax.tree_util.tree_map(lambda g: g * clip_scale, grads)

        step_num = state["step"] + 1
        do_retract = (step_num % 4) == 0
        lr_muon = float(muon_lr) * lr_mult
        lr_adam = 0.004 * lr_mult
        b1_corr = 1.0 - (0.9 ** step_num.astype(jnp.float32))
        b2_corr = 1.0 - (0.95 ** step_num.astype(jnp.float32))

        def update_muon_2d(w: jax.Array, g: jax.Array, m: jax.Array) -> tuple[jax.Array, jax.Array]:
            new_m = 0.95 * m + g
            if stiefel_optimizer == "tangent":
                m_tan = project_stiefel_tangent_jax(w, new_m)
                u_ns = newton_schulz5_bf16(m_tan, steps=5)
                u = project_stiefel_tangent_jax(w, u_ns)
                cwd_mask = (w * u > 0.0).astype(jnp.float32)
                w_step = w - (lr_muon * 0.01) * (w * cwd_mask) - lr_muon * u
                w_next = stiefel_quadratic_retract_jax(w_step)
                return w_next, new_m
            u = newton_schulz5_bf16(new_m, steps=5)
            cwd_mask = (w * u > 0.0).astype(jnp.float32)
            w_next = w - (lr_muon * 0.01) * (w * cwd_mask)
            r_grad = jax.lax.cond(
                do_retract,
                stiefel_retract_grad,
                lambda x: jnp.zeros_like(x),
                w_next,
            )
            w_next = w_next - (0.005 * 4.0) * r_grad - lr_muon * u
            return w_next, new_m

        def update_adam(w: jax.Array, g: jax.Array, m: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            new_m = 0.9 * m + 0.1 * g
            new_v = 0.95 * v + 0.05 * (g * g)
            m_hat = new_m / b1_corr
            v_hat = new_v / b2_corr
            w_next = w * (1.0 - lr_adam * 0.01) - lr_adam * (m_hat / (jnp.sqrt(v_hat) + 1e-8))
            return w_next, new_m, new_v

        new_p = {}
        new_mom = {}
        new_v = {}

        for top_k in p:
            if top_k == "blocks":
                new_p["blocks"] = {}
                new_mom["blocks"] = {}
                new_v["blocks"] = {}
                for sub_k in p["blocks"]:
                    w_l = p["blocks"][sub_k]
                    g_l = grads["blocks"][sub_k]
                    m_l = state["mom"]["blocks"][sub_k]
                    v_l = state["v_adam"]["blocks"][sub_k]
                    if is_muon_leaf("blocks", sub_k):
                        w_u, m_u = jax.vmap(update_muon_2d)(w_l, g_l, m_l)
                        new_p["blocks"][sub_k] = w_u
                        new_mom["blocks"][sub_k] = m_u
                        new_v["blocks"][sub_k] = v_l
                    else:
                        w_u, m_u, v_u = update_adam(w_l, g_l, m_l, v_l)
                        new_p["blocks"][sub_k] = w_u
                        new_mom["blocks"][sub_k] = m_u
                        new_v["blocks"][sub_k] = v_u
            elif top_k in ("unet_gates", "mtp_rotations"):
                w_u, m_u = jax.vmap(update_muon_2d)(p[top_k], grads[top_k], state["mom"][top_k])
                new_p[top_k] = w_u
                new_mom[top_k] = m_u
                new_v[top_k] = state["v_adam"][top_k]
            elif is_muon_leaf(top_k, None):
                w_u, m_u = update_muon_2d(p[top_k], grads[top_k], state["mom"][top_k])
                new_p[top_k] = w_u
                new_mom[top_k] = m_u
                new_v[top_k] = state["v_adam"][top_k]
            else:
                w_u, m_u, v_u = update_adam(p[top_k], grads[top_k], state["mom"][top_k], state["v_adam"][top_k])
                new_p[top_k] = w_u
                new_mom[top_k] = m_u
                new_v[top_k] = v_u

        new_state = {"step": step_num, "mom": new_mom, "v_adam": new_v}
        return new_p, new_state, loss_val

    @jax.jit
    def eval_batch_step(p: dict, x_val: jax.Array, y_val: jax.Array):
        phi_seed_3d, M_seed, inv_t, layer_states = encode_backbone(p, x_val, None, False)
        phi_seed = phi_seed_3d.reshape(-1, out_dim)
        M_seed_valid = M_seed[:FINEWEB_GPT2_VOCAB_SIZE]
        N = phi_seed.shape[0]
        num_chunks = 4
        chunk_sz = max(1, N // num_chunks)
        phi_c = phi_seed.reshape(num_chunks, chunk_sz, out_dim)
        y_c = y_val.reshape(num_chunks, chunk_sz)
        beta = jnp.clip(inv_t.astype(jnp.bfloat16), 1.0, 60.0)

        def eval_chunk(carry, inputs):
            loss_acc, t1_acc, t5_acc = carry
            zc, yc = inputs
            logits = (jnp.matmul(zc, M_seed_valid.T, preferred_element_type=jnp.bfloat16) * beta).astype(jnp.float32)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(log_probs, yc[:, None], axis=-1).squeeze(-1)
            _, top5_idx = jax.lax.top_k(logits, 5)
            t1 = jnp.sum(top5_idx[:, 0] == yc)
            t5 = jnp.sum(jnp.any(top5_idx == yc[:, None], axis=-1))
            return (loss_acc + jnp.sum(nll), t1_acc + t1, t5_acc + t5), None

        (loss_sum, t1_hits, t5_hits), _ = jax.lax.scan(
            eval_chunk,
            (jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.int32)),
            (phi_c, y_c),
        )
        sub_logits = (jnp.matmul(phi_seed[:1024], M_seed_valid.T, preferred_element_type=jnp.bfloat16) * beta).astype(jnp.float32)
        sub_y = y_val.reshape(-1)[:1024]
        target_scores = jnp.take_along_axis(sub_logits, sub_y[:, None], axis=-1)
        ranks = jnp.sum(sub_logits > target_scores, axis=-1) + 1

        # Compute 10-qubit (5+5 split) bipartite entanglement entropy across all layers on 64 tokens
        def layer_ent(l_psi):
            raw_sub = l_psi.reshape(-1, D)[:64].astype(jnp.float32)
            if D == 1024:
                sub_psi = raw_sub
            elif D > 1024:
                sub_psi = raw_sub[:, :1024]
                sub_psi = sub_psi / jnp.sqrt(jnp.maximum(jnp.sum(sub_psi * sub_psi, axis=-1, keepdims=True), 1e-12))
            else:
                sub_psi = jnp.pad(raw_sub, ((0, 0), (0, 1024 - D)))
                sub_psi = sub_psi / jnp.sqrt(jnp.maximum(jnp.sum(sub_psi * sub_psi, axis=-1, keepdims=True), 1e-12))
            return jnp.mean(bipartite_entanglement_entropy(sub_psi, num_qubits=10, keep_qubits=(0, 1, 2, 3, 4)))

        layer_ents = jax.vmap(layer_ent)(layer_states)
        return loss_sum / float(N), t1_hits, t5_hits, ranks, layer_ents, phi_seed[:512].astype(jnp.float32)

    def compute_isometry_error_tpu(p: dict) -> float:
        errs: list[float] = []
        mats: list[jax.Array] = [p["w_lift"]]
        if "readout_rot" in p:
            mats.append(p["readout_rot"])
        for i in range(L // 2):
            mats.append(p["unet_gates"][i])
        if "mtp_rotations" in p:
            for i in range(p["mtp_rotations"].shape[0]):
                mats.append(p["mtp_rotations"][i])
        for sub_k in muon_block_keys:
            for i in range(L):
                mats.append(p["blocks"][sub_k][i])
        for w in mats:
            wf = np.asarray(w, dtype=np.float32)
            r, c = wf.shape
            if r >= c:
                gram = wf.T @ wf
                err = np.linalg.norm(gram - np.eye(c, dtype=np.float32), ord="fro") / math.sqrt(c)
            else:
                gram = wf @ wf.T
                err = np.linalg.norm(gram - np.eye(r, dtype=np.float32), ord="fro") / math.sqrt(r)
            errs.append(float(err))
        return float(np.mean(errs))

    def evaluate_validation(p: dict, num_batches: int | None = None) -> dict[str, float | list[float]]:
        losses: list[float] = []
        t1_total = 0
        t5_total = 0
        tok_total = 0
        all_ranks: list[np.ndarray] = []
        ent_layers: list[float] = []
        seed_eff_rank = 0.0
        seed_pr = 0.0
        stride = B_micro * T
        max_avail = max(1, (len(val_tokens) - 1) // stride)
        if num_batches is not None and num_batches > 0:
            eff_batches = min(int(num_batches), max_avail)
        else:
            target_batches = max(1, int(val_tokens_budget) // stride)
            eff_batches = min(target_batches, max_avail)
        for b_idx in range(eff_batches):
            base = b_idx * stride
            chunk = val_tokens[base : base + stride + 1].astype(np.int32)
            x_v = jnp.asarray(chunk[:-1].reshape(B_micro, T))
            y_v = jnp.asarray(chunk[1:].reshape(B_micro, T))
            l_b, t1_b, t5_b, ranks_b, l_ents, phi_seed_0 = eval_batch_step(p, x_v, y_v)
            losses.append(float(l_b))
            t1_total += int(t1_b)
            t5_total += int(t5_b)
            tok_total += stride
            if b_idx == 0:
                ent_layers = [round(float(v), 4) for v in np.asarray(l_ents).tolist()]
                seed_eff_rank = round(float(effective_quantum_rank(phi_seed_0)), 2)
                seed_pr = round(float(purity_participation_ratio(phi_seed_0)), 2)
            if b_idx < 2:
                all_ranks.append(np.asarray(ranks_b))
        val_ce = float(np.mean(losses))
        val_ppl = float(math.exp(min(val_ce, 20.0)))
        med_rank = float(np.median(np.concatenate(all_ranks))) if all_ranks else -1.0
        iso_err = compute_isometry_error_tpu(p)
        hs_np = np.clip(np.asarray(p["blocks"]["head_scale"], dtype=np.float32).reshape(L, H), 1.0, 40.0)
        beta_val = float(np.clip(float(p["tied_inv_temp"]), 1.0, 60.0))
        return {
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "top1_acc": 100.0 * t1_total / max(1, tok_total),
            "top5_acc": 100.0 * t5_total / max(1, tok_total),
            "med_rank": med_rank,
            "iso_err": iso_err,
            "entanglement_5v5_mean": round(float(np.mean(ent_layers)), 4) if ent_layers else 0.0,
            "entanglement_5v5_layers": ent_layers,
            "seed_eff_rank": seed_eff_rank,
            "seed_participation_ratio": seed_pr,
            "gamma_head_mean": round(float(np.mean(hs_np)), 4),
            "gamma_head_min": round(float(np.min(hs_np)), 4),
            "gamma_head_max": round(float(np.max(hs_np)), 4),
            "gamma_layer_means": [round(float(x), 4) for x in np.mean(hs_np, axis=1).tolist()],
            "povm_beta": round(beta_val, 4),
        }

    print("Compiling XLA HLO training & evaluation programs on TPU v5e-1...", flush=True)
    t_comp0 = time.perf_counter()
    x_warm = jnp.zeros((grad_accum, B_micro, T), dtype=jnp.int32)
    y_warm = jnp.zeros((grad_accum, B_micro, T), dtype=jnp.int32)
    p_copy = jax.tree_util.tree_map(lambda x: x.copy(), params)
    s_copy = jax.tree_util.tree_map(lambda x: x.copy(), opt_state)
    p_copy, s_copy, l_warm = train_step(p_copy, s_copy, x_warm, y_warm, jnp.array(0.0, dtype=jnp.float32), key)
    jax.block_until_ready(l_warm)
    del p_copy, s_copy
    gc.collect()
    val0 = evaluate_validation(params)
    print(f"XLA compilation complete in {time.perf_counter() - t_comp0:.2f}s", flush=True)

    history: list[dict[str, float]] = []
    print(
        f"[Epoch 0.00 | Step    0 |   0.00M tok] Val CE: {val0['val_ce']:.4f} | Val PPL: {val0['val_ppl']:8.1f} | "
        f"Top-1: {val0['top1_acc']:5.2f}% | Top-5: {val0['top5_acc']:5.2f}% | MedRank: {val0['med_rank']:5.0f} | "
        f"IsoErr: {val0['iso_err']:.4f} | S(rho_A): {val0['entanglement_5v5_mean']:.4f} nats",
        flush=True,
    )
    history.append({"epoch": 0.0, "step": 0, "tokens_M": 0.0, "train_ce": float(val0["val_ce"]), "tok_per_s": 0.0, **val0})  # type: ignore[arg-type]

    warmup_steps = min(150, max(1, total_steps // 5))
    running_loss_arr = jnp.zeros((), dtype=jnp.float32)
    running_steps = 0
    train_t0 = time.perf_counter()
    window_t0 = time.perf_counter()
    window_tokens = 0
    save_path = out_json if out_json is not None else Path("/content/tpu_100M_results.json")

    global_step = 0
    step_rng = jax.random.PRNGKey(100)

    for ep in range(num_epochs):
        if stream is not None and ep > 0:
            stream.prefetch_shard(1)

        for shard_idx in range(1, num_shards + 1):
            if stream is not None:
                if ep == 0 and shard_idx == 1:
                    active_shard = train_tokens
                    shard_wait_ms = first_wait_ms
                else:
                    active_shard, shard_wait_ms = stream.get_shard(shard_idx)
                print(
                    f"  [Shard {shard_idx}/{num_shards} | Epoch {ep + 1}/{num_epochs}] "
                    f"Active ({len(active_shard):,} tokens | prefetch wait: {shard_wait_ms:.1f} ms)",
                    flush=True,
                )
                rng = np.random.default_rng(42 + ep * 1000 + shard_idx)
            else:
                active_shard = train_tokens
                rng = np.random.default_rng(42 + ep)

            batch_order = rng.permutation(steps_per_shard)

            for b_pos in range(steps_per_shard):
                global_step += 1
                if schedule == "wsd":
                    lr_mult = wsd_lr_multiplier(
                        global_step,
                        total_steps,
                        warmup_steps=warmup_steps,
                        decay_fraction=wsd_decay_fraction,
                        min_lr_ratio=wsd_min_lr_ratio,
                    )
                else:
                    lr_mult = cosine_lr_multiplier(
                        global_step,
                        total_steps,
                        warmup_steps=warmup_steps,
                        min_lr_ratio=0.15,
                    )

                chunk_idx = int(batch_order[b_pos])
                start_tok = chunk_idx * tok_per_step
                buf = active_shard[start_tok : start_tok + tok_per_step + 1].astype(np.int32)
                x_np = buf[:-1].reshape(grad_accum, B_micro, T)
                y_np = buf[1:].reshape(grad_accum, B_micro, T)
                x_j = jnp.asarray(x_np)
                y_j = jnp.asarray(y_np)

                step_rng, sub_k = jax.random.split(step_rng)
                params, opt_state, loss_j = train_step(
                    params, opt_state, x_j, y_j, jnp.array(lr_mult, dtype=jnp.float32), sub_k
                )

                running_loss_arr = running_loss_arr + loss_j
                running_steps += 1
                window_tokens += tok_per_step

                shard_step = b_pos + 1
                if num_shards == 1:
                    do_eval = (global_step == eval_every + (ep * steps_per_epoch)) or (
                        global_step == (ep + 1) * steps_per_epoch
                    )
                else:
                    do_eval = (
                        shard_step % eval_every == 0
                        and (steps_per_shard - shard_step) >= max(1, eval_every // 2)
                    ) or (shard_step == steps_per_shard)

                if global_step % 150 == 0 and (not do_eval) and (global_step != total_steps):
                    train_ce_mid = float(running_loss_arr) / max(1, running_steps)
                    dt_w = time.perf_counter() - window_t0
                    tps = window_tokens / max(dt_w, 1e-6)
                    ep_frac = global_step / float(steps_per_epoch)
                    tok_m = (global_step * tok_per_step) / 1e6
                    print(
                        f"  -> Step {global_step:4d}/{total_steps} (Epoch {ep_frac:.2f} | {tok_m:6.2f}M tok) | "
                        f"Train CE: {train_ce_mid:.4f} | Speed: {tps:,.0f} tok/s",
                        flush=True,
                    )

                if do_eval:
                    train_ce = float(running_loss_arr) / max(1, running_steps)
                    dt_w = time.perf_counter() - window_t0
                    tps = window_tokens / max(dt_w, 1e-6)
                    running_loss_arr = jnp.zeros((), dtype=jnp.float32)
                    running_steps = 0

                    val_metrics = evaluate_validation(params)
                    if num_shards == 1:
                        ep_frac = round((global_step / float(steps_per_epoch)) * 2.0) / 2.0
                    else:
                        ep_frac = round(global_step / float(steps_per_epoch), 4)
                    tok_m = (global_step * tok_per_step) / 1e6
                    print(
                        f"[Epoch {ep_frac:4.2f} | Step {global_step:4d} | {tok_m:6.2f}M tok] "
                        f"Train CE: {train_ce:.4f} | Val CE: {val_metrics['val_ce']:.4f} | Val PPL: {val_metrics['val_ppl']:6.1f} | "
                        f"Top-1: {val_metrics['top1_acc']:5.2f}% | Top-5: {val_metrics['top5_acc']:5.2f}% | "
                        f"MedRank: {val_metrics['med_rank']:4.0f} | IsoErr: {val_metrics['iso_err']:.4f} | "
                        f"S(rho_A): {val_metrics['entanglement_5v5_mean']:.4f} | gamma_h: {val_metrics['gamma_head_mean']:.2f} | "
                        f"beta: {val_metrics['povm_beta']:.2f} | {tps:,.0f} tok/s",
                        flush=True,
                    )
                    history.append(
                        {
                            "epoch": ep_frac,
                            "shard_idx": shard_idx,
                            "step": global_step,
                            "tokens_M": tok_m,
                            "lr_mult": round(lr_mult, 4),
                            "train_ce": train_ce,
                            "tok_per_s": tps,
                            **val_metrics,  # type: ignore[arg-type]
                        }
                    )
                    try:
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_text(json.dumps(history, indent=2))
                    except Exception:
                        pass
                    window_t0 = time.perf_counter()
                    window_tokens = 0

    if stream is not None:
        stream.close()
    total_time = time.perf_counter() - train_t0
    print("=" * 90, flush=True)
    print(f"COMPLETED {num_epochs} FULL EPOCHS ON TPU v5e-1 IN {total_time:.1f}s ({total_time/60:.2f} min)", flush=True)
    print("=" * 90, flush=True)
    return history


def main() -> None:
    args = parse_args()
    backend = args.backend
    if backend == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                backend = "cuda"
        except ImportError:
            pass
    if backend == "auto":
        try:
            import jax

            if jax.default_backend() == "tpu":
                backend = "tpu"
        except ImportError:
            pass

    if backend == "cuda":
        run_cuda_benchmark(
            num_epochs=args.epochs,
            num_shards=args.num_shards,
            eval_every_tokens_m=args.eval_every_tokens_m,
            data_dir=args.data_dir,
            schedule=args.schedule,
            wsd_decay_fraction=args.wsd_decay_fraction,
            wsd_min_lr_ratio=args.wsd_min_lr_ratio,
            seq_curriculum_str=args.seq_curriculum,
            mtp_horizons=args.mtp_horizons,
            mtp_weight=args.mtp_weight,
            seed_dim=args.seed_dim,
            mtp_stride=args.mtp_stride,
            mtp_detach_readout=args.mtp_detach_readout,
            mtp_decoupled_temp=args.mtp_decoupled_temp,
            stiefel_optimizer=args.stiefel_optimizer,
            kraus_beams=args.kraus_beams,
            fused_povm_tile_size=args.fused_povm_tile_size,
            overcomplete_frame=args.overcomplete_frame,
            overcomplete_seed_adapter=args.overcomplete_seed_adapter,
            overcomplete_two_beam=args.overcomplete_two_beam,
            two_beam_seed_adapter=args.two_beam_seed_adapter,
            num_layers=args.num_layers,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            frame_expansion=args.frame_expansion,
            readout_mode=args.readout_mode,
            depth_scaled_residual=args.depth_scaled_residual,
            architecture=args.architecture,
            dynamic_povm_beta=args.dynamic_povm_beta,
            ortho_decouple=args.ortho_decouple,
            povm_log_prior=args.povm_log_prior,
            gated_spherical_residual=args.gated_spherical_residual,
            purified_born_attn=args.purified_born_attn,
            zero_init_classical=args.zero_init_classical,
            true_layer0_v_residual=args.true_layer0_v_residual,
            classical_stiefel_penalty=args.classical_stiefel_penalty,
            stage_shard_on_gpu=args.stage_shard_on_gpu,
            direct_chunked_ce=args.direct_chunked_ce,
            unet_skip_mode=args.unet_skip_mode,
            fused_qkv=args.fused_qkv,
            num_value_embeddings=args.num_value_embeddings,
            skip_attn_layers_str=args.skip_attn_layers,
            zero_half_rope=args.zero_half_rope,
            doc_mask_attn=args.doc_mask_attn,
            sliding_window=args.sliding_window,
            leaderboard_exact_classical=args.leaderboard_exact_classical,
            progressive_window=args.progressive_window,
            ppmi_svd_init=args.ppmi_svd_init,
            ppmi_svd_shards=args.ppmi_svd_shards,
            ppmi_svd_alpha=args.ppmi_svd_alpha,
            ngram_hash=args.ngram_hash,
            ngram_vocab_size=args.ngram_vocab_size,
            polar_express=args.polar_express,
            normuon=args.normuon,
            twin_rail_muon=args.twin_rail_muon,
            cautious_wd=args.cautious_wd,
            tail_ema=args.tail_ema,
            mudd_and_gates=args.mudd_and_gates,
            smear_gate=args.smear_gate,
            tapered_mtp=args.tapered_mtp,
            align_to_bos=args.align_to_bos,
            sampled_softmax_candidates=args.sampled_softmax_candidates,
            embed_step_every=args.embed_step_every,
            fp8_linear=args.fp8_linear,
            sparse_64qubit=args.sparse_64qubit,
            num_qubits=args.num_qubits,
            sparse_ngram_k=args.sparse_ngram_k,
            sparse_density_layer=args.sparse_density_layer,
            density_matrix_rank=args.density_matrix_rank,
            no_fock_coupler=args.no_fock_coupler,
            hierarchical_16qubit=args.hierarchical_16qubit,
            qubit_registers=args.qubit_registers,
            qubit_amplitudes=args.qubit_amplitudes,
            num_higher_states=args.num_higher_states,
            no_block_hierarchy_bridge=args.no_block_hierarchy_bridge,
            last_layer_softmax=args.last_layer_softmax,
            two_beam_kboson_readout=args.two_beam_kboson_readout,
            diffuse_rank=args.diffuse_rank,
            logit_softcap=args.logit_softcap,
            muon_momentum_warmup=args.muon_momentum_warmup,
            muon_lr=args.muon_lr,
            optimizer_mode=args.optimizer_mode,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seq_len=args.seq_len,
            val_tokens_budget=args.val_tokens,
            synthetic_tokens=args.synthetic_tokens,
            out_json=args.out_json,
        )
    elif backend == "tpu":
        run_tpu_benchmark(
            num_epochs=args.epochs,
            num_shards=args.num_shards,
            eval_every_tokens_m=args.eval_every_tokens_m,
            data_dir=args.data_dir,
            schedule=args.schedule,
            wsd_decay_fraction=args.wsd_decay_fraction,
            wsd_min_lr_ratio=args.wsd_min_lr_ratio,
            mtp_horizons=args.mtp_horizons,
            mtp_weight=args.mtp_weight,
            seed_dim=args.seed_dim,
            mtp_stride=args.mtp_stride,
            mtp_detach_readout=args.mtp_detach_readout,
            mtp_decoupled_temp=args.mtp_decoupled_temp,
            stiefel_optimizer=args.stiefel_optimizer,
            fused_povm_tile_size=args.fused_povm_tile_size,
            num_layers=args.num_layers,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            frame_expansion=args.frame_expansion,
            readout_mode=args.readout_mode,
            depth_scaled_residual=args.depth_scaled_residual,
            architecture=args.architecture,
            dynamic_povm_beta=args.dynamic_povm_beta,
            ortho_decouple=args.ortho_decouple,
            povm_log_prior=args.povm_log_prior,
            gated_spherical_residual=args.gated_spherical_residual,
            purified_born_attn=args.purified_born_attn,
            zero_init_classical=args.zero_init_classical,
            true_layer0_v_residual=args.true_layer0_v_residual,
            classical_stiefel_penalty=args.classical_stiefel_penalty,
            muon_lr=args.muon_lr,
            tpu_attn_kernel=args.tpu_attn_kernel,
            val_tokens_budget=args.val_tokens,
            synthetic_tokens=args.synthetic_tokens,
            out_json=args.out_json,
        )
    else:
        raise RuntimeError(f"No CUDA GPU or Cloud TPU detected for backend={args.backend!r}.")


if __name__ == "__main__":
    main()


