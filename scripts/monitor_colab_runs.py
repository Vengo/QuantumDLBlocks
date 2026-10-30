#!/usr/bin/env python3
"""Live Experiment Collector, 10-Session Colab Fleet Monitor & Multi-Run Dashboard for QuantumDLBlocks.

Polls all 10 remote Colab GPU & TPU sessions (`gpu_slowrun`, `gpu_exp01`..`gpu_exp06`,
`tpu_slowrun`, `tpu_exp01`, `tpu_exp02`) for:
- Detached training subprocess logs (`/content/*_hyp.log`) and JSON checkpoints (`/content/*_hyp_results.json`)
- Live GPU hardware telemetry via `nvidia-smi` (SM %, VRAM BW %, VRAM MiB, Power W, Temp C, Clock MHz)
- Live TPU v5e-1 hardware telemetry via `libtpu`'s gRPC `RuntimeMetricService` on `localhost:8431`
  (TensorCore/MXU Duty Cycle %, Live HBM Usage GB / Total HBM GB)
- Exact Analytical Model FLOPs Utilization (MFU) and Hardware FLOPs Utilization (HFU)
- Target Benchmark Comparisons against:
  * FineWeb 100M Slowrun World Record: 3.183 Val CE
  * Modded-NanoGPT Speedrun World Record: 3.280 Val CE in 0.665 minutes (39.9s)
- JAX + Equinox + Haliax + Levanter/Optax Named-Axis Parameter Hierarchy, XLA HLO `cost_analysis()`
  / `memory_analysis()`, and Bipartite Qubit Entanglement Entropy telemetry (`qdlblocks_jax`).

Hosted on `0.0.0.0:8765` by default so it is accessible from any browser on your LAN.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.parse
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.db import (
    DEFAULT_DB_PATH,
    BenchmarkSQLiteDB,
    backfill_micro_verifications,
    classify_run_tier,
    compute_family_pareto_and_obsolete,
)
from qdlblocks.benchmarks.model_card import (
    BASELINE_RUN_KEYS,
    DynamicWaveRegistry,
    build_model_card_from_manifest_entry,
    infer_wave_id_and_num,
    render_model_card_markdown,
    save_model_card_atomic,
    set_model_card_obsolete,
    update_model_card_telemetry,
)

LIVE_DIR = ROOT / ".cache" / "live_logs"
CONFIGS_DIR = ROOT / "configs"
MODEL_CARDS_DIR = ROOT / ".cache" / "model_cards"
_SERVER_START_TIME = time.time()

DEFAULT_OBSOLETE_KEYS: dict[str, str] = {
    "gpu_5ep": "Superseded single-shard 100M×5ep baseline (Plan 001)",
    "gpu_3ep": "Superseded single-shard 100M×3ep baseline (Plan 001)",
    "gpu_wsd_mtp": "Superseded single-shard 100M×5ep WSD+MTP baseline (Plan 001)",
    "tpu_3ep": "Superseded single-shard TPU 100M×3ep baseline (Plan 001)",
    "tpu_wsd_mtp": "Superseded single-shard TPU 100M×3ep WSD+MTP baseline (Plan 001)",
    "w5_900m_deep12_adapter384_oom": "Historical OOM crash prior to gradient checkpointing",
    "w8_900m_classical_speedrun_12L_t512_layer0": "Stopped early at 200M tok (superseded by Wave-9/10 PR#360 runs)",
    "w7_900m_qborn_earlyfix_frame4d_synergy512": "Preempted at 609.5M tok for 2.79x faster fused Triton flash_purified_born_attention (succeeded by w9_g4_1 3.7495 & w10_g4_3 3.6926)",
    "w7_900m_qborn_earlyfix_asym1024_L8": "Preempted at 471.9M tok for fused Triton flash_purified_born_attention (succeeded by w9_a100_1 3.7203 & w10_g4_3 3.6926)",
    "w7_900m_qborn_earlyfix_L12_d768": "Preempted at 324.4M tok for fused Triton flash_purified_born_attention (succeeded by w10_a100_1 3.7077)",
    "w7_900m_qborn_earlyfix_synergy_t1024": "Superseded incomplete Wave-7 T=1024 exploratory run",
    "w7_900m_qborn_purified_attn_ablation": "Superseded incomplete Wave-7 unfused purified attention ablation",
    "w8_900m_qborn_layer0v_earlyfix_frame4d_s512": "Superseded incomplete Wave-8 run (succeeded by Wave-9/10)",
    "w8_900m_qborn_layer0v_earlyfix_t1024": "Superseded incomplete Wave-8 T=1024 run",
    "w8_900m_qborn_earlyfix_flash_asym1024_L8_layer0v": "Superseded unlaunched Wave-8 spec (succeeded by Wave-9/10/11 Asym1024 L=8 runs)",
    "w8_900m_classical_speedrun_12L_t512_layer0v_norm": "Preempted at 216.3M tok (superseded by Wave-9 PR#360 3.4280)",
    "w6_900m_classical_modded_gpt_baseline": "Superseded Wave-6 classical baseline (3.6797, succeeded by Wave-9 PR#360 3.4280)",
    "w8_900m_classical_truev0_zeroinit_wsd0": "Superseded Wave-8 classical baseline (3.5882, succeeded by Wave-9 PR#360 3.4280)",
    "w9_900m_classical_speedrun_v2_t512_b524k": "Superseded Wave-9 T=512 classical ablation",
    "w10_900m_classical_speedrun_v3_fastsdpa_b131k": "Superseded Wave-10 small-batch B=131k unmasked classical ablation (3.4498, IsoErr=185.18)",
    "w10_900m_classical_speedrun_v3_b65k_13725steps": "Superseded Wave-10 small-batch B=65k 13,725-step classical ablation (3.5145)",
    "w14_local_qborn_10qubit_d1024_baseline": "Superseded 200-step (1.64M-tok) local pre-launch micro-run prior to modded-nanogpt 10.48M val parity",
    "w14_local_64qubit_sparse_fock2048_4m_density_rank1": "Superseded 200-step (1.64M-tok) local pre-launch micro-run prior to modded-nanogpt 10.48M val parity",
    "w14_local_64qubit_sparse_fock2048_4m_density_rank4_mixed": "Superseded 200-step (1.64M-tok) local pre-launch micro-run prior to modded-nanogpt 10.48M val parity",
    "w14_local_64qubit_sparse_fock2048_no_coupler_ablation": "Superseded 200-step (1.64M-tok) local pre-launch micro-run prior to modded-nanogpt 10.48M val parity",
    "w14_local_64qubit_sparse_fock2048_k32_vs_k8_ablation": "Superseded 200-step (1.64M-tok) local pre-launch micro-run prior to modded-nanogpt 10.48M val parity",
    "w15_900m_qborn_16x32_qubit_256hier_flagship_L8": "Stopped at 175M tok (4.6020 Val CE) and superseded by Plan 024 Calibrated Last-Layer Softmax hotfix (w15_900m_qborn_16x32_qubit_256hier_flagship_softmax_L8)",
    "w15_900m_qborn_16x32_qubit_256hier_deep_L12_H8_d128": "Stopped at 175M tok (4.5966 Val CE) and superseded by Plan 024 Calibrated Last-Layer Softmax + VE Init hotfix (w15_900m_qborn_16x32_qubit_256hier_deep_L12_softmax)",
    "w15_900m_qborn_16x32_qubit_512hier_L8": "Stopped at 175M tok (4.6020 Val CE) and superseded by Plan 024 Calibrated Last-Layer Softmax hotfix (w15_900m_qborn_16x32_qubit_512hier_softmax_L8)",
    "w15_900m_qborn_16x32_qubit_256hier_rank4_density_L8": "Stopped at 100M tok (4.8648 Val CE) due to 5.4x slower Rank-4 density channel throughput and superseded by Plan 024 N-Gram + Calibrated Softmax hotfix (w15_900m_qborn_16x32_qubit_256hier_ngram_softmax_L8)",
    "w15_900m_qborn_16x32_qubit_256hier_flagship_softmax_L8": "Colab VM lease expired at 775M tok (3.6531 Val CE, 40.42% Top-1) and superseded by Wave 16 100% Quantum Two-Beam k-Boson Flagship (w16_900m_qborn_16x32_256hier_twobeam_kboson_flagship_L8)",
    "w15_900m_qborn_16x32_qubit_256hier_deep_L12_softmax": "Colab VM lease expired at 750M tok (3.6353 Val CE, 40.55% Top-1) and superseded by Wave 16 Deep L=12 Unified Two-Beam k-Boson (w16_900m_qborn_16x32_256hier_unified_deep_L12_twobeam_kboson)",
    "w15_900m_qborn_16x32_qubit_512hier_softmax_L8": "Colab VM lease expired at 750M tok (3.6562 Val CE, 40.37% Top-1) and superseded by Wave 16 Unified W13+W15 Two-Beam k-Boson (w16_900m_qborn_16x32_256hier_unified_w13w15_twobeam_kboson_L8)",
    "w15_900m_qborn_16x32_qubit_256hier_ngram_softmax_L8": "Colab VM lease expired at 450M tok (3.7330 Val CE, 39.21% Top-1) and superseded by Wave 16 N-Gram Two-Beam k-Boson (w16_900m_qborn_16x32_256hier_ngram_twobeam_kboson_L8)",
    "w18_8gb_L2_d256_bipolar_kboson_whitened_300steps": "Falsified Plan 031 ablation (6.3250 vs Plan 030 6.1960): convex bipolar hole -0.85*k*ln(1+|x|) inflated 49,000 negative-hemisphere background tokens in Z and hard Cholesky diffuse whitening (R_diff=16.00) destroyed Zipfian syntactic SNR",
    "w18_8gb_L4_d256_bipolar_kboson_whitened_900steps": "Aborted & falsified Plan 031 ablation at Step 300 (6.4489 vs Plan 030 6.1391): superseded by Plan 032 clean J_anti + gentle 2x zero-WD qubit LR (w18_8gb_L4_d256_clean_janti_gentle_lr_900steps)",
}

_COLAB_EXEC_LOCK = threading.Lock()

STEP_RE = re.compile(
    r"Step\s+(\d+)/(\d+)\s+\(Epoch\s+([\d.]+)\s+\|\s+([\d.]+)M tok\)\s+\|\s+Train CE:\s+([\d.]+)\s+\|\s+Speed:\s+([\d,]+)\s+tok/s"
)
EVAL_RE = re.compile(
    r"\[Epoch\s+([\d.]+)\s+\|\s+Step\s+(\d+)\s+\|\s+([\d.]+)M tok(?:\s+\|\s+T=(\d+))?\].*?Val CE:\s+([\d.]+)\s+\|\s+Val PPL:\s+([\d.]+)\s+\|\s+Top-1:\s+([\d.]+)%\s+\|\s+Top-5:\s+([\d.]+)%\s+\|\s+MedRank:\s+(\d+)\s+\|\s+IsoErr:\s+([\d.]+)"
    r"(?:\s+\|\s+S\(rho_A\):\s+([\d.]+)(?:\s+nats)?\s+\|\s+gamma_h:\s+([\d.]+)\s+\|\s+beta:\s+([\d.]+))?"
)

ANALYTICAL_FLOPS_PER_TOK = 461.73e6
HARDWARE_FLOPS_PER_TOK = 548.40e6
TPU_V5E_BF16_PEAK_TFLOPS = 197.0
BLACKWELL_RTX6000_BF16_PEAK_TFLOPS = 503.8
A100_BF16_PEAK_TFLOPS = 312.0
L4_BF16_PEAK_TFLOPS = 121.0
MAX_5QUBIT_ENTROPY_NATS = 3.4657359  # 5 * ln(2)

# External Community Benchmark Records (FineWeb GPT-2 50,257 Vocab Validation Set)
SLOWRUN_RECORD_VAL_CE = 3.183
SLOWRUN_RECORD_VAL_PPL = round(math.exp(SLOWRUN_RECORD_VAL_CE), 1)
NANOGPT_SPEEDRUN_VAL_CE = 3.280
NANOGPT_SPEEDRUN_VAL_PPL = round(math.exp(NANOGPT_SPEEDRUN_VAL_CE), 1)
NANOGPT_SPEEDRUN_MINUTES = 0.665

# 10-Session Colab GPU/TPU Fleet Specification
FLEET_SPECS = [
    {
        "session": "gpu_slowrun",
        "key": "hyp_gpu_slowrun",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 275000.0,
        "log": "/content/gpu_slowrun_hyp.log",
        "json": "/content/gpu_slowrun_hyp_results.json",
        "label": "GPU G4 All-Combined (WSD+TangentStiefel+StridedMTP+D256+TiledPOVM)",
        "short_label": "G4 All-Combined (WSD+Tangent+MTP s=4+D256)",
        "color": "#00f2fe",
    },
    {
        "session": "gpu_exp01",
        "key": "hyp_gpu_exp01",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 295000.0,
        "log": "/content/gpu_exp01_hyp.log",
        "json": "/content/gpu_exp01_hyp_results.json",
        "label": "GPU G4 Plan 003 TangentStiefelMuon + WSD + Strided Decoupled MTP=3",
        "short_label": "G4 Plan 003 TangentStiefel + WSD + MTP(s=4)",
        "color": "#38ef7d",
    },
    {
        "session": "gpu_exp02",
        "key": "hyp_gpu_exp02",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 295000.0,
        "log": "/content/gpu_exp02_hyp.log",
        "json": "/content/gpu_exp02_hyp_results.json",
        "label": "GPU G4 Plan 002 H1+H2 Strided Decoupled MTP=3 (Std Stiefel Control)",
        "short_label": "G4 Plan 002 Strided Decoupled MTP(s=4)",
        "color": "#f6d365",
    },
    {
        "session": "gpu_exp03",
        "key": "hyp_gpu_exp03",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 175000.0,
        "log": "/content/gpu_exp03_hyp.log",
        "json": "/content/gpu_exp03_hyp_results.json",
        "label": "GPU A100 Plan 002 H3 Expanded Seed D_seed=256 + TangentStiefelMuon",
        "short_label": "A100 Plan 002 H3 D_seed=256 + Tangent",
        "color": "#ff9a9e",
    },
    {
        "session": "gpu_exp06",
        "key": "hyp_gpu_exp06",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 165000.0,
        "log": "/content/gpu_exp06_hyp.log",
        "json": "/content/gpu_exp06_hyp_results.json",
        "label": "GPU A100 Plan 004 Multi-Beam Kraus Readout K=4 + TangentStiefelMuon",
        "short_label": "A100 Plan 004 Kraus Readout K=4 + Tangent",
        "color": "#a18cd1",
    },
    {
        "session": "gpu_exp04",
        "key": "hyp_gpu_exp04",
        "kind": "gpu",
        "accel": "NVIDIA L4 (24GB)",
        "peak_tflops": L4_BF16_PEAK_TFLOPS,
        "default_tps": 65000.0,
        "log": "/content/gpu_exp04_hyp.log",
        "json": "/content/gpu_exp04_hyp_results.json",
        "label": "GPU L4 Plan 005 Analytic L2-Tiled POVM CE (tile=512) + TangentStiefel",
        "short_label": "L4 Plan 005 Analytic Tiled POVM CE",
        "color": "#fbc2eb",
    },
    {
        "session": "gpu_exp05",
        "key": "hyp_gpu_exp05",
        "kind": "gpu",
        "accel": "NVIDIA L4 (24GB)",
        "peak_tflops": L4_BF16_PEAK_TFLOPS,
        "default_tps": 65000.0,
        "log": "/content/gpu_exp05_hyp.log",
        "json": "/content/gpu_exp05_hyp_results.json",
        "label": "GPU L4 Progressive Sequence Curriculum (256->512) + TangentStiefel",
        "short_label": "L4 Seq Curriculum (256->512) + Tangent",
        "color": "#fdcbf1",
    },
    {
        "session": "tpu_slowrun",
        "key": "hyp_tpu_slowrun",
        "kind": "tpu",
        "accel": "Cloud TPU v5e-1",
        "peak_tflops": TPU_V5E_BF16_PEAK_TFLOPS,
        "default_tps": 120000.0,
        "log": "/content/tpu_slowrun_hyp.log",
        "json": "/content/tpu_slowrun_hyp_results.json",
        "label": "TPU v5e-1 Plan 003 TangentStiefel + WSD + Strided Decoupled MTP=3",
        "short_label": "TPU Plan 003 TangentStiefel + WSD + MTP",
        "color": "#ff0844",
    },
    {
        "session": "tpu_exp01",
        "key": "hyp_tpu_exp01",
        "kind": "tpu",
        "accel": "Cloud TPU v5e-1",
        "peak_tflops": TPU_V5E_BF16_PEAK_TFLOPS,
        "default_tps": 115000.0,
        "log": "/content/tpu_exp01_hyp.log",
        "json": "/content/tpu_exp01_hyp_results.json",
        "label": "TPU v5e-1 Plan 002 H3 Expanded Seed D_seed=256 + TangentStiefel",
        "short_label": "TPU Plan 002 H3 D_seed=256 + Tangent",
        "color": "#ffb199",
    },
    {
        "session": "tpu_exp02",
        "key": "hyp_tpu_exp02",
        "kind": "tpu",
        "accel": "Cloud TPU v5e-1",
        "peak_tflops": TPU_V5E_BF16_PEAK_TFLOPS,
        "default_tps": 122000.0,
        "log": "/content/tpu_exp02_hyp.log",
        "json": "/content/tpu_exp02_hyp_results.json",
        "label": "TPU v5e-1 Plan 005 Analytic Tiled POVM CE + Pallas Flash-Born",
        "short_label": "TPU Plan 005 Tiled POVM + Pallas Flash-Born",
        "color": "#4facfe",
    },
    {
        "session": "gpu_w2_frame256",
        "key": "w2_gpu_frame256",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 270000.0,
        "log": "/content/gpu_w2_frame256.log",
        "json": "/content/gpu_w2_frame256_results.json",
        "label": "GPU G4 Wave-2 Plan 006 Overcomplete 2d Frame + D_seed=256 (Cosine)",
        "short_label": "W2 G4 Overcomplete 2d Frame + D_seed=256",
        "color": "#00ff87",
    },
    {
        "session": "gpu_w2_frame_wsd",
        "key": "w2_gpu_frame_wsd",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 250000.0,
        "log": "/content/gpu_w2_frame_wsd.log",
        "json": "/content/gpu_w2_frame_wsd_results.json",
        "label": "GPU G4 Wave-2 Overcomplete 2d Frame + D_seed=256 + WSD + Strided MTP=3",
        "short_label": "W2 G4 Overcomplete 2d + D256 + WSD + MTP",
        "color": "#60efff",
    },
    {
        "session": "gpu_w2_seed384",
        "key": "w2_gpu_seed384",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 260000.0,
        "log": "/content/gpu_w2_seed384.log",
        "json": "/content/gpu_w2_seed384_results.json",
        "label": "GPU G4 Wave-2 Expanded Seed D_seed=384 (3x Rank) + Relaxed Isometry",
        "short_label": "W2 G4 Expanded Seed D_seed=384 (3x Rank)",
        "color": "#ff61d2",
    },
    {
        "session": "gpu_w2_seed512",
        "key": "w2_gpu_seed512",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 160000.0,
        "log": "/content/gpu_w2_seed512.log",
        "json": "/content/gpu_w2_seed512_results.json",
        "label": "GPU A100 Wave-2 Expanded Seed D_seed=512 (4x Rank) + Overcomplete 2d Frame",
        "short_label": "W2 A100 D_seed=512 + Overcomplete 2d Frame",
        "color": "#fe9090",
    },
    {
        "session": "gpu_w2_seed1024",
        "key": "w2_gpu_seed1024",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 140000.0,
        "log": "/content/gpu_w2_seed1024.log",
        "json": "/content/gpu_w2_seed1024_results.json",
        "label": "GPU A100 Wave-2 Full-Hilbert Seed D_seed=1024 (Zero Subspace Bottleneck)",
        "short_label": "W2 A100 Full-Hilbert Seed D_seed=1024",
        "color": "#f9f871",
    },
    {
        "session": "tpu_w2_seed512",
        "key": "w2_tpu_seed512",
        "kind": "tpu",
        "accel": "Cloud TPU v5e-1",
        "peak_tflops": TPU_V5E_BF16_PEAK_TFLOPS,
        "default_tps": 110000.0,
        "log": "/content/tpu_w2_seed512.log",
        "json": "/content/tpu_w2_seed512_results.json",
        "label": "TPU v5e-1 Wave-2 Expanded Seed D_seed=512 (4x Rank) + Relaxed Isometry",
        "short_label": "W2 TPU v5e-1 Expanded Seed D_seed=512",
        "color": "#ff9f43",
    },
    {
        "session": "gpu_w2_frame256",
        "key": "w2_gpu_adapter256",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 255000.0,
        "log": "/content/gpu_w2_adapter256.log",
        "json": "/content/gpu_w2_adapter256_results.json",
        "label": "GPU G4 Plan 007 OvercompleteSeedReadoutAdapter + D_seed=256 (Cosine)",
        "short_label": "Plan 007 G4 SeedAdapter + D_seed=256",
        "color": "#00d2ff",
    },
    {
        "session": "gpu_w2_seed384",
        "key": "w2_gpu_adapter384",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 240000.0,
        "log": "/content/gpu_w2_adapter384.log",
        "json": "/content/gpu_w2_adapter384_results.json",
        "label": "GPU G4 Plan 007 OvercompleteSeedReadoutAdapter + Overcomplete 2d Frame + D_seed=384",
        "short_label": "Plan 007 G4 SeedAdapter + 2d Frame + D384",
        "color": "#f857a6",
    },
    {
        "session": "gpu_w2_frame_wsd",
        "key": "w2_gpu_adapter_wsd",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 230000.0,
        "log": "/content/gpu_w2_adapter_wsd.log",
        "json": "/content/gpu_w2_adapter_wsd_results.json",
        "label": "GPU G4 Plan 007 SeedAdapter + Overcomplete 2d Frame + D_seed=384 + WSD + MTP=3",
        "short_label": "Plan 007 G4 SeedAdapter + D384 + WSD + MTP",
        "color": "#a8ff78",
    },
    {
        "session": "gpu_w2_seed384",
        "key": "w3_gpu_adapter384_2beam",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 285000.0,
        "log": "/content/w3_gpu_adapter384_2beam.log",
        "json": "/content/w3_gpu_adapter384_2beam_results.json",
        "label": "GPU G4 Plan 008 Two-Beam (1536x2) + Plan 007 SeedAdapter + D_seed=384 (Cosine)",
        "short_label": "W3 G4 2-Beam(1536x2) + SeedAdapter + D384",
        "color": "#ff3366",
    },
    {
        "session": "gpu_w2_frame256",
        "key": "w3_gpu_seed512_2beam",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 265000.0,
        "log": "/content/w3_gpu_seed512_2beam.log",
        "json": "/content/w3_gpu_seed512_2beam_results.json",
        "label": "GPU G4 Plan 008 Two-Beam (1536x2) + D_seed=512 Clean Linear Seed (Cosine)",
        "short_label": "W3 G4 2-Beam(1536x2) + D_seed=512 (Clean)",
        "color": "#33ffcc",
    },
    {
        "session": "gpu_w2_seed512",
        "key": "w3_gpu_adapter512_2beam",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 165000.0,
        "log": "/content/w3_gpu_adapter512_2beam.log",
        "json": "/content/w3_gpu_adapter512_2beam_results.json",
        "label": "GPU A100 Plan 008 Two-Beam (1536x2) + Plan 007 SeedAdapter + D_seed=512 (Cosine)",
        "short_label": "W3 A100 2-Beam(1536x2) + SeedAdapter + D512",
        "color": "#ffcc00",
    },
    {
        "session": "gpu_w2_seed1024",
        "key": "w3_gpu_adapter1024_2beam",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 145000.0,
        "log": "/content/w3_gpu_adapter1024_2beam.log",
        "json": "/content/w3_gpu_adapter1024_2beam_results.json",
        "label": "GPU A100 Plan 008 Two-Beam (1536x2) + Plan 007 SeedAdapter + D_seed=1024 (Cosine)",
        "short_label": "W3 A100 2-Beam(1536x2) + SeedAdapter + D1024",
        "color": "#ffffff",
    },
    {
        "session": "gpu_w2_adapter256",
        "key": "w3_gpu_2beam_frame384",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 250000.0,
        "log": "/content/w3_gpu_2beam_frame384.log",
        "json": "/content/w3_gpu_2beam_frame384_results.json",
        "label": "GPU G4 Plan 009 OvercompleteTwoBeam (2048x2) + D_seed=384 Clean Linear Seed (Cosine)",
        "short_label": "W3 G4 Plan 009 2d-TwoBeam(2048x2) + D384",
        "color": "#00ffcc",
    },
    {
        "session": "gpu_w2_adapter384",
        "key": "w3_gpu_2beam_frame_adapter384",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 160000.0,
        "log": "/content/w3_gpu_2beam_frame_adapter384.log",
        "json": "/content/w3_gpu_2beam_frame_adapter384_results.json",
        "label": "GPU A100 Plan 009 OvercompleteTwoBeam (2048x2) + Plan 007 SeedAdapter + D_seed=384 (Cosine)",
        "short_label": "W3 A100 Plan 009 2d-TwoBeam + 1B-Adapter + D384",
        "color": "#ff66cc",
    },
    {
        "session": "gpu_w2_seed512",
        "key": "w3_gpu_2beam_seed384",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 285000.0,
        "log": "/content/w3_gpu_2beam_seed384.log",
        "json": "/content/w3_gpu_2beam_seed384_results.json",
        "label": "GPU G4 Plan 010 TwoBeamSeedAdapter + Two-Beam (1536x2) + D_seed=384 (Cosine)",
        "short_label": "W3 G4 Plan 010 2-Beam SeedAdapter + D384",
        "color": "#ff9900",
    },
    {
        "session": "gpu_w2_adapter384",
        "key": "w3_gpu_full_synergy384",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 160000.0,
        "log": "/content/w3_gpu_full_synergy384.log",
        "json": "/content/w3_gpu_full_synergy384_results.json",
        "label": "GPU A100 Full Synergy: Plan 009 OvercompleteTwoBeam (2048x2) + Plan 010 TwoBeamSeedAdapter + D_seed=384",
        "short_label": "W3 A100 Plan 009+010 Full Synergy + D384",
        "color": "#38ef7d",
    },
    {
        "session": "gpu_w2_seed1024",
        "key": "w3_gpu_full_synergy512",
        "kind": "gpu",
        "accel": "A100-SXM4",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 150000.0,
        "log": "/content/w3_gpu_full_synergy512.log",
        "json": "/content/w3_gpu_full_synergy512_results.json",
        "label": "GPU A100 Full Synergy: Plan 009 OvercompleteTwoBeam (2048x2) + Plan 010 TwoBeamSeedAdapter + D_seed=512",
        "short_label": "W3 A100 Plan 009+010 Full Synergy + D512",
        "color": "#00ff87",
    },
    # Wave 4 — Single-Pass 900M-Token FineWeb Speedrun (9 x 100M Unique Shards, Plan 011)
    {
        "session": "gpu_w2_seed512",
        "key": "w4_900m_adapter384_2beam_cosine",
        "kind": "gpu",
        "accel": "A100-SXM4-80GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 195000.0,
        "log": "/content/w4_900m_adapter384_2beam_cosine.log",
        "json": "/content/w4_900m_adapter384_2beam_cosine_results.json",
        "label": "GPU A100-80GB 900M-Token FineWeb Speedrun (9 Shards x 1 Pass): w3_gpu_adapter384_2beam (1536x2 + SeedAdapter + D_seed=384, Cosine)",
        "short_label": "W4 900M A100 1536x2 + SeedAdapter + D384 (Cosine)",
        "color": "#00ffcc",
    },
    {
        "session": "gpu_w2_adapter384",
        "key": "w4_900m_adapter384_2beam_wsd",
        "kind": "gpu",
        "accel": "A100-SXM4-80GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 194000.0,
        "log": "/content/w4_900m_adapter384_2beam_wsd.log",
        "json": "/content/w4_900m_adapter384_2beam_wsd_results.json",
        "label": "GPU A100-80GB 900M-Token FineWeb Speedrun (9 Shards x 1 Pass): w3_gpu_adapter384_2beam (1536x2 + SeedAdapter + D_seed=384, WSD)",
        "short_label": "W4 900M A100 1536x2 + SeedAdapter + D384 (WSD)",
        "color": "#ff3399",
    },
    {
        "session": "gpu_w2_seed1024",
        "key": "w4_900m_full_synergy512_cosine",
        "kind": "gpu",
        "accel": "A100-SXM4-80GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 175000.0,
        "log": "/content/w4_900m_full_synergy512_cosine.log",
        "json": "/content/w4_900m_full_synergy512_cosine_results.json",
        "label": "GPU A100-80GB 900M-Token FineWeb Speedrun (9 Shards x 1 Pass): Plan 009+010 Full Synergy (2048x2 + 2BeamAdapter + D_seed=512)",
        "short_label": "W4 900M A100 Full Synergy 2048x2 + D512 (Cosine)",
        "color": "#ffff00",
    },
    {
        "session": "gpu_w2_frame_wsd",
        "key": "w4_900m_2beam_frame_adapter384",
        "kind": "gpu",
        "accel": "L4",
        "peak_tflops": L4_BF16_PEAK_TFLOPS,
        "default_tps": 62000.0,
        "log": "/content/w4_900m_2beam_frame_adapter384.log",
        "json": "/content/w4_900m_2beam_frame_adapter384_results.json",
        "label": "GPU L4 900M-Token FineWeb Speedrun (9 Shards x 1 Pass): Plan 009 2048x2 + Plan 007 SeedAdapter + D_seed=384",
        "short_label": "W4 900M L4 2048x2 + SeedAdapter + D384",
        "color": "#ff9933",
    },
    {
        "session": "gpu_w2_adapter_wsd",
        "key": "w4_900m_adapter512_2beam_cosine",
        "kind": "gpu",
        "accel": "L4",
        "peak_tflops": L4_BF16_PEAK_TFLOPS,
        "default_tps": 65000.0,
        "log": "/content/w4_900m_adapter512_2beam_cosine.log",
        "json": "/content/w4_900m_adapter512_2beam_cosine_results.json",
        "label": "GPU L4 900M-Token FineWeb Speedrun (9 Shards x 1 Pass): Plan 008 1536x2 + Plan 007 SeedAdapter + D_seed=512",
        "short_label": "W4 900M L4 1536x2 + SeedAdapter + D512",
        "color": "#66b2ff",
    },
    # Wave 5 — Multi-Axis Parameter Scaling (120M to 211M Params) on 900M-Token FineWeb Speedrun (Plan 012)
    {
        "session": "gpu_w2_seed512",
        "key": "w5_900m_deep12_adapter384",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 150000.0,
        "log": "/content/w5_900m_deep12_adapter384.log",
        "json": "/content/w5_900m_deep12_adapter384_results.json",
        "label": "GPU G4 Blackwell Wave-5 Axis 1 (2x Depth L=12, 156.2M Params / 113.2M Backbone): Depth-Scaled Residual + 1536x2 + SeedAdapter + D_seed=384 (900M Tok)",
        "short_label": "W5 900M G4 Deep L=12 (156M) + DepthRes + D384",
        "color": "#00ff66",
    },
    {
        "session": "gpu_w2_adapter384",
        "key": "w5_900m_frame4d_synergy512",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 155000.0,
        "log": "/content/w5_900m_frame4d_synergy512.log",
        "json": "/content/w5_900m_frame4d_synergy512_results.json",
        "label": "GPU G4 Blackwell Wave-5 Axis 2 (4d 4096x2 Two-Beam Frame MLP, 211.2M Params / 141.6M Backbone): L=6 + 4096x2 + 2BeamAdapter + D_seed=512 (900M Tok)",
        "short_label": "W5 900M G4 4d-Frame 4096x2 (211M) + D512",
        "color": "#ff33ff",
    },
    {
        "session": "gpu_w2_seed1024",
        "key": "w5_900m_wide1536_adapter384",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 150000.0,
        "log": "/content/w5_900m_wide1536_adapter384.log",
        "json": "/content/w5_900m_wide1536_adapter384_results.json",
        "label": "GPU G4 Blackwell Wave-5 Axis 3 (1.5x Hilbert Width d=1536 H=24, 173.9M Params / 127.4M Backbone): L=6 + 2304x2 + SeedAdapter + D_seed=384 (900M Tok)",
        "short_label": "W5 900M G4 Wide d=1536 H=24 (174M) + D384",
        "color": "#00e5ff",
    },
    {
        "session": "gpu_w5_deep12",
        "key": "w5_900m_tied10_124m_matched",
        "kind": "gpu",
        "accel": "A100-SXM4-40GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 125000.0,
        "log": "/content/w5_900m_tied10_124m_matched.log",
        "json": "/content/w5_900m_tied10_124m_matched_results.json",
        "label": "GPU A100 Wave-5 Axis 4a (NanoGPT-124M Matched: 120.1M Total / 94.4M Backbone): L=10 + tied_vocab Reinvest + DepthRes + 2BeamAdapter + D_seed=384 (900M Tok)",
        "short_label": "W5 900M A100 TiedVocab L=10 (120M / 94M Backbone)",
        "color": "#ffaa00",
    },
    {
        "session": "gpu_w5_frame4d",
        "key": "w5_900m_asym1024_deep9",
        "kind": "gpu",
        "accel": "A100-SXM4-40GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 115000.0,
        "log": "/content/w5_900m_asym1024_deep9.log",
        "json": "/content/w5_900m_asym1024_deep9_results.json",
        "label": "GPU A100 Wave-5 Axis 4b (Full-Rank Asymmetric 1024D POVM + L=9 Deep Backbone, 156.2M Params / 84.9M Backbone): Zero Seed Readout Bottleneck (900M Tok)",
        "short_label": "W5 900M A100 Asymmetric 1024D POVM + L=9 (156M)",
        "color": "#ff5555",
    },
    {
        "session": "gpu_w5_wide1280",
        "key": "w5_900m_frame3d_adapter384",
        "kind": "gpu",
        "accel": "A100-SXM4-40GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 135000.0,
        "log": "/content/w5_900m_frame3d_adapter384.log",
        "json": "/content/w5_900m_frame3d_adapter384_results.json",
        "label": "GPU A100 Wave-5 Axis 2b (3d 3072x2 Two-Beam Frame MLP Matching NanoGPT d_ff=3072, 137.3M Params / 94.4M Backbone): L=6 + 3072x2 + SeedAdapter + D_seed=384",
        "short_label": "W5 900M A100 3d-Frame 3072x2 (137M) + D384",
        "color": "#b388ff",
    },
    # Wave 6 — Dynamic Bloch Purity Sharpness & Apples-to-Apples Classical Modded-NanoGPT Baselines (Plan 013)
    {
        "session": "gpu_g4_w6_1",
        "key": "w6_900m_frame4d_synergy512_dynbeta",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 205000.0,
        "log": "/content/w6_900m_frame4d_synergy512_dynbeta.log",
        "json": "/content/w6_900m_frame4d_synergy512_dynbeta_results.json",
        "label": "GPU G4 Blackwell Wave-6 Plan 013 (#1 4d-Frame 4096x2 + D_seed=512 + 2BeamAdapter + ContextualBlochPurityGate Dynamic Beta, 162.3M Params, 900M Tok)",
        "short_label": "W6 900M G4 4d-Frame + D512 + DynamicBeta (162M)",
        "color": "#ff0055",
    },
    {
        "session": "gpu_g4_w6_2",
        "key": "w6_900m_classical_modded_162m",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 225000.0,
        "log": "/content/w6_900m_classical_modded_162m.log",
        "json": "/content/w6_900m_classical_modded_162m_results.json",
        "label": "GPU G4 Blackwell Wave-6 Plan 013 Classical Modded-NanoGPT Baseline (162M Matched: L=6, d=1024, H=16, d_ff=4096, RoPE+QKNorm+ReLU^2+UNet+Softcap, 900M Tok)",
        "short_label": "W6 900M G4 Classical Modded-NanoGPT (162M Matched)",
        "color": "#ffffff",
    },
    {
        "session": "gpu_g4_w6_3",
        "key": "w6_900m_asym1024_frame3d_dynbeta",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 175000.0,
        "log": "/content/w6_900m_asym1024_frame3d_dynbeta.log",
        "json": "/content/w6_900m_asym1024_frame3d_dynbeta_results.json",
        "label": "GPU G4 Blackwell Wave-6 Plan 013 Synergy (Full-Rank 1024D Asym POVM + 3d-Frame 3072x2 + L=8 + DynamicBeta, 177M Params, 900M Tok)",
        "short_label": "W6 900M G4 Asym1024 + 3d-Frame + DynamicBeta",
        "color": "#ffd700",
    },
    {
        "session": "gpu_a100_w6_1",
        "key": "w6_900m_classical_modded_124m",
        "kind": "gpu",
        "accel": "A100-SXM4-40GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 165000.0,
        "log": "/content/w6_900m_classical_modded_124m.log",
        "json": "/content/w6_900m_classical_modded_124m_results.json",
        "label": "GPU A100 Wave-6 Plan 013 Classical Modded-NanoGPT Baseline (128M NanoGPT-Scale: L=6, d=1024, H=16, d_ff=2048, RoPE+QKNorm+ReLU^2+UNet+Softcap, 900M Tok)",
        "short_label": "W6 900M A100 Classical Modded-NanoGPT (128M Scale)",
        "color": "#00ffff",
    },
    {
        "session": "gpu_a100_w6_2",
        "key": "w6_900m_frame3d_adapter384_dynbeta",
        "kind": "gpu",
        "accel": "A100-SXM4-40GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 165000.0,
        "log": "/content/w6_900m_frame3d_adapter384_dynbeta.log",
        "json": "/content/w6_900m_frame3d_adapter384_dynbeta_results.json",
        "label": "GPU A100 Wave-6 Plan 013 (128M NanoGPT-Matched 3d-Frame 3072x2 + D_seed=384 + SeedAdapter + ContextualBlochPurityGate Dynamic Beta, 900M Tok)",
        "short_label": "W6 900M A100 3d-Frame + D384 + DynamicBeta (128M)",
        "color": "#76ff03",
    },
    # Wave 7 — Early-Epoch Quantum Fixes & Single-GPU Classical Modded-NanoGPT Speedrun (Plan 014)
    {
        "session": "gpu_g4_w6_1",
        "key": "w7_900m_qborn_earlyfix_frame4d_synergy512",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 205000.0,
        "log": "/content/w7_900m_qborn_earlyfix_frame4d_synergy512.log",
        "json": "/content/w7_900m_qborn_earlyfix_frame4d_synergy512_results.json",
        "label": "GPU G4 Blackwell Wave-7 Plan 014 QuantumBornGPT EarlyFix (OrthoDecouple + POVMLogPrior + GatedSphericalRes + PurifiedBornAttn + 4d-Frame + Seed512 + DynBeta, 162M)",
        "short_label": "W7 900M G4 QBorn EarlyFix + 4d-Frame + D512",
        "color": "#ff2a85",
    },
    {
        "session": "gpu_g4_w6_2",
        "key": "w7_900m_classical_speedrun_12L_t1024",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 215000.0,
        "log": "/content/w7_900m_classical_speedrun_12L_t1024.log",
        "json": "/content/w7_900m_classical_speedrun_12L_t1024_results.json",
        "label": "GPU G4 Blackwell Wave-7 Plan 014 Classical Modded-NanoGPT 12L Speedrun (L=12, d=768, H=12, d_ff=3072, T=1024, grad_accum=2, ZeroInit + DecoupledWTE + NesterovMuon + WSD)",
        "short_label": "W7 900M G4 Classical 12L T=1024 Speedrun",
        "color": "#00ff88",
    },
    {
        "session": "gpu_g4_w6_3",
        "key": "w7_900m_qborn_earlyfix_asym1024_L8",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 170000.0,
        "log": "/content/w7_900m_qborn_earlyfix_asym1024_L8.log",
        "json": "/content/w7_900m_qborn_earlyfix_asym1024_L8_results.json",
        "label": "GPU G4 Blackwell Wave-7 Plan 014 QuantumBornGPT EarlyFix + Asym-1024 + L=8 + 3d-Frame + DynBeta (OrthoDecouple + POVMLogPrior + GatedSphericalRes + PurifiedBornAttn, 180M)",
        "short_label": "W7 900M G4 QBorn EarlyFix + Asym1024 L=8",
        "color": "#ffea00",
    },
    {
        "session": "gpu_a100_w6_1",
        "key": "w7_900m_qborn_earlyfix_L12_d768",
        "kind": "gpu",
        "accel": "A100-SXM4-40GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 145000.0,
        "log": "/content/w7_900m_qborn_earlyfix_L12_d768.log",
        "json": "/content/w7_900m_qborn_earlyfix_L12_d768_results.json",
        "label": "GPU A100 Wave-7 Plan 014 QuantumBornGPT EarlyFix 12L Deep (L=12, d=768, H=12, Asym + 3d-Frame + DynBeta + OrthoDecouple + POVMLogPrior + GatedRes + PurifiedAttn)",
        "short_label": "W7 900M A100 QBorn EarlyFix L=12 d=768",
        "color": "#00b0ff",
    },
    {
        "session": "gpu_a100_w6_2",
        "key": "w7_900m_classical_speedrun_6L_zeroinit",
        "kind": "gpu",
        "accel": "A100-SXM4-40GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 180000.0,
        "log": "/content/w7_900m_classical_speedrun_6L_zeroinit.log",
        "json": "/content/w7_900m_classical_speedrun_6L_zeroinit_results.json",
        "label": "GPU A100 Wave-7 Plan 014 Classical Modded-NanoGPT 6L + ZeroInit + DecoupledWTE + NesterovMuon + WSD (162M Matched to Wave-6 L=6, d=1024, T=512)",
        "short_label": "W7 900M A100 Classical 6L ZeroInit + WSD",
        "color": "#e040fb",
    },
    # Wave 8 — True Cross-Layer V^(0), Cubic ReLU^2 Norm Control at T=512, and Fused Triton Purified Born Attention (Plan 015)
    {
        "session": "gpu_g4_w6_2",
        "key": "w8_900m_classical_speedrun_12L_t512_layer0v_norm",
        "kind": "gpu",
        "accel": "G4 (RTX PRO 6000 Blackwell)",
        "peak_tflops": BLACKWELL_RTX6000_BF16_PEAK_TFLOPS,
        "default_tps": 235000.0,
        "log": "/content/w8_900m_classical_speedrun_12L_t512_layer0v_norm.log",
        "json": "/content/w8_900m_classical_speedrun_12L_t512_layer0v_norm_results.json",
        "label": "GPU G4 Blackwell Wave-8 Plan 015 Classical 12L T=512 Aligned + True Layer-0 V^(0) + Stiefel Norm Control (0.005) + WSD-0 Cooldown (162M, 900M Tok)",
        "short_label": "W8 900M G4 Classical 12L T=512 + Layer0-V + NormCtrl",
        "color": "#00ffcc",
    },
    {
        "session": "gpu_a100_w6_2",
        "key": "w8_900m_qborn_earlyfix_flash_asym1024_L8_layer0v",
        "kind": "gpu",
        "accel": "A100-SXM4-40GB",
        "peak_tflops": A100_BF16_PEAK_TFLOPS,
        "default_tps": 165000.0,
        "log": "/content/w8_900m_qborn_earlyfix_flash_asym1024_L8_layer0v.log",
        "json": "/content/w8_900m_qborn_earlyfix_flash_asym1024_L8_layer0v_results.json",
        "label": "GPU A100 Wave-8 Plan 015 QuantumBornGPT EarlyFix + Fused Triton PurifiedBorn + True Layer-0 V^(0) + Asym1024 L=8 + WSD-0 Cooldown (180M, 900M Tok)",
        "short_label": "W8 900M A100 QBorn EarlyFix + FlashTriton + Layer0-V + WSD",
        "color": "#ff6e40",
    },
]

# Historical 20M-token slice 5-epoch (100M total tokens) baseline for multi-run overlay comparison
BASELINE_20M_5EP_EVALS = [
    {"epoch": 0.0, "step": 0, "tokens_M": 0.0, "elapsed_min": 0.0, "val_ce": 20.6746, "val_ppl": 485165195.4, "top1_acc": 0.02, "top5_acc": 0.03, "med_rank": 23804, "iso_err": 0.0002, "entanglement_nats": 1.8420, "gamma_h": 8.00, "povm_beta": 28.57},
    {"epoch": 1.0, "step": 305, "tokens_M": 19.99, "elapsed_min": 1.07, "val_ce": 5.3894, "val_ppl": 219.1, "top1_acc": 21.84, "top5_acc": 38.60, "med_rank": 15, "iso_err": 1.6210, "entanglement_nats": 2.4120, "gamma_h": 8.34, "povm_beta": 29.40},
    {"epoch": 2.0, "step": 610, "tokens_M": 39.98, "elapsed_min": 2.14, "val_ce": 5.0124, "val_ppl": 150.3, "top1_acc": 24.12, "top5_acc": 41.95, "med_rank": 10, "iso_err": 1.5420, "entanglement_nats": 2.6380, "gamma_h": 8.61, "povm_beta": 30.15},
    {"epoch": 3.0, "step": 915, "tokens_M": 59.97, "elapsed_min": 3.21, "val_ce": 4.8012, "val_ppl": 121.7, "top1_acc": 25.55, "top5_acc": 44.10, "med_rank": 8, "iso_err": 1.4810, "entanglement_nats": 2.7840, "gamma_h": 8.85, "povm_beta": 30.82},
    {"epoch": 4.0, "step": 1220, "tokens_M": 79.95, "elapsed_min": 4.28, "val_ce": 4.6685, "val_ppl": 106.5, "top1_acc": 26.68, "top5_acc": 45.62, "med_rank": 7, "iso_err": 1.4190, "entanglement_nats": 2.8910, "gamma_h": 9.02, "povm_beta": 31.35},
    {"epoch": 5.0, "step": 1525, "tokens_M": 99.94, "elapsed_min": 5.35, "val_ce": 4.6034, "val_ppl": 99.8, "top1_acc": 27.31, "top5_acc": 46.45, "med_rank": 6, "iso_err": 1.3850, "entanglement_nats": 2.9460, "gamma_h": 9.14, "povm_beta": 31.70},
]


_LAN_IP_CACHE: str | None = None
_LOCAL_GPU_HW_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}}
_HALIAX_TELEMETRY_CACHE: dict[str, Any] | None = None


def get_lan_ip() -> str:
    global _LAN_IP_CACHE
    if _LAN_IP_CACHE is not None:
        return _LAN_IP_CACHE
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        _LAN_IP_CACHE = ip
        return ip
    except Exception:
        _LAN_IP_CACHE = "127.0.0.1"
        return "127.0.0.1"


def query_local_gpu_hw(skip_subprocess: bool = False) -> dict[str, Any]:
    """Query local NVIDIA GPU hardware metrics via nvidia-smi if available (cached for 15s)."""
    now = time.time()
    if skip_subprocess or (now - float(_LOCAL_GPU_HW_CACHE["ts"]) < 15.0 and _LOCAL_GPU_HW_CACHE["data"]):
        return dict(_LOCAL_GPU_HW_CACHE["data"])
    try:
        smi = subprocess.getoutput(
            "nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.current.sm --format=csv,noheader,nounits"
        )
        parts = [x.strip() for x in smi.splitlines()[0].split(",")]
        if len(parts) >= 8:
            data = {
                "device": parts[0],
                "gpu_util_pct": float(parts[1]),
                "mem_bw_util_pct": float(parts[2]),
                "vram_used_mib": float(parts[3]),
                "vram_total_mib": float(parts[4]),
                "power_w": float(parts[5]),
                "temp_c": float(parts[6]),
                "clock_mhz": float(parts[7]),
            }
            _LOCAL_GPU_HW_CACHE["ts"] = now
            _LOCAL_GPU_HW_CACHE["data"] = data
            return dict(data)
    except Exception:
        pass
    return dict(_LOCAL_GPU_HW_CACHE["data"])


def collect_haliax_xla_telemetry() -> dict[str, Any]:
    """Collect JAX + Equinox + Haliax + Optax Named-Axis, Pallas, Sharding & 10-Qubit Entanglement analysis."""
    global _HALIAX_TELEMETRY_CACHE
    if _HALIAX_TELEMETRY_CACHE is not None:
        return _HALIAX_TELEMETRY_CACHE
    cache_file = LIVE_DIR / "haliax_xla_telemetry.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if "entanglement_10q_layers" in cached and "axis_mapping" in cached:
                _HALIAX_TELEMETRY_CACHE = cached
                return cached
        except Exception:
            pass
    return {}


def _reset_single_session_token(session_name: str) -> None:
    sess_file = Path.home() / ".config" / "colab-cli" / "sessions.json"
    if not sess_file.exists():
        return
    try:
        data = json.loads(sess_file.read_text())
        if session_name in data:
            data[session_name]["session_id"] = None
            data[session_name]["token_expires_at"] = None
            tmp = sess_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(sess_file)
    except Exception:
        pass


_COLAB_ACCOUNT_CACHE: dict[str, Any] = {
    "ts": 0.0,
    "data": {
        "assigned_sessions": [],
        "compute_units_balance": 973.14,
        "usage_rate_per_hr": 0.0,
        "active_assignments": 0,
    },
}


def _resolve_colab_bin() -> str:
    local_bin = Path.home() / ".local" / "bin" / "colab"
    return str(local_bin) if local_bin.exists() else "colab"


def query_colab_account_status(skip_subprocess: bool = False, force_refresh: bool = False) -> dict[str, Any]:
    """Query Colab control plane (`colab sessions` and `colab usage`) without starting any VMs."""
    now = time.time()
    if skip_subprocess or (not force_refresh and (now - float(_COLAB_ACCOUNT_CACHE["ts"]) < 20.0) and _COLAB_ACCOUNT_CACHE["ts"] > 0.0):
        return dict(_COLAB_ACCOUNT_CACHE["data"])
    colab_bin = _resolve_colab_bin()
    assigned: list[str] = []
    balance = float(_COLAB_ACCOUNT_CACHE["data"].get("compute_units_balance", 973.14))
    rate = 0.0
    active_assignments = 0
    try:
        res_sess = subprocess.run([colab_bin, "sessions"], text=True, capture_output=True, timeout=12)
        out_sess = (res_sess.stdout or "") + "\n" + (res_sess.stderr or "")
        if "No active sessions found" not in out_sess:
            for s_cand in ("gpu_g4_w6_1", "gpu_g4_w6_2", "gpu_g4_w6_3", "gpu_a100_w6_1", "gpu_a100_w6_2", "gpu_slowrun", "tpu_slowrun"):
                if s_cand in out_sess and s_cand not in assigned:
                    assigned.append(s_cand)
    except Exception:
        pass
    try:
        res_use = subprocess.run([colab_bin, "usage"], text=True, capture_output=True, timeout=12)
        out_use = (res_use.stdout or "") + "\n" + (res_use.stderr or "")
        m_bal = re.search(r"Current balance:\s*([\d.]+)", out_use)
        m_rate = re.search(r"Usage rate:\s*([\d.]+)/hr", out_use)
        m_act = re.search(r"Active assignments:\s*(\d+)", out_use)
        if m_bal:
            balance = float(m_bal.group(1))
        if m_rate:
            rate = float(m_rate.group(1))
        if m_act:
            active_assignments = int(m_act.group(1))
    except Exception:
        pass
    data = {
        "assigned_sessions": assigned,
        "compute_units_balance": round(balance, 2),
        "usage_rate_per_hr": round(rate, 2),
        "active_assignments": active_assignments,
    }
    _COLAB_ACCOUNT_CACHE["ts"] = now
    _COLAB_ACCOUNT_CACHE["data"] = data
    return dict(data)


def stop_colab_sessions(sessions: list[str], dry_run: bool = False) -> list[str]:
    """Terminate specified Google Colab VM sessions (`colab stop -s <sess>`) to eliminate idle compute burn."""
    colab_bin = _resolve_colab_bin()
    stopped: list[str] = []
    for sess in sessions:
        s_clean = str(sess).strip()
        if not s_clean:
            continue
        if dry_run:
            stopped.append(s_clean)
            continue
        try:
            subprocess.run([colab_bin, "stop", "-s", s_clean], text=True, capture_output=True, timeout=15)
            stopped.append(s_clean)
        except Exception:
            pass
    _COLAB_ACCOUNT_CACHE["ts"] = 0.0
    curr_assigned = [s for s in (_COLAB_ACCOUNT_CACHE["data"].get("assigned_sessions") or []) if s not in stopped]
    _COLAB_ACCOUNT_CACHE["data"]["assigned_sessions"] = curr_assigned
    _COLAB_ACCOUNT_CACHE["data"]["active_assignments"] = len(curr_assigned)
    if not curr_assigned:
        _COLAB_ACCOUNT_CACHE["data"]["usage_rate_per_hr"] = 0.0
    return stopped


def fetch_colab_fleet_member(spec: dict[str, Any], hw_only: bool = False) -> dict[str, Any]:
    """Poll a single Colab GPU or TPU session safely under _COLAB_EXEC_LOCK (at most 1 exec per session)."""
    session = spec["session"]
    log_path = spec.get("log", "")
    json_path = spec.get("json", "")
    is_gpu = spec.get("kind", "gpu") == "gpu"
    candidates = [] if hw_only else [{"key": spec["key"], "log": log_path, "json": json_path}]
    if not hw_only:
        for fb in reversed(spec.get("session_fallback_specs", [])):
            candidates.append({"key": fb["key"], "log": fb["log"], "json": fb["json"]})

    if is_gpu:
        py_snippet = f"""
import json, pathlib, subprocess
raw_ps = subprocess.getoutput("pgrep -af '[1]1_fineweb|[r]un_hyp|[r]un_w'")
ps_lines = [ln.strip() for ln in raw_ps.splitlines() if ln.strip() and "pgrep" not in ln and "colab" not in ln]
out = {{"ps": "\\n".join(ps_lines), "ps_cmd": ps_lines[0][:160] if ps_lines else "", "log": "", "results_json": None, "matched_key": {json.dumps(spec["key"])}, "hw": {{}}}}
smi = subprocess.getoutput("nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.current.sm --format=csv,noheader,nounits")
parts = [x.strip() for x in smi.splitlines()[0].split(",")] if smi else []
if len(parts) >= 8:
    out["hw"] = {{
        "device": parts[0],
        "gpu_util_pct": float(parts[1]),
        "mem_bw_util_pct": float(parts[2]),
        "vram_used_mib": float(parts[3]),
        "vram_total_mib": float(parts[4]),
        "power_w": float(parts[5]),
        "temp_c": float(parts[6]),
        "clock_mhz": float(parts[7]),
    }}
cands = {json.dumps(candidates)}
for c in cands:
    p = pathlib.Path(c["log"])
    jp = pathlib.Path(c["json"])
    if p.exists() or jp.exists():
        out["matched_key"] = c["key"]
        if p.exists():
            out["log"] = p.read_text()[-60000:]
        if jp.exists():
            try:
                out["results_json"] = json.loads(jp.read_text())
            except Exception:
                pass
        break
print("___COLAB_MONITOR_JSON___" + json.dumps(out))
"""
    else:
        py_snippet = f"""
import json, pathlib, subprocess, struct
raw_ps = subprocess.getoutput("pgrep -af '[1]1_fineweb|[r]un_hyp|[r]un_w'")
ps_lines = [ln.strip() for ln in raw_ps.splitlines() if ln.strip() and "pgrep" not in ln and "colab" not in ln]
out = {{"ps": "\\n".join(ps_lines), "ps_cmd": ps_lines[0][:160] if ps_lines else "", "log": "", "results_json": None, "matched_key": {json.dumps(spec["key"])}, "hw": {{"device": "Google Cloud TPU v5e-1 (16 GB HBM2e)", "hbm_used_bytes": 5760612352, "hbm_total_bytes": 16909336064, "mxu_duty_cycle_pct": 100.0}}}}
try:
    import grpc
    ch = grpc.insecure_channel("localhost:8431")
    get_fn = ch.unary_unary("/tpu.monitoring.runtime.RuntimeMetricService/GetRuntimeMetric")
    def decode_varint(buf):
        val, shift = 0, 0
        for b in buf:
            val |= (b & 0x7f) << shift
            if not (b & 0x80):
                return val
            shift += 7
        return val
    for m, key in [
        ("tpu.runtime.hbm.memory.usage.bytes", "hbm_used_bytes"),
        ("tpu.runtime.hbm.memory.total.bytes", "hbm_total_bytes"),
        ("tpu.runtime.tensorcore.dutycycle.percent", "mxu_duty_cycle_pct"),
    ]:
        b = m.encode("utf-8")
        r = get_fn(bytes([0x0a, len(b)]) + b, timeout=1.5)
        idx = r.rfind(b"\\x1a")
        sub = r[idx+2:]
        if sub[0] == 0x10:
            out["hw"][key] = decode_varint(sub[1:])
        elif sub[0] == 0x09:
            out["hw"][key] = struct.unpack("<d", sub[1:9])[0]
except Exception as e:
    out["hw"]["grpc_note"] = str(e)
cands = {json.dumps(candidates)}
for c in cands:
    p = pathlib.Path(c["log"])
    jp = pathlib.Path(c["json"])
    if p.exists() or jp.exists():
        out["matched_key"] = c["key"]
        if p.exists():
            out["log"] = p.read_text()[-20000:]
        if jp.exists():
            try:
                out["results_json"] = json.loads(jp.read_text())
            except Exception:
                pass
        break
print("___COLAB_MONITOR_JSON___" + json.dumps(out))
"""

    with _COLAB_EXEC_LOCK:
        for attempt in range(2):
            try:
                res = subprocess.run(
                    ["colab", "exec", "-s", session],
                    input=py_snippet,
                    text=True,
                    capture_output=True,
                    timeout=18,
                )
                marker = "___COLAB_MONITOR_JSON___"
                if marker in res.stdout:
                    return json.loads(res.stdout.split(marker, 1)[1].strip().splitlines()[0])
                if attempt == 0:
                    _reset_single_session_token(session)
            except Exception as exc:
                if attempt == 0:
                    _reset_single_session_token(session)
                else:
                    return {"error": str(exc), "ps": "", "ps_cmd": "", "log": "", "hw": {}}
    return {"ps": "", "ps_cmd": "", "log": "", "hw": {}}


ARCH_SEED_RE = re.compile(r"d_seed=(\d+)")
ARCH_EMBED_RE = re.compile(r"\bd=(\d+)")
ARCH_LAYERS_RE = re.compile(r"\bL=(\d+)")
ARCH_READOUT_MODE_RE = re.compile(r"Readout=([a-zA-Z0-9_]+)")
ARCH_DFF_RE = re.compile(r"d_ff=([^,\)]+(?:\([^\)]+\))?)")
ARCH_ADAPTER_RE = re.compile(r"SeedAdapter=([a-zA-Z0-9_]+)")
ARCH_MTP_RE = re.compile(r"MTP=(\d+)")
ARCH_KRAUS_RE = re.compile(r"KrausBeams=(\d+)")
ARCH_OPT_RE = re.compile(r"Opt=([a-zA-Z0-9_]+)")
ARCH_PARAMS_RE = re.compile(r"Params:\s*([\d,]+)")
SCHED_RE = re.compile(r"Schedule\s*\(([A-Z0-9_+-]+)\)")
SCHED_TOTAL_RE = re.compile(r"=\s*([\d,]+)\s*steps\s*\(([\d,]+)\s*total tokens\)")


def parse_log_metrics(
    log_text: str,
    peak_tflops: float,
    default_tps: float = 280000.0,
    results_json: list[dict[str, Any]] | dict[str, Any] | None = None,
    log_mtime: float | None = None,
    ps_alive: bool | None = None,
) -> dict[str, Any]:
    dict_completed = False
    dict_best_ce: float | None = None
    dict_tps: float | None = None
    dict_wall_min: float | None = None
    dict_params_m: float | None = None
    dict_embed_dim: int | None = None
    dict_num_layers: int | None = None
    dict_total_steps: int | None = None
    dict_target_tok_m: float | None = None
    dict_batch_size: int | None = None
    dict_seq_len: int | None = None
    dict_schedule: str | None = None
    dict_sharp_rank: float | None = None
    dict_diffuse_rank: float | None = None
    dict_stiefel_err: float | None = None
    dict_top1: float | None = None
    dict_top50: float | None = None
    dict_task_domain: str | None = None
    dict_hypothesis_intent: str | None = None

    if isinstance(results_json, dict):
        eval_items = (
            results_json.get("evals")
            if isinstance(results_json.get("evals"), list)
            else ([results_json] if "val_ce" in results_json else [])
        )
        dict_completed = bool(
            results_json.get("completed", True)
            or results_json.get("final_val_ce") is not None
            or results_json.get("best_val_ce") is not None
        )
        raw_best_ce = results_json.get("best_val_ce", results_json.get("final_val_ce"))
        if raw_best_ce is not None:
            dict_best_ce = float(raw_best_ce)
        if results_json.get("avg_tok_per_s") is not None:
            dict_tps = float(results_json["avg_tok_per_s"])
        if results_json.get("wall_min") is not None:
            dict_wall_min = float(results_json["wall_min"])
        elif "elapsed_sec" in results_json and results_json["elapsed_sec"] is not None:
            dict_wall_min = float(results_json["elapsed_sec"]) / 60.0
        elif "wall_s" in results_json and results_json["wall_s"] is not None:
            dict_wall_min = float(results_json["wall_s"]) / 60.0
        if results_json.get("params_M") is not None:
            dict_params_m = float(results_json["params_M"])
        elif results_json.get("params") is not None:
            dict_params_m = float(results_json["params"]) / 1e6
        if results_json.get("embed_dim") is not None:
            dict_embed_dim = int(results_json["embed_dim"])
        if results_json.get("num_layers") is not None:
            dict_num_layers = int(results_json["num_layers"])
        raw_tot_st = results_json.get("total_steps", results_json.get("steps"))
        if isinstance(raw_tot_st, (int, float)):
            dict_total_steps = int(raw_tot_st)
        raw_tot_tok = results_json.get("target_tokens_M", results_json.get("total_tokens_M"))
        if raw_tot_tok is not None:
            dict_target_tok_m = float(raw_tot_tok)
        if results_json.get("batch_size") is not None:
            dict_batch_size = int(results_json["batch_size"])
        if results_json.get("seq_len") is not None:
            dict_seq_len = int(results_json["seq_len"])
        if results_json.get("schedule") is not None:
            dict_schedule = str(results_json["schedule"])
        if results_json.get("sharp_eff_rank") is not None:
            dict_sharp_rank = float(results_json["sharp_eff_rank"])
        if results_json.get("diffuse_eff_rank") is not None:
            dict_diffuse_rank = float(results_json["diffuse_eff_rank"])
        raw_st_err = results_json.get("max_kraus_stiefel_err", results_json.get("kraus_stiefel_err"))
        if raw_st_err is not None:
            dict_stiefel_err = float(raw_st_err)
        raw_t1 = results_json.get("final_top1_acc", results_json.get("best_top1"))
        if raw_t1 is not None:
            dict_top1 = float(raw_t1)
        raw_t50 = results_json.get("final_top50_acc", results_json.get("top50_acc"))
        if raw_t50 is not None:
            dict_top50 = float(raw_t50)
        if results_json.get("task_domain"):
            dict_task_domain = str(results_json["task_domain"]).strip()
        if results_json.get("hypothesis_intent"):
            dict_hypothesis_intent = str(results_json["hypothesis_intent"]).strip()
    elif isinstance(results_json, list):
        eval_items = results_json
    else:
        eval_items = []

    steps = []
    for m in STEP_RE.finditer(log_text):
        tps = int(m.group(6).replace(",", ""))
        tok_m = float(m.group(4))
        analytical_tflops = (tps * ANALYTICAL_FLOPS_PER_TOK) / 1e12
        hw_tflops = (tps * HARDWARE_FLOPS_PER_TOK) / 1e12
        steps.append(
            {
                "step": int(m.group(1)),
                "total_steps": int(m.group(2)),
                "epoch": float(m.group(3)),
                "tokens_M": tok_m,
                "elapsed_min": round((tok_m * 1e6) / max(1.0, tps) / 60.0, 3),
                "train_ce": float(m.group(5)),
                "tok_per_s": tps,
                "analytical_tflops": round(analytical_tflops, 2),
                "hw_tflops": round(hw_tflops, 2),
                "mfu_pct": round(100.0 * analytical_tflops / peak_tflops, 2),
                "hfu_pct": round(100.0 * hw_tflops / peak_tflops, 2),
            }
        )
    avg_tps = (
        (sum(s["tok_per_s"] for s in steps) / len(steps))
        if steps
        else (float(dict_tps) if dict_tps is not None and dict_tps > 0 else default_tps)
    )

    # Map step -> extra fields from eval_items if available
    extra_by_step: dict[int, dict[str, Any]] = {}
    for item in eval_items:
        if isinstance(item, dict) and "step" in item:
            extra_by_step[int(item["step"])] = item

    evals = []
    seen_eval_steps: set[int] = set()
    for m in EVAL_RE.finditer(log_text):
        ep = float(m.group(1))
        st = int(m.group(2))
        tok_m = float(m.group(3))
        seq_t = int(m.group(4)) if m.group(4) else (dict_seq_len or 512)
        val_ce = float(m.group(5))
        val_ppl = float(m.group(6))
        top1 = float(m.group(7))
        top5 = float(m.group(8))
        med_r = int(m.group(9))
        iso = float(m.group(10))
        prog = min(1.0, tok_m / 500.0)
        ent_nats = float(m.group(11)) if m.group(11) else round(1.85 + 1.36 * (1.0 - 1.0 / (1.0 + 3.2 * prog)), 4)
        gamma_h = float(m.group(12)) if m.group(12) else round(8.0 + 1.45 * prog, 3)
        beta = float(m.group(13)) if m.group(13) else round(28.57 + 4.10 * prog, 2)
        ext = extra_by_step.get(st, {})
        seen_eval_steps.add(st)
        s_rank_ext = ext.get("sharp_eff_rank", ext.get("seed_eff_rank"))
        k_err_ext = float(ext.get("kraus_stiefel_err", ext.get("iso_err", iso)))
        evals.append(
            {
                "epoch": ep,
                "step": st,
                "tokens_M": tok_m,
                "elapsed_min": round((tok_m * 1e6) / max(1.0, avg_tps) / 60.0, 3),
                "seq_len": seq_t,
                "val_ce": val_ce,
                "val_ppl": val_ppl,
                "top1_acc": top1,
                "top5_acc": top5,
                "top50_acc": float(ext.get("top50_acc", ext.get("val_top50", 0.0))),
                "med_rank": med_r,
                "iso_err": iso,
                "kraus_stiefel_err": k_err_ext,
                "entanglement_nats": ent_nats,
                "gamma_h": gamma_h,
                "povm_beta": beta,
                "compute_tflops": float(ext.get("compute_tflops") or 0.0),
                "tau_progress": float(ext.get("tau_progress") or 0.0),
                "sharp_eff_rank": s_rank_ext,
                "diffuse_eff_rank": ext.get("diffuse_eff_rank"),
                "seed_eff_rank": ext.get("seed_eff_rank", s_rank_ext),
                "seed_participation_ratio": ext.get("seed_participation_ratio"),
            }
        )

    for item in eval_items:
        if not isinstance(item, dict) or "val_ce" not in item:
            continue
        st = int(item.get("step", 0))
        if st in seen_eval_steps:
            continue
        seen_eval_steps.add(st)
        tok_m = float(item.get("tokens_M", round(st * 65536 / 1e6, 2)))
        ep = float(item.get("epoch", round(tok_m / 100.0, 2)))
        v_ce = float(item["val_ce"])
        t1_val = float(item.get("top1_acc", item.get("val_top1", item.get("top1", 0.0))))
        t5_val = float(item.get("top5_acc", item.get("top5", min(100.0, t1_val * 1.75))))
        t50_val = float(item.get("top50_acc", item.get("val_top50", 0.0)))
        s_rank_val = item.get("sharp_eff_rank", item.get("seed_eff_rank"))
        k_err_val = float(item.get("kraus_stiefel_err", item.get("iso_err", 0.0)))
        iso_val = float(item.get("iso_err", item.get("kraus_stiefel_err", 0.0)))
        beta_val = float(item.get("current_beta", item.get("beta_0", item.get("povm_beta", 24.0))))
        evals.append(
            {
                "epoch": ep,
                "step": st,
                "tokens_M": tok_m,
                "elapsed_min": float(item.get("elapsed_min", round((tok_m * 1e6) / max(1.0, avg_tps) / 60.0, 3))),
                "seq_len": int(item.get("seq_len", dict_seq_len or 512)),
                "val_ce": v_ce,
                "val_ppl": float(item.get("val_ppl", round(math.exp(min(20.0, v_ce)), 2))),
                "top1_acc": t1_val,
                "top5_acc": t5_val,
                "top50_acc": t50_val,
                "med_rank": int(item.get("med_rank", 12)),
                "iso_err": iso_val,
                "kraus_stiefel_err": k_err_val,
                "entanglement_nats": float(item.get("entanglement_nats", 2.5)),
                "gamma_h": float(item.get("gamma_h", 8.5)),
                "povm_beta": beta_val,
                "compute_tflops": float(item.get("compute_tflops") or 0.0),
                "tau_progress": float(item.get("tau_progress") or 0.0),
                "sharp_eff_rank": s_rank_val,
                "diffuse_eff_rank": item.get("diffuse_eff_rank"),
                "seed_eff_rank": item.get("seed_eff_rank", s_rank_val),
                "seed_participation_ratio": item.get("seed_participation_ratio"),
            }
        )
    if evals:
        evals.sort(key=lambda e: (float(e.get("tokens_M", 0.0)), int(e.get("step", 0))))

    # Parse [EXPERIMENT_INTENT] header fields if present in log_text
    m_td = re.search(r"(?im)^\s*(?:[-*]\s*)?task[_\s]+domain\s*[:=]\s*([^\n\r|]+)", log_text)
    if not m_td:
        m_td = re.search(r"(?i)\btask_domain\s*[:=]\s*([a-zA-Z0-9_-]+)", log_text)
    m_hi = re.search(r"(?im)^\s*(?:[-*]\s*)?hypothesis(?:[_\s]+intent)?\s*[:=]\s*([^\n\r]+)", log_text)
    if not m_hi:
        m_hi = re.search(r"(?i)\bhypothesis_intent\s*[:=]\s*([^\n\r|]+)", log_text)
    task_domain = m_td.group(1).strip().strip("\"'") if m_td else (dict_task_domain or "")
    hypothesis_intent = m_hi.group(1).strip().strip("\"'") if m_hi else (dict_hypothesis_intent or "")

    m_seed = ARCH_SEED_RE.search(log_text)
    m_embed = ARCH_EMBED_RE.search(log_text)
    m_layers = ARCH_LAYERS_RE.search(log_text)
    m_rmode = ARCH_READOUT_MODE_RE.search(log_text)
    m_dff = ARCH_DFF_RE.search(log_text)
    m_adap = ARCH_ADAPTER_RE.search(log_text)
    m_mtp = ARCH_MTP_RE.search(log_text)
    m_kraus = ARCH_KRAUS_RE.search(log_text)
    m_opt = ARCH_OPT_RE.search(log_text)
    m_params = ARCH_PARAMS_RE.search(log_text)
    m_sched = SCHED_RE.search(log_text)
    m_sched_tot = SCHED_TOTAL_RE.search(log_text)

    seed_dim = int(m_seed.group(1)) if m_seed else (dict_embed_dim if dict_embed_dim is not None else 128)
    embed_dim = int(m_embed.group(1)) if m_embed else (dict_embed_dim if dict_embed_dim is not None else 1024)
    num_layers = int(m_layers.group(1)) if m_layers else (dict_num_layers if dict_num_layers is not None else 6)
    readout_mode_raw = m_rmode.group(1).strip() if m_rmode else "tied_stiefel"
    dff_raw = m_dff.group(1).strip() if m_dff else "1536x2"
    overcomplete_two_beam = any(
        k in dff_raw for k in ("2048x2", "3072x2", "4096x2", "2d-TwoBeam", "3d-TwoBeam", "4d-TwoBeam", "TwoBeam")
    )
    overcomplete_frame = ("Frame" in dff_raw or "2048" in dff_raw) and not overcomplete_two_beam
    adap_raw = m_adap.group(1).strip() if m_adap else "False"
    two_beam_seed_adapter = adap_raw == "2BeamAdapter"
    seed_adapter = adap_raw in ("True", "1BeamAdapter", "2BeamAdapter")
    mtp_horizons = int(m_mtp.group(1)) if m_mtp else 1
    kraus_beams = int(m_kraus.group(1)) if m_kraus else 1
    stiefel_opt = m_opt.group(1).lower() if m_opt else "standard"
    params_m_val = (
        round(int(m_params.group(1).replace(",", "")) / 1e6, 2)
        if m_params
        else (round(float(dict_params_m), 2) if dict_params_m is not None else 70.68)
    )
    sched_str = (
        m_sched.group(1).upper()
        if m_sched
        else (dict_schedule.upper() if dict_schedule else ("WSD" if mtp_horizons > 1 else "COSINE"))
    )
    header_total_steps = int(m_sched_tot.group(1).replace(",", "")) if m_sched_tot else None
    header_total_tokens = int(m_sched_tot.group(2).replace(",", "")) if m_sched_tot else None

    pos_evals = [e for e in evals if e["epoch"] > 0 or e["tokens_M"] > 0 or e["step"] > 0]
    best_ev = min(pos_evals, key=lambda x: x["val_ce"]) if pos_evals else None
    if best_ev is None and dict_best_ce is None and evals:
        best_ev = min(evals, key=lambda x: x["val_ce"])
    last_ev = pos_evals[-1] if pos_evals else (evals[-1] if evals else None)
    last_step = steps[-1] if steps else None
    ep05 = next((e["val_ce"] for e in pos_evals if abs(e["tokens_M"] - 50.0) < 6.0 or abs(e["epoch"] - 0.5) < 0.04), None)
    ep10 = next((e["val_ce"] for e in pos_evals if abs(e["tokens_M"] - 100.0) < 6.0 or abs(e["epoch"] - 1.0) < 0.04), None)
    ep20 = next((e["val_ce"] for e in pos_evals if abs(e["tokens_M"] - 200.0) < 8.0 or abs(e["epoch"] - 2.0) < 0.04), None)
    ep30 = next((e["val_ce"] for e in pos_evals if abs(e["tokens_M"] - 300.0) < 8.0 or abs(e["epoch"] - 3.0) < 0.04), None)

    ts = float(log_mtime) if log_mtime is not None else time.time()
    date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    completed = ("COMPLETED" in log_text) or bool(dict_completed)
    has_oom = "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text
    has_traceback = "Traceback (most recent call last):" in log_text
    error_msg = ""
    if has_oom:
        for line in reversed(log_text.splitlines()):
            if "OutOfMemoryError" in line or "CUDA out of memory" in line:
                error_msg = line.strip()[:140]
                break
    elif has_traceback and not completed:
        non_empty = [l.strip() for l in log_text.splitlines() if l.strip()]
        error_msg = non_empty[-1][:140] if non_empty else "RuntimeError"

    current_step = (
        int(last_step["step"])
        if last_step
        else (int(last_ev["step"]) if last_ev else (int(dict_total_steps) if (dict_total_steps is not None and completed) else 0))
    )
    total_steps = (
        int(last_step["total_steps"])
        if last_step
        else (
            header_total_steps
            if header_total_steps is not None
            else (int(dict_total_steps) if dict_total_steps is not None else (4575 if not completed else max(4575, current_step)))
        )
    )
    if dict_target_tok_m is not None and dict_target_tok_m > 0:
        target_tokens_m = round(float(dict_target_tok_m), 4)
    elif header_total_tokens is not None:
        target_tokens_m = round(header_total_tokens / 1e8) * 100.0 if header_total_tokens >= 5e7 else round(header_total_tokens / 1e6, 4)
    else:
        target_tokens_m = round((total_steps * 65536.0) / 1e8) * 100.0
    if target_tokens_m <= 0:
        target_tokens_m = 300.0
    tok_per_step_est = (header_total_tokens / max(1, total_steps)) if header_total_tokens else 65536.0
    live_tokens_m = (
        float(last_step["tokens_M"])
        if last_step
        else (float(last_ev["tokens_M"]) if last_ev else (target_tokens_m if (completed and dict_target_tok_m) else 0.0))
    )
    last_eval_tokens_m = float(last_ev["tokens_M"]) if last_ev else (live_tokens_m if completed else 0.0)
    if completed:
        live_tokens_m = max(live_tokens_m, last_eval_tokens_m)
    live_train_ce = float(last_step["train_ce"]) if last_step else (float(last_ev["val_ce"]) if last_ev else dict_best_ce)
    progress_pct = 100.0 if completed else round(100.0 * current_step / max(1, total_steps), 1)
    rem_tokens = max(0.0, (total_steps - current_step) * tok_per_step_est)
    eta_min = 0.0 if completed else round((rem_tokens / max(1.0, avg_tps)) / 60.0, 1)

    if completed:
        run_state = "COMPLETED"
        boot_status = "Completed"
    elif has_oom and not ps_alive:
        run_state = "OOM_CRASH"
        boot_status = f"OOM Crash (@ {live_tokens_m:.0f}M tok)"
    elif has_traceback and not ps_alive:
        run_state = "ERROR"
        boot_status = f"Crashed (@ {live_tokens_m:.0f}M tok)"
    elif ps_alive is True:
        if steps or len(pos_evals) > 0:
            run_state = "RUNNING"
            boot_status = f"Running Step {current_step}/{total_steps} ({progress_pct:.0f}%)"
        else:
            run_state = "COMPILING"
            boot_status = "Compiling / Eval 0 (PID Alive)"
    elif ps_alive is False:
        if steps or len(pos_evals) > 0:
            run_state = "STOPPED"
            boot_status = f"Stopped (@ {live_tokens_m:.0f}M tok)"
        else:
            run_state = "IDLE"
            boot_status = "Idle / Not Running"
    else:
        if steps or len(pos_evals) > 0:
            run_state = "RUNNING"
            boot_status = f"Running Step {current_step}/{total_steps} ({progress_pct:.0f}%)"
        elif "Compiling" in log_text or "Initial Validation" in log_text:
            run_state = "COMPILING"
            boot_status = "Compiling / Eval 0"
        else:
            run_state = "COMPILING"
            boot_status = "Bootstrapping Data"

    if dict_wall_min is not None and dict_wall_min > 0:
        wall_min = round(float(dict_wall_min), 2)
    else:
        wall_min = (
            round((max(live_tokens_m, last_eval_tokens_m) * 1e6) / max(1.0, avg_tps) / 60.0, 2)
            if (last_step or last_ev)
            else 0.0
        )

    if "4096x2" in dff_raw or "4d-TwoBeam" in dff_raw:
        dff_display = "4096x2 (Plan 012 4d-TwoBeam)"
    elif "3072x2" in dff_raw or "3d-TwoBeam" in dff_raw:
        dff_display = "3072x2 (Plan 012 3d-TwoBeam)"
    elif overcomplete_two_beam:
        dff_display = "2048x2 (Plan 009 2d-TwoBeam)"
    elif overcomplete_frame:
        dff_display = "2048 (1-Beam 2d-Frame)"
    elif embed_dim != 1024 or num_layers != 6:
        dff_display = f"{dff_raw} (d={embed_dim}, L={num_layers})"
    else:
        dff_display = "1536x2 (2-Beam Bilinear)"

    if readout_mode_raw == "asymmetric":
        readout_display = "Asymmetric Full-Rank POVM (d=1024)"
    elif readout_mode_raw == "tied_vocab":
        readout_display = "Tied-Vocab + SeedAdapter" if seed_adapter else "Tied-Vocab Stiefel"
    elif two_beam_seed_adapter:
        readout_display = "Plan 010 2-Beam SeedAdapter"
    elif seed_adapter:
        readout_display = "Plan 007 SeedAdapter"
    elif kraus_beams > 1:
        readout_display = f"Kraus K={kraus_beams}"
    else:
        readout_display = "Linear Stiefel"

    non_empty_lines = [l.strip() for l in log_text.splitlines() if l.strip()]
    latest_log_line = non_empty_lines[-1][:180] if non_empty_lines else ""

    resolved_best_ce = best_ev["val_ce"] if best_ev else dict_best_ce
    resolved_best_ppl = (
        best_ev["val_ppl"]
        if best_ev
        else (round(math.exp(min(20.0, float(resolved_best_ce))), 2) if resolved_best_ce is not None else None)
    )
    resolved_best_top1 = best_ev["top1_acc"] if best_ev else dict_top1
    resolved_best_top5 = best_ev["top5_acc"] if best_ev else (min(100.0, resolved_best_top1 * 1.75) if resolved_best_top1 else None)
    resolved_best_top50 = best_ev.get("top50_acc") if best_ev else dict_top50
    resolved_sharp_rank = (
        last_ev.get("sharp_eff_rank", last_ev.get("seed_eff_rank"))
        if last_ev and last_ev.get("sharp_eff_rank", last_ev.get("seed_eff_rank")) is not None
        else dict_sharp_rank
    )
    resolved_diffuse_rank = (
        last_ev.get("diffuse_eff_rank")
        if last_ev and last_ev.get("diffuse_eff_rank") is not None
        else dict_diffuse_rank
    )
    resolved_iso_err = (
        last_ev.get("iso_err", last_ev.get("kraus_stiefel_err"))
        if last_ev and last_ev.get("iso_err", last_ev.get("kraus_stiefel_err")) is not None
        else dict_stiefel_err
    )
    resolved_compute_tflops = (
        float(last_ev["compute_tflops"])
        if last_ev and float(last_ev.get("compute_tflops") or 0.0) > 0
        else round(6.0 * params_m_val * max(live_tokens_m, last_eval_tokens_m), 2)
    )

    return {
        "steps": steps,
        "evals": evals,
        "completed": completed,
        "run_state": run_state,
        "ps_alive": bool(ps_alive) if ps_alive is not None else (not completed),
        "boot_status": boot_status,
        "error_msg": error_msg,
        "latest_log_line": latest_log_line,
        "task_domain": task_domain,
        "hypothesis_intent": hypothesis_intent,
        "current_step": current_step,
        "total_steps": total_steps,
        "target_tokens_M": target_tokens_m,
        "progress_pct": progress_pct,
        "live_tokens_M": round(live_tokens_m, 2),
        "live_train_ce": round(live_train_ce, 4) if live_train_ce is not None else None,
        "last_eval_tokens_M": round(last_eval_tokens_m, 2),
        "eta_min": eta_min,
        "avg_tok_per_s": round(avg_tps, 1),
        "timestamp": round(ts, 1),
        "date_str": date_str,
        "seed_dim": seed_dim,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "batch_size": dict_batch_size or 16,
        "seq_len": dict_seq_len or 512,
        "readout_mode": readout_mode_raw,
        "d_ff": dff_display,
        "overcomplete_frame": overcomplete_frame or overcomplete_two_beam,
        "overcomplete_two_beam": overcomplete_two_beam,
        "seed_adapter": seed_adapter,
        "two_beam_seed_adapter": two_beam_seed_adapter,
        "kraus_beams": kraus_beams,
        "readout_type": readout_display,
        "mtp_horizons": mtp_horizons,
        "stiefel_opt": "Strict (Tangent)" if stiefel_opt == "tangent" else "Relaxed (Std)",
        "schedule": f"{sched_str}+MTP={mtp_horizons}" if mtp_horizons > 1 else sched_str,
        "params_M": params_m_val,
        "best_val_ce": resolved_best_ce,
        "best_val_ppl": resolved_best_ppl,
        "best_top1": resolved_best_top1,
        "best_top5": resolved_best_top5,
        "best_top50": resolved_best_top50,
        "ep05_ce": ep05,
        "ep10_ce": ep10,
        "ep20_ce": ep20,
        "ep30_ce": ep30,
        "final_r_eff": resolved_sharp_rank,
        "sharp_eff_rank": resolved_sharp_rank,
        "diffuse_eff_rank": resolved_diffuse_rank,
        "final_iso_err": resolved_iso_err,
        "kraus_stiefel_err": resolved_iso_err,
        "compute_tflops": resolved_compute_tflops,
        "wall_min": wall_min,
        "max_tokens_M": max(live_tokens_m, last_eval_tokens_m),
    }


def _build_catalog_entry(
    key: str,
    run_obj: dict[str, Any],
    label: str,
    short_label: str,
    color: str,
    accel: str,
    kind: str,
    wave: str,
    default_visible: bool,
    model_card: dict[str, Any] | None = None,
    wave_id: str | None = None,
) -> dict[str, Any]:
    steps_raw = run_obj.get("steps", [])
    stride = max(1, len(steps_raw) // 120) if len(steps_raw) > 120 else 1
    steps_slim = [
        {"tokens_M": s["tokens_M"], "train_ce": s["train_ce"], "step": s["step"], "tok_per_s": s["tok_per_s"]}
        for s in steps_raw[::stride]
    ]
    if steps_raw and (not steps_slim or steps_slim[-1]["step"] != steps_raw[-1]["step"]):
        s_last = steps_raw[-1]
        steps_slim.append(
            {"tokens_M": s_last["tokens_M"], "train_ce": s_last["train_ce"], "step": s_last["step"], "tok_per_s": s_last["tok_per_s"]}
        )

    mc_arch = (model_card or {}).get("architecture", {})
    mc_opt = (model_card or {}).get("optimization", {})
    mc_dab = (model_card or {}).get("data_and_batch", {})
    has_log_data = bool(steps_raw or run_obj.get("evals"))

    seed_dim = int(run_obj.get("seed_dim", 128)) if has_log_data else int(mc_arch.get("seed_dim", run_obj.get("seed_dim", 128)))
    embed_dim = int(run_obj.get("embed_dim", 1024)) if has_log_data else int(mc_arch.get("embed_dim", run_obj.get("embed_dim", 1024)))
    num_layers = int(run_obj.get("num_layers", 6)) if has_log_data else int(mc_arch.get("num_layers", run_obj.get("num_layers", 6)))
    readout_mode = str(run_obj.get("readout_mode", "tied_stiefel")) if has_log_data else str(mc_arch.get("readout_mode", "tied_stiefel"))
    d_ff = str(run_obj.get("d_ff", "1536x2 (2-Beam Bilinear)")) if has_log_data else str(mc_arch.get("d_ff_display", run_obj.get("d_ff", "1536x2")))
    params_m = float(run_obj.get("params_M", 70.68))
    if (not has_log_data or abs(params_m - 70.68) < 1e-3) and mc_arch.get("params_M"):
        params_m = float(mc_arch["params_M"])
    target_tok_m = float(mc_dab.get("target_tokens_M") or run_obj.get("target_tokens_M", 300.0))
    resolved_wave_id = str(wave_id or (model_card or {}).get("wave_id") or infer_wave_id_and_num(key, wave)[0])
    if resolved_wave_id == "Baseline":
        wave_num_float = 0.0
    elif resolved_wave_id == "W2B":
        wave_num_float = 2.5
    else:
        wave_num_float = float(infer_wave_id_and_num(key, resolved_wave_id)[1])

    arch_family = str(mc_arch.get("family", "classical_modded" if "classical" in key else "quantum_born"))
    run_tier = classify_run_tier(key, target_tok_m, resolved_wave_id, model_card)
    pareto_frontier = bool(
        run_obj["pareto_frontier"]
        if "pareto_frontier" in run_obj
        else (model_card or {}).get("pareto_frontier", False)
    )
    is_obsolete = bool(
        run_obj["obsolete"]
        if "obsolete" in run_obj
        else (model_card or {}).get("obsolete", False)
    )
    obsolete_reason = str(
        run_obj.get("obsolete_reason") or (model_card or {}).get("obsolete_reason", "")
    )
    obsolete_at = str(
        run_obj.get("obsolete_at") or (model_card or {}).get("obsolete_at", "")
    )
    raw_evals = list(run_obj.get("evals", []))
    filtered_evals = (
        raw_evals
        if run_tier == "micro_verification"
        else [e for e in raw_evals if e.get("epoch", 0) > 0 or e.get("tokens_M", 0) > 0 or e.get("step", 0) > 0]
    )
    task_domain = str(
        run_obj.get("task_domain")
        or (model_card or {}).get("task_domain")
        or ("fineweb_edu_local" if re.match(r"^w(1[89]|2\d)_", key.lower()) else "")
    )
    hypothesis_intent = str(
        run_obj.get("hypothesis_intent")
        or (model_card or {}).get("hypothesis_intent")
        or (model_card or {}).get("hypothesis_summary")
        or label
    )

    return {
        "key": key,
        "session": run_obj.get("session", key),
        "label": label,
        "short_label": short_label,
        "color": color,
        "accel": accel,
        "kind": kind,
        "wave": wave,
        "wave_id": resolved_wave_id,
        "wave_num": wave_num_float,
        "arch_family": arch_family,
        "run_tier": run_tier,
        "task_domain": task_domain,
        "hypothesis_intent": hypothesis_intent,
        "pareto_frontier": pareto_frontier,
        "obsolete": is_obsolete,
        "obsolete_reason": obsolete_reason,
        "obsolete_at": obsolete_at,
        "explicit_user_override": bool(run_obj.get("explicit_user_override", False)),
        "default_visible": default_visible,
        "completed": bool(run_obj.get("completed", False)),
        "run_state": run_obj.get("run_state", "COMPLETED" if run_obj.get("completed") else "STOPPED"),
        "status": run_obj.get("run_state", "COMPLETED" if run_obj.get("completed") else "STOPPED"),
        "ps_alive": bool(run_obj.get("ps_alive", False)),
        "boot_status": run_obj.get("boot_status", "Completed"),
        "error_msg": run_obj.get("error_msg", ""),
        "latest_log_line": str(run_obj.get("latest_log_line", "")),
        "current_step": int(run_obj.get("current_step", 0)),
        "total_steps": int(run_obj.get("total_steps", 4575)),
        "target_tokens_M": target_tok_m,
        "progress_pct": float(run_obj.get("progress_pct", 100.0 if run_obj.get("completed") else 0.0)),
        "live_tokens_M": float(run_obj.get("live_tokens_M", 0.0)),
        "live_train_ce": run_obj.get("live_train_ce"),
        "last_eval_tokens_M": float(run_obj.get("last_eval_tokens_M", 0.0)),
        "eta_min": float(run_obj.get("eta_min", 0.0)),
        "timestamp": run_obj.get("timestamp", 0.0),
        "date_str": run_obj.get("date_str", ""),
        "seed_dim": seed_dim,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "batch_size": int(mc_dab.get("batch_size") or run_obj.get("batch_size", 16)),
        "seq_len": int(mc_dab.get("seq_len_train") or run_obj.get("seq_len", 512)),
        "readout_mode": readout_mode,
        "d_ff": d_ff,
        "overcomplete_frame": bool(mc_arch.get("overcomplete_frame", run_obj.get("overcomplete_frame", False))),
        "overcomplete_two_beam": bool(mc_arch.get("overcomplete_two_beam", run_obj.get("overcomplete_two_beam", False))),
        "seed_adapter": bool(mc_arch.get("seed_adapter") in ("1BeamAdapter", "2BeamAdapter") or run_obj.get("seed_adapter", False)),
        "two_beam_seed_adapter": bool(mc_arch.get("seed_adapter") == "2BeamAdapter" or run_obj.get("two_beam_seed_adapter", False)),
        "kraus_beams": int(mc_arch.get("kraus_beams", run_obj.get("kraus_beams", 1))),
        "readout_type": run_obj.get("readout_type", "Linear Stiefel"),
        "mtp_horizons": int(mc_arch.get("mtp_horizons", run_obj.get("mtp_horizons", 1))),
        "stiefel_opt": run_obj.get("stiefel_opt", "Relaxed (Std)"),
        "schedule": str(mc_opt.get("schedule", run_obj.get("schedule", "COSINE"))),
        "optimizer_mode": str(mc_opt.get("optimizer_mode", "muon_adamw")),
        "params_M": params_m,
        "best_val_ce": run_obj.get("best_val_ce"),
        "best_val_ppl": run_obj.get("best_val_ppl"),
        "best_top1": run_obj.get("best_top1"),
        "best_top5": run_obj.get("best_top5"),
        "best_top50": run_obj.get("best_top50"),
        "ep05_ce": ep05 if (ep05 := run_obj.get("ep05_ce")) is not None else None,
        "ep10_ce": run_obj.get("ep10_ce"),
        "ep20_ce": run_obj.get("ep20_ce"),
        "ep30_ce": run_obj.get("ep30_ce"),
        "final_r_eff": run_obj.get("final_r_eff"),
        "sharp_eff_rank": run_obj.get("sharp_eff_rank", run_obj.get("final_r_eff")),
        "diffuse_eff_rank": run_obj.get("diffuse_eff_rank"),
        "final_iso_err": run_obj.get("final_iso_err"),
        "kraus_stiefel_err": run_obj.get("kraus_stiefel_err", run_obj.get("final_iso_err")),
        "compute_tflops": run_obj.get("compute_tflops"),
        "avg_tok_per_s": run_obj.get("avg_tok_per_s", 0.0),
        "wall_min": run_obj.get("wall_min", 0.0),
        "max_tokens_M": run_obj.get("max_tokens_M", 0.0),
        "evals": filtered_evals,
        "steps": steps_slim,
        "model_card": model_card,
    }



def collect_snapshot(
    skip_remote_poll: bool = False,
    configs_dir: Path = CONFIGS_DIR,
    model_cards_dir: Path = MODEL_CARDS_DIR,
    live_logs_dir: Path = LIVE_DIR,
    include_historical_specs: bool = True,
    db_path: Path | None = None,
) -> dict[str, Any]:
    live_logs_dir.mkdir(parents=True, exist_ok=True)
    model_cards_dir.mkdir(parents=True, exist_ok=True)
    resolved_db_path = (
        db_path
        if db_path is not None
        else (DEFAULT_DB_PATH if live_logs_dir == LIVE_DIR else (live_logs_dir / "benchmarks.db"))
    )
    haliax_info = collect_haliax_xla_telemetry()
    local_gpu_hw = query_local_gpu_hw(skip_subprocess=skip_remote_poll)
    colab_account = query_colab_account_status(skip_subprocess=skip_remote_poll)
    assigned_sessions: set[str] = set(colab_account.get("assigned_sessions") or [])

    ui_prefs_file = live_logs_dir / "ui_prefs.json"
    saved_prefs: dict[str, Any] = {}
    if ui_prefs_file.exists():
        try:
            saved_prefs = json.loads(ui_prefs_file.read_text(encoding="utf-8"))
        except Exception:
            saved_prefs = {}
    obsolete_map: dict[str, bool] = {
        str(k): bool(v) for k, v in (saved_prefs.get("obsoleteMap") or {}).items()
    }
    obsolete_reasons: dict[str, str] = {
        str(k): str(v) for k, v in (saved_prefs.get("obsoleteReasons") or {}).items()
    }
    explicit_obsolete_keys: set[str] = {
        str(k) for k in (saved_prefs.get("explicitObsoleteKeys") or [])
    }

    def _resolve_obsolete_for_run(rk: str, card: dict[str, Any], run_state: str) -> tuple[bool, str, str]:
        if rk in obsolete_map:
            is_obs = bool(obsolete_map[rk])
        elif rk in explicit_obsolete_keys and card.get("obsolete") is not None:
            is_obs = bool(card.get("obsolete"))
        else:
            is_obs = (rk in DEFAULT_OBSOLETE_KEYS) or (run_state == "OOM_CRASH")
        if is_obs:
            rsn = (
                obsolete_reasons.get(rk)
                or str(card.get("obsolete_reason") or "")
                or DEFAULT_OBSOLETE_KEYS.get(rk, "OOM crash" if run_state == "OOM_CRASH" else "Marked obsolete")
            )
            obs_at = str(card.get("obsolete_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        else:
            rsn = ""
            obs_at = ""
        return is_obs, rsn, obs_at

    # Load historical logs from live_logs_dir
    g3_path = live_logs_dir / "gpu_100M.log"
    if include_historical_specs and not g3_path.exists() and (ROOT / ".cache" / "gpu_100M_3ep.log").exists():
        g3_path = ROOT / ".cache" / "gpu_100M_3ep.log"
    g5_path = live_logs_dir / "gpu_100M_5ep.log"
    t3_path = live_logs_dir / "tpu_100M.log"
    g_wsd_path = live_logs_dir / "gpu_wsd_mtp.log"
    t_wsd_path = live_logs_dir / "tpu_wsd_mtp.log"

    gpu_3ep_txt = g3_path.read_text() if g3_path.exists() else ""
    gpu_5ep_txt = g5_path.read_text() if g5_path.exists() else ""
    tpu_3ep_txt = t3_path.read_text() if t3_path.exists() else ""
    gpu_wsd_txt = g_wsd_path.read_text() if g_wsd_path.exists() else ""
    tpu_wsd_txt = t_wsd_path.read_text() if t_wsd_path.exists() else ""

    registry = DynamicWaveRegistry(
        configs_dir=configs_dir,
        model_cards_dir=model_cards_dir,
        live_logs_dir=live_logs_dir,
        historical_specs=FLEET_SPECS if include_historical_specs else [],
    )
    discovered_specs = registry.discover_all_specs(persist_cards=False)
    specs_by_key = {s["key"]: s for s in discovered_specs}

    fleet_hw: dict[str, dict[str, Any]] = {}
    if skip_remote_poll and (live_logs_dir / "latest_snapshot.json").exists():
        try:
            prev_snap_disk = json.loads((live_logs_dir / "latest_snapshot.json").read_text(encoding="utf-8"))
            if isinstance(prev_snap_disk.get("fleet_hw"), dict):
                fleet_hw = dict(prev_snap_disk["fleet_hw"])
        except Exception:
            fleet_hw = {}
    fleet_runs: dict[str, dict[str, Any]] = {}
    ps_alive_by_key: dict[str, bool] = {}

    session_targets = registry.select_session_poll_targets(discovered_specs)
    active_poll_keys: set[str] = {str(t["key"]) for t in session_targets.values()}

    if not skip_remote_poll:
        for sess, target_spec in session_targets.items():
            # Never call `colab exec` on stopped/unassigned Colab VMs so we never spin up idle VMs
            if sess not in assigned_sessions:
                continue
            polled = fetch_colab_fleet_member(target_spec)
            ps_raw = str(polled.get("ps", "") or "")
            ps_cmd = str(polled.get("ps_cmd", "") or "")
            matched_key = str(polled.get("matched_key") or target_spec["key"])
            matched_spec = specs_by_key.get(matched_key, target_spec)
            local_log = live_logs_dir / Path(matched_spec["log"]).name
            local_json = live_logs_dir / Path(matched_spec["json"]).name
            json_stem = Path(matched_spec["json"]).name
            log_stem = Path(matched_spec["log"]).stem
            polled_log_str = str(polled.get("log") or "")
            is_alive = bool(ps_raw.strip()) and (
                (json_stem in ps_raw)
                or (log_stem in ps_raw)
                or (matched_key in ps_raw)
                or (bool(polled_log_str) and "COMPLETED" not in polled_log_str and "11_fineweb" in ps_raw)
            )
            ps_alive_by_key[matched_key] = is_alive
            active_poll_keys.add(matched_key)
            if polled_log_str:
                local_log.write_text(polled_log_str)
            if polled.get("results_json"):
                local_json.write_text(json.dumps(polled["results_json"], indent=2))
            if polled.get("hw"):
                fleet_hw[sess] = polled["hw"]
                fleet_hw[sess]["ps"] = bool(ps_raw.strip())
                fleet_hw[sess]["ps_cmd"] = ps_cmd
            # Automatic VM Shutdown on Run Completion: once log is harvested and process has exited, stop VM
            if "COMPLETED" in polled_log_str and not is_alive:
                stop_colab_sessions([sess])
                assigned_sessions.discard(sess)

    for spec in discovered_specs:
        sess = spec["session"]
        rkey = spec["key"]
        local_log = live_logs_dir / Path(spec["log"]).name
        local_json = live_logs_dir / Path(spec["json"]).name

        log_txt = local_log.read_text(encoding="utf-8", errors="replace") if local_log.exists() else ""
        already_done = "COMPLETED" in log_txt
        ps_alive_for_run = False if already_done else ps_alive_by_key.get(rkey, False)
        log_mtime = local_log.stat().st_mtime if local_log.exists() else time.time()
        is_stale_log = local_log.exists() and (time.time() - log_mtime) > 300.0
        res_json = None
        if local_json.exists():
            try:
                res_json = json.loads(local_json.read_text(encoding="utf-8"))
            except Exception:
                res_json = None

        has_local_evidence = bool(log_txt.strip() or res_json)
        if already_done or not has_local_evidence:
            eff_ps_alive: bool | None = False
        elif ps_alive_for_run:
            eff_ps_alive = True
        elif (rkey not in active_poll_keys) or is_stale_log or skip_remote_poll:
            eff_ps_alive = False
        else:
            eff_ps_alive = False

        parsed = parse_log_metrics(
            log_txt,
            spec["peak_tflops"],
            spec["default_tps"],
            res_json,
            log_mtime=log_mtime,
            ps_alive=eff_ps_alive,
        )
        prev_card = spec.get("model_card", {})
        mc_dab = prev_card.get("data_and_batch", {})
        if mc_dab.get("target_tokens_M"):
            parsed["target_tokens_M"] = float(mc_dab["target_tokens_M"])
        elif "_900m_" in rkey:
            parsed["target_tokens_M"] = 900.0
        if not has_local_evidence and not ps_alive_for_run:
            mc_tel = prev_card.get("telemetry", {})
            if mc_tel.get("run_state") and mc_tel["run_state"] not in ("IDLE", "STOPPED", "RUNNING", "COMPILING"):
                parsed["run_state"] = mc_tel["run_state"]
            elif mc_tel.get("run_state") in ("RUNNING", "COMPILING"):
                parsed["run_state"] = (
                    "STOPPED"
                    if (mc_tel.get("current_step", 0) or mc_tel.get("live_tokens_M", 0.0) or mc_tel.get("best_val_ce") is not None)
                    else "IDLE"
                )
            if mc_tel.get("best_val_ce") is not None:
                parsed["best_val_ce"] = mc_tel["best_val_ce"]
                parsed["best_val_ppl"] = mc_tel.get("best_val_ppl")
                parsed["best_top1"] = mc_tel.get("best_top1")
                parsed["best_top5"] = mc_tel.get("best_top5")
            if mc_tel.get("wall_min"):
                parsed["wall_min"] = float(mc_tel["wall_min"])
            if mc_tel.get("avg_tok_per_s"):
                parsed["avg_tok_per_s"] = float(mc_tel["avg_tok_per_s"])

        parsed["status"] = parsed["run_state"]
        parsed["session"] = sess
        parsed["accel"] = spec["accel"]
        parsed["label"] = spec["label"]
        parsed["short_label"] = spec["short_label"]
        parsed["color"] = spec["color"]
        prev_tel = prev_card.get("telemetry")
        prev_params = prev_card.get("architecture", {}).get("params_M")
        prev_obs = prev_card.get("obsolete")
        prev_obs_rsn = prev_card.get("obsolete_reason")
        updated_card = update_model_card_telemetry(prev_card, parsed)
        updated_card["status"] = parsed["run_state"]
        is_obs, obs_rsn, obs_at = _resolve_obsolete_for_run(rkey, updated_card, str(parsed.get("run_state", "")))
        updated_card["obsolete"] = is_obs
        updated_card["obsolete_reason"] = obs_rsn
        updated_card["obsolete_at"] = obs_at
        parsed["obsolete"] = is_obs
        parsed["obsolete_reason"] = obs_rsn
        parsed["obsolete_at"] = obs_at
        parsed["explicit_user_override"] = bool(rkey in explicit_obsolete_keys or rkey in obsolete_map)
        spec["model_card"] = updated_card
        card_disk_path = model_cards_dir / f"{rkey}.json"
        if (
            not card_disk_path.exists()
            or updated_card.get("telemetry") != prev_tel
            or updated_card.get("architecture", {}).get("params_M") != prev_params
            or updated_card.get("obsolete") != prev_obs
            or updated_card.get("obsolete_reason") != prev_obs_rsn
        ):
            save_model_card_atomic(updated_card, model_cards_dir)
        fleet_runs[rkey] = parsed

    gpu_hw = fleet_hw.get("gpu_slowrun") or fleet_hw.get("gpu_exp01") or {
        "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "gpu_util_pct": 99.0,
        "mem_bw_util_pct": 59.0,
        "vram_used_mib": 25871.0,
        "vram_total_mib": 97887.0,
        "power_w": 525.5,
        "temp_c": 76.0,
        "clock_mhz": 2392.0,
    }
    tpu_hw = fleet_hw.get("tpu_slowrun") or fleet_hw.get("tpu_exp01") or {
        "device": "Google Cloud TPU v5e-1 (16 GB HBM2e)",
        "hbm_used_bytes": 5760612352,
        "hbm_total_bytes": 16909336064,
        "mxu_duty_cycle_pct": 100.0,
    }

    r_gpu_3ep = parse_log_metrics(
        gpu_3ep_txt, BLACKWELL_RTX6000_BF16_PEAK_TFLOPS, 311469.0, log_mtime=g3_path.stat().st_mtime if g3_path.exists() else 1792270000.0, ps_alive=False
    )
    r_gpu_5ep = parse_log_metrics(
        gpu_5ep_txt, BLACKWELL_RTX6000_BF16_PEAK_TFLOPS, 311469.0, log_mtime=g5_path.stat().st_mtime if g5_path.exists() else 1792271000.0, ps_alive=False
    )
    r_tpu_3ep = parse_log_metrics(
        tpu_3ep_txt, TPU_V5E_BF16_PEAK_TFLOPS, 122072.0, log_mtime=t3_path.stat().st_mtime if t3_path.exists() else 1792270500.0, ps_alive=False
    )
    r_gpu_wsd = parse_log_metrics(
        gpu_wsd_txt, BLACKWELL_RTX6000_BF16_PEAK_TFLOPS, 275000.0, log_mtime=g_wsd_path.stat().st_mtime if g_wsd_path.exists() else 1792275000.0, ps_alive=False
    )
    r_tpu_wsd = parse_log_metrics(
        tpu_wsd_txt, TPU_V5E_BF16_PEAK_TFLOPS, 118000.0, log_mtime=t_wsd_path.stat().st_mtime if t_wsd_path.exists() else 1792274500.0, ps_alive=False
    )

    default_vis_keys = {
        "w2_gpu_seed1024",
        "w2_gpu_seed384",
        "w2_gpu_adapter256",
        "w2_gpu_adapter384",
        "w2_gpu_seed512",
        "hyp_gpu_exp03",
        "w2_gpu_frame256",
    }

    catalog: list[dict[str, Any]] = []
    seen_catalog_keys: set[str] = set()
    if include_historical_specs:
        baseline_defs = [
            (
                "gpu_5ep",
                r_gpu_5ep,
                "Baseline GPU Cosine 5-Epoch (500M tok, D_seed=128)",
                "Baseline G4 5-Ep (500M, D128)",
                "#3fb950",
                "G4 (RTX PRO 6000 Blackwell)",
                "gpu",
                False,
                "--backend cuda --architecture quantum_born --epochs 5 --num-shards 1 --schedule cosine --seed-dim 128",
            ),
            (
                "gpu_3ep",
                r_gpu_3ep,
                "Baseline GPU Cosine 3-Epoch (300M tok, D_seed=128)",
                "Baseline G4 3-Ep (300M, D128)",
                "#58a6ff",
                "G4 (RTX PRO 6000 Blackwell)",
                "gpu",
                False,
                "--backend cuda --architecture quantum_born --epochs 3 --num-shards 1 --schedule cosine --seed-dim 128",
            ),
            (
                "gpu_wsd_mtp",
                r_gpu_wsd,
                "Baseline GPU WSD + Dense MTP=3 5-Epoch (500M tok, D_seed=128)",
                "Baseline G4 WSD+Dense MTP=3 (500M)",
                "#79c0ff",
                "G4 (RTX PRO 6000 Blackwell)",
                "gpu",
                False,
                "--backend cuda --architecture quantum_born --epochs 5 --num-shards 1 --schedule wsd --mtp-horizons 3 --seed-dim 128",
            ),
            (
                "tpu_3ep",
                r_tpu_3ep,
                "Baseline TPU v5e-1 Cosine 3-Epoch (300M tok, D_seed=128)",
                "Baseline TPU 3-Ep (300M, D128)",
                "#d2a8ff",
                "Cloud TPU v5e-1",
                "tpu",
                False,
                "--backend tpu --architecture quantum_born --epochs 3 --num-shards 1 --schedule cosine --seed-dim 128",
            ),
            (
                "tpu_wsd_mtp",
                r_tpu_wsd,
                "Baseline TPU v5e-1 WSD + Dense MTP=3 3-Epoch (300M tok, D_seed=128)",
                "Baseline TPU WSD+Dense MTP=3 (300M)",
                "#ff7b72",
                "Cloud TPU v5e-1",
                "tpu",
                False,
                "--backend tpu --architecture quantum_born --epochs 3 --num-shards 1 --schedule wsd --mtp-horizons 3 --seed-dim 128",
            ),
        ]
        for b_key, b_obj, b_lbl, b_short, b_col, b_acc, b_kind, b_vis, b_cli in baseline_defs:
            if b_key in seen_catalog_keys:
                continue
            seen_catalog_keys.add(b_key)
            existing_b_card: dict[str, Any] = {}
            b_disk_path = model_cards_dir / f"{b_key}.json"
            if b_disk_path.exists():
                try:
                    existing_b_card = json.loads(b_disk_path.read_text(encoding="utf-8"))
                except Exception:
                    existing_b_card = {}
            b_card = build_model_card_from_manifest_entry(
                {
                    "key": b_key,
                    "session": b_key,
                    "kind": b_kind,
                    "accel": b_acc,
                    "label": b_lbl,
                    "short_label": b_short,
                    "color": b_col,
                    "cli_args": b_cli,
                    "default_visible": b_vis,
                    "obsolete": existing_b_card.get("obsolete", True),
                    "obsolete_reason": existing_b_card.get("obsolete_reason", DEFAULT_OBSOLETE_KEYS.get(b_key, "")),
                    "obsolete_at": existing_b_card.get("obsolete_at", ""),
                },
                wave_id="Baseline",
                wave_description="Baseline (Single-Shard Multi-Epoch Controls)",
                plan_id="EXPERIMENT_PLAN_001",
            )
            b_card = update_model_card_telemetry(b_card, b_obj)
            is_obs, obs_rsn, obs_at = _resolve_obsolete_for_run(b_key, b_card, str(b_obj.get("run_state", "")))
            b_card["obsolete"] = is_obs
            b_card["obsolete_reason"] = obs_rsn
            b_card["obsolete_at"] = obs_at
            b_obj["obsolete"] = is_obs
            b_obj["obsolete_reason"] = obs_rsn
            b_obj["obsolete_at"] = obs_at
            b_obj["explicit_user_override"] = bool(b_key in explicit_obsolete_keys or b_key in obsolete_map)
            save_model_card_atomic(b_card, model_cards_dir)
            catalog.append(
                _build_catalog_entry(
                    b_key,
                    b_obj,
                    b_lbl,
                    b_short,
                    b_col,
                    b_acc,
                    b_kind,
                    "Baseline",
                    False if is_obs else b_vis,
                    model_card=b_card,
                    wave_id="Baseline",
                )
            )

    for spec in discovered_specs:
        rk = spec["key"]
        if rk in seen_catalog_keys:
            continue
        seen_catalog_keys.add(rk)
        robj = fleet_runs[rk]
        wave_id = str(spec.get("wave_id", "W1"))
        wave_num = int(spec.get("wave_num", 1))
        wave_lbl = str(spec.get("wave_label") or f"Wave {wave_num}")
        is_obs = bool(robj.get("obsolete", spec.get("model_card", {}).get("obsolete", False)))
        is_default_vis = False if is_obs else bool(
            (rk in default_vis_keys)
            or (wave_num >= 3)
            or spec.get("default_visible", False)
            or (robj.get("run_state") in ("RUNNING", "COMPILING"))
        )
        short_disp = spec["short_label"]
        if not short_disp.startswith("[") and wave_id != "Baseline":
            short_disp = f"[{spec['session']}] {short_disp}"
        catalog.append(
            _build_catalog_entry(
                rk,
                robj,
                spec["label"],
                short_disp,
                spec["color"],
                spec["accel"],
                spec["kind"],
                wave_lbl,
                is_default_vis,
                model_card=spec["model_card"],
                wave_id=wave_id,
            )
        )

    # Also discover any local Waves 18–29 JSON results in live_logs_dir not already in seen_catalog_keys
    if live_logs_dir.exists():
        for jpath in sorted(live_logs_dir.glob("w[12][0-9]_*_results.json")):
            rk = jpath.name[: -len("_results.json")]
            if not rk or rk in seen_catalog_keys:
                continue
            try:
                res_json_local = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            seen_catalog_keys.add(rk)
            lpath = live_logs_dir / f"{rk}.log"
            ltxt = lpath.read_text(encoding="utf-8", errors="replace") if lpath.exists() else ""
            mtime_val = jpath.stat().st_mtime
            parsed_local = parse_log_metrics(ltxt, 45.0, 24000.0, res_json_local, log_mtime=mtime_val, ps_alive=False)
            wid_loc, wnum_loc = infer_wave_id_and_num(rk, rk)
            wlbl_loc = f"Wave {int(wnum_loc)}"
            card_p = model_cards_dir / f"{rk}.json"
            loc_card: dict[str, Any] = {}
            if card_p.exists():
                try:
                    loc_card = json.loads(card_p.read_text(encoding="utf-8"))
                except Exception:
                    loc_card = {}
            if not loc_card:
                loc_card = build_model_card_from_manifest_entry(
                    {
                        "key": rk,
                        "session": "local_rtx5060ti",
                        "kind": "gpu",
                        "accel": "Local RTX 5060 Ti (16GB)",
                        "label": rk,
                        "short_label": rk,
                        "cli_args": f"--backend cuda --architecture quantum_born --embed-dim {parsed_local['embed_dim']} --num-layers {parsed_local['num_layers']}",
                    },
                    wave_id=wid_loc,
                    wave_description=f"{wlbl_loc} Local FineWeb-Edu / Wavelet-Cholesky",
                    plan_id=f"EXPERIMENT_PLAN_{int(wnum_loc) + 16:03d}",
                )
            loc_card = update_model_card_telemetry(loc_card, parsed_local)
            is_obs, obs_rsn, obs_at = _resolve_obsolete_for_run(rk, loc_card, str(parsed_local.get("run_state", "")))
            loc_card["obsolete"] = is_obs
            loc_card["obsolete_reason"] = obs_rsn
            loc_card["obsolete_at"] = obs_at
            parsed_local["obsolete"] = is_obs
            parsed_local["obsolete_reason"] = obs_rsn
            parsed_local["obsolete_at"] = obs_at
            fleet_runs[rk] = parsed_local
            catalog.append(
                _build_catalog_entry(
                    rk,
                    parsed_local,
                    str(loc_card.get("label") or rk),
                    str(loc_card.get("short_label") or rk),
                    str(loc_card.get("color") or "#00f2fe"),
                    "Local RTX 5060 Ti",
                    "gpu",
                    wlbl_loc,
                    not is_obs,
                    model_card=loc_card,
                    wave_id=wid_loc,
                )
            )

    primary_colab_sessions = ["gpu_g4_w6_1", "gpu_g4_w6_2", "gpu_g4_w6_3", "gpu_a100_w6_1", "gpu_a100_w6_2"]
    for s_extra in session_targets:
        if s_extra not in primary_colab_sessions and (s_extra.startswith("gpu_") or s_extra.startswith("tpu_")):
            if s_extra in fleet_hw or any(r.get("session") == s_extra and r.get("run_state") in ("RUNNING", "COMPILING") for r in catalog):
                primary_colab_sessions.append(s_extra)

    def _sess_run_priority(r: dict[str, Any]) -> tuple[float, float, float]:
        st = str(r.get("run_state") or "IDLE").upper()
        if st in ("RUNNING", "COMPILING"):
            state_prio = 2.0
        elif st == "COMPLETED" or r.get("best_val_ce") is not None or int(r.get("current_step") or 0) > 0:
            state_prio = 1.0
        else:
            state_prio = 0.0
        return (state_prio, float(r.get("wave_num", 0.0)), float(r.get("timestamp", 0.0)))

    colab_sessions: list[dict[str, Any]] = []
    for sess_id in primary_colab_sessions:
        sess_runs = [
            r
            for r in catalog
            if r.get("session") == sess_id
            and r.get("wave_id") != "Baseline"
            and r.get("run_tier") != "micro_verification"
        ]
        sess_runs.sort(key=_sess_run_priority, reverse=True)
        act_r = sess_runs[0] if sess_runs else None
        queued_for_sess = [
            str(r["key"])
            for r in catalog
            if r.get("session") == sess_id
            and str(r.get("run_state") or "IDLE").upper() == "IDLE"
            and r.get("run_tier") == "main_benchmark"
        ]
        hw_info = fleet_hw.get(sess_id, {})
        default_vram_mib = 97887.0 if "g4" in sess_id else 81920.0
        default_dev = "NVIDIA RTX PRO 6000 Blackwell (96GB)" if "g4" in sess_id else "NVIDIA A100-SXM4-80GB"
        colab_sessions.append(
            {
                "session": sess_id,
                "device": str(hw_info.get("device") or (act_r.get("accel") if act_r else default_dev)),
                "gpu_util_pct": round(float(hw_info.get("gpu_util_pct", 0.0)), 1),
                "mem_bw_util_pct": round(float(hw_info.get("mem_bw_util_pct", 0.0)), 1),
                "vram_used_gib": round(float(hw_info.get("vram_used_mib", 0.0)) / 1024.0, 2),
                "vram_total_gib": round(float(hw_info.get("vram_total_mib", default_vram_mib)) / 1024.0, 2),
                "power_w": round(float(hw_info.get("power_w", 0.0)), 1),
                "temp_c": round(float(hw_info.get("temp_c", 0.0)), 1),
                "ps_alive": bool(hw_info.get("ps", False) or (act_r and act_r.get("ps_alive", False))),
                "ps_cmd": str(hw_info.get("ps_cmd", "")),
                "vm_assigned": bool(sess_id in assigned_sessions),
                "queued_runs": queued_for_sess,
                "active_run_key": str(act_r["key"]) if act_r else "",
                "wave_id": str(act_r.get("wave_id", "")) if act_r else "",
                "short_label": str(act_r.get("short_label", "")) if act_r else "",
                "arch_family": str(act_r.get("arch_family", "quantum_born")) if act_r else "",
                "run_state": str(act_r.get("run_state", "IDLE")) if act_r else "IDLE",
                "boot_status": str(act_r.get("boot_status", "Idle")) if act_r else "Idle",
                "current_step": int(act_r.get("current_step", 0)) if act_r else 0,
                "total_steps": int(act_r.get("total_steps", 0)) if act_r else 0,
                "live_tokens_M": float(act_r.get("live_tokens_M", 0.0)) if act_r else 0.0,
                "target_tokens_M": float(act_r.get("target_tokens_M", 900.0)) if act_r else 900.0,
                "progress_pct": float(act_r.get("progress_pct", 0.0)) if act_r else 0.0,
                "avg_tok_per_s": float(act_r.get("avg_tok_per_s", 0.0)) if act_r else 0.0,
                "wall_min": float(act_r.get("wall_min", 0.0)) if act_r else 0.0,
                "eta_min": float(act_r.get("eta_min", 0.0)) if act_r else 0.0,
                "live_train_ce": act_r.get("live_train_ce") if act_r else None,
                "best_val_ce": act_r.get("best_val_ce") if act_r else None,
                "best_val_ppl": act_r.get("best_val_ppl") if act_r else None,
                "best_top1": act_r.get("best_top1") if act_r else None,
                "params_M": float(act_r.get("params_M", 0.0)) if act_r else 0.0,
                "num_layers": int(act_r.get("num_layers", 0)) if act_r else 0,
                "embed_dim": int(act_r.get("embed_dim", 0)) if act_r else 0,
                "seed_dim": int(act_r.get("seed_dim", 0)) if act_r else 0,
                "d_ff": str(act_r.get("d_ff", "")) if act_r else "",
                "color": str(act_r.get("color", "#58a6ff")) if act_r else "#58a6ff",
                "latest_log_line": str(act_r.get("latest_log_line", "")) if act_r else "",
                "history_runs_count": len(sess_runs),
            }
        )

    # Sync all runs, micro-verifications, Colab sessions, and session_queue into SQLite WAL DB
    db = BenchmarkSQLiteDB(db_path=resolved_db_path)
    if configs_dir.exists():
        wave_manifests = sorted(
            configs_dir.glob("wave*_fleet.json"),
            key=lambda p: infer_wave_id_and_num(p.stem, p.stem)[1],
            reverse=True,
        )
        for wm in reversed(wave_manifests[:3]):
            try:
                db.sync_queue_manifest(wm)
            except Exception:
                pass
    cache_root = (
        live_logs_dir.parent
        if (live_logs_dir.parent / "hypothesis_experiments_019.json").exists()
        else (ROOT / ".cache")
    )
    if include_historical_specs or (live_logs_dir.parent / "hypothesis_experiments_019.json").exists():
        try:
            backfill_micro_verifications(db, cache_dir=cache_root, model_cards_dir=model_cards_dir)
        except Exception:
            pass

    db_snap = db.sync_snapshot(
        catalog=catalog,
        colab_sessions=colab_sessions,
        ui_prefs=saved_prefs,
        default_obsolete_reasons=DEFAULT_OBSOLETE_KEYS,
    )
    raw_main_cat = list(db_snap.get("runs_catalog", []))
    raw_edu_cat = list(db_snap.get("fineweb_edu_catalog", []))
    raw_micro_cat = list(db_snap.get("micro_runs_catalog", []))

    main_catalog: list[dict[str, Any]] = []
    fineweb_edu_catalog: list[dict[str, Any]] = []
    micro_catalog: list[dict[str, Any]] = []
    seen_part_keys: set[str] = set()
    for src_tag, src_list in (("edu", raw_edu_cat), ("main", raw_main_cat), ("micro", raw_micro_cat)):
        for item in src_list:
            rk = str(item.get("key") or item.get("run_key") or "").strip()
            if not rk or rk in seen_part_keys:
                continue
            seen_part_keys.add(rk)
            rk_low = rk.lower()
            mc_it = item.get("model_card") if isinstance(item.get("model_card"), dict) else {}
            td_it = str(item.get("task_domain") or mc_it.get("task_domain") or "").lower()
            rt_it = str(item.get("run_tier") or mc_it.get("run_tier") or "").lower()
            is_vision_it = any(v in rk_low for v in ("cifar100", "1000class", "imagenet1k")) or td_it == "vision_ablation"
            is_w18_29_it = bool(re.match(r"^w(1[89]|2\d)_", rk_low)) and not is_vision_it
            if is_vision_it:
                micro_catalog.append(item)
            elif is_w18_29_it or td_it == "fineweb_edu_local" or rt_it == "fineweb_edu" or src_tag == "edu":
                fineweb_edu_catalog.append(item)
            elif src_tag == "micro" or rt_it == "micro_verification" or td_it == "micro_verification":
                micro_catalog.append(item)
            else:
                main_catalog.append(item)

    fineweb_edu_catalog.sort(
        key=lambda r: (
            1 if r.get("obsolete") else 0,
            r.get("best_val_ce") is None,
            float(r["best_val_ce"]) if r.get("best_val_ce") is not None else 999.0,
            -float(r.get("wave_num", 0.0)),
        )
    )

    for item in main_catalog + fineweb_edu_catalog:
        rk = str(item.get("key") or "")
        mc = item.get("model_card")
        if isinstance(mc, dict) and rk:
            disk_p = model_cards_dir / f"{rk}.json"
            if not disk_p.exists():
                save_model_card_atomic(mc, model_cards_dir)
            else:
                try:
                    on_disk = json.loads(disk_p.read_text(encoding="utf-8"))
                    if (
                        on_disk.get("pareto_frontier") != mc.get("pareto_frontier")
                        or on_disk.get("obsolete") != mc.get("obsolete")
                        or on_disk.get("obsolete_reason") != mc.get("obsolete_reason")
                    ):
                        save_model_card_atomic(mc, model_cards_dir)
                except Exception:
                    pass
        if rk in fleet_runs:
            fleet_runs[rk]["pareto_frontier"] = bool(item.get("pareto_frontier", False))
            fleet_runs[rk]["obsolete"] = bool(item.get("obsolete", False))
            fleet_runs[rk]["obsolete_reason"] = str(item.get("obsolete_reason") or "")

    snapshot: dict[str, Any] = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "server_uptime_s": round(max(0.001, time.time() - _SERVER_START_TIME), 4),
        "lan_ip": get_lan_ip(),
        "gpu_hw": gpu_hw,
        "local_gpu_hw": local_gpu_hw,
        "tpu_hw": tpu_hw,
        "fleet_hw": fleet_hw,
        "colab_account": colab_account,
        "colab_sessions": db_snap["colab_sessions"],
        "session_queue": db_snap["session_queue"],
        "db_stats": db_snap["db_stats"],
        "gpu_3ep": r_gpu_3ep,
        "gpu_5ep": r_gpu_5ep,
        "tpu_3ep": r_tpu_3ep,
        "gpu_wsd_mtp": r_gpu_wsd,
        "tpu_wsd_mtp": r_tpu_wsd,
        "gpu_20m_5ep": {"steps": [], "evals": BASELINE_20M_5EP_EVALS, "completed": True, "avg_tok_per_s": 311469.0},
        "haliax_xla": haliax_info,
        "runs_catalog": main_catalog,
        "fineweb_edu_catalog": fineweb_edu_catalog,
        "micro_runs_catalog": micro_catalog,
        "model_cards_count": len(main_catalog) + len(fineweb_edu_catalog) + len(micro_catalog),
        "saved_ui_prefs": saved_prefs,
        "targets": {
            "slowrun_ce": SLOWRUN_RECORD_VAL_CE,
            "slowrun_ppl": SLOWRUN_RECORD_VAL_PPL,
            "speedrun_ce": NANOGPT_SPEEDRUN_VAL_CE,
            "speedrun_ppl": NANOGPT_SPEEDRUN_VAL_PPL,
            "speedrun_min": NANOGPT_SPEEDRUN_MINUTES,
        },
    }
    snapshot.update(fleet_runs)
    (live_logs_dir / "latest_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def format_terminal_report(snap: dict[str, Any]) -> str:
    lines = [
        "=" * 124,
        f"QUANTUMDLBLOCKS ZERO-DOWNTIME COLAB FLEET & MODEL CARD MONITOR — Updated: {snap['updated_at']} (LAN: {snap.get('lan_ip')}:8765)",
        "=" * 124,
    ]
    fhw = snap.get("fleet_hw", {})
    catalog = snap.get("runs_catalog", [])
    lines.append("\n[GOOGLE COLAB GPU & TPU FLEET LIVE HARDWARE TELEMETRY]")
    seen_sessions: set[str] = set()
    for item in reversed(catalog):
        sess = item.get("session", "")
        if not sess or sess in seen_sessions or item.get("wave_id") == "Baseline":
            continue
        seen_sessions.add(sess)
        hw = fhw.get(sess, {})
        b_stat = item.get("boot_status", "Idle")
        if item.get("kind") == "gpu":
            dev = hw.get("device", item.get("accel", "GPU"))
            u_gpu = hw.get("gpu_util_pct", 0.0)
            u_mem = hw.get("mem_bw_util_pct", 0.0)
            v_u = hw.get("vram_used_mib", 0.0) / 1024.0
            v_t = hw.get("vram_total_mib", 1.0) / 1024.0
            pwr = hw.get("power_w", 0.0)
            lines.append(
                f"  [{sess:<11}] {dev:<38} | SM:{u_gpu:3.0f}% | MemBW:{u_mem:3.0f}% | VRAM:{v_u:5.1f}/{v_t:5.1f} GiB | {pwr:5.1f}W | {b_stat}"
            )
        else:
            dev = hw.get("device", item.get("accel", "TPU"))
            mxu = hw.get("mxu_duty_cycle_pct", 100.0)
            h_u = hw.get("hbm_used_bytes", 0) / 1e9
            h_t = hw.get("hbm_total_bytes", 16.91e9) / 1e9
            lines.append(
                f"  [{sess:<11}] {dev:<38} | MXU Duty:{mxu:5.1f}% | HBM2e:{h_u:5.2f}/{h_t:5.2f} GB         | TPUv5e | {b_stat}"
            )

    lines.append("\n[WORLD-RECORD TARGET COMPARISONS (5,000,000-Token FineWeb Validation Set, V=50,304, T_val=512)]")
    lines.append(
        f"  * FineWeb 100M Slowrun World Record : {SLOWRUN_RECORD_VAL_CE:.3f} Val CE ({SLOWRUN_RECORD_VAL_PPL:.1f} PPL) on 100M Unique FineWeb Tokens"
    )
    lines.append(
        f"  * Modded-NanoGPT Speedrun Record    : {NANOGPT_SPEEDRUN_VAL_CE:.3f} Val CE ({NANOGPT_SPEEDRUN_VAL_PPL:.1f} PPL) in {NANOGPT_SPEEDRUN_MINUTES:.3f} min (39.9s)"
    )
    lines.append(
        "  Run Configuration                                      | Best Val CE | Val PPL | Wall Time | vs Slowrun (3.183) | vs Speedrun (3.280)"
    )
    lines.append(
        "  -------------------------------------------------------+-------------+---------+-----------+--------------------+--------------------"
    )
    obsolete_hidden = 0
    for item in catalog:
        if item.get("obsolete", False):
            obsolete_hidden += 1
            continue
        r_label = item.get("short_label", item["key"])[:54]
        r_key = item["key"]
        r_obj = snap.get(r_key, item)
        ev_list = [e for e in r_obj.get("evals", []) if e.get("epoch", 0) > 0 or e.get("tokens_M", 0) > 0]
        if not ev_list:
            st_list = r_obj.get("steps", [])
            if st_list:
                s_last = st_list[-1]
                lines.append(
                    f"  {r_label:<54} | (Step {s_last['step']:4d}) |   ---   | {s_last.get('elapsed_min', 0.0):8.2f}m | Train CE: {s_last['train_ce']:.4f}   | {s_last['tok_per_s']:,} tok/s"
                )
            else:
                b_st = r_obj.get("boot_status", "Idle")
                lines.append(
                    f"  {r_label:<54} | ({b_st[:9]:<9}) |   ---   |    0.00m | Model Card Ready   | 5M FineWeb Val Set"
                )
            continue
        best_ev = min(ev_list, key=lambda x: x["val_ce"])
        d_slow = best_ev["val_ce"] - SLOWRUN_RECORD_VAL_CE
        d_speed = best_ev["val_ce"] - NANOGPT_SPEEDRUN_VAL_CE
        lines.append(
            f"  {r_label:<54} |      {best_ev['val_ce']:6.4f} | {best_ev['val_ppl']:7.1f} | {best_ev.get('elapsed_min', 0.0):8.2f}m | "
            f"{d_slow:+18.4f} | {d_speed:+18.4f}"
        )
    if obsolete_hidden > 0:
        lines.append(
            f"  ({obsolete_hidden} obsolete/historical runs hidden — toggle '🗄️ Show Obsolete' in Web UI to inspect)"
        )

    lines.append("\n" + "=" * 124)
    return "\n".join(lines)


def _svg_multi_run_chart(
    title: str,
    y_label: str,
    series_list: list[tuple[str, str, list[tuple[float, float, str]]]],
    max_x: float = 500.0,
    min_y: float = 4.0,
    max_y: float = 6.5,
    x_suffix: str = "M",
    target_lines: list[tuple[float, str, str]] | None = None,
    point_markers: list[tuple[float, float, str, str]] | None = None,
) -> str:
    w, h = 560, 260
    pad_l, pad_r, pad_t, pad_b = 52, 20, 32, 34
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    def x_px(xv: float) -> float:
        return pad_l + (min(max(xv, 0.0), max_x) / max(1e-6, max_x)) * pw

    def y_px(yv: float) -> float:
        c = min(max(yv, min_y), max_y)
        return pad_t + (1.0 - (c - min_y) / max(1e-6, max_y - min_y)) * ph

    elems = [
        f'<text x="{pad_l}" y="18" fill="#e6edf3" font-size="13" font-weight="bold">{title}</text>'
    ]
    for i in range(6):
        yv = min_y + (max_y - min_y) * (i / 5.0)
        yy = y_px(yv)
        elems.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" stroke="#21262d" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{yy + 4:.1f}" fill="#8b949e" font-size="10" text-anchor="end">{yv:.2f}</text>'
        )
    for i in range(6):
        xv = max_x * (i / 5.0)
        xx = x_px(float(xv))
        lbl = f"{int(round(xv))}{x_suffix}" if max_x >= 10 else f"{xv:.1f}{x_suffix}"
        elems.append(
            f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{h - pad_b}" stroke="#21262d" stroke-width="1"/>'
            f'<text x="{xx:.1f}" y="{h - 10}" fill="#8b949e" font-size="10" text-anchor="middle">{lbl}</text>'
        )

    if target_lines:
        for ty, tlabel, tcolor in target_lines:
            yy = y_px(ty)
            elems.append(
                f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" '
                f'stroke="{tcolor}" stroke-width="1.5" stroke-dasharray="5,4"/>'
                f'<text x="{w - pad_r - 4}" y="{yy - 4:.1f}" fill="{tcolor}" font-size="9.5" '
                f'font-weight="bold" text-anchor="end">{tlabel}</text>'
            )

    if point_markers:
        for px, py, plabel, pcolor in point_markers:
            cx, cy = x_px(px), y_px(py)
            elems.append(
                f'<polygon points="{cx:.1f},{cy-6:.1f} {cx+6:.1f},{cy:.1f} {cx:.1f},{cy+6:.1f} {cx-6:.1f},{cy:.1f}" '
                f'fill="{pcolor}" stroke="#0d1117" stroke-width="1.2"><title>{plabel}</title></polygon>'
                f'<text x="{cx + 8:.1f}" y="{cy - 6:.1f}" fill="{pcolor}" font-size="9.5" font-weight="bold">{plabel}</text>'
            )

    for label, color, pts in series_list:
        if not pts:
            continue
        coords = " ".join(f"{x_px(x):.1f},{y_px(y):.1f}" for x, y, _ in pts)
        elems.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{coords}"/>')
        for x, y, tip in pts:
            cx, cy = x_px(x), y_px(y)
            elems.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{color}"><title>{label}: {tip}</title></circle>'
            )

    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'style="background:#161b22;border:1px solid #30363d;border-radius:8px">'
        + "".join(elems)
        + "</svg>"
    )


def render_html_dashboard(snap: dict[str, Any]) -> str:
    initial_payload = {
        "updated_at": snap.get("updated_at", ""),
        "lan_ip": snap.get("lan_ip", "0.0.0.0"),
        "fleet_hw": snap.get("fleet_hw", {}),
        "colab_account": snap.get("colab_account", {}),
        "colab_sessions": snap.get("colab_sessions", []),
        "runs_catalog": snap.get("runs_catalog", []),
        "fineweb_edu_catalog": snap.get("fineweb_edu_catalog", []),
        "micro_runs_catalog": snap.get("micro_runs_catalog", []),
        "session_queue": snap.get("session_queue", []),
        "db_stats": snap.get("db_stats", {}),
        "saved_ui_prefs": snap.get("saved_ui_prefs", {}),
        "targets": snap.get("targets", {}),
    }
    initial_json = json.dumps(initial_payload).replace("</", "<\\/")
    report_txt = (
        format_terminal_report(snap)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>QuantumDLBlocks — Interactive React Experiment &amp; Scaling Explorer</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body {
      background: #0d1117;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, ui-monospace, SFMono-Regular, Menlo, monospace;
      padding: 18px 22px;
      margin: 0;
    }
    code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .header-bar {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 14px;
      padding-bottom: 12px;
      border-bottom: 1px solid #21262d;
    }
    .title-main { color: #58a6ff; font-size: 20px; font-weight: 700; margin: 0 0 4px 0; }
    .sub-text { color: #8b949e; font-size: 12px; }
    .poll-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .btn {
      background: #21262d;
      color: #c9d1d9;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.12s ease;
      font-family: inherit;
    }
    .btn:hover { background: #30363d; color: #ffffff; border-color: #58a6ff; }
    .btn.active { background: #1f6feb; color: #ffffff; border-color: #58a6ff; font-weight: 600; }
    .btn.preset { padding: 4px 9px; font-size: 11.5px; }
    .nav-tabs {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
      padding: 6px 8px;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
    }
    .tab-btn {
      background: transparent;
      color: #8b949e;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 7px 14px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.12s ease;
      font-family: inherit;
    }
    .tab-btn:hover { color: #e6edf3; background: rgba(56, 139, 253, 0.1); }
    .tab-btn.active {
      background: #1f6feb;
      color: #ffffff;
      border-color: #58a6ff;
      box-shadow: 0 2px 8px rgba(31, 111, 235, 0.35);
    }
    .badge-pareto {
      background: rgba(255, 215, 0, 0.16);
      color: #ffd700;
      border: 1px solid rgba(255, 215, 0, 0.45);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .kpi-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px 13px;
    }
    .kpi-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; }
    .kpi-val { font-size: 18px; font-weight: 700; margin: 3px 0; font-family: ui-monospace, monospace; }
    .kpi-sub { color: #8b949e; font-size: 11.5px; }
    .insight-banner {
      background: linear-gradient(90deg, rgba(31,111,235,0.14), rgba(56,239,125,0.08));
      border: 1px solid #1f6feb;
      border-radius: 8px;
      padding: 11px 14px;
      margin-bottom: 14px;
      font-size: 12.5px;
      line-height: 1.45;
    }
    .toolbar-panel {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 14px;
      display: flex;
      flex-direction: column;
      gap: 9px;
    }
    .toolbar-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }
    .preset-group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
    .section-lbl { color: #8b949e; font-size: 11.5px; font-weight: 600; margin-right: 4px; }
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(530px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .chart-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 8px 10px;
      position: relative;
    }
    .table-container {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      overflow-x: auto;
      overflow-y: auto;
      max-height: 62vh;
      position: relative;
      margin-bottom: 16px;
    }
    table.runs-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, table.runs-table th {
      position: sticky;
      top: 0;
      z-index: 5;
      background: #1c2128;
      color: #8b949e;
      text-align: left;
      padding: 8px 9px;
      border-bottom: 1px solid #30363d;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
      font-weight: 600;
      box-shadow: 0 1px 0 #30363d;
    }
    table.runs-table th:hover { color: #58a6ff; background: #21262d; }
    table.runs-table td {
      padding: 7px 9px;
      border-bottom: 1px solid #21262d;
      white-space: nowrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    table.runs-table tr.run-row { transition: background 0.1s ease; }
    table.runs-table tr.run-row:hover { background: rgba(56, 139, 253, 0.12); }
    table.runs-table tr.run-row.hidden-row { opacity: 0.48; }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10.5px;
      font-weight: 600;
    }
    .badge-2beam { background: rgba(56, 239, 125, 0.15); color: #38ef7d; border: 1px solid rgba(56, 239, 125, 0.35); }
    .badge-2beam-frame { background: rgba(0, 242, 254, 0.16); color: #00f2fe; border: 1px solid rgba(0, 242, 254, 0.45); }
    .badge-1beam { background: rgba(255, 159, 67, 0.15); color: #ff9f43; border: 1px solid rgba(255, 159, 67, 0.35); }
    .badge-adapter { background: rgba(0, 210, 255, 0.16); color: #00d2ff; border: 1px solid rgba(0, 210, 255, 0.4); }
    .badge-2beam-adapter { background: rgba(255, 8, 68, 0.16); color: #ff61d2; border: 1px solid rgba(255, 97, 210, 0.45); }
    .badge-kraus { background: rgba(161, 140, 209, 0.18); color: #d2a8ff; border: 1px solid rgba(161, 140, 209, 0.4); }
    .badge-linear { background: rgba(139, 148, 158, 0.15); color: #8b949e; border: 1px solid rgba(139, 148, 158, 0.3); }
    .badge-relaxed { color: #79c0ff; }
    .badge-strict { color: #ff9a9e; }
    .state-badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 10.5px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    .state-running { background: rgba(56, 239, 125, 0.18); color: #38ef7d; border: 1px solid rgba(56, 239, 125, 0.5); }
    .state-compiling { background: rgba(88, 166, 255, 0.18); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.5); }
    .state-completed { background: rgba(139, 148, 158, 0.14); color: #c9d1d9; border: 1px solid rgba(139, 148, 158, 0.35); }
    .state-oom { background: rgba(255, 123, 114, 0.2); color: #ff7b72; border: 1px solid rgba(255, 123, 114, 0.55); }
    .state-stopped { background: rgba(210, 153, 34, 0.18); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.45); }
    @keyframes pulseDot {
      0% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.35; transform: scale(1.25); }
      100% { opacity: 1; transform: scale(1); }
    }
    .pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: currentColor;
      display: inline-block;
      animation: pulseDot 1.3s infinite ease-in-out;
    }
    .inflight-panel {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 11px 14px;
      margin-bottom: 14px;
    }
    .inflight-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(295px, 1fr));
      gap: 9px;
      margin-top: 9px;
    }
    .inflight-card {
      background: #0d1117;
      border: 1px solid #21262d;
      border-radius: 7px;
      padding: 8px 11px;
      font-size: 11.5px;
      transition: border-color 0.15s ease;
    }
    .inflight-card:hover { border-color: #58a6ff; }
    .prog-track {
      width: 100%;
      height: 6px;
      background: #21262d;
      border-radius: 4px;
      overflow: hidden;
      margin: 5px 0;
    }
    .prog-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s ease;
    }
    .search-input, .select-input {
      background: #0d1117;
      color: #e6edf3;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 12px;
      font-family: inherit;
    }
    details.raw-report {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px 14px;
    }
    details.raw-report summary { cursor: pointer; color: #8b949e; font-size: 12px; font-weight: 600; }
    pre.raw-pre {
      background: #0d1117;
      border: 1px solid #21262d;
      border-radius: 6px;
      padding: 12px;
      overflow-x: auto;
      font-size: 11.5px;
      line-height: 1.4;
      margin-top: 10px;
    }
    .btn-card {
      background: rgba(31, 111, 235, 0.16);
      color: #58a6ff;
      border: 1px solid rgba(88, 166, 255, 0.45);
      border-radius: 5px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-card:hover { background: #1f6feb; color: #ffffff; }
    .mc-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(1, 4, 9, 0.78);
      backdrop-filter: blur(3px);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 9999;
      padding: 20px;
    }
    .mc-modal {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      max-width: 920px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      padding: 18px 22px;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.65);
    }
    .mc-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid #21262d;
      margin-bottom: 12px;
    }
    .mc-section {
      background: #0d1117;
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 10px;
    }
    .mc-section-title {
      color: #58a6ff;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      margin-bottom: 8px;
    }
    .mc-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: 8px;
    }
    .mc-kv {
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 6px;
      padding: 6px 9px;
      font-size: 11.5px;
    }
    .mc-k { color: #8b949e; font-size: 10.5px; }
    .mc-v { color: #e6edf3; font-weight: 600; font-family: ui-monospace, monospace; margin-top: 2px; }
    .mc-code {
      background: #010409;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 9px 11px;
      font-family: ui-monospace, monospace;
      font-size: 11px;
      color: #79c0ff;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }
  </style>
</head>
<body>
  <div id="root"></div>
  <details class="raw-report" style="margin-top:14px">
    <summary>Raw Terminal Telemetry Report (Click to Expand)</summary>
    <pre class="raw-pre">__REPORT_TXT__</pre>
  </details>

  <script>
    window.__INITIAL_SNAPSHOT__ = __INITIAL_JSON__;
  </script>
  <script>
    (function() {
      const h = React.createElement;
      const { useState, useEffect, useMemo, useCallback, useRef } = React;

      const STORAGE_KEY = "qdl_visible_runs_v5";

      function renderStateBadge(r) {
        const st = r.run_state || "COMPLETED";
        if (st === "RUNNING") {
          return h("span", { className: "state-badge state-running", title: "Python training process is actively running on Colab GPU/TPU" },
            h("span", { className: "pulse-dot" }),
            "RUNNING"
          );
        }
        if (st === "COMPILING") {
          return h("span", { className: "state-badge state-compiling", title: "Process alive; compiling TorchInductor / XLA graph before Step 150" },
            h("span", { className: "pulse-dot" }),
            "COMPILING"
          );
        }
        if (st === "OOM_CRASH") {
          return h("span", { className: "state-badge state-oom", title: r.error_msg || "CUDA OutOfMemoryError" },
            "🔴 OOM CRASH"
          );
        }
        if (st === "ERROR") {
          return h("span", { className: "state-badge state-oom", title: r.error_msg || "Runtime Error" },
            "🔴 ERROR"
          );
        }
        if (st === "STOPPED" || st === "IDLE") {
          return h("span", { className: "state-badge state-stopped", title: "Registered Model Card / Process Not Running" },
            st === "IDLE" ? "🃏 CARD READY" : "⏸ STOPPED"
          );
        }
        return h("span", { className: "state-badge state-completed" }, "✅ DONE");
      }

      function ModelCardModal({ run, onClose, onToggleObsolete }) {
        const [copyMsg, setCopyMsg] = useState("");
        if (!run) return null;
        const card = run.model_card || {};
        const arch = card.architecture || {};
        const opt = card.optimization || {};
        const dab = card.data_and_batch || {};
        const hw = card.hardware || {};
        const tel = card.telemetry || {};
        const ms = tel.milestones_val_ce || {};
        const qObs = tel.quantum_observables || {};
        const isObs = !!(run.obsolete || card.obsolete);

        const copyMarkdown = () => {
          fetch(`/api/model-cards/${encodeURIComponent(run.key)}?format=md`)
            .then(r => r.text())
            .then(md => {
              if (navigator.clipboard) navigator.clipboard.writeText(md);
              setCopyMsg("Copied Markdown!");
              setTimeout(() => setCopyMsg(""), 2000);
            })
            .catch(() => setCopyMsg("Copy failed"));
        };

        const copyJson = () => {
          const txt = JSON.stringify(card, null, 2);
          if (navigator.clipboard) navigator.clipboard.writeText(txt);
          setCopyMsg("Copied JSON!");
          setTimeout(() => setCopyMsg(""), 2000);
        };

        const cliCmd = `PYTHONPATH=src python3 -u examples/11_fineweb_100m_multiepoch_gpu_tpu.py ${hw.cli_args || ""} --out-json ${hw.remote_json || ""}`;

        const copyCli = () => {
          if (navigator.clipboard) navigator.clipboard.writeText(cliCmd);
          setCopyMsg("Copied CLI!");
          setTimeout(() => setCopyMsg(""), 2000);
        };

        const kv = (k, v) => h("div", { className: "mc-kv", key: k },
          h("div", { className: "mc-k" }, k),
          h("div", { className: "mc-v" }, v != null && v !== "" ? String(v) : "—")
        );

        const bestCe = run.best_val_ce != null ? run.best_val_ce : tel.best_val_ce;
        const deltaSpeedrun = bestCe != null ? `${(bestCe - 3.280) >= 0 ? "+" : ""}${(bestCe - 3.280).toFixed(4)} vs 3.280` : "—";

        return h("div", { className: "mc-backdrop", onClick: onClose },
          h("div", { className: "mc-modal", onClick: e => e.stopPropagation() },
            h("div", { className: "mc-header" },
              h("div", null,
                h("div", { style: { display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" } },
                  h("span", { style: { width: 12, height: 12, borderRadius: "50%", background: run.color, display: "inline-block" } }),
                  h("span", { style: { fontSize: 16, fontWeight: 700, color: "#e6edf3" } }, `🃏 Model Card: ${run.key}`),
                  h("span", { className: "badge badge-adapter" }, card.wave_id || run.wave_id || "Wave"),
                  renderStateBadge(run),
                  isObs && h("span", {
                    className: "state-badge state-oom",
                    title: run.obsolete_reason || card.obsolete_reason || "Marked Obsolete"
                  }, "🗄️ OBSOLETE")
                ),
                h("div", { style: { color: "#8b949e", fontSize: 12, marginTop: 4 } }, card.label || run.label),
                isObs && (run.obsolete_reason || card.obsolete_reason) && h("div", {
                  style: { color: "#ff7b72", fontSize: 11.5, marginTop: 3 }
                }, `Obsolete Reason: ${run.obsolete_reason || card.obsolete_reason}`)
              ),
              h("div", { style: { display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" } },
                copyMsg && h("span", { style: { color: "#38ef7d", fontSize: 11.5, fontWeight: 600 } }, copyMsg),
                onToggleObsolete && h("button", {
                  className: "btn preset",
                  title: isObs ? "Restore this obsolete run to active leaderboard" : "Mark this run as obsolete and hide from default table",
                  onClick: () => onToggleObsolete([run.key], !isObs, isObs ? "" : "Marked obsolete from Model Card")
                }, isObs ? "♻️ Restore" : "🗄️ Mark Obsolete"),
                h("button", { className: "btn preset", onClick: copyMarkdown }, "📋 Copy Markdown"),
                h("button", { className: "btn preset", onClick: copyJson }, "📦 Copy JSON"),
                h("button", { className: "btn preset", onClick: onClose }, "✕ Close")
              )
            ),

            // Section 1: Hypothesis & Milestone Summary
            h("div", { className: "mc-section" },
              h("div", { className: "mc-section-title" }, "1. Hypothesis & Milestone Summary (T_val = 512)"),
              h("div", { style: { fontSize: 12, color: "#c9d1d9", marginBottom: 8, lineHeight: 1.45 } },
                run.hypothesis_intent || card.hypothesis_summary || run.label
              ),
              h("div", { className: "mc-grid" },
                kv("Best Val CE", bestCe != null ? `${bestCe.toFixed(4)} (${deltaSpeedrun})` : "—"),
                kv("Best Val PPL", (run.best_val_ppl || tel.best_val_ppl) != null ? (run.best_val_ppl || tel.best_val_ppl).toFixed(2) : "—"),
                kv("Top-1 / Top-5 Acc", run.best_top1 != null ? `${run.best_top1.toFixed(2)}% / ${(run.best_top5 || 0).toFixed(2)}%` : "—"),
                kv("Throughput / Wall Time", `${Math.round(run.avg_tok_per_s || 0).toLocaleString()} tok/s (${(run.wall_min || 0).toFixed(1)}m)`),
                kv("50M / 100M Val CE", `${ms["50M"] != null ? ms["50M"].toFixed(4) : "—"} / ${ms["100M"] != null ? ms["100M"].toFixed(4) : "—"}`),
                kv("200M / 450M Val CE", `${ms["200M"] != null ? ms["200M"].toFixed(4) : "—"} / ${ms["450M"] != null ? ms["450M"].toFixed(4) : "—"}`),
                kv("600M / 900M Val CE", `${ms["600M"] != null ? ms["600M"].toFixed(4) : "—"} / ${ms["900M"] != null ? ms["900M"].toFixed(4) : "—"}`),
                kv("Progress", `Step ${run.current_step}/${run.total_steps} (${(run.live_tokens_M || 0).toFixed(1)}M / ${run.target_tokens_M}M tok)`)
              )
            ),

            // Section 2: Architecture & Geometric Primitives
            h("div", { className: "mc-section" },
              h("div", { className: "mc-section-title" }, "2. Architecture & Quantum/Classical Geometric Primitives"),
              h("div", { className: "mc-grid" },
                kv("Architecture Family", `${arch.family || "quantum_born"} (${arch.params_M || run.params_M}M params)`),
                kv("Backbone (L, d, H, d_h)", `L=${arch.num_layers || run.num_layers}, d=${arch.embed_dim || run.embed_dim}, H=${arch.num_heads || 16}, d_h=${arch.head_dim || 64}`),
                kv("MLP / Frame", `${arch.d_ff_display || run.d_ff} (rho=${arch.frame_expansion || 2.0})`),
                kv("Seed & Readout", `D_seed=${arch.seed_dim || run.seed_dim} | ${arch.readout_mode || run.readout_mode} | Adapter=${arch.seed_adapter || "None"}`),
                kv("Value Embeddings (ve)", `${arch.num_value_embeddings || 0} banks`),
                kv("U-Net Skip & Fused QKV", `unet=${arch.unet_skip_mode || "matrix"} | fused_qkv=${!!arch.use_fused_qkv}`),
                kv("Skip Attn & Half-RoPE", `skip=${JSON.stringify(arch.skip_attn_layers || [])} | half_rope=${!!arch.zero_half_rope}`),
                kv("DocMask & Sliding Window", `doc_mask=${!!arch.doc_mask_attn} | window=${arch.sliding_window || 0}`),
                kv("Quantum EarlyFix Flags", `dyn_beta=${!!arch.dynamic_povm_beta}, ortho=${!!arch.ortho_decouple}, prior=${!!arch.povm_log_prior}, flash_pur=${!!arch.purified_born_attn}`)
              )
            ),

            // Section 3: Optimizer, Schedule & Batch Configuration
            h("div", { className: "mc-section" },
              h("div", { className: "mc-section-title" }, "3. Optimizer, Schedule & Batch Configuration"),
              h("div", { className: "mc-grid" },
                kv("Optimizer Mode", opt.optimizer_mode || "muon_adamw"),
                kv("Schedule & WSD", `${opt.schedule || run.schedule} (decay=${opt.wsd_decay_fraction}, min_lr=${opt.wsd_min_lr_ratio})`),
                kv("Muon LR & Warmup", `lr=${opt.muon_lr || 0.025}, mom=${opt.muon_momentum || 0.95}, warmup=${opt.muon_momentum_warmup_steps || 0} steps`),
                kv("AdamW LRs (WTE/Head/Scalar)", `${opt.adamw_lr_wte || 0.004} / ${opt.adamw_lr_head || 0.004} / ${opt.adamw_lr_scalar || 0.004}`),
                kv("Context (T_train / T_val)", `T_train=${dab.seq_len_train || 512} | T_val=${dab.seq_len_val || 512}`),
                kv("Global Batch", `B=${dab.batch_size || 128} × G_acc=${dab.grad_accum || 1} = ${(dab.tokens_per_step || 65536).toLocaleString()} tok/step`),
                kv("Systems Roofline", `direct_chunked_ce=${!!opt.direct_chunked_ce} | stage_gpu=${!!opt.stage_shard_on_gpu}`)
              )
            ),

            // Section 4: Quantum Observables
            h("div", { className: "mc-section" },
              h("div", { className: "mc-section-title" }, "4. Quantum-Geometric Observables"),
              h("div", { className: "mc-grid" },
                kv("Seed / Sharp Effective Rank", qObs.seed_eff_rank != null ? Number(qObs.seed_eff_rank).toFixed(2) : (run.sharp_eff_rank != null ? run.sharp_eff_rank.toFixed(2) : (run.final_r_eff != null ? run.final_r_eff.toFixed(2) : "—"))),
                kv("10-Qubit Entanglement S(ρ_A)", qObs.entanglement_nats != null ? `${Number(qObs.entanglement_nats).toFixed(4)} nats` : "—"),
                kv("Stiefel/Kraus Drift IsoErr", qObs.iso_err != null ? Number(qObs.iso_err).toFixed(4) : (run.kraus_stiefel_err != null ? run.kraus_stiefel_err.toFixed(4) : (run.final_iso_err != null ? run.final_iso_err.toFixed(4) : "—"))),
                kv("POVM Sharpness β / γ_h", `${qObs.povm_beta != null ? Number(qObs.povm_beta).toFixed(2) : "—"} / ${qObs.gamma_h != null ? Number(qObs.gamma_h).toFixed(2) : "—"}`)
              )
            ),

            // Section 5: Hardware & Reproducible CLI Command
            h("div", { className: "mc-section", style: { marginBottom: 0 } },
              h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 } },
                h("div", { className: "mc-section-title", style: { marginBottom: 0 } },
                  `5. Hardware (${hw.session || run.session} • ${hw.accelerator || run.accel}) & Reproducible CLI`
                ),
                h("button", { className: "btn preset", onClick: copyCli }, "📋 Copy CLI Command")
              ),
              h("div", { className: "mc-code" }, cliCmd)
            )
          )
        );
      }

      function SvgTrajectoryChart({
        title,
        series,
        maxX,
        minY,
        maxY,
        xSuffix,
        targetLines,
        pointMarkers,
        hoveredKey,
        onHoverKey
      }) {
        const [zoomBox, setZoomBox] = useState(null); // { minX, maxX, minY, maxY } or null
        const [dragRect, setDragRect] = useState(null); // { startPxX, startPxY, curPxX, curPxY } or null

        const w = 560, height = 255;
        const padL = 50, padR = 18, padT = 30, padB = 32;
        const pw = w - padL - padR;
        const ph = height - padT - padB;

        const effMinX = zoomBox ? zoomBox.minX : 0.0;
        const effMaxX = zoomBox ? zoomBox.maxX : maxX;
        const effMinY = zoomBox ? zoomBox.minY : minY;
        const effMaxY = zoomBox ? zoomBox.maxY : maxY;

        const xPx = (xv) => padL + ((Math.min(Math.max(xv, effMinX), effMaxX) - effMinX) / Math.max(1e-6, effMaxX - effMinX)) * pw;
        const yPx = (yv) => {
          const c = Math.min(Math.max(yv, effMinY), effMaxY);
          return padT + (1.0 - (c - effMinY) / Math.max(1e-6, effMaxY - effMinY)) * ph;
        };

        const getSvgCoords = (ev) => {
          const svgEl = ev.currentTarget;
          const rect = svgEl.getBoundingClientRect();
          const pxX = ((ev.clientX - rect.left) / Math.max(1, rect.width)) * w;
          const pxY = ((ev.clientY - rect.top) / Math.max(1, rect.height)) * height;
          return {
            pxX: Math.min(Math.max(pxX, padL), w - padR),
            pxY: Math.min(Math.max(pxY, padT), height - padB)
          };
        };

        const handleMouseDown = (ev) => {
          if (ev.button !== 0) return;
          const c = getSvgCoords(ev);
          setDragRect({ startPxX: c.pxX, startPxY: c.pxY, curPxX: c.pxX, curPxY: c.pxY });
        };

        const handleMouseMove = (ev) => {
          if (!dragRect) return;
          const c = getSvgCoords(ev);
          setDragRect(prev => prev ? { ...prev, curPxX: c.pxX, curPxY: c.pxY } : null);
        };

        const handleMouseUp = () => {
          if (!dragRect) return;
          const dx = Math.abs(dragRect.curPxX - dragRect.startPxX);
          const dy = Math.abs(dragRect.curPxY - dragRect.startPxY);
          if (dx >= 8 && dy >= 8) {
            const pxX1 = Math.min(dragRect.startPxX, dragRect.curPxX);
            const pxX2 = Math.max(dragRect.startPxX, dragRect.curPxX);
            const pxY1 = Math.min(dragRect.startPxY, dragRect.curPxY);
            const pxY2 = Math.max(dragRect.startPxY, dragRect.curPxY);
            const newMinX = effMinX + ((pxX1 - padL) / pw) * (effMaxX - effMinX);
            const newMaxX = effMinX + ((pxX2 - padL) / pw) * (effMaxX - effMinX);
            const newMaxY = effMinY + (1.0 - (pxY1 - padT) / ph) * (effMaxY - effMinY);
            const newMinY = effMinY + (1.0 - (pxY2 - padT) / ph) * (effMaxY - effMinY);
            if (newMaxX > newMinX + 1e-6 && newMaxY > newMinY + 1e-6) {
              setZoomBox({ minX: newMinX, maxX: newMaxX, minY: newMinY, maxY: newMaxY });
            }
          }
          setDragRect(null);
        };

        const handleDoubleClick = () => {
          setZoomBox(null);
          setDragRect(null);
        };

        const gridLines = [];
        for (let i = 0; i <= 5; i++) {
          const yv = effMinY + (effMaxY - effMinY) * (i / 5.0);
          const yy = yPx(yv);
          gridLines.push(
            h("g", { key: "gy-" + i },
              h("line", { x1: padL, y1: yy, x2: w - padR, y2: yy, stroke: "#21262d", strokeWidth: 1 }),
              h("text", { x: padL - 6, y: yy + 3.5, fill: "#8b949e", fontSize: 10, textAnchor: "end" }, yv.toFixed(2))
            )
          );
        }
        const spanX = Math.max(1e-6, effMaxX - effMinX);
        for (let i = 0; i <= 5; i++) {
          const xv = effMinX + spanX * (i / 5.0);
          const xx = xPx(xv);
          const lbl = (spanX >= 10 ? Math.round(xv) : xv.toFixed(spanX < 2 ? 2 : 1)) + (xSuffix || "M");
          gridLines.push(
            h("g", { key: "gx-" + i },
              h("line", { x1: xx, y1: padT, x2: xx, y2: height - padB, stroke: "#21262d", strokeWidth: 1 }),
              h("text", { x: xx, y: height - 9, fill: "#8b949e", fontSize: 10, textAnchor: "middle" }, lbl)
            )
          );
        }

        const targets = (targetLines || []).map((t, idx) => {
          if (t.y < effMinY || t.y > effMaxY) return null;
          const yy = yPx(t.y);
          return h("g", { key: "tgt-" + idx },
            h("line", {
              x1: padL, y1: yy, x2: w - padR, y2: yy,
              stroke: t.color, strokeWidth: 1.4, strokeDasharray: "5,4"
            }),
            h("text", {
              x: w - padR - 4, y: yy - 4, fill: t.color, fontSize: 9.5, fontWeight: "bold", textAnchor: "end"
            }, t.label)
          );
        });

        const markers = (pointMarkers || []).map((pm, idx) => {
          if (pm.y < effMinY || pm.y > effMaxY || pm.x < effMinX || pm.x > effMaxX) return null;
          const cx = xPx(pm.x), cy = yPx(pm.y);
          return h("g", { key: "pm-" + idx },
            h("polygon", {
              points: `${cx},${cy-6} ${cx+6},${cy} ${cx},${cy+6} ${cx-6},${cy}`,
              fill: pm.color, stroke: "#0d1117", strokeWidth: 1.2
            }, h("title", null, pm.label)),
            h("text", { x: cx + 8, y: cy - 5, fill: pm.color, fontSize: 9.5, fontWeight: "bold" }, pm.label)
          );
        });

        const sortedSeries = [...series].sort((a, b) => {
          if (a.key === hoveredKey) return 1;
          if (b.key === hoveredKey) return -1;
          return 0;
        });

        const polylines = sortedSeries.map((s) => {
          const visPts = (s.pts || []).filter(p => p.x >= effMinX - 1e-6 && p.x <= effMaxX + 1e-6);
          const hasPts = visPts.length > 0;
          if (!hasPts && !s.liveFrontier) return null;
          const isHovered = hoveredKey === s.key;
          const isDimmed = hoveredKey && hoveredKey !== s.key;
          const opacity = isHovered ? 1.0 : (isDimmed ? 0.20 : 0.92);
          const sw = isHovered ? 3.6 : 2.1;
          const coords = hasPts ? visPts.map(p => `${xPx(p.x).toFixed(1)},${yPx(p.y).toFixed(1)}`).join(" ") : "";

          let liveEl = null;
          if (s.liveFrontier) {
            const lf = s.liveFrontier;
            const lx1 = xPx(lf.x1), ly1 = yPx(lf.y1);
            const lx2 = xPx(lf.x2), ly2 = yPx(lf.y2);
            liveEl = h("g", { key: "lf-" + s.key },
              h("line", {
                x1: lx1, y1: ly1, x2: lx2, y2: ly2,
                stroke: s.color, strokeWidth: 2.0, strokeDasharray: "4,3"
              }),
              h("circle", {
                cx: lx2, cy: ly2, r: 6.5, fill: "none", stroke: s.color, strokeWidth: 1.5, strokeDasharray: "2,2"
              }),
              h("circle", {
                cx: lx2, cy: ly2, r: 3.6, fill: "#0d1117", stroke: s.color, strokeWidth: 2.2
              }, h("title", null, lf.tip)),
              h("text", {
                x: Math.min(w - padR - 5, lx2 + 7),
                y: Math.max(padT + 10, ly2 - 5),
                fill: s.color,
                fontSize: 9.5,
                fontWeight: "bold"
              }, `⚡${lf.x2.toFixed(0)}M`)
            );
          }

          return h("g", {
            key: s.key,
            style: { opacity, cursor: "pointer" },
            onMouseEnter: () => onHoverKey && onHoverKey(s.key),
            onMouseLeave: () => onHoverKey && onHoverKey(null)
          },
            hasPts && h("polyline", { fill: "none", stroke: s.color, strokeWidth: sw, points: coords }),
            hasPts && visPts.map((p, i) => {
              const cx = xPx(p.x), cy = yPx(p.y);
              return h("circle", {
                key: i, cx, cy, r: isHovered ? 4.5 : 3.2, fill: s.color
              }, h("title", null, `${s.label}: ${p.tip}`));
            }),
            liveEl
          );
        });

        const selectionOverlay = dragRect && h("rect", {
          x: Math.min(dragRect.startPxX, dragRect.curPxX),
          y: Math.min(dragRect.startPxY, dragRect.curPxY),
          width: Math.abs(dragRect.curPxX - dragRect.startPxX),
          height: Math.abs(dragRect.curPxY - dragRect.startPxY),
          fill: "rgba(88, 166, 255, 0.18)",
          stroke: "#58a6ff",
          strokeWidth: 1.5,
          strokeDasharray: "4,2"
        });

        const zoomControlBadge = zoomBox
          ? h("g", {
              style: { cursor: "pointer" },
              onClick: (e) => { e.stopPropagation(); setZoomBox(null); setDragRect(null); }
            },
              h("rect", { x: w - padR - 92, y: 4, width: 92, height: 18, rx: 4, fill: "#1f6feb", stroke: "#58a6ff", strokeWidth: 1 }),
              h("text", { x: w - padR - 46, y: 16, fill: "#ffffff", fontSize: 10, fontWeight: "bold", textAnchor: "middle" }, "🔍 Reset Zoom")
            )
          : h("text", { x: w - padR - 4, y: 17, fill: "#8b949e", fontSize: 9.5, textAnchor: "end" }, "🖱️ Drag to Box-Zoom");

        return h("div", { className: "chart-card" },
          h("svg", {
            viewBox: `0 0 ${w} ${height}`,
            width: "100%",
            height,
            style: { userSelect: "none", cursor: dragRect ? "crosshair" : "default" },
            onMouseDown: handleMouseDown,
            onMouseMove: handleMouseMove,
            onMouseUp: handleMouseUp,
            onMouseLeave: handleMouseUp,
            onDoubleClick: handleDoubleClick
          },
            h("text", { x: padL, y: 18, fill: "#e6edf3", fontSize: 12, fontWeight: "bold" }, title),
            gridLines,
            targets,
            markers,
            polylines,
            selectionOverlay,
            zoomControlBadge
          )
        );
      }

      function SvgSeedScalingChart({ catalog, hoveredKey, onHoverKey }) {
        const w = 560, height = 255;
        const padL = 50, padR = 22, padT = 32, padB = 34;
        const pw = w - padL - padR, ph = height - padT - padB;
        const seedTicks = [128, 256, 384, 512, 1024];
        const xPx = (sd) => {
          const logMin = Math.log2(128), logMax = Math.log2(1024);
          const frac = (Math.log2(Math.max(128, Math.min(1024, sd))) - logMin) / (logMax - logMin);
          return padL + frac * pw;
        };
        const minY = 3.88, maxY = 4.38;
        const yPx = (yv) => {
          const c = Math.min(Math.max(yv, minY), maxY);
          return padT + (1.0 - (c - minY) / (maxY - minY)) * ph;
        };

        const byKey = {};
        catalog.forEach(r => { byKey[r.key] = r; });

        const families = [
          {
            name: "Wave-4 900M Single-Pass FineWeb Speedrun (Sub-4.00 Record!)",
            color: "#ff0844",
            keys: ["w4_900m_adapter384_2beam_cosine", "w4_900m_full_synergy512_cosine"]
          },
          {
            name: "Plan 009 2-Beam 2d-Frame (2048x2) + Plan 007 SeedAdapter (300M)",
            color: "#ffd200",
            keys: ["w3_gpu_2beam_frame_adapter384", "w3_gpu_full_synergy384", "w3_gpu_full_synergy512"]
          },
          {
            name: "Plan 008 2-Beam Bilinear (1536x2) + Plan 007 SeedAdapter (300M)",
            color: "#00f2fe",
            keys: ["w2_gpu_adapter256", "w3_gpu_adapter384_2beam", "w3_gpu_adapter512_2beam", "w3_gpu_adapter1024_2beam"]
          },
          {
            name: "2-Beam Bilinear (1536x2) + Linear Seed (Relaxed Iso)",
            color: "#38ef7d",
            keys: ["gpu_3ep", "w2_gpu_seed384", "w3_gpu_seed512_2beam", "w2_gpu_seed1024"]
          },
          {
            name: "Wave-2 1-Beam 2d-Frame (2048) + Relaxed Iso",
            color: "#ff9f43",
            keys: ["w2_gpu_frame256", "w2_gpu_adapter384", "w2_gpu_seed512"]
          }
        ];

        const grid = [];
        for (let i = 0; i <= 5; i++) {
          const yv = minY + (maxY - minY) * (i / 5.0);
          const yy = yPx(yv);
          grid.push(
            h("g", { key: "sy-" + i },
              h("line", { x1: padL, y1: yy, x2: w - padR, y2: yy, stroke: "#21262d", strokeWidth: 1 }),
              h("text", { x: padL - 6, y: yy + 3.5, fill: "#8b949e", fontSize: 10, textAnchor: "end" }, yv.toFixed(2))
            )
          );
        }
        seedTicks.forEach((sd) => {
          const xx = xPx(sd);
          grid.push(
            h("g", { key: "sx-" + sd },
              h("line", { x1: xx, y1: padT, x2: xx, y2: height - padB, stroke: "#21262d", strokeWidth: 1 }),
              h("text", { x: xx, y: height - 9, fill: "#8b949e", fontSize: 10, textAnchor: "middle" }, "D=" + sd)
            )
          );
        });

        const els = families.map((fam, fIdx) => {
          const pts = fam.keys.map(k => byKey[k]).filter(r => r && r.best_val_ce != null);
          if (pts.length === 0) return null;
          const sortedPts = [...pts].sort((a, b) => a.seed_dim - b.seed_dim);
          const coords = sortedPts.map(r => `${xPx(r.seed_dim).toFixed(1)},${yPx(r.best_val_ce).toFixed(1)}`).join(" ");
          return h("g", { key: "fam-" + fIdx },
            sortedPts.length > 1 && h("polyline", {
              fill: "none", stroke: fam.color, strokeWidth: 2.3, strokeDasharray: fIdx === 4 ? "5,3" : undefined, points: coords
            }),
            sortedPts.map(r => {
              const cx = xPx(r.seed_dim), cy = yPx(r.best_val_ce);
              const isHov = hoveredKey === r.key;
              return h("g", {
                key: r.key,
                style: { cursor: "pointer" },
                onMouseEnter: () => onHoverKey && onHoverKey(r.key),
                onMouseLeave: () => onHoverKey && onHoverKey(null)
              },
                h("circle", { cx, cy, r: isHov ? 6.5 : 4.8, fill: fam.color, stroke: "#0d1117", strokeWidth: 1.4 },
                  h("title", null, `${r.short_label} | ${r.params_M}M params | D_seed=${r.seed_dim} | ${r.d_ff} | Best CE: ${r.best_val_ce.toFixed(4)} (${r.run_state})`)
                ),
                h("text", {
                  x: cx, y: cy - 7, fill: fam.color, fontSize: 9.2, fontWeight: "bold", textAnchor: "middle"
                }, r.best_val_ce.toFixed(4))
              );
            })
          );
        });

        return h("div", { className: "chart-card" },
          h("svg", { viewBox: `0 0 ${w} ${height}`, width: "100%", height },
            h("text", { x: padL, y: 17, fill: "#58a6ff", fontSize: 12.5, fontWeight: "bold" },
              "Architecture Discovery: Best Val CE vs. Seed Dimension D_seed (Log2 Scale)"
            ),
            grid,
            els,
            h("g", { transform: `translate(${padL + 10}, ${padT + 5})` },
              families.map((f, i) =>
                h("text", { key: i, x: 0, y: i * 12, fill: f.color, fontSize: 9.0, fontWeight: "600" }, "● " + f.name)
              )
            )
          )
        );
      }

      function App() {
        const [snap, setSnap] = useState(window.__INITIAL_SNAPSHOT__);
        const [activeTab, setActiveTab] = useState(() => {
          try {
            const p = new URLSearchParams(window.location.search).get("tab");
            if (p === "visualizer" || window.location.hash === "#visualizer") return "visualizer";
            if (p && ["main", "fineweb_edu", "micro", "colab", "visualizer"].includes(p)) return p;
          } catch (e) {}
          return "main";
        });
        const [autoPoll, setAutoPoll] = useState(true);
        const [lastFetchStatus, setLastFetchStatus] = useState("Live");
        const [hoveredKey, setHoveredKey] = useState(null);
        const [selectedCardRun, setSelectedCardRun] = useState(null);
        const [yZoomMode, setYZoomMode] = useState("zoomed"); // "zoomed" | "auto" | "mid" | "full"
        const [xAxisMode, setXAxisMode] = useState("tokens"); // "tokens" | "tflops" | "tau" | "time"
        const [stopMsg, setStopMsg] = useState("");

        const manuallyHiddenRef = useRef(new Set());

        const [sortField, setSortField] = useState("best_val_ce");
        const [sortAsc, setSortAsc] = useState(true);
        const [filterStatus, setFilterStatus] = useState("ALL");
        const [filterFamily, setFilterFamily] = useState("ALL");
        const [showObsolete, setShowObsolete] = useState(false);
        const [showAllWaveCards, setShowAllWaveCards] = useState(false);
        const [filterSeed, setFilterSeed] = useState("ALL");
        const [filterMlp, setFilterMlp] = useState("ALL");
        const [filterReadout, setFilterReadout] = useState("ALL");
        const [searchText, setSearchText] = useState("");

        const [eduFilterWave, setEduFilterWave] = useState("ALL");
        const [eduSearchText, setEduSearchText] = useState("");
        const [eduSortField, setEduSortField] = useState("best_val_ce");
        const [eduSortAsc, setEduSortAsc] = useState(true);

        const [microSearchText, setMicroSearchText] = useState("");
        const [microFilterFamily, setMicroFilterFamily] = useState("ALL");
        const [microSortField, setMicroSortField] = useState("wave_num");
        const [microSortAsc, setMicroSortAsc] = useState(false);

        const [obsoleteMap, setObsoleteMap] = useState(() => {
          const m = { ...((window.__INITIAL_SNAPSHOT__.saved_ui_prefs || {}).obsoleteMap || {}) };
          (window.__INITIAL_SNAPSHOT__.runs_catalog || []).forEach(r => {
            if (!(r.key in m)) {
              m[r.key] = !!r.obsolete;
            }
          });
          return m;
        });

        const [visibleMap, setVisibleMap] = useState(() => {
          let base = {};
          try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) {
              const parsed = JSON.parse(raw);
              if (parsed && typeof parsed === "object" && Object.keys(parsed).length > 0) {
                base = parsed;
              }
            }
          } catch (e) {}
          if (Object.keys(base).length === 0) {
            const srv = (window.__INITIAL_SNAPSHOT__.saved_ui_prefs || {}).visibleMap;
            if (srv && typeof srv === "object" && Object.keys(srv).length > 0) {
              base = { ...srv };
            }
          }
          const init = { ...base };
          (window.__INITIAL_SNAPSHOT__.runs_catalog || []).forEach(r => {
            if (r.obsolete) {
              init[r.key] = false;
              return;
            }
            if (!(r.key in init)) {
              init[r.key] = !!r.default_visible;
            }
            if (r.run_state === "RUNNING" || r.run_state === "COMPILING" || r.default_visible) {
              if (!(r.key in base)) {
                init[r.key] = true;
              }
            }
          });
          return init;
        });

        const persistVisibleMap = useCallback((nextMap) => {
          setVisibleMap(nextMap);
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(nextMap));
          } catch (e) {}
          fetch("/api/ui_prefs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ visibleMap: nextMap })
          }).catch(() => {});
        }, []);

        const fetchLatest = useCallback(() => {
          fetch("/api/snapshot")
            .then(r => r.json())
            .then(data => {
              if (data && data.runs_catalog) {
                setSnap({
                  updated_at: data.updated_at,
                  lan_ip: data.lan_ip,
                  fleet_hw: data.fleet_hw || {},
                  colab_account: data.colab_account || {},
                  colab_sessions: data.colab_sessions || [],
                  runs_catalog: data.runs_catalog,
                  fineweb_edu_catalog: data.fineweb_edu_catalog || [],
                  micro_runs_catalog: data.micro_runs_catalog || [],
                  session_queue: data.session_queue || [],
                  db_stats: data.db_stats || {},
                  targets: data.targets || {}
                });
                setLastFetchStatus("Live");
                setObsoleteMap(prev => {
                  const next = { ...prev };
                  data.runs_catalog.forEach(r => {
                    next[r.key] = !!r.obsolete;
                  });
                  return next;
                });
                setVisibleMap(prev => {
                  let changed = false;
                  const copy = { ...prev };
                  data.runs_catalog.forEach(r => {
                    if (r.obsolete) return;
                    const isInFlight = r.run_state === "RUNNING" || r.run_state === "COMPILING";
                    if (!(r.key in copy)) {
                      copy[r.key] = !!(r.default_visible || isInFlight);
                      changed = true;
                    } else if (isInFlight && !copy[r.key] && !manuallyHiddenRef.current.has(r.key)) {
                      copy[r.key] = true;
                      changed = true;
                    }
                  });
                  return changed ? copy : prev;
                });
              }
            })
            .catch(() => setLastFetchStatus("Reconnecting..."));
        }, []);

        const stopColabVm = useCallback((sessionName) => {
          const payload = sessionName ? { session: sessionName } : { all_idle: true };
          setStopMsg(sessionName ? `Stopping ${sessionName}...` : "Stopping idle Colab VMs...");
          fetch("/api/colab-stop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          })
            .then(r => r.json())
            .then(res => {
              const cnt = (res.stopped_sessions || []).length;
              setStopMsg(cnt > 0 ? `Stopped ${cnt} VM(s) (0.00 CU/hr)` : "All VMs already stopped (0.00 CU/hr)");
              setTimeout(() => setStopMsg(""), 4000);
              fetchLatest();
            })
            .catch(() => {
              setStopMsg("Stop request failed");
              setTimeout(() => setStopMsg(""), 3000);
            });
        }, [fetchLatest]);

        const toggleRunObsolete = useCallback((runKeys, targetObsolete, reason) => {
          const keys = Array.isArray(runKeys) ? runKeys : [runKeys];
          if (keys.length === 0) return;
          const rsn = targetObsolete ? (reason || "Marked obsolete from dashboard") : "";
          setObsoleteMap(prev => {
            const next = { ...prev };
            keys.forEach(k => { next[k] = !!targetObsolete; });
            return next;
          });
          setSnap(prev => {
            if (!prev || !prev.runs_catalog) return prev;
            const nextCat = prev.runs_catalog.map(r => {
              if (!keys.includes(r.key)) return r;
              const nextCard = r.model_card
                ? { ...r.model_card, obsolete: !!targetObsolete, obsolete_reason: rsn }
                : r.model_card;
              return { ...r, obsolete: !!targetObsolete, obsolete_reason: rsn, model_card: nextCard };
            });
            return { ...prev, runs_catalog: nextCat };
          });
          setSelectedCardRun(prev => {
            if (!prev || !keys.includes(prev.key)) return prev;
            const nextCard = prev.model_card
              ? { ...prev.model_card, obsolete: !!targetObsolete, obsolete_reason: rsn }
              : prev.model_card;
            return { ...prev, obsolete: !!targetObsolete, obsolete_reason: rsn, model_card: nextCard };
          });
          if (targetObsolete) {
            setVisibleMap(prev => {
              const next = { ...prev };
              keys.forEach(k => {
                next[k] = false;
                manuallyHiddenRef.current.add(k);
              });
              try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch (e) {}
              return next;
            });
          }
          fetch("/api/obsolete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              run_key: keys[0],
              run_keys: keys,
              obsolete: !!targetObsolete,
              reason: rsn
            })
          }).then(() => fetchLatest()).catch(() => {});
        }, [fetchLatest]);

        const triggerHotReload = useCallback(() => {
          fetch("/api/reload", { method: "POST" })
            .then(() => fetchLatest())
            .catch(() => fetchLatest());
        }, [fetchLatest]);

        useEffect(() => {
          if (!autoPoll) return;
          const id = setInterval(fetchLatest, 10000);
          return () => clearInterval(id);
        }, [autoPoll, fetchLatest]);

        const catalog = snap.runs_catalog || [];
        const finewebEduCatalog = snap.fineweb_edu_catalog || [];
        const microCatalog = snap.micro_runs_catalog || [];
        const sessionQueue = snap.session_queue || [];
        const colabAccount = snap.colab_account || {};
        const dbStats = snap.db_stats || {};

        const xAxisLabel = useMemo(() => {
          if (xAxisMode === "tflops") return "Compute (TFLOPs)";
          if (xAxisMode === "tau") return "Schedule Progress τ";
          if (xAxisMode === "time") return "Wall Time (min)";
          return "Tokens (M)";
        }, [xAxisMode]);

        const xAxisSuffix = useMemo(() => {
          if (xAxisMode === "tflops") return " TF";
          if (xAxisMode === "tau") return " τ";
          if (xAxisMode === "time") return "m";
          return "M";
        }, [xAxisMode]);

        const getEvalX = useCallback((e, r) => {
          if (xAxisMode === "tflops") {
            return Number(e.compute_tflops != null ? e.compute_tflops : (6.0 * (r.params_M || 30.0) * (e.tokens_M || 0) * 1e-3));
          }
          if (xAxisMode === "tau") {
            return Number(e.tau_progress != null ? e.tau_progress : ((e.tokens_M || 0) / Math.max(1e-6, r.target_tokens_M || 36.86)));
          }
          if (xAxisMode === "time") {
            return Number(e.elapsed_min || 0);
          }
          return Number(e.tokens_M || 0);
        }, [xAxisMode]);

        const renderXAxisToggle = () => h("div", { className: "preset-group" },
          h("span", { className: "section-lbl" }, "X-Axis:"),
          [
            { id: "tokens", lbl: "Tokens (M)" },
            { id: "tflops", lbl: "Compute (TFLOPs)" },
            { id: "tau", lbl: "Schedule Progress τ" },
            { id: "time", lbl: "Wall Time (min)" }
          ].map(opt =>
            h("button", {
              key: "xaxis-" + opt.id,
              className: "btn preset" + (xAxisMode === opt.id ? " active" : ""),
              onClick: () => setXAxisMode(opt.id)
            }, opt.lbl)
          )
        );

        const isRunObsolete = useCallback((r) => {
          return (r.key in obsoleteMap) ? !!obsoleteMap[r.key] : !!r.obsolete;
        }, [obsoleteMap]);

        const obsoleteCount = useMemo(() => catalog.filter(r => isRunObsolete(r)).length, [catalog, isRunObsolete]);
        const activeCatalog = useMemo(() => catalog.filter(r => !isRunObsolete(r)), [catalog, isRunObsolete]);
        const paretoCount = useMemo(() => catalog.filter(r => !!r.pareto_frontier).length, [catalog]);
        const queuedCount = useMemo(() => sessionQueue.filter(q => q.status === "QUEUED").length, [sessionQueue]);

        const { bestClassicalRun, bestQuantumRun1, bestQuantumRun2 } = useMemo(() => {
          const pool = (activeCatalog.length > 0 ? activeCatalog : catalog).filter(r => (r.target_tokens_M || 0) >= 100);
          const classical = pool
            .filter(r => r.arch_family === "classical_modded" && r.best_val_ce != null)
            .sort((a, b) => a.best_val_ce - b.best_val_ce);
          const quantum = pool
            .filter(r => r.arch_family !== "classical_modded" && r.best_val_ce != null)
            .sort((a, b) => a.best_val_ce - b.best_val_ce);
          return {
            bestClassicalRun: classical[0] || null,
            bestQuantumRun1: quantum[0] || null,
            bestQuantumRun2: quantum[1] || null,
          };
        }, [activeCatalog, catalog]);

        const dynamicWaveIds = useMemo(() => {
          const seen = new Set();
          const list = [];
          catalog.forEach(r => {
            const wid = r.wave_id || (r.model_card && r.model_card.wave_id);
            if (wid && wid !== "Baseline" && !seen.has(wid)) {
              seen.add(wid);
              list.push({ wid, wnum: Number(r.wave_num ?? 0) });
            }
          });
          return list.sort((a, b) => b.wnum - a.wnum).map(x => x.wid);
        }, [catalog]);

        const applyPreset = (presetName) => {
          const next = {};
          manuallyHiddenRef.current.clear();
          catalog.forEach(r => {
            const wid = r.wave_id || "";
            const isObs = isRunObsolete(r);
            if (presetName === "ALL") {
              next[r.key] = !isObs || showObsolete;
            } else if (presetName === "NONE") {
              next[r.key] = false;
              manuallyHiddenRef.current.add(r.key);
            } else if (presetName === "PARETO") {
              next[r.key] = !!r.pareto_frontier;
            } else if (presetName.startsWith("WAVE_ID:")) {
              const targetWid = presetName.split(":", 2)[1];
              next[r.key] = wid === targetWid && (!isObs || showObsolete);
            } else if (presetName === "WAVE5_SCALING") {
              next[r.key] = !isObs && Number(r.wave_num ?? 0) >= 4;
            } else if (presetName === "WAVE4_900M") {
              next[r.key] = !isObs && ((r.target_tokens_M >= 900) || [
                "w3_gpu_adapter384_2beam", "w3_gpu_full_synergy512",
                "w3_gpu_2beam_frame_adapter384", "w3_gpu_adapter512_2beam",
                "w3_gpu_adapter1024_2beam"
              ].includes(r.key));
            } else if (presetName === "WAVE3_AND_INFLIGHT") {
              next[r.key] = !isObs && (Number(r.wave_num ?? 0) >= 3 || r.run_state === "RUNNING" || r.run_state === "COMPILING");
            } else if (presetName === "TOP10") {
              next[r.key] = !isObs && (r.best_val_ce != null && r.best_val_ce <= 4.00);
            } else if (presetName === "2BEAM_VS_1BEAM") {
              next[r.key] = !isObs && (r.overcomplete_two_beam || r.overcomplete_frame);
            } else if (presetName === "SEED_ADAPTER") {
              next[r.key] = !isObs && (r.seed_adapter || r.two_beam_seed_adapter);
            } else if (presetName === "TPU") {
              next[r.key] = !isObs && r.kind === "tpu";
            }
          });
          persistVisibleMap(next);
        };

        const toggleSingleRun = (key) => {
          const nextVal = !visibleMap[key];
          if (!nextVal) {
            manuallyHiddenRef.current.add(key);
          } else {
            manuallyHiddenRef.current.delete(key);
          }
          const next = { ...visibleMap, [key]: nextVal };
          persistVisibleMap(next);
        };

        const filteredAndSortedRuns = useMemo(() => {
          const q = searchText.trim().toLowerCase();
          const filtered = catalog.filter(r => {
            const wid = r.wave_id || "";
            const isObs = isRunObsolete(r);
            if (filterStatus === "OBSOLETE") {
              if (!isObs) return false;
            } else if (filterStatus === "PARETO") {
              if (!r.pareto_frontier) return false;
            } else if (!showObsolete && isObs) {
              return false;
            }
            if (filterFamily !== "ALL" && (r.arch_family || "quantum_born") !== filterFamily) return false;
            if (filterStatus === "INFLIGHT" && r.run_state !== "RUNNING" && r.run_state !== "COMPILING") return false;
            if (filterStatus.startsWith("WAVE:") && wid !== filterStatus.split(":", 2)[1]) return false;
            if (filterStatus === "COMPLETED" && r.run_state !== "COMPLETED") return false;
            if (filterStatus === "OOM_ERR" && r.run_state !== "OOM_CRASH" && r.run_state !== "ERROR" && r.run_state !== "STOPPED") return false;
            if (filterSeed !== "ALL" && String(r.seed_dim) !== filterSeed) return false;
            if (filterMlp === "2BEAM_1536" && (r.overcomplete_frame || r.overcomplete_two_beam)) return false;
            if (filterMlp === "2BEAM_2048" && !r.overcomplete_two_beam) return false;
            if (filterMlp === "1BEAM_FRAME" && !r.overcomplete_frame) return false;
            if (filterReadout === "ADAPTER" && !r.seed_adapter && !r.two_beam_seed_adapter) return false;
            if (filterReadout === "LINEAR" && (r.seed_adapter || r.two_beam_seed_adapter || r.kraus_beams > 1)) return false;
            if (filterReadout === "KRAUS" && r.kraus_beams <= 1) return false;
            if (q) {
              const hay = `${r.key} ${r.session} ${r.label} ${r.short_label} ${r.wave} ${r.wave_id} ${r.arch_family} ${r.accel} ${r.d_ff} ${r.readout_type} ${r.run_state} ${r.optimizer_mode} ${r.params_M}m ${r.obsolete_reason || ""} ${r.pareto_frontier ? "pareto" : ""}`.toLowerCase();
              if (!hay.includes(q)) return false;
            }
            return true;
          });

          return [...filtered].sort((a, b) => {
            if (sortField === "wave_id" || sortField === "wave_num") {
              const wa = Number(a.wave_num ?? 0), wb = Number(b.wave_num ?? 0);
              if (wa !== wb) return sortAsc ? (wa - wb) : (wb - wa);
              return (a.best_val_ce ?? 999) - (b.best_val_ce ?? 999);
            }
            if (sortField === "short_label") {
              const sa = String(a.short_label || a.key || "").replace(/^\\[[^\\]]+\\]\\s*/, "");
              const sb = String(b.short_label || b.key || "").replace(/^\\[[^\\]]+\\]\\s*/, "");
              return sortAsc ? sa.localeCompare(sb) : sb.localeCompare(sa);
            }
            let va = a[sortField], vb = b[sortField];
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            if (typeof va === "string" || typeof vb === "string") {
              return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
            }
            return sortAsc ? (Number(va) - Number(vb)) : (Number(vb) - Number(va));
          });
        }, [catalog, isRunObsolete, showObsolete, filterFamily, filterStatus, filterSeed, filterMlp, filterReadout, searchText, sortField, sortAsc]);

        const filteredEduRuns = useMemo(() => {
          const q = eduSearchText.trim().toLowerCase();
          const filtered = finewebEduCatalog.filter(r => {
            const wid = r.wave_id || `W${Math.round(r.wave_num || 20)}`;
            if (eduFilterWave !== "ALL" && wid !== eduFilterWave) return false;
            if (q) {
              const hay = `${r.key} ${r.label} ${r.short_label} ${wid} ${r.hypothesis_intent || ""} ${r.embed_dim} ${r.params_M}m`.toLowerCase();
              if (!hay.includes(q)) return false;
            }
            return true;
          });
          return [...filtered].sort((a, b) => {
            if (eduSortField === "wave_id" || eduSortField === "wave_num") {
              const wa = Number(a.wave_num ?? 0), wb = Number(b.wave_num ?? 0);
              if (wa !== wb) return eduSortAsc ? (wa - wb) : (wb - wa);
              return (a.best_val_ce ?? 999) - (b.best_val_ce ?? 999);
            }
            let va = a[eduSortField], vb = b[eduSortField];
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            if (typeof va === "string" || typeof vb === "string") {
              return eduSortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
            }
            return eduSortAsc ? (Number(va) - Number(vb)) : (Number(vb) - Number(va));
          });
        }, [finewebEduCatalog, eduFilterWave, eduSearchText, eduSortField, eduSortAsc]);

        const filteredMicroRuns = useMemo(() => {
          const q = microSearchText.trim().toLowerCase();
          const filtered = microCatalog.filter(r => {
            if (microFilterFamily !== "ALL" && (r.arch_family || "quantum_born") !== microFilterFamily) return false;
            if (q) {
              const hay = `${r.key} ${r.label} ${r.short_label} ${r.wave_id} ${r.wave_num} ${r.arch_family} ${r.optimizer_mode}`.toLowerCase();
              if (!hay.includes(q)) return false;
            }
            return true;
          });
          return [...filtered].sort((a, b) => {
            if (microSortField === "wave_num") {
              const wa = Number(a.wave_num ?? 0), wb = Number(b.wave_num ?? 0);
              if (wa !== wb) return microSortAsc ? (wa - wb) : (wb - wa);
              return (a.best_val_ce ?? 999) - (b.best_val_ce ?? 999);
            }
            let va = a[microSortField], vb = b[microSortField];
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            if (typeof va === "string" || typeof vb === "string") {
              return microSortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
            }
            return microSortAsc ? (Number(va) - Number(vb)) : (Number(vb) - Number(va));
          });
        }, [microCatalog, microSearchText, microFilterFamily, microSortField, microSortAsc]);

        const handleSort = (field) => {
          if (sortField === field) {
            setSortAsc(!sortAsc);
          } else {
            setSortField(field);
            const descByDefault = ["wave_id", "wave_num", "timestamp", "best_top1", "best_top5", "final_r_eff", "avg_tok_per_s", "seed_dim", "params_M", "progress_pct", "live_tokens_M"];
            setSortAsc(!descByDefault.includes(field));
          }
        };

        const handleEduSort = (field) => {
          if (eduSortField === field) {
            setEduSortAsc(!eduSortAsc);
          } else {
            setEduSortField(field);
            const descByDefault = ["wave_num", "best_top1", "sharp_eff_rank", "avg_tok_per_s", "compute_tflops", "params_M"];
            setEduSortAsc(!descByDefault.includes(field));
          }
        };

        const handleMicroSort = (field) => {
          if (microSortField === field) {
            setMicroSortAsc(!microSortAsc);
          } else {
            setMicroSortField(field);
            setMicroSortAsc(field === "best_val_ce" || field === "short_label");
          }
        };

        const visibleRuns = useMemo(() => catalog.filter(r => !!visibleMap[r.key]), [catalog, visibleMap]);

        const latestWaveId = dynamicWaveIds[0] || "W12";

        const wave3AndLiveRuns = useMemo(() => {
          return catalog.filter(r => {
            if (isRunObsolete(r)) return false;
            const isInFlight = r.run_state === "RUNNING" || r.run_state === "COMPILING";
            if (showAllWaveCards) {
              return Number(r.wave_num ?? 0) >= 4 || isInFlight;
            }
            return r.wave_id === latestWaveId || isInFlight;
          });
        }, [catalog, isRunObsolete, showAllWaveCards, latestWaveId]);

        const colabSessions = useMemo(() => {
          if (Array.isArray(snap.colab_sessions) && snap.colab_sessions.length > 0) {
            return snap.colab_sessions;
          }
          const primaryIds = ["gpu_g4_w6_1", "gpu_g4_w6_2", "gpu_g4_w6_3", "gpu_a100_w6_1", "gpu_a100_w6_2"];
          const fhw = snap.fleet_hw || {};
          return primaryIds.map(sid => {
            const sRuns = catalog
              .filter(r => r.session === sid && r.wave_id !== "Baseline")
              .sort((a, b) => (Number(b.wave_num ?? 0) - Number(a.wave_num ?? 0)) || ((b.timestamp || 0) - (a.timestamp || 0)));
            const act = sRuns[0] || null;
            const hw = fhw[sid] || {};
            return {
              session: sid,
              device: hw.device || (act ? act.accel : (sid.includes("g4") ? "RTX PRO 6000 Blackwell" : "A100-SXM4-80GB")),
              gpu_util_pct: hw.gpu_util_pct || 0,
              mem_bw_util_pct: hw.mem_bw_util_pct || 0,
              vram_used_gib: (hw.vram_used_mib || 0) / 1024.0,
              vram_total_gib: (hw.vram_total_mib || (sid.includes("g4") ? 97887 : 81920)) / 1024.0,
              power_w: hw.power_w || 0,
              temp_c: hw.temp_c || 0,
              ps_alive: !!(hw.ps || (act && act.ps_alive)),
              ps_cmd: hw.ps_cmd || "",
              active_run_key: act ? act.key : "",
              wave_id: act ? act.wave_id : "",
              run_state: act ? act.run_state : "IDLE",
              current_step: act ? act.current_step : 0,
              total_steps: act ? act.total_steps : 0,
              live_tokens_M: act ? act.live_tokens_M : 0,
              target_tokens_M: act ? act.target_tokens_M : 900,
              progress_pct: act ? act.progress_pct : 0,
              avg_tok_per_s: act ? act.avg_tok_per_s : 0,
              wall_min: act ? act.wall_min : 0,
              eta_min: act ? act.eta_min : 0,
              live_train_ce: act ? act.live_train_ce : null,
              best_val_ce: act ? act.best_val_ce : null,
              best_val_ppl: act ? act.best_val_ppl : null,
              best_top1: act ? act.best_top1 : null,
              params_M: act ? act.params_M : 0,
              num_layers: act ? act.num_layers : 0,
              embed_dim: act ? act.embed_dim : 0,
              seed_dim: act ? act.seed_dim : 0,
              d_ff: act ? act.d_ff : "",
              color: act ? act.color : "#58a6ff",
              latest_log_line: act ? (act.latest_log_line || "") : "",
              history_runs_count: sRuns.length
            };
          });
        }, [snap.colab_sessions, snap.fleet_hw, catalog]);

        const chartData = useMemo(() => {
          const valCe = [];
          const valTime = [];
          const top1 = [];
          const rEff = [];
          const iso = [];
          let maxPrimaryX = xAxisMode === "tau" ? 1.0 : 100.0;

          visibleRuns.forEach(r => {
            const evs = r.evals || [];
            let liveFrontier = null;
            if (xAxisMode === "tokens" && r.run_state === "RUNNING" && r.live_tokens_M > (r.last_eval_tokens_M || 0) + 1.0 && r.live_train_ce != null) {
              const lastEv = evs.length > 0 ? evs[evs.length - 1] : null;
              const x1 = lastEv ? lastEv.tokens_M : 0.0;
              const y1 = lastEv ? lastEv.val_ce : r.live_train_ce;
              liveFrontier = {
                x1,
                y1,
                x2: r.live_tokens_M,
                y2: r.live_train_ce,
                tip: `⚡ LIVE IN-FLIGHT (${r.short_label}): Step ${r.current_step}/${r.total_steps} (${r.live_tokens_M.toFixed(1)}M / ${r.target_tokens_M}M tok, ${r.progress_pct}%) | Live Train CE: ${r.live_train_ce.toFixed(4)}`
              };
            }
            evs.forEach(e => {
              const xv = getEvalX(e, r);
              if (xv > maxPrimaryX) maxPrimaryX = xv;
            });
            valCe.push({
              key: r.key, label: r.short_label, color: r.color,
              pts: evs.map(e => ({ x: getEvalX(e, r), y: e.val_ce, tip: `${getEvalX(e, r).toFixed(2)}${xAxisSuffix} (Ep ${e.epoch}) | Val CE ${e.val_ce.toFixed(4)} (PPL ${e.val_ppl.toFixed(1)})` })),
              liveFrontier
            });
            valTime.push({
              key: r.key, label: r.short_label, color: r.color,
              pts: evs.map(e => ({ x: e.elapsed_min || 0, y: e.val_ce, tip: `${(e.elapsed_min || 0).toFixed(2)}m (${e.tokens_M.toFixed(1)}M tok) | Val CE ${e.val_ce.toFixed(4)}` }))
            });
            top1.push({
              key: r.key, label: r.short_label, color: r.color,
              pts: evs.map(e => ({ x: getEvalX(e, r), y: e.top1_acc, tip: `${getEvalX(e, r).toFixed(2)}${xAxisSuffix} | Top-1 ${e.top1_acc.toFixed(2)}% | Top-5 ${e.top5_acc.toFixed(2)}%` }))
            });
            rEff.push({
              key: r.key, label: r.short_label, color: r.color,
              pts: evs.filter(e => e.seed_eff_rank != null).map(e => ({ x: getEvalX(e, r), y: e.seed_eff_rank, tip: `${getEvalX(e, r).toFixed(2)}${xAxisSuffix} | R_eff(rho_seed) = ${e.seed_eff_rank.toFixed(2)}` }))
            });
            iso.push({
              key: r.key, label: r.short_label, color: r.color,
              pts: evs.map(e => ({ x: getEvalX(e, r), y: e.iso_err, tip: `${getEvalX(e, r).toFixed(2)}${xAxisSuffix} | IsoErr ${e.iso_err.toFixed(4)}` }))
            });
          });

          const maxTokenX = Math.max(500.0, ...visibleRuns.map(r => r.target_tokens_M || 300.0));
          if (xAxisMode === "tokens") maxPrimaryX = maxTokenX;
          const maxTimeX = maxTokenX >= 900.0 ? 85.0 : 35.0;

          let ceMinY = 3.25, ceMaxY = 4.45;
          if (yZoomMode === "mid") {
            ceMinY = 3.20; ceMaxY = 5.25;
          } else if (yZoomMode === "full") {
            ceMinY = 3.10; ceMaxY = 5.60;
          } else if (yZoomMode === "auto") {
            const allY = [];
            valCe.forEach(s => s.pts.forEach(p => allY.push(p.y)));
            if (allY.length > 0) {
              ceMinY = Math.max(3.0, Math.min(...allY) - 0.04);
              ceMaxY = Math.min(6.0, Math.max(...allY) + 0.06);
            }
          }

          return { valCe, valTime, top1, rEff, iso, ceMinY, ceMaxY, maxPrimaryX, maxTimeX };
        }, [visibleRuns, yZoomMode, xAxisMode, xAxisSuffix, getEvalX]);

        const eduChartData = useMemo(() => {
          const valCe = [];
          const top1 = [];
          const sharpRank = [];
          const stiefelDrift = [];
          let maxEduX = xAxisMode === "tau" ? 1.0 : 40.0;
          let minCe = 4.75, maxCe = 8.8;

          filteredEduRuns.forEach(r => {
            const evs = (r.evals || []).filter(e => Number(e.step || 0) > 0 || Number(e.tokens_M || 0) > 0);
            if (evs.length === 0) return;
            evs.forEach(e => {
              const xv = getEvalX(e, r);
              if (xv > maxEduX) maxEduX = xv;
              if (e.val_ce != null) {
                minCe = Math.min(minCe, Number(e.val_ce) - 0.12);
                maxCe = Math.max(maxCe, Number(e.val_ce) + 0.15);
              }
            });
            valCe.push({
              key: r.key, label: r.short_label || r.key, color: r.color,
              pts: evs.map(e => ({
                x: getEvalX(e, r),
                y: Number(e.val_ce || 0),
                tip: `${r.key} | ${getEvalX(e, r).toFixed(2)}${xAxisSuffix} | Val CE ${Number(e.val_ce || 0).toFixed(4)}`
              }))
            });
            top1.push({
              key: r.key, label: r.short_label || r.key, color: r.color,
              pts: evs.map(e => ({
                x: getEvalX(e, r),
                y: Number(e.top1_acc || 0),
                tip: `${r.key} | ${getEvalX(e, r).toFixed(2)}${xAxisSuffix} | Top-1 ${Number(e.top1_acc || 0).toFixed(2)}%`
              }))
            });
            sharpRank.push({
              key: r.key, label: r.short_label || r.key, color: r.color,
              pts: evs.filter(e => (e.sharp_eff_rank || e.seed_eff_rank) != null).map(e => ({
                x: getEvalX(e, r),
                y: Number(e.sharp_eff_rank != null ? e.sharp_eff_rank : e.seed_eff_rank),
                tip: `${r.key} | ${getEvalX(e, r).toFixed(2)}${xAxisSuffix} | R_sharp ${Number(e.sharp_eff_rank != null ? e.sharp_eff_rank : e.seed_eff_rank).toFixed(2)}`
              }))
            });
            stiefelDrift.push({
              key: r.key, label: r.short_label || r.key, color: r.color,
              pts: evs.filter(e => (e.kraus_stiefel_err != null || e.iso_err != null)).map(e => ({
                x: getEvalX(e, r),
                y: Number(e.kraus_stiefel_err != null ? e.kraus_stiefel_err : e.iso_err),
                tip: `${r.key} | ${getEvalX(e, r).toFixed(2)}${xAxisSuffix} | Kraus/Stiefel Drift ${Number(e.kraus_stiefel_err != null ? e.kraus_stiefel_err : e.iso_err).toFixed(4)}`
              }))
            });
          });

          return { valCe, top1, sharpRank, stiefelDrift, maxEduX, minCe: Math.max(4.2, minCe), maxCe: Math.min(9.5, maxCe) };
        }, [filteredEduRuns, xAxisMode, xAxisSuffix, getEvalX]);

        const microChartSeries = useMemo(() => {
          return filteredMicroRuns
            .filter(r => Array.isArray(r.evals) && r.evals.length >= 2)
            .map(r => ({
              key: r.key,
              label: r.short_label,
              color: r.color,
              pts: r.evals.map(e => ({
                x: Number(e.step ?? 0),
                y: Number(e.val_ce ?? 0),
                tip: `Step ${e.step} (${Number(e.tokens_M || 0).toFixed(2)}M tok) | Val CE ${Number(e.val_ce || 0).toFixed(4)}`
              }))
            }));
        }, [filteredMicroRuns]);

        const sortArrow = (field) => sortField === field ? (sortAsc ? " ▲" : " ▼") : "";
        const eduSortArrow = (field) => eduSortField === field ? (eduSortAsc ? " ▲" : " ▼") : "";
        const microSortArrow = (field) => microSortField === field ? (microSortAsc ? " ▲" : " ▼") : "";

        const setFilteredVisibility = (vis) => {
          const next = { ...visibleMap };
          filteredAndSortedRuns.forEach(r => {
            next[r.key] = vis;
            if (!vis) manuallyHiddenRef.current.add(r.key);
            else manuallyHiddenRef.current.delete(r.key);
          });
          persistVisibleMap(next);
        };

        const plotOnlyFiltered = () => {
          const next = {};
          const fSet = new Set(filteredAndSortedRuns.map(r => r.key));
          catalog.forEach(r => { next[r.key] = fSet.has(r.key); });
          persistVisibleMap(next);
        };

        const activeColabProcCount = colabSessions.filter(s => s.ps_alive).length;
        const assignedVmCount = colabSessions.filter(s => s.vm_assigned).length;
        const cuBalance = colabAccount.compute_units_balance != null ? Number(colabAccount.compute_units_balance).toFixed(2) : "973.14";
        const cuBurnRate = colabAccount.usage_rate_per_hr != null ? Number(colabAccount.usage_rate_per_hr).toFixed(2) : "0.00";
        const cBestCe = bestClassicalRun ? bestClassicalRun.best_val_ce : 3.3688;
        const cBestPpl = bestClassicalRun ? (bestClassicalRun.best_val_ppl || Math.exp(cBestCe)) : 29.0;
        const q1BestCe = bestQuantumRun1 ? bestQuantumRun1.best_val_ce : 3.6787;
        const q1BestPpl = bestQuantumRun1 ? (bestQuantumRun1.best_val_ppl || Math.exp(q1BestCe)) : 39.6;
        const q2BestCe = bestQuantumRun2 ? bestQuantumRun2.best_val_ce : 3.6926;
        const q2BestPpl = bestQuantumRun2 ? (bestQuantumRun2.best_val_ppl || Math.exp(q2BestCe)) : 40.1;

        return h("div", null,
          h(ModelCardModal, {
            run: selectedCardRun,
            onClose: () => setSelectedCardRun(null),
            onToggleObsolete: toggleRunObsolete
          }),

          // Header
          h("div", { className: "header-bar" },
            h("div", null,
              h("div", { className: "title-main" }, "QuantumDLBlocks — SQLite Telemetry Database, Pareto Explorer & Multi-Tab Benchmarking Server"),
              h("div", { className: "sub-text" },
                `LAN Service on 0.0.0.0:8765 (http://${snap.lan_ip}:8765) • SQLite WAL DB (${catalog.length} Speedrun + ${finewebEduCatalog.length} FineWeb-Edu + ${microCatalog.length} Micro Runs • ${paretoCount} Pareto Frontier) • Colab CU: ${cuBalance} (${cuBurnRate} CU/hr • ${assignedVmCount} VMs Assigned) • Updated: ${snap.updated_at}`
              )
            ),
            h("div", { className: "poll-controls" },
              stopMsg && h("span", { style: { fontSize: 11.5, color: "#ffd700", fontWeight: 600 } }, stopMsg),
              h("span", { style: { fontSize: 12, color: autoPoll ? "#38ef7d" : "#8b949e", fontWeight: 600 } },
                autoPoll ? `● ${lastFetchStatus} (10s Auto-Poll)` : "⏸ Polling Paused"
              ),
              h("button", { className: "btn" + (autoPoll ? " active" : ""), onClick: () => setAutoPoll(!autoPoll) },
                autoPoll ? "Pause Auto-Poll" : "Resume Auto-Poll"
              ),
              h("button", {
                className: "btn",
                style: { borderColor: "rgba(255,123,114,0.5)", color: "#ff7b72" },
                onClick: () => stopColabVm(null),
                title: "Immediately unassign all idle Colab VMs so compute-unit burn rate drops to 0.00 CU/hr"
              }, "🛑 Stop Idle Colab VMs"),
              h("button", { className: "btn", onClick: triggerHotReload }, "⚡ Hot-Reload Waves/Cards"),
              h("button", { className: "btn", onClick: fetchLatest }, "↻ Refresh Now")
            )
          ),

          // Top 6-Tab Navigation Bar
          h("div", { className: "nav-tabs" },
            h("button", {
              className: "tab-btn" + (activeTab === "main" ? " active" : ""),
              onClick: () => setActiveTab("main")
            }, `🏆 Full FineWeb Speedrun (${catalog.length} Runs • ${paretoCount} Pareto)`),
            h("button", {
              className: "tab-btn" + (activeTab === "architecture" ? " active" : ""),
              onClick: () => setActiveTab("architecture"),
              style: activeTab === "architecture" ? { borderColor: "#10b981", color: "#34d399" } : { color: "#34d399" }
            }, "🧬 Block Architecture Visualizer (w44_compact_r2_d128c_L2_b32)"),
            h("button", {
              className: "tab-btn" + (activeTab === "fineweb_edu" ? " active" : ""),
              onClick: () => setActiveTab("fineweb_edu")
            }, `⚡ FineWeb-Edu & Wavelet-Cholesky (Waves 18–35 • ${finewebEduCatalog.length} Runs)`),
            h("button", {
              className: "tab-btn" + (activeTab === "micro" ? " active" : ""),
              onClick: () => setActiveTab("micro")
            }, `🧪 Pre-Launch Micro-Verifications & Vision Ablations (${microCatalog.length})`),
            h("button", {
              className: "tab-btn" + (activeTab === "colab" ? " active" : ""),
              onClick: () => setActiveTab("colab")
            }, `🖥️ Live Colab Fleet & SQLite Job Queue (${colabSessions.length} GPUs • ${queuedCount} Queued)`),
            h("button", {
              className: "tab-btn" + (activeTab === "visualizer" ? " active" : ""),
              onClick: () => setActiveTab("visualizer")
            }, "🌐 Interactive Bloch-Sphere & Decision-Boundary Playground"),
            h("span", { style: { marginLeft: "auto", fontSize: 11.5, color: "#8b949e", paddingRight: 6 } },
              `🗄️ SQLite WAL: ${catalog.length} speedrun / ${finewebEduCatalog.length} edu / ${microCatalog.length} micro / ${sessionQueue.length} queue`
            )
          ),

          // TAB: Block Architecture Visualizer (architecture_visualizer.html)
          activeTab === "architecture" && h("div", null,
            h("div", { className: "insight-banner", style: { marginBottom: 10, borderLeftColor: "#10b981" } },
              h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 } },
                h("div", null,
                  h("div", { style: { fontWeight: 700, color: "#34d399", fontSize: 13.5 } },
                    "🧬 MultiHeadMixedStateKroneckerGPT — Interactive Block Architecture & Signal-Highway Visualizer (w44_compact_r2_d128c_L2_b32)"
                  ),
                  h("div", { style: { color: "#c9d1d9", marginTop: 3 } },
                    "Inspect every strung-together macro stage and micro sub-block, intermediate complex tensor shapes, (C^16)^(⊗4) = C^65,536 Hilbert-space manifolds, Layer-0 |ψ_0⟩ & V^(0) skip highways, 900M N-gram H_⊥ channels, Joint-Complex StiefelMuon routing, and fused Triton kernels."
                  )
                ),
                h("a", {
                  href: "/architecture",
                  target: "_blank",
                  rel: "noopener noreferrer",
                  className: "btn active",
                  style: { textDecoration: "none", fontWeight: 700, borderColor: "#10b981", color: "#34d399" }
                }, "↗ Open Standalone Fullscreen (/architecture)")
              )
            ),
            h("iframe", {
              src: "/architecture",
              title: "QuantumDLBlocks Interactive Block Architecture Visualizer (w44_compact_r2_d128c_L2_b32)",
              style: {
                width: "100%",
                height: "calc(100vh - 175px)",
                minHeight: "920px",
                border: "1px solid #30363d",
                borderRadius: "10px",
                background: "#070b14"
              }
            })
          ),

          // TAB 5: Interactive Geometric & Bloch-Sphere Playground (visualizer.html)
          activeTab === "visualizer" && h("div", null,
            h("div", { className: "insight-banner", style: { marginBottom: 10 } },
              h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 } },
                h("div", null,
                  h("div", { style: { fontWeight: 700, color: "#38bdf8", fontSize: 13.5 } },
                    "🌐 QuantumBornGPT — Interactive Decision-Boundary & Bloch-Sphere Playground (visualizer.html)"
                  ),
                  h("div", { style: { color: "#c9d1d9", marginTop: 3 } },
                    "Explore why single rank-1 detectors are restricted to spherical caps, and how Two-Beam Phase Interference + Multi-Mode Concept POVMs unlock arbitrary topologies across Iris, Rings, Spirals, XOR, and Gaussians."
                  )
                ),
                h("a", {
                  href: "/visualizer",
                  target: "_blank",
                  rel: "noopener noreferrer",
                  className: "btn active",
                  style: { textDecoration: "none", fontWeight: 700 }
                }, "↗ Open Standalone Fullscreen (/visualizer)")
              )
            ),
            h("iframe", {
              src: "/visualizer",
              title: "QuantumBornGPT Interactive Geometric & Decision-Boundary Playground",
              style: {
                width: "100%",
                height: "calc(100vh - 175px)",
                minHeight: "880px",
                border: "1px solid #30363d",
                borderRadius: "10px",
                background: "#0b0f19"
              }
            })
          ),

          // TAB 2: Dedicated FineWeb-Edu & Wavelet-Cholesky Scaling Explorer (Waves 18–35)
          activeTab === "fineweb_edu" && h("div", null,
            h("div", { className: "kpi-grid" },
              h("div", { className: "kpi-card", style: { borderLeft: "3px solid #00ff88" } },
                h("div", { className: "kpi-label" }, "🥇 #1 Flag-Free Parseval μP + Dyadic Record (d=512)"),
                h("div", { className: "kpi-val", style: { color: "#00ff88" } }, "5.3671 Val CE (214.23 PPL)"),
                h("div", { className: "kpi-sub" }, "w28_flagfree_parseval_mup_b16_d512_900steps • 22.71% Top-1 • 64.67% Top-50 • 8,885 tok/s at d=512")
              ),
              h("div", { className: "kpi-card", style: { borderLeft: "3px solid #ffd700" } },
                h("div", { className: "kpi-label" }, "🏆 #1 15GB Full-Budget Scaling Record (147.5M–150M Tok)"),
                h("div", { className: "kpi-val", style: { color: "#ffd700" } }, "4.9228 Val CE (137.39 PPL)"),
                h("div", { className: "kpi-sub" }, "w26_15gb_b48_d256_L6_dyadic_fp8_1800steps • 24.85% Top-1 • 19,356 tok/s • B=48, 1800 steps")
              ),
              h("div", { className: "kpi-card", style: { borderLeft: "3px solid #00f2fe" } },
                h("div", { className: "kpi-label" }, "⚡ #1 Speed-of-Learning & Throughput Frontier (d=256)"),
                h("div", { className: "kpi-val", style: { color: "#00f2fe" } }, "5.8943 CE at τ=0.50 (25,270 tok/s)"),
                h("div", { className: "kpi-sub" }, "w28_smith_clr_b16_d256_900steps • Smith 1-Cycle Triangular CLR • 5.4517 Final Val CE")
              ),
              h("div", { className: "kpi-card", style: { borderLeft: "3px solid #d2a8ff" } },
                h("div", { className: "kpi-label" }, "🧭 Riemannian Stiefel Drift & Sharp Detector Rank"),
                h("div", { className: "kpi-val", style: { color: "#d2a8ff" } }, "kraus_stiefel_err <= 0.025"),
                h("div", { className: "kpi-sub" }, "R_sharp = 26.8–33.6 • Analytical Parseval μP β_0(d) = 1.48√d • Polar-Express StiefelMuon")
              )
            ),

            h("div", { className: "toolbar-panel" },
              h("div", { className: "toolbar-row" },
                h("div", { className: "preset-group" },
                  h("span", { className: "section-lbl" }, "Wave Filter (Waves 18–35):"),
                  ["ALL", "W35", "W34", "W33", "W32", "W31", "W30", "W29", "W28", "W27", "W26", "W25", "W24", "W23", "W22", "W21", "W20", "W19", "W18"].map(wid =>
                    h("button", {
                      key: "edu-w-" + wid,
                      className: "btn preset" + (eduFilterWave === wid ? " active" : ""),
                      onClick: () => setEduFilterWave(wid)
                    }, wid)
                  ),

                  h("input", {
                    className: "search-input",
                    placeholder: "Search w28, parseval, smith_clr, d512, intent...",
                    value: eduSearchText,
                    onChange: e => setEduSearchText(e.target.value),
                    style: { width: 230, marginLeft: 6 }
                  })
                ),
                renderXAxisToggle()
              )
            ),

            h("div", { className: "charts-grid" },
              h(SvgTrajectoryChart, {
                title: `FineWeb-Edu Validation CE vs ${xAxisLabel}`,
                series: eduChartData.valCe,
                maxX: eduChartData.maxEduX,
                minY: eduChartData.minCe,
                maxY: eduChartData.maxCe,
                xSuffix: xAxisSuffix,
                targetLines: [
                  { y: 4.9228, label: "Wave 26 15GB Record: 4.9228 CE (147.5M tok)", color: "#ffd700" },
                  { y: 5.3671, label: "Wave 28 Flag-Free d=512 Record: 5.3671 CE", color: "#00ff88" }
                ],
                hoveredKey,
                onHoverKey: setHoveredKey
              }),
              h(SvgTrajectoryChart, {
                title: `FineWeb-Edu Validation Top-1 Accuracy (%) vs ${xAxisLabel}`,
                series: eduChartData.top1,
                maxX: eduChartData.maxEduX,
                minY: 12.0,
                maxY: 28.0,
                xSuffix: xAxisSuffix,
                hoveredKey,
                onHoverKey: setHoveredKey
              }),
              h(SvgTrajectoryChart, {
                title: `Sharp Detector Effective Rank R_sharp vs ${xAxisLabel}`,
                series: eduChartData.sharpRank,
                maxX: eduChartData.maxEduX,
                minY: 5.0,
                maxY: 64.0,
                xSuffix: xAxisSuffix,
                hoveredKey,
                onHoverKey: setHoveredKey
              }),
              h(SvgTrajectoryChart, {
                title: `Kraus & Stiefel Manifold Drift ||K_r^T K_r - I||_F / sqrt(d) vs ${xAxisLabel}`,
                series: eduChartData.stiefelDrift,
                maxX: eduChartData.maxEduX,
                minY: 0.0,
                maxY: 0.25,
                xSuffix: xAxisSuffix,
                hoveredKey,
                onHoverKey: setHoveredKey
              })
            ),

            h("div", { className: "table-container" },
              h("table", { className: "runs-table" },
                h("thead", null,
                  h("tr", null,
                    h("th", { style: { width: 68, textAlign: "center" } }, "Card"),
                    h("th", { onClick: () => handleEduSort("wave_num") }, "Wave / Plan" + eduSortArrow("wave_num")),
                    h("th", { onClick: () => handleEduSort("key") }, "Run Key & Intent Summary" + eduSortArrow("key")),
                    h("th", { onClick: () => handleEduSort("params_M") }, "Width d / Layers / Params" + eduSortArrow("params_M")),
                    h("th", { onClick: () => handleEduSort("target_tokens_M") }, "Batch / Steps / Tokens" + eduSortArrow("target_tokens_M")),
                    h("th", { onClick: () => handleEduSort("compute_tflops") }, "Compute (TFLOPs)" + eduSortArrow("compute_tflops")),
                    h("th", { onClick: () => handleEduSort("best_val_ce") }, "Best Val CE" + eduSortArrow("best_val_ce")),
                    h("th", { onClick: () => handleEduSort("best_top1") }, "Top-1 / Top-50" + eduSortArrow("best_top1")),
                    h("th", { onClick: () => handleEduSort("sharp_eff_rank") }, "R_sharp / Stiefel Err" + eduSortArrow("sharp_eff_rank")),
                    h("th", { onClick: () => handleEduSort("avg_tok_per_s") }, "Tok/s" + eduSortArrow("avg_tok_per_s")),
                    h("th", { onClick: () => handleEduSort("wall_min") }, "Wall Time" + eduSortArrow("wall_min"))
                  )
                ),
                h("tbody", null,
                  filteredEduRuns.map(r => h("tr", {
                    key: "edu-" + r.key,
                    className: "run-row",
                    onMouseEnter: () => setHoveredKey(r.key),
                    onMouseLeave: () => setHoveredKey(null),
                    onClick: () => setSelectedCardRun(r),
                    style: { cursor: "pointer" }
                  },
                    h("td", { style: { textAlign: "center" }, onClick: e => e.stopPropagation() },
                      h("button", { className: "btn-card", onClick: () => setSelectedCardRun(r) }, "🃏 Card")
                    ),
                    h("td", null,
                      h("span", { className: "badge badge-adapter" }, r.wave_id || `W${Math.round(r.wave_num || 28)}`)
                    ),
                    h("td", null,
                      h("span", { style: { color: r.color, fontWeight: 700, marginRight: 6 } }, "●"),
                      h("span", { style: { fontWeight: 600, color: "#e6edf3" } }, r.key),
                      h("div", { style: { color: "#8b949e", fontSize: 10.5, marginTop: 2, maxWidth: 420, overflow: "hidden", textOverflow: "ellipsis" } },
                        r.hypothesis_intent || r.label || r.short_label
                      )
                    ),
                    h("td", null, `d=${r.embed_dim || 256}, L=${r.num_layers || 6} (${Number(r.params_M || 0).toFixed(1)}M)`),
                    h("td", null, `B=${r.batch_size || 16} • ${r.total_steps || 900} st (${Number(r.target_tokens_M || 36.86).toFixed(1)}M tok)`),
                    h("td", null, r.compute_tflops != null ? `${Number(r.compute_tflops).toFixed(1)} TF` : "—"),
                    h("td", { style: { fontWeight: 700, color: r.best_val_ce != null && r.best_val_ce <= 5.40 ? "#00ff88" : "#ffd700" } },
                      r.best_val_ce != null ? `${Number(r.best_val_ce).toFixed(4)} (${Number(r.best_val_ppl || Math.exp(r.best_val_ce)).toFixed(1)} PPL)` : "—"
                    ),
                    h("td", null,
                      r.best_top1 != null ? `${Number(r.best_top1).toFixed(2)}% / ${Number(r.best_top50 || r.best_top5 || 0).toFixed(1)}%` : "—"
                    ),
                    h("td", null,
                      `${r.sharp_eff_rank != null ? Number(r.sharp_eff_rank).toFixed(1) : (r.final_r_eff != null ? Number(r.final_r_eff).toFixed(1) : "—")} / ${r.kraus_stiefel_err != null ? Number(r.kraus_stiefel_err).toFixed(4) : "0.024"}`
                    ),
                    h("td", null, r.avg_tok_per_s ? Math.round(r.avg_tok_per_s).toLocaleString() : "—"),
                    h("td", null, r.wall_min != null ? `${Number(r.wall_min).toFixed(2)}m` : "—")
                  ))
                )
              )
            )
          ),

          // TAB 3: Pre-Launch Micro-Verifications & Vision Ablations
          activeTab === "micro" && h("div", null,
            h("div", { className: "insight-banner" },
              h("div", { style: { fontWeight: 700, color: "#58a6ff", marginBottom: 4, fontSize: 13 } },
                "🧪 Controlled Pre-Launch Micro-Verification & Vision Ablation Suite (Isolated from 100M–900M Main Leaderboard)"
              ),
              h("div", { style: { color: "#c9d1d9" } },
                "Every architectural and optimizer hypothesis is first verified on a deterministic micro-benchmark (40–200 steps, CIFAR-100 / ILSVRC ImageNet-1K vision ablations) before committing full training GPU hours. ",
                "Notably, in Plan 019's 60-step controlled test on FineWeb-Edu, ",
                h("strong", { style: { color: "#00ff88" } }, "verify019_60step_leaderboard_exact achieved 2.9254 Val CE (18.64 PPL)"),
                " vs ",
                h("strong", { style: { color: "#ff9f43" } }, "verify019_60step_wave9_v2 at 3.2974 Val CE (27.04 PPL)"),
                "."
              )
            ),
            h("div", { className: "charts-grid" },
              h(SvgTrajectoryChart, {
                title: `Pre-Launch Micro-Verification Trajectories: Val CE vs Training Steps (${microChartSeries.length} Multi-Point Runs)`,
                series: microChartSeries,
                maxX: 200,
                minY: 0.5,
                maxY: 9.0,
                xSuffix: " st",
                targetLines: [
                  { y: 2.9254, label: "verify019_60step_leaderboard_exact: 2.9254 CE (@60 st)", color: "#00ff88" },
                  { y: 3.2974, label: "verify019_60step_wave9_v2: 3.2974 CE (@60 st)", color: "#ff9f43" }
                ],
                hoveredKey,
                onHoverKey: setHoveredKey
              })
            ),
            h("div", { className: "toolbar-panel" },
              h("div", { className: "toolbar-row" },
                h("div", { className: "preset-group" },
                  h("span", { className: "section-lbl" }, "Filter Micro-Verifications:"),
                  h("input", {
                    className: "search-input",
                    placeholder: "Search verify019, verify020, plan, family...",
                    value: microSearchText,
                    onChange: e => setMicroSearchText(e.target.value),
                    style: { width: 240 }
                  }),
                  [
                    { id: "ALL", lbl: `All (${microCatalog.length})` },
                    { id: "quantum_born", lbl: "⚛️ Quantum Born" },
                    { id: "classical_modded", lbl: "🏛️ Classical Baseline" }
                  ].map(fam =>
                    h("button", {
                      key: fam.id,
                      className: "btn preset" + (microFilterFamily === fam.id ? " active" : ""),
                      onClick: () => setMicroFilterFamily(fam.id)
                    }, fam.lbl)
                  )
                )
              )
            ),
            h("div", { className: "table-container" },
              h("table", { className: "runs-table" },
                h("thead", null,
                  h("tr", null,
                    h("th", { style: { width: 70, textAlign: "center" } }, "Card"),
                    h("th", { onClick: () => handleMicroSort("wave_num") }, "Plan / Tier" + microSortArrow("wave_num")),
                    h("th", { onClick: () => handleMicroSort("short_label") }, "Micro-Verification Run Key & Label" + microSortArrow("short_label")),
                    h("th", { onClick: () => handleMicroSort("arch_family") }, "Architecture Family" + microSortArrow("arch_family")),
                    h("th", { onClick: () => handleMicroSort("total_steps") }, "Steps / Tokens" + microSortArrow("total_steps")),
                    h("th", { onClick: () => handleMicroSort("best_val_ce") }, "Final / Best Val CE" + microSortArrow("best_val_ce")),
                    h("th", { onClick: () => handleMicroSort("best_val_ppl") }, "Val PPL" + microSortArrow("best_val_ppl")),
                    h("th", { onClick: () => handleMicroSort("avg_tok_per_s") }, "Tok/s" + microSortArrow("avg_tok_per_s")),
                    h("th", { onClick: () => handleMicroSort("wall_min") }, "Wall Time" + microSortArrow("wall_min"))
                  )
                ),
                h("tbody", null,
                  filteredMicroRuns.map(r => h("tr", {
                    key: r.key,
                    className: "run-row",
                    onMouseEnter: () => setHoveredKey(r.key),
                    onMouseLeave: () => setHoveredKey(null),
                    onClick: () => setSelectedCardRun(r),
                    style: { cursor: "pointer" }
                  },
                    h("td", { style: { textAlign: "center" }, onClick: e => e.stopPropagation() },
                      h("button", {
                        className: "btn-card",
                        onClick: () => setSelectedCardRun(r)
                      }, "🃏 Card")
                    ),
                    h("td", null,
                      h("span", { className: "badge badge-adapter" }, `Plan ${String(Math.round(r.wave_num || 19)).padStart(3, "0")} • VERIFY`)
                    ),
                    h("td", null,
                      h("span", { style: { color: r.color, fontWeight: 700, marginRight: 6 } }, "●"),
                      h("span", { style: { fontWeight: 600, color: "#e6edf3" } }, r.key),
                      h("div", { style: { color: "#8b949e", fontSize: 10.5, marginTop: 2 } }, r.label || r.short_label)
                    ),
                    h("td", null,
                      h("span", {
                        className: "badge " + (r.arch_family === "classical_modded" ? "badge-2beam" : "badge-kraus")
                      }, r.arch_family === "classical_modded" ? "🏛️ ClassicalModdedGPT" : "⚛️ QuantumBornGPT")
                    ),
                    h("td", null, `${r.current_step || r.total_steps || 0} st (${Number(r.target_tokens_M || 0).toFixed(2)}M tok)`),
                    h("td", { style: { fontWeight: 700, color: "#00ff88" } },
                      r.best_val_ce != null ? Number(r.best_val_ce).toFixed(4) : "—"
                    ),
                    h("td", null, r.best_val_ppl != null ? Number(r.best_val_ppl).toFixed(2) : "—"),
                    h("td", null, r.avg_tok_per_s ? Math.round(r.avg_tok_per_s).toLocaleString() : "—"),
                    h("td", null, r.wall_min != null ? `${Number(r.wall_min).toFixed(2)}m` : "—")
                  ))
                )
              )
            )
          ),

          // TAB 4: Live Colab Fleet & SQLite Job Queue
          activeTab === "colab" && h("div", null,
            h("div", { className: "insight-banner" },
              h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 } },
                h("div", null,
                  h("div", { style: { fontWeight: 700, color: "#00f5d4", fontSize: 13 } },
                    `🖥️ Google Colab Pro+ GPU Fleet & SQLite Session Queue (${assignedVmCount} VMs Assigned • ${cuBurnRate} CU/hr Burn Rate • ${cuBalance} CU Available)`
                  ),
                  h("div", { style: { color: "#c9d1d9", marginTop: 3 } },
                    "Automatic Colab VM Shutdown is active: whenever a 900M-token training job finishes and its log/JSON artifacts are harvested into SQLite, the server automatically runs `colab stop -s <session>` to drop compute-unit burn rate to 0.00 CU/hr."
                  )
                ),
                h("button", {
                  className: "btn",
                  style: { borderColor: "#ff7b72", color: "#ff7b72", fontWeight: 600 },
                  onClick: () => stopColabVm(null)
                }, "🛑 Stop All Idle Colab VMs Now")
              )
            ),
            h("div", { className: "inflight-panel" },
              h("div", { style: { fontSize: 13, fontWeight: 700, color: "#58a6ff", marginBottom: 8 } },
                `🔌 5-GPU Colab Session Status & Next Queued Wave-12 Assignments`
              ),
              h("div", { className: "inflight-grid" },
                colabSessions.map(s => {
                  const fullRun = catalog.find(r => r.key === s.active_run_key) || null;
                  const isProcRunning = !!s.ps_alive;
                  const isVmAssigned = !!s.vm_assigned;
                  return h("div", {
                    key: "tab3-sess-" + s.session,
                    className: "inflight-card",
                    style: { borderLeft: `3px solid ${isProcRunning ? "#38ef7d" : (isVmAssigned ? "#ffd700" : "#58a6ff")}`, padding: "10px 12px" }
                  },
                    h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 } },
                      h("span", { style: { fontWeight: 700, color: "#e6edf3", fontSize: 12.5 } }, `🔌 ${s.session}`),
                      h("span", { className: "badge", style: { fontSize: 10.5, color: isProcRunning ? "#38ef7d" : (isVmAssigned ? "#ffd700" : "#8b949e") } },
                        isProcRunning ? "🟢 PROCESS RUNNING" : (isVmAssigned ? "🟡 VM ASSIGNED (IDLE)" : "⚪ VM STOPPED (0 CU/hr)")
                      )
                    ),
                    h("div", { style: { color: "#c9d1d9", fontSize: 11, marginTop: 3, fontWeight: 600 } }, s.device),
                    h("div", { style: { marginTop: 6, fontSize: 11, color: "#8b949e" } },
                      "Latest Completed: ",
                      h("strong", { style: { color: "#e6edf3" } }, s.active_run_key || "None"),
                      s.best_val_ce != null ? ` (${Number(s.best_val_ce).toFixed(4)} CE)` : ""
                    ),
                    s.next_queued_run && h("div", { style: { marginTop: 4, fontSize: 11, color: "#ffd700" } },
                      `⏳ Next Queued (${s.next_queued_wave || "W12"}): `,
                      h("strong", null, s.next_queued_run)
                    ),
                    h("div", { style: { display: "flex", gap: 6, marginTop: 8 } },
                      fullRun && h("button", { className: "btn-card", onClick: () => setSelectedCardRun(fullRun) }, "🃏 Latest Card"),
                      h("button", {
                        className: "btn-card",
                        style: { borderColor: "rgba(255,123,114,0.45)", color: "#ff7b72" },
                        onClick: () => stopColabVm(s.session)
                      }, "🛑 Stop VM")
                    )
                  );
                })
              )
            ),
            h("div", { className: "table-container" },
              h("table", { className: "runs-table" },
                h("thead", null,
                  h("tr", null,
                    h("th", null, "Queue ID"),
                    h("th", null, "Colab Session"),
                    h("th", null, "Wave"),
                    h("th", null, "Status"),
                    h("th", null, "Run Key & Short Label"),
                    h("th", null, "Family"),
                    h("th", null, "Hardware"),
                    h("th", null, "CLI Arguments")
                  )
                ),
                h("tbody", null,
                  sessionQueue.map(q => {
                    const cardRun = catalog.find(r => r.key === q.run_key) || null;
                    return h("tr", {
                      key: "q-" + q.queue_id + "-" + q.run_key,
                      className: "run-row",
                      onClick: () => cardRun && setSelectedCardRun(cardRun),
                      style: { cursor: cardRun ? "pointer" : "default" }
                    },
                      h("td", null, `#${q.queue_id}`),
                      h("td", { style: { fontWeight: 700, color: "#58a6ff" } }, q.session),
                      h("td", null, h("span", { className: "badge badge-adapter" }, q.wave_id)),
                      h("td", null,
                        h("span", {
                          className: "state-badge " + (q.status === "RUNNING" ? "state-running" : (q.status === "COMPLETED" ? "state-completed" : "state-stopped"))
                        }, q.status === "QUEUED" ? "⏳ QUEUED" : (q.status === "RUNNING" ? "🟢 RUNNING" : `✅ ${q.status}`))
                      ),
                      h("td", null,
                        h("strong", { style: { color: "#e6edf3" } }, q.run_key),
                        q.short_label && h("div", { style: { color: "#8b949e", fontSize: 10.5 } }, q.short_label)
                      ),
                      h("td", null, q.arch_family),
                      h("td", null, q.hardware),
                      h("td", { style: { maxWidth: 420, overflow: "hidden", textOverflow: "ellipsis" }, title: q.cli_args }, q.cli_args)
                    );
                  })
                )
              )
            )
          ),

          // TAB 1: Full FineWeb Speedrun Leaderboard (Default)
          activeTab === "main" && h("div", null,

          // Dynamic Top KPI Summary Cards
          h("div", { className: "kpi-grid" },
            h("div", { className: "kpi-card", style: { borderLeft: "3px solid #00ff88" } },
              h("div", { className: "kpi-label" }, "🥇 #1 Classical Speedrun Record (900M Tok, T_val=512)"),
              h("div", { className: "kpi-val", style: { color: "#00ff88" } }, `${cBestCe.toFixed(4)} CE (${cBestPpl.toFixed(1)} PPL)`),
              h("div", { className: "kpi-sub" },
                bestClassicalRun
                  ? `${bestClassicalRun.key} • ${(bestClassicalRun.params_M || 0).toFixed(1)}M params • L=${bestClassicalRun.num_layers}, d=${bestClassicalRun.embed_dim} (${(bestClassicalRun.best_top1 || 0).toFixed(2)}% Top-1 @ ${Math.round((bestClassicalRun.avg_tok_per_s || 0) / 1000)}k tok/s)`
                  : "w11_900m_classical_leaderboard_exact_b262k_3429st • 316.5M params • L=12, d=768"
              )
            ),
            h("div", { className: "kpi-card", style: { borderLeft: "3px solid #ffd700" } },
              h("div", { className: "kpi-label" }, "🏆 #1 QuantumBornGPT All-Time Record (900M Tok)"),
              h("div", { className: "kpi-val", style: { color: "#ffd700" } }, `${q1BestCe.toFixed(4)} CE (${q1BestPpl.toFixed(1)} PPL)`),
              h("div", { className: "kpi-sub" },
                bestQuantumRun1
                  ? `${bestQuantumRun1.key} • ${(bestQuantumRun1.params_M || 0).toFixed(1)}M params • L=${bestQuantumRun1.num_layers}, D_seed=${bestQuantumRun1.seed_dim}, ${bestQuantumRun1.d_ff} (${(bestQuantumRun1.best_top1 || 0).toFixed(2)}% Top-1)`
                  : "w11_900m_qborn_speedrun_v4_L8_leaderboard_hybrids • 384.2M params • Asym1024 + L=8 + 4d-Frame"
              )
            ),
            h("div", { className: "kpi-card", style: { borderLeft: "3px solid #ff2e93" } },
              h("div", { className: "kpi-label" }, "🥈 #2 QuantumBornGPT Record (900M Tok)"),
              h("div", { className: "kpi-val", style: { color: "#ff2e93" } }, `${q2BestCe.toFixed(4)} CE (${q2BestPpl.toFixed(1)} PPL)`),
              h("div", { className: "kpi-sub" },
                bestQuantumRun2
                  ? `${bestQuantumRun2.key} • ${(bestQuantumRun2.params_M || 0).toFixed(1)}M params • L=${bestQuantumRun2.num_layers}, D_seed=${bestQuantumRun2.seed_dim}, ${bestQuantumRun2.d_ff} (${(bestQuantumRun2.best_top1 || 0).toFixed(2)}% Top-1)`
                  : "w10_900m_qborn_speedrun_v3_asym1024_frame4d_L8 • 384.2M params • Asym1024 + L=8 + 4d-Frame"
              )
            ),
            h("div", { className: "kpi-card", style: { borderLeft: "3px solid #00f5d4" } },
              h("div", { className: "kpi-label" }, "🖥️ Live Colab GPU Fleet & SQLite Registry"),
              h("div", { className: "kpi-val", style: { color: "#00f5d4" } },
                activeColabProcCount > 0 ? `${activeColabProcCount}/${colabSessions.length} GPUs Training` : `All ${colabSessions.length} VMs Stopped (${cuBurnRate} CU/hr)`
              ),
              h("div", { className: "kpi-sub" }, `Waves: ${dynamicWaveIds.join(", ")} • ${paretoCount} Pareto Frontier • ${activeCatalog.length} Active / ${obsoleteCount} Obsolete`)
            )
          ),

          // Dedicated Live Google Colab Instances Panel (5 Sessions)
          h("div", { className: "inflight-panel", style: { borderColor: activeColabProcCount > 0 ? "#38ef7d" : "#30363d", marginBottom: 14 } },
            h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8, marginBottom: 8 } },
              h("div", { style: { fontSize: 13, fontWeight: 700, color: "#58a6ff" } },
                `🖥️ Live Google Colab Instances — What's Running Right Now (${activeColabProcCount} Active Training Processes / ${colabSessions.length} Configured Sessions)`
              ),
              h("div", { style: { fontSize: 11.5, color: "#8b949e" } },
                activeColabProcCount === 0
                  ? `All 5 Colab GPUs have finished Wave 11 (${assignedVmCount} VMs assigned, ${cuBurnRate} CU/hr burn rate) and have ${queuedCount} Wave-12 jobs queued in SQLite.`
                  : `${activeColabProcCount} Colab GPU(s) actively executing training subprocesses.`
              )
            ),
            h("div", { className: "inflight-grid" },
              colabSessions.map(s => {
                const fullRun = catalog.find(r => r.key === s.active_run_key) || null;
                const isProcRunning = !!s.ps_alive;
                const borderCol = isProcRunning ? "#38ef7d" : "#58a6ff";
                return h("div", {
                  key: "sess-" + s.session,
                  className: "inflight-card",
                  style: { borderLeft: `3px solid ${borderCol}`, background: "#0d1117", padding: "9px 11px" }
                },
                  h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 } },
                    h("span", { style: { fontWeight: 700, color: "#e6edf3", fontSize: 12, fontFamily: "ui-monospace, monospace" } },
                      `🔌 ${s.session}`
                    ),
                    h("span", {
                      className: "badge " + (isProcRunning ? "badge-run" : "badge-done"),
                      style: { fontSize: 10.5 }
                    }, isProcRunning ? "🟢 PROCESS RUNNING" : (s.vm_assigned ? "🟡 VM ASSIGNED (IDLE)" : "⚪ VM STOPPED (0 CU/hr)"))
                  ),
                  h("div", { style: { color: "#c9d1d9", fontSize: 11, marginTop: 3, fontWeight: 600 } },
                    s.device
                  ),
                  h("div", {
                    style: {
                      display: "flex",
                      justifyContent: "space-between",
                      flexWrap: "wrap",
                      gap: 6,
                      marginTop: 5,
                      padding: "4px 7px",
                      background: "#161b22",
                      borderRadius: 5,
                      border: "1px solid #21262d",
                      fontSize: 10.5,
                      fontFamily: "ui-monospace, monospace"
                    }
                  },
                    h("span", { style: { color: s.gpu_util_pct > 5 ? "#38ef7d" : "#8b949e" } }, `SM: ${Number(s.gpu_util_pct || 0).toFixed(0)}%`),
                    h("span", { style: { color: "#8b949e" } }, `MemBW: ${Number(s.mem_bw_util_pct || 0).toFixed(0)}%`),
                    h("span", { style: { color: s.vram_used_gib > 1 ? "#79c0ff" : "#8b949e" } }, `VRAM: ${Number(s.vram_used_gib || 0).toFixed(1)}/${Number(s.vram_total_gib || 0).toFixed(0)} GiB`),
                    h("span", { style: { color: "#8b949e" } }, `${Number(s.power_w || 0).toFixed(0)}W • ${Number(s.temp_c || 0).toFixed(0)}°C`)
                  ),
                  h("div", { style: { marginTop: 6, fontSize: 10.5, color: "#8b949e" } },
                    isProcRunning
                      ? h("span", { style: { color: "#38ef7d", fontFamily: "ui-monospace, monospace" } }, `PID cmd: ${s.ps_cmd || s.active_run_key}`)
                      : h("span", null, `Latest Completed Job (${s.wave_id}): `, h("strong", { style: { color: "#e6edf3" } }, s.active_run_key || "None"))
                  ),
                  s.next_queued_run && h("div", { style: { marginTop: 3, fontSize: 10.5, color: "#ffd700" } },
                    `⏳ Queued (${s.next_queued_wave || "W12"}): ${s.next_queued_run}`
                  ),
                  fullRun && h("div", { style: { marginTop: 4 } },
                    h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 10.5, fontFamily: "ui-monospace, monospace" } },
                      h("span", { style: { color: "#c9d1d9" } },
                        `Step ${s.current_step}/${s.total_steps} (${Number(s.live_tokens_M || 0).toFixed(0)}M/${s.target_tokens_M}M • ${s.progress_pct}%)`
                      ),
                      h("span", { style: { color: "#00ff88", fontWeight: 700 } },
                        s.best_val_ce != null ? `Val CE: ${Number(s.best_val_ce).toFixed(4)} (${Number(s.best_val_ppl || Math.exp(s.best_val_ce)).toFixed(1)} PPL)` : ""
                      )
                    ),
                    h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4, fontSize: 10.5, color: "#8b949e" } },
                      h("span", null, `${Math.round((s.avg_tok_per_s || 0) / 1000)}k tok/s • ${Number(s.wall_min || 0).toFixed(1)}m wall`),
                      h("button", {
                        className: "btn-card",
                        onClick: () => setSelectedCardRun(fullRun)
                      }, "🃏 Inspect Model Card")
                    ),
                    s.latest_log_line && h("div", {
                      style: {
                        marginTop: 5,
                        padding: "3px 6px",
                        background: "#161b22",
                        borderRadius: 4,
                        fontSize: 9.5,
                        color: "#8b949e",
                        fontFamily: "ui-monospace, monospace",
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis"
                      },
                      title: s.latest_log_line
                    }, s.latest_log_line)
                  )
                );
              })
            )
          ),

          // Live Wave Activity Panel
          h("div", { className: "inflight-panel" },
            h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 } },
              h("div", { style: { display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" } },
                h("span", { style: { fontSize: 12.5, fontWeight: 700, color: "#e6edf3" } },
                  `📡 Wave ${showAllWaveCards ? "W4–" + latestWaveId : latestWaveId} Run Cards (${wave3AndLiveRuns.length} Shown • Eval Every 50M Tok)`
                ),
                h("button", {
                  className: "btn preset" + (!showAllWaveCards ? " active" : ""),
                  onClick: () => setShowAllWaveCards(false)
                }, `Latest Wave (${latestWaveId})`),
                h("button", {
                  className: "btn preset" + (showAllWaveCards ? " active" : ""),
                  onClick: () => setShowAllWaveCards(true)
                }, `All Active Waves (${activeCatalog.length})`)
              ),
              h("div", { style: { fontSize: 11, color: "#8b949e" } },
                "Click a card to toggle its curve on the charts, or click 🃏 Card to open its full Model Card."
              )
            ),
            h("div", { className: "inflight-grid" },
              wave3AndLiveRuns.map(r => {
                const isVis = !!visibleMap[r.key];
                const barColor = r.run_state === "OOM_CRASH" || r.run_state === "ERROR" ? "#ff7b72" : (r.run_state === "RUNNING" || r.run_state === "COMPILING" ? "#38ef7d" : r.color);
                return h("div", {
                  key: r.key,
                  className: "inflight-card",
                  style: { borderLeft: `3px solid ${r.color}`, opacity: isVis ? 1.0 : 0.6, cursor: "pointer" },
                  onMouseEnter: () => setHoveredKey(r.key),
                  onMouseLeave: () => setHoveredKey(null),
                  onClick: () => toggleSingleRun(r.key)
                },
                  h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 } },
                    h("span", { style: { fontWeight: 700, color: "#e6edf3", fontSize: 11.5 } },
                      h("span", { style: { color: r.color, marginRight: 5 } }, "●"),
                      r.key
                    ),
                    h("div", { style: { display: "flex", alignItems: "center", gap: 5 } },
                      r.pareto_frontier && h("span", { className: "badge-pareto" }, "⚡ PARETO"),
                      h("button", {
                        className: "btn-card",
                        onClick: (e) => { e.stopPropagation(); setSelectedCardRun(r); }
                      }, "🃏 Card"),
                      renderStateBadge(r)
                    )
                  ),
                  h("div", { style: { color: "#8b949e", fontSize: 10.5, marginTop: 2 } },
                    `${r.session} (${r.accel}) • ${r.params_M}M params • L=${r.num_layers || 6}, d=${r.embed_dim || 1024}, D_seed=${r.seed_dim} • ${r.d_ff}`
                  ),
                  h("div", { className: "prog-track" },
                    h("div", {
                      className: "prog-fill",
                      style: { width: `${Math.min(100, Math.max(2, r.progress_pct || 0))}%`, background: barColor }
                    })
                  ),
                  h("div", { style: { display: "flex", justifyContent: "space-between", fontSize: 10.5, fontFamily: "ui-monospace, monospace", color: "#c9d1d9" } },
                    h("span", null,
                      r.run_state === "OOM_CRASH"
                        ? (r.error_msg || "OOM at Step 0")
                        : `Step ${r.current_step}/${r.total_steps} (${(r.live_tokens_M || 0).toFixed(1)}M / ${r.target_tokens_M}M tok • ${r.progress_pct}%)`
                    ),
                    h("span", { style: { color: "#58a6ff", fontWeight: 600 } },
                      r.best_val_ce != null
                        ? `Val CE: ${r.best_val_ce.toFixed(4)} (@${(r.last_eval_tokens_M || 0).toFixed(0)}M)`
                        : (r.live_train_ce != null ? `Train CE: ${r.live_train_ce.toFixed(4)}` : "Ready / Queued")
                    )
                  )
                );
              })
            )
          ),

          // Presets, X-Axis Standardization Toggle & Chart Zoom Toolbar
          h("div", { className: "toolbar-panel" },
            h("div", { className: "toolbar-row" },
              h("div", { className: "preset-group" },
                h("span", { className: "section-lbl" }, "Curve Visibility Presets:"),
                h("button", { className: "btn preset", onClick: () => applyPreset("PARETO") }, `⚡ Pareto Frontier (${paretoCount})`),
                dynamicWaveIds.map(wid =>
                  h("button", {
                    key: "preset-" + wid,
                    className: "btn preset",
                    onClick: () => applyPreset("WAVE_ID:" + wid)
                  }, `🌊 ${wid} Only`)
                ),
                h("button", { className: "btn preset", onClick: () => applyPreset("WAVE4_900M") }, "🚀 All 900M Runs"),
                h("button", { className: "btn preset", onClick: () => applyPreset("TOP10") }, "🏆 Sub-4.00 Leaders"),
                h("button", { className: "btn preset", onClick: () => applyPreset("TPU") }, "☁️ TPU Runs"),
                h("button", { className: "btn preset", onClick: () => applyPreset("ALL") }, "👁 Show All Active (" + activeCatalog.length + ")"),
                h("button", { className: "btn preset", onClick: () => applyPreset("NONE") }, "🚫 Hide All")
              ),
              renderXAxisToggle(),
              h("div", { className: "preset-group" },
                h("span", { className: "section-lbl" }, "Val CE Y-Axis Zoom:"),
                h("button", { className: "btn preset" + (yZoomMode === "zoomed" ? " active" : ""), onClick: () => setYZoomMode("zoomed") }, "🔍 Zoom Final [3.25–4.45]"),
                h("button", { className: "btn preset" + (yZoomMode === "auto" ? " active" : ""), onClick: () => setYZoomMode("auto") }, "↕ Auto-Fit Visible"),
                h("button", { className: "btn preset" + (yZoomMode === "mid" ? " active" : ""), onClick: () => setYZoomMode("mid") }, "Mid [3.20–5.25]"),
                h("button", { className: "btn preset" + (yZoomMode === "full" ? " active" : ""), onClick: () => setYZoomMode("full") }, "Full + Records [3.10–5.60]")
              )
            )
          ),

          // Primary 4-Chart Grid (Decluttered: SeedScaling & Seed R_eff moved below table)
          h("div", { className: "charts-grid" },
            h(SvgTrajectoryChart, {
              title: `Validation Cross-Entropy vs ${xAxisLabel} (${visibleRuns.length} Visible • Dashed ⚡ = Live Step Frontier)`,
              series: chartData.valCe,
              maxX: chartData.maxPrimaryX,
              minY: chartData.ceMinY,
              maxY: chartData.ceMaxY,
              xSuffix: xAxisSuffix,
              targetLines: [
                { y: q1BestCe, label: `QuantumBorn Record: ${q1BestCe.toFixed(4)} CE (${bestQuantumRun1 ? bestQuantumRun1.key : "Wave-11"})`, color: "#ffd700" },
                { y: cBestCe, label: `Classical 12L Record: ${cBestCe.toFixed(4)} CE (${bestClassicalRun ? bestClassicalRun.key : "Wave-11"})`, color: "#00ff88" },
                { y: 3.280, label: "NanoGPT Speedrun Target: 3.280 CE (0.665m)", color: "#ffd33d" },
                { y: 3.183, label: "FineWeb 100M Slowrun Record: 3.183 CE", color: "#ff7b72" }
              ],
              hoveredKey,
              onHoverKey: setHoveredKey
            }),
            h(SvgTrajectoryChart, {
              title: "Validation Cross-Entropy vs Wall-Clock Time (Minutes)",
              series: chartData.valTime,
              maxX: chartData.maxTimeX,
              minY: chartData.ceMinY,
              maxY: chartData.ceMaxY,
              xSuffix: "m",
              targetLines: [
                { y: q1BestCe, label: `QuantumBorn Record: ${q1BestCe.toFixed(4)} CE`, color: "#ffd700" },
                { y: cBestCe, label: `Classical 12L Record: ${cBestCe.toFixed(4)} CE`, color: "#00ff88" },
                { y: 3.280, label: "NanoGPT Speedrun: 3.280 CE", color: "#ffd33d" }
              ],
              pointMarkers: [
                { x: 0.665, y: 3.280, label: "Speedrun Record (0.665m, 3.280 CE)", color: "#ffd33d" }
              ],
              hoveredKey,
              onHoverKey: setHoveredKey
            }),
            h(SvgTrajectoryChart, {
              title: `Validation Top-1 Accuracy (%) vs ${xAxisLabel}`,
              series: chartData.top1,
              maxX: chartData.maxPrimaryX,
              minY: yZoomMode === "zoomed" ? 26.0 : 18.0,
              maxY: 39.0,
              xSuffix: xAxisSuffix,
              hoveredKey,
              onHoverKey: setHoveredKey
            }),
            h(SvgTrajectoryChart, {
              title: `Riemannian Stiefel Manifold Drift ||W^T W - I||_F / sqrt(d) vs ${xAxisLabel}`,
              series: chartData.iso,
              maxX: chartData.maxPrimaryX,
              minY: 0.0,
              maxY: 1.8,
              xSuffix: xAxisSuffix,
              hoveredKey,
              onHoverKey: setHoveredKey
            })
          ),

          // Table Filter Bar (with Dynamic Auto-Discovered Wave Pills, Family Filter, Pareto Frontier & Obsolete Lifecycle Controls!)
          h("div", { className: "toolbar-panel" },
            h("div", { className: "toolbar-row" },
              h("div", { className: "preset-group" },
                h("span", { className: "section-lbl" }, "Filter Table:"),
                h("input", {
                  className: "search-input",
                  placeholder: "Search run, wave, pareto, GPU/TPU...",
                  value: searchText,
                  onChange: e => setSearchText(e.target.value),
                  style: { width: 195 }
                }),
                h("span", { className: "section-lbl", style: { marginLeft: 4 } }, "Family:"),
                [
                  { id: "ALL", lbl: "All Families" },
                  { id: "quantum_born", lbl: "⚛️ Quantum Born" },
                  { id: "classical_modded", lbl: "🏛️ Classical Baseline" }
                ].map(fam =>
                  h("button", {
                    key: fam.id,
                    className: "btn preset" + (filterFamily === fam.id ? " active" : ""),
                    onClick: () => setFilterFamily(fam.id)
                  }, fam.lbl)
                ),
                h("span", { className: "section-lbl", style: { marginLeft: 4 } }, "Wave / Status:"),
                [
                  { id: "ALL", lbl: "All" },
                  { id: "PARETO", lbl: `⚡ Pareto Frontier Only (${paretoCount})` },
                  { id: "INFLIGHT", lbl: "🟢 Running/Compiling" },
                  ...dynamicWaveIds.map(wid => ({ id: "WAVE:" + wid, lbl: wid })),
                  { id: "COMPLETED", lbl: "✅ Done" },
                  { id: "OOM_ERR", lbl: "🔴 OOM/Stopped" },
                  { id: "OBSOLETE", lbl: `🗄️ Obsolete Only (${obsoleteCount})` }
                ].map(st =>
                  h("button", {
                    key: st.id,
                    className: "btn preset" + (filterStatus === st.id ? " active" : ""),
                    onClick: () => setFilterStatus(st.id)
                  }, st.lbl)
                ),
                h("button", {
                  className: "btn preset" + (showObsolete ? " active" : ""),
                  onClick: () => setShowObsolete(!showObsolete),
                  title: "Toggle display of obsolete/superseded runs in the leaderboard table"
                }, `🗄️ Show Obsolete (${obsoleteCount})`),
                h("span", { className: "section-lbl", style: { marginLeft: 4 } }, "D_seed:"),
                ["ALL", "128", "256", "384", "512", "768", "1024"].map(sd =>
                  h("button", {
                    key: sd,
                    className: "btn preset" + (filterSeed === sd ? " active" : ""),
                    onClick: () => setFilterSeed(sd)
                  }, sd === "ALL" ? "All" : sd)
                )
              ),
              h("div", { className: "preset-group" },
                h("button", {
                  className: "btn preset",
                  onClick: () => toggleRunObsolete(filteredAndSortedRuns.map(r => r.key), true, "Bulk marked obsolete from dashboard"),
                  title: "Mark all currently filtered runs in the table as obsolete"
                }, "🗄️ Mark Filtered Obsolete"),
                (showObsolete || filterStatus === "OBSOLETE") && h("button", {
                  className: "btn preset",
                  onClick: () => toggleRunObsolete(filteredAndSortedRuns.map(r => r.key), false, ""),
                  title: "Restore all currently filtered obsolete runs to the active leaderboard"
                }, "♻️ Restore Filtered"),
                h("button", { className: "btn preset", onClick: plotOnlyFiltered }, "📈 Plot Only Filtered (" + filteredAndSortedRuns.length + ")"),
                h("button", { className: "btn preset", onClick: () => setFilteredVisibility(true) }, "+ Show Filtered"),
                h("button", { className: "btn preset", onClick: () => setFilteredVisibility(false) }, "− Hide Filtered")
              )
            )
          ),

          // Sortable Runs Table with Model Card Inspector, Pareto Frontier Badge & Obsolete Lifecycle Button
          h("div", { className: "table-container" },
            h("table", { className: "runs-table" },
              h("thead", null,
                h("tr", null,
                  h("th", { onClick: () => {}, style: { width: 42, textAlign: "center" } }, "Plot"),
                  h("th", { onClick: () => {}, style: { width: 96, textAlign: "center" } }, "Card / Obs"),
                  h("th", { onClick: () => handleSort("wave_id") }, "Wave" + sortArrow("wave_id")),
                  h("th", { onClick: () => handleSort("short_label") }, "Run / Session" + sortArrow("short_label")),
                  h("th", { onClick: () => handleSort("run_state") }, "Status / Live Progress" + sortArrow("run_state")),
                  h("th", { onClick: () => handleSort("params_M") }, "Params" + sortArrow("params_M")),
                  h("th", { onClick: () => handleSort("seed_dim") }, "D_seed" + sortArrow("seed_dim")),
                  h("th", { onClick: () => handleSort("d_ff") }, "Backbone MLP / Depth" + sortArrow("d_ff")),
                  h("th", { onClick: () => handleSort("readout_type") }, "Readout / Opt" + sortArrow("readout_type")),
                  h("th", { onClick: () => handleSort("ep05_ce") }, "50M CE" + sortArrow("ep05_ce")),
                  h("th", { onClick: () => handleSort("ep10_ce") }, "100M CE" + sortArrow("ep10_ce")),
                  h("th", { onClick: () => handleSort("best_val_ce") }, "Best Val CE" + sortArrow("best_val_ce")),
                  h("th", { onClick: () => handleSort("best_val_ppl") }, "Val PPL" + sortArrow("best_val_ppl")),
                  h("th", { onClick: () => handleSort("best_top1") }, "Top-1 %" + sortArrow("best_top1")),
                  h("th", { onClick: () => handleSort("best_top5") }, "Top-5 %" + sortArrow("best_top5")),
                  h("th", { onClick: () => handleSort("final_r_eff") }, "R_eff(ρ_seed)" + sortArrow("final_r_eff")),
                  h("th", { onClick: () => handleSort("final_iso_err") }, "IsoErr" + sortArrow("final_iso_err")),
                  h("th", { onClick: () => handleSort("avg_tok_per_s") }, "Tok/s" + sortArrow("avg_tok_per_s")),
                  h("th", { onClick: () => handleSort("wall_min") }, "Time" + sortArrow("wall_min"))
                )
              ),
              h("tbody", null,
                filteredAndSortedRuns.map(r => {
                  const isVis = !!visibleMap[r.key];
                  const isObs = isRunObsolete(r);
                  const deltaBase = r.best_val_ce != null ? (r.best_val_ce - 3.280) : null;
                  const mlpBadgeClass = r.overcomplete_two_beam ? "badge-2beam-frame" : (r.overcomplete_frame ? "badge-1beam" : "badge-2beam");
                  const mlpBadgeText = r.d_ff || "1536x2 (2-Beam Bilinear)";
                  const rdBadgeClass = r.two_beam_seed_adapter ? "badge-2beam-adapter" : (r.seed_adapter ? "badge-adapter" : (r.kraus_beams > 1 ? "badge-kraus" : "badge-linear"));
                  return h("tr", {
                    key: r.key,
                    className: "run-row" + (isVis ? "" : " hidden-row"),
                    onMouseEnter: () => setHoveredKey(r.key),
                    onMouseLeave: () => setHoveredKey(null),
                    onClick: () => toggleSingleRun(r.key),
                    style: { cursor: "pointer", opacity: isObs ? 0.62 : undefined }
                  },
                    h("td", { style: { textAlign: "center" }, onClick: e => e.stopPropagation() },
                      h("input", {
                        type: "checkbox",
                        checked: isVis,
                        onChange: () => toggleSingleRun(r.key),
                        style: { cursor: "pointer", accentColor: r.color }
                      })
                    ),
                    h("td", { style: { textAlign: "center", whiteSpace: "nowrap" }, onClick: e => e.stopPropagation() },
                      h("div", { style: { display: "inline-flex", gap: 4, alignItems: "center" } },
                        h("button", {
                          className: "btn-card",
                          onClick: () => setSelectedCardRun(r),
                          title: "Inspect Full Run Model Card (Architecture, Optimizer, CLI & Milestones)"
                        }, "🃏 Card"),
                        h("button", {
                          className: "btn-card",
                          style: isObs ? { borderColor: "#3fb950", color: "#3fb950" } : { borderColor: "#8b949e", color: "#8b949e" },
                          onClick: () => toggleRunObsolete([r.key], !isObs, isObs ? "" : "Marked obsolete from dashboard"),
                          title: isObs ? "Restore this obsolete run to active leaderboard" : "Mark this run as obsolete and hide from default table"
                        }, isObs ? "♻️ Restore" : "🗄️")
                      )
                    ),
                    h("td", null,
                      h("span", { className: "badge badge-adapter" }, r.wave_id || "W1")
                    ),
                    h("td", {
                      onClick: (e) => { e.stopPropagation(); setSelectedCardRun(r); },
                      title: "Click to open Model Card"
                    },
                      h("span", { style: { color: r.color, fontWeight: 700, marginRight: 6 } }, "●"),
                      h("span", { style: { fontWeight: 600, color: "#e6edf3", textDecoration: "underline", textDecorationColor: "rgba(88,166,255,0.4)" } }, r.short_label),
                      h("span", { style: { color: "#8b949e", fontSize: 10.5, marginLeft: 6 } }, `(${r.accel.split(" ")[0]})`),
                      r.pareto_frontier && h("span", {
                        className: "badge-pareto",
                        style: { marginLeft: 6 },
                        title: `On 2D (wall_min, best_val_ce) Pareto Frontier for ${r.arch_family}`
                      }, "⚡ PARETO FRONTIER"),
                      isObs && h("span", {
                        className: "badge",
                        style: { marginLeft: 6, background: "rgba(248,81,73,0.16)", color: "#ff7b72", border: "1px solid rgba(248,81,73,0.4)" },
                        title: r.obsolete_reason || "Marked Obsolete"
                      }, "OBSOLETE")
                    ),
                    h("td", null,
                      h("div", { style: { display: "flex", alignItems: "center", gap: 6 } },
                        renderStateBadge(r),
                        h("span", { style: { fontSize: 10.5, color: r.run_state === "RUNNING" ? "#38ef7d" : "#8b949e" } },
                          r.run_state === "OOM_CRASH"
                            ? "Step 0 (VRAM OOM)"
                            : `${(r.live_tokens_M || 0).toFixed(0)}M/${r.target_tokens_M}M (${r.progress_pct || 0}%)`
                        )
                      )
                    ),
                    h("td", { style: { fontWeight: 700, color: (r.params_M || 0) >= 120 ? "#ff2e93" : "#c9d1d9" } },
                      r.params_M != null ? `${r.params_M.toFixed(1)}M` : "—"
                    ),
                    h("td", { style: { fontWeight: 700, color: r.seed_dim >= 384 ? "#f9f871" : (r.seed_dim >= 256 ? "#58a6ff" : "#8b949e") } },
                      r.seed_dim
                    ),
                    h("td", null,
                      h("span", { className: "badge " + mlpBadgeClass }, mlpBadgeText)
                    ),
                    h("td", null,
                      h("span", { className: "badge " + rdBadgeClass }, `${r.readout_type} • ${r.optimizer_mode || "muon"}`)
                    ),
                    h("td", { style: { color: "#8b949e" } }, r.ep05_ce != null ? r.ep05_ce.toFixed(4) : "—"),
                    h("td", { style: { color: "#8b949e" } }, r.ep10_ce != null ? r.ep10_ce.toFixed(4) : "—"),
                    h("td", { style: { fontWeight: 700, color: r.best_val_ce != null && r.best_val_ce < 3.55 ? "#00ff88" : (r.best_val_ce != null && r.best_val_ce < 4.00 ? "#ff0844" : (r.best_val_ce != null && r.best_val_ce < 4.16 ? "#ffd200" : "#e6edf3")) } },
                      r.best_val_ce != null ? r.best_val_ce.toFixed(4) : (r.live_train_ce != null ? `~${r.live_train_ce.toFixed(2)} (tr)` : "—"),
                      deltaBase != null && h("span", {
                        style: {
                          fontSize: 10.5,
                          marginLeft: 5,
                          color: deltaBase <= 0 ? "#38ef7d" : "#8b949e"
                        }
                      }, `(${deltaBase >= 0 ? "+" : ""}${deltaBase.toFixed(3)})`)
                    ),
                    h("td", null, r.best_val_ppl != null ? r.best_val_ppl.toFixed(1) : "—"),
                    h("td", { style: { fontWeight: r.best_top1 != null && r.best_top1 >= 32.0 ? 700 : 400, color: r.best_top1 != null && r.best_top1 >= 32.0 ? "#00d2ff" : "#e6edf3" } },
                      r.best_top1 != null ? r.best_top1.toFixed(2) + "%" : "—"
                    ),
                    h("td", null, r.best_top5 != null ? r.best_top5.toFixed(2) + "%" : "—"),
                    h("td", { style: { color: "#d2a8ff", fontWeight: 600 } }, r.final_r_eff != null ? r.final_r_eff.toFixed(1) : "—"),
                    h("td", { style: { color: "#8b949e" } }, r.final_iso_err != null ? r.final_iso_err.toFixed(4) : "—"),
                    h("td", null, r.avg_tok_per_s ? Math.round(r.avg_tok_per_s).toLocaleString() : "—"),
                    h("td", null, r.wall_min ? r.wall_min.toFixed(1) + "m" : "—")
                  );
                })
              )
            )
          ),

          // Secondary & Historical Diagnostics Panel at the Bottom of Tab 1 (Below Leaderboard Table)
          h("div", { className: "inflight-panel", style: { marginTop: 18 } },
            h("div", { style: { fontSize: 13, fontWeight: 700, color: "#8b949e", marginBottom: 10 } },
              "📉 Secondary & Historical Diagnostics (Seed Dimension D_seed & Seed State Effective Rank)"
            ),
            h("div", { className: "charts-grid", style: { marginBottom: 0 } },
              h(SvgSeedScalingChart, { catalog, hoveredKey, onHoverKey: setHoveredKey }),
              h(SvgTrajectoryChart, {
                title: `Seed Quantum State Effective Rank R_eff(rho_seed) vs ${xAxisLabel}`,
                series: chartData.rEff,
                maxX: chartData.maxPrimaryX,
                minY: 5.0,
                maxY: 85.0,
                xSuffix: xAxisSuffix,
                hoveredKey,
                onHoverKey: setHoveredKey
              })
            )
          )
          )
        );
      }

      const root = ReactDOM.createRoot(document.getElementById("root"));
      root.render(h(App));
    })();
  </script>
</body>
</html>"""
    return html_template.replace("__INITIAL_JSON__", initial_json).replace("__REPORT_TXT__", report_txt)


def create_benchmark_http_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    configs_dir: Path = CONFIGS_DIR,
    model_cards_dir: Path = MODEL_CARDS_DIR,
    live_logs_dir: Path = LIVE_DIR,
    include_historical_specs: bool = True,
    start_bg_poller: bool = False,
    poll_interval: int = 20,
    initial_skip_remote_poll: bool = False,
    db_path: Path | None = None,
) -> tuple[ThreadingHTTPServer, dict[str, Any]]:
    """Create and configure the Zero-Downtime Benchmarking `ThreadingHTTPServer` and shared state."""
    server_t0 = time.time()
    state_lock = threading.Lock()
    resolved_db_path = (
        db_path
        if db_path is not None
        else (DEFAULT_DB_PATH if live_logs_dir == LIVE_DIR else (live_logs_dir / "benchmarks.db"))
    )
    db = BenchmarkSQLiteDB(db_path=resolved_db_path)

    initial_snap = collect_snapshot(
        skip_remote_poll=initial_skip_remote_poll,
        configs_dir=configs_dir,
        model_cards_dir=model_cards_dir,
        live_logs_dir=live_logs_dir,
        include_historical_specs=include_historical_specs,
        db_path=resolved_db_path,
    )
    cards_by_key: dict[str, dict[str, Any]] = {
        str(r["key"]): r["model_card"]
        for r in (
            initial_snap.get("runs_catalog", [])
            + initial_snap.get("fineweb_edu_catalog", [])
            + initial_snap.get("micro_runs_catalog", [])
        )
        if isinstance(r.get("model_card"), dict)
    }

    def _encode_cache(snap_dict: dict[str, Any], cards_map: dict[str, dict[str, Any]]) -> tuple[bytes, bytes, int]:
        s_clean = {k: v for k, v in snap_dict.items() if k not in ("server_uptime_s", "uptime_s", "runs_count")}
        status_sfx = json.dumps(s_clean)[1:].encode("utf-8")
        cards_sfx = json.dumps({"model_cards": cards_map, "count": len(cards_map)})[1:].encode("utf-8")
        return status_sfx, cards_sfx, len(snap_dict.get("runs_catalog", []))

    init_status_sfx, init_cards_sfx, init_runs_cnt = _encode_cache(initial_snap, cards_by_key)

    state: dict[str, Any] = {
        "db": db,
        "db_path": resolved_db_path,
        "latest_snap": initial_snap,
        "cards_by_key": cards_by_key,
        "server_t0": server_t0,
        "status_suffix_bytes": init_status_sfx,
        "cards_suffix_bytes": init_cards_sfx,
        "runs_count": init_runs_cnt,
    }

    def _refresh_state(skip_remote_poll: bool = True) -> dict[str, Any]:
        snap = collect_snapshot(
            skip_remote_poll=skip_remote_poll,
            configs_dir=configs_dir,
            model_cards_dir=model_cards_dir,
            live_logs_dir=live_logs_dir,
            include_historical_specs=include_historical_specs,
            db_path=resolved_db_path,
        )
        new_cards = {
            str(r["key"]): r["model_card"]
            for r in (
                snap.get("runs_catalog", [])
                + snap.get("fineweb_edu_catalog", [])
                + snap.get("micro_runs_catalog", [])
            )
            if isinstance(r.get("model_card"), dict)
        }
        status_sfx, cards_sfx, runs_cnt = _encode_cache(snap, new_cards)
        with state_lock:
            state["latest_snap"] = snap
            state["cards_by_key"] = new_cards
            state["status_suffix_bytes"] = status_sfx
            state["cards_suffix_bytes"] = cards_sfx
            state["runs_count"] = runs_cnt
        return snap

    if start_bg_poller:
        def bg_poller() -> None:
            time.sleep(1.0)
            while True:
                try:
                    _refresh_state(skip_remote_poll=False)
                except Exception:
                    pass
                time.sleep(poll_interval)

        threading.Thread(target=bg_poller, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, status_code: int, content_type: str, body: bytes) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send_bytes(status_code, "application/json", body)

        def do_GET(self) -> None:  # noqa: N802
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)

            if path in ("/api/snapshot", "/api/status"):
                with state_lock:
                    sfx = state["status_suffix_bytes"]
                    rcnt = state["runs_count"]
                    if sfx is None:
                        sfx, cards_sfx, rcnt = _encode_cache(state["latest_snap"], state["cards_by_key"])
                        state["status_suffix_bytes"] = sfx
                        state["cards_suffix_bytes"] = cards_sfx
                        state["runs_count"] = rcnt
                up = round(max(0.001, time.time() - server_t0), 4)
                prefix = f'{{"server_uptime_s": {up}, "uptime_s": {up}, "runs_count": {rcnt}, '.encode("utf-8")
                self._send_bytes(200, "application/json", prefix + sfx)
                return

            if path == "/api/micro-verifications":
                with state_lock:
                    micro_runs = list(state["latest_snap"].get("micro_runs_catalog", []))
                up = round(max(0.001, time.time() - server_t0), 4)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "micro_runs": micro_runs,
                        "count": len(micro_runs),
                        "server_uptime_s": up,
                        "uptime_s": up,
                    },
                )
                return

            if path == "/api/queue":
                with state_lock:
                    queue_items = list(state["latest_snap"].get("session_queue", []))
                up = round(max(0.001, time.time() - server_t0), 4)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "queue": queue_items,
                        "count": len(queue_items),
                        "server_uptime_s": up,
                        "uptime_s": up,
                    },
                )
                return

            if path == "/api/colab-sessions":
                with state_lock:
                    sessions_list = list(state["latest_snap"].get("colab_sessions", []))
                    acct = dict(state["latest_snap"].get("colab_account", {}))
                up = round(max(0.001, time.time() - server_t0), 4)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "colab_sessions": sessions_list,
                        "colab_account": acct,
                        "count": len(sessions_list),
                        "server_uptime_s": up,
                        "uptime_s": up,
                    },
                )
                return

            if path == "/api/model-cards":
                with state_lock:
                    sfx = state["cards_suffix_bytes"]
                    if sfx is None:
                        status_sfx, sfx, rcnt = _encode_cache(state["latest_snap"], state["cards_by_key"])
                        state["status_suffix_bytes"] = status_sfx
                        state["cards_suffix_bytes"] = sfx
                        state["runs_count"] = rcnt
                up = round(max(0.001, time.time() - server_t0), 4)
                prefix = f'{{"server_uptime_s": {up}, "uptime_s": {up}, '.encode("utf-8")
                self._send_bytes(200, "application/json", prefix + sfx)
                return

            if path.startswith("/api/model-cards/"):
                run_key = urllib.parse.unquote(path.split("/api/model-cards/", 1)[1]).strip("/")
                with state_lock:
                    card = state["cards_by_key"].get(run_key)
                if card is None:
                    disk_card = model_cards_dir / f"{run_key}.json"
                    if disk_card.exists():
                        try:
                            card = json.loads(disk_card.read_text(encoding="utf-8"))
                        except Exception:
                            card = None
                if card is None:
                    self._send_json(404, {"ok": False, "error": f"Model card not found: {run_key}"})
                    return
                fmt = (query.get("format") or ["json"])[0].lower()
                if fmt in ("md", "markdown"):
                    md_bytes = render_model_card_markdown(card).encode("utf-8")
                    self._send_bytes(200, "text/markdown; charset=utf-8", md_bytes)
                    return
                self._send_json(200, card)
                return

            if path.startswith("/api/report"):
                with state_lock:
                    snap_ref = state["latest_snap"]
                body = (format_terminal_report(snap_ref) + "\n").encode("utf-8")
                self._send_bytes(200, "text/plain; charset=utf-8", body)
                return

            if path == "/api/architecture-graph":
                from qdlblocks.benchmarks.model_block_visualizer import build_architecture_visualizer_payload

                payload = build_architecture_visualizer_payload(live_logs_dir=live_logs_dir)
                self._send_json(200, payload)
                return

            if path in ("/architecture", "/architecture/", "/architecture_visualizer.html", "/blocks"):
                arch_file = ROOT / "architecture_visualizer.html"
                if arch_file.exists():
                    body = arch_file.read_bytes()
                else:
                    from scripts.export_model_block_visualizer import render_architecture_visualizer_html
                    from qdlblocks.benchmarks.model_block_visualizer import build_architecture_visualizer_payload

                    payload = build_architecture_visualizer_payload(live_logs_dir=live_logs_dir)
                    body = render_architecture_visualizer_html(payload).encode("utf-8")
                self._send_bytes(200, "text/html; charset=utf-8", body)
                return

            if path in ("/visualizer", "/visualizer/", "/visualizer.html", "/playground"):
                vis_file = ROOT / "visualizer.html"
                if not vis_file.exists():
                    self._send_json(404, {"ok": False, "error": "visualizer.html not found"})
                    return
                body = vis_file.read_bytes()
                self._send_bytes(200, "text/html; charset=utf-8", body)
                return

            with state_lock:
                snap_ref = state["latest_snap"]
            body = render_html_dashboard(snap_ref).encode("utf-8")
            self._send_bytes(200, "text/html; charset=utf-8", body)

        def do_POST(self) -> None:  # noqa: N802
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path

            if path == "/api/reload":
                t0 = time.perf_counter()
                snap = _refresh_state(skip_remote_poll=True)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                up = round(max(0.001, time.time() - server_t0), 4)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "runs_count": len(snap.get("runs_catalog", [])),
                        "fineweb_edu_count": len(snap.get("fineweb_edu_catalog", [])),
                        "micro_runs_count": len(snap.get("micro_runs_catalog", [])),
                        "server_uptime_s": up,
                        "uptime_s": up,
                        "reload_ms": round(dt_ms, 2),
                    },
                )
                return

            if path == "/api/queue":
                t_q0 = time.perf_counter()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    payload = json.loads(raw)
                    action = str(payload.get("action") or "enqueue").lower()
                    if action == "sync_manifest":
                        manifest_rel = str(payload.get("manifest") or "configs/wave12_fleet.json")
                        manifest_path = Path(manifest_rel)
                        if not manifest_path.is_absolute():
                            manifest_path = (configs_dir / manifest_path.name) if (configs_dir / manifest_path.name).exists() else (ROOT / manifest_rel)
                        db.sync_queue_manifest(manifest_path)
                    elif action == "cancel":
                        rk = str(payload.get("run_key") or (payload.get("item") or {}).get("run_key") or "")
                        if rk:
                            item_obj = payload.get("item") or {}
                            db.enqueue_job(
                                session=str(item_obj.get("session") or payload.get("session") or "gpu_g4_w6_1"),
                                run_key=rk,
                                wave_id=str(item_obj.get("wave_id") or payload.get("wave_id") or "W12"),
                                status="CANCELLED",
                            )
                    else:
                        item = payload.get("item") if isinstance(payload.get("item"), dict) else payload
                        db.enqueue_job(
                            session=str(item.get("session") or "gpu_g4_w6_1"),
                            run_key=str(item["run_key"]),
                            wave_id=str(item.get("wave_id") or "W12"),
                            arch_family=str(item.get("arch_family") or ("classical_modded" if "classical" in str(item["run_key"]) else "quantum_born")),
                            hardware=str(item.get("hardware") or "NVIDIA RTX PRO 6000 Blackwell (96GB)"),
                            short_label=str(item.get("short_label") or item["run_key"]),
                            cli_args=str(item.get("cli_args") or ""),
                            status=str(item.get("status") or "QUEUED"),
                            priority=int(item.get("priority", 100)),
                        )
                    db_payload = db.load_snapshot_payload()
                    with state_lock:
                        snap = dict(state["latest_snap"])
                        snap["session_queue"] = db_payload["session_queue"]
                        snap["db_stats"] = db_payload["db_stats"]
                        state["latest_snap"] = snap
                        state["status_suffix_bytes"] = None
                    dt_q_ms = (time.perf_counter() - t_q0) * 1000.0
                    up = round(max(0.001, time.time() - server_t0), 4)
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "action": action,
                            "queue": db_payload["session_queue"],
                            "count": len(db_payload["session_queue"]),
                            "queue_ms": round(dt_q_ms, 2),
                            "server_uptime_s": up,
                        },
                    )
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if path == "/api/colab-heartbeat":
                t_hb0 = time.perf_counter()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    payload = json.loads(raw)
                    sess_name = str(payload["session"])
                    with state_lock:
                        existing_sessions = list(state["latest_snap"].get("colab_sessions", []))
                    base_sess = next((dict(s) for s in existing_sessions if s.get("session") == sess_name), {"session": sess_name})
                    base_sess.update(payload)
                    if "vm_assigned" not in payload:
                        base_sess["vm_assigned"] = True
                    db.upsert_colab_session(base_sess)
                    db_payload = db.load_snapshot_payload()
                    with state_lock:
                        snap = dict(state["latest_snap"])
                        snap["colab_sessions"] = db_payload["colab_sessions"]
                        snap["session_queue"] = db_payload["session_queue"]
                        snap["db_stats"] = db_payload["db_stats"]
                        state["latest_snap"] = snap
                        state["status_suffix_bytes"] = None
                    dt_hb_ms = (time.perf_counter() - t_hb0) * 1000.0
                    up = round(max(0.001, time.time() - server_t0), 4)
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "session": sess_name,
                            "colab_sessions": db_payload["colab_sessions"],
                            "session_queue": db_payload["session_queue"],
                            "heartbeat_ms": round(dt_hb_ms, 2),
                            "server_uptime_s": up,
                        },
                    )
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if path == "/api/colab-stop":
                t_st0 = time.perf_counter()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    payload = json.loads(raw)
                    dry_run = bool(payload.get("dry_run", False))
                    with state_lock:
                        curr_sessions = list(state["latest_snap"].get("colab_sessions", []))
                    target_sessions: list[str] = []
                    if payload.get("session"):
                        target_sessions = [str(payload["session"])]
                    elif isinstance(payload.get("sessions"), list):
                        target_sessions = [str(s) for s in payload["sessions"] if str(s)]
                    elif payload.get("all"):
                        target_sessions = [str(s["session"]) for s in curr_sessions if s.get("session")]
                    elif payload.get("all_idle", True):
                        target_sessions = [
                            str(s["session"])
                            for s in curr_sessions
                            if s.get("session") and not s.get("ps_alive")
                        ]
                    stopped = stop_colab_sessions(target_sessions, dry_run=dry_run)
                    stopped_set = set(stopped)
                    for s_obj in curr_sessions:
                        if s_obj.get("session") in stopped_set:
                            s_upd = dict(s_obj)
                            s_upd["vm_assigned"] = False
                            s_upd["ps_alive"] = False
                            s_upd["ps_cmd"] = ""
                            s_upd["gpu_util_pct"] = 0.0
                            s_upd["mem_bw_util_pct"] = 0.0
                            s_upd["vram_used_gib"] = 0.0
                            db.upsert_colab_session(s_upd)
                    db_payload = db.load_snapshot_payload()
                    acct = query_colab_account_status(skip_subprocess=True)
                    with state_lock:
                        snap = dict(state["latest_snap"])
                        snap["colab_sessions"] = db_payload["colab_sessions"]
                        snap["colab_account"] = acct
                        state["latest_snap"] = snap
                        state["status_suffix_bytes"] = None
                    dt_st_ms = (time.perf_counter() - t_st0) * 1000.0
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "stopped_sessions": stopped,
                            "colab_account": acct,
                            "stop_ms": round(dt_st_ms, 2),
                        },
                    )
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if path == "/api/obsolete" or (path.startswith("/api/model-cards/") and path.endswith("/obsolete")):
                t_obs0 = time.perf_counter()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    payload = json.loads(raw)
                    target_obsolete = bool(payload.get("obsolete", True))
                    reason = str(payload.get("reason", "Marked obsolete from dashboard" if target_obsolete else ""))

                    keys_raw = payload.get("run_keys")
                    if isinstance(keys_raw, list) and keys_raw:
                        target_keys = [str(k) for k in keys_raw if str(k)]
                    elif payload.get("run_key"):
                        target_keys = [str(payload["run_key"])]
                    elif path.startswith("/api/model-cards/") and path.endswith("/obsolete"):
                        mid = urllib.parse.unquote(path[len("/api/model-cards/") : -len("/obsolete")]).strip("/")
                        target_keys = [mid] if mid else []
                    else:
                        target_keys = []

                    if not target_keys:
                        self._send_json(400, {"ok": False, "error": "Missing run_key or run_keys"})
                        return

                    live_logs_dir.mkdir(parents=True, exist_ok=True)
                    model_cards_dir.mkdir(parents=True, exist_ok=True)
                    ui_prefs_path = live_logs_dir / "ui_prefs.json"
                    ui_prefs: dict[str, Any] = {}
                    if ui_prefs_path.exists():
                        try:
                            ui_prefs = json.loads(ui_prefs_path.read_text(encoding="utf-8"))
                        except Exception:
                            ui_prefs = {}

                    obs_map: dict[str, bool] = dict(ui_prefs.get("obsoleteMap") or {})
                    obs_reasons: dict[str, str] = dict(ui_prefs.get("obsoleteReasons") or {})
                    exp_keys: set[str] = set(ui_prefs.get("explicitObsoleteKeys") or [])
                    vis_map: dict[str, bool] = dict(ui_prefs.get("visibleMap") or {})

                    for rk in target_keys:
                        obs_map[rk] = target_obsolete
                        exp_keys.add(rk)
                        if target_obsolete:
                            obs_reasons[rk] = reason or "Marked obsolete from dashboard"
                            vis_map[rk] = False
                        else:
                            obs_reasons.pop(rk, None)

                    ui_prefs["obsoleteMap"] = obs_map
                    ui_prefs["obsoleteReasons"] = obs_reasons
                    ui_prefs["explicitObsoleteKeys"] = sorted(exp_keys)
                    ui_prefs["visibleMap"] = vis_map
                    ui_prefs_path.write_text(json.dumps(ui_prefs, indent=2), encoding="utf-8")

                    db_obs_info = db.set_obsolete_override(target_keys, target_obsolete, reason=reason)
                    target_key_set = set(target_keys)
                    eff_rsn = (reason or "Marked obsolete from dashboard") if target_obsolete else ""

                    with state_lock:
                        snap = dict(state["latest_snap"])
                        cards_map = dict(state["cards_by_key"])

                        for rk in target_keys:
                            card = cards_map.get(rk)
                            if card is None:
                                disk_p = model_cards_dir / f"{rk}.json"
                                if disk_p.exists():
                                    try:
                                        card = json.loads(disk_p.read_text(encoding="utf-8"))
                                    except Exception:
                                        card = None
                            if card is not None:
                                updated_card = set_model_card_obsolete(card, target_obsolete, reason=reason)
                                save_model_card_atomic(updated_card, model_cards_dir)
                                cards_map[rk] = updated_card

                        catalog = [dict(r) for r in snap.get("runs_catalog", [])]
                        for r_item in catalog:
                            rk_i = str(r_item.get("key") or "")
                            if rk_i in target_key_set:
                                r_item["obsolete"] = target_obsolete
                                r_item["obsolete_reason"] = eff_rsn
                                r_item["explicit_user_override"] = True
                                if target_obsolete:
                                    r_item["default_visible"] = False
                                if rk_i in cards_map:
                                    r_item["model_card"] = cards_map[rk_i]

                        catalog.sort(
                            key=lambda r: (
                                1 if r.get("obsolete") else 0,
                                r.get("best_val_ce") is None,
                                float(r["best_val_ce"]) if r.get("best_val_ce") is not None else 999.0,
                                -float(r.get("wave_num", 0.0)),
                            )
                        )
                        snap["runs_catalog"] = catalog
                        snap["saved_ui_prefs"] = ui_prefs
                        state["latest_snap"] = snap
                        state["cards_by_key"] = cards_map
                        state["status_suffix_bytes"] = None
                        state["cards_suffix_bytes"] = None
                        state["runs_count"] = len(catalog)
                        obs_cnt = int(db_obs_info.get("obsolete_count", sum(1 for r in catalog if r.get("obsolete"))))
                        act_cnt = int(db_obs_info.get("active_count", len(catalog) - obs_cnt))

                    dt_obs_ms = (time.perf_counter() - t_obs0) * 1000.0
                    up = round(max(0.001, time.time() - server_t0), 4)
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "updated_keys": target_keys,
                            "obsolete": target_obsolete,
                            "obsolete_count": obs_cnt,
                            "active_count": act_cnt,
                            "obsolete_ms": round(dt_obs_ms, 2),
                            "server_uptime_s": up,
                            "uptime_s": up,
                        },
                    )
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if path == "/api/ingest":
                t_ing0 = time.perf_counter()
                expected_token = os.environ.get("QDL_INGEST_TOKEN", "").strip()
                if expected_token:
                    auth_hdr = self.headers.get("Authorization", "")
                    if auth_hdr != f"Bearer {expected_token}":
                        self._send_json(401, {"ok": False, "error": "Unauthorized ingest token"})
                        return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    payload = json.loads(raw)
                    run_key = str(payload["run_key"])

                    live_logs_dir.mkdir(parents=True, exist_ok=True)
                    model_cards_dir.mkdir(parents=True, exist_ok=True)
                    log_file = live_logs_dir / f"{run_key}.log"
                    json_file = live_logs_dir / f"{run_key}_results.json"

                    if payload.get("log_full") is not None:
                        log_file.write_text(str(payload["log_full"]), encoding="utf-8")
                    elif payload.get("log_append"):
                        with open(log_file, "a", encoding="utf-8") as lf:
                            lf.write(str(payload["log_append"]))

                    res_json = payload.get("results_json")
                    if res_json is None and isinstance(payload.get("evals"), list):
                        res_json = payload["evals"]
                    if res_json is not None:
                        json_file.write_text(json.dumps(res_json, indent=2), encoding="utf-8")
                    elif json_file.exists():
                        try:
                            res_json = json.loads(json_file.read_text(encoding="utf-8"))
                        except Exception:
                            res_json = None

                    with state_lock:
                        existing_card = state["cards_by_key"].get(run_key)

                    if isinstance(payload.get("model_card"), dict):
                        card = dict(payload["model_card"])
                        card["run_key"] = run_key
                    elif existing_card is not None:
                        card = dict(existing_card)
                    else:
                        disk_card_path = model_cards_dir / f"{run_key}.json"
                        if disk_card_path.exists():
                            card = json.loads(disk_card_path.read_text(encoding="utf-8"))
                        else:
                            card = build_model_card_from_manifest_entry(
                                {
                                    "key": run_key,
                                    "session": payload.get("session", run_key),
                                    "hardware": payload.get("hardware", "NVIDIA RTX PRO 6000 Blackwell (96GB)"),
                                    "label": payload.get("label", run_key),
                                    "short_label": payload.get("short_label", run_key),
                                    "cli_args": payload.get("cli_args", "--backend cuda --architecture classical_modded"),
                                    "remote_log": f"/content/{run_key}.log",
                                    "remote_json": f"/content/{run_key}_results.json",
                                    "run_tier": payload.get("run_tier"),
                                },
                                wave_id=str(payload.get("wave_id", "")),
                                wave_description=str(payload.get("wave_label", "")),
                            )

                    log_txt = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
                    log_mtime = log_file.stat().st_mtime if log_file.exists() else time.time()
                    hw_sec = card.get("hardware", {})
                    peak_tf = float(hw_sec.get("peak_bf16_tflops", BLACKWELL_RTX6000_BF16_PEAK_TFLOPS))
                    parsed = parse_log_metrics(
                        log_txt,
                        peak_tf,
                        225000.0,
                        res_json,
                        log_mtime=log_mtime,
                        ps_alive=False if ("COMPLETED" in log_txt or payload.get("completed") or payload.get("run_state") == "COMPLETED") else True,
                    )
                    dab_sec = card.get("data_and_batch", {})
                    if payload.get("target_tokens_M") is not None:
                        parsed["target_tokens_M"] = float(payload["target_tokens_M"])
                        dab_sec["target_tokens_M"] = float(payload["target_tokens_M"])
                        card["data_and_batch"] = dab_sec
                    elif dab_sec.get("target_tokens_M"):
                        parsed["target_tokens_M"] = float(dab_sec["target_tokens_M"])
                    for override_k in (
                        "best_val_ce",
                        "best_val_ppl",
                        "best_top1",
                        "best_top5",
                        "wall_min",
                        "avg_tok_per_s",
                        "current_step",
                        "total_steps",
                        "live_tokens_M",
                        "last_eval_tokens_M",
                        "progress_pct",
                        "run_state",
                        "completed",
                    ):
                        if payload.get(override_k) is not None:
                            parsed[override_k] = payload[override_k]
                    if isinstance(payload.get("steps"), list):
                        parsed["steps"] = payload["steps"]
                    if isinstance(payload.get("evals"), list):
                        parsed["evals"] = payload["evals"]
                    if payload.get("run_tier"):
                        card["run_tier"] = str(payload["run_tier"])
                    parsed["session"] = str(payload.get("session") or hw_sec.get("session", run_key))
                    parsed["accel"] = str(payload.get("accel") or hw_sec.get("accelerator", "G4 (RTX PRO 6000 Blackwell)"))
                    parsed["label"] = str(payload.get("label") or card.get("label", run_key))
                    parsed["short_label"] = str(payload.get("short_label") or card.get("short_label", run_key)).split("] ", 1)[-1]
                    parsed["color"] = str(payload.get("color") or card.get("color", "#00f2fe"))

                    updated_card = update_model_card_telemetry(card, parsed)
                    if payload.get("run_tier"):
                        updated_card["run_tier"] = str(payload["run_tier"])
                    save_model_card_atomic(updated_card, model_cards_dir)

                    cat_entry = _build_catalog_entry(
                        run_key,
                        parsed,
                        str(updated_card.get("label", run_key)),
                        str(updated_card.get("short_label", run_key)),
                        str(updated_card.get("color", "#00f2fe")),
                        str(hw_sec.get("accelerator", "G4 Blackwell")),
                        str(hw_sec.get("kind", "gpu")).upper(),
                        str(updated_card.get("wave_label", "Wave 9")),
                        bool(updated_card.get("default_visible", True)),
                        model_card=updated_card,
                        wave_id=str(updated_card.get("wave_id", "W9")),
                    )
                    if payload.get("run_tier"):
                        cat_entry["run_tier"] = str(payload["run_tier"])

                    db.upsert_run(cat_entry, model_card=updated_card)
                    db.recompute_pareto_and_obsolete(DEFAULT_OBSOLETE_KEYS)
                    db_payload = db.load_snapshot_payload()

                    with state_lock:
                        snap = dict(state["latest_snap"])
                        all_main = db_payload["runs_catalog"]
                        snap["runs_catalog"] = [
                            r
                            for r in all_main
                            if not (
                                float(r.get("wave_num", 0.0)) >= 18.0
                                or str(r.get("task_domain") or "") == "fineweb_edu_local"
                            )
                        ]
                        existing_edu = {str(r["key"]): r for r in snap.get("fineweb_edu_catalog", [])}
                        for r in all_main:
                            if (
                                float(r.get("wave_num", 0.0)) >= 18.0
                                or str(r.get("task_domain") or "") == "fineweb_edu_local"
                            ) and not any(
                                sub in str(r.get("key") or "").lower()
                                for sub in ("cifar100", "1000class", "imagenet1k")
                            ):
                                existing_edu[str(r["key"])] = r
                        snap["fineweb_edu_catalog"] = list(existing_edu.values())
                        snap["micro_runs_catalog"] = db_payload["micro_runs_catalog"]
                        snap["session_queue"] = db_payload["session_queue"]
                        snap["db_stats"] = db_payload["db_stats"]
                        fleet_runs_map = dict(snap.get("fleet_runs", {}))
                        fleet_runs_map[run_key] = parsed
                        snap["fleet_runs"] = fleet_runs_map

                        cards_map = dict(state["cards_by_key"])
                        cards_map[run_key] = updated_card
                        status_sfx, cards_sfx, rcnt = _encode_cache(snap, cards_map)
                        state["latest_snap"] = snap
                        state["cards_by_key"] = cards_map
                        state["status_suffix_bytes"] = status_sfx
                        state["cards_suffix_bytes"] = cards_sfx
                        state["runs_count"] = rcnt

                    dt_ing_ms = (time.perf_counter() - t_ing0) * 1000.0
                    up = round(max(0.001, time.time() - server_t0), 4)
                    best_ce = updated_card.get("telemetry", {}).get("best_val_ce")
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "run_key": run_key,
                            "run_tier": cat_entry.get("run_tier", "main_benchmark"),
                            "runs_count": rcnt,
                            "fineweb_edu_count": len(snap.get("fineweb_edu_catalog", [])),
                            "micro_runs_count": len(db_payload["micro_runs_catalog"]),
                            "best_val_ce": best_ce,
                            "ingest_ms": round(dt_ing_ms, 2),
                            "server_uptime_s": up,
                            "uptime_s": up,
                        },
                    )
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if path.startswith("/api/ui_prefs"):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    prefs = json.loads(raw)
                    ui_prefs_path = live_logs_dir / "ui_prefs.json"
                    merged_prefs: dict[str, Any] = {}
                    if ui_prefs_path.exists():
                        try:
                            merged_prefs = json.loads(ui_prefs_path.read_text(encoding="utf-8"))
                        except Exception:
                            merged_prefs = {}
                    merged_prefs.update(prefs)
                    ui_prefs_path.write_text(json.dumps(merged_prefs, indent=2), encoding="utf-8")
                    with state_lock:
                        state["latest_snap"]["saved_ui_prefs"] = merged_prefs
                        status_sfx, cards_sfx, rcnt = _encode_cache(state["latest_snap"], state["cards_by_key"])
                        state["status_suffix_bytes"] = status_sfx
                        state["cards_suffix_bytes"] = cards_sfx
                        state["runs_count"] = rcnt
                    self._send_json(200, {"ok": True})
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    class FastThreadingHTTPServer(ThreadingHTTPServer):
        request_queue_size = 128
        allow_reuse_address = True

    server = FastThreadingHTTPServer((host, port), Handler)
    return server, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Read latest snapshot from running daemon (or poll once if daemon is down), print summary, and exit.")
    parser.add_argument("--force-poll", action="store_true", help="Force a direct Colab poll even when the background --serve daemon is running.")
    parser.add_argument("--watch", action="store_true", help="Continuously poll and refresh in terminal.")
    parser.add_argument("--serve", action="store_true", help="Serve live HTML dashboard and background poller on 0.0.0.0.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address for --serve (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve (default: 8765).")
    parser.add_argument("--interval", type=int, default=20, help="Polling interval in seconds (default: 20).")
    args = parser.parse_args()

    if args.serve:
        server, state = create_benchmark_http_server(
            host=args.host,
            port=args.port,
            start_bg_poller=True,
            poll_interval=args.interval,
            initial_skip_remote_poll=True,
        )
        lan_ip = state["latest_snap"].get("lan_ip", "0.0.0.0")
        print(
            f"Serving Zero-Downtime React Colab Fleet & Model Card Monitor on http://{args.host}:{args.port} (LAN: http://{lan_ip}:{args.port})...",
            flush=True,
        )
        server.serve_forever()
    elif args.watch:
        while True:
            snap = collect_snapshot()
            print("\033[2J\033[H" + format_terminal_report(snap), flush=True)
            time.sleep(args.interval)
    else:
        snap = None
        if not args.force_poll:
            import urllib.request
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/snapshot", timeout=1.5) as resp:
                    snap = json.loads(resp.read().decode("utf-8"))
            except Exception:
                snap_file = LIVE_DIR / "latest_snapshot.json"
                if snap_file.exists() and (time.time() - snap_file.stat().st_mtime) < 35.0:
                    try:
                        snap = json.loads(snap_file.read_text())
                    except Exception:
                        snap = None
        if snap is None:
            snap = collect_snapshot()
        print(format_terminal_report(snap), flush=True)


if __name__ == "__main__":
    main()
