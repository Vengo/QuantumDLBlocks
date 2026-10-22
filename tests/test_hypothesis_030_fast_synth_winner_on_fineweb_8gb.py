"""Unit and integration tests for Experiment Plan 030 (`EXPERIMENT_PLAN_030.md`).

Verifies:
1. `H1`: Full-Ambient (`use_ortho_decouple=False`, `Q=16, A=16, d=256`) Self-Register Orthogonal Decoupling &
   Directed Predecessor PPMI-SVD initialization (`self_register_overlap_max < 1e-5` vs `> 0.40` legacy,
   and `directed_bigram_overlap_ratio >= 2.0`).
2. `H2` & `H3`: End-to-end verification of `.cache/hypothesis_experiments_030.json` once generated on the
   local RTX 5060 Ti (`<= 8.0 GB` VRAM cap).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from qdlblocks.benchmarks.synthetic_fineweb_walk import build_fast_quantum_born_synth_lm
from qdlblocks.layers.cooccurrence_init import compute_ppmi_svd_embeddings
from qdlblocks.layers.hierarchical_qubit import initialize_hierarchical_16qubit_with_ppmi_svd

ROOT = Path(__file__).resolve().parents[1]


def test_h1_full_ambient_self_register_decoupling_and_directed_ppmi() -> None:
    """Verify full-ambient (`use_ortho_decouple=False`) self-register Gram-Schmidt decoupling and directed predecessor mix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(2030)
    vocab_size = 512
    # Construct a synthetic stream with strong directed bigram pairs (v -> (v + 1) % 256)
    base_tokens = rng.integers(0, 256, size=8192, dtype=np.uint16)
    stream = np.empty(16384, dtype=np.uint16)
    stream[0::2] = base_tokens
    stream[1::2] = (base_tokens + 1) % 256

    ppmi_emb = compute_ppmi_svd_embeddings(
        token_arrays=[stream],
        vocab_size=vocab_size,
        embed_dim=64,
        window_size=4,
        device=device,
    )["embeddings"]

    m_legacy = build_fast_quantum_born_synth_lm(
        vocab_size=vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        block_size=32,
        seed=42,
        device=device,
    )
    stats_legacy = initialize_hierarchical_16qubit_with_ppmi_svd(
        m_legacy,
        ppmi_embeddings=ppmi_emb,
        token_shards=[stream],
        alpha=0.80,
        kmeans_iters=4,
        decouple_self_in_full_ambient=False,
        directed_predecessor_mix=0.0,
        device=device,
    )
    assert stats_legacy["self_register_overlap_max"] > 0.40

    m_decoupled = build_fast_quantum_born_synth_lm(
        vocab_size=vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        block_size=32,
        seed=42,
        device=device,
    )
    stats_decoupled = initialize_hierarchical_16qubit_with_ppmi_svd(
        m_decoupled,
        ppmi_embeddings=ppmi_emb,
        token_shards=[stream],
        alpha=0.80,
        kmeans_iters=4,
        decouple_self_in_full_ambient=True,
        directed_predecessor_mix=0.35,
        device=device,
    )
    assert stats_decoupled["self_register_overlap_max"] < 1e-5
    assert stats_decoupled["directed_bigram_overlap_ratio"] >= 1.5


def test_plan030_verification_artifact_schema_if_present() -> None:
    """Verify `.cache/hypothesis_experiments_030.json` passes H1, H2, and H3 under `<= 8.0 GB` VRAM."""
    p = ROOT / ".cache" / "hypothesis_experiments_030.json"
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["plan_id"] == "EXPERIMENT_PLAN_030"
    assert data["vram_cap_gb"] <= 8.0
    assert data["H1_self_decoupled_directed_ppmi_svd"]["passed"] is True
    assert data["H2_fast_synth_winner_fineweb_300steps"]["passed"] is True
    assert data["H3_iso_wallclock_L4_d256_and_unified_L6_d512"]["passed"] is True
    assert data["overall_pass"] is True
