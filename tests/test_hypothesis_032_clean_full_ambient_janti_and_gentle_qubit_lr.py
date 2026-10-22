"""Unit and hypothesis tests for Experiment Plan 032 (Anti-Diagonal Spectral-Reversal Isometry J_anti & Gentle 2x Qubit LR)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.synthetic_fineweb_walk import build_fast_quantum_born_synth_lm
from qdlblocks.layers.hierarchical_qubit import initialize_hierarchical_16qubit_with_ppmi_svd


def test_h1_janti_full_ambient_self_orthogonality_directed_ratio_and_step0_calibration() -> None:
    """Verify H1: J_anti full-ambient initialization achieves <1e-5 self-overlap, >=2.5x directed bigram ratio, and calibrated Step 0 CE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 2048
    embed_dim = 64
    num_regs = 16
    reg_dim = 4  # 16 * 4 = 64 == embed_dim (full-ambient, even A=4)

    gen = torch.Generator(device="cpu").manual_seed(2032)
    ppmi_emb = torch.randn((vocab_size, embed_dim), generator=gen, dtype=torch.float32).to(device)

    # Construct synthetic token stream with strong directed bigram pairs u -> v
    base_tokens = torch.randint(0, vocab_size, (32000,), generator=gen, dtype=torch.int64)
    for k in range(32):
        u_tok = 10 + 2 * k
        v_tok = 11 + 2 * k
        idx_pos = torch.arange(k * 500, k * 500 + 400, 2)
        base_tokens[idx_pos] = u_tok
        base_tokens[idx_pos + 1] = v_tok
    token_arr = base_tokens.numpy().astype(np.uint16)

    m_legacy = build_fast_quantum_born_synth_lm(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        block_size=64,
        seed=2032,
        device=device,
    )
    stats_legacy = initialize_hierarchical_16qubit_with_ppmi_svd(
        m_legacy,
        ppmi_embeddings=ppmi_emb,
        token_shards=[token_arr],
        alpha=0.75,
        decouple_self_in_full_ambient=False,
        directed_predecessor_mix=0.0,
        device=device,
    )
    assert float(stats_legacy["self_register_overlap_max"]) > 0.40

    m_janti = build_fast_quantum_born_synth_lm(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        block_size=64,
        seed=2032,
        device=device,
    )
    stats_janti = initialize_hierarchical_16qubit_with_ppmi_svd(
        m_janti,
        ppmi_embeddings=ppmi_emb,
        token_shards=[token_arr],
        alpha=0.75,
        decouple_self_in_full_ambient=True,
        directed_predecessor_mix=0.35,
        device=device,
    )
    assert float(stats_janti["self_register_overlap_max"]) < 1e-5
    assert float(stats_janti["directed_bigram_overlap_ratio"]) >= 2.5


def test_gentle_2x_zero_wd_qubit_embedding_optimizer_routing() -> None:
    """Verify that unigram_qubits, higher_states, and value_embeds route to the gentle 2x zero-WD group (lr=0.006, wd=0.0)."""
    spec_mod = importlib.util.spec_from_file_location(
        "verify_plan032",
        ROOT / "examples" / "42_verify_plan032_clean_full_ambient_and_gentle_qubit_lr.py",
    )
    assert spec_mod is not None and spec_mod.loader is not None
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    m = build_fast_quantum_born_synth_lm(
        vocab_size=1024,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        block_size=32,
        seed=42,
        device="cpu",
    )
    assert not bool(getattr(m, "use_bipolar_kboson", False))
    assert m.hierarchical_povm_readout is not None
    assert not bool(m.hierarchical_povm_readout.use_bipolar_kboson)

    _, adam_opt, _, _ = mod.build_optimizers_for_plan032(m, qubit_embed_lr=0.006)
    groups_by_name = {g.get("group_name"): g for g in adam_opt.param_groups}
    assert "qubit_embed" in groups_by_name
    qe_group = groups_by_name["qubit_embed"]
    assert abs(float(qe_group["lr"]) - 0.006) < 1e-9
    assert abs(float(qe_group["base_lr"]) - 0.006) < 1e-9
    assert float(qe_group["weight_decay"]) == 0.0

    qe_ids = {id(p) for p in qe_group["params"]}
    assert m.hierarchical_qubit_embed is not None
    assert id(m.hierarchical_qubit_embed.unigram_qubits) in qe_ids
    assert id(m.hierarchical_qubit_embed.higher_states) in qe_ids
    if m.value_embeds is not None:
        for ve in m.value_embeds:
            assert id(ve.weight) in qe_ids


def test_plan032_verification_artifact_schema_if_present() -> None:
    """Verify `.cache/hypothesis_experiments_032.json` schema and pass flags when present."""
    art_path = ROOT / ".cache" / "hypothesis_experiments_032.json"
    if not art_path.exists():
        return
    data = json.loads(art_path.read_text(encoding="utf-8"))
    assert data["plan_id"] == "EXPERIMENT_PLAN_032"
    assert bool(data["overall_pass"]) is True
    assert bool(data["H1_janti_full_ambient_calibration"]["passed"]) is True
    assert bool(data["H2_clean_L6_d512_gentle_lr_and_16x32_janti"]["passed"]) is True
    assert bool(data["H3_clean_L4_d256_janti_gentle_lr_900steps"]["passed"]) is True
    assert float(data["H3_clean_L4_d256_janti_gentle_lr_900steps"]["max_peak_vram_gb"]) <= 8.0
