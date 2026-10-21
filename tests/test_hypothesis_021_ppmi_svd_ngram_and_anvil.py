"""Unit tests for Experiment Plan 021:
- 9-Shard 900M-Token Shifted PPMI + Randomized SVD Co-Occurrence Spectral Initialization
- Hashed Bigram + Trigram Sign-Modulated Embeddings (HashedNgramEmbedding)
- Polar Express 6-Step Quintic Schedule (ANVIL_QUINTIC_MAPS), NorMuon Per-Lane Equalizer,
  Twin-Rail Momentum, Sign-Aligned Cautious Weight Decay, & Norm-Preserving TailWeightAverager
- MUDDDynamicSkip, InputChannelGate, GatedXSA, ForwardSmearGate, YaRNRotarySchedule,
  apply_partial_key_offset, find_bos_aligned_starts, compute_tapered_mtp_weight
- Sampled-Softmax Candidate Cross-Entropy (sampled_softcap_linear_ce, sampled_povm_ce) & FP8LinearWrapper
- Wave-13 Fleet Manifest & Model Card Schema Validation
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from qdlblocks.benchmarks.fineweb_stream import (
    compute_tapered_mtp_weight,
    find_bos_aligned_starts,
)
from qdlblocks.benchmarks.model_card import (
    WAVE_DEFAULT_LABELS,
    build_model_card_from_manifest_entry,
    parse_cli_args_to_card_sections,
    validate_model_card,
)
from qdlblocks.functional import (
    ANVIL_QUINTIC_MAPS,
    normuon_equalize,
    zeropower_via_newtonschulz5,
    zeropower_via_polarexpress5,
    zeropower_via_polarexpress5_batched,
)
from qdlblocks.layers.cooccurrence_init import (
    compute_collocation_coherence_metrics,
    compute_ppmi_svd_embeddings,
    initialize_classical_with_ppmi_svd,
    initialize_quantum_born_with_ppmi_svd,
)
from qdlblocks.layers.direct_chunked_ce import (
    build_sampled_candidate_set,
    direct_chunked_povm_ce,
    direct_chunked_softcap_linear_ce,
    sampled_povm_ce,
    sampled_softcap_linear_ce,
)
from qdlblocks.layers.fp8_linear import (
    FP8LinearWrapper,
    convert_linear_layers_to_fp8,
    is_fp8_scaled_mm_supported,
)
from qdlblocks.layers.isometric import (
    YaRNRotarySchedule,
    apply_partial_key_offset,
)
from qdlblocks.layers.mudd_and_gates import (
    ForwardSmearGate,
    InputChannelGate,
    MUDDDynamicSkip,
    apply_gated_xsa,
)
from qdlblocks.layers.ngram_hash import (
    HashedNgramEmbedding,
    compute_ngram_hashes_and_signs,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks.optim.muon import StiefelMuon
from qdlblocks.optim.tail_ema import TailWeightAverager


def _make_contextual_cluster_shards(
    vocab_size: int = 512,
    active_vocab: int = 384,
    num_clusters: int = 16,
    tokens_per_shard: int = 32768,
    num_shards: int = 3,
    seed: int = 42,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Generate multi-shard Markov-cluster token streams with unseen upper token IDs (`[active_vocab, vocab_size)`)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cluster_size = active_vocab // num_clusters

    shards: list[torch.Tensor] = []
    for _ in range(num_shards):
        # Sample cluster sequence where tokens within a 4-token span share the same latent topic cluster
        num_spans = tokens_per_shard // 4
        span_clusters = torch.randint(0, num_clusters, (num_spans, 1), generator=gen)
        offsets = torch.randint(0, cluster_size, (num_spans, 4), generator=gen)
        toks = (span_clusters * cluster_size + offsets).reshape(-1)
        shards.append(toks)

    # Collocated pairs (same cluster) vs random cross-cluster pairs
    colloc_a = []
    colloc_b = []
    rand_a = []
    rand_b = []
    for c in range(num_clusters):
        base = c * cluster_size
        next_base = ((c + 1) % num_clusters) * cluster_size
        for i in range(min(cluster_size - 1, 8)):
            colloc_a.append(base + i)
            colloc_b.append(base + i + 1)
            rand_a.append(base + i)
            rand_b.append(next_base + i)

    colloc_pairs = torch.tensor(list(zip(colloc_a, colloc_b)), dtype=torch.long)
    random_pairs = torch.tensor(list(zip(rand_a, rand_b)), dtype=torch.long)
    return shards, colloc_pairs, random_pairs


def test_ppmi_svd_cooccurrence_init_ortho_and_collocation_snr(tmp_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 512
    embed_dim = 64
    shards, colloc_pairs, random_pairs = _make_contextual_cluster_shards(
        vocab_size=vocab_size,
        active_vocab=384,  # Tokens [384..511] are completely unseen padding/rare tokens
        num_clusters=16,
        tokens_per_shard=24576,
        num_shards=3,
        seed=1337,
    )

    cache_file = tmp_path / "ppmi_svd_cache.pt"
    stats = compute_ppmi_svd_embeddings(
        token_arrays=shards,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_context_buckets=512,
        window_size=4,
        cds_alpha=0.75,
        shift_kappa=1.0,
        svd_oversample=24,
        svd_niter=3,
        chunk_tokens=8192,
        seed=1337,
        device=device,
        cache_path=cache_file,
    )
    u_spec = stats["embeddings"]
    assert cache_file.exists()
    assert u_spec.shape == (vocab_size, embed_dim)
    assert not torch.isnan(u_spec).any()
    assert stats["total_tokens"] == 3 * 24576
    assert stats["loaded_from_cache"] is False

    # Verify all rows (including unseen tokens 384..511) have unit L2 norm 1.0
    row_norms = u_spec.float().norm(dim=-1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)

    # Verify deterministic cache reload matches bit-for-bit
    stats_cached = compute_ppmi_svd_embeddings(
        token_arrays=shards,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_context_buckets=512,
        cache_path=cache_file,
        device=device,
    )
    u_cached = stats_cached["embeddings"]
    assert stats_cached["loaded_from_cache"] is True
    assert torch.allclose(u_spec.cpu(), u_cached.cpu(), atol=1e-6)

    # Compare isotropic vs PPMI-SVD initialization on QuantumBornGPT with ortho_decouple=True
    torch.manual_seed(2026)
    model_iso = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        block_size=64,
        use_ortho_decouple=True,
        use_purified_attn=True,
    ).to(device)
    tokens_1d = torch.cat(shards, dim=0)
    metrics_iso = compute_collocation_coherence_metrics(
        tokens_1d,
        model_iso.wte.weight,
        top_k=128,
        seed=1337,
    )

    torch.manual_seed(2026)
    model_ppmi = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        block_size=64,
        use_ortho_decouple=True,
        use_purified_attn=True,
    ).to(device)
    init_info = initialize_quantum_born_with_ppmi_svd(
        model_ppmi,
        u_spec,
        alpha=0.80,
        seed=1337,
    )
    assert init_info["max_self_overlap"] < 1e-6
    assert abs(init_info["min_row_norm"] - 1.0) < 1e-5
    assert abs(init_info["max_row_norm"] - 1.0) < 1e-5

    # Verify explicit Step-0 input-output orthogonality across all V rows
    w_in = torch.nn.functional.normalize(model_ppmi.wte.weight.detach().float(), p=2.0, dim=-1)
    m_seed = torch.nn.functional.normalize(model_ppmi.povm_readout.povm_head.detach().float(), p=2.0, dim=-1)
    max_overlap = float((w_in * m_seed).sum(dim=-1).abs().max().item())
    assert max_overlap < 1e-6

    metrics_ppmi = compute_collocation_coherence_metrics(
        tokens_1d,
        model_ppmi.wte.weight,
        top_k=128,
        seed=1337,
    )
    assert metrics_ppmi["cos_ratio"] >= 3.0
    assert metrics_ppmi["colloc_purified_born_energy"] >= 1.5 * metrics_iso["colloc_purified_born_energy"]

    # Also verify ClassicalModdedGPT initialization preserves RMS scale
    c_model = ClassicalModdedGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        block_size=64,
    ).to(device)
    c_info = initialize_classical_with_ppmi_svd(c_model, u_spec, alpha=0.5, seed=1337)
    assert c_info["wte_rms"] > 0.0
    assert not torch.isnan(c_model.wte.weight).any()


def test_hashed_ngram_embedding_sign_trick_and_shapes() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, d = 4, 32, 64
    ngram_vocab = 1024
    sign_pool_rows = 512
    idx = torch.randint(0, 500, (b, t), device=device)

    bi_idx, tri_idx, bi_sgn_id, tri_sgn_id = compute_ngram_hashes_and_signs(
        idx, ngram_vocab_size=ngram_vocab, sign_pool_rows=sign_pool_rows
    )
    assert bi_idx.shape == (b, t)
    assert tri_idx.shape == (b, t)
    assert bi_sgn_id.shape == (b, t)
    assert tri_sgn_id.shape == (b, t)
    assert ((bi_idx >= 0) & (bi_idx < ngram_vocab)).all()
    assert ((tri_idx >= 0) & (tri_idx < ngram_vocab)).all()
    assert ((bi_sgn_id >= 0) & (bi_sgn_id < sign_pool_rows)).all()
    assert ((tri_sgn_id >= 0) & (tri_sgn_id < sign_pool_rows)).all()

    # Verify Step-0 exact zero output when zero_init=True
    ngram_emb = HashedNgramEmbedding(
        ngram_vocab_size=ngram_vocab,
        embed_dim=d,
        num_layers=4,
        sign_pool_rows=sign_pool_rows,
        inject_scale_init=0.05,
        zero_init=True,
    ).to(device)
    assert set(ngram_emb.sign_pool.unique().cpu().tolist()).issubset({-1.0, 1.0})
    out0 = ngram_emb(idx)
    assert out0.shape == (b, t, d)
    assert float(out0.abs().max().item()) == 0.0

    # Verify gradient flows cleanly to table.weight on backward
    loss = (out0 * torch.randn_like(out0)).sum()
    loss.backward()
    assert ngram_emb.table.weight.grad is not None
    assert float(ngram_emb.table.weight.grad.abs().sum().item()) > 0.0

    # Verify sign cancellation across distinct n-grams colliding in a small table
    small_pairs = torch.arange(0, 2048, dtype=torch.long, device=device).view(32, 64)
    _, _, sgn_bi_id, sgn_tri_id = compute_ngram_hashes_and_signs(
        small_pairs, ngram_vocab_size=64, sign_pool_rows=sign_pool_rows
    )
    sp = ngram_emb.sign_pool.to(device=device)
    bi_signs = sp[sgn_bi_id]
    tri_signs = sp[sgn_tri_id]
    assert abs(float(bi_signs.mean().item())) < 0.05
    assert abs(float(tri_signs.mean().item())) < 0.05


def test_polar_express_normuon_twin_rail_and_tail_ema() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(ANVIL_QUINTIC_MAPS) == 6

    torch.manual_seed(99)
    # Create an ill-conditioned rectangular matrix with condition number kappa(G) = 60 >= 50
    u_true, _ = torch.linalg.qr(torch.randn(128, 64, device=device))
    v_true, _ = torch.linalg.qr(torch.randn(64, 64, device=device))
    s_vals = torch.logspace(0.0, math.log10(1.0 / 60.0), 64, device=device)
    g_mat = (u_true * s_vals.unsqueeze(0)) @ v_true.T

    x_pe = zeropower_via_polarexpress5(g_mat, steps=6).float()
    x_ns = zeropower_via_newtonschulz5(g_mat, steps=5).float()

    sv_pe = torch.linalg.svdvals(x_pe)
    sv_ns = torch.linalg.svdvals(x_ns)
    rmse_pe = float((sv_pe - 1.0).square().mean().sqrt().item())
    rmse_ns = float((sv_ns - 1.0).square().mean().sqrt().item())
    assert rmse_pe < 0.08
    assert rmse_pe < rmse_ns

    # Verify batched Polar Express matches single-matrix Polar Express
    g_batch = torch.stack([g_mat, g_mat * 0.5], dim=0)
    x_pe_batch = zeropower_via_polarexpress5_batched(g_batch, steps=6).float()
    assert torch.allclose(x_pe_batch[0], x_pe, atol=2e-2)

    # Verify NorMuon per-lane energy equalizer preserves Frobenius norm within < 1e-5
    u_skewed = x_pe.clone()
    u_skewed[:, :16] *= 4.0
    pre_frob = float(u_skewed.norm().item())
    u_eq, lane_state = normuon_equalize(u_skewed, None, beta2=0.95)
    post_frob = float(u_eq.norm().item())
    assert abs(pre_frob - post_frob) / max(1.0, pre_frob) < 1e-5
    assert lane_state.shape == (1, 64)

    # Verify StiefelMuon with polar_method="polar_express", use_normuon=True, use_twin_rail=True, decay_mode="cautious"
    w1 = torch.nn.Parameter(torch.randn(64, 32, device=device))
    w2 = torch.nn.Parameter(torch.randn(64, 32, device=device))
    opt = StiefelMuon(
        [w1, w2],
        lr=0.02,
        polar_method="polar_express",
        use_normuon=True,
        normuon_beta2=0.95,
        use_twin_rail=True,
        rail_fast_beta=0.85,
        rail_slow_beta=0.98,
        rail_fast_weight=0.4385,
        rail_engage_step=2,
        weight_decay=0.01,
        decay_mode="cautious",
        batched_ns=True,
    )
    for _ in range(3):
        opt.zero_grad()
        loss = (w1.square().sum() + w2.square().sum()) * 0.5
        loss.backward()
        opt.step()
    assert "slow_momentum_buffer" in opt.state[w1]
    assert "lane_energy" in opt.state[w1]

    # Verify TailWeightAverager Frobenius & spherical decontraction and lossless context manager swap
    toy = QuantumBornGPT(
        vocab_size=128,
        embed_dim=32,
        seed_dim=32,
        num_heads=4,
        num_layers=2,
        block_size=16,
    ).to(device)
    averager = TailWeightAverager(
        toy,
        beta=0.90,
        start_step=2,
        preserve_frobenius_norm=True,
        spherical_param_patterns=("wte", "povm_head"),
    )
    orig_wte = toy.wte.weight.detach().clone()
    for s in range(1, 5):
        with torch.no_grad():
            toy.wte.weight.copy_(
                torch.nn.functional.normalize(torch.randn_like(toy.wte.weight), dim=-1)
            )
        active = averager.step(toy, current_step=s)
        if s < 2:
            assert not active
        else:
            assert active

    live_snapshot = toy.wte.weight.detach().clone()
    with averager.average_parameters(toy):
        ema_wte = toy.wte.weight.detach().clone()
        ema_row_norms = ema_wte.float().norm(dim=-1)
        # Spherical parameters must have exact unit L2 row norms after decontraction
        assert torch.allclose(ema_row_norms, torch.ones_like(ema_row_norms), atol=1e-5)
    # Restored live weights must match live_snapshot bit-for-bit
    assert torch.equal(toy.wte.weight, live_snapshot)
    assert not torch.equal(live_snapshot, orig_wte)


def test_mudd_gates_xsa_bos_align_and_sampled_softmax() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, h, dh = 2, 16, 4, 16
    d = h * dh
    x = torch.randn(b, t, d, device=device)

    # 1. InputChannelGate Step-0 exact identity (g(x) == 1.0)
    gate = InputChannelGate(num_heads=h, gate_channels=6, head_first=True).to(device)
    g0 = gate(x)
    assert g0.shape == (b, h, t, 1)
    assert torch.allclose(g0, torch.ones_like(g0), atol=1e-6)

    # 2. apply_gated_xsa Step-0 identity and non-zero projection removal when alpha != 0
    y_heads = torch.randn(b, h, t, dh, device=device)
    v_heads = torch.randn(b, h, t, dh, device=device)
    alpha0 = torch.zeros(h, device=device)
    y_xsa0 = apply_gated_xsa(y_heads, v_heads, alpha0)
    assert torch.allclose(y_xsa0, y_heads, atol=1e-6)
    alpha_full = torch.full((h,), 10.0, device=device)  # tanh(10) ~ 1.0
    y_xsa1 = apply_gated_xsa(v_heads, v_heads, alpha_full)
    assert float(y_xsa1.norm().item()) < 1e-3

    # 3. MUDDDynamicSkip Step-0 identity (spherical=False and spherical=True)
    mudd_lin = MUDDDynamicSkip(embed_dim=d, num_sources=2, hidden_dim=32, spherical=False).to(device)
    s0 = torch.randn(b, t, d, device=device)
    s1 = torch.randn(b, t, d, device=device)
    out_lin = mudd_lin(x, [s0, s1])
    assert torch.allclose(out_lin, x, atol=1e-5)

    x_sph = torch.nn.functional.normalize(x, p=2.0, dim=-1)
    s0_sph = torch.nn.functional.normalize(s0, p=2.0, dim=-1)
    mudd_sph = MUDDDynamicSkip(embed_dim=d, num_sources=1, hidden_dim=32, spherical=True).to(device)
    out_sph = mudd_sph(x_sph, [s0_sph])
    assert torch.allclose(out_sph, x_sph, atol=1e-5)

    # 4. ForwardSmearGate Step-0 identity
    smear = ForwardSmearGate(embed_dim=d, gate_dim=12, spherical=False).to(device)
    assert torch.allclose(smear(x), x, atol=1e-6)

    # 5. YaRNRotarySchedule & apply_partial_key_offset
    yarn = YaRNRotarySchedule(head_dim=dh, max_seq_len=128, base_attn_scale=0.12).to(device)
    new_scale = yarn.apply_yarn(old_window=8, new_window=16)
    assert new_scale > 0.12
    q_rot, k_rot = yarn(y_heads, v_heads)
    assert q_rot.shape == y_heads.shape and k_rot.shape == v_heads.shape
    k_shifted = apply_partial_key_offset(v_heads, rotary_dim=8)
    assert torch.equal(k_shifted[:, :, 1:, 8:], v_heads[:, :, :-1, 8:])
    assert torch.equal(k_shifted[:, :, :, :8], v_heads[:, :, :, :8])

    # 6. find_bos_aligned_starts & compute_tapered_mtp_weight
    stream = np.full((1024,), 10, dtype=np.uint16)
    stream[99] = 50256
    stream[299] = 50256
    starts = find_bos_aligned_starts(stream, seq_len=64, batch_size=8, eot_token_id=50256)
    assert set(starts.tolist()).issubset({0, 100, 300})
    assert abs(compute_tapered_mtp_weight(0, 100, 0.25, 0.65) - 0.25) < 1e-6
    assert compute_tapered_mtp_weight(70, 100, 0.25, 0.65) == 0.0

    # 7. Sampled-Softmax Candidate Set & Cross-Entropy (Classical + POVM with log_prior)
    vocab_small = 200  # Not coprime with 10, tests coprime fallback
    targets = torch.randint(0, vocab_small, (b, t), device=device)
    cand_ids, remapped = build_sampled_candidate_set(
        targets, vocab_size=vocab_small, num_candidates=64, step=3
    )
    assert cand_ids.shape == (64,)
    assert torch.equal(cand_ids[remapped], targets.reshape(-1))

    # Full candidate set (num_candidates >= vocab_size) must match exact direct_chunked CE
    lm_w = torch.randn(vocab_small, d, device=device, requires_grad=True)
    x_in = torch.randn(b, t, d, device=device, requires_grad=True)
    loss_exact = direct_chunked_softcap_linear_ce(x_in, lm_w, targets, softcap=15.0)
    loss_samp_full = sampled_softcap_linear_ce(
        x_in, lm_w, targets, num_candidates=vocab_small, step=0, softcap=15.0
    )
    assert torch.allclose(loss_exact, loss_samp_full, atol=1e-5)

    loss_samp_64 = sampled_softcap_linear_ce(
        x_in, lm_w, targets, num_candidates=64, step=1, softcap=15.0
    )
    loss_samp_64.backward()
    assert lm_w.grad is not None and x_in.grad is not None

    # POVM sampled CE with log_prior
    psi = torch.nn.functional.normalize(torch.randn(b, t, d, device=device), dim=-1).requires_grad_(True)
    dm_seed = torch.nn.functional.normalize(torch.randn(vocab_small, d, device=device), dim=-1).requires_grad_(True)
    wte_w = torch.nn.functional.normalize(torch.randn(vocab_small, d, device=device), dim=-1)
    w_lift = torch.eye(d, device=device)
    r_ortho = torch.eye(d, device=device)
    inv_t = torch.tensor(10.0, device=device)
    log_prior = torch.randn(vocab_small, device=device)

    povm_exact = direct_chunked_povm_ce(
        psi.reshape(-1, d), dm_seed, inv_t, targets.reshape(-1), log_prior=log_prior
    )
    povm_samp_full = sampled_povm_ce(
        psi, dm_seed, wte_w, w_lift, r_ortho, inv_t, targets,
        num_candidates=vocab_small, readout_mode="asymmetric", log_prior=log_prior,
    )
    assert torch.allclose(povm_exact, povm_samp_full, atol=1e-5)

    povm_samp_64 = sampled_povm_ce(
        psi, dm_seed, wte_w, w_lift, r_ortho, inv_t, targets,
        num_candidates=64, step=2, readout_mode="asymmetric", log_prior=log_prior,
    )
    povm_samp_64.backward()
    assert dm_seed.grad is not None and psi.grad is not None

    # 8. FP8LinearWrapper forward/backward
    lin = torch.nn.Linear(64, 64, bias=False).to(device)
    fp8_lin = FP8LinearWrapper(lin, enabled=True).to(device)
    x_fp8 = torch.randn(4, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    out_fp8 = fp8_lin(x_fp8)
    assert out_fp8.shape == (4, 16, 64)
    out_fp8.sum().backward()
    assert fp8_lin.weight.grad is not None and x_fp8.grad is not None

    # 9. Full QuantumBornGPT & ClassicalModdedGPT forward/backward with all Plan 021 flags
    q_full = QuantumBornGPT(
        vocab_size=256,
        embed_dim=64,
        seed_dim=64,
        num_heads=4,
        num_layers=4,
        block_size=32,
        use_ngram_hash=True,
        ngram_vocab_size=512,
        use_mudd_and_gates=True,
        use_smear_gate=True,
        use_partial_key_offset=True,
    ).to(device)
    _, q_loss = q_full(targets, targets)
    assert q_loss is not None
    q_loss.backward()

    c_full = ClassicalModdedGPT(
        vocab_size=256,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        block_size=32,
        leaderboard_exact=True,
        use_ngram_hash=True,
        ngram_vocab_size=512,
        use_mudd_and_gates=True,
        use_smear_gate=True,
        use_partial_key_offset=True,
    ).to(device)
    convert_linear_layers_to_fp8(c_full, enabled=True)
    _, c_loss = c_full(targets, targets)
    assert c_loss is not None
    c_loss.backward()

    # 10. Validate configs/wave13_fleet.json & model_card schema
    repo_root = Path(__file__).resolve().parents[1]
    w13_path = repo_root / "configs" / "wave13_fleet.json"
    assert w13_path.exists()
    assert "W13" in WAVE_DEFAULT_LABELS
    w13_data = json.loads(w13_path.read_text(encoding="utf-8"))
    assert w13_data.get("wave_id", "W13") == "W13"
    assert "Wave 13" in w13_data["wave"]
    assert len(w13_data["runs"]) == 5
    for entry in w13_data["runs"]:
        card = build_model_card_from_manifest_entry(
            entry,
            wave_id="W13",
            wave_description=w13_data.get("description", w13_data["wave"]),
            plan_id=w13_data["plan_id"],
        )
        errs = validate_model_card(card)
        assert errs == [], f"Model card validation errors for {entry['key']}: {errs}"
        arch_sec, opt_sec, _ = parse_cli_args_to_card_sections(entry["cli_args"])
        if "--ppmi-svd-init" in entry["cli_args"]:
            assert arch_sec["ppmi_svd_init"] is True
        if "--ngram-hash" in entry["cli_args"]:
            assert arch_sec["use_ngram_hash"] is True
        if "--polar-express" in entry["cli_args"]:
            assert opt_sec["polar_method"] == "polar_express"

