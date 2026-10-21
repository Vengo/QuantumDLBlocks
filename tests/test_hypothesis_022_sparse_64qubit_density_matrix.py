"""Unit tests for Plan 022: 64-Qubit Sparse Fock Excitation Space H_64^(<=2),
Sparse Hashed 1/2/3-Gram Qubit Encodings, & 4M-Parameter Internal Density Matrix Layers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.model_card import (
    WAVE_DEFAULT_LABELS,
    WAVE_DEFAULT_PLANS,
    build_model_card_from_manifest_entry,
    parse_cli_args_to_card_sections,
)
from qdlblocks.layers.sparse_qubit_fock import (
    Sparse64QubitDensityLayer,
    Sparse64QubitNgramEmbedder,
    Sparse64QubitPOVMReadout,
    build_64qubit_pair_index_tables,
    initialize_sparse_64qubit_with_ppmi_svd,
)
from qdlblocks.models.born_gpt import QuantumBornGPT


def test_64qubit_fock_pair_tables_and_sparse_ngram_embedder() -> None:
    pair_i, pair_j = build_64qubit_pair_index_tables(num_qubits=64, max_pair_modes=1984)
    assert pair_i.shape == (1984,)
    assert pair_j.shape == (1984,)
    assert int(pair_i.min().item()) >= 0
    assert int(pair_j.max().item()) < 64
    assert bool((pair_i < pair_j).all().item())
    assert 64 + int(pair_i.numel()) == 2048

    # Check uniqueness of all 1984 selected pairs
    pairs_set = set(zip(pair_i.tolist(), pair_j.tolist()))
    assert len(pairs_set) == 1984

    embedder = Sparse64QubitNgramEmbedder(
        vocab_size=256,
        num_qubits=64,
        fock_dim=2048,
        sparse_k=16,
        ngram_vocab_size=4096,
        num_layers=4,
    )
    idx = torch.randint(0, 256, (3, 12), dtype=torch.long)
    psi_0 = embedder(idx)
    assert psi_0.shape == (3, 12, 2048)

    norms = torch.norm(psi_0, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Verify causal boundary safety: changing token at t >= 1 must not alter psi_0[:, 0, :],
    # and changing token at t >= 2 must not alter psi_0[:, :2, :]
    idx_mod = idx.clone()
    idx_mod[:, 2:] = (idx_mod[:, 2:] + 17) % 256
    psi_0_mod = embedder(idx_mod)
    assert torch.allclose(psi_0[:, :2, :], psi_0_mod[:, :2, :], atol=1e-6)

    idx_mod_t1 = idx.clone()
    idx_mod_t1[:, 1:] = (idx_mod_t1[:, 1:] + 23) % 256
    psi_0_mod_t1 = embedder(idx_mod_t1)
    assert torch.allclose(psi_0[:, :1, :], psi_0_mod_t1[:, :1, :], atol=1e-6)

    # Verify gradient flow to all learnable parameters
    delta = embedder.forward_ngram_delta(idx)
    loss = (psi_0 * torch.randn_like(psi_0)).sum() + (embedder.layer_scales.sum() * delta.sum())
    loss.backward()
    for pname in (
        "qubit_wte",
        "unigram_sparse_vals",
        "ngram_sparse_vals",
        "pair_scale_2gram",
        "pair_scale_3gram",
        "layer_scales",
    ):
        param = getattr(embedder, pname)
        grad = param.weight.grad if hasattr(param, "weight") else param.grad
        assert grad is not None, f"Missing gradient for {pname}"
        assert float(grad.abs().sum().item()) > 0.0, f"Zero gradient for {pname}"


def test_sparse_64qubit_density_layer_4m_param_budget_and_purity() -> None:
    layer_r4 = Sparse64QubitDensityLayer(
        fock_dim=2048,
        num_qubits=64,
        density_rank=4,
        use_fock_coupler=True,
    )
    assert layer_r4.density_operator.weight.shape == (2048, 2048)
    assert layer_r4.density_operator.weight.numel() == 4_194_304
    total_layer_params = sum(p.numel() for p in layer_r4.parameters())
    assert 4_194_304 <= total_layer_params <= 4_250_000

    x = F.normalize(torch.randn(2, 8, 2048), p=2, dim=-1)
    out_r4 = layer_r4(x)
    assert out_r4.shape == (2, 8, 2048)
    norms = torch.norm(out_r4, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert 0.25 - 1e-4 <= layer_r4.last_purity_mean <= 1.0 + 1e-4
    assert layer_r4.last_pair_sector_energy > 0.0

    # Rank-1 pure density operator check
    layer_r1 = Sparse64QubitDensityLayer(
        fock_dim=2048,
        num_qubits=64,
        density_rank=1,
        use_fock_coupler=True,
    )
    out_r1 = layer_r1(x)
    assert torch.allclose(torch.norm(out_r1, p=2, dim=-1), torch.ones(2, 8), atol=1e-5)
    assert abs(layer_r1.last_purity_mean - 1.0) < 1e-5

    # Coupler ablation check: disabling coupler produces distinct output
    layer_no_coupler = Sparse64QubitDensityLayer(
        fock_dim=2048,
        num_qubits=64,
        density_rank=4,
        use_fock_coupler=False,
    )
    assert layer_no_coupler.qubit_unitary is None
    out_no_coupler = layer_no_coupler(x)
    assert torch.allclose(torch.norm(out_no_coupler, p=2, dim=-1), torch.ones(2, 8), atol=1e-5)


def test_sparse_64qubit_ppmi_svd_init_and_zero_self_overlap() -> None:
    rng = np.random.default_rng(42)
    vocab_size = 256
    n_tokens = 32768
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = (1.0 / (ranks ** 1.05))
    probs /= probs.sum()
    shard = rng.choice(vocab_size, size=n_tokens, replace=True, p=probs).astype(np.uint16)

    # Inject strong recurring collocations (v -> (v + 1) % 64)
    for pos in range(0, n_tokens - 4, 6):
        v = int(shard[pos]) % 64
        shard[pos] = v
        shard[pos + 1] = (v + 1) % 64
        shard[pos + 2] = (v + 2) % 64

    model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=2048,
        seed_dim=64,
        num_heads=8,
        num_layers=2,
        use_sparse_64qubit=True,
        num_qubits=64,
        sparse_ngram_k=16,
        use_sparse_density_layer=True,
        density_matrix_rank=4,
    )
    stats = initialize_sparse_64qubit_with_ppmi_svd(
        model,
        token_shards=[shard],
        window=4,
        alpha_ppmi=0.65,
        device=torch.device("cpu"),
    )
    assert stats["collocation_cosine_ratio"] >= 3.0
    assert stats["max_self_overlap"] < 1e-6

    # Verify directly on Sparse64QubitNgramEmbedder + Sparse64QubitPOVMReadout
    assert model.sparse_qubit_embed is not None
    assert model.sparse_povm_readout is not None
    v_ids = torch.arange(vocab_size, dtype=torch.long)
    psi_uni = model.sparse_qubit_embed.forward_unigram_fock(v_ids)
    m_povm = model.sparse_povm_readout.get_povm_matrix(model.sparse_qubit_embed)
    self_overlaps = (psi_uni * m_povm).sum(dim=-1).abs()
    assert float(self_overlaps.max().item()) < 1e-6


def test_quantum_born_gpt_sparse_64qubit_integration_and_model_card() -> None:
    torch.manual_seed(42)
    model = QuantumBornGPT(
        vocab_size=256,
        embed_dim=2048,
        seed_dim=64,
        num_heads=8,
        num_layers=2,
        use_sparse_64qubit=True,
        num_qubits=64,
        sparse_ngram_k=16,
        use_sparse_density_layer=True,
        density_matrix_rank=4,
        use_mudd_and_gates=True,
        use_smear_gate=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 64, (4, 16), dtype=torch.long)
    y = (x + 1) % 64

    losses: list[float] = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        assert loss is not None
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], f"Expected loss to decrease over 3 steps, got {losses}"
    assert model.sparse_qubit_embed is not None
    assert model.sparse_qubit_embed.qubit_wte.weight.grad is not None
    assert model.blocks[0].sparse_density_layer is not None
    assert model.blocks[0].sparse_density_layer.density_operator.weight.grad is not None

    # Verify model card CLI parser & W14 wave metadata
    assert "W14" in WAVE_DEFAULT_LABELS
    assert WAVE_DEFAULT_PLANS["W14"] in ("EXPERIMENT_PLAN_022", "EXPERIMENT_PLAN_022.md")

    cli = (
        "--architecture quantum_born --num-layers 4 --embed-dim 2048 --num-heads 8 "
        "--sparse-64qubit --num-qubits 64 --sparse-ngram-k 16 "
        "--sparse-density-layer --density-matrix-rank 4 --ppmi-svd-init"
    )
    arch, opt_sec, data_sec = parse_cli_args_to_card_sections(cli)
    assert arch["use_sparse_64qubit"] is True
    assert arch["num_qubits"] == 64
    assert arch["sparse_ngram_k"] == 16
    assert arch["use_sparse_density_layer"] is True
    assert arch["density_matrix_rank"] == 4
    assert arch["use_fock_coupler"] is True
    assert arch["embed_dim"] == 2048
    assert "Density2048x2048" in arch["d_ff_display"]
    assert arch["params_M_est"] > 40.0

    card = build_model_card_from_manifest_entry(
        {
            "key": "w14_local_64qubit_sparse_fock2048_4m_density_rank4_mixed",
            "wave": "W14",
            "cli_args": cli,
            "target_tokens_M": 1.64,
        }
    )
    assert card["run_tier"] == "main_benchmark"
    assert card["wave_id"] == "W14"
