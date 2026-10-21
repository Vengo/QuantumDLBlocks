"""Unit tests for Plan 023: 16-Qubit Multi-Register Amplitudes (32 = 2^5 States/Register)
& 256-State Interacting Quantum Hierarchy.
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
from qdlblocks.layers.hierarchical_qubit import (
    Hierarchical16QubitEmbedder,
    Hierarchical16QubitPOVMReadout,
    initialize_hierarchical_16qubit_with_ppmi_svd,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import (
    StiefelMuon,
    polar_snap_weights,
    split_muon_adamw_params,
)
from qdlblocks.optim.tail_ema import TailWeightAverager


def test_hierarchical_16qubit_embedder_geometry_and_causality() -> None:
    torch.manual_seed(42)
    embedder = Hierarchical16QubitEmbedder(
        vocab_size=512,
        num_registers=16,
        reg_dim=32,
        embed_dim=1024,
        num_higher_states=256,
        num_layers=4,
        use_ortho_decouple=True,
    )
    assert embedder.unigram_qubits.shape == (512, 16, 32)
    assert embedder.higher_states is not None
    assert embedder.higher_states.shape == (256, 16, 32)
    assert embedder.higher_interaction is not None
    assert embedder.higher_interaction.shape == (16, 256, 256)
    assert embedder.higher_sub_unitary.shape == (16, 32, 32)

    # Check sub-register normalization ||psi_v^(q)||_2 == 0.25 +- 1e-5 and flattened 512-D norm == 1.0 +- 1e-5
    u_norm = embedder.get_normalized_unigram_qubits()
    reg_norms = torch.norm(u_norm, p=2, dim=-1)
    assert torch.allclose(reg_norms, torch.full_like(reg_norms, 0.25), atol=1e-5)
    flat_norms = torch.norm(u_norm.reshape(512, 512), p=2, dim=-1)
    assert torch.allclose(flat_norms, torch.ones_like(flat_norms), atol=1e-5)

    # Check raw parameter sub-register norms as well
    raw_reg_norms = torch.norm(embedder.unigram_qubits, p=2, dim=-1)
    assert torch.allclose(raw_reg_norms, torch.full_like(raw_reg_norms, 0.25), atol=1e-5)

    idx = torch.randint(0, 512, (3, 12), dtype=torch.long)
    psi_0, p_seq = embedder(idx, return_p_seq=True)
    assert psi_0.shape == (3, 12, 1024)
    assert p_seq is not None
    assert p_seq.shape == (3, 12, 256)

    psi_norms = torch.norm(psi_0, p=2, dim=-1)
    assert torch.allclose(psi_norms, torch.ones_like(psi_norms), atol=1e-5)

    # Verify strict causality for t >= 1
    for t_cut in (1, 2, 5):
        idx_mod = idx.clone()
        idx_mod[:, t_cut:] = (idx_mod[:, t_cut:] + 37) % 512
        psi_0_mod = embedder(idx_mod)
        assert torch.allclose(psi_0[:, :t_cut, :], psi_0_mod[:, :t_cut, :], atol=1e-6), (
            f"Causality violation at t_cut={t_cut}"
        )

    # Verify per-block bridge output norm & gradient flow to all learnable parameters
    x_bridged = embedder.forward_block_bridge(psi_0, p_seq, layer_idx=0)
    assert x_bridged.shape == (3, 12, 1024)
    assert torch.allclose(torch.norm(x_bridged, p=2, dim=-1), torch.ones(3, 12), atol=1e-5)

    loss = (x_bridged * torch.randn_like(x_bridged)).sum() + (embedder.block_bridge_gates.sum() * 1e-3)
    loss.backward()
    for pname in (
        "unigram_qubits",
        "higher_states",
        "higher_interaction",
        "higher_sub_unitary",
        "horizon_weights",
        "lift_mix_gate",
        "block_bridge_gates",
    ):
        param = getattr(embedder, pname)
        assert param is not None, f"Missing parameter {pname}"
        assert param.grad is not None, f"Missing gradient for {pname}"
        assert float(param.grad.abs().sum().item()) > 0.0, f"Zero gradient for {pname}"


def test_hierarchical_16qubit_povm_readout_and_zero_self_overlap() -> None:
    torch.manual_seed(42)
    V = 256
    embedder = Hierarchical16QubitEmbedder(
        vocab_size=V,
        num_registers=16,
        reg_dim=32,
        embed_dim=1024,
        num_higher_states=64,
        num_layers=2,
        use_ortho_decouple=True,
    )
    readout = Hierarchical16QubitPOVMReadout(
        vocab_size=V,
        num_registers=16,
        reg_dim=32,
        embed_dim=1024,
        num_higher_states=64,
        use_ortho_decouple=True,
    )

    m_povm = readout.get_coupled_povm_matrix(embedder)
    assert m_povm.shape == (V, 1024)

    # Verify unit norm of M_v and exclusive odd-coordinate support (even coordinates == 0)
    m_norms = torch.norm(m_povm, p=2, dim=-1)
    assert torch.allclose(m_norms, torch.ones_like(m_norms), atol=1e-5)
    assert float(m_povm[:, 0::2].abs().max().item()) < 1e-6

    # Verify Step-0 overlap between pure unigram input states (0::2) and coupled POVM axes (1::2) is < 1e-6
    v_ids = torch.arange(V, dtype=torch.long)
    psi_uni = embedder.forward_unigram_1024(v_ids)
    assert float(psi_uni[:, 1::2].abs().max().item()) < 1e-6
    self_overlaps = (psi_uni * m_povm).sum(dim=-1).abs()
    assert float(self_overlaps.max().item()) < 1e-6

    # Verify direct_chunked_ce=True and direct_chunked_ce=False match and propagate gradients
    model = QuantumBornGPT(
        vocab_size=V,
        embed_dim=1024,
        seed_dim=512,
        num_heads=8,
        num_layers=2,
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=32,
        num_higher_states=64,
    )
    x = torch.randint(0, V, (2, 8), dtype=torch.long)
    y = torch.randint(0, V, (2, 8), dtype=torch.long)

    _, loss_mat = model(x, targets=y, direct_chunked_ce=False)
    _, loss_dir = model(x, targets=y, direct_chunked_ce=True)
    assert loss_mat is not None and loss_dir is not None
    assert torch.allclose(loss_mat, loss_dir, atol=1e-4), (
        f"Materialized loss {loss_mat.item()} != direct chunked loss {loss_dir.item()}"
    )

    model.zero_grad(set_to_none=True)
    loss_dir.backward()
    assert model.hierarchical_readout is not None
    assert model.hierarchical_embed is not None
    assert model.hierarchical_readout.povm_odd_qubits.grad is not None
    assert float(model.hierarchical_readout.povm_odd_qubits.grad.abs().sum().item()) > 0.0
    assert model.hierarchical_readout.vocab_higher_affinity is not None
    assert model.hierarchical_readout.vocab_higher_affinity.grad is not None
    assert float(model.hierarchical_readout.vocab_higher_affinity.grad.abs().sum().item()) > 0.0
    assert model.hierarchical_embed.higher_states is not None
    assert model.hierarchical_embed.higher_states.grad is not None
    assert float(model.hierarchical_embed.higher_states.grad.abs().sum().item()) > 0.0


def test_hierarchical_16qubit_ppmi_svd_spectral_centroid_init() -> None:
    rng = np.random.default_rng(42)
    vocab_size = 512
    k_high = 64
    n_tokens = 40960
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = 1.0 / (ranks ** 1.02)
    probs /= probs.sum()
    shard = rng.choice(vocab_size, size=n_tokens, replace=True, p=probs).astype(np.uint16)

    # Implant 8 distinct multi-token concept blocks (each block of 8 tokens co-occurs tightly)
    for pos in range(0, n_tokens - 6, 6):
        block_id = int(pos // 6) % 8
        base_tok = block_id * 8
        shard[pos] = base_tok + (pos % 8)
        shard[pos + 1] = base_tok + ((pos + 1) % 8)
        shard[pos + 2] = base_tok + ((pos + 2) % 8)
        shard[pos + 3] = base_tok + ((pos + 3) % 8)

    model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=1024,
        seed_dim=512,
        num_heads=8,
        num_layers=2,
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=32,
        num_higher_states=k_high,
    )
    stats = initialize_hierarchical_16qubit_with_ppmi_svd(
        model,
        token_shards=[shard],
        window=4,
        alpha_ppmi=0.65,
        kmeans_iters=15,
        device=torch.device("cpu"),
    )
    assert stats["collocation_cosine_ratio"] >= 3.0, f"Got {stats['collocation_cosine_ratio']}"
    assert stats["hierarchy_concept_sim_ratio"] >= 2.5, f"Got {stats['hierarchy_concept_sim_ratio']}"
    assert stats["max_self_overlap"] < 1e-6, f"Got {stats['max_self_overlap']}"


def test_quantum_born_gpt_hierarchical_16qubit_integration_and_stiefel_muon() -> None:
    torch.manual_seed(42)
    vocab_size = 256
    model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=1024,
        seed_dim=512,
        num_heads=8,
        num_layers=2,
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=32,
        num_higher_states=256,
        use_mudd_and_gates=True,
        use_smear_gate=True,
    )
    muon_p, adamw_p = split_muon_adamw_params(model)
    assert len(muon_p) > 0
    assert any(p.shape == (16, 256, 256) for p in muon_p)
    assert any(p.shape == (16, 32, 32) for p in muon_p)

    opt_muon = StiefelMuon(
        muon_p,
        lr=0.015,
        momentum=0.95,
        ns_steps=5,
        stiefel_strength=0.02,
        polar_method="polar_express",
        use_normuon=True,
        cautious_wd=True,
    )
    opt_adam = torch.optim.AdamW(adamw_p, lr=3e-3, weight_decay=0.0)
    tail_averager = TailWeightAverager(model, start_fraction=0.0, decay=0.9)

    x = torch.randint(0, 64, (4, 16), dtype=torch.long)
    y = (x + 1) % 64

    losses: list[float] = []
    for step in range(3):
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, direct_chunked_ce=True)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        opt_muon.step()
        opt_adam.step()
        polar_snap_weights(model, polar_method="polar_express")
        tail_averager.update(step=step, total_steps=3)
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], f"Expected loss to decrease over 3 steps, got {losses}"

    # Verify sub-register norms == 0.25 and 3D unitary interaction matrices remain near-orthonormal
    assert model.hierarchical_embed is not None
    assert model.hierarchical_readout is not None
    for p_sub in (
        model.hierarchical_embed.unigram_qubits,
        model.hierarchical_embed.higher_states,
        model.hierarchical_readout.povm_odd_qubits,
    ):
        assert p_sub is not None
        norms = torch.norm(p_sub.detach(), p=2, dim=-1)
        assert torch.allclose(norms, torch.full_like(norms, 0.25), atol=1e-4)

    w_int = model.hierarchical_embed.higher_interaction
    assert w_int is not None
    gram_int = torch.bmm(w_int.detach().transpose(-2, -1), w_int.detach())
    eye_256 = torch.eye(256).unsqueeze(0).expand(16, -1, -1)
    assert float((gram_int - eye_256).abs().max().item()) < 0.05

    # Verify TailWeightAverager preserves sub-register geometry when applied
    with tail_averager.average_parameters(model):
        norms_ema = torch.norm(model.hierarchical_embed.unigram_qubits.detach(), p=2, dim=-1)
        assert torch.allclose(norms_ema, torch.full_like(norms_ema, 0.25), atol=1e-4)

    # Verify W15 model card parser & wave metadata
    assert "W15" in WAVE_DEFAULT_LABELS
    assert WAVE_DEFAULT_PLANS["W15"] in ("EXPERIMENT_PLAN_023", "EXPERIMENT_PLAN_023.md")

    cli = (
        "--architecture quantum_born --num-layers 6 --embed-dim 1024 --num-heads 8 "
        "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 "
        "--num-higher-states 256 --ortho-decouple --ppmi-svd-init"
    )
    arch, _, _ = parse_cli_args_to_card_sections(cli)
    assert arch["use_hierarchical_16qubit"] is True
    assert arch["qubit_registers"] == 16
    assert arch["qubit_amplitudes"] == 32
    assert arch["num_higher_states"] == 256
    assert arch["use_block_hierarchy_bridge"] is True
    assert arch["readout_mode"] == "hierarchical_povm"
    assert arch["seed_dim"] == 512

    card = build_model_card_from_manifest_entry(
        {
            "key": "w15_local_qborn_16x32_qubit_256hier_flagship",
            "wave": "W15",
            "cli_args": cli,
            "target_tokens_M": 1.64,
        }
    )
    assert card["run_tier"] == "main_benchmark"
    assert card["wave_id"] == "W15"


def test_modded_nanogpt_evaluate_validation_parity() -> None:
    from qdlblocks.benchmarks.fineweb_stream import (
        FINEWEB_GPT2_VOCAB_SIZE,
        FINEWEB_VAL_TOKENS,
        compute_modded_nanogpt_val_steps,
        iter_modded_nanogpt_val_batches,
    )
    from qdlblocks.layers.direct_chunked_ce import (
        direct_chunked_povm_ce,
        direct_chunked_softcap_linear_ce,
    )

    assert FINEWEB_VAL_TOKENS == 10_485_760
    assert FINEWEB_GPT2_VOCAB_SIZE == 50257
    assert compute_modded_nanogpt_val_steps(FINEWEB_VAL_TOKENS, batch_size=64, seq_len=512) == 320
    assert compute_modded_nanogpt_val_steps(FINEWEB_VAL_TOKENS, batch_size=32, seq_len=1024) == 320

    # Verify sequential non-overlapping slicing from pos=0 (never skipping tokens at EOT=50256)
    val_stream = np.arange(1024, dtype=np.uint16)
    val_stream[5] = 50256  # EOT token inside first batch must NOT shift batch start
    batches = list(iter_modded_nanogpt_val_batches(val_stream, batch_size=2, seq_len=16, val_tokens_budget=64))
    assert len(batches) == 2
    b0, x0, y0 = batches[0]
    b1, x1, y1 = batches[1]
    assert b0 == 0 and b1 == 1
    assert np.array_equal(x0.reshape(-1), val_stream[0:32])
    assert np.array_equal(y0.reshape(-1), val_stream[1:33])
    assert np.array_equal(x1.reshape(-1), val_stream[32:64])
    assert np.array_equal(y1.reshape(-1), val_stream[33:65])

    # Verify valid_vocab_size=50257 masks out padding columns (e.g. 64..79 in a toy 80-token padded vocab)
    torch.manual_seed(123)
    v_padded = 80
    v_valid = 64
    n_tok = 32
    d_dim = 16
    targets = torch.randint(0, v_valid, (n_tok,), dtype=torch.long)

    # 1. Classical softcap linear CE parity with valid_vocab_size
    x_hid = torch.randn(n_tok, d_dim, requires_grad=True)
    w_head = torch.randn(v_padded, d_dim, requires_grad=True)
    loss_chunked = direct_chunked_softcap_linear_ce(
        x_hid, w_head, targets, softcap=15.0, chunk_tokens=16, valid_vocab_size=v_valid
    )
    raw_logits = (x_hid.float() @ w_head[:v_valid].float().T)
    capped_logits = 15.0 * torch.tanh(raw_logits / 15.0)
    loss_ref = F.cross_entropy(capped_logits, targets, reduction="mean")
    assert torch.allclose(loss_chunked, loss_ref, atol=1e-5)
    loss_chunked.backward()
    assert w_head.grad is not None
    assert float(w_head.grad[v_valid:].abs().max().item()) == 0.0

    # 2. Quantum POVM CE parity with valid_vocab_size (even when padding POVM rows 64..79 are non-zero unit vectors!)
    phi = F.normalize(torch.randn(n_tok, d_dim), p=2, dim=-1).requires_grad_(True)
    m_seed = F.normalize(torch.randn(v_padded, d_dim), p=2, dim=-1).requires_grad_(True)
    inv_t = torch.tensor(20.0, requires_grad=True)
    log_prior = torch.randn(v_padded, requires_grad=True)

    loss_povm = direct_chunked_povm_ce(
        phi, m_seed, inv_t, targets, log_prior=log_prior, chunk_tokens=16, valid_vocab_size=v_valid
    )
    logits_povm_ref = (phi.float() @ m_seed[:v_valid].float().T) * 20.0 + log_prior[:v_valid].float().unsqueeze(0)
    loss_povm_ref = F.cross_entropy(logits_povm_ref, targets, reduction="mean")
    assert torch.allclose(loss_povm, loss_povm_ref, atol=1e-5)
    loss_povm.backward()
    assert m_seed.grad is not None and log_prior.grad is not None
    assert float(m_seed.grad[v_valid:].abs().max().item()) == 0.0
    assert float(log_prior.grad[v_valid:].abs().max().item()) == 0.0

