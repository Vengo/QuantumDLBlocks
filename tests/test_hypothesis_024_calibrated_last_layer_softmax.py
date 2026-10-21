"""Unit tests for Experiment Plan 024: Calibrated Last-Layer Softmax Readout for 16-Qubit Hierarchical QuantumBornGPT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.model_card import (
    build_model_card_from_manifest_entry,
    parse_cli_args_to_card_sections,
)
from qdlblocks.layers.direct_chunked_ce import (
    direct_chunked_softcap_linear_ce,
    sampled_softcap_linear_ce,
)
from qdlblocks.layers.hierarchical_qubit import (
    initialize_hierarchical_16qubit_with_ppmi_svd,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import (
    StiefelMuon,
    polar_snap_weights,
    split_muon_adamw_params,
)


def test_direct_chunked_softcap_linear_ce_with_bias_forward_backward_parity() -> None:
    torch.manual_seed(2024)
    v_padded = 80
    v_valid = 64
    n_tok = 32
    d_dim = 16
    softcap = 15.0
    targets = torch.randint(0, v_valid, (n_tok,), dtype=torch.long)

    x_chunked = torch.randn(n_tok, d_dim, requires_grad=True)
    w_chunked = torch.randn(v_padded, d_dim, requires_grad=True)
    b_chunked = torch.randn(v_padded, requires_grad=True)

    x_ref = x_chunked.detach().clone().requires_grad_(True)
    w_ref = w_chunked.detach().clone().requires_grad_(True)
    b_ref = b_chunked.detach().clone().requires_grad_(True)

    loss_chunked = direct_chunked_softcap_linear_ce(
        x_chunked,
        w_chunked,
        targets,
        softcap=softcap,
        chunk_tokens=16,
        valid_vocab_size=v_valid,
        bias=b_chunked,
    )

    raw_logits = (x_ref.float() @ w_ref[:v_valid].float().T) + b_ref[:v_valid].float().unsqueeze(0)
    capped_logits = softcap * torch.tanh(raw_logits / softcap)
    loss_ref = F.cross_entropy(capped_logits, targets, reduction="mean")

    assert torch.allclose(loss_chunked, loss_ref, atol=1e-5)

    loss_chunked.backward()
    loss_ref.backward()

    assert x_chunked.grad is not None and x_ref.grad is not None
    assert w_chunked.grad is not None and w_ref.grad is not None
    assert b_chunked.grad is not None and b_ref.grad is not None

    assert torch.allclose(x_chunked.grad, x_ref.grad, atol=1e-4)
    assert torch.allclose(w_chunked.grad[:v_valid], w_ref.grad[:v_valid], atol=1e-4)
    assert torch.allclose(b_chunked.grad[:v_valid], b_ref.grad[:v_valid], atol=1e-4)

    # Padding columns [v_valid:v_padded] must have exact zero gradient
    assert float(w_chunked.grad[v_valid:].abs().max().item()) == 0.0
    assert float(b_chunked.grad[v_valid:].abs().max().item()) == 0.0

    # Also verify sampled_softcap_linear_ce with bias
    loss_sampled = sampled_softcap_linear_ce(
        x_chunked.detach(),
        w_chunked.detach(),
        targets,
        num_candidates=32,
        softcap=softcap,
        bias=b_chunked.detach(),
    )
    assert torch.isfinite(loss_sampled)


def test_calibrated_softmax_readout_even_odd_gradient_flow_and_non_unit_norm() -> None:
    torch.manual_seed(42)
    model = QuantumBornGPT(
        vocab_size=64,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        seed_dim=64,
        block_size=32,
        use_ortho_decouple=True,
        use_povm_log_prior=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=4,
        num_higher_states=16,
        use_last_layer_softmax=True,
        logit_softcap=15.0,
    )
    assert model.hierarchical_readout is not None
    assert model.hierarchical_embed is not None

    # Step-0 parity: hier_even_proj is initialized to 0.0 so even coordinates [:, 0::2] of calibrated weight are 0 at step 0
    w_cal_0 = model.hierarchical_readout.get_calibrated_softmax_weight(model.hierarchical_embed)
    assert float(w_cal_0[:, 0::2].abs().max().item()) == 0.0
    assert float(w_cal_0[:, 1::2].abs().max().item()) > 0.0

    x = torch.randint(0, 64, (4, 16), dtype=torch.long)
    y = (x + 1) % 64

    # Check forward parity between compute_logits and direct_chunked_ce
    logits_full, loss_full = model(x, targets=y, direct_chunked_ce=False)
    _, loss_chunked = model(x, targets=y, direct_chunked_ce=True)
    assert logits_full is not None and loss_full is not None and loss_chunked is not None
    assert torch.allclose(loss_full, loss_chunked, atol=1e-4)

    loss_chunked.backward()

    hr = model.hierarchical_readout
    assert hr.hier_even_proj.grad is not None
    assert float(hr.hier_even_proj.grad.abs().max().item()) > 0.0, "hier_even_proj must receive non-zero gradient!"
    assert hr.povm_odd_qubits.grad is not None
    assert float(hr.povm_odd_qubits.grad.abs().max().item()) > 0.0
    assert hr.vocab_higher_affinity is not None and hr.vocab_higher_affinity.grad is not None
    assert float(hr.vocab_higher_affinity.grad.abs().max().item()) > 0.0
    assert hr.token_radial_scale.grad is not None
    assert float(hr.token_radial_scale.grad.abs().max().item()) > 0.0
    assert hr.readout_rms_gain.grad is not None
    assert float(hr.readout_rms_gain.grad.abs().max().item()) > 0.0
    assert hr.qubit_readout_gain.grad is not None
    assert float(hr.qubit_readout_gain.grad.abs().max().item()) > 0.0
    assert model.povm_log_prior is not None and model.povm_log_prior.grad is not None
    assert float(model.povm_log_prior.grad.abs().max().item()) > 0.0

    # Take an AdamW step and run polar_snap_weights; verify povm_odd_qubits is NOT snapped back to 0.25
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    opt.step()
    polar_snap_weights(model, polar_method="polar_express")

    odd_norms = torch.norm(hr.povm_odd_qubits.detach(), p=2, dim=-1)
    assert not torch.allclose(odd_norms, torch.full_like(odd_norms, 0.25), atol=1e-4), (
        "When use_last_layer_softmax=True, polar_snap_weights must NOT snap povm_odd_qubits back to 0.25 norm!"
    )
    # Meanwhile unigram_qubits and higher_states ARE still snapped to 0.25 norm
    uni_norms = torch.norm(model.hierarchical_embed.unigram_qubits.detach(), p=2, dim=-1)
    assert torch.allclose(uni_norms, torch.full_like(uni_norms, 0.25), atol=1e-4)

    # After 1 step, hier_even_proj is non-zero so even coordinates [:, 0::2] of calibrated weight become non-zero
    w_cal_1 = hr.get_calibrated_softmax_weight(model.hierarchical_embed)
    assert float(w_cal_1[:, 0::2].abs().max().item()) > 0.0


def test_split_muon_adamw_params_excludes_hier_even_proj() -> None:
    model = QuantumBornGPT(
        vocab_size=64,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        seed_dim=64,
        block_size=32,
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=4,
        num_higher_states=16,
        use_last_layer_softmax=True,
    )
    assert model.hierarchical_readout is not None
    assert model.hierarchical_embed is not None

    muon_params, adamw_params = split_muon_adamw_params(model)
    muon_ids = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}

    hr = model.hierarchical_readout
    he = model.hierarchical_embed

    # Only higher_interaction and higher_sub_unitary are 3D tensors in StiefelMuon (hier_even_proj is excluded)
    muon_3d = [p for p in muon_params if p.ndim == 3]
    assert len(muon_3d) == 2
    assert id(he.higher_interaction) in muon_ids
    assert id(he.higher_sub_unitary) in muon_ids
    assert id(hr.hier_even_proj) not in muon_ids

    # hier_even_proj, token_radial_scale, readout_rms_gain, qubit_readout_gain must be in AdamW
    for p_adam in (
        hr.hier_even_proj,
        hr.token_radial_scale,
        hr.readout_rms_gain,
        hr.qubit_readout_gain,
        hr.povm_odd_qubits,
        hr.vocab_higher_affinity,
    ):
        assert p_adam is not None
        assert id(p_adam) not in muon_ids
        assert id(p_adam) in adamw_ids


def test_corrected_ppmi_svd_init_preserves_distinct_tail_tokens_and_ve() -> None:
    # Create a synthetic token stream where only tokens 0..23 appear in the stream and tokens 24..63 are rare/unseen
    rng = np.random.default_rng(77)
    stream = rng.integers(0, 24, size=16384, dtype=np.uint16)

    model = QuantumBornGPT(
        vocab_size=64,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        seed_dim=64,
        block_size=32,
        use_ortho_decouple=True,
        use_povm_log_prior=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=4,
        num_higher_states=16,
        use_last_layer_softmax=True,
        num_value_embeddings=3,
    )

    stats = initialize_hierarchical_16qubit_with_ppmi_svd(
        model,
        token_shards=[stream],
        alpha=0.75,
        kmeans_iters=8,
        device=torch.device("cpu"),
    )

    assert stats["max_self_overlap"] < 1e-6
    assert stats["num_value_embeds_initialized"] == 3

    # Verify all 3 value_embeds were initialized with non-zero weights
    for ve in model.value_embeds:
        assert float(torch.norm(ve.weight.detach()).item()) > 0.0

    # Verify tail tokens 24..63 have distinct unit embeddings (not identical rank-1 clones!)
    assert model.hierarchical_embed is not None
    tail_flat = model.hierarchical_embed.unigram_qubits.detach()[24:].reshape(40, 64)
    tail_norms = torch.norm(tail_flat, p=2, dim=-1)
    assert torch.allclose(tail_norms, torch.ones_like(tail_norms), atol=1e-4)

    tail_cos = tail_flat @ tail_flat.T
    off_diag_mask = ~torch.eye(40, dtype=torch.bool)
    max_tail_off_diag_cos = float(tail_cos[off_diag_mask].abs().max().item())
    assert max_tail_off_diag_cos < 0.65, (
        f"Expected distinct isotropic tail token embeddings (< 0.65 cosine), got {max_tail_off_diag_cos:.4f}"
    )


def test_calibrated_softmax_vs_uncalibrated_povm_micro_convergence() -> None:
    # Build a skewed Zipfian + Bigram synthetic token dataset on V=128
    torch.manual_seed(1024)
    vocab_size = 128
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
    unigram_probs = 1.0 / (ranks**1.15)
    unigram_probs = unigram_probs / unigram_probs.sum()

    # Deterministic bigram transition boost + Zipfian base
    seq_len = 32
    batch_size = 16
    x_batches = []
    y_batches = []
    for _ in range(12):
        x = torch.multinomial(unigram_probs, batch_size * seq_len, replacement=True).view(batch_size, seq_len)
        # Next token is 60% deterministic bigram shift ((x * 3 + 1) % 32) and 40% Zipfian sample
        z_sample = torch.multinomial(unigram_probs, batch_size * seq_len, replacement=True).view(batch_size, seq_len)
        mask = torch.rand(batch_size, seq_len) < 0.60
        y = torch.where(mask, (x * 3 + 1) % 32, z_sample)
        x_batches.append(x)
        y_batches.append(y)

    def _run_arm(use_softmax: bool) -> list[float]:
        torch.manual_seed(2048)
        m = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            seed_dim=64,
            block_size=seq_len,
            use_ortho_decouple=True,
            use_povm_log_prior=True,
            use_hierarchical_16qubit=True,
            qubit_registers=16,
            qubit_amplitudes=4,
            num_higher_states=16,
            use_last_layer_softmax=use_softmax,
            logit_softcap=15.0,
        )
        muon_p, adam_p = split_muon_adamw_params(m)
        opt_m = StiefelMuon(muon_p, lr=0.025, polar_method="polar_express")
        opt_a = torch.optim.AdamW(adam_p, lr=0.008 if use_softmax else 0.004, weight_decay=0.0)
        losses: list[float] = []
        for step in range(10):
            opt_m.zero_grad(set_to_none=True)
            opt_a.zero_grad(set_to_none=True)
            _, loss = m(x_batches[step], targets=y_batches[step], direct_chunked_ce=True)
            assert loss is not None and torch.isfinite(loss)
            loss.backward()
            opt_m.step()
            opt_a.step()
            polar_snap_weights(m, polar_method="polar_express")
            losses.append(float(loss.item()))
        return losses

    losses_povm = _run_arm(use_softmax=False)
    losses_softmax = _run_arm(use_softmax=True)
    assert losses_softmax[-1] < losses_povm[-1], (
        f"Expected Calibrated Last-Layer Softmax final loss ({losses_softmax[-1]:.4f}) < "
        f"Uncalibrated POVM final loss ({losses_povm[-1]:.4f})"
    )


def test_wave15_hotfix_manifest_excludes_a100_w6_1() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hotfix_path = repo_root / "configs" / "wave15_hotfix_softmax_fleet.json"
    fleet_path = repo_root / "configs" / "wave15_fleet.json"

    assert hotfix_path.exists()
    assert fleet_path.exists()

    hotfix_data = json.loads(hotfix_path.read_text(encoding="utf-8"))
    hotfix_runs = hotfix_data["runs"]
    hotfix_sessions = [r["session"] for r in hotfix_runs]

    # Critical safety requirement: gpu_a100_w6_1 must NOT be in wave15_hotfix_softmax_fleet.json
    assert "gpu_a100_w6_1" not in hotfix_sessions
    assert set(hotfix_sessions) == {"gpu_g4_w6_1", "gpu_g4_w6_2", "gpu_g4_w6_3", "gpu_a100_w6_2"}

    for r in hotfix_runs:
        assert "--last-layer-softmax" in r["cli_args"]
        assert "--logit-softcap 15.0" in r["cli_args"]
        arch, opt, _ = parse_cli_args_to_card_sections(r["cli_args"])
        assert arch["use_last_layer_softmax"] is True
        assert arch["logit_softcap"] == 15.0
        assert arch["readout_mode"] == "calibrated_hierarchical_softmax"
        assert opt["adamw_lr_head"] == 0.008
        card = build_model_card_from_manifest_entry(r, wave_id="W15")
        assert card["run_tier"] == "main_benchmark"

    # Full wave15_fleet.json must retain gpu_a100_w6_1 (without --last-layer-softmax) and mark the 4 stopped runs obsolete
    fleet_data = json.loads(fleet_path.read_text(encoding="utf-8"))
    fleet_runs_by_key = {r["key"]: r for r in fleet_data["runs"]}
    assert "w15_900m_qborn_16x32_qubit_256hier_ngram_L8" in fleet_runs_by_key
    a100_1_run = fleet_runs_by_key["w15_900m_qborn_16x32_qubit_256hier_ngram_L8"]
    assert a100_1_run["session"] == "gpu_a100_w6_1"
    assert "--last-layer-softmax" not in a100_1_run["cli_args"]
    assert not bool(a100_1_run.get("obsolete", False))

    for obs_key in (
        "w15_900m_qborn_16x32_qubit_256hier_flagship_L8",
        "w15_900m_qborn_16x32_qubit_256hier_deep_L12_H8_d128",
        "w15_900m_qborn_16x32_qubit_512hier_L8",
        "w15_900m_qborn_16x32_qubit_256hier_rank4_density_L8",
    ):
        assert obs_key in fleet_runs_by_key
        assert fleet_runs_by_key[obs_key]["obsolete"] is True
