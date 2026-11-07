r"""Verification Script for `EXPERIMENT_PLAN_085.md` (`Wave 65`): Fixed-Once Subsystem Quantum Vocabulary Architectures (`256 ⊕ 197` Lüders Instrument, `16^4` Right-Canonical MPS, & Shared `St(512, 128)` Frame)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from qdlblocks.benchmarks.quantum_subsystem_vocab_lab import (
    LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES, LUDERS_FULL_ROW_CUTOFF, MPS_SITE3_FULL_PREFIX_CUTOFF,
    MPS_SITE3_MAX_OUTCOMES, QuantumSubsystemVocabGPT, QuantumSubsystemVocabSpec, VocabSubsystemMode,
    build_luders_hierarchical_zipf_priors, build_mps_4qudit_hierarchical_zipf_priors,
    build_subsystem_vocab_optimizers, step_subsystem_vocab_optimizers,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import RAW_GPT2_VOCAB_SIZE
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import build_zero_data_bpe_rank_zipf_prior


def main() -> None:
    device = torch.device("cpu")
    torch.manual_seed(1337)

    # 1. Verify exact hierarchical Zipf prior factorization for both Lüders (256x197) and MPS (16x16x16x13)
    log_p_flat, _, _ = build_zero_data_bpe_rank_zipf_prior(50304, device=device)
    log_p_valid = log_p_flat[:RAW_GPT2_VOCAB_SIZE] - torch.logsumexp(log_p_flat[:RAW_GPT2_VOCAB_SIZE], dim=0)

    b_coarse, b_fine_cond, valid_2d = build_luders_hierarchical_zipf_priors(RAW_GPT2_VOCAB_SIZE, device=device)
    assert int(valid_2d.sum().item()) == RAW_GPT2_VOCAB_SIZE
    assert int((LUDERS_FULL_ROW_CUTOFF * LUDERS_FINE_MAX_OUTCOMES) + ((LUDERS_COARSE_OUTCOMES - LUDERS_FULL_ROW_CUTOFF) * (LUDERS_FINE_MAX_OUTCOMES - 1))) == RAW_GPT2_VOCAB_SIZE
    v_ids = torch.arange(RAW_GPT2_VOCAB_SIZE, device=device, dtype=torch.long)
    recon_luders = b_coarse[v_ids & 0xFF] + b_fine_cond[v_ids & 0xFF, v_ids >> 8]
    luders_prior_err = float((recon_luders - log_p_valid).abs().max().item())
    assert luders_prior_err < 1e-5, f"Lüders hierarchical Zipf prior error too high: {luders_prior_err}"

    b0, b1, b2, b3, valid_s3 = build_mps_4qudit_hierarchical_zipf_priors(RAW_GPT2_VOCAB_SIZE, device=device)
    assert int(valid_s3.sum().item()) == RAW_GPT2_VOCAB_SIZE
    assert int((MPS_SITE3_FULL_PREFIX_CUTOFF * MPS_SITE3_MAX_OUTCOMES) + ((4096 - MPS_SITE3_FULL_PREFIX_CUTOFF) * (MPS_SITE3_MAX_OUTCOMES - 1))) == RAW_GPT2_VOCAB_SIZE
    s0, s1, s2, s3 = v_ids & 0xF, (v_ids >> 4) & 0xF, (v_ids >> 8) & 0xF, v_ids >> 12
    p01, p012 = v_ids & 0xFF, v_ids & 0xFFF
    recon_mps = b0[s0] + b1[s0, s1] + b2[p01, s2] + b3[p012, s3]
    mps_prior_err = float((recon_mps - log_p_valid).abs().max().item())
    assert mps_prior_err < 1e-5, f"MPS 4-qudit hierarchical Zipf prior error too high: {mps_prior_err}"

    # 2. Verify all three Fixed-Once Subsystem Vocabulary Modes (Partition Function == 1.0, Forward/Backward, Parameter Compression)
    idx_in = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 16), device=device, dtype=torch.long)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 16), device=device, dtype=torch.long)

    mode_reports: dict[str, dict] = {}
    for mode in (
        VocabSubsystemMode.LUDERS_256X256_INSTRUMENT,
        VocabSubsystemMode.MPS_16POW4_BORN_CHAIN,
        VocabSubsystemMode.SHARED_STIEFEL_128D_FRAME,
    ):
        spec = QuantumSubsystemVocabSpec(vocab_mode=mode, num_layers=2, kraus_rank=2, seed=1337)
        model = QuantumSubsystemVocabGPT(spec).to(device=device)
        init_ms = model.initialize_model_()
        budget = model.parameter_budget_summary()
        part_check = model.verify_exact_50257_partition_function(idx_in[:, :4])
        assert part_check["max_partition_err"] < 1e-5, f"{mode.value} partition error: {part_check['max_partition_err']}"

        # Verify forward_loss_and_metrics matches gathering from full_50257_log_probs
        with torch.no_grad():
            loss_fast, _ = model.forward_loss_and_metrics(idx_in[:, :4], targets[:, :4])
            u_voc, u_vac, rho_re, rho_im = model.encode_trunk(idx_in[:, :4])
            full_lp = model.boundary.full_50257_log_probs(
                u_voc, u_vac, model.vac_rest_mass, rho_re, rho_im, model.current_analytical_beta()
            )
            loss_exact = -full_lp.gather(-1, targets[:, :4].reshape(-1, 1)).mean()
            nll_consistency_err = float((loss_fast - loss_exact).abs().item())
            assert nll_consistency_err < 1e-5, f"{mode.value} NLL consistency err: {nll_consistency_err}"

        # Run 2 optimization steps and verify manifold preservation
        opts = build_subsystem_vocab_optimizers(model)
        losses = []
        for step in range(1, 3):
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            loss_t, m_dict = model.forward_loss_and_metrics(idx_in, targets)
            losses.append(round(float(loss_t.item()), 4))
            loss_t.backward()
            step_subsystem_vocab_optimizers(model, opts, tokens_in_step=32, lr_mult=1.0, mom_beta=0.50)

        telem = model.get_manifold_telemetry()
        assert telem["kraus_stiefel_err"] < 1e-4
        assert telem["boundary_stiefel_err"] < 1e-4

        mode_reports[mode.value] = {
            "init_ms": round(init_ms, 2),
            "parameter_budget": budget,
            "partition_function_check": part_check,
            "nll_consistency_err": float(f"{nll_consistency_err:.3e}"),
            "step_losses": losses,
            "post_step_telemetry": telem,
        }

    summary = {
        "plan": "EXPERIMENT_PLAN_085",
        "wave": "W65",
        "status": "PASS",
        "luders_hierarchical_zipf_max_err": float(f"{luders_prior_err:.3e}"),
        "mps_4qudit_hierarchical_zipf_max_err": float(f"{mps_prior_err:.3e}"),
        "modes": mode_reports,
    }
    out_path = Path(".cache/verify_plan085_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
