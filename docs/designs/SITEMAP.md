# QuantumDLBlocks Design Documents & EGM Track Sitemap (`docs/designs/SITEMAP.md`)

This sitemap indexes all 7 certified Engineering Goldfish Mode (EGM) design documents in `docs/designs/egm_tracks/`, the `Wave 44` master plan ([`EXPERIMENT_PLAN_059.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_059.md)), core engineering standards, and the active diagnostic CLI scripts in `scripts/`.

---

## 1. Certified Wave 44 EGM Track Design Documents (`docs/designs/egm_tracks/`)

| Track | Design Document | Core Problem Addressed | Certified Technical Solution & Key Symbols |
| :---: | :--- | :--- | :--- |
| **Track 1** | [`docs/designs/egm_tracks/track1_joint_complex_stiefel_muon.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/designs/egm_tracks/track1_joint_complex_stiefel_muon.md) | `StiefelMuon` previously orthogonalized `weight_re` and `weight_im` as two independent real matrices, yielding $(U_{\text{re}} + i U_{\text{im}})(U_{\text{re}} + i U_{\text{im}})^\dagger = 2I + i(\dots) \ne I$ and breaking $U(1)$ phase equivariance. | Pairs `(param_re, param_im)` in `StiefelMuon` (`is_complex_pair_group=True`) and runs 5-step quintic Polar Express on the Hermitian Gram matrix $A = X X^\dagger \in \mathbb{C}^{m \times m}$ via `zeropower_via_polarexpress5_complex_2d`, `zeropower_via_polarexpress5_complex_batched`, and `normuon_equalize_complex` in [`src/qdlblocks/optim/muon.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/muon.py). |
| **Track 2** | [`docs/designs/egm_tracks/track2_head_dim_and_universal_muon.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/designs/egm_tracks/track2_head_dim_and_universal_muon.md) | `DEFAULT_COMPLEX_REGISTER_DIM = 32` concatenated two distinct $\mathbb{C}^{16}$ Kronecker factors into each attention head ($H=4 \times \mathbb{C}^{32}$), while `MIN_COMPLEX_MUON_DIM = 16` excluded small unitary mixers (`factor_mix`, `meaning_mix`) from `StiefelMuon`. | Aligns `register_dim = DEFAULT_FACTOR_DIM = 16` ($H = 4R \times \mathbb{C}^{16}$) and introduces `KroneckerParameterPartition` + `classify_kronecker_model_parameters` in [`src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py) (`MIN_KRONECKER_MUON_OPERATOR_DIM = 2`). |
| **Track 3** | [`docs/designs/egm_tracks/track3_self_echo_and_repetition_ablation.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/designs/egm_tracks/track3_self_echo_and_repetition_ablation.md) | Legacy self-echo subtraction `op - ov_re * p0` dropped the imaginary inner product `ov_im` (breaking $U(1)$ phase invariance) and subtracted along ambient `p0`, perturbing the calibrated unigram carrier $\lvert c_{\text{ref}}\rangle$. | Projects `|p_0>` onto the 15D carrier-orthogonal subspace $\mathcal{H}_{\perp}$ first and removes the full complex inner product $(\text{ov}_{\text{re}} + i\,\text{ov}_{\text{im}})\lvert\hat{p}_{0,\perp}\rangle$ (`apply_carrier_orthogonal_complex_self_echo`), plus adds repetition telemetry (`rep_val_ce`, `non_rep_val_ce`, `mean_self_echo_prob`, `rep_top1_acc`). |
| **Track 4** | [`docs/designs/egm_tracks/track4_polysemy_branch_init_and_routing.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/designs/egm_tracks/track4_polysemy_branch_init_and_routing.md) | `DEFAULT_NGRAM_SECONDARY_BRANCH_SCALE = 0.22` left `98.1%` random Haar noise in branches $r \ge 1$, while `1/\sqrt{4R}` pre-readout scaling shrank per-factor overlaps at $R=4$ and norm-only routing ignored phase alignment. | Initializes all $R$ meaning branches at full spectral strength ($\alpha_{\text{spec}} = 0.88$, `DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE = 1.0`) via distinct Helmert rotations $Q_{r,m} \in \mathrm{O}(30)$, locks factor norm to `REFERENCE_RAW_PHI_FACTOR_SCALE = 1/\sqrt{8}`, and adds a phase-sensitive complex inner-product router $\beta_{t,r}$. |
| **Track 5** | [`docs/designs/egm_tracks/track5_backward_dedup_and_autograd_op_audit.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/designs/egm_tracks/track5_backward_dedup_and_autograd_op_audit.md) | `_calibrated_kronecker_chunk_bwd_step` called `_kronecker_chunk_povm_grad_step` after `_kronecker_chunk_bwd_step`, recomputing `_compute_kronecker_chunk_intensity` and `torch.pow(i_tv, boson_p)` a second time on every chunk; validation recomputed `get_tied_vocab_factors()` every batch. | Deduplicates backward into a single pass via `_kronecker_chunk_bwd_step_with_prior_grad`, hoists vocabulary tensors into `_EvalChunkedVocabContext` during validation, and introduces the two-layer Custom Autograd Efficiency Auditor (`src/qdlblocks/benchmarks/autograd_efficiency_auditor.py`). |
| **Track 6** | [`docs/designs/egm_tracks/track6_multihead_superposition_and_mixed_states.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/designs/egm_tracks/track6_multihead_superposition_and_mixed_states.md) | `factored_complex_krylov_step` executed `11` $(T \times T)$ matrix multiplications per layer, creating a quadratic latency bottleneck at $T \in \{512, 1024\}$. | Replaces intra-head time-domain Krylov GEMMs with single-pass causal multi-head Born attention (`2` $(T \times T)$ GEMMs/layer) + $O(H_g^2 d_f)$ Cross-Head Hermitian Gram matrix purification (`MultiHeadStinespringSuperpositionAttention`) and incoherent mixed-state readout (`MultiHeadMixedStateKroneckerGPT`). |
| **Track 7** | [`docs/designs/egm_tracks/track7_tooling_backlog_and_subagent_surfacing.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/designs/egm_tracks/track7_tooling_backlog_and_subagent_surfacing.md) | `scripts/TOOLING_BACKLOG.md` mixed implemented, active candidate, and superseded Wave 6–38 scripts, and sub-agents lacked a single skill indexing all CLI probes and Wave 44 invariants. | Implements [`src/qdlblocks/benchmarks/kronecker_diagnostic_utils.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/kronecker_diagnostic_utils.py), all active candidate CLI scripts in `scripts/`, [`.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md`](file:///home/festivity/agy/QuantumDLBlocks/.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md), and reconciles [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md). |

---

## 2. Core Repository Guides, Standards, & Skills

- **Agent Guidelines**: [`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md)
- **Reusable Building-Blocks Catalog**: [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md)
- **Script & Module Best Practices (Rules 1–9)**: [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md)
- **Tooling & Wave 44 Invariants Skill**: [`.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md`](file:///home/festivity/agy/QuantumDLBlocks/.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md)
- **Colab Experiment Monitor Skill**: [`.agents/skills/colab-experiment-monitor/SKILL.md`](file:///home/festivity/agy/QuantumDLBlocks/.agents/skills/colab-experiment-monitor/SKILL.md)
- **Operational Scripts & Tooling Backlog**: [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md)
- **Wave 44 Experiment Plan**: [`EXPERIMENT_PLAN_059.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_059.md)

---

## 3. Wave 44 Diagnostic & Operational CLI Scripts (`scripts/`)

### 3.1 Track 7 Backlog Candidate CLI Scripts (`scripts/TOOLING_BACKLOG.md`)
1. [`scripts/multihead_von_neumann_entropy_and_coherence_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multihead_von_neumann_entropy_and_coherence_inspector.py)
2. [`scripts/polysemy_router_token_sense_clustering_cli.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/polysemy_router_token_sense_clustering_cli.py)
3. [`scripts/polysemy_branch_sense_disambiguation_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/polysemy_branch_sense_disambiguation_inspector.py)
4. [`scripts/kronecker_factor_entanglement_and_mutual_info_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kronecker_factor_entanglement_and_mutual_info_probe.py)
5. [`scripts/povm_scale_drift_and_rarity_decile_analyzer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/povm_scale_drift_and_rarity_decile_analyzer.py)
6. [`scripts/rare_token_gradient_starvation_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/rare_token_gradient_starvation_tracker.py)
7. [`scripts/ngram_quantum_operator_vs_attention_gate_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ngram_quantum_operator_vs_attention_gate_tracker.py)
8. [`scripts/modded_nanogpt_10m_per_decile_ce_breakdown.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/modded_nanogpt_10m_per_decile_ce_breakdown.py)
9. [`scripts/born_povm_boson_power_and_floor_calibrator.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/born_povm_boson_power_and_floor_calibrator.py)
10. [`scripts/random_init_attention_carrier_spectrum_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/random_init_attention_carrier_spectrum_probe.py)
11. [`scripts/grad_accum_effective_batch_scaling_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/grad_accum_effective_batch_scaling_profiler.py)
12. [`scripts/wall_clock_chinchilla_pareto_predictor.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wall_clock_chinchilla_pareto_predictor.py)
13. [`scripts/verify_plan.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_plan.py)
14. [`scripts/wave_report.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wave_report.py)

### 3.2 Wave 44 Mechanism & Track Verification Probes (`scripts/`)
1. [`scripts/ngram_helmert_carrier_calibration_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ngram_helmert_carrier_calibration_inspector.py)
2. [`scripts/trigram_context_smearing_ablation_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/trigram_context_smearing_ablation_probe.py)
3. [`scripts/multibranch_polysemy_entropy_and_subspace_angle_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multibranch_polysemy_entropy_and_subspace_angle_probe.py)
4. [`scripts/kronecker_coincidence_gradient_snr_and_rank_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kronecker_coincidence_gradient_snr_and_rank_probe.py)
5. [`scripts/calibrated_kronecker_vocab_spectrum_and_povm_drift_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/calibrated_kronecker_vocab_spectrum_and_povm_drift_probe.py)
6. [`scripts/context_adaptive_branch_beta_and_self_echo_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/context_adaptive_branch_beta_and_self_echo_probe.py)
7. [`scripts/born_exponent_and_holographic_coincidence_sweep.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/born_exponent_and_holographic_coincidence_sweep.py)
8. [`scripts/multilag_window_and_coincidence_schedule_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multilag_window_and_coincidence_schedule_probe.py)
9. [`scripts/multihead_gram_krylov_vs_intrahead_ablation_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multihead_gram_krylov_vs_intrahead_ablation_probe.py)
10. [`scripts/quantum_superposition_purity_and_gram_spectrum_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/quantum_superposition_purity_and_gram_spectrum_probe.py)
11. [`scripts/complex_stiefel_orthogonality_and_large_batch_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complex_stiefel_orthogonality_and_large_batch_probe.py)
12. [`scripts/self_echo_repetition_penalty_and_nll_breakdown_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/self_echo_repetition_penalty_and_nll_breakdown_probe.py)
13. [`scripts/wave_preflight_hardware_and_throughput_gate.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wave_preflight_hardware_and_throughput_gate.py)
14. [`scripts/colab_fleet_watchdog_and_auto_requeue.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet_watchdog_and_auto_requeue.py)
