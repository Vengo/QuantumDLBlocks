# Experiment Plan 063 (Wave 48): Standalone Phase-Locked Krylov & Impure Density Quantum Born GPT

- **Wave**: 48 (`EXPERIMENT_PLAN_063.md`)
- **Design Specification**: [`docs/designs/wave48_standalone_quantum_born_gpt.md`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md)
- **Date**: 2026-10-30
- **Hardware Target**: `2x NVIDIA GeForce RTX 3090 (24 GB VRAM each)`
- **Validation Benchmark**: Official `Modded-NanoGPT` `FineWeb` Shard 0 (`10,485,760` validation tokens in [`fineweb_val_000000.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin))

---

## 1. Executive Summary & Architectural Motivation

Wave 47 ([`EXPERIMENT_PLAN_062.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_062.md)) demonstrated that a 12-layer `RiemannianImpureDensityGPT` could update its internal QKV and unitary weights via `RiemannianStiefelMuon`, but relied on a pre-trained `400M`-token classical teacher checkpoint for its embeddings and periodically overwrote its 3-factor SVD readout matrices (`w_obs_u`, `w_obs_vt`, `w_chan_u`, `w_chan_vt`) via Golub-Pereyra ridge regression against cached teacher activations. Furthermore, passing only `256`-dimensional quadratic density matrices $\rho = |v\rangle\langle v|$ through causal attention erased all first-moment linear sign and complex $U(1)$ phase information ($|v\rangle \mapsto e^{i\theta}|v\rangle$).

Wave 48 (`EXPERIMENT_PLAN_063.md`) eliminates all teacher checkpoints and ridge-regression refits to establish **100% standalone, Step-0 end-to-end training on FineWeb**:
1. **Fused `288`-Width Isometric Hermitian + First-Moment Ket Attention & Option B (`320`-Dim/Head) Uhlmann Phase-Locked Rank-2 Krylov Readout**: Packs the $\sqrt{2}$-isometric Rank-4 density matrix (`256` dims) and the unit-norm first-moment coherent value ket (`32` dims) into a single `288`-wide Tensor-Core causal $p$-boson Born attention GEMM per head, applies 1-step Bures-Wasserstein McWeeny purification and $U(16)$ unitary evolution inside a `@torch.compile` helper (`bfloat16` matmuls, `float32` trace/norm reductions), extracts the Uhlmann phase-locked Rank-2 Lanczos-Krylov eigenpair $((\lambda_1, |\phi_1\rangle), (\lambda_2, |\phi_2\rangle))$ seeded by the contextual first-moment ket $|\bar{v}\rangle$, and feeds both eigenmodes alongside the unit-scale purified density (`256 + 64 = 320` dims/head, `1,920` total) into a single direct Stiefel readout $W_{\text{obs}} \in \text{St}(1920, 768)$.
2. **Tikhonov-Damped `RiemannianStiefelMuon` Across 100% of Backbone Operators**: Replaces the unstable 3-factor SVD readout with a single direct semi-orthogonal Stiefel matrix + learnable residual vector gain updated at every step by tangent-projected Newton-Schulz with Tikhonov Gram damping ($\lambda_{\text{tik}} = 10^{-3}$) and $1/\sqrt{1+\eta^2}$ geodesic retraction.
3. **Active-Token Vocabulary Manifold Optimizers & Rule-9 Single-Pass Chunked Readouts**: Implements `BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer` (updating untied `wte.weight` strictly on $\mathcal{I}_{\text{in}} = \operatorname{unique}(\text{idx})$ and `lm_head.weight` on $\mathcal{I}_{\text{act}} = \operatorname{unique}(\text{idx} \cup \text{targets})$) and benchmarks them against `AdamW` across both `VocabularyReadoutMode.RANK2_KRYLOV_BORN` (`Rank2KrylovBornChunkedCEFn`) and `VocabularyReadoutMode.SOFTCAP_LINEAR_CE` (`_CompiledChunkedSoftcapLinearCEFn`).

---

## 2. Quantitative Hypotheses (`H1`, `H2`, `H3`, `H4`)

- **Hypothesis H1 (Uhlmann Phase-Locked Rank-2 Krylov `320`-Dim/Head Readout vs. Phase-Erased `256`-Width Density Attention)**:
  - Fused `288`-width packing + Uhlmann phase-locked Rank-2 Krylov extraction (`QuantumAttentionMode.UHLMANN_KRYLOV_320`) achieves exact single-GEMM equivalence (`max_abs_diff < 1e-6`), preserves non-zero linear phase covariance (`phase_covariance_cos > 0.999` under $|v\rangle \mapsto e^{i\theta}|v\rangle$, vs. `0.0` for quadratic density-only packing), and achieves strictly lower FineWeb validation Cross-Entropy (`val_ce_uhlmann_320 < val_ce_phase_erased_256`) than `QuantumAttentionMode.PHASE_ERASED_DENSITY_256` on an equal-token standalone FineWeb run.
- **Hypothesis H2 (Teacher-Free Direct Stiefel Readout + Tikhonov-Damped `RiemannianStiefelMuon` Stability)**:
  - Direct semi-orthogonal Stiefel readouts ($W_{\text{obs}} \in \text{St}(1920, 768)$, $W_{\text{chan}} \in \text{St}(1536, 768)$) and complex head unitaries $U_h \in U(16)$ updated 100% end-to-end by Tikhonov-damped `RiemannianStiefelMuon` ($\lambda_{\text{tik}} = 10^{-3}$, zero teacher checkpoints, zero ridge refits) maintain `mean_stiefel_err < 1e-5` and `mean_unitary_err < 1e-5` and achieve lower validation Cross-Entropy (`val_ce_tikhonov_stiefel < val_ce_undamped_euclidean`) than undamped Euclidean updates.
- **Hypothesis H3 (Active-Token Vocabulary Manifold Optimizers & Readout Modes on FineWeb)**:
  - Updating untied vocabulary embeddings strictly on active batch tokens via `BatchSubspaceTikhonovGramMuon` (`VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON`) and `RiemannianSphereTangentSNROptimizer` (`VocabOptimizerMode.SPHERE_TANGENT_SNR_MUON`) maintains exact zero inactive-token drift (`inactive_row_drift == 0.0`) and unit-RMS sphere preservation (`mean_sphere_rms_err < 1e-5`), while converging monotonically from Step 0 on FineWeb across both `VocabularyReadoutMode.RANK2_KRYLOV_BORN` and `VocabularyReadoutMode.SOFTCAP_LINEAR_CE` (`final_val_ce < 6.0` in fast screening and competitive with `AdamW`).
- **Hypothesis H4 (Standalone Multi-GPU FineWeb Shard 0 Benchmark Trajectory)**:
  - Training `StandaloneQuantumBornGPT` ($L=12, d=768, H=6, d_h=16, R=4$) standalone from scratch on FineWeb training shards across `2x NVIDIA GeForce RTX 3090` GPUs with zero teacher dependency achieves monotonic validation Cross-Entropy improvement evaluated on the official `10,485,760`-token `fineweb_val_000000.bin` Shard 0 validation set (`val_ce < 4.20` on intermediate multi-million-token horizon and progressing toward `< 3.28` on the `<= 330M`-token horizon).

---

## 3. Top-of-File Named Constants Ledger (Rule 7)

| Constant Name | Value | Module Location | Rationale |
| :--- | :---: | :--- | :--- |
| `DEFAULT_FUSED_HEAD_PACK_DIM` | `288` | `impure_density_mirror_block.py` | Fused per-head attention GEMM width $d_h^2 + 2 d_h = 256 + 32 = 288$ ($9 \times 32$) |
| `DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM` | `320` | `impure_density_mirror_block.py` | Option B per-head observable readout width $256 + 64 = 320$ ($6 \times 320 = 1,920$ total) |
| `DEFAULT_TIKHONOV_GRAM_LAMBDA` | `1e-3` | `impure_density_mirror_block.py` | Relative Tikhonov Gram damping $\lambda_{\text{tik}}$ in Newton-Schulz & active-token Gram Muon |
| `DEFAULT_UHLMANN_PHASE_EPS` | `1e-7` | `impure_density_mirror_block.py` | Numerical floor for Uhlmann parallel-transport phase lock $\langle \tilde{\phi}_1 \mid \bar{v}\rangle / |\langle \tilde{\phi}_1 \mid \bar{v}\rangle|$ |
| `DEFAULT_SPHERE_SNR_TRUST_TAU` | `0.50` | `vocab_manifold_muon.py` | SNR trust-region half-saturation constant $\tau_{\text{snr}}$ on $\sqrt{d}\,\mathbb{S}^{d-1}$ |
| `DEFAULT_SPHERE_GEODESIC_LR` | `0.08` | `vocab_manifold_muon.py` | Base angular geodesic step size $\eta_{\text{sphere}}$ (radians) for active vocabulary rows |
| `DEFAULT_BATCH_GRAM_MUON_LR` | `0.06` | `vocab_manifold_muon.py` | Base step size $\eta_{\text{gram}}$ for active-token `BatchSubspaceTikhonovGramMuon` |
| `DEFAULT_GRAM_WHITENING_MIX` | `0.25` | `vocab_manifold_muon.py` | Trace-balanced Gram whitening mix $\alpha_{\text{gram}}$ preventing rank-1 unigram carrier suppression |
| `DEFAULT_DIRECT_READOUT_INIT_SIGMA` | `0.06` | `standalone_quantum_born_gpt.py` | Step-0 learnable residual scale initialization for direct Stiefel readouts `w_obs`, `w_chan` |
| `DEFAULT_KRYLOV_BORN_LOG_TEMP` | `6.0` | `standalone_quantum_born_gpt.py` | Log-Born temperature $\tau_{\text{born}}$ inside $15 \tanh(\tau_{\text{born}} \log(I_{t,v}) / 15)$ |
| `DEFAULT_BORN_LOG_INTENSITY_EPS` | `0.01` | `standalone_quantum_born_gpt.py` | Bosonic vacuum intensity floor $\varepsilon_{\text{born}}$ in coherent homodyne Born readout |

---

## 4. Planned Changes

1. **[`src/qdlblocks/layers/impure_density_mirror_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py) & [`src/qdlblocks/layers/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/__init__.py)**:
   - Add `fast_rank_r_hermitian_and_ket_pack`, `extract_rank2_krylov_eigenpair_uhlmann`, Tikhonov Gram damping in `_newton_schulz5_tall_real` and `riemannian_stiefel_muon_step`, and promoted `_complex_newton_schulz5` + batched `riemannian_complex_unitary_muon_step` (`<= 8` parameters per function).
2. **[`src/qdlblocks/optim/vocab_manifold_muon.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py) & [`src/qdlblocks/optim/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/__init__.py)**:
   - Create active-token vocabulary manifold optimizers `BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer` with `_newton_schulz_coupled_inv_sqrt`.
3. **[`src/qdlblocks/models/standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py) & [`src/qdlblocks/models/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/__init__.py)**:
   - Create `QuantumAttentionMode`, `VocabularyReadoutMode`, `VocabOptimizerMode`, `StandaloneQuantumBornConfig` (`<= 8` fields), `_compiled_mcweeny_uhlmann_krylov_step`, `StandaloneQuantumBornBlock`, `Rank2KrylovBornChunkedCEFn` (Rule-9 single-pass chunked autograd), `StandaloneQuantumBornGPT`, and `StandaloneRiemannianOptimizerBundle`.
4. **[`examples/73_verify_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py)**:
   - Create the Wave 48 verification & Dual-RTX 3090 FineWeb speedrun runner supporting `--mode verify` and `--mode speedrun-330m`.
5. **[`tests/test_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan063_standalone_quantum_born_gpt.py)**:
   - Create the deterministic unit test suite for Wave 48 primitives.

---

## 5. Operational Tooling, Script Opportunities, & Testing

### 5.1 Existing Scripts Used
- `scripts/ast_signature_and_length_linter.py`: Enforces Rule 1 (`<= 1,000` SLOC soft ceiling, zero semicolons) and Rule 2 (`<= 8` parameters per function/dataclass).
- `scripts/verify_plan.py`: Runs automated plan structure verification, AST linting, pytest execution, and JSON hypothesis verification (`python scripts/verify_plan.py --plan-num 63`).
- `scripts/sync_waves_to_webui.py`: Synchronizes Wave 48 benchmark telemetry and model cards with the SQLite database and WebUI (`python scripts/sync_waves_to_webui.py --trigger-reload`).

### 5.2 Build-Before-Launch Scripts
- `examples/73_verify_plan063_standalone_quantum_born_gpt.py`: Executes `H1`–`H4` verification and multi-GPU (`2x RTX 3090`) standalone FineWeb training evaluated on `fineweb_val_000000.bin` (`10,485,760` tokens).

### 5.3 Nice-to-Have Scripts
- `scripts/uhlmann_krylov_phase_and_vocab_snr_probe.py`: Diagnostic CLI for inspecting Uhlmann phase-locking sensitivity, per-layer Krylov eigenvalue gap $\lambda_1 - \lambda_2$, and per-decile vocabulary SNR distributions.

### 5.4 Testing & Verification Commands
```bash
/home/nixos/scratch/.venv/bin/python scripts/ast_signature_and_length_linter.py \
  src/qdlblocks/layers/impure_density_mirror_block.py \
  src/qdlblocks/optim/vocab_manifold_muon.py \
  src/qdlblocks/models/standalone_quantum_born_gpt.py \
  examples/73_verify_plan063_standalone_quantum_born_gpt.py \
  tests/test_plan063_standalone_quantum_born_gpt.py

/home/nixos/scratch/.venv/bin/pytest tests/test_plan063_standalone_quantum_born_gpt.py tests/test_plan062_riemannian_impure_density_gpt.py -q

/home/nixos/scratch/.venv/bin/python examples/73_verify_plan063_standalone_quantum_born_gpt.py --mode verify

/home/nixos/scratch/.venv/bin/python scripts/verify_plan.py --plan-num 63
```
