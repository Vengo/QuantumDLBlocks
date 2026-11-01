# EXPERIMENT PLAN 064 (Wave 49): Modular Quantum Statistical-Mechanics GPT, Mean-Centered Isometric Born Trace Readout, CPTP Kraus Channels, and Extended FineWeb Scaling

## 1. Executive Summary & Prioritized Experimental Objectives

Building on Wave 48 (`EXPERIMENT_PLAN_063.md`, documented in [`docs/SESSION_WRAPUP_WAVE48_QUANTUM_BORN_GPT.md`](file:///home/nixos/QuantumDLBlocks/docs/SESSION_WRAPUP_WAVE48_QUANTUM_BORN_GPT.md)) and our theoretical analysis in *"Top-1 Accuracy vs Cross-Entropy Divergence"*, Wave 49 (`EXPERIMENT_PLAN_064.md`) factors the monolithic [`StandaloneQuantumBornGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py) implementation into reusable library modules under `src/qdlblocks/` and executes seven prioritized experimental investigations on FineWeb:

1. **Priority 1 — Extended `600M`-Token Benchmark of Upgraded `Wave 48 v3` (+ Readout/Vocab Fixes)**:
   - Scale the rebuilt `Wave 48 v3` architecture (`SOFTCAP_LINEAR_CE`, clamped `logit_scale`, `0.75 * step_active + 0.25 * step_full` vocabulary hypersphere updates, $Q, K \in \mathbb{C}^{64}$ with $p=3$ Born coincidence $|\langle q_t \mid k_s \rangle|^6$, and $R=4$ wavepacket branch transport) up to `600M` FineWeb tokens (`73,242` steps at `8,192` tokens/step across Shards 1–6), recording validation cross-entropy, perplexity, Top-1/Top-5 accuracy, rank-bucketed NLL, and quantum purity/effective-rank telemetry at regular milestones (`4.10M`, `8.19M`, `20.48M`, `50M`, `100M`, `200M`, `300M`, `400M`, `500M`, `600M` tokens).
2. **Priority 2 — Modular Refactoring & `100M`-Token Comparison (Rebuilt `Wave 48 v3` vs. Mean-Centered Isometric Born Trace Readout)**:
   - Factor reusable quantum statistical-mechanics operators into [`src/qdlblocks/layers/quantum_statmech_channels.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_statmech_channels.py) and [`src/qdlblocks/layers/isometric_born_trace_readout.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/isometric_born_trace_readout.py), and compose them in [`src/qdlblocks/models/modular_quantum_statmech_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/modular_quantum_statmech_gpt.py) (`ModularQuantumStatMechGPT`).
   - Verify exact numerical equivalence with `StandaloneQuantumBornGPT` (`Wave 48 v3`) at Step 0 and across multi-step updates, and run a head-to-head `100M`-token comparison between:
     - **Arm 2A**: Rebuilt `Wave 48 v3` (`VocabularyReadoutMode.SOFTCAP_LINEAR_CE`), and
     - **Arm 2B**: **Mean-Centered Isometric Born Trace Readout** (`VocabularyReadoutMode.MEAN_CENTERED_ISOMETRIC_TRACE`), which combines signed coherent amplitude overlap $c_{\text{sharp}}(t,v) = s_{\text{eff}}\langle h_{\text{final},t}, w_{\text{lm},v}\rangle$ with the mean-centered register-averaged Hilbert-Schmidt Born trace $\tilde{c}_{\text{trace}}(t,v) = \frac{1}{H}\sum_{h=1}^H \langle \tilde{P}(\rho_{t,h}), \tilde{P}(E_{v,h})\rangle - \frac{1}{d_h}$ inside a Rule-9 single-pass chunked autograd function (`MeanCenteredIsometricBornTraceCEFn`).
3. **Priority 3 — `50M`-Token Completely Positive Trace-Preserving (CPTP) Kraus Channel Benchmark**:
   - Replace closed-system unitary conjugation $U_h \rho_{t,h} U_h^\dagger$ ($U_h \in U(16)$) with an open-system **Rank-$K=4$ CPTP Kraus quantum channel**:
     $$\mathcal{E}_h(\rho_{t,h}) = \sum_{k=1}^K A_{h,k}\,\rho_{t,h}\,A_{h,k}^\dagger, \qquad \sum_{k=1}^K A_{h,k}^\dagger A_{h,k} = I_{16}$$
     parameterized by a tall complex Stiefel matrix $A_h = [A_{h,1}^\top, \dots, A_{h,K}^\top]^\top \in \mathrm{St}_{\mathbb{C}}(K d_h, d_h) = \mathrm{St}_{\mathbb{C}}(64, 16)$ updated via Tikhonov-damped complex Stiefel Newton-Schulz retraction, and benchmark on `50M` FineWeb tokens (`6,104` steps).
4. **Priority 4 — Embedding Trainability & Representational Geometry Investigation**:
   - Test whether the Stiefel projection matrices ($W_{\text{in}}, W_q, W_k, W_v \in \mathrm{St}(768, 768)$) and untied `lm_head` provide sufficient representational power when initial token embeddings `wte` are frozen, comparing:
     - **Arm 4A (`TRAINABLE_SPHERE_MANIFOLD`)**: Random sphere init + active-token `BatchSubspaceTikhonovGramMuon`,
     - **Arm 4B (`FROZEN_RANDOM_SPHERE`)**: Frozen random unit-RMS sphere embeddings (Johnson-Lindenstrauss random projection),
     - **Arm 4C (`FROZEN_PPMI_SVD_SPHERE`)**: Frozen Shifted-PPMI + Randomized SVD co-occurrence spectral embeddings on $S^{767}(\sqrt{768})$.
5. **Priority 5 — Readout Divergence Benchmark (Rank-Bucketed NLL Telemetry)**:
   - Compare `SOFTCAP_LINEAR_CE`, `RANK2_KRYLOV_BORN`, and `MEAN_CENTERED_ISOMETRIC_TRACE` using rank-bucketed validation NLL decomposition (`Val CE`, `Top-1`, `Top-5`, `Gap(Top5 - Top1)`, `NLL(Rank 1)`, `NLL(Rank 2-5)`, `NLL(Rank 6-50)`, `NLL(Rank > 50)`).
6. **Priority 6 — Quantum Purification Comparison (Quadratic vs. Cubic Fermi-Dirac vs. Learnable Temperature)**:
   - Compare `PurificationMode.QUADRATIC_MCWEENY` ($\rho^2 / \mathrm{Tr}(\rho^2)$), `PurificationMode.CUBIC_FERMI_DIRAC` ($(3\rho^2 - 2\rho^3) / \mathrm{Tr}(3\rho^2 - 2\rho^3)$), and `PurificationMode.LEARNABLE_TEMP_MCWEENY` (per-head learnable convex polynomial purification).
7. **Priority 7 — Detached Purity / Tsallis-2 Entropy Gate Investigation**:
   - Evaluate `EntropyGateMode.DETACHED_TSALLIS_PURITY_GATE`, which uses detached purity $\mathcal{P}_{\text{det}} = \mathrm{Tr}(\rho^2).\text{detach}()$ and Tsallis-2 linear entropy $S_2 = 1 - \mathcal{P}_{\text{det}}$ to gate the coherent wavepacket branch $\psi_U$ vs. the impure density/Krylov branches while restoring head-level thermodynamic confidence weighting without radial purity-collapse gradients.

---

## 2. Top-of-File Named Constants Ledger (Rule 7)

| Constant Name | Value | Description |
| :--- | :---: | :--- |
| `DEFAULT_STATMECH_VOCAB_SIZE` | `50304` | Padded GPT-2 vocabulary size (`64`-aligned) |
| `DEFAULT_STATMECH_EMBED_DIM` | `768` | Model hidden dimension $d_{\text{model}}$ |
| `DEFAULT_STATMECH_NUM_LAYERS` | `12` | Number of quantum transformer blocks $L$ |
| `DEFAULT_STATMECH_NUM_HEADS` | `6` | Number of quantum heads $H$ |
| `DEFAULT_STATMECH_HEAD_COMPLEX_DIM` | `16` | Complex Hilbert dimension $d_h$ per head ($\mathrm{Herm}(16) \cong \mathbb{R}^{256}$) |
| `DEFAULT_STATMECH_VALUE_BRANCHES` | `4` | Wavepacket branch rank $R$ per head ($Q, K \in \mathbb{C}^{64}$) |
| `DEFAULT_KRAUS_CHANNEL_RANK` | `4` | Default CPTP Kraus operator rank $K$ ($A_h \in \mathrm{St}_{\mathbb{C}}(64, 16)$) |
| `DEFAULT_KRAUS_DOMINANT_WEIGHT` | `0.85` | Step-0 trace weight on dominant coherent Kraus branch $A_{h,1}$ |
| `DEFAULT_ISOMETRIC_TRACE_UNIFORM_MEAN` | `0.0625` | Uniform trace baseline $1 / d_h = 1 / 16$ for mean-centered Born trace |
| `DEFAULT_ISOMETRIC_TRACE_INIT_BETA` | `2.0` | Initial coupling weight $\beta_{\text{trace}}$ for mean-centered Born trace |
| `DEFAULT_TSALLIS_PURITY_CENTER` | `0.35` | Equilibrium purity center $\mathcal{P}_0$ for detached Tsallis/purity gate |

---

## 3. Verification Gates (`H1`–`H5`)

- **Gate `H1` (Exact Refactoring Equivalence)**: With identical random seed, `ModularQuantumStatMechGPT` configured with default `Wave 48 v3` modes produces Step-0 forward loss and logits matching `StandaloneQuantumBornGPT` within `< 1e-6` (`float32`) / exact `bfloat16` match.
- **Gate `H2` (Single-Pass Mean-Centered Isometric Born Trace Autograd)**: `MeanCenteredIsometricBornTraceCEFn` executes `0` matmuls in `backward()` (Rule 9) and matches analytical PyTorch reference gradients with cosine similarity `> 0.9999`.
- **Gate `H3` (CPTP Kraus Complex Stiefel Manifold Preservation)**: Across training steps under `QuantumChannelMode.CPTP_KRAUS_RANK_K`, `max_kraus_stiefel_err` ($\frac{1}{\sqrt{d_h}}\|\sum_{k=1}^K A_{h,k}^\dagger A_{h,k} - I_{d_h}\|_F$) remains `< 1e-5` and output density matrices remain Hermitian PSD with unit trace.
- **Gate `H4` (Purification & Tsallis Gate Physicality)**: `QUADRATIC_MCWEENY`, `CUBIC_FERMI_DIRAC`, and `LEARNABLE_TEMP_MCWEENY` preserve unit trace and non-negative Rayleigh quotients, and `DETACHED_TSALLIS_PURITY_GATE` induces zero radial gradient through $\mathrm{Tr}(\rho^2)$.
- **Gate `H5` (AST & SLOC Linter Compliance)**: All new modules, tests, and scripts pass `scripts/ast_signature_and_length_linter.py` (`<= 1,000` SLOC, `<= 8` args per function, `0` semicolons).
