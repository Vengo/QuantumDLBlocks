# Consolidated Goldfish Review Report — `wave48_standalone_quantum_born_gpt.md`

**Target Document**: [`/home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md) (`FeatureRange: 1..263`, `DetailedRange: 264..345`)
**Stage Reports**:
1. **Stage 5 (Initial Zoom Out)**: [`zoom_out_report.md`](file:///home/nixos/QuantumDLBlocks/zoom_out_report.md)
2. **Stage 6 (Comprehension Test)**: [`comprehension_report.md`](file:///home/nixos/QuantumDLBlocks/comprehension_report.md)
3. **Stage 7 (Critic Review)**: [`critic_report.md`](file:///home/nixos/QuantumDLBlocks/critic_report.md)
4. **Stage 8 (Intermediate Zoom Out)**: [`intermediate_zoom_out_report.md`](file:///home/nixos/QuantumDLBlocks/intermediate_zoom_out_report.md)
5. **Stage 9 (Implementation Readiness Test)**: [`implementation_readiness_report.md`](file:///home/nixos/QuantumDLBlocks/implementation_readiness_report.md)

**Consolidated Severity Summary**: `3` 🔴 Blockers, `5` 🟡 Majors, `2` 🟢 Nits

---

## 1. Consolidated Actionable Tasks (Prioritized: 🔴 Blocker $\to$ 🟡 Major $\to$ 🟢 Nit)

### 🔴 Blocker 1 (`C-B1`): Lock `RANK2_KRYLOV_BORN` Readout Equations and Explicit Sparse (`wte`) vs. Dense (`lm_head`) Optimizer Assignment
* **Source Findings**: Stage 5 Finding 1, Stage 6 Finding 1, Stage 7 Finding 2, Stage 8 Finding 1, Stage 9 Finding 3.
* **Affected Lines**: [`lines 31–33, 130–135, 184–204, 306`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L130-L204)
* **Issue**:
  1. Section 2.5 leaves `VocabularyReadoutMode.RANK2_KRYLOV_BORN` (`Rank2KrylovBornChunkedCEFn`) parenthetically undecided between complex $\mathbb{C}^{384} \times \mathbb{C}^{384}$ vs. real $\mathbb{R}^{768}$ branches, omits the source of $(\hat{\lambda}_{1,t}, \hat{\lambda}_{2,t})$, and omits the closed-form calibrated logit equation $z_{t,v}(I_{t,v})$.
  2. Section 2.4 does not specify whether `wte` and `lm_head` are weight-tied or untied, or how `wte` (sparse non-zero gradients strictly on $\mathcal{I}_{\text{in}} = \operatorname{unique}(\text{idx})$) vs. `lm_head` (dense non-zero partition-function gradients across all $V = 50,304$ rows) are stepped.
* **Required Remediation**:
  1. **Zero-Extra-Parameter Branch Partition**: Partition the final RMS-normalized hidden state $h_t = [h_{1,t}, h_{2,t}] \in \mathbb{R}^{384} \oplus \mathbb{R}^{384}$ and vocabulary matrix $E = [E_1, E_2] \in \mathbb{R}^{V \times 384} \oplus \mathbb{R}^{V \times 384}$, weighted by the normalized head-averaged Krylov Rayleigh quotients $(\hat{\lambda}_{1,t}, \hat{\lambda}_{2,t})$ forwarded from the final quantum block:
     $$I_{t,v} = \hat{\lambda}_{1,t} \langle e_{1,v}, h_{1,t}\rangle^2 + \hat{\lambda}_{2,t} \langle e_{2,v}, h_{2,t}\rangle^2 + \varepsilon, \qquad z_{t,v} = 15 \tanh\!\left(\frac{\tau_{\text{born}} \log(I_{t,v})}{15}\right)$$
     with top-of-file constant `DEFAULT_BORN_READOUT_TEMP = 4.0`, computed in chunks of `chunk_tokens = 8192` with single-pass analytical backward gradient accumulation.
  2. **Untied `wte` & `lm_head` Optimizer Assignment**: Keep `wte` and `lm_head` untied (matching [`RiemannianImpureDensityGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L706-L732)). Step `wte.weight` via the active-token manifold optimizer strictly on $\mathcal{I}_{\text{in}} = \operatorname{unique}(\text{idx})$, and step `lm_head.weight` on active union rows $\mathcal{I}_{\text{act}} = \operatorname{unique}(\text{idx} \cup \text{targets})$ (or via the `AdamW` head group).

---

### 🔴 Blocker 2 (`C-B2`): Promote `_complex_newton_schulz5` and `riemannian_complex_unitary_muon_step` Into `impure_density_mirror_block.py` (`<= 8` Args)
* **Source Findings**: Stage 5 Finding 2, Stage 8 Finding 3, Stage 9 Findings 2 & 4.
* **Affected Lines**: [`lines 182, 276–285, 307`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L276-L285)
* **Issue**:
  * [`_complex_newton_schulz5`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L114-L138) and [`riemannian_complex_unitary_muon_step`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L141-L210) currently reside in [`src/qdlblocks/models/riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py), which `NG3` (line 47) freezes against modification.
  * In addition, `riemannian_complex_unitary_muon_step` already has 8 parameters in Wave 47, so appending `tikhonov_lambda` as a 9th parameter would fail [`scripts/ast_signature_and_length_linter.py`](file:///home/nixos/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) (Rule 2: `<= 8` args).
* **Required Remediation**:
  * Promote `_complex_newton_schulz5(g_re, g_im, steps=5, eps=1e-7, tikhonov_lambda=0.0)` (`5` args) and `riemannian_complex_unitary_muon_step(u_re, u_im, grad_re, grad_im, mom_re, mom_im, lr=0.02, tikhonov_lambda=0.0)` (`8` args, using constant `DEFAULT_STIEFEL_MOMENTUM_BETA = 0.95` internally or tuple packing) into [`src/qdlblocks/layers/impure_density_mirror_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py) with batched 3D `(H, 16, 16)` support.

---

### 🔴 Blocker 3 (`C-B3`): Include Mandatory Headings Required by `scripts/verify_plan.py` in `EXPERIMENT_PLAN_063.md`
* **Source Findings**: Stage 5 Finding 5, Stage 8 Finding 5, Stage 9 Finding 1.
* **Affected Lines**: [`lines 268–275, 335–336`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L268-L275)
* **Issue**:
  * [`scripts/verify_plan.py`](file:///home/nixos/QuantumDLBlocks/scripts/verify_plan.py#L31-L36) enforces `REQUIRED_PLAN_HEADINGS = ("Hypotheses", "Planned", "Testing", "Operational Tooling")`. Target File 1's specification omitted `"Planned"`, `"Testing"`, and `"Operational Tooling"`.
* **Required Remediation**:
  * Specify the 5 mandatory sections in `EXPERIMENT_PLAN_063.md`: `1. Executive Summary`, `2. Quantitative Hypotheses (H1–H4)`, `3. Top-of-File Named Constants Ledger`, `4. Planned Changes`, and `5. Operational Tooling, Script Opportunities, & Testing`.

---

### 🟡 Major 1 (`C-M1`): Reuse `unpack_hermitian_isometric` in `_compiled_mcweeny_uhlmann_krylov_step` and Fix `_CompiledChunkedSoftcapLinearCEFn` Link
* **Source Findings**: Stage 5 Finding 3, Stage 6 Finding 4, Stage 7 Finding 1, Stage 8 Finding 3, Stage 9 Finding 4.
* **Affected Lines**: [`lines 202–203, 304`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L202-L304)
* **Required Remediation**:
  * Simplify `_compiled_mcweeny_uhlmann_krylov_step(packed_mixed, u_re, u_im, re_pack_mask, im_pack_mask)` to 5 arguments by reusing [`unpack_hermitian_isometric`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L106-L117) instead of passing 5 custom index/scale/mask buffers.
  * Update line 203 to reference [`_CompiledChunkedSoftcapLinearCEFn`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L329-L433).

---

### 🟡 Major 2 (`C-M2`): Specify Coupled Newton-Schulz Recurrence and Remove Confounded Second SNR Gate in `BatchSubspaceTikhonovGramMuon`
* **Source Findings**: Stage 5 Finding 4, Stage 6 Finding 2, Stage 8 Finding 2, Stage 9 Finding 5.
* **Affected Lines**: [`lines 186–195, 290–295`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L186-L295)
* **Required Remediation**:
  * Specify the coupled Newton-Schulz recurrence $Y_0 = A, Z_0 = I_d$, $T_k = \frac{1}{2}(3 I_d - Z_k Y_k), Y_{k+1} = Y_k T_k, Z_{k+1} = T_k Z_k$ for $A^{-1/2}$.
  * Remove the duplicate adaptive median SNR gate from `BatchSubspaceTikhonovGramMuon` so `BatchSubspaceTikhonovGramMuon` isolates $d \times d$ Tikhonov Gram preconditioning + radial sphere retraction while `RiemannianSphereTangentSNROptimizer` isolates per-row SNR trust-region geodesic steps.

---

### 🟡 Major 3 (`C-M3`): Align Causal $p$-Boson Born Attention, $V_0$ Highway, Mach-Zehnder Polarization Identity, and Unit Isometric Observable Scale
* **Source Findings**: Stage 5 Finding 4, Stage 6 Finding 3, Stage 7 Findings 3, 4, & 6, Stage 8 Finding 4, Stage 9 Finding 6.
* **Affected Lines**: [`lines 77–127, 154–173, 305`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L77-L173)
* **Required Remediation**:
  * Lock causal $p$-boson Born attention weights $W_{ts,h} = |\langle q_{t,h} \mid k_{s,h}\rangle_{\mathbb{C}}|^{2p} \mathbf{1}_{s \le t} / (\sum_{s' \le t} |\langle q_{t,h} \mid k_{s',h}\rangle_{\mathbb{C}}|^{2p} + \varepsilon)$ ($p=2$).
  * Document `block_lambdas`, `v_lambdas` ($V_0$ highway), and the Mach-Zehnder polarization identity $\operatorname{ReLU}(u_a + u_b)^2 - \frac{1}{4}\operatorname{ReLU}(u_a - u_b)^2$ with `d_chan = 1536` (`w_beam_a, w_beam_b` of shape `(768, 1536)` and `w_chan` of shape `(1536, 768)`), matching [`RiemannianImpureDensityBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L280-L479).
  * Specify unit isometric scale (`1.0`, no `* 16.0` multiplier) on $\tilde{P}(\rho_{\text{out}}) \in \mathbb{R}^{256}$ so the `256` density channels and `64` Krylov channels are balanced entering $W_{\text{obs}} \in \text{St}(1920, 768)$.

---

### 🟡 Major 4 (`C-M4`): Align Rectangular Stiefel Horizontal Projection ($m > n$) in Section 2.3.2
* **Source Findings**: Stage 7 Finding 5, Stage 8 Finding 3.
* **Affected Lines**: [`lines 174–182`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L174-L182)
* **Required Remediation**:
  * Specify horizontal projection $G - W(W^\top G)$ for rectangular Stiefel matrices ($m \neq n$) and symmetric projection $G - W\operatorname{sym}(W^\top G)$ for square Stiefel matrices ($m = n$), followed by Tikhonov-damped `_newton_schulz5_tall_real`, geodesic retraction $(W - \eta U_{\text{tan}})/\sqrt{1+\eta^2}$, and 1-step Schulz manifold polish, matching [`riemannian_stiefel_muon_step`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L266-L290).

---

### 🟡 Major 5 (`C-M5`): Specify `H1` Phase-Erased `256`-Width Baseline Dispatch and Package `__init__.py` Exports
* **Source Findings**: Stage 9 Findings 7 & 8.
* **Affected Lines**: [`lines 266–328`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L266-L328)
* **Required Remediation**:
  * Replace fixed `head_dim_complex=16` on `StandaloneQuantumBornConfig` with `attention_mode: QuantumAttentionMode = QuantumAttentionMode.UHLMANN_KRYLOV_320` (`UHLMANN_KRYLOV_320` vs. `PHASE_ERASED_DENSITY_256`, keeping `<= 8` fields on `StandaloneQuantumBornConfig`) so `H1` compares both modes under identical architecture/optimizer framing.
  * Include `__init__.py` export updates in [`src/qdlblocks/layers/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/__init__.py), [`src/qdlblocks/optim/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/__init__.py), and [`src/qdlblocks/models/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/__init__.py).

---

### 🟢 Nit 1 (`C-N1`): Define Traps 1–3 Inline and Fix Rule 1 / Rule 3 Citation & Clickable `file://` Links
* **Source Findings**: Stage 6 Findings 5 & 6, Stage 7 Finding 7, Stage 8 Finding 5, Stage 9 Finding 9.
* **Affected Lines**: [`lines 58, 63, 187, 227, 244–245, 252, 327`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L58-L327)

### 🟢 Nit 2 (`C-N2`): Strip Subjective Adjectives ("clean", "graceful") in Implementation Descriptions
* **Source Findings**: Stage 9 Finding 10.
* **Affected Lines**: [`lines 300, 304`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L300-L304)

---

## 2. Cross-Stage Traceability Matrix

| Consolidated ID | Severity | Stage 5 (`zoom_out`) | Stage 6 (`comprehension`) | Stage 7 (`critic`) | Stage 8 (`interm_zoom_out`) | Stage 9 (`impl_readiness`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C-B1`** (`RANK2_KRYLOV_BORN` & `wte`/`lm_head` split) | 🔴 Blocker | Finding 1 | Finding 1 | Finding 2 | Finding 1 | Finding 3 |
| **`C-B2`** (Promote complex $U(16)$ Muon, `<= 8` args) | 🔴 Blocker | Finding 2 | — | — | Finding 3 | Findings 2, 4 |
| **`C-B3`** (`EXPERIMENT_PLAN_063.md` `verify_plan.py` headings) | 🔴 Blocker | Finding 5 | — | — | Finding 5 | Finding 1 |
| **`C-M1`** (Reuse `unpack_hermitian_isometric` & `_CompiledChunkedSoftcapLinearCEFn`) | 🟡 Major | Finding 3 | Finding 4 | Finding 1 | Finding 3 | Finding 4 |
| **`C-M2`** (Coupled NS $A^{-1/2}$ & drop duplicate SNR gate in Gram Muon) | 🟡 Major | Finding 4 | Finding 2 | — | Finding 2 | Finding 5 |
| **`C-M3`** ($p$-Boson Born weights, $V_0$ highway, Mach-Zehnder `1536`, unit scale) | 🟡 Major | Finding 4 | Finding 3 | Findings 3, 4, 6 | Finding 4 | Finding 6 |
| **`C-M4`** (Rectangular Stiefel horizontal projection in Section 2.3.2) | 🟡 Major | — | — | Finding 5 | Finding 3 | — |
| **`C-M5`** (`QuantumAttentionMode` for `H1` (`<= 8` fields) & `__init__.py` exports) | 🟡 Major | — | — | — | — | Findings 7, 8 |
| **`C-N1`** (Traps 1–3 inline definitions, Rule 1/3 citation, `file://` links) | 🟢 Nit | — | Findings 5, 6 | Finding 7 | Finding 5 | Finding 9 |
| **`C-N2`** (Strip subjective adjectives `"clean"`, `"graceful"`) | 🟢 Nit | — | — | — | — | Finding 10 |
