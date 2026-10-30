# Goldfish Review Report (Stage 7: Critic Review)

**Target:** [`/home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L1-L263) (`Lines 1..263 (FeatureRange)`)

## Reviewer Status

| Reviewer | Status |
| --- | --- |
| `egm-design` | completed |
| `egm-structural` | completed |
| `egm-style` | completed |

**Severity Breakdown:** 0 Blockers, 5 Major, 2 Nits

---

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. Broken Symbol & Line Reference for `SOFTCAP_LINEAR_CE` Readout - Severity: Major
* **Reviewer Lens:** `egm-structural`, `egm-design`
* **Affected Scope:** [`Lines 202-203`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L202-L203) (`Section 2.5: Rule-9 Single-Pass Chunked Vocabulary Readouts`)
* **Discrepancy Description**: Line 203 states that `VocabularyReadoutMode.SOFTCAP_LINEAR_CE` reuses [`compiled_chunked_softcap_linear_ce`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L436-L453). However, no symbol named `compiled_chunked_softcap_linear_ce` exists in [`compiled_leaderboard_classical_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py), and lines `436-453` point to [`CompiledLeaderboardBlock.__init__`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L436-L453). The actual single-pass custom autograd class is [`_CompiledChunkedSoftcapLinearCEFn`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L329-L433) at lines `329-433` (which is also how [`riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L62) imports it).
* **Rationale & Recommendation**: Update line 203 to reference [`_CompiledChunkedSoftcapLinearCEFn`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L329-L433) (or specify a public wrapper function if one is being added) with the corrected line range `#L329-L433`.

### 2. Underspecified `RANK2_KRYLOV_BORN` Readout Equations, Tensor Shapes, and Eigenvalue Routing - Severity: Major
* **Reviewer Lens:** `egm-design`, `egm-structural`
* **Affected Scope:** [`Lines 130-135`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L130-L135), [`Lines 198-201`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L198-L201) (`Section 1` & `Section 2.5.1`)
* **Discrepancy Description**: Section 2.5.1 leaves three implementation-critical aspects of `VocabularyReadoutMode.RANK2_KRYLOV_BORN` (`Rank2KrylovBornChunkedCEFn`) ambiguous:
  1. **Branch Representation & Dimension**: Line 200 writes *"Maps the final normalized hidden state $h_t \in \mathbb{R}^{768}$ into a Rank-2 complex measurement state pair $(|\Psi_{1,t}\rangle, |\Psi_{2,t}\rangle) \in \mathbb{C}^{384} \times \mathbb{C}^{384}$ (or two real measurement branches in $\mathbb{R}^{768}$)"* and contracts against $E \in \mathbb{R}^{V \times 768}$. Leaving $\mathbb{C}^{384}$ vs. $\mathbb{R}^{768}$ parenthetically undecided means the projection parameters and inner-product algebra ($|\langle e_v \mid \Psi_{m,t}\rangle_{\mathbb{C}}|^2$ viewing $e_v \in \mathbb{R}^{768}$ as $\mathbb{C}^{384}$ vs. real dot products) are not locked.
  2. **Source of $(\hat{\lambda}_{1,t}, \hat{\lambda}_{2,t})$**: Each `StandaloneQuantumBornBlock` projects its per-head Krylov modes into $\Delta x_{\text{attn}} \in \mathbb{R}^{768}$ (`Lines 118-120`). It is unspecified whether $(\hat{\lambda}_{1,t}, \hat{\lambda}_{2,t})$ at the final readout are head-averaged Rayleigh quotients forwarded from Layer $L-1$ or predicted from $h_t$.
  3. **Exact Log-Intensity / Softcap Calibration Formula**: Line 133 writes $p_t(v) \propto \lambda_1 |\langle e_v|\Psi_1\rangle|^2 + \lambda_2 |\langle e_v|\Psi_2\rangle|^2$, whereas line 201 states *"applies temperature-scaled log-intensity / softcap calibration"*. Without the exact closed-form logit equation $z_{t,v}(I_{t,v})$, the single-pass backward gradient $\partial \mathcal{L} / \partial h_t$ and $\partial \mathcal{L} / \partial E$ required by Rule 9 cannot be uniquely implemented.
* **Rationale & Recommendation**: Lock a single mathematical formulation in Section 2.5.1 specifying: (a) how $|\Psi_{1,t}\rangle, |\Psi_{2,t}\rangle$ and $\hat{\lambda}_{1,t}, \hat{\lambda}_{2,t}$ are computed from $h_t$ (or final-layer telemetry), (b) how complex/real inner products with $E \in \mathbb{R}^{V \times 768}$ are formed to yield $I_{t,v}$, and (c) the exact calibrated logit formula $z_{t,v}$ and its single-pass derivative.

### 3. Contradictory Causal Attention Weight Formulation (`Softmax_s(Scale * Re⟨q_t | k_s⟩)` vs. Polynomial $p$-Boson Born Weights) - Severity: Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** [`Lines 100-102`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L100-L102), [`Line 154`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L154) (`Section 1` & `Section 2.2`)
* **Discrepancy Description**: In Section 1 (Box 3, Line 101), the system diagram defines the causal attention weights as:
  `Causal Born / Softmax-Aligned Weights W_ts,h = Softmax_s(Scale * Re⟨q_t | k_s⟩)`
  which uses exponential `Softmax` on only the real part $\operatorname{Re}\langle q_t \mid k_s\rangle$ (discarding the imaginary part $\operatorname{Im}\langle q_t \mid k_s\rangle$ of the complex inner product). By contrast, Wave 47's [`RiemannianImpureDensityBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L429-L436) and [`AgyRewrittenGeminiBornComplexPackedWithRoPE.py`](file:///home/nixos/QuantumDLBlocks/AgyRewrittenGeminiBornComplexPackedWithRoPE.py#L196-L203) compute full complex modulus-squared $p$-boson Born weights $W_{ts,h} = \frac{|\langle q_t \mid k_s\rangle_{\mathbb{C}}|^{2p}}{\sum_{s' \le t} |\langle q_t \mid k_{s'}\rangle_{\mathbb{C}}|^{2p} + \varepsilon}$ ($p=2$). Section 2.2 then refers to $W_{\text{attn}}$ without defining its equation.
* **Rationale & Recommendation**: Explicitly specify the exact causal attention weight equation in Section 2 (and align Box 3 in Section 1) so there is no ambiguity as to whether `StandaloneQuantumBornBlock` uses $p$-boson Born modulus-squared weights $|\langle q_t \mid k_s\rangle_{\mathbb{C}}|^{2p}$ or scaled softmax attention.

### 4. Mach-Zehnder Channel Mixer Dimension (`3072` vs. `1536`), Gating Formula, and Stiefel Transpose Convention - Severity: Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** [`Lines 122-126`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L122-L126), [`Lines 172-173`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L172-L173) (`Section 1` & `Section 2.3.1`)
* **Discrepancy Description**:
  1. In Wave 47 ([`riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L280)), `channel_hidden_dim` is `2 * embed_dim = 1536`, whereas Lines 123–126 and 173 specify `3072` ($4 \times 768$) while Line 29 earlier writes $W_{\text{chan}} \in \text{St}(d_{\text{chan}}, 768)$.
  2. Line 123 writes `w_beam_a, w_beam_b ∈ St(3072, 768)` and Line 126 writes `W_chan ∈ St(3072, 768)`. In PyTorch (`m_norm @ w_beam_a` and `h_int @ w_chan`), `w_beam_a` has shape `(768, D_chan)` (wide matrix with $W W^\top = I_{768}$), whereas `w_chan` has shape `(D_chan, 768)` (tall matrix with $W^\top W = I_{768}$).
  3. Line 124 writes `Interferometric gating φ_a(u_a) ⊙ u_b`, whereas [`RiemannianImpureDensityBlock.compute_channel_features`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L473-L479) computes the Mach-Zehnder polarization identity `F.relu(beam_a + beam_b).square() - 0.25 * F.relu(beam_a - beam_b).square()`.
* **Rationale & Recommendation**: Clarify the exact `channel_hidden_dim` (`1536` vs. `3072`), state the exact interferometric activation equation, and distinguish the PyTorch tensor shapes `(768, D_chan)` for `w_beam_a, w_beam_b` from `(D_chan, 768)` for `w_chan`.

### 5. Rectangular Stiefel Horizontal Projection ($m > n$) vs. Symmetric Projection ($m = n$) in `RiemannianStiefelMuon` - Severity: Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** [`Lines 174-182`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L174-L182) (`Section 2.3.2: Tikhonov-Damped Tangent Newton-Schulz`)
* **Discrepancy Description**: Section 2.3.2 states that for any tall Stiefel parameter $W \in \text{St}(m, n)$ ($m \ge n$), the gradient is projected via $G_{\text{tan}} = G - W \operatorname{sym}(W^\top G)$, followed by 5 Newton-Schulz steps and Frobenius rescaling $U_{\text{tan}} = \Pi_{T_W}(X_5) \frac{\sqrt{n}}{\|\Pi_{T_W}(X_5)\|_F + \varepsilon}$. However, as documented and implemented in [`riemannian_stiefel_muon_step`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L268-L290):
  - When $m > n$ (rectangular Stiefel, such as $W_{\text{obs}} \in \text{St}(1920, 768)$ and $W_{\text{q}}, W_{\text{k}} \in \text{St}(768, 192)$), Newton-Schulz orthogonalization does **not** commute with a non-zero skew-symmetric block $W^\top G - G^\top W$; keeping $W^\top U_{\text{tan}} + U_{\text{tan}}^\top W = 0$ and $U_{\text{tan}}^\top U_{\text{tan}} = I_n$ so that $(W - \eta U_{\text{tan}}) / \sqrt{1 + \eta^2}$ stays on $\text{St}(m, n)$ requires the **horizontal Grassmann projection** $M_{\text{horiz}} = G_{\text{nest}} - W(W^\top G_{\text{nest}})$ ($W^\top U_{\text{tan}} = 0$) when $m \neq n$, plus a post-retraction 1-step Schulz polish.
  - Using Frobenius scalar normalization instead of unit-singular-value orthogonalization after $\Pi_{T_W}$ breaks $U_{\text{tan}}^\top U_{\text{tan}} = I_n$, causing the scalar geodesic retraction $(W - \eta U_{\text{tan}}) / \sqrt{1 + \eta^2}$ to drift off the Stiefel manifold and violate `G3` (`< 1e-5` manifold error).
* **Rationale & Recommendation**: Update Section 2.3.2 to match [`riemannian_stiefel_muon_step`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L268-L290): specify horizontal projection $G - W(W^\top G)$ for rectangular $m > n$ (and $G - W\operatorname{sym}(W^\top G)$ for square $m = n$), inject Tikhonov Gram damping $A_0 = X_0^\top X_0 + \lambda_{\text{tik}} I_n$ inside `_newton_schulz5_tall_real`, and retain the 1-step Schulz manifold polish after retraction.

### 6. Unspecified Scaling of Packed Density Relative to Krylov Modes in Option B (`320`-Dim/Head) Readout - Severity: Nit
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** [`Line 118`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L118), [`Lines 167-169`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L167-L169) (`Section 1` & `Section 2.2.4`)
* **Discrepancy Description**: In Wave 47 ([`riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L467-L469)), the packed output density was multiplied by `float(self.d_h)` ($16.0$) before readout. In Wave 48's Option B readout $O_{b,h,t} = [\tilde{P}(\rho_{\text{out}}), \sqrt{\lambda_1}\phi_{1,\text{re}}, \sqrt{\lambda_1}\phi_{1,\text{im}}, \sqrt{\lambda_2}\phi_{2,\text{re}}, \sqrt{\lambda_2}\phi_{2,\text{im}}] \in \mathbb{R}^{320}$, multiplying $\tilde{P}(\rho_{\text{out}})$ by $16.0$ would make the `256` quadratic density channels $16\times$ larger in L2 norm than the `64` linear Krylov eigenmode channels (whose combined L2 norm is $\sqrt{\lambda_1 + \lambda_2} \le 1$).
* **Rationale & Recommendation**: Explicitly state in Section 2.2.4 whether $\tilde{P}(\rho_{\text{out}})$ is concatenated at unit isometric scale ($\|\tilde{P}(\rho_{\text{out}})\|_2 = \sqrt{\operatorname{Tr}(\rho_{\text{out}}^2)} \le 1$, matching the unit scale of $\sqrt{\lambda_1}\phi_1 \oplus \sqrt{\lambda_2}\phi_2$) so the linear Krylov modes are not drowned out by a legacy $d_h = 16$ multiplier.

### 7. Misattributed `AGENTS.md` Rule 3 Description and Missing Clickable `file://` Links - Severity: Nit
* **Reviewer Lens:** `egm-style`, `egm-structural`
* **Affected Scope:** [`Line 58`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L58), [`Line 68`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L68), [`Line 184`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L184), [`Lines 244-245`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L244-L245)
* **Discrepancy Description**:
  1. Line 245 states *"zero semicolons per Rule 3"*, whereas in [`AGENTS.md`](file:///home/nixos/QuantumDLBlocks/AGENTS.md#L65-L70) semicolon packing is prohibited under **Rule 1**, while **Rule 3** is *"Strategy / Polymorphic Dispatch Over Boolean Flag Jungles"*.
  2. File references in [`Line 58`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L58) (`examples/73_verify_plan063_standalone_quantum_born_gpt.py`), [`Line 68`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L68), [`Line 184`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L184) (`src/qdlblocks/optim/vocab_manifold_muon.py`), and [`Lines 244-245`](file:///home/nixos/QuantumDLBlocks/docs/designs/wave48_standalone_quantum_born_gpt.md#L244-L245) (`AGENTS.md`, `scripts/ast_signature_and_length_linter.py`) are plain inline code rather than clickable `file://` links.
* **Rationale & Recommendation**: Correct the Rule 1 / Rule 3 attribution on line 245 and convert the unlinked repository paths in `FeatureRange` to clickable `file://` Markdown links.
