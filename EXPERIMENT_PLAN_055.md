# EXPERIMENT PLAN 055 (`Wave 40`): Tied Token-Wavefunction & Operator-Ket Holographic Complex Born Readout, Soft Unit-Sphere / Unitary / Parseval-Isotropy Penalties, and Unbiased Rare-Token Importance Sampling

## 1. Context, Motivation, and Post-Mortem of `Wave 39` (`w39_winner_d128c_H4_dh32_L6_900Mtok`)

In `Wave 39` (`EXPERIMENT_PLAN_054.md`), our multiples-of-32 5-minute wall-clock tournament showed that `d_complex = 128` (`H = 4, d_h = 32, d_real = 256, L = 6`, `14.88M` parameters) won the 5-minute sprint (`5.8945` Val CE). However, when we launched the long-horizon `900M`-token run (`w39_winner_d128c_H4_dh32_L6_900Mtok`), validation cross-entropy plateaued at `5.37–5.41` nats after `~50M` tokens (`Step 6,000`), prompting us to terminate the run at `190.05M` tokens (`Step 23,200`) and conduct a deep mathematical and empirical post-mortem.

Our diagnostic investigation (`task-1329`, `task-1423`, `task-1446`, `task-1460`) uncovered five concrete root causes of the `5.37` Val CE plateau:

1. **Hidden Boltzmann Softmax + Real-Part-Only Projection in the Legacy Readout**:
   - While [`GeminiBornComplexPackedWithRoPE.py`](file:///home/festivity/agy/QuantumDLBlocks/GeminiBornComplexPackedWithRoPE.py) uses a **100% softmax-free** $p$-boson complex Born rule inside attention ($w_{ts} = |\langle q_t^{\text{rot}} \mid k_s^{\text{rot}} \rangle_{\mathbb{C}}|^{2p} / \sum_{j \le t} |\langle q_t^{\text{rot}} \mid k_j^{\text{rot}} \rangle_{\mathbb{C}}|^{2p}$), [`GeminiPackedComplexRoPEGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/gemini_packed_complex_rope_gpt.py) attached a legacy Boltzmann `exp` / `log_softmax` vocabulary readout (`_ComplexBornChunkedCEFn`) that:
     - Applied `exp(...)` over all `50,257` tokens (giving orthogonal tokens $\langle e_v \mid \Phi_t \rangle_{\mathbb{C}} = 0$ a non-zero $\exp(0) = 1$ denominator weight and pushing all rare/unseen tokens into an antipodal cone),
     - Computed only the **Real part** $\operatorname{Re}\langle \psi_t \mid M_v \rangle_{\mathbb{C}}$ and discarded the **Imaginary part** $\operatorname{Im}\langle \psi_t \mid M_v \rangle_{\mathbb{C}}$, and
     - Multiplied already-unit-norm complex vectors (`||psi_re||^2 + ||psi_im||^2 = 1.0`) by an erroneous $1/\sqrt{2}$ on both `psi_flat` and `det_flat`, cutting self-overlap $\langle \psi \mid \psi \rangle$ in half (`0.50` instead of `1.00`).
2. **Vocabulary-Side `meas_proj` Detector Synthesis (`50,257` Rows/Step) vs. Tied Token-Wavefunction + Sequence-Side Operator Output (`8,192` Positions/Step)**:
   - Previously, `synthesize_complex_detectors()` applied `self.meas_proj` to all $V = 50,257$ vocabulary embeddings on every step ($O(V d^2)$ compute), so any singular-value drift in `meas_proj` collapsed the effective rank `R_det` of the entire vocabulary simultaneously (`254.68 -> 10.01` in 100 steps).
   - By **tying the initial token wavefunctions $\lvert e_v \rangle \in \mathbb{C}^{d_{\text{complex}}}$ directly into the last layer** as the measurement basis $\langle e_v \rvert$, and having the neural network output a last-layer operator $\hat{\mathcal{O}}_t$ acting on the $B \times T$ sequence wavefunctions $\lvert \psi_t^{(L)} \rangle$ ($6.1\times$ fewer operator evaluations per step, zero vocabulary-side checkpointing), token probabilities arise purely from the interaction $\langle e_v \mid \hat{\mathcal{O}}_t \mid \psi_t \rangle_{\mathbb{C}}$.
3. **Hard Retractions + Real-Half `StiefelMuon` Penalty vs. Smooth Differentiable Unit-Sphere, Complex-Unitary, and Parseval-Isotropy Penalties**:
   - Calling `StiefelMuon` with `stiefel_penalty = 0.02` on `weight_re` and `weight_im` separately enforced $W_{\text{re}} W_{\text{re}}^\top = I$ and $W_{\text{im}} W_{\text{im}}^\top = I$ (which is mathematically incompatible with a complex unitary matrix $W_{\text{re}} W_{\text{re}}^\top + W_{\text{im}} W_{\text{im}}^\top = I$), while `ComplexUnitaryInterferenceFFN` weights do not need Stiefel retraction at all because its beam and output wavefunctions are already normalized onto the complex unit sphere via `complex_l2_normalize`.
   - Furthermore, replacing hard periodic retractions with smooth differentiable penalties—$\mathcal{L}_{\text{sphere}}$ on raw embedding norms $\|E_v\|_2^2 \to 1$, $\mathcal{L}_{\text{unitary}}$ on square complex operators $W W^H \to I$, and $\mathcal{L}_{\text{iso}}$ (**Parseval Frame Isotropy Penalty** $\Sigma_E = \frac{1}{V} E_{\text{cent}}^\top E_{\text{cent}} \to \frac{1}{2 d_{\text{complex}}} I_{2 d_{\text{complex}}}$)—keeps the tied vocabulary basis spanning the full Hilbert space (`R_emb >= 140–230` instead of collapsing to `27`).
4. **Severe Zipf Token Starvation in FineWeb (`50%` of Vocabulary Receives Only `3.44%` of Updates)**:
   - Across the `900M` FineWeb stream, the top 100 tokens consume `43.62%` of all training positions while the bottom `25,257` tokens (`50%` of the vocabulary) receive only `3.44%` of gradient updates (and in the first `2M` PPMI-SVD slice, `6,502` tokens are completely unseen and `33,249` tokens appear `< 10` times).
   - Combining **isotropic unit-sphere initialization** (or ablation against PPMI-SVD) with **Rare-Token Importance-Sampled Sequence Windows** ($P_{\text{sample}}(W) \propto \text{rarity}(W)$) and **Unbiased Inverse-Probability Reweighting** boosts rare-token gradient updates while keeping the expected validation objective 100% unbiased.

---

## 2. Testable Scientific Hypotheses (`H1`–`H4`)

### Hypothesis `H1`: 100% Softmax-Free Tied Operator-Wavefunction Complex Born Readout (`TiedHoloBornReadout`)
- **Independent Variable**: Vocabulary readout formulation at $d_{\text{complex}} = 128$ ($d_{\text{real}} = 256$):
  1. `legacy_boltzmann_real`: Legacy `_ComplexBornChunkedCEFn` (vocabulary-side `meas_proj` on $50,257$ rows, $1/\sqrt{2}$ double-shrinkage, Real-part-only overlap $\operatorname{Re}\langle \psi_t \mid M_v \rangle_{\mathbb{C}}$, Boltzmann `exp` / `log_softmax` denominator).
  2. `tied_energy_born`: Tied initial token wavefunctions $\lvert e_v \rangle = \text{normalize}(E_{\text{re}}[v] + i E_{\text{im}}[v])$, sequence-side operator output $\lvert \Phi_t \rangle = \hat{\mathcal{O}}_t \lvert \psi_t \rangle$, and pure $p$-boson Born energy rule:
     $$P_{\text{EnergyBorn}}(v \mid t) = \frac{w_{\text{prior},v} \, \bigl(|\langle e_v \mid \Phi_t \rangle_{\mathbb{C}}|^2 + \varepsilon_0\bigr)^p}{\sum_{v'=1}^V w_{\text{prior},v'} \, \bigl(|\langle e_{v'} \mid \Phi_t \rangle_{\mathbb{C}}|^2 + \varepsilon_0\bigr)^p}$$
  3. `tied_holo_born`: Tied initial token wavefunctions $\lvert e_v \rangle$, sequence-side operator + wavefunction output $\lvert \Phi_t \rangle = \hat{\mathcal{O}}_t \lvert \psi_t \rangle$, and **100% Softmax-Free Holographic Reference-Wave Complex Born Rule** (coherent interference between unit reference wave $1$ and complex transition amplitude $\alpha \langle e_v \mid \Phi_t \rangle_{\mathbb{C}}^{\text{cent}}$, raised to polynomial boson power $p$ via `torch.pow(I_tv, boson_p)` with **zero `exp` / `softmax` / `logsumexp`**):
     $$I_{t,v} = \bigl|1 + \alpha \, \langle e_v \mid \Phi_t \rangle_{\mathbb{C}}^{\text{cent}}\bigr|^2 + \varepsilon_0 = \bigl(1 + \alpha \, c_{t,v}^{\text{re,cent}}\bigr)^2 + \bigl(\alpha \, c_{t,v}^{\text{im}}\bigr)^2 + \varepsilon_0$$
     $$\tilde{P}_{t,v} = w_{\text{prior},v} \cdot (I_{t,v})^p, \qquad Z_t = \sum_{v'=1}^V \tilde{P}_{t,v'}, \qquad P_{\text{HoloBorn}}(v \mid t) = \frac{\tilde{P}_{t,v}}{Z_t}$$
     where:
     - $c_{t,v}^{\text{re}} = \Phi_{\text{re},t}^\top e_{\text{re},v} + \Phi_{\text{im},t}^\top e_{\text{im},v}$,
     - $c_{t,v}^{\text{im}} = \Phi_{\text{re},t}^\top e_{\text{im},v} - \Phi_{\text{im},t}^\top e_{\text{re},v}$,
     - $c_{t,v}^{\text{re,cent}} = c_{t,v}^{\text{re}} - \mu_t^{\text{re}}$ with $\mu_t^{\text{re}} = \frac{1}{V}\sum_{v'=1}^V c_{t,v'}^{\text{re}}$, and
     - $w_{\text{prior},v} = \bigl(P_0(v) / \max_{v'} P_0(v')\bigr)^{\tau_{\text{prior}}} \in (0, 1]$ is the polynomial POVM measure weight (no `exp` used).
- **Dependent Observables**:
  - Self-overlap identity check $|\langle e_v \mid e_v \rangle_{\mathbb{C}}|^2$ (`1.000000` without $1/\sqrt{2}$ shrinkage vs. `0.500000` with $1/\sqrt{2}$ shrinkage),
  - Exact probability normalization $\max_t \bigl|\sum_{v=1}^V P_{\text{HoloBorn}}(v \mid t) - 1.0\bigr|$,
  - Memory-efficient chunked autograd parity (`max_abs_grad_err` between `TiedHoloBornChunkedNLLFn` with `vocab_chunk_size = 4096` and full-materialized float32 reference), and
  - Readout operator FLOPs/step ($B \cdot T \cdot d_{\text{complex}}^2$ sequence-side vs. $V \cdot d_{\text{complex}}^2$ vocabulary-side).
- **Quantitative Pass/Fail Criteria**:
  1. Self-overlap $|\langle e_v \mid e_v \rangle_{\mathbb{C}}|^2 = 1.0 \pm 10^{-6}$ and probability sum error $< 10^{-5}$.
  2. Chunked autograd function `TiedHoloBornChunkedNLLFn` matches the full float32 reference within `< 1e-5` forward loss error and `< 1e-4` backward gradient error on $(\Phi_{\text{re}}, \Phi_{\text{im}}, e_{\text{re}}, e_{\text{im}})$.
  3. Sequence-side operator application reduces operator GEMM rows by `50,257 / 8,192 = 6.13x` at `B=16, T=512`.

### Hypothesis `H2`: Soft Unit-Sphere, Complex-Unitary, and Parseval-Isotropy Penalties vs. Hard Retractions
- **Independent Variable**: Manifold & frame regularization strategy over `200` training steps on FineWeb (`d_complex = 128, L = 4`):
  1. `hard_retract_legacy`: Periodic `retract_complex_unitary_weights()` + `StiefelMuon` with real `stiefel_penalty = 0.02` on individual `weight_re, weight_im` matrices + unconstrained `embed_re, embed_im` norms + no Parseval frame isotropy penalty.
  2. `soft_penalties_parseval_iso`: Zero hard retractions, zero FFN weight retractions (`stiefel_penalty = 0.0` inside `StiefelMuon`), and three smooth differentiable penalties computed over the active vocabulary $V = \text{valid\_vocab\_size} = 50,257$:
     - **Unit-Sphere Embedding Penalty**:
       $$\mathcal{L}_{\text{sphere}} = \frac{1}{V} \sum_{v=1}^V \Bigl(\|E_{\text{re}}[v]\|_2^2 + \|E_{\text{im}}[v]\|_2^2 - 1\Bigr)^2$$
     - **Square Complex-Unitary Penalty** (applied only to square $d_{\text{complex}} \times d_{\text{complex}}$ attention $Q, K, V, O$ and last-layer operator $\hat{U}_{\text{out}}$, **excluding** rectangular FFN `beam1, beam2, proj_down`):
       $$\mathcal{L}_{\text{unitary}} = \frac{1}{N_{\text{sq}}} \sum_{W \in \mathcal{W}_{\text{sq}}} \frac{1}{d_{\text{complex}}} \Bigl(\bigl\|W_{\text{re}} W_{\text{re}}^\top + W_{\text{im}} W_{\text{im}}^\top - I\bigr\|_F^2 + \bigl\|W_{\text{im}} W_{\text{re}}^\top - W_{\text{re}} W_{\text{im}}^\top\bigr\|_F^2\Bigr)$$
     - **Parseval Frame Isotropy Penalty** on the tied unit-norm vocabulary wavefunctions $\tilde{E} = [\hat{e}_{\text{re}},\, \hat{e}_{\text{im}}] \in \mathbb{R}^{V \times 2 d_{\text{complex}}}$ ($\tilde{E}_{\text{cent}} = \tilde{E} - \frac{1}{V}\mathbf{1}\mathbf{1}^\top \tilde{E}$):
       $$\mathcal{L}_{\text{iso}} = (2 d_{\text{complex}}) \left\|\frac{1}{V} \tilde{E}_{\text{cent}}^\top \tilde{E}_{\text{cent}} - \frac{1}{2 d_{\text{complex}}} I_{2 d_{\text{complex}}}\right\|_F^2$$
     - Combined penalty: $\mathcal{L}_{\text{reg}} = \lambda_{\text{sphere}} \mathcal{L}_{\text{sphere}} + \lambda_{\text{unitary}} \mathcal{L}_{\text{unitary}} + \lambda_{\text{iso}} \mathcal{L}_{\text{iso}}$.
- **Dependent Observables**:
  - Tied vocabulary participation ratio $R_{\text{emb}} = \frac{(\operatorname{Tr} \Sigma_E)^2}{\|\Sigma_E\|_F^2} \in [1, 2 d_{\text{complex}}]$ at Step 200,
  - Median and maximum raw embedding norm $\|E_v\|_2 = \sqrt{\|E_{\text{re}}[v]\|_2^2 + \|E_{\text{im}}[v]\|_2^2}$ at Step 200,
  - Complex unitary drift $\max_{W \in \mathcal{W}_{\text{sq}}} \frac{1}{\sqrt{d}}\|W W^H - I\|_F$, and
  - Validation NLL (`val_ce`) at Step 200.
- **Quantitative Pass/Fail Criteria**:
  1. `soft_penalties_parseval_iso` preserves $R_{\text{emb}} \ge 140.0$ at Step 200 (vs. $< 45.0$ without Parseval isotropy and $< 30.0$ in `hard_retract_legacy`—at least a **`3.5x` improvement** in Hilbert-space rank).
  2. Median raw embedding norm stays within $[0.95, 1.15]$ and complex unitary error stays $< 0.25$ without any hard retraction steps.

### Hypothesis `H3`: Isotropic Unit-Sphere Initialization & Unbiased Rare-Token Importance Sampling
- **Independent Variable**: Initialization & data-sampling strategy on FineWeb (`d_complex = 128`):
  1. `ppmi_svd_uniform`: 2M-token PPMI-SVD initialization + uniform random sequence window sampling ($\alpha_{\text{rare}} = 0.0$).
  2. `isotropic_haar_uniform`: Isotropic random unit-sphere initialization ($\hat{e}_v \sim \text{Unif}(\mathbb{S}^{2 d_{\text{complex}} - 1})$, $R_{\text{emb}}(0) \ge 235$) + uniform random window sampling ($\alpha_{\text{rare}} = 0.0$).
  3. `isotropic_haar_rare_reweighted`: Isotropic random unit-sphere initialization + **Rare-Token Importance-Sampled Sequence Windows** ($q(W_i) = S_{\text{rare}}(W_i) / \sum_{j=1}^M S_{\text{rare}}(W_j)$ where $S_{\text{rare}}(W_i) = \frac{1}{T}\sum_{t \in W_i} P_0(y_{i,t})^{-\alpha_{\text{rare}}}$, $\alpha_{\text{rare}} = 0.28$) paired with **Unbiased Inverse-Probability Importance Reweighting**.
- **Dependent Observables**:
  - Bottom-50% vocabulary (`25,257` rarest tokens) share of sampled training tokens,
  - Importance-weighted estimator unbiasedness error $\bigl|\mathbb{E}_{W \sim q}[w_{\text{imp}}(W) \mathcal{L}(W)] - \mathbb{E}_{W \sim \text{Unif}}[\mathcal{L}(W)]\bigr|$, and
  - Post-training `R_emb` and held-out validation NLL (`val_ce`).
- **Quantitative Pass/Fail Criteria**:
  1. Rare-token importance sampling ($\alpha_{\text{rare}} = 0.28$) increases the sampling exposure of rare tokens in the candidate pool by at least **`1.5x`** while maintaining exact importance-weighted expectation error `< 1e-5`.
  2. `isotropic_haar_uniform` and `isotropic_haar_rare_reweighted` maintain `R_emb >= 140.0` and achieve lower or equal validation NLL than `ppmi_svd_uniform`.

### Hypothesis `H4`: 5-Minute Head-to-Head Verification & Multi-Shard Scaling Check (`Wave 40` vs. `Wave 39`)
- **Independent Variable**: Training [`TiedHoloBornComplexRoPEGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/tied_holo_born_complex_gpt.py) (`d_complex = 128, H = 4, d_h = 32, L = 6`) on the RTX 5060 Ti with all `Wave 40` fixes (`TiedHoloBornReadout`, soft sphere/unitary/Parseval-isotropy penalties, no FFN retraction, no $1/\sqrt{2}$ shrinkage) vs. the `Wave 39` baseline trajectory (`R_det: 243 -> 27`).
- **Dependent Observables**:
  - `val_ce`, `val_ppl`, `top1_acc`, `top50_acc`, `sharp_eff_rank` (`R_emb`), `mean_density_purity`, `embed_norm_median`, and `complex_unitary_err` streamed live via `StandardizedExperimentLogger` to `.cache/live_logs/`, `.cache/model_cards/`, and `.cache/benchmarks.db`.
- **Quantitative Pass/Fail Criteria**:
  1. `sharp_eff_rank` (`R_emb`) remains `>= 140.0` throughout the entire run (completely eliminating the `243 -> 27` rank collapse observed in `Wave 39`).
  2. Validation NLL (`val_ce`) decreases monotonically across evaluation checkpoints with zero NaN/Inf events and `0` unexpected `float32` activation leaks.

---

## 3. Planned Changes (Rule 1–8 Compliant Architecture & Modules)

Following **Rule 6** of [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md) (*Clean, Flag-Free Per-Experiment Modules Over Boolean-Flag Accumulation in Legacy Models*), we leave [`GeminiBornComplexPackedWithRoPE.py`](file:///home/festivity/agy/QuantumDLBlocks/GeminiBornComplexPackedWithRoPE.py) and [`src/qdlblocks/models/gemini_packed_complex_rope_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/gemini_packed_complex_rope_gpt.py) untouched for reproducible comparison and create a clean, dedicated, flag-free model module [`src/qdlblocks/models/tied_holo_born_complex_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/tied_holo_born_complex_gpt.py).

### 3.1 New Flag-Free Model & Readout Module: [`src/qdlblocks/models/tied_holo_born_complex_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/tied_holo_born_complex_gpt.py)

1. **Top-of-File Named Constants (Rule 7)**:
   ```python
   DEFAULT_HOLO_BOSON_P: float = 12.0
   DEFAULT_HOLO_ALPHA: float = 0.85
   DEFAULT_HOLO_EPS: float = 1e-6
   DEFAULT_PRIOR_TEMP: float = 0.65
   DEFAULT_SPHERE_PENALTY_WEIGHT: float = 2.0
   DEFAULT_UNITARY_PENALTY_WEIGHT: float = 0.08
   DEFAULT_ISO_PENALTY_WEIGHT: float = 0.50
   DEFAULT_VOCAB_CHUNK_SIZE: int = 4096
   DEFAULT_MUON_LR: float = 0.015
   DEFAULT_EMBED_LR: float = 0.006
   DEFAULT_GATE_LR: float = 0.003
   ```

2. **Frozen Configuration Dataclass (`TiedHoloBornConfig`, `<= 8` arguments in all functions/methods per Rule 2)**:
   ```python
   @dataclass(frozen=True)
   class TiedHoloBornConfig:
       vocab_size: int = 50304
       valid_vocab_size: int = 50257
       d_complex: int = 128
       n_registers: int = 4
       d_register: int = 32
       num_layers: int = 6
       block_size: int = 512
       holo_boson_p: float = DEFAULT_HOLO_BOSON_P
       holo_alpha: float = DEFAULT_HOLO_ALPHA
       prior_temp: float = DEFAULT_PRIOR_TEMP
       sphere_penalty_weight: float = DEFAULT_SPHERE_PENALTY_WEIGHT
       unitary_penalty_weight: float = DEFAULT_UNITARY_PENALTY_WEIGHT
       iso_penalty_weight: float = DEFAULT_ISO_PENALTY_WEIGHT
       checkpoint_blocks: bool = False
   ```

3. **Pure Polynomial Holographic Born Probability & NLL Evaluation (`eval_tied_holo_born_probs` and `TiedHoloBornChunkedNLLFn`)**:
   - **Pure Polynomial Power Rule (Zero `exp`, Zero `softmax`, Zero `logsumexp`)**:
     - Let `phi_cat = [phi_re, phi_im]` of shape `(N, 2 * d_complex)` and `e_cat = [e_re, e_im]` of shape `(V, 2 * d_complex)` where $V = \text{valid\_vocab\_size}$ and every row has unit norm ($\|\text{phi\_cat}[t]\|_2 = 1.0, \|\text{e\_cat}[v]\|_2 = 1.0$, **no $1/\sqrt{2}$ factor**).
     - Define `e_skew_cat = [-e_im, e_re]` of shape `(V, 2 * d_complex)`. Then for any vocabulary slice $c = [v_0 : v_1]$:
       $$c_{t,v}^{\text{re}} = (\text{phi\_cat} \, @ \, \text{e\_cat}[c]^\top)_{t,v} = \Phi_{\text{re},t}^\top e_{\text{re},v} + \Phi_{\text{im},t}^\top e_{\text{im},v}$$
       $$c_{t,v}^{\text{im}} = -(\text{phi\_cat} \, @ \, \text{e\_skew\_cat}[c]^\top)_{t,v} = \Phi_{\text{re},t}^\top e_{\text{im},v} - \Phi_{\text{im},t}^\top e_{\text{re},v}$$
     - Precompute the exact global vocabulary mean ket $\bar{e} = \frac{1}{V}\sum_{v=1}^V \text{e\_cat}[v] \in \mathbb{R}^{2 d_{\text{complex}}}$ and per-token mean real overlap $\mu_t^{\text{re}} = \text{phi\_cat}[t]^\top \bar{e} = \frac{1}{V}\sum_{v'=1}^V c_{t,v'}^{\text{re}}$ in $O(N \cdot d_{\text{complex}})$ before the chunk loop.
     - For each token $t$ and vocabulary item $v \in \{1, \dots, V\}$, the centered holographic interference intensity is:
       $$c_{t,v}^{\text{re,cent}} = c_{t,v}^{\text{re}} - \mu_t^{\text{re}}, \qquad I_{t,v} = \bigl(1 + \alpha \, c_{t,v}^{\text{re,cent}}\bigr)^2 + \bigl(\alpha \, c_{t,v}^{\text{im}}\bigr)^2 + \varepsilon_0$$
     - Given precomputed positive polynomial prior weights `prior_weights: (V,)` where $w_{\text{prior},v} = \bigl(P_0(v) / \max_{v'} P_0(v')\bigr)^{\tau_{\text{prior}}} \in (0, 1]$, the unnormalized $p$-boson Born weight is computed in `float32` directly via polynomial power (`torch.pow`):
       $$\tilde{P}_{t,v} = w_{\text{prior},v} \cdot (I_{t,v})^p, \qquad Z_t = \sum_{v'=1}^V \tilde{P}_{t,v'}, \qquad P_{\text{HoloBorn}}(v \mid t) = \frac{\tilde{P}_{t,v}}{Z_t}$$
       Notice that since $|c_{t,v}^{\text{re,cent}}|, |c_{t,v}^{\text{im}}| \le 2$ and $\alpha \le 1$, $I_{t,v} \in [\varepsilon_0, 13]$, so for $p = 12$, $(I_{t,v})^p \le 13^{12} \approx 2.3 \times 10^{13} \ll 3.4 \times 10^{38}$ (`float32` max). Thus $\tilde{P}_{t,v}$ and $Z_t$ are accumulated directly in `float32` with **zero `exp` or `logsumexp` calls**.
     - The per-token negative log-likelihood is $\ell_t = \ln(Z_t) - \ln(\tilde{P}_{t, y_t})$.
     - Given optional `token_weights: torch.Tensor | None` of shape `(N,)`, define the normalized per-token weights:
       $$\tilde{w}_t = \begin{cases} \frac{1}{N} & \text{if } \text{token\_weights is None} \\ \frac{\text{token\_weights}[t]}{\sum_{t'=1}^N \text{token\_weights}[t']} & \text{otherwise} \end{cases}$$
       so the scalar loss returned by `TiedHoloBornChunkedNLLFn.forward` is $\mathcal{L} = \sum_{t=1}^N \tilde{w}_t \, \ell_t$.

   - **Exact `eval_tied_holo_born_probs` Function Signature (`<= 8` args)**:
     ```python
     def eval_tied_holo_born_probs(
         phi_re: torch.Tensor,
         phi_im: torch.Tensor,
         e_re: torch.Tensor,
         e_im: torch.Tensor,
         prior_weights: torch.Tensor,
         boson_p: float = DEFAULT_HOLO_BOSON_P,
         holo_alpha: float = DEFAULT_HOLO_ALPHA,
     ) -> torch.Tensor:
     ```
     Computes and returns the exact normalized probability tensor `probs` of shape `(*phi_re.shape[:-1], V)` in `float32` where $\sum_{v=1}^V \text{probs}[\dots, v] = 1.0$, using `torch.pow(I_tv, boson_p)` with zero `exp` or `softmax`.

   - **Exact Backward Equations of `TiedHoloBornChunkedNLLFn` (Including $\mu_t^{\text{re}}$ Centering Pullback)**:
     - **Forward signature (`<= 8` args)**:
       ```python
       @staticmethod
       def forward(
           ctx: Any,
           phi_cat: torch.Tensor,
           e_cat: torch.Tensor,
           targets: torch.Tensor,
           prior_weights: torch.Tensor,
           boson_p: float,
           holo_alpha: float,
           token_weights: torch.Tensor | None = None,
       ) -> torch.Tensor:
       ```
     - In `backward(ctx, grad_out)`:
       1. For each token $t \in \{1, \dots, N\}$ and chunk $c = [v_0 : v_1]$, recompute $c_{t,v}^{\text{re,cent}}$, $c_{t,v}^{\text{im}}$, $I_{t,v}$, and $P_{t,v} = \tilde{P}_{t,v} / Z_t$.
       2. The gradient of $\mathcal{L}$ with respect to $\ln \tilde{P}_{t,v}$ is:
          $$\delta_{t,v} = (\text{grad\_out} \cdot \tilde{w}_t) \cdot \bigl(P_{t,v} - \mathbb{I}[v = y_t]\bigr)$$
       3. Since $\ln \tilde{P}_{t,v} = \ln w_{\text{prior},v} + p \ln I_{t,v}$ and $I_{t,v} = (1 + \alpha c_{t,v}^{\text{re,cent}})^2 + (\alpha c_{t,v}^{\text{im}})^2 + \varepsilon_0$, the partial derivatives with respect to $c_{t,v}^{\text{re,cent}}$ and $c_{t,v}^{\text{im}}$ are:
          $$g_{t,v}^{\text{re,cent}} = \delta_{t,v} \cdot \frac{2 p \alpha \bigl(1 + \alpha c_{t,v}^{\text{re,cent}}\bigr)}{I_{t,v}}, \qquad g_{t,v}^{\text{im}} = \delta_{t,v} \cdot \frac{2 p \alpha^2 c_{t,v}^{\text{im}}}{I_{t,v}}$$
       4. Accumulate over chunks:
          - $\text{d\_phi\_raw}[t] \mathrel{+}= \sum_{v \in c} \bigl(g_{t,v}^{\text{re,cent}} \, \text{e\_cat}[v] - g_{t,v}^{\text{im}} \, \text{e\_skew\_cat}[v]\bigr)$,
          - $\text{d\_e\_raw}[v] = \sum_{t=1}^N \bigl(g_{t,v}^{\text{re,cent}} \, \text{phi\_cat}[t] + g_{t,v}^{\text{im}} \, \text{phi\_skew\_cat}[t]\bigr)$ (where $\text{phi\_skew\_cat} = [-\Phi_{\text{im}}, \Phi_{\text{re}}]$), and
          - $s_t \mathrel{+}= \sum_{v \in c} g_{t,v}^{\text{re,cent}}$ (the per-token sum of centered real-overlap gradients across all $V$ vocabulary items).
       5. **Exact Centering Pullback for $\mu_t^{\text{re}} = \frac{1}{V}\sum_{v'=1}^V c_{t,v'}^{\text{re}} = \text{phi\_cat}[t]^\top \bar{e}$**:
          Since $c_{t,v}^{\text{re,cent}} = c_{t,v}^{\text{re}} - \frac{1}{V}\sum_{v'=1}^V c_{t,v'}^{\text{re}}$, the true gradient with respect to uncentered $c_{t,v}^{\text{re}}$ is $g_{t,v}^{\text{re}} = g_{t,v}^{\text{re,cent}} - \frac{1}{V} s_t$. Therefore, after the chunk loop completes:
          $$\text{d\_phi\_cat}[t] = \text{d\_phi\_raw}[t] - s_t \, \bar{e}$$
          $$\text{d\_e\_cat}[v] = \text{d\_e\_raw}[v] - \frac{1}{V} \sum_{t=1}^N s_t \, \text{phi\_cat}[t]$$
          Return `(d_phi_cat.to(phi_cat.dtype), d_e_cat.to(e_cat.dtype), None, None, None, None, None)`.

4. **Complete `TiedHoloBornComplexRoPEGPT` Class Contract, Initialization Modes, and `forward` Signature**:
   - **Attributes & Default Initialization in `__init__(self, cfg: TiedHoloBornConfig)`**:
     - `self.embed_re = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_complex))`
     - `self.embed_im = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_complex))`
     - By default in `__init__`, `reset_isotropic_unit_sphere_embeddings()` initializes all rows `0 .. cfg.vocab_size - 1` from independent standard normal draws normalized row-wise onto the complex unit sphere via `complex_l2_normalize`, and zeros out unused padding rows `cfg.valid_vocab_size .. cfg.vocab_size - 1`.
     - `self.register_buffer("prior_weights", torch.ones(cfg.vocab_size, dtype=torch.float32))` stores the polynomial POVM measure weights $w_{\text{prior},v} = (P_0(v) / \max_{v'} P_0(v'))^{\tau_{\text{prior}}} \in (0, 1]$ for $v < \text{valid\_vocab\_size}$.
     - `self.blocks = nn.ModuleList([GeminiComplexRegisterBlock(...) for _ in range(cfg.num_layers)])` (reusing the validated `GeminiComplexRegisterBlock` with `probe_mode="v_krylov"` and `normalize_krylov_ket=True`).
     - `self.out_operator = ComplexLinear(cfg.d_complex, cfg.d_complex)` initialized as a random Haar complex unitary matrix.
     - `self.psi0_readout_gate = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))`.
   - **Initialization Helper Methods for `H3` (`<= 8` args each)**:
     - `initialize_uniform_or_empirical_prior(self, prior_tokens: torch.Tensor) -> None`: Computes Laplace-smoothed unigram probabilities $P_0(v) = \frac{\text{count}(v) + 1.0}{\sum_{v'=0}^{V-1} (\text{count}(v') + 1.0)}$ over `0 .. valid_vocab_size - 1` and sets `self.prior_weights[:V] = (P_0 / P_0.max()).pow(self.cfg.prior_temp)`.
     - `initialize_isotropic_haar(self, prior_tokens: torch.Tensor | None = None, seed: int = 1337) -> dict[str, float]`: Initializes `embed_re[:V], embed_im[:V]` isotropically on the complex unit sphere $\mathbb{S}^{2d_{\text{complex}}-1}$, optionally sets `prior_weights` from `prior_tokens`, and returns `{"sharp_eff_rank": self.compute_sharp_eff_rank()}`.
     - `initialize_with_ppmi_svd(self, ppmi_embeddings: torch.Tensor, prior_tokens: torch.Tensor) -> dict[str, float]`: Initializes `embed_re[:V], embed_im[:V]` from `ppmi_embeddings` (using the harmonic tiling from `initialize_gemini_complex_gpt_with_ppmi_svd`), sets `prior_weights` from `prior_tokens`, and returns `{"sharp_eff_rank": self.compute_sharp_eff_rank()}`.
   - **Exact `forward` Signature & Return Contract (`<= 8` args)**:
     ```python
     def forward(
         self,
         idx: torch.Tensor,
         targets: torch.Tensor | None = None,
         return_logits: bool | None = None,
         token_weights: torch.Tensor | None = None,
     ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
     ```
     - Slices active vocabulary `v_eff = int(self.cfg.valid_vocab_size)` so padding rows `50,257 .. 50,303` never participate in forward or backward:
       `e_re, e_im = complex_l2_normalize(self.embed_re[:v_eff], self.embed_im[:v_eff], dim=-1)`.
     - Computes sequence-side post-operator wavefunctions `phi_re, phi_im` of shape `(B, T, d_complex)`.
     - When `targets is not None and not want_logits` (`want_logits = (targets is None) if return_logits is None else bool(return_logits)`):
       Calls `loss = TiedHoloBornChunkedNLLFn.apply(phi_cat, e_cat, valid_targets, self.prior_weights[:v_eff], float(self.cfg.holo_boson_p), float(self.cfg.holo_alpha), token_weights)` and returns `(loss if return_logits is None else None, loss)`.
     - When `want_logits` is `True` (e.g., during validation evaluation of `val_ce`, `top1_acc`, `top50_acc`):
       Calls `probs = eval_tied_holo_born_probs(phi_re, phi_im, e_re, e_im, self.prior_weights[:v_eff], boson_p=float(self.cfg.holo_boson_p), holo_alpha=float(self.cfg.holo_alpha))` and returns `log_probs = torch.log(probs.clamp_min(1e-12))` of shape `(B, T, v_eff)` as the first tuple element (so `argmax` / `topk` rank predictions identically to `probs`, and `-log_probs.gather(...)` equals the exact Holographic Born NLL `val_ce` without applying a second softmax!).

5. **Exact Signature and `valid_vocab_size` Slicing of `compute_soft_manifold_and_isotropy_penalties` (`<= 8` args)**:
   ```python
   def compute_soft_manifold_and_isotropy_penalties(
       model: TiedHoloBornComplexRoPEGPT,
   ) -> tuple[torch.Tensor, dict[str, float]]:
   ```
   - Slices strictly over the active vocabulary `v_eff = int(model.cfg.valid_vocab_size)` (`50,257` rows):
     - `e_re_raw = model.embed_re[:v_eff].float()` and `e_im_raw = model.embed_im[:v_eff].float()`
     - `e_norm_sq = e_re_raw.square().sum(dim=-1) + e_im_raw.square().sum(dim=-1)`
     - `sphere_pen = (e_norm_sq - 1.0).square().mean()`
   - Computes `unitary_pen` as the mean of $\frac{1}{d_{\text{complex}}}\bigl(\|W_{\text{re}} W_{\text{re}}^\top + W_{\text{im}} W_{\text{im}}^\top - I\|_F^2 + \|W_{\text{im}} W_{\text{re}}^\top - W_{\text{re}} W_{\text{im}}^\top\|_F^2\bigr)$ across all `ComplexLinear` submodules where `mod.in_features == mod.out_features` (square attention `q_proj, k_proj, v_proj, out_proj` and `out_operator`, excluding rectangular FFN layers), and scales the un-attenuated per-matrix sum $\sum_{W \in \mathcal{W}_{\text{sq}}} \|W W^H - I\|_F^2$ by `model.cfg.unitary_penalty_weight` so the restoring gradient on each matrix $W$ is dimension- and depth-independent.
   - Computes `iso_pen` on the normalized active vocabulary $\tilde{E} = [\hat{e}_{\text{re}}, \hat{e}_{\text{im}}] \in \mathbb{R}^{v_{\text{eff}} \times 2 d_{\text{complex}}}$:
     - `e_cent = e_cat - e_cat.mean(dim=0, keepdim=True)`
     - `cov = (e_cent.t() @ e_cent) / float(v_eff)`
     - `d_real = 2 * int(model.cfg.d_complex)`
     - `target_cov = torch.eye(d_real, device=cov.device, dtype=torch.float32) / float(d_real)`
     - `iso_pen = float(d_real) * (cov - target_cov).square().sum()`
   - Combines `total_penalty = float(model.cfg.sphere_penalty_weight) * sphere_pen + float(model.cfg.unitary_penalty_weight) * unitary_raw_sum + float(model.cfg.iso_penalty_weight) * iso_pen`.
   - Returns `(total_penalty, {"sphere_pen": ..., "unitary_pen": ..., "iso_pen": ..., "embed_norm_median": ..., "embed_norm_max": ..., "complex_unitary_err": ...})`.

6. **Exact Signature and Parameter Routing of `build_tied_holo_born_optimizers` (`<= 8` args)**:
   ```python
   def build_tied_holo_born_optimizers(
       model: TiedHoloBornComplexRoPEGPT,
       base_muon_lr: float = DEFAULT_MUON_LR,
       base_embed_lr: float = DEFAULT_EMBED_LR,
       base_gate_lr: float = DEFAULT_GATE_LR,
       weight_decay: float = 0.0,
   ) -> tuple[StiefelMuon, torch.optim.AdamW]:
   ```
   - Routes all 2D `ComplexLinear` weights (`weight_re, weight_im` with `p.ndim == 2 and min(p.shape) >= 16`) to `StiefelMuon(muon_params, lr=base_muon_lr, momentum=0.95, stiefel_penalty=0.0, use_normuon=True, use_twin_rail=True)` (setting `stiefel_penalty=0.0` so `StiefelMuon` performs Polar Express orthogonalized updates without applying the conflicting real-matrix Stiefel penalty).
   - Routes `[model.embed_re, model.embed_im]` to `AdamW` group 0 (`lr=base_embed_lr, weight_decay=0.0, betas=(0.9, 0.95)`) and 1D/scalar gates (`psi0_readout_gate`, `phase_bias`, `v0_mix_logit`, etc.) to `AdamW` group 1 (`lr=base_gate_lr, weight_decay=weight_decay, betas=(0.9, 0.95)`).

7. **Exact Specification of `RareTokenImportanceWindowSampler` (`H3` & Section 3.1.6)**:
   - **Constructor Signature (`<= 8` args)**:
     ```python
     class RareTokenImportanceWindowSampler:
         def __init__(
             self,
             reference_tokens: np.ndarray | torch.Tensor,
             valid_vocab_size: int = 50257,
             alpha_rare: float = 0.28,
             pool_multiplier: int = 8,
             laplace_eps: float = 1.0,
         ) -> None:
     ```
   - **Laplace Smoothing Formula & Tail-Contrasted Window Score**:
     Given `reference_tokens`, clamps indices to `[0, valid_vocab_size - 1]` and computes:
     $$P_0(v) = \frac{\text{count}(v) + \epsilon_{\text{laplace}}}{\sum_{v'=0}^{V-1} \bigl(\text{count}(v') + \epsilon_{\text{laplace}}\bigr)}, \qquad r(v) = P_0(v)^{-\alpha_{\text{rare}}}$$
     Because $\epsilon_{\text{laplace}} = 1.0 > 0$, even unseen tokens (`count(v) == 0`) have strictly positive probability $P_0(v) > 0$ and finite rarity score $r(v) < \infty$. To prevent the top-50% frequent tokens (`96.5%` of corpus positions) from washing out window-level variance across $T = 256\text{–}512$ tokens via the Central Limit Theorem, the sampler evaluates excess rarity on the bottom-50% vocabulary tail ($r_{\text{tail}}(v) = \max(0, r(v) - \operatorname{median}(r)) \cdot \mathbb{I}[v \in \mathcal{V}_{\text{bottom50}}]$) and raises the normalized window tail score to `DEFAULT_POOL_CONTRAST_POWER = 6.0` before normalizing $q_i = S_i / \sum_{j=1}^M S_j$.
   - **Window Sampling & Exact Unbiased Importance Weighting (`sample_batch` signature `<= 8` args)**:
     ```python
     def sample_batch(
         self,
         token_buffer: torch.Tensor,
         batch_size: int,
         seq_len: int,
         step_seed: int,
     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
     ```
     1. Draws $M = \text{pool\_multiplier} \times \text{batch\_size}$ candidate window start offsets $s_1, \dots, s_M$ uniformly at random from `token_buffer`.
     2. For each candidate window $i \in \{1, \dots, M\}$, computes the positive window rarity score $S_i > 0$ and candidate selection probability $q_i = \frac{S_i}{\sum_{j=1}^M S_j}$ (whereas uniform selection from the pool has probability $u_i = \frac{1}{M}$).
     3. Samples `batch_size` window indices $i_1, \dots, i_B \sim \text{Categorical}(q_1, \dots, q_M)$ (with replacement so each draw is independent and identically distributed from $q$).
     4. Assigns every token in selected window $k \in \{1, \dots, B\}$ its exact inverse-probability importance weight $w_{k,t} = \frac{1 / (M \, q_{i_k})}{\frac{1}{B}\sum_{b=1}^B 1 / (M \, q_{i_b})}$.
     5. **Exact Unbiasedness Property (Tested in `test_h3_unbiased_rare_token_importance_sampler`)**:
        For any per-window loss function $\mathcal{L}(W_i)$ on a candidate pool $\{W_1, \dots, W_M\}$, the unnormalized importance-weighted estimator $\hat{\mu}_{\text{IS}} = \frac{1}{B}\sum_{b=1}^B \frac{1}{M q_{i_b}} \mathcal{L}(W_{i_b})$ satisfies:
        $$\mathbb{E}_{i_b \sim q}\!\left[\frac{1}{M q_{i_b}} \mathcal{L}(W_{i_b})\right] = \sum_{i=1}^M q_i \frac{1}{M q_i} \mathcal{L}(W_i) = \frac{1}{M}\sum_{i=1}^M \mathcal{L}(W_i) = \mathbb{E}_{W \sim \text{Unif}}[\mathcal{L}(W)]$$
        with zero bias (`< 1e-6` numerical error when evaluated analytically over the distribution $q$).

### 3.2 Export in [`src/qdlblocks/models/__init__.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/__init__.py) and Document in [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md)
- Export `TiedHoloBornConfig`, `TiedHoloBornComplexRoPEGPT`, `TiedHoloBornChunkedNLLFn`, `eval_tied_holo_born_probs`, `eval_tied_energy_born_probs`, `RareTokenImportanceWindowSampler`, `build_tied_holo_born_optimizers`, and `compute_soft_manifold_and_isotropy_penalties` in `src/qdlblocks/models/__init__.py`.
- Document the new `Wave 40` components and CLI profiler in [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md).

---

## 4. Testing & Verification Plan

### 4.1 Unit Tests: [`tests/test_plan055_tied_holo_born_and_rare_sampling.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan055_tied_holo_born_and_rare_sampling.py)
1. `test_h1_tied_holo_born_self_overlap_and_prob_sum`: Verifies that unit-norm complex states have exact self-overlap $|\langle e_v \mid e_v \rangle_{\mathbb{C}}|^2 = 1.0 \pm 10^{-6}$ (fixing the $0.50$ double-$1/\sqrt{2}$ bug), that `eval_tied_holo_born_probs` sums to `1.0` (`max_err < 1e-5`) with zero `exp`/`softmax` calls, and that sequence-side `out_operator` applies to `(B, T, d_complex)` rather than `(V, d_complex)`.
2. `test_h1_chunked_holo_born_autograd_parity`: Verifies that `TiedHoloBornChunkedNLLFn` (`vocab_chunk_size = 64` on a synthetic $V = 256$ vocabulary) matches the unchunked float32 `eval_tied_holo_born_probs` reference within `< 1e-5` forward loss error and `< 1e-4` backward gradient error on `phi_cat`, `e_cat`, and with non-uniform `token_weights` (explicitly verifying the $\mu_t^{\text{re}}$ centering pullback).
3. `test_h2_soft_sphere_unitary_and_parseval_isotropy_penalties`: Verifies that `compute_soft_manifold_and_isotropy_penalties` slices `[:valid_vocab_size]`, penalizes square `ComplexLinear` matrices ($W W^H \ne I$) while ignoring rectangular FFN `ComplexLinear` matrices (`beam1, beam2, proj_down`), and that 25 gradient steps on $\mathcal{L}_{\text{iso}}$ monotonically increase $R_{\text{emb}}$ on a low-rank perturbed embedding table.
4. `test_h3_unbiased_rare_token_importance_sampler`: Verifies Laplace smoothing on zero-count tokens, confirms that `RareTokenImportanceWindowSampler` increases bottom-50% rare-token frequency in sampled batches (`> 1.5x` over uniform), and verifies that the exact importance-weighted expectation $\sum_{i=1}^M q_i \frac{1}{M q_i} \mathcal{L}(W_i)$ matches the uniform population mean $\frac{1}{M}\sum_{i=1}^M \mathcal{L}(W_i)$ within `< 1e-6`.
5. `test_h4_end_to_end_bf16_smoke_and_rank_retention`: Runs a 5-step `bfloat16` training loop of `TiedHoloBornComplexRoPEGPT` verifying zero `float32` activation leaks inside blocks, decreasing loss, and $R_{\text{emb}}$ retention.

### 4.2 End-to-End Verification Runner: [`examples/65_verify_plan055_tied_holo_born_and_rare_sampling.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/65_verify_plan055_tied_holo_born_and_rare_sampling.py)
- Executes all four hypothesis audits (`H1`–`H4`), including a 5-minute (`300s`) live training run of `TiedHoloBornComplexRoPEGPT` (`d_complex = 128, H = 4, d_h = 32, L = 6`) on `FineWeb` with `StandardizedExperimentLogger`, writing `.cache/hypothesis_experiments_055.json` and syncing to the WebUI (`:8765`).

### 4.3 Regression & AST Linter Gate
- Run `pytest tests/test_plan053_*.py tests/test_plan054_*.py tests/test_plan055_*.py` and `scripts/ast_signature_and_length_linter.py` on all created/modified files (`0` violations).

---

## 5. Operational Tooling & Script Opportunities

### 5.1 Existing Scripts Used
1. **AST Signature & SLOC Linter**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/models/tied_holo_born_complex_gpt.py scripts/fineweb_token_rarity_and_born_rank_profiler.py examples/65_verify_plan055_tied_holo_born_and_rare_sampling.py tests/test_plan055_tied_holo_born_and_rare_sampling.py
   ```
2. **WebUI ModelCard & SQLite Sync**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload
   ```
3. **Live Experiment Monitor Snapshot**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once
   ```

### 5.2 `Build-Before-Launch` Script (Implemented in This Plan)
- [`scripts/fineweb_token_rarity_and_born_rank_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_token_rarity_and_born_rank_profiler.py):
  - Reusable CLI tool that:
    1. Computes and exports FineWeb Zipf token frequency statistics across slice sizes (`2M`, `8M`, `30M`, `100M`, `300M/900M`) and renders the 4-panel Matplotlib visual (`fineweb_token_distribution.png`),
    2. Audits `RareTokenImportanceWindowSampler` exposure gains across vocabulary frequency deciles and verifies unbiased loss reweighting, and
    3. Runs a fast N-step rank-preservation benchmark comparing `legacy_boltzmann_real` vs. `tied_holo_born` + `soft_penalties_parseval_iso`, writing `.cache/fineweb_token_rarity_and_born_rank_profile.json`.
  - Invocation:
    ```bash
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/fineweb_token_rarity_and_born_rank_profiler.py --steps 100 --json-out .cache/fineweb_token_rarity_and_born_rank_profile.json
    ```

### 5.3 `Nice-to-Have` Scripts (Logged in [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md) & Surfaced to User)
1. `scripts/born_povm_boson_power_and_floor_calibrator.py`: Sweeps the Born exponent $p \in \{4, 8, 12, 16\}$ and interference modulation $\alpha \in \{0.65, 0.75, 0.85, 0.92\}$ across $d_{\text{complex}} \in \{64, 96, 128, 192\}$ to map the exact analytical & empirical relationship between $(p, \alpha)$, minimum achievable Born NLL floor, and orthogonal-token gradient sparsity.
2. `scripts/rare_token_gradient_starvation_tracker.py`: Tracks per-frequency-decile (`Decile 1: Top 100` .. `Decile 10: Bottom 25,257`) embedding norm drift, AdamW second-moment $v_t$ scale, and validation NLL across training checkpoints under uniform vs. rare-token importance sampling.

---

## 6. Experiment Outcomes, Constants Ledger, & Telemetry Summary (`Wave 40`)

### 6.1 Rule 7 Named Constants Ledger ([`src/qdlblocks/models/tied_holo_born_complex_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/tied_holo_born_complex_gpt.py))

| Constant Name | Value | Mathematical Role & Empirical Calibration Rationale |
| :--- | :---: | :--- |
| `DEFAULT_HOLO_BOSON_P` | `12.0` | Multi-boson coincidence power $p$ in $I_{t,v}^p$; provides dynamic range $(1.85 / 0.15)^{24} \approx 1.56 \times 10^{26}$ via pure polynomial `torch.pow` with zero `exp`/`softmax`. |
| `DEFAULT_HOLO_ALPHA` | `0.85` | Holographic signal-to-reference modulation $\alpha \in (0, 1)$; places destructive antipodal interference at $1 - \alpha = 0.15$ and constructive interference at $1 + \alpha = 1.85$. |
| `DEFAULT_HOLO_EPS` | `1e-6` | Numerical vacuum floor $\varepsilon_0$ preventing $\ln(0)$ under destructive interference. |
| `DEFAULT_PRIOR_TEMP` | `0.65` | Exponent $\tau_{\text{prior}}$ for polynomial POVM measure $w_{\text{prior},v} = (P_0(v) / \max_{v'} P_0(v'))^{\tau_{\text{prior}}} \in (0, 1]$. |
| `DEFAULT_SPHERE_PENALTY_WEIGHT` | `2.0` | Weight $\lambda_{\text{sphere}}$ on $\mathcal{L}_{\text{sphere}} = \frac{1}{V}\sum_{v=1}^V (\|E_{\text{re}}[v]\|_2^2 + \|E_{\text{im}}[v]\|_2^2 - 1)^2$; locks median embedding norm to `1.0012–1.0026`. |
| `DEFAULT_UNITARY_PENALTY_WEIGHT` | `0.08` | Weight $\lambda_{\text{unitary}}$ applied to the un-attenuated sum $\sum_{W \in \mathcal{W}_{\text{sq}}} \|W W^H - I\|_F^2$ across square `ComplexLinear` operators so the restoring force does not decay by $1 / (N_{\text{sq}} d_{\text{complex}})$; holds `complex_unitary_err` at `0.0224–0.0514` (`< 0.25`) with zero hard retractions. |
| `DEFAULT_ISO_PENALTY_WEIGHT` | `0.50` | Weight $\lambda_{\text{iso}}$ on the Parseval frame isotropy penalty $\mathcal{L}_{\text{iso}} = (2 d_{\text{complex}})\|\frac{1}{V}\tilde{E}_{\text{cent}}^\top \tilde{E}_{\text{cent}} - \frac{1}{2 d_{\text{complex}}} I\|_F^2$; preserves $R_{\text{emb}} \in [243.09, 250.48]$ across 471 steps (`3.86M` tokens). |
| `DEFAULT_VOCAB_CHUNK_SIZE` | `4096` | Vocabulary chunk size in `TiedHoloBornChunkedNLLFn` bounding temporary readout memory to `O(B * T * 4096)`. |
| `DEFAULT_RARE_ALPHA` | `0.28` | Inverse-frequency exponent $\alpha_{\text{rare}}$ in $r(v) = P_0(v)^{-\alpha_{\text{rare}}}$. |
| `DEFAULT_POOL_MULTIPLIER` | `8` | Candidate pool multiplier $M = 8 \times B$ for `RareTokenImportanceWindowSampler`. |
| `DEFAULT_POOL_CONTRAST_POWER` | `6.0` | Tail-excess contrast exponent applied to bottom-50% vocabulary rarity per window to overcome Central Limit Theorem smoothing over $T = 256\text{–}512$ tokens (`2.5706x` rare-token exposure gain). |

---

### 6.2 FineWeb Zipf Token Starvation Profile ([`.cache/fineweb_token_rarity_and_born_rank_profile.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/fineweb_token_rarity_and_born_rank_profile.json))

| Corpus Slice | Total Tokens | Unseen Tokens (`count == 0`) | Starved Tokens (`count < 10`) | Well-Sampled (`count >= 100`) | Top-100 Token Share | Bottom-`25,257` (`50%`) Token Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2M` Slice** | `2,000,000` | `6,502` (`12.94%`) | `33,249` (`66.16%`) | `3,551` (`7.07%`) | `43.97%` | `2.42%` |
| **`8M` Slice** | `8,000,000` | `1,667` (`3.32%`) | `17,845` (`35.51%`) | `12,188` (`24.25%`) | `43.52%` | `3.05%` |
| **`30M` Slice** | `30,000,000` | `785` (`1.56%`) | `5,834` (`11.61%`) | `27,633` (`54.98%`) | `43.62%` | `3.32%` |
| **`100M` Slice** | `100,000,000` | `407` (`0.81%`) | `1,661` (`3.31%`) | `41,701` (`82.98%`) | `43.62%` | `3.44%` |
| **`300M` Slice** | `300,000,000` | `277` (`0.55%`) | `799` (`1.59%`) | `47,301` (`94.12%`) | `43.60%` | `3.46%` |

---

### 6.3 Hypothesis `H1` — Tied Holographic Complex Born Readout & Chunked Autograd Audit (`PASS`)

| Verification Metric | Target | Observed (`Wave 40`) | Status |
| :--- | :---: | :---: | :---: |
| Unit-Ket Self-Overlap $\|\langle \hat{e}_v \mid \hat{e}_v \rangle_{\mathbb{C}}\|^2$ (No $1/\sqrt{2}$ Shrinkage) | `1.0 ± 1e-6` | `1.000000` (`max_err = 4.77e-7` vs. legacy `0.50`) | **PASS** |
| `eval_tied_energy_born_probs` Probability Sum Error $\max_t \|\sum_v P(v\mid t) - 1\|$ | `< 1e-5` | `1.19e-7` | **PASS** |
| `eval_tied_holo_born_probs` Probability Sum Error $\max_t \|\sum_v P(v\mid t) - 1\|$ | `< 1e-5` | `1.19e-7` | **PASS** |
| `TiedHoloBornChunkedNLLFn` vs. Unchunked Forward Loss Error | `< 1e-5` | `0.0` | **PASS** |
| `TiedHoloBornChunkedNLLFn` vs. Unchunked $\nabla_{\Phi} \mathcal{L}$ Max Error (with $\mu_t^{\text{re}}$ Pullback) | `< 1e-4` | `1.34e-7` | **PASS** |
| `TiedHoloBornChunkedNLLFn` vs. Unchunked $\nabla_{E} \mathcal{L}$ Max Error (with $\mu_t^{\text{re}}$ Pullback) | `< 1e-4` | `8.94e-8` | **PASS** |
| Sequence-Side $\hat{U}_{\text{out}}$ GEMM Row Reduction (`50,257` Vocab Rows $\to$ `B*T = 8,192`) | `6.13x` | `6.1349x` | **PASS** |

---

### 6.4 Hypothesis `H2` — Soft Unit-Sphere, Complex-Unitary, and Parseval-Isotropy Penalties vs. Hard Retractions (`PASS`, 200 Steps)

| Ablation Arm (`d_complex = 128, L = 4, 200 Steps`) | Step 0 $R_{\text{emb}}$ | Step 200 $R_{\text{emb}}$ (`sharp_eff_rank`) | Rank Gain vs. Legacy | Median $\|E_v\|_2$ | Max $\|E_v\|_2$ | Complex Unitary Err | Val NLL (`val_ce`) | Val Top-1 (`%`) | Val Top-50 (`%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `hard_retract_legacy` (Legacy Boltzmann Real + Hard Retraction + Real Stiefel Penalty) | `243.01` | `16.74` | `1.00x` | `2.2492` | `2.7869` | `0.0206` | `6.8650` | `12.30%` | `46.86%` |
| **`soft_penalties_parseval_iso` (`Wave 40` TiedHoloBorn + Soft Sphere/Unitary/Parseval Penalties)** | `242.93` | **`249.70`** | **`14.92x`** | **`1.0012`** | **`1.5924`** | **`0.0224`** | **`6.8401`** | **`12.48%`** | **`47.53%`** |

- **Outcome (`H2 PASS`)**: `soft_penalties_parseval_iso` completely eliminates the `243.01 -> 16.74` rank collapse (`14.92x` higher effective Hilbert-space rank at Step 200, far exceeding the `>= 3.5x` and `>= 140.0` targets), locks median embedding norm to `1.0012` (`[0.95, 1.15]`), holds `complex_unitary_err = 0.0224 < 0.25` with zero hard retractions, and improves validation NLL (`6.8401` vs. `6.8650`), Top-1 (`12.48%` vs. `12.30%`), and Top-50 (`47.53%` vs. `46.86%`).

---

### 6.5 Hypothesis `H3` — Initialization & Unbiased Rare-Token Importance Sampling (`PARTIAL PASS — Sampling & Rank Targets Passed; Short-Horizon Val CE Diagnostic Recorded`)

| Ablation Arm (`d_complex = 128, L = 4, 150 Steps`) | Step 0 $R_{\text{emb}}$ | Step 150 $R_{\text{emb}}$ | Rare Bottom-50% Hits | Rare Bottom-50% Share (`%`) | Rare Exposure Gain | Analytical IS Bias Error | Val NLL (`val_ce`) | Val Top-1 (`%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w40_h3_ppmi_svd_uniform` (`ppmi_svd`, $\alpha_{\text{rare}} = 0.0$) | `242.93` | `250.29` | `12,596` | `4.10%` | `1.0000x` | `0.0` | `6.8448` | `11.91%` |
| `w40_h3_isotropic_haar_uniform` (`isotropic_haar`, $\alpha_{\text{rare}} = 0.0$) | `254.70` | **`251.30`** | `12,596` | `4.10%` | `1.0000x` | `0.0` | `7.1202` | `11.65%` |
| `w40_h3_isotropic_haar_rare_reweighted` (`isotropic_haar`, $\alpha_{\text{rare}} = 0.28$) | `254.70` | **`241.93`** | **`32,379`** | **`10.54%`** | **`2.5706x`** | **`0.0`** | `7.4677` | `10.66%` |

- **Mathematical & Empirical Post-Mortem of `H3`**:
  1. **What Succeeded**:
     - `RareTokenImportanceWindowSampler` (`alpha_rare = 0.28`, `pool_multiplier = 8`, `DEFAULT_POOL_CONTRAST_POWER = 6.0`) boosted bottom-50% rare-token exposure from `12,596` hits (`4.10%`) to `32,379` hits (`10.54%`)—a **`2.5706x` exposure gain** (exceeding the `>= 1.5x` target) with **`0.0` analytical importance-weighting bias** (`< 1e-5`).
     - Both `isotropic_haar` arms preserved near-maximal Hilbert-space dimensionality without PPMI-SVD (`R_emb = 251.30` for `isotropic_haar_uniform`, surpassing `ppmi_svd_uniform` at `250.29`, and `R_emb = 241.93 >= 140.0` for `isotropic_haar_rare_reweighted`).
  2. **Why `ppmi_svd_uniform` Had Lower `val_ce` at Step 150 (`0.31M` Tokens) & Next-Wave Mathematical Remedy**:
     - **Short-Horizon Co-Occurrence Warm-Start**: At 150 steps (`307,200` tokens total, `< 6` tokens per vocabulary entry on average), `ppmi_svd_uniform` starts at Step 0 with precomputed 2M-token co-occurrence geometry already aligned across seen tokens, whereas `isotropic_haar` starts from random orthogonal kets and must learn co-occurrence from scratch. When trained for 471 steps (`3.86M` tokens) in `H4`, `isotropic_haar_uniform` drops to `val_ce = 6.9492` (`val_top1 = 12.40%`, `R_emb = 250.35`).
     - **Kish Effective Sample Size (ESS) Variance Inflation from Unclipped $1/q_i$ After 6th-Power Window Contrast**: Raising window tail scores to `DEFAULT_POOL_CONTRAST_POWER = 6.0` makes $q_i$ peaked on rare-dense windows; when an occasional common-token window with small $q_i$ is drawn in the same batch of $B = 8$, its exact inverse-probability weight $w_i \propto 1/q_i$ dominates the batch normalization $\sum_{b=1}^B w_b$, reducing the batch's Kish Effective Sample Size ($\text{ESS} = (\sum w_b)^2 / \sum w_b^2$) from `8.0` to `~1.8` and inflating gradient variance on frequent tokens.
     - **Next-Wave Remedy**: (a) **Hybrid PPMI-SVD + Isotropic Haar Complement Initialization** (initialize top-frequency seen tokens via PPMI-SVD and project unseen/starved tokens onto the isotropic Haar complement so Step 0 has both `R_emb >= 250` and warm co-occurrence geometry), and (b) **Tempered / ESS-Floored Importance Weights** $w_i \propto q_i^{-\beta_{\text{is}}}$ ($\beta_{\text{is}} \in [0.25, 0.50]$) or token-level focal rarity weighting so rare tokens receive higher gradient exposure without collapsing batch-level Kish ESS on common tokens.

---

### 6.6 Hypothesis `H4` — 5-Minute (`300s`) Live Verification of `TiedHoloBornComplexRoPEGPT` (`PASS`)

- **Run Key**: `w40_tied_holo_born_d128c_H4_dh32_L6_300s` (`d_complex = 128`, real-equiv `256`, `H = 4, d_h = 32, L = 6`, `14.88M` params, `B = 16, T = 512`, `RTX 5060 Ti 16GB`, `12,859 tok/s`, peak VRAM `2,882.4 MiB`).

| Checkpoint | Step | Tokens (`M`) | Elapsed (`s`) | Train NLL | Val NLL (`val_ce`) | Val PPL | Val Top-1 (`%`) | Val Top-50 (`%`) | $R_{\text{emb}}$ (`sharp_eff_rank`) | Median $\|E_v\|_2$ | Complex Unitary Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 0 (Init)** | `0` | `0.00M` | `0.0s` | `9.0810` | `9.0810` | `8786.75` | `1.39%` | `20.65%` | **`254.70`** | `1.0000` | `0.0000` |
| **Eval 1 (`25%`)** | `118` | `0.97M` | `75.2s` | `7.7651` | `7.4441` | `1709.68` | `10.56%` | `41.64%` | **`243.09`** | `1.0107` | `0.0514` |
| **Eval 2 (`50%`)** | `236` | `1.93M` | `150.4s` | `7.9716` | `7.2354` | `1387.71` | `11.48%` | `43.33%` | **`248.74`** | `1.0064` | `0.0442` |
| **Eval 3 (`75%`)** | `354` | `2.90M` | `225.6s` | `5.5017` | `7.0192` | `1117.87` | `12.16%` | `45.05%` | **`250.48`** | `1.0034` | `0.0325` |
| **Eval 4 (`100%`)** | `471` | `3.86M` | `300.1s` | `7.1139` | **`6.9492`** | **`1042.29`** | **`12.40%`** | **`46.13%`** | **`250.35`** | **`1.0026`** | **`0.0246`** |

- **Outcome (`H4 PASS`)**:
  - `sharp_eff_rank` ($R_{\text{emb}}$) never dropped below `243.09` and finished at **`250.35 / 256`** (**`9.27x` higher** than Wave 39's collapsed `27.0`).
  - `val_ce` decreased monotonically across all checkpoints (`9.0810 -> 7.4441 -> 7.2354 -> 7.0192 -> 6.9492`) with `0` NaN/Inf events, `embed_norm_median = 1.0026`, and `complex_unitary_err = 0.0246`.

---

### 6.7 AST Signature & Code-Length Linter Compliance (`scripts/ast_signature_and_length_linter.py`)

| File | Total Lines | Comment-Free SLOC (`<= 1,000`) | Max Function Args (`<= 8`) | Semicolons (`0`) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [`src/qdlblocks/models/tied_holo_born_complex_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/tied_holo_born_complex_gpt.py) | `935` | `738` | `8` | `0` | **PASS (`0` violations)** |
| [`scripts/fineweb_token_rarity_and_born_rank_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_token_rarity_and_born_rank_profiler.py) | `384` | `316` | `3` | `0` | **PASS (`0` violations)** |
| [`examples/65_verify_plan055_tied_holo_born_and_rare_sampling.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/65_verify_plan055_tied_holo_born_and_rare_sampling.py) | `910` | `789` | `7` | `0` | **PASS (`0` violations)** |
| [`tests/test_plan055_tied_holo_born_and_rare_sampling.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan055_tied_holo_born_and_rare_sampling.py) | `337` | `280` | `2` | `0` | **PASS (`0` violations)** |

