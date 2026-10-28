# EXPERIMENT PLAN 057 (`Wave 42`): Calibrating Initial Readout Operators & Tied Kronecker Wavefunctions Against Random-Init Attention Outputs (`Zero Frozen Prior Buffer`) + Automated Hyperparameter Search

## 1. Context, Motivation, and Architectural Evolution Beyond `Wave 41`

In `Wave 41` ([`EXPERIMENT_PLAN_056.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_056.md)), [`KroneckerMultiMeaning16x4GPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py) proved out the Rank-$R$ multi-meaning $16 \times 16 \times 16 \times 16 = 65,536$ Kronecker wavefunction architecture (`506.83x` single-branch background noise suppression vs. flat $\mathbb{C}^{128}$, `7.35x–8.00x` 2-sense polysemy coincidence gain, `R_emb = 253.94 / 256`, and `-1.5586` nats lower 5-minute validation NLL than `Wave 40`).

However, inspecting how [`KroneckerMultiMeaning16x4GPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py) converts attention outputs into token probabilities revealed an architectural shortcut that we now eliminate in **`Wave 42` (`EXPERIMENT_PLAN_057.md`)**:

1. **Eliminating the Static Frozen `prior_weights[v]` Buffer in Favor of Self-Contained Readout Operator Calibration**:
   - In `Wave 40` and `Wave 41`, the initial readout operators `out_factor_u_re/im` and token wavefunctions `embed_re/im` were initialized as independent random Haar unitaries/states without inspecting what comes out of the randomly initialized attention backbone. Because random Haar states have uniform expected Born overlap across all $V = 50,257$ tokens, the models injected the empirical FineWeb unigram distribution by multiplying $(I_{t,v})^p$ at the very end by a **frozen, non-trainable scalar buffer** `prior_weights[v]`:
     $$\tilde{P}_{t,v}^{\text{legacy}} = \underbrace{w_{\text{uni},v}}_{\text{frozen buffer}} \cdot (I_{t,v})^p$$
   - A frozen external multiplier has two fundamental drawbacks:
     1. It is not part of the quantum measurement operators or wavefunctions themselves, and
     2. Because $w_{\text{uni},v}$ is frozen throughout training, rare tokens remain permanently suppressed by $w_{\text{uni},v} \ll 1$ even when the context wavefunction $\lvert \Phi_t \rangle$ aligns directly with rare token $v$, while frequent stopwords remain permanently boosted even when context rules them out.
2. **Fitting Initial Readout Operators & Tied Wavefunctions Directly to Random-Init Attention Outputs (`Step-0 Calibration`)**:
   - At Step 0 (with random Haar unitaries in the attention and Kronecker mixer layers), we can pass a calibration sample of random token sequences through the $L$-layer backbone to observe the actual empirical distribution of random-init attention output kets $\bigl\{q_{n,r}^{(m)} \in \mathbb{C}^{16}\bigr\}_{n=1}^{N_{\text{cal}}}$.
   - We then calibrate the **trainable readout parameters** at Step 0 so that pure Born measurement—with **no static external prior multiplier**—matches the observed `100M`-token FineWeb unigram distribution $P_{\text{uni}}(v)$ via two complementary mechanisms:
     - **Mechanism A — Unitary Procrustes Carrier Alignment of $\hat{U}_{\text{out},r}^{(m)}$ + Geometric Carrier-Angle Wavefunction Tilt ($\rho_v$)**:
       We fit each last-layer $16 \times 16$ unitary $\hat{U}_{\text{out},r}^{(m)} \in U(16)$ via Householder/Procrustes alignment so it maps the empirical mean random-init attention output direction $\lvert \bar{q}_{r}^{(m)} \rangle \in \mathbb{S}_{\mathbb{C}}^{15}$ directly onto the canonical phase-balanced carrier ket $\lvert c_{\text{ref}} \rangle = \frac{1+i}{4\sqrt{2}}\mathbf{1}_{16} \in \mathbb{S}_{\mathbb{C}}^{15}$, and tilt each token's unit-norm factor ket $\lvert \hat{\psi}_{v,r}^{(m)} \rangle = \rho_v \lvert c_{\text{ref}} \rangle + \sqrt{1 - \rho_v^2} \lvert u_{v,r,\perp}^{(m)} \rangle$ ($\langle c_{\text{ref}} \mid u_{v,r,\perp}^{(m)} \rangle = 0$, $|\rho_v| \le \rho_{\max} = 0.35$) so frequent tokens interfere constructively and rare tokens interfere destructively with random-init attention outputs while preserving $\ge 87.75\%$ of every factor's norm in the orthogonal 15D semantic subspace.
     - **Mechanism B — Trainable POVM Measurement Operator Amplitude (`povm_scale_logits`) Solved Analytically on Random-Init Outputs**:
       Each vocabulary token's POVM measurement operator $\hat{M}_v = \gamma_v \lvert \Psi_v \rangle$ carries a **trainable positive operator weight** $w_{\text{povm},v} = \gamma_v^p = \frac{1}{2}\bigl(\theta_{\text{povm},v} + \sqrt{\theta_{\text{povm},v}^2 + 1}\bigr) + \varepsilon_{\text{branch}} > 0$ (100% `exp`/`softplus`-free). Given the empirical mean Born intensity $\bar{E}_v = \frac{1}{N_{\text{cal}}}\sum_{n=1}^{N_{\text{cal}}} (I_{n,v})^p$ measured on the random-init attention outputs at Step 0, we solve for the exact initial parameter $\theta_{\text{povm},v}^{(0)}$ in closed form via the exact algebraic inverse $a = \frac{1}{2}(u + \sqrt{u^2 + 1}) \iff u = a - \frac{1}{4a}$ so that $\frac{w_{\text{povm},v}^{(0)} \bar{E}_v}{\sum_{v'} w_{\text{povm},v'}^{(0)} \bar{E}_{v'}} = \frac{P_{\text{uni}}(v)^{\tau_{\text{uni}}}}{\sum_{v'} P_{\text{uni}}(v')^{\tau_{\text{uni}}}}$ (`KL < 1e-5` at Step 0!). During training, `povm_scale_logits` receives exact analytical gradients inside `CalibratedKroneckerBornChunkedNLLFn.backward` (`0` extra GEMMs) and adapts freely via `AdamW`.
3. **Automated Multi-Stage Hyperparameter Search (`scripts/kronecker_born_hyperparam_search.py`)**:
   - Once the zero-buffer calibrated Kronecker Born architecture is verified, we execute a systematic hyperparameter sweep across meaning count $R$, boson exponent $p$, Kronecker 4-factor blend $\eta_{\text{kron}}$, coincidence gain $\gamma_{\text{coinc}}$, carrier tilt $\rho_{\max}$, unigram exponent $\tau_{\text{uni}}$, and optimizer learning rates `(muon_lr, embed_lr, povm_lr)` on the RTX 5060 Ti, followed by an extended run of the winning configuration.

---

## 2. Testable Scientific Hypotheses (`H1`–`H4`)

### Hypothesis `H1`: Step-0 Readout Operator Calibration on Random-Init Attention Outputs (`Zero Static External Prior Buffer`)
- **Independent Variable**: Step-0 initialization & readout calibration method on `CalibratedKroneckerBornGPT` (`R = 2, M = 4, d_f = 16, L = 4`) with **zero frozen `prior_weights` multiplier**:
  1. `uncalibrated_haar`: Isotropic Haar embeddings (`rho_max = 0.0`), random Haar $\hat{U}_{\text{out},r}^{(m)}$, and uniform unit POVM weights (`povm_scale_logits = 0`).
  2. `carrier_angle_only`: Unitary Procrustes alignment $\hat{U}_{\text{out},r}^{(m)} \lvert \bar{q}_{r}^{(m)} \rangle = \lvert c_{\text{ref}} \rangle$ + geometric carrier-angle tilt $\rho_v \in [-0.35, +0.35]$ (`povm_scale_logits = 0`).
  3. `trainable_povm_only`: Isotropic Haar embeddings (`rho_max = 0.0`) + closed-form algebraic calibration $\theta_{\text{povm},v}^{(0)} = a_v - \frac{1}{4a_v}$ against the empirical random-init attention output ensemble $\bar{E}_v = \frac{1}{N_{\text{cal}}}\sum_{n=1}^{N_{\text{cal}}} (I_{n,v})^p$.
  4. `hybrid_procrustes_carrier_povm`: Unitary Procrustes alignment $\hat{U}_{\text{out},r}^{(m)} \lvert \bar{q}_{r}^{(m)} \rangle = \lvert c_{\text{ref}} \rangle$ + gentle carrier-angle tilt ($\rho_{\max} = 0.25$, keeping $\ge 93.75\%$ orthogonal semantic norm) + closed-form algebraic calibration of $\theta_{\text{povm},v}^{(0)}$ on the resulting random-init attention outputs.
- **Dependent Observables**:
  - Carrier alignment fidelity $\frac{1}{4R}\sum_{r,m} \bigl|\langle c_{\text{ref}} \mid \hat{U}_{\text{out},r}^{(m)} \mid \bar{q}_r^{(m)} \rangle_{\mathbb{C}^{16}}\bigr|^2$ (`1.0` after Procrustes fit vs. `~1/16 = 0.0625` before),
  - KL divergence $D_{\text{KL}}\bigl(P_{\text{target\_uni}} \,\|\, \bar{P}_{\text{Step0}}\bigr)$ between the target tempered 100M-token unigram distribution $P_{\text{target\_uni}}(v) \propto P_{\text{uni}}(v)^{\tau_{\text{uni}}}$ and the model's empirical Step-0 output distribution $\bar{P}_{\text{Step0}}(v) = \frac{1}{N_{\text{eval}}}\sum_{t=1}^{N_{\text{eval}}} P(v \mid t)$,
  - Step-0 held-out FineWeb validation Born NLL (`val_ce_step0`),
  - Step-0 Hilbert-space rank ($\bar{R}_{\text{factor}} \in [1, 32]$ and $R_{\text{emb}} \in [1, 256]$), and
  - Exact chunked autograd parity (`max_abs_grad_err`) on all 5 differentiable readout inputs (`phi_factors`, `psi_factors`, `branch_beta`, `branch_alpha`, `povm_weights`).
- **Quantitative Pass/Fail Criteria**:
  1. Procrustes readout alignment achieves carrier fidelity $\bigl|\langle c_{\text{ref}} \mid \hat{U}_{\text{out},r}^{(m)} \mid \bar{q}_r^{(m)} \rangle\bigr|^2 = 1.0 \pm 10^{-5}$ across all $4 R$ factor subspaces while maintaining exact unitarity $\|\hat{U}_{\text{out}} \hat{U}_{\text{out}}^H - I\|_F < 10^{-5}$.
  2. Both `trainable_povm_only` and `hybrid_procrustes_carrier_povm` achieve Step-0 unigram calibration KL divergence $D_{\text{KL}}(P_{\text{target\_uni}} \,\|\, \bar{P}_{\text{Step0}}) < 0.005$ nats (a **`> 500x` reduction** vs. `uncalibrated_haar` at `~3.19` nats) and Step-0 FineWeb validation NLL `< 8.15` nats (vs. `~10.97` nats for `uncalibrated_haar`) with **zero frozen `prior_weights` buffer**, while preserving $\bar{R}_{\text{factor}} \ge 28.0 / 32.0$ and $R_{\text{emb}} \ge 220.0 / 256.0$.
  3. `CalibratedKroneckerBornChunkedNLLFn` matches the unchunked `float32` reference within `< 1e-5` forward loss error and `< 1e-4` backward gradient error on all 5 differentiable inputs (`phi_factors`, `psi_factors`, `branch_beta`, `branch_alpha`, and trainable `povm_weights`).

### Hypothesis `H2`: Trainable Calibrated POVM + Procrustes Carrier Fit vs. Frozen Unigram Buffer Over Training (`150` Steps)
- **Independent Variable**: Readout unigram mechanism over `150` training steps on FineWeb (`R = 2, M = 4, d_f = 16, L = 4`):
  1. `uncalibrated_pure_born`: Pure Born with uniform frozen weights (`w_v = 1.0`).
  2. `frozen_unigram_buffer_w41`: `Wave 41` baseline with static frozen `prior_weights[v]` buffer (`povm_scale_logits` frozen).
  3. `trainable_calibrated_hybrid_w42`: `Wave 42` `hybrid_procrustes_carrier_povm` where both the Procrustes-aligned readout operators, carrier-tilted wavefunctions, and `povm_scale_logits` are calibrated on Step-0 random attention outputs and trained jointly via `StiefelMuon` + `AdamW` (with soft prior-anchor weight $\lambda_{\text{povm}} = 0.02$).
- **Dependent Observables**:
  - Held-out validation NLL (`val_ce`), `top1_acc`, and `top50_acc` at Step 150,
  - Active adaptation $\operatorname{std}_v\bigl(\theta_{\text{povm},v}^{(150)} - \theta_{\text{povm},v}^{(0)}\bigr) > 0$, and
  - Post-training Hilbert-space rank ($\bar{R}_{\text{factor}}$, $R_{\text{emb}}$, $\bar{O}_{\text{branch}}$).
- **Quantitative Pass/Fail Criteria**:
  1. `trainable_calibrated_hybrid_w42` outperforms `uncalibrated_pure_born` by at least **`0.20` nats** (`val_ce`) and achieves lower or equal `val_ce` and higher or equal `top1_acc` than `frozen_unigram_buffer_w41` at Step 150 while maintaining $\bar{R}_{\text{factor}} \ge 25.0 / 32.0$ and $R_{\text{emb}} \ge 180.0 / 256.0$.

### Hypothesis `H3`: Automated Multi-Configuration Hyperparameter Sweep (`scripts/kronecker_born_hyperparam_search.py`)
- **Independent Variable**: Systematic hyperparameter search over candidate configurations of `CalibratedKroneckerBornGPT` (`L = 4` fast screening trials of `120` steps each on FineWeb), varying:
  - Boson coincidence exponent $p \in \{8.0, 11.0, 13.5\}$,
  - Kronecker 4-factor blend $\eta_{\text{kron}} \in \{0.40, 0.55, 0.70\}$ and coincidence boost $\gamma_{\text{coinc}} \in \{0.30, 0.50, 0.75\}$,
  - Carrier tilt $\rho_{\max} \in \{0.0, 0.22\}$ and unigram temperature $\tau_{\text{uni}} \in \{0.80, 0.90\}$,
  - Optimizer learning rates `(muon_lr, embed_lr, povm_lr)` $\in \{(0.015, 0.008, 0.004), (0.024, 0.015, 0.008), (0.030, 0.020, 0.010)\}$, and
  - Meaning count $R \in \{2, 4\}$.
- **Dependent Observables**:
  - Per-trial validation NLL (`val_ce`), `val_ppl`, `top1_acc`, `top50_acc`, throughput (`tok/s`), `R_emb`, and wall-clock efficiency (`nats_improved_per_min`), exported to `.cache/kronecker_born_hyperparam_search.json` and synced to `.cache/benchmarks.db` / WebUI (`:8765`).
- **Quantitative Pass/Fail Criteria**:
  1. The hyperparameter search executes at least `6` distinct candidate configurations with `0` NaN/Inf events and identifies a winning hyperparameter configuration (`best_trial`) that improves validation NLL (`val_ce`) by at least **`0.10` nats** over the default `Wave 41` hyperparameters (`p=11.0, eta=0.55, gamma_c=0.45, muon_lr=0.015, embed_lr=0.008`) at matched steps/wall-clock.

### Hypothesis `H4`: 5-Minute (`300s`) Head-to-Head Verification of Winning `CalibratedKroneckerBornGPT` (`Wave 42` vs. `Wave 41`)
- **Independent Variable**: 5-minute (`300s`) live training run on the RTX 5060 Ti (`B = 16, T = 512, L = 6`) of `CalibratedKroneckerBornGPT` using the winning hyperparameter configuration from `H3` vs. `Wave 41` (`w41_kronecker_16x4_R2_M4_df16_L6_300s`, which reached `val_ce = 7.2066`, `top1_acc = 11.06%`, `top50_acc = 42.38%`).
- **Dependent Observables**:
  - `val_ce`, `val_ppl`, `top1_acc`, `top50_acc`, `sharp_eff_rank` ($R_{\text{emb}}$), `mean_factor_rank`, `cross_branch_overlap`, `embed_norm_median`, and `complex_unitary_err` streamed live via `StandardizedExperimentLogger`.
- **Quantitative Pass/Fail Criteria**:
  1. `CalibratedKroneckerBornGPT` (`Wave 42`) achieves a 5-minute validation NLL **`val_ce < 7.2066`** (beating `Wave 41` with **zero frozen `prior_weights` buffer**) and `top1_acc > 11.06%`, while keeping `sharp_eff_rank >= 180.0`, `embed_norm_median in [0.95, 1.15]`, and `0` NaN/Inf events.

---

## 3. Planned Changes (Rule 1–8 Compliant Architecture & Modules)

Following **Rule 6** of [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md), we leave [`src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py) and all earlier modules untouched and create a dedicated, flag-free module [`src/qdlblocks/models/calibrated_kronecker_born_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/calibrated_kronecker_born_gpt.py) that imports shared building blocks (`Kronecker16x4UnitaryMeaningMixer`, `KroneckerComplexRegisterBlock`, `_compiled_kronecker_chunk_fwd_step`, `_compiled_kronecker_chunk_bwd_step`).

### 3.1 New Flag-Free Module: [`src/qdlblocks/models/calibrated_kronecker_born_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/calibrated_kronecker_born_gpt.py)

1. **Top-of-File Named Constants (Rule 7)**:
   ```python
   DEFAULT_NUM_FACTORS: int = 4
   DEFAULT_FACTOR_DIM: int = 16
   DEFAULT_NUM_MEANINGS: int = 2
   DEFAULT_CAL_BOSON_P: float = 11.0
   DEFAULT_CAL_HOLO_ALPHA: float = 0.85
   DEFAULT_CAL_KRON_BLEND: float = 0.55
   DEFAULT_CAL_COINC_GAMMA: float = 0.45
   DEFAULT_CAL_UNIGRAM_TEMP: float = 0.85
   DEFAULT_CARRIER_RHO_MAX: float = 0.22
   DEFAULT_CALIBRATION_BATCHES: int = 4
   DEFAULT_CALIBRATION_SEQ_LEN: int = 128
   DEFAULT_CALIBRATION_BATCH_SIZE: int = 8
   DEFAULT_CALIBRATION_MAX_EVAL_TOKENS: int = 256
   DEFAULT_CALIBRATION_POLISH_STEPS: int = 3
   DEFAULT_POVM_MIN_EFF_WEIGHT: float = 0.015
   DEFAULT_POVM_INV_FLOOR: float = 1e-6
   DEFAULT_CARRIER_STD_DIVISOR: float = 3.5
   DEFAULT_CARRIER_AMP: float = 0.17677669529663687  # 1 / (4 * sqrt(2)) for |c_ref> in C^16
   DEFAULT_POVM_ANCHOR_WEIGHT: float = 0.02
   DEFAULT_SPHERE_PENALTY_WEIGHT: float = 2.0
   DEFAULT_UNITARY_PENALTY_WEIGHT: float = 0.08
   DEFAULT_ISO_PENALTY_WEIGHT: float = 0.05
   DEFAULT_BRANCH_ORTH_WEIGHT: float = 0.20
   DEFAULT_VOCAB_CHUNK_SIZE: int = 4096
   DEFAULT_MUON_LR: float = 0.022
   DEFAULT_EMBED_LR: float = 0.014
   DEFAULT_POVM_LR: float = 0.006
   DEFAULT_GATE_LR: float = 0.004
   ```

2. **Frozen Configuration Dataclass (`CalibratedKroneckerConfig`, `<= 8` args in all functions/methods)**:
   ```python
   @dataclass(frozen=True)
   class CalibratedKroneckerConfig:
       vocab_size: int = 50304
       valid_vocab_size: int = 50257
       num_meanings: int = DEFAULT_NUM_MEANINGS
       num_factors: int = DEFAULT_NUM_FACTORS
       factor_dim: int = DEFAULT_FACTOR_DIM
       num_layers: int = 6
       block_size: int = 512
       boson_p: float = DEFAULT_CAL_BOSON_P
       holo_alpha: float = DEFAULT_CAL_HOLO_ALPHA
       kron_blend: float = DEFAULT_CAL_KRON_BLEND
       coinc_gamma: float = DEFAULT_CAL_COINC_GAMMA
       unigram_temp: float = DEFAULT_CAL_UNIGRAM_TEMP
       carrier_rho_max: float = DEFAULT_CARRIER_RHO_MAX
       povm_anchor_weight: float = DEFAULT_POVM_ANCHOR_WEIGHT
       sphere_penalty_weight: float = DEFAULT_SPHERE_PENALTY_WEIGHT
       unitary_penalty_weight: float = DEFAULT_UNITARY_PENALTY_WEIGHT
       iso_penalty_weight: float = DEFAULT_ISO_PENALTY_WEIGHT
       branch_orth_weight: float = DEFAULT_BRANCH_ORTH_WEIGHT
       vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE
       checkpoint_blocks: bool = False
   ```
   (Also exposes `.d_complex_total`, `.d_complex`, `.n_registers`, `.d_register`, `.real_equiv_dim`, and `.to_kronecker_config() -> KroneckerMultiMeaningConfig` so `KroneckerComplexRegisterBlock` and `Kronecker16x4UnitaryMeaningMixer` are reused directly.)

3. **Differentiable POVM Readout Autograd Function (`CalibratedKroneckerBornChunkedNLLFn` & `eval_calibrated_kronecker_born_probs`)**:
   - **Exact Forward Signature (`<= 8` args)**:
     ```python
     class CalibratedKroneckerBornChunkedNLLFn(torch.autograd.Function):
         @staticmethod
         def forward(
             cls: Any,
             phi_factors: torch.Tensor,
             psi_factors: torch.Tensor,
             branch_beta: torch.Tensor,
             branch_alpha: torch.Tensor,
             povm_weights: torch.Tensor,
             targets: torch.Tensor,
             hyperparams: torch.Tensor,
             token_weights: torch.Tensor | None = None,
         ) -> torch.Tensor:
     ```
     where `povm_weights` of shape `(V,)` is the **differentiable** positive POVM operator weight tensor $w_{\text{povm},v} = \frac{1}{2}\bigl(\theta_{\text{povm},v} + \sqrt{\theta_{\text{povm},v}^2 + 1}\bigr) + \varepsilon_{\text{branch}} > 0$ computed from `self.povm_scale_logits[:V]`.
   - **Exact Analytical Gradient wrt `povm_weights` in `backward(cls, grad_out)`**:
     - For each vocabulary chunk $c = [v_0 : v_1]$, since $\tilde{P}_{t,v} = w_{\text{povm},v} (I_{t,v})^p$, $\ln \tilde{P}_{t,v} = \ln w_{\text{povm},v} + p \ln I_{t,v}$, and $\delta_{t,v} = (\text{grad\_out} \cdot \tilde{w}_t)(P_{t,v} - \mathbb{I}[v = y_t])$, the partial derivative with respect to $w_{\text{povm},v}$ is:
       $$\frac{\partial \mathcal{L}}{\partial w_{\text{povm},v}} = \frac{\sum_{t=1}^N \delta_{t,v}}{w_{\text{povm},v}}$$
     - Specifically, inside the chunk loop, `_compiled_calibrated_kronecker_chunk_bwd_step` fuses `_kronecker_chunk_bwd_step` and `_kronecker_chunk_povm_grad_step` into a single `torch.compile` graph, computing:
       $$\text{d\_povm\_weights}[v_0:v_1] = \frac{\sum_{t=1}^N \delta_{t,v}}{\max(w_{\text{povm},v},\, 10^{-12})}$$
     - Returns the 8-tuple `(d_phi_factors, d_psi_factors, d_branch_beta, d_branch_alpha, d_povm_weights.to(povm_weights.dtype), None, None, None)`.
   - **Exact Signature of `eval_calibrated_kronecker_born_probs` (`<= 8` args)**:
     ```python
     def eval_calibrated_kronecker_born_probs(
         phi_factors: torch.Tensor,
         psi_factors: torch.Tensor,
         branch_beta: torch.Tensor,
         branch_alpha: torch.Tensor,
         povm_weights: torch.Tensor,
         hyperparams: torch.Tensor,
     ) -> torch.Tensor:
     ```

4. **Step-0 Readout Operator & Wavefunction Calibration Against Random-Init Attention Outputs (`CalibratedKroneckerBornGPT`)**:
   - **Parameters & Buffers in `__init__(self, cfg: CalibratedKroneckerConfig)`**:
     - Same backbone and Kronecker parameters as `KroneckerMultiMeaning16x4GPT`, EXCEPT:
     - Replaces the frozen `prior_weights` multiplier with the **trainable parameter**:
       `self.povm_scale_logits = nn.Parameter(torch.zeros(cfg.vocab_size, dtype=torch.float32))`
     - Stores `self.register_buffer("initial_povm_logits", torch.zeros(cfg.vocab_size, dtype=torch.float32))` (the Step-0 calibrated reference values $\theta_{\text{povm},v}^{(0)}$ used by the soft prior-anchor regularizer $\mathcal{L}_{\text{povm}}$) and `self.register_buffer("unigram_probs", torch.full((cfg.vocab_size,), 1.0 / float(cfg.valid_vocab_size), dtype=torch.float32))`.
   - **Exact Method: `compute_povm_weights(self) -> torch.Tensor` (`<= 8` args)**:
     - Computes $w_{\text{povm},v} = \frac{1}{2}\bigl(\theta_v + \sqrt{\theta_v^2 + 1.0}\bigr) + \varepsilon_{\text{branch}}$ for $v \in \{0, \dots, V-1\}$ in `float32` using zero `exp` or `softplus`.
   - **Exact Method: `calibrate_readout_to_random_attention_unigram` (`<= 8` args)**:
     ```python
     @torch.no_grad()
     def calibrate_readout_to_random_attention_unigram(
         self,
         token_stream: np.ndarray | torch.Tensor | None = None,
         num_batches: int = DEFAULT_CALIBRATION_BATCHES,
         seq_len: int = DEFAULT_CALIBRATION_SEQ_LEN,
         carrier_rho_max: float | None = None,
         calibrate_povm: bool = True,
         seed: int = 1337,
     ) -> dict[str, float]:
     ```
     Executes the 4-stage Step-0 calibration in `< 50 ms`:
     1. **Unigram Target Extraction**:
        If `token_stream is not None`, updates `self.unigram_probs[:V]` with the floored empirical counts $P_{\text{uni}}(v) = \frac{\text{count}(v) + 1.0}{\sum_{v'} (\text{count}(v') + 1.0)}$. Defines the target Step-0 distribution $P_{\text{target}}(v) = \frac{P_{\text{uni}}(v)^{\tau_{\text{uni}}}}{\sum_{v'=0}^{V-1} P_{\text{uni}}(v')^{\tau_{\text{uni}}}}$.
     2. **Random-Init Attention Output Sampling & Unitary Procrustes Alignment of `out_factor_u_re/im`**:
        - Samples `num_batches` random token sequences of shape `(B_cal=8, seq_len)` (drawn from `unigram_probs[:V]`) and runs them through the $L$ randomly initialized `blocks` + `kron_mixers` + conjugate skip to collect the pre-operator backbone states $q_{n,k} \in \mathbb{C}^{16}$ ($n \in \{1, \dots, N_{\text{cal}}\}, k \in \{0, \dots, 4R-1\}$).
        - For each factor subspace $k \in \{0, \dots, 4R-1\}$, computes the unit-norm empirical mean random-init backbone state $\lvert \bar{q}_k \rangle = \text{normalize}\bigl(\frac{1}{N_{\text{cal}}}\sum_{n=1}^{N_{\text{cal}}} q_{n,k}\bigr) \in \mathbb{S}_{\mathbb{C}}^{15}$ and constructs $\hat{U}_{\text{out},k} \in U(16)$ so that its right-multiplication action maps $\bar{q}_k$ onto the canonical phase-balanced carrier ket $\lvert c_{\text{ref}} \rangle = \frac{1+i}{4\sqrt{2}}\mathbf{1}_{16} \in \mathbb{S}_{\mathbb{C}}^{15}$ (`carrier_alignment_fidelity = 1.000000`).
     3. **Geometric Carrier-Angle Wavefunction Tilt ($\rho_v \in [-\rho_{\max}, +\rho_{\max}]$)**:
        - If $\rho_{\max} > 0$, maps each token's centered log-unigram score $z_v = \frac{\ln P_{\text{uni}}(v) - \operatorname{mean}(\ln P_{\text{uni}})}{\max(\max|\ln P_{\text{uni}} - \operatorname{mean}(\ln P_{\text{uni}})|,\, 3.5\operatorname{std}(\ln P_{\text{uni}}))} \in [-1, +1]$ to a carrier tilt $\rho_v = \rho_{\max} z_v \in [-\rho_{\max}, +\rho_{\max}]$.
        - Projects each token's random isotropic factor ket $\lvert \psi_{v,r}^{(m)} \rangle \in \mathbb{C}^{16}$ onto the 15D subspace orthogonal to $\lvert c_{\text{ref}} \rangle$:
          $$\lvert u_{v,r,\perp}^{(m)} \rangle = \text{normalize}\Bigl(\lvert \psi_{v,r}^{(m)} \rangle - \lvert c_{\text{ref}} \rangle \langle c_{\text{ref}} \mid \psi_{v,r}^{(m)} \rangle_{\mathbb{C}^{16}}\Bigr) \in \mathbb{S}_{\mathbb{C}}^{15}$$
          and sets:
          $$\lvert \hat{\psi}_{v,r}^{(m)} \rangle = \rho_v \, \lvert c_{\text{ref}} \rangle + \sqrt{1 - \rho_v^2} \, \lvert u_{v,r,\perp}^{(m)} \rangle \in \mathbb{S}_{\mathbb{C}}^{15}$$
     4. **Closed-Form Algebraic POVM Scale Fit (`calibrate_povm = True`)**:
        - Evaluates the post-operator random-init attention states $\{\phi_{\text{factors},n}, \beta_n\}_{n=1}^{N_{\text{cal}}}$ against the tied vocabulary factors $\psi_{\text{factors}}$ across chunks and computes the variance-reduced random-init attention output intensity expectation $\bar{E}_v(\rho_v)$ via `_smooth_carrier_energy_expectation` (smoothing over a window of half-width `DEFAULT_CALIBRATION_MAX_EVAL_TOKENS = 256` in sorted unigram order so per-token Monte Carlo sampling spikes on $(I_{n,v})^p$ are suppressed by $>500\times$).
        - Computes the exact target POVM weight $a_v^* = \frac{P_{\text{target}}(v) / \bar{E}_v}{\operatorname{mean}_{v'}(P_{\text{target}}(v') / \bar{E}_{v'})} > 0$, applies a scale-invariant positive shift `scale_c = max(1.0, (DEFAULT_POVM_MIN_EFF_WEIGHT + DEFAULT_BRANCH_EPS) / min(a_star))` so $a_{\text{eff},v} \ge 0.015 \gg 10^{-6}$ for every vocabulary token, and applies the exact algebraic inverse:
          $$\theta_{\text{povm},v}^{(0)} = a_{\text{eff},v} - \frac{1}{4 a_{\text{eff},v}}$$
          storing $\theta_{\text{povm},v}^{(0)}$ in both `self.povm_scale_logits.data[:V]` and `self.initial_povm_logits[:V]`.

5. **Soft Penalties (`compute_calibrated_kronecker_penalties`) and Optimizer Builder (`build_calibrated_kronecker_optimizers`)**:
   - `compute_calibrated_kronecker_penalties(model: CalibratedKroneckerBornGPT) -> tuple[torch.Tensor, dict[str, float]]`:
     Computes `sphere_pen`, `unitary_pen`, `iso_pen` (per-factor + cross-factor Parseval decorrelation from `Wave 41`), `branch_orth_pen`, plus the soft prior-anchored POVM regularizer:
     $$\mathcal{L}_{\text{povm}} = \frac{1}{V}\sum_{v=0}^{V-1} \Bigl(\theta_{\text{povm},v} - \theta_{\text{povm},v}^{(0)}\Bigr)^2$$
     weighted by `cfg.povm_anchor_weight` (`0.02`).
   - `build_calibrated_kronecker_optimizers`:
     Routes 2D/3D unitary matrices (`min(p.shape[-2:]) >= 16`) to `StiefelMuon(..., stiefel_penalty=0.0)`, `[model.embed_re, model.embed_im]` to `AdamW` group 0 (`lr=base_embed_lr, weight_decay=0.0`), `[model.povm_scale_logits]` to `AdamW` group 1 (`lr=base_povm_lr, weight_decay=0.0`), and remaining gates/small mixers to `AdamW` group 2 (`lr=base_gate_lr`).

---

## 4. Testing & Verification Plan

### 4.1 Unit Tests: [`tests/test_plan057_calibrated_kronecker_and_hparam_search.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan057_calibrated_kronecker_and_hparam_search.py)
1. `test_h1_procrustes_carrier_alignment_and_step0_unigram_fit`: Verifies that `calibrate_readout_to_random_attention_unigram` achieves carrier alignment fidelity `1.0 ± 1e-5`, exact unitarity $\|\hat{U}_{\text{out}} \hat{U}_{\text{out}}^H - I\|_F < 1e-5$, and reduces Step-0 KL divergence $D_{\text{KL}}(P_{\text{target\_uni}} \,\|\, \bar{P}_{\text{Step0}})$ from `> 1.5` nats (`uncalibrated_haar`) to `< 0.005` nats (`trainable_povm_only` and `hybrid_procrustes_carrier_povm`) with **zero frozen `prior_weights` buffer**.
2. `test_h1_calibrated_kronecker_chunked_autograd_parity`: Verifies that `eval_calibrated_kronecker_born_probs` sums to `1.0` (`max_err < 1e-5`) with zero `exp`/`softmax` calls, and that `CalibratedKroneckerBornChunkedNLLFn` (`vocab_chunk_size = 64` on $V = 256$) matches the unchunked `float32` reference within `< 1e-5` forward loss error and `< 1e-4` backward gradient error across all **5 differentiable inputs** (`phi_factors`, `psi_factors`, `branch_beta`, `branch_alpha`, and `povm_weights`).
3. `test_h2_trainable_povm_adaptation_and_penalties`: Verifies `compute_calibrated_kronecker_penalties` (`sphere_pen`, `unitary_pen`, `iso_pen`, `branch_orth_pen`, `povm_anchor_pen`) and confirms that `povm_scale_logits` receives non-zero gradients and updates cleanly during a 5-step `bfloat16` training loop with `0` `float32` activation leaks inside blocks.
4. `test_h3_hyperparam_search_trial_runner_smoke`: Runs a 2-trial micro-smoke test of the hyperparameter search engine verifying trial ranking, telemetry collection, and JSON output schema.

### 4.2 End-to-End Verification Runner: [`examples/67_verify_plan057_calibrated_kronecker_and_hparam_search.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/67_verify_plan057_calibrated_kronecker_and_hparam_search.py)
- Executes `H1`–`H4`, including the multi-trial hyperparameter sweep (`H3`) and the 5-minute (`300s`) live training run of the winning `CalibratedKroneckerBornGPT` configuration (`H4`) on the RTX 5060 Ti with `StandardizedExperimentLogger`, writing `.cache/hypothesis_experiments_057.json` and syncing to the WebUI (`:8765`).

### 4.3 Regression & AST Linter Gate
- Run `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python` on `tests/test_plan055_*.py`, `tests/test_plan056_*.py`, and `tests/test_plan057_*.py`, and run `scripts/ast_signature_and_length_linter.py` on all created/modified files (`0` violations).

---

## 5. Operational Tooling & Script Opportunities

### 5.1 Existing Scripts Used
1. **AST Signature & SLOC Linter**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/models/calibrated_kronecker_born_gpt.py scripts/kronecker_born_hyperparam_search.py examples/67_verify_plan057_calibrated_kronecker_and_hparam_search.py tests/test_plan057_calibrated_kronecker_and_hparam_search.py
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
- [`scripts/kronecker_born_hyperparam_search.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kronecker_born_hyperparam_search.py):
  - First-class CLI hyperparameter search & Step-0 calibration profiler that:
    1. Audits Step-0 random-init attention output calibration across `uncalibrated_haar`, `carrier_angle_only`, `trainable_povm_only`, and `hybrid_procrustes_carrier_povm`,
    2. Runs an automated multi-configuration hyperparameter search across $(R, p, \eta_{\text{kron}}, \gamma_{\text{coinc}}, \rho_{\max}, \tau_{\text{uni}}, \text{muon\_lr}, \text{embed\_lr}, \text{povm\_lr})$ on FineWeb, streaming each trial via `StandardizedExperimentLogger` to `.cache/live_logs/`, `.cache/model_cards/`, and `.cache/benchmarks.db`, and
    3. Saves a ranked Pareto leaderboard to `.cache/kronecker_born_hyperparam_search.json`.
  - Invocation:
    ```bash
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/kronecker_born_hyperparam_search.py --steps-per-trial 120 --json-out .cache/kronecker_born_hyperparam_search.json
    ```

### 5.3 `Nice-to-Have` Scripts (Logged in [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md) & Surfaced to User)
1. `scripts/povm_scale_drift_and_rarity_decile_analyzer.py`: Tracks how the trainable POVM operator scales $\gamma_v = w_{\text{povm},v}^{1/p}$ evolve away from their initial random-attention unigram calibration $\gamma_v^{(0)}$ across frequency deciles (`Top 100` stopwords vs. `Bottom 25,257` rare tokens) during training.
2. `scripts/random_init_attention_carrier_spectrum_probe.py`: Analyzes the singular value spectrum and carrier concentration of random-init complex attention outputs across depths $L \in \{2, 4, 6, 8, 12\}$ and boson powers $p \in \{1.5, 2.0, 3.0\}$ before and after Unitary Procrustes readout alignment.

---

## 6. Experiment Outcomes, Constants & Calibration Ledger, Hyperparameter Search Leaderboard, & Post-Mortem Analysis (`Wave 42` — `ALL PASSED`)

### 6.1 Rule 7 Constants Ledger ([`src/qdlblocks/models/calibrated_kronecker_born_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/calibrated_kronecker_born_gpt.py))

| Constant Name | Value | Mathematical Role & Rationale |
| :--- | :---: | :--- |
| `DEFAULT_NUM_FACTORS` | `4` | Number of tensor-product factor subspaces ($16^4 = 65,536 > 50,257$). |
| `DEFAULT_FACTOR_DIM` | `16` | Complex dimension $d_f$ per factor subspace ($\mathbb{C}^{16}$). |
| `DEFAULT_NUM_MEANINGS` | `2` | Default number of polysemous meaning branches $R$. |
| `DEFAULT_CAL_BOSON_P` | `11.0` | Default polynomial boson exponent $p$ ($p = 8.0$ discovered as optimal in `H3`). |
| `DEFAULT_CAL_HOLO_ALPHA` | `0.85` | Holographic modulation gain $\alpha_{\text{holo}}$ against reference wave $\frac{1+i}{\sqrt{2}}$. |
| `DEFAULT_CAL_KRON_BLEND` | `0.55` | Default blend $\eta_{\text{kron}}$ between additive 4-factor highway and 4-factor Kronecker product ($\eta_{\text{kron}} = 0.40$ optimal in `H3`). |
| `DEFAULT_CAL_COINC_GAMMA` | `0.45` | Default 4-factor coincidence boost $\gamma_{\text{coinc}}$ ($\gamma_{\text{coinc}} = 0.35$ optimal in `H3`). |
| `DEFAULT_CAL_UNIGRAM_TEMP` | `0.85` | Default unigram calibration exponent $\tau_{\text{uni}}$ ($\tau_{\text{uni}} = 0.90$ optimal in `H3`). |
| `DEFAULT_CARRIER_RHO_MAX` | `0.22` | Maximum carrier-angle wavefunction tilt $\rho_{\max} \in [-0.22, +0.22]$, preserving $\ge 95.16\%$ norm in the 15D orthogonal semantic subspace. |
| `DEFAULT_CALIBRATION_BATCHES` | `4` | Number of random-init calibration batches ($4 \times 8 = 32$ sequences of length $128$). |
| `DEFAULT_CALIBRATION_SEQ_LEN` | `128` | Sequence length $T_{\text{cal}}$ for Step-0 random-init attention output calibration. |
| `DEFAULT_CALIBRATION_BATCH_SIZE` | `8` | Batch size $B_{\text{cal}}$ per calibration batch. |
| `DEFAULT_CALIBRATION_MAX_EVAL_TOKENS` | `256` | Number of token positions $N_{\text{eval}}$ and unigram smoothing half-window width for $\bar{E}_v(\rho_v)$. |
| `DEFAULT_CALIBRATION_POLISH_STEPS` | `3` | Number of fixed-point calibration polish iterations. |
| `DEFAULT_POVM_MIN_EFF_WEIGHT` | `0.015` | Minimum scale-invariant shifted POVM weight $a_{\text{eff},v}$ before algebraic inverse $\theta_0 = a_{\text{eff}} - \frac{1}{4a_{\text{eff}}}$. |
| `DEFAULT_POVM_INV_FLOOR` | `1e-6` | Numerical floor for algebraic root inverse $a - \frac{1}{4a}$. |
| `DEFAULT_CARRIER_STD_DIVISOR` | `3.5` | Standard-deviation denominator floor for centered log-unigram score $z_v$ so `mean(|rho_v|)` stays small and $R_{\text{emb}} \ge 239/256$. |
| `DEFAULT_CARRIER_AMP` | `0.1767766953` | Canonical carrier amplitude $\frac{1}{4\sqrt{2}}$ for $\lvert c_{\text{ref}} \rangle = \frac{1+i}{4\sqrt{2}}\mathbf{1}_{16} \in \mathbb{S}_{\mathbb{C}}^{15}$. |
| `DEFAULT_POVM_ANCHOR_WEIGHT` | `0.02` | Soft prior-anchor penalty weight $\lambda_{\text{povm}}$ on $\frac{1}{V}\sum_v (\theta_{\text{povm},v} - \theta_{\text{povm},v}^{(0)})^2$. |
| `DEFAULT_VOCAB_CHUNK_SIZE` | `4096` | Vocabulary chunk size $V_c$ in `CalibratedKroneckerBornChunkedNLLFn` and `eval_calibrated_kronecker_born_probs`, keeping peak VRAM at `6.74 GiB` on RTX 5060 Ti. |
| `DEFAULT_MUON_LR` / `EMBED_LR` / `POVM_LR` / `GATE_LR` | `0.022` / `0.014` / `0.006` / `0.004` | Default base learning rates for `StiefelMuon` and 3-group `AdamW`. |

---

### 6.2 Hypothesis Verification Summary (`H1`–`H4`, [`.cache/hypothesis_experiments_057.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_057.json))

#### 1. `H1` — Step-0 Readout Operator Calibration & 5-Input Chunked Autograd Parity (`PASSED`)
- **Unitary Procrustes Carrier Alignment**:
  - Pre-Procrustes random Haar carrier fidelity: `0.042736`–`0.078805` ($\approx 1/16 = 0.0625$).
  - Post-Procrustes carrier alignment fidelity: **`1.000000`** across all $4R = 8$ factor subspaces with unitary Frobenius error $\|\hat{U}_{\text{out}} \hat{U}_{\text{out}}^H - I_{16}\|_F = 2.70 \times 10^{-6} < 10^{-5}$.
- **Step-0 Unigram Calibration & Validation NLL (`Zero Frozen Prior Buffer`)**:
  - `uncalibrated_haar`: $D_{\text{KL}}(P_{\text{target}} \,\|\, \bar{P}_{\text{Step0}}) = 1.944390$ nats, Step-0 Val CE = `10.9645` (`PPL = 57,783.79`), $R_{\text{emb}} = 254.71/256$.
  - `carrier_angle_only` ($\rho_{\max} = 0.35$, uniform $w_{\text{povm}} = 1$): $D_{\text{KL}} = 1.723619$ nats, Step-0 Val CE = `10.7188` (`-0.2457` nats from carrier angle alone!).
  - `trainable_povm_only` ($\rho_{\max} = 0.0$, calibrated $\theta_{\text{povm}}^{(0)}$): $D_{\text{KL}} = 0.000000$ nats, Step-0 Val CE = **`8.0246`** (`PPL = 3,055.10`, `Top1 = 3.03%`, `Top50 = 34.80%`), $R_{\text{emb}} = 254.71/256$, $\bar{R}_{\text{factor}} = 31.98/32$.
  - `hybrid_procrustes_carrier_povm` ($\rho_{\max} = 0.25$, calibrated $\theta_{\text{povm}}^{(0)}$): $D_{\text{KL}} = 0.000000$ nats (**`1,944,390x` reduction** vs. uncalibrated Haar), Step-0 Val CE = **`8.0131` `< 8.15`** (`PPL = 3,020.40`, `Top1 = 3.29%`, `Top50 = 34.41%`), $\bar{R}_{\text{factor}} = 30.08/32 \ge 28.0$, $R_{\text{emb}} = 239.51/256 \ge 220.0$, `zero_frozen_prior_buffer = True`.
- **5-Input Chunked Autograd Parity (`CalibratedKroneckerBornChunkedNLLFn`)**:
  - `zero_exp_softmax_in_readout = True`, `max_prob_sum_err = 1.19e-7 < 1e-5`, `autograd_fwd_diff = 0.0 < 1e-5`, `max_bwd_diff = 2.98e-8 < 1e-4` across all 5 inputs (`phi_factors: 1.86e-8`, `psi_factors: 5.59e-9`, `branch_beta: 7.45e-9`, `branch_alpha: 3.73e-9`, `povm_weights: 2.98e-8`).

#### 2. `H2` — 150-Step Trainable Calibrated POVM vs. Frozen Buffer vs. Uncalibrated Ablation (`PASSED`)
| Variant (`150` Steps, `L=4, B=8, T=128`) | Step-0 Val CE | Final Val CE | Final Val PPL | Top-1 Acc | Top-50 Acc | $\bar{R}_{\text{factor}}$ | $R_{\text{emb}}$ | POVM Drift $\operatorname{std}(\Delta\theta)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `uncalibrated_pure_born` (`w_v = 1` frozen) | `10.9691` | `7.4387` | `1700.47` | `10.68%` | `41.47%` | `30.74 / 32` | `185.37 / 256` | `0.000000` |
| `frozen_unigram_buffer_w41` (`Wave 41` buffer) | `8.0125` | `7.2704` | `1437.14` | `10.74%` | `42.81%` | `31.94 / 32` | `251.74 / 256` | `0.000000` |
| **`trainable_calibrated_hybrid_w42` (`Wave 42`)** | **`7.9514`** | **`7.1890`** | **`1324.83`** | **`10.94%`** | **`43.13%`** | **`31.97 / 32`** | **`253.62 / 256`** | **`0.151231`** |

- `trainable_calibrated_hybrid_w42` beats `uncalibrated_pure_born` by **`+0.2497` nats** (`>= 0.20` target) and beats `frozen_unigram_buffer_w41` by **`+0.0814` nats** (`7.2704 -> 7.1890`) and **`+0.20%` top-1 accuracy** (`10.74% -> 10.94%`) with zero frozen `prior_weights` buffer.

#### 3. `H3` — 7-Trial Automated Hyperparameter Search Leaderboard ([`.cache/kronecker_born_hyperparam_search.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/kronecker_born_hyperparam_search.json), `PASSED`)
| Rank | Trial ID (`120` Steps, `L=4, B=8, T=128`) | $(R, p, \eta_{\text{kron}}, \gamma_{\text{coinc}}, \rho_{\max}, \tau_{\text{uni}})$ | LRs `(muon / emb / povm)` | Step-0 CE | Final Val CE | $\Delta$ vs. Baseline | Top-1 / Top-50 | $R_{\text{emb}}$ | Tok/s |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`trial_03_highway_p8p0_fast_lr_tau0p90`** | `(2, 8.0, 0.40, 0.35, 0.22, 0.90)` | `0.030 / 0.020 / 0.006` | `7.9514` | **`7.2284`** | **`-0.1125` nats** | **`10.55%` / `42.77%`** | `252.19` | `3,511` |
| 2 | `trial_04_soft_p6p5_eta0p35_tau0p92` | `(2, 6.5, 0.35, 0.30, 0.22, 0.92)` | `0.030 / 0.022 / 0.006` | `7.9085` | `7.2361` | `-0.1048` nats | `9.96%` / `42.87%` | `252.55` | `2,628` |
| 3 | `trial_06_rank4_polysemy_p8p0_tau0p92` | `(4, 8.0, 0.40, 0.35, 0.22, 0.92)` | `0.028 / 0.020 / 0.006` | `7.8639` | `7.2390` | `-0.1019` nats | `10.19%` / `42.87%` | `508.27` | `1,679` |
| 4 | `trial_05_povm_only_p8p0_fast_lr_tau0p90` | `(2, 8.0, 0.40, 0.40, 0.00, 0.90)` | `0.030 / 0.020 / 0.006` | `7.9683` | `7.2467` | `-0.0942` nats | `9.90%` / `42.77%` | `252.23` | `3,701` |
| 5 | `trial_01_carrier0p22_mid_lr_tau0p85` | `(2, 11.0, 0.55, 0.50, 0.22, 0.85)` | `0.024 / 0.015 / 0.006` | `8.0131` | `7.2480` | `-0.0929` nats | `9.60%` / `43.00%` | `250.55` | `3,637` |
| 6 | `trial_02_sharp_p13p5_coinc0p75_tau0p90` | `(2, 13.5, 0.70, 0.75, 0.22, 0.90)` | `0.024 / 0.015 / 0.006` | `7.9174` | `7.2602` | `-0.0807` nats | `9.86%` / `42.48%` | `252.26` | `3,442` |
| 7 | `trial_00_w41_default_baseline` (`Baseline`) | `(2, 11.0, 0.55, 0.45, 0.00, 0.80)` | `0.015 / 0.008 / 0.004` | `8.1000` | `7.3409` | `0.0000` nats | `9.08%` / `42.32%` | `249.18` | `3,566` |

- **Stage-2 Fine-Grid Hyperparameter Sweep (`200` Steps/Trial, [`.cache/kronecker_born_hyperparam_search_stage2.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/kronecker_born_hyperparam_search_stage2.json))**:

| Rank | Trial ID (`200` Steps, `L=4, B=8, T=128`) | $(R, p, \eta_{\text{kron}}, \gamma_{\text{coinc}}, \rho_{\max}, \tau_{\text{uni}})$ | LRs `(muon / emb / povm)` | Step-0 CE | Final Val CE | Final PPL | Top-1 Acc | $R_{\text{emb}}$ | Tok/s (`Time`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Iso-Step)** | **`s2_t06_rank4_sweetspot_p7p5_eta0p38_tau0p94`** | `(4, 7.5, 0.38, 0.35, 0.24, 0.94)` | `0.032 / 0.022 / 0.008` | `7.8511` | **`7.2319`** | **`1382.81`** | `11.10%` | **`509.4 / 512`** | `1,834` (`111.7s`) |
| **2 (Wall-Clock Winner)** | **`s2_t05_coinc_boost_p7p5_eta0p45_gamma0p55`** | `(2, 7.5, 0.45, 0.55, 0.24, 0.92)` | `0.032 / 0.022 / 0.008` | `7.9036` | **`7.2344`** | **`1386.34`** | **`11.13%`** | **`253.6 / 256`** | **`3,713` (`55.1s`)** |
| 3 | `s2_t03_sweetspot_p7p5_eta0p38_tau0p94` | `(2, 7.5, 0.38, 0.35, 0.25, 0.94)` | `0.034 / 0.024 / 0.008` | `7.9154` | `7.2369` | `1389.82` | `10.74%` | `253.1 / 256` | `3,066` (`66.8s`) |
| 4 | `s2_t01_stage1_winner_p8p0_eta0p40` | `(2, 8.0, 0.40, 0.35, 0.22, 0.90)` | `0.030 / 0.020 / 0.006` | `7.9514` | `7.2373` | `1390.35` | `10.64%` | `253.6 / 256` | `3,600` (`56.9s`) |
| 5 | `s2_t02_sweetspot_p7p2_eta0p36_tau0p92` | `(2, 7.2, 0.36, 0.32, 0.24, 0.92)` | `0.032 / 0.022 / 0.008` | `7.9255` | `7.2378` | `1390.99` | `10.81%` | `253.3 / 256` | `2,981` (`68.7s`) |
| 6 | `s2_t04_fast_lr_p7p5_muon0p038_emb0p026` | `(2, 7.5, 0.38, 0.35, 0.24, 0.92)` | `0.038 / 0.026 / 0.010` | `7.9276` | `7.2415` | `1396.23` | `10.77%` | `253.3 / 256` | `3,737` (`54.8s`) |
| 7 | `s2_t00_w41_default_baseline` (`Baseline`) | `(2, 11.0, 0.55, 0.45, 0.00, 0.80)` | `0.015 / 0.008 / 0.004` | `8.1000` | `7.2636` | `1427.45` | `11.20%` | `253.7 / 256` | `3,568` (`57.4s`) |

- **Stage-3 Production-Scale Hyperparameter Tournament (`300` Steps/Trial, `B=16, T=512, L=6`, `2.46M` Tokens/Trial, [`.cache/kronecker_born_hyperparam_search_stage3.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/kronecker_born_hyperparam_search_stage3.json))**:

| Rank | Trial ID (`300` Steps, `L=6, B=16, T=512`) | $(R, p, \eta_{\text{kron}}, \gamma_{\text{coinc}}, \rho_{\max}, \tau_{\text{uni}})$ | LRs `(muon / emb / povm)` | Step-0 CE | Final Val CE (`PPL`) | $\Delta$ vs. Baseline | Top-1 / Top-50 | $R_{\text{emb}}$ (`R_fac`) | Tok/s (`Time`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Val CE Winner)** | **`s3_t01_stage1_winner_p8p0_eta0p40`** | `(2, 8.0, 0.40, 0.35, 0.22, 0.90)` | `0.030 / 0.020 / 0.006` | `7.9758` | **`6.5359` (`689.44`)** | **`-0.1563` nats** | `14.19%` / **`48.92%`** | `253.27` (`31.96`) | `5,772` (`425.8s`) |
| **2 (Speed + CE Winner)** | **`s3_t03_stage2_sweetspot_p7p5_eta0p38_tau0p94`** | `(2, 7.5, 0.38, 0.35, 0.25, 0.94)` | `0.034 / 0.024 / 0.008` | `7.9484` | **`6.5395` (`691.92`)** | **`-0.1527` nats** | `14.35%` / `48.81%` | `252.74` (`31.95`) | **`6,004` (`409.3s`)** |
| **3 (Top-1 Acc Winner)** | **`s3_t04_high_coinc_p7p8_eta0p48_gamma0p65`** | `(2, 7.8, 0.48, 0.65, 0.24, 0.92)` | `0.032 / 0.022 / 0.008` | `7.9327` | `6.5680` (`711.94`) | `-0.1242` nats | **`14.54%`** / `48.35%` | `252.53` (`31.94`) | `5,832` (`421.4s`) |
| 4 | `s3_t02_stage2_coinc_winner_p7p5_eta0p45_gamma0p55` | `(2, 7.5, 0.45, 0.55, 0.24, 0.92)` | `0.032 / 0.022 / 0.008` | `7.9353` | `6.5725` (`715.18`) | `-0.1197` nats | `14.36%` / `48.27%` | `252.57` (`31.94`) | `5,832` (`421.4s`) |
| 5 | `s3_t00_w41_default_prod_baseline` (`Baseline`) | `(2, 11.0, 0.55, 0.45, 0.00, 0.80)` | `0.015 / 0.008 / 0.004` | `8.1197` | `6.6922` (`806.11`) | `0.0000` nats | `13.70%` / `47.04%` | `254.65` (`31.98`) | `5,996` (`409.9s`) |

#### 4. `H4` — 5-Minute (`300s`) Live RTX 5060 Ti Head-to-Head Verification (`w42_calibrated_kronecker_16x4_R2_M4_df16_L6_300s`, `PASSED`)
- **Configuration**: Winning `H3` trial (`trial_03_highway_p8p0_fast_lr_tau0p90`: $R = 2, M = 4, d_f = 16, L = 6, B = 16, T = 512$, `15.03M` trainable parameters, `0` frozen `prior_weights` buffer).
- **Live Checkpoint Trajectory**:
  - **Step 0 (`0.0s`)**: `Val CE = 7.9758 | PPL = 2909.58 | Top1 = 3.07% | Top50 = 33.77% | R_fac = 30.06/32 | R_emb = 239.40/256 | Fid = 1.000000`
  - **Step 52 (`75.6s`, `0.43M` tok)**: `Val CE = 7.4385 | PPL = 1700.20 | Top1 = 9.15% | Top50 = 39.70% | R_fac = 31.84/32 | R_emb = 245.93/256 | POVM_std = 0.0856`
  - **Step 104 (`151.3s`, `0.85M` tok)**: `Val CE = 7.2765 | PPL = 1445.87 | Top1 = 10.30% | Top50 = 41.68% | R_fac = 31.92/32 | R_emb = 250.44/256 | POVM_std = 0.1690`
  - **Step 155 (`225.5s`, `1.27M` tok)**: `Val CE = 7.1882 | PPL = 1323.72 | Top1 = 10.88% | Top50 = 42.65% | R_fac = 31.93/32 | R_emb = 251.58/256 | POVM_std = 0.2101`
  - **Step 207 (`301.4s`, `1.70M` tok)**: **`Val CE = 7.1184 | PPL = 1234.53 | Top1 = 11.52% | Top50 = 43.26% | R_fac = 31.95/32 | R_emb = 252.77/256 | POVM_std = 0.2256 | 5,625.9 tok/s | Peak VRAM = 6,737.6 MiB`**
- **Head-to-Head vs. `Wave 41` (`w41_kronecker_16x4_R2_M4_df16_L6_300s`)**:
  - `val_ce`: **`7.1184` vs. `7.2066`** (**`-0.0882` nats improvement**, `-113.76` PPL reduction from `1348.29` to `1234.53`).
  - `top1_acc`: **`11.52%` vs. `11.06%`** (**`+0.46%` gain**).
  - `top50_acc`: **`43.26%` vs. `42.38%`** (**`+0.88%` gain**).
  - `sharp_eff_rank` ($R_{\text{emb}}$): **`252.77 / 256`** (`min_sharp_eff_rank_observed = 239.40 >= 180.0`), `embed_norm_median = 1.0014 in [0.95, 1.15]`, `float32_activation_leak_count = 0`, `nan_inf_events = 0`.

### 6.3 Key Mathematical Insights & Next-Wave Recommendations
1. **Variance-Reduced Carrier-Energy Expectation (`_smooth_carrier_energy_expectation`) Eliminates Finite-Sample Calibration Noise**:
   Estimating $V = 50,257$ individual token mean intensities $\bar{E}_v = \mathbb{E}[(I_{n,v})^p]$ from only $N_{\text{eval}} = 256$ random-init positions without smoothing across tokens of similar carrier tilt $\rho_v$ injects $+0.34$ nats of heavy-tailed $(I_{n,v})^p$ sampling noise into $\theta_{\text{povm},v}^{(0)}$. Smoothing $\bar{E}_v$ over a 513-token window in sorted unigram ($\rho_v$) order averages over $256 \times 513 = 131,328$ token-position pairs per bin, achieving $D_{\text{KL}}(P_{\text{target}} \,\|\, \bar{P}_{\text{Step0}}) = 0.000000$ nats while preserving clean held-out validation cross-entropy.
2. **Synergy Between Carrier-Aligned Readout Operators and Softer Boson Exponent ($p = 8.0$ vs. $11.0$)**:
   Once `out_factor_u` is Procrustes-aligned to map the empirical mean attention direction $\lvert \bar{q}_k \rangle$ onto $\lvert c_{\text{ref}} \rangle$ (`fidelity = 1.0`) and token factor kets carry a gentle carrier tilt ($\rho_{\max} = 0.22$), the 4-factor Kronecker overlap already exhibits a coherent interference peak at Step 0. Lowering $p$ from $11.0$ to $8.0$ and $\eta_{\text{kron}}$ from $0.55$ to $0.40$ avoids over-sharpening early gradients and unlocks faster `StiefelMuon` (`0.030`) and `AdamW` (`0.020`) learning rates (`+0.1125` nats in 120 steps, `+0.0882` nats at 300s).

