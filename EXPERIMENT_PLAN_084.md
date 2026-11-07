# EXPERIMENT PLAN 084: Analytical Parseval Temperature ($\beta_0 = 10.0$), Large-Batch Gradient Accumulation (`8K -> 262K` Tok/Step), Orthonormal Two-Basis $U(16)$ MUB Tomography, Rank-2 CPTP Stiefel Kraus Channels ($\operatorname{St}_{\mathbb{C}}(32, 16)$), `L = 8` Trunk Rebalancing, & Closed-Loop Shadow-EMA Thermostat (`Wave 64`)

## 1. Executive Summary & Architectural Motivation

In [`docs/SESSION_WRAPUP_EXPLICIT_DENSITY_AND_RIEMANNIAN_LR_2026_11_06.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SESSION_WRAPUP_EXPLICIT_DENSITY_AND_RIEMANNIAN_LR_2026_11_06.md) (`Wave 63` / `EXPERIMENT_PLAN_083.md`), we consolidated our quantum density-matrix language model (`ExplicitDensityTomographicGPT`, `T2`) onto a 100% power-of-2 `512D` ($H = 16$ heads of $\mathbb{C}^{16 \times 16}$) geometry, eliminated `AdamW` completely (`0` Step-1 temperature quenches up to `32.0x` constant LR), and proved via our Hot + Shadow-EMA + Branch-and-Quench probe that higher-LR duty-matched trajectories (`PROBE_3_UNIFORM_2X` at `6.2885` Val CE and `PROBE_4_DUTY_MATCHED` at `6.3047` Val CE with `RhoRank = 3.46 / 16`) overtake `1.0x` (`6.2989`) after a 60-step geodesic quench.

Our `Wave 63` diagnostics also exposed five concrete mathematical and systems bottlenecks that `Wave 64` (`EXPERIMENT_PLAN_084.md`) resolves in sequence:

1. **Step 1 — Eliminating $\mathcal{O}(\beta^2)$ Hessian Curvature Drift via Analytical Parseval Temperature ($\beta_0 = 10.0$) & Magnitude-Aware Muon Damping ($g_{\text{mag}}$)**:
   - When `readout_temp` ($\beta$) is a free learnable scalar, it drifts upward from `0.50` (Step 0) $\to$ `4.81` (Step 60) $\to$ `10.82` (`100M` tokens). Because the logit Hessian scales quadratically with temperature ($\|H_z\| = \beta^2 \|\operatorname{diag}(p) - p p^\top\| \propto \beta^2$), this $2.25\times$ post-warmup temperature drift silently multiplies valley-wall curvature by **`5.06x` mid-run**, while also penalizing `PROBE_4_DUTY_MATCHED` (`scalar_mult = 0.75x`) during early warmup.
   - Replacing `readout_temp` with a **non-trainable analytical Parseval schedule $\beta_0(t)$** (`0.50` at Step 0 for exact BPE-rank Zipf prior calibration, ramping to `4.0` by `0.16M` tokens and `10.0` by `1.23M` tokens / `150` steps at `8K` batch, then fixed at $\beta_0 = 10.0$) locks global logit curvature after `1.23M` tokens while leaving per-subspace gain adjustment to the bounded singular spectrum `exit_bridge.sigma` $\in [0.5, 2.5]^{512}$.
   - In addition, we equip `ModularRiemannianStiefelKrausMuon` and `BatchSubspaceTikhonovGramMuon` with a **soft gradient-norm gate** $g_{\text{mag}} = \frac{\|M_t\|_F}{\sqrt{\|M_t\|_F^2 + \tau_{\text{grad}}^2}}$ so geodesic steps remain scale-invariant when $\|M_t\|_F \gg \tau_{\text{grad}}$ (`1e-4`) and smoothly self-anneal to zero near a minimum.

2. **Step 2 — Large-Batch Gradient Accumulation (`8,192 -> 65,536 -> 262,144` Tokens/Step)**:
   - At `8,192` tokens/step (`B = 16, T = 512`), only `~2,603 / 50,304` unique tokens (`5.18%` duty cycle) appear in `active_ids` per step, causing `wte_512` to progress `2.05x` slower than the `100%`-duty-cycle trunk at `1.0x`.
   - Accumulating over $K_{\text{accum}} \in \{1, 8, 32\}$ micro-steps (`8,192`, `65,536`, and `262,144 = 2^18` tokens/step, matching an `8x H100` speedrun batch) jumps the per-step vocabulary duty cycle from `5.18%` to **`~20%` (`65K`) and `~38%–42%` (`262K`)**, co-orthogonalizing `~20,000` active token rows together in a single $512 \times 512$ Newton-Schulz Gram step and raising gradient SNR by $\sqrt{32} \approx 5.66\times$.

3. **Step 3 — Rebalancing Trunk vs. Readout Compute (`L = 4 -> L = 8`, `52.1%` Trunk FLOPs), Orthonormal Two-Basis $U(16)$ MUB Tomography, & Rank-2 CPTP Stiefel Kraus Channels ($\operatorname{St}_{\mathbb{C}}(32, 16)$)**:
   - **Depth Scaling (`L = 4 -> L = 8`)**: At $L = 4$, the quantum trunk performs only `35.2%` of total FLOPs (`85.7 MFLOPs` trunk vs. `157.8 MFLOPs` vocabulary readout). Scaling to **$L = 8$ layers** brings the trunk to **`52.1%` of total FLOPs** (`171.4 MFLOPs` trunk vs. `157.8 MFLOPs` readout) while increasing step time by only `~1.35x`.
   - **Orthonormal Two-Basis Probe Frame (`2 × U(16) = 32` Mutually Unbiased Bases per Head)**: Instead of 32 independent $\mathbb{S}^{31}$ probe vectors that can collapse onto the dominant eigenvector of $\rho_{t,h}$, we group each head's 32 probes into **two $16 \times 16$ unitary matrices** $C_h^{(1)}, C_h^{(2)} \in U(16)$ initialized at Step 0 as a Haar unitary $C_h^{(1)}$ and its Discrete Fourier Transform Mutually Unbiased Basis $C_h^{(2)} = C_h^{(1)} F_{16}$ ($|\langle c_{j,h}^{(1)} \mid c_{k,h}^{(2)} \rangle|^2 = \frac{1}{16}$) and updated on $U(16)$ via `ModularRiemannianStiefelKrausMuon`. Each 16-probe bank forms an exact resolution of the identity ($\sum_{k=1}^{16} |c_{k,h}^{(m)}\rangle\langle c_{k,h}^{(m)}| = I_{16}$), guaranteeing zero intra-basis collinearity.
   - **Rank-$R = 2$ CPTP Stiefel Kraus Channels ($\mathcal{E}_h(\rho) = \sum_{r=1}^2 K_{h,r} \rho K_{h,r}^\dagger$, $\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(32, 16)$)**: Stacking $R = 2$ Kraus operators vertically into $\mathbf{K}_h = \begin{bmatrix} K_{h,1} \\ K_{h,2} \end{bmatrix} \in \mathbb{C}^{32 \times 16}$, the complex Stiefel constraint $\mathbf{K}_h^\dagger \mathbf{K}_h = K_{h,1}^\dagger K_{h,1} + K_{h,2}^\dagger K_{h,2} = I_{16}$ is identical to the CPTP trace-preservation law, enabling non-isospectral quantum channels that actively purify or mix $\rho_{t,h}$.

4. **Step 4 — Closed-Loop Hot-vs-Shadow-EMA Riemannian Thermostat (`RiemannianShadowEMAThermostat`)**:
   - Automates our Step-5 thermal-gap discovery by tracking a fast `20`-step (`ema_decay = 0.95`) ambient lerp shadow state $\tilde{\theta}_{\text{ema}}$, retracting it onto $\mathbb{S}^{511}(\sqrt{512}) \times \operatorname{St}(n, p) \times \operatorname{St}_{\mathbb{C}}(32, 16) \times U(16)$ at evaluation checkpoints, and whenever $\Delta L_{\text{thermal}} = L_{\text{val}}(\theta_{\text{hot}}) - L_{\text{val}}(\bar{\theta}_{\text{ema}}) > +0.02$ nats, banking $\theta_{\text{hot}} \leftarrow \bar{\theta}_{\text{ema}}$ and scaling the learning rate by $\max(0.50, 1 / \sqrt{1 + 8.0\,\Delta L_{\text{thermal}}})$.

---

## 2. Quantitative Hypotheses (`H1–H4`)

- **`H1` (Step 1 — Analytical Parseval Temperature $\beta_0 = 10.0$ & Magnitude-Aware Muon Gate $g_{\text{mag}}$)**:
  - Replacing the trainable scalar `readout_temp` with the deterministic analytical ramp $\beta_0(t)$ (`0.50 -> 4.0 -> 10.0` over `1.23M` tokens, zero trainable global temperature parameters) preserves the exact Step-0 Zero-PPMI BPE-rank Zipf prior CE (`9.1473`) while eliminating post-warmup $\beta^2$ curvature drift (`0.0%` drift after Step 150 vs. `+406%` curvature drift in Wave 63).
  - On the `300`-step (`240` plateau + `60` quench) duty-matched benchmark (`trunk=1.5x, vocab=2.5x, ngram=1.5x, scalar=0.75x`), fixing $\beta_0(t)$ analytically removes the `scalar_mult = 0.75x` temperature-lag penalty and improves Post-Quench Val CE over Wave 63's `PROBE_4_DUTY_MATCHED` (`6.3047`) while maintaining high quantum density rank ($\text{RhoRank} \ge 3.0 / 16$).
- **`H2` (Step 2 — Large-Batch Gradient Accumulation `8,192 -> 65,536 -> 262,144` Tokens/Step)**:
  - Increasing effective batch size via gradient accumulation ($K_{\text{accum}} \in \{1, 8, 32\}$) increases per-step active vocabulary duty cycle from `~5.18%` (`8,192` tok/step) to **`>= 18%` at `65,536` tok/step** and **`>= 35%` at `262,144` tok/step**, reducing the `1.0x` trunk-to-vocab geodesic speed imbalance from `2.05x` (`K=1`) toward `1.0x` (`K=32`) and shifting Lookahead Step Alignment Cosines $\cos(\Delta W_{t-1}, -g_t)$ upward (confirming that larger batches support `2.0x–3.0x` larger geodesic step sizes).
- **`H3` (Step 3 — Orthonormal Two-Basis $U(16)$ MUB Probes, Rank-2 Stiefel Kraus Channels $\operatorname{St}_{\mathbb{C}}(32, 16)$, & `L = 8` Depth Scaling)**:
  - **Two-Basis $U(16)$ MUB Probes**: Enforces exact intra-basis probe orthogonality ($\|C_h^{(m)} (C_h^{(m)})^\dagger - I_{16}\|_F / \sqrt{16} < 10^{-3}$ throughout training vs. unconstrained $\mathbb{S}^{31}$ collinearity drift) and exact Step-0 mutually unbiased cross-basis overlap ($|\langle c_{j,h}^{(1)} \mid c_{k,h}^{(2)} \rangle|^2 = 1/16 = 0.0625$ to `< 1e-6`).
  - **Rank-2 CPTP Stiefel Kraus Channel**: Enforces exact CPTP trace preservation ($\|\sum_{r=1}^2 K_{h,r}^\dagger K_{h,r} - I_{16}\|_F / \sqrt{16} < 10^{-3}$ and $|\operatorname{Tr}(\mathcal{E}_h(\rho)) - 1| < 10^{-5}$).
  - **300-Step Controlled Ablation (`W64_A0..A3`)**: Scaling from `L = 4` (`35.2%` trunk FLOPs) to `L = 8` (`52.1%` trunk FLOPs) with Orthonormal Two-Basis $U(16)$ Probes and Rank-2 Stiefel Kraus Channels improves 300-step Post-Quench Validation CE below Wave 63's best `6.2885` while keeping step-time increase `<= 1.45x`.
- **`H4` (Step 4 — Closed-Loop Hot-vs-Shadow-EMA Thermostat Verification)**:
  - On a multi-checkpoint trajectory with deliberate initial hot step multipliers (`2.0x`), `RiemannianShadowEMAThermostat` automatically detects positive thermal gaps ($\Delta L_{\text{thermal}} = L_{\text{hot}} - L_{\text{ema}} > 0.02$ nats), banks the manifold-retracted Shadow-EMA state ($\theta_{\text{hot}} \leftarrow \bar{\theta}_{\text{ema}}$), and steps down the learning rate via $\eta \leftarrow \eta / \sqrt{1 + 8.0\,\Delta L_{\text{thermal}}}$, beating un-thermostatted constant-LR training.

---

## 3. Mathematical Formulation

### 3.1 Analytical Parseval Temperature Schedule $\beta_0(t)$ & Magnitude-Aware Muon Gate $g_{\text{mag}}$
1. **Analytical Logit Temperature $\beta_0(\text{tokens})$**:
   Instead of a free scalar parameter `readout_temp`, the inverse temperature entering `apply_unified_512d_vacuum_sphere_norm` is a deterministic function of cumulative training tokens $N_{\text{tok}}$:
   $$\beta_0(N_{\text{tok}}) = \begin{cases} \beta_{\text{init}} + (\beta_{\text{mid}} - \beta_{\text{init}})\,\dfrac{N_{\text{tok}}}{N_{\text{fast}}}, & 0 \le N_{\text{tok}} < N_{\text{fast}} \\[6pt] \beta_{\text{mid}} + (\beta_{\text{target}} - \beta_{\text{mid}})\,\dfrac{N_{\text{tok}} - N_{\text{fast}}}{N_{\text{warmup}} - N_{\text{fast}}}, & N_{\text{fast}} \le N_{\text{tok}} < N_{\text{warmup}} \\[6pt] \beta_{\text{target}}, & N_{\text{tok}} \ge N_{\text{warmup}} \end{cases}$$
   with $\beta_{\text{init}} = 0.50$, $\beta_{\text{mid}} = 4.0$, $\beta_{\text{target}} = 10.0$, $N_{\text{fast}} = 163,840$ (`20` steps at `8K` batch), and $N_{\text{warmup}} = 1,228,800$ (`150` steps at `8K` batch). For $N_{\text{tok}} \ge 1.23\text{M}$, $\beta_0 \equiv 10.0$ is strictly constant.
2. **Soft Gradient-Norm Magnitude Gate ($g_{\text{mag}}$)**:
   Given tangent momentum $M_{\text{tan}}$ with Frobenius/row-L2 norm $\|M_{\text{tan}}\|$ and threshold $\tau_{\text{grad}} = 10^{-4}$ (`DEFAULT_MUON_GRAD_NORM_GATE_TAU`), the geodesic step size is multiplied by:
   $$g_{\text{mag}} = \frac{\|M_{\text{tan}}\|}{\sqrt{\|M_{\text{tan}}\|^2 + \tau_{\text{grad}}^2}} \;\in\; [0, 1)$$

### 3.2 Orthonormal Two-Basis $U(16)$ MUB Quantum State Tomography
Each head $h \in \{0, \dots, 15\}$ parameterizes its $K = 32$ tomographic probe kets as two $16 \times 16$ unitary matrices $C_h^{(1)}, C_h^{(2)} \in U(16)$:
1. **Step-0 Mutually Unbiased Initialization**:
   Let $C_h^{(1)} \in U(16)$ be a random Haar unitary and let $(F_{16})_{a,b} = \frac{1}{4}\exp(2\pi i a b / 16)$ be the $16 \times 16$ Discrete Fourier Transform unitary. Setting $C_h^{(2)} = C_h^{(1)} F_{16} \in U(16)$ guarantees:
   $$C_h^{(m)} (C_h^{(m)})^\dagger = I_{16} \quad (m \in \{1, 2\}), \qquad \big|\langle c_{j,h}^{(1)} \mid c_{k,h}^{(2)} \rangle\big|^2 = \frac{1}{16} \quad \forall\, j, k \in \{0, \dots, 15\}$$
2. **Trace Resolution Identity**:
   Because each 16-probe bank sums to the identity ($\sum_{k=1}^{16} |c_{k,h}^{(m)}\rangle\langle c_{k,h}^{(m)}| = I_{16}$), the sum of all 32 probe expectations in head $h$ equals $2\operatorname{Tr}(\rho_{t,h}) = 2$, preventing probe-frame collapse onto a single dominant eigenspace.

### 3.3 Rank-$R = 2$ CPTP Stiefel Kraus Channels ($\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(32, 16)$)
In each highway block $\ell \in \{0, \dots, L-1\}$, head $h$ evolves its damped density matrix $\rho_{\text{damped},t,h} \in \operatorname{Herm}(\mathbb{C}^{16})$ via a Rank-$R = 2$ CPTP Kraus operator stacked as $\mathbf{K}_h = \begin{bmatrix} K_{h,1} \\ K_{h,2} \end{bmatrix} \in \operatorname{St}_{\mathbb{C}}(32, 16)$:
$$\mathcal{E}_h(\rho_{\text{damped},t,h}) = K_{h,1}\,\rho_{\text{damped},t,h}\,K_{h,1}^\dagger + K_{h,2}\,\rho_{\text{damped},t,h}\,K_{h,2}^\dagger, \qquad \mathbf{K}_h^\dagger \mathbf{K}_h = K_{h,1}^\dagger K_{h,1} + K_{h,2}^\dagger K_{h,2} = I_{16}$$

### 3.4 Closed-Loop Hot-vs-Shadow-EMA Riemannian Thermostat (`RiemannianShadowEMAThermostat`)
At every step $t$, $\tilde{\theta}_{\text{ema}}^{(t)} = (1 - \alpha_{\text{ema}})\tilde{\theta}_{\text{ema}}^{(t-1)} + \alpha_{\text{ema}}\theta_{\text{hot}}^{(t)}$ ($\alpha_{\text{ema}} = 0.05$). At thermostat checkpoints, $\bar{\theta}_{\text{ema}} = \operatorname{Retract}(\tilde{\theta}_{\text{ema}})$ is evaluated on the validation set to measure $\Delta L_{\text{thermal}} = L_{\text{val}}(\theta_{\text{hot}}) - L_{\text{val}}(\bar{\theta}_{\text{ema}})$. Whenever $\Delta L_{\text{thermal}} > \tau_{\text{thermal}}$ (`0.02` nats), we copy $\theta_{\text{hot}} \leftarrow \bar{\theta}_{\text{ema}}$ and multiply the optimizer LR scale by:
$$\gamma_{\text{thermostat}} = \max\!\left(0.50,\; \frac{1}{\sqrt{1 + \kappa_{\text{therm}} \cdot \max(0,\, \Delta L_{\text{thermal}})}}\right), \qquad \kappa_{\text{therm}} = 8.0$$

---

## 4. Planned Changes & Testing Architecture

1. **Riemannian Optimizer Upgrades (Magnitude Gate & Multi-Shape Complex Operator Bucketing)**:
   - [`src/qdlblocks/layers/quantum_statmech_channels.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/quantum_statmech_channels.py): Add `DEFAULT_MUON_GRAD_NORM_GATE_TAU` and soft Frobenius magnitude gating $g_{\text{mag}} = \|M_{\text{tan}}\|_F / \sqrt{\|M_{\text{tan}}\|_F^2 + \tau_{\text{grad}}^2}$ inside `riemannian_stiefel_tangent_transport_muon_step` and `riemannian_complex_tangent_transport_muon_step`.
   - [`src/qdlblocks/models/modular_quantum_statmech_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/modular_quantum_statmech_gpt.py): Upgrade `ModularRiemannianStiefelKrausMuon` to accept `grad_norm_gate_tau: float = 0.0` and bucket `complex_operator_pairs` by `(u_re.size(-2), u_re.size(-1))` so `(16, 32, 16)` Rank-2 Kraus Stiefel operators and `(16, 16, 16)` $U(16)$ probe bases step seamlessly in the same optimizer.
   - [`src/qdlblocks/optim/vocab_manifold_muon.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py): Add `grad_norm_gate_tau: float = 0.0` to `BatchSubspaceTikhonovGramMuon` (`step_active` and `_step_full_single_param`).
2. **New Flag-Free Wave 64 Architecture & Thermostat Lab ([`src/qdlblocks/benchmarks/kraus_mub_thermostat_8l_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/kraus_mub_thermostat_8l_lab.py))**:
   - Implements `compute_analytical_parseval_beta`, `build_mub_two_basis_unitary_probes`, `conjugate_packed_hermitian_by_kraus_stiefel`, `compute_mub_two_basis_probe_expectations`, `KrausMUBThermostatGPT` (`L = 4` and `L = 8`), `RiemannianShadowEMAThermostat`, `run_grad_accum_duty_cycle_and_curvature_probe`, and `run_wave64_step1_to_step4_benchmarks`.
3. **First-Class CLI Scripts (`scripts/`)**:
   - [`scripts/grad_accum_262k_duty_cycle_and_lr_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/grad_accum_262k_duty_cycle_and_lr_probe.py): Executes Step 2 (`K_accum in {1, 8, 32}` $\implies$ `8,192 -> 65,536 -> 262,144` tokens/step duty-cycle & lookahead alignment probe).
   - [`scripts/kraus_mub_8l_and_thermostat_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kraus_mub_8l_and_thermostat_bench.py): Executes Step 1 (Analytical $\beta_0 = 10.0$ + Magnitude-Gated Muon), Step 3 (`L=4` vs. `2×U(16)` MUB Probes vs. `Rank-2` Stiefel Kraus vs. `L=8` Consolidated), and Step 4 (`RiemannianShadowEMAThermostat` trajectory).
4. **Verification Script & Unit Tests**:
   - [`examples/94_verify_plan084_kraus_mub_thermostat_8l.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/94_verify_plan084_kraus_mub_thermostat_8l.py)
   - [`tests/test_plan084_kraus_mub_thermostat_8l.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan084_kraus_mub_thermostat_8l.py)

---

## 5. Operational Tooling & Script Identification

### 5.1 Existing Scripts Used
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/benchmarks/kraus_mub_thermostat_8l_lab.py src/qdlblocks/optim/vocab_manifold_muon.py src/qdlblocks/layers/quantum_statmech_channels.py src/qdlblocks/models/modular_quantum_statmech_gpt.py scripts/grad_accum_262k_duty_cycle_and_lr_probe.py scripts/kraus_mub_8l_and_thermostat_bench.py examples/94_verify_plan084_kraus_mub_thermostat_8l.py tests/test_plan084_kraus_mub_thermostat_8l.py`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/verify_plan.py --plan-num 84 --json-out .cache/verify_plan084.json`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload`

### 5.2 `Build-Before-Launch` Scripts
1. [`scripts/grad_accum_262k_duty_cycle_and_lr_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/grad_accum_262k_duty_cycle_and_lr_probe.py): Profiles active vocabulary & N-gram duty cycles, geodesic progress ratios, and lookahead step alignment cosines across `8,192`, `65,536`, and `262,144` (`2^18`) effective batch sizes.
2. [`scripts/kraus_mub_8l_and_thermostat_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kraus_mub_8l_and_thermostat_bench.py): Runs the Step-1 Analytical $\beta_0 = 10.0$ + Magnitude-Gate benchmark, the Step-3 `L=8` + `2×U(16)` MUB + `Rank-2` Stiefel Kraus ablation suite, and the Step-4 Closed-Loop Shadow-EMA Thermostat run (supporting `--tokens-target-m 300.0` for the full `300M`-token run).

### 5.3 `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/mub_probe_collinearity_and_kraus_entropy_inspector.py`: Inspects the $32 \times 32$ cross-probe Gram matrix of `(probe_u1, probe_u2)` and the per-layer von Neumann entropy change $\Delta S(\rho_{t,h}) = S(\mathcal{E}_h(\rho_{t,h})) - S(\rho_{t,h})$ induced by Rank-$R$ CPTP Kraus channels across layers $\ell = 0..L-1$.

---

## 6. Constants Ledger & Empirical Results (`Wave 64`)

### 6.1 Rule 7 Named Constants Ledger (`src/qdlblocks/benchmarks/kraus_mub_thermostat_8l_lab.py`)

| Constant Name | Value | Mathematical Role & Rationale |
| :--- | :---: | :--- |
| `DEFAULT_ANALYTICAL_BETA_INIT` | `0.50` | Step-0 inverse temperature $\beta_{\text{init}}$ preserving exact BPE-rank Zipf prior calibration (`9.1473` CE) |
| `DEFAULT_ANALYTICAL_BETA_MID` | `4.00` | Intermediate inverse temperature reached at $N_{\text{fast}} = 163,840$ tokens (`20` steps @ `8K`) |
| `DEFAULT_ANALYTICAL_BETA_TARGET` | `10.00` | Fixed analytical Parseval inverse temperature $\beta_0 = 10.0$ locked for all $N_{\text{tok}} \ge 1,228,800$ |
| `DEFAULT_ANALYTICAL_BETA_FAST_TOKENS` | `163840` | Fast initial warmup token horizon (`20 × 8,192` tokens) |
| `DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS` | `1228800` | Full temperature ramp horizon (`150 × 8,192` tokens) after which $\beta_0 \equiv 10.0$ |
| `DEFAULT_MUON_GRAD_NORM_GATE_TAU` | `1e-4` | Soft gradient-norm threshold $\tau_{\text{grad}}$ in $g_{\text{mag}} = \|M\|_F / \sqrt{\|M\|_F^2 + \tau_{\text{grad}}^2}$ |
| `DEFAULT_KRAUS_BRANCH_RANK` | `2` | Number of CPTP Kraus operators per head ($R = 2$, $\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(32, 16)$) |
| `DEFAULT_KRAUS_PRIMARY_WEIGHT` | `0.88` | Step-0 dominant branch weight $w_1 = \sqrt{0.88}$ in $\mathbf{K}_h = [K_{h,1}; K_{h,2}] \in \operatorname{St}_{\mathbb{C}}(32, 16)$ |
| `DEFAULT_MUB_UNITARY_BANKS` | `2` | Number of $U(16)$ orthonormal probe bases per head ($2 \times 16 = 32$ probes/head) |
| `DEFAULT_REBALANCED_TRUNK_LAYERS` | `8` | Rebalanced quantum density highway depth ($L = 8$, `52.1%` trunk FLOP share vs. `47.9%` readout) |
| `DEFAULT_THERMOSTAT_EMA_DECAY` | `0.80` | Short `4`-step thermal horizon ($\tau_{\text{lag}} = \frac{0.80}{0.20} = 4.0$ steps) eliminating secular drift lag while damping 2–5 step zig-zag oscillations |
| `DEFAULT_THERMOSTAT_GAP_THRESHOLD` | `0.015` | Thermal gap threshold $\Delta L_{\text{thermal}} > 0.015$ nats triggering $\theta_{\text{hot}} \leftarrow \bar{\theta}_{\text{ema}}$ & LR step-down |
| `DEFAULT_THERMOSTAT_GAIN_KAPPA` | `8.0` | Calibrated thermal sensitivity gain $\kappa = 8.0$ in $\eta_{\text{next}} = \eta_{\text{cur}} / \sqrt{1 + \kappa\,\Delta L_{\text{thermal}}}$ |
| `DEFAULT_THERMOSTAT_MIN_SCALE_STEP` | `0.50` | Minimum single-checkpoint LR reduction factor per thermostat trigger |
| `DEFAULT_THERMOSTAT_POLAR_STEPS` | `3` | Un-divided quadratic polar retraction steps ($W \leftarrow \frac{1}{2}W(3I - W^\top W)$) preserving exact unit singular values (`1.000000`) |
| `DEFAULT_THERMOSTAT_RETRACT_CADENCE` | `2` | Step cadence for batched un-divided Polar Stiefel/Unitary retraction (`~0.45 ms` amortized) |

### 6.2 Empirical Results (`Wave 64` — Completed)

#### A. Step 2: Large-Batch Gradient Accumulation (`8,192 -> 65,536 -> 262,144` Tokens/Step, `.cache/plan084_grad_accum_262k_probe.json`)
- **Critical Step-0 Sphere-Radius Fix Discovered During Step 2**:
  - In `Wave 63` (`ExplicitDensityTomographicGPT`), `wte_512` was initialized with unit complex norm per head (row L2 norm $\sqrt{16} = 4.0$), whereas `BatchSubspaceTikhonovGramMuon` retracts active rows onto the unit-RMS sphere $\mathbb{S}^{511}(\sqrt{512})$ (row L2 norm $\sqrt{512} \approx 22.63$), creating an artificial $5.66\times$ norm jump on a token's first active step (`wte_active_geodesic_step_rel = 2.355`, `lookahead_cos_wte = +0.0010`).
  - Scaling `wte_512` by $\sqrt{2 d_h} = \sqrt{32}$ at Step 0 in [`initialize_kraus_mub_model_`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/kraus_mub_thermostat_8l_lab.py#L690-L693) places all `50,304` rows directly on $\mathbb{S}^{511}(\sqrt{512})$, eliminating the first-visit norm jump and revealing the true duty-cycle-driven geodesic speed ratio:

| $K_{\text{accum}}$ | Effective Batch (`tok/step`) | Vocab Duty Cycle (`%`) | N-Gram Duty Cycle (`%`) | Trunk/WTE Speed Ratio | Lookahead $\cos(\Delta W_{t-1}, -g_t)$ (`Trunk / WTE / Det`) | 10-Step Val CE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1`** | `8,192` | `5.17%` (`2,601.2` tok) | `84.54%` | `60.15x` | `+0.1025 / +0.1880 / +0.4100` | `7.8472` |
| **`8`** | `65,536` | `22.41%` (`11,272.0` tok) | `100.00%` | `29.45x` | `+0.1714 / +0.2163 / +0.1997` | `7.3898` |
| **`32`** | `262,144` (`2^18`) | **`47.20%`** (`23,742.4` tok) | **`100.00%`** | **`16.60x`** | **`+0.2142 / +0.1227 / +0.0747`** | **`7.3870`** |

#### B. Steps 1 & 3: 300-Step (`2.46M` Tokens, `524,288` Val Tokens) Ablation Suite (`.cache/plan084_kraus_mub_8l_thermostat_bench.json`)

| Variant ID | Architecture & Optimizer Configuration | Step 0 CE | Step 60 CE | Step 120 CE | Step 180 CE | Step 240 CE | Step 300 CE (`PPL`) | Top-1 / Top-5 | $\rho$ Eff. Rank | Kraus / Probe Err |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *(Wave 63 Ref)* | `W63_A3` (`L=4`, free `readout_temp -> 22.5`, un-gated Muon, norm-4 `wte_512`) | `9.1473` | `7.3504` | `7.1292` | `7.0245` | `6.9602` | `6.8473` (`941.3`) | `11.27% / 24.51%` | `3.21 / 16` | `0.00e+00` |
| **`W64_A0`** | **Step 1**: Analytical $\beta_0(N_{\text{tok}}) \to 10.0$ + Mag-Gated Muon (`tau=1e-4`) + $\mathbb{S}^{511}(\sqrt{512})$ `wte_512` (`L=4, R=1`) | `9.0834` | `7.1470` | `6.9181` | `6.8607` | `6.7389` | **`6.5725`** (`715.2`) | `15.47% / 30.18%` | `2.18 / 16` | `0.00e+00 / 0.00e+00` |
| **`W64_A1`** | **Step 3a**: `W64_A0` + Orthonormal Two-Basis $2 \times U(16)$ DFT MUB Tomographic Probes (`L=4, R=1`) | `9.0834` | `7.1309` | `6.9157` | `6.8652` | `6.7498` | **`6.5790`** (`719.8`) | `15.43% / 30.08%` | `2.26 / 16` | `0.00e+00 / 0.00e+00` |
| **`W64_A2`** | **Step 3b**: `W64_A1` + Rank-$R=2$ CPTP Stiefel Kraus Channels $\operatorname{St}_{\mathbb{C}}(32, 16)$ (`L=4, R=2`) | `9.0834` | `7.1319` | `6.9157` | `6.8645` | `6.7486` | **`6.5811`** (`721.3`) | `15.43% / 30.04%` | `2.18 / 16` | `0.00e+00 / 0.00e+00` |
| **`W64_A3`** | **Steps 3c & 4**: Rebalanced **`L = 8`** (`52.1%` Trunk FLOPs) + Rank-$R=2$ Kraus + $2 \times U(16)$ MUB + Thermostat | `9.0834` | **`7.1236`** | **`6.9162`** | **`6.8553`** | **`6.7363`** | **`6.5591`** (`705.6`) | **`15.70% / 30.22%`** | `2.20 / 16` | `0.00e+00 / 0.00e+00` |

#### C. Step 4: Fixed Two-Timescale Riemannian Polar EMA Thermostat Verification (`.cache/plan084_w64_a3_fixed_ema_verify.json`)
- **Three Geometric Bugs Fixed in [`RiemannianShadowEMAThermostat`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/kraus_mub_thermostat_8l_lab.py#L961-L1070)**:
  1. **Un-Divided 3-Step Quadratic Polar Retraction (`retract_real_stiefel_polar_tensor`, `retract_complex_stiefel_polar_pair`)**: `_newton_schulz5_tall_real` divided $512 \times 512$ Stiefel matrices by $\|W\|_F = \sqrt{512} \approx 22.63$, shrinking singular values to `0.3186` (`ortho_err = 0.6814`). Because any convex combination of Stiefel/Unitary matrices satisfies $\|W\|_2 \le 1.0 < \sqrt{3}$ by operator-norm convexity, 3 un-divided steps of $W \leftarrow \frac{1}{2}W(3I - W^\top W)$ converge quadratically to singular values `1.000000` (`ortho_err < 1e-6`) in `< 0.45 ms` amortized per step.
  2. **Two-Timescale Sparse vs. Dense Smoothing (`_collect_fast_sparse_param_ids`)**: Zero-initialized N-gram tables (`tok_ngram_in`, `tok_ngram_out`, `tok_unigram_out`) use soft-zero regularized RMSNorm `ng * rsqrt(ng.square().mean() + 0.0625)`. Damping newly visited buckets by slow EMA (`alpha = 0.05`) shrank `ng.square().mean()` by `400x` below `0.0625`, attenuating N-gram readout amplitude in `ema_model`. Fast synchronization (`alpha_fast >= 0.55`) on sparse/zero-init tables eliminates this attenuation.
  3. **Short Thermal Horizon (`DEFAULT_THERMOSTAT_EMA_DECAY = 0.80`, $\tau_{\text{lag}} = 4.0$ steps) + EMA-Synchronized `tokens_seen_buf`**: Aligns `ema_model`'s analytical Parseval temperature $\beta_0(N_{\text{tok}})$ with the exact effective age of its weights.
- **120-Step `W64_A3` Verification & 90-Step `2.20x` Hot Stress Test (`.cache/plan084_w64_a3_fixed_ema_verify.json`)**:
  - On `W64_A3_FULL_CONSOLIDATED_8L_THERMOSTAT` (`120` steps), the fixed Shadow EMA beat the hot model at Step 60 (`EMA CE = 7.2085` vs `Hot CE = 7.2681`, `gap = +0.0596 nats`, `Trig: True`, `lr: 1.00x -> 0.82x`) and Step 90 (`EMA CE = 7.0767` vs `Hot CE = 7.0946`, `gap = +0.0179 nats`, `Trig: True`, `lr: 0.82x -> 0.77x`), finishing at **`7.0078` Val CE (`12.30%` Top-1)** vs `7.0222` (`12.05%` Top-1) un-triggered.
  - On the `2.20x` Hot Stress Test (`90` steps), the thermostat triggered at Steps 44, 66, and 88 (`LR: 2.200x -> 2.068x -> 1.735x -> 1.621x`), finishing at **`7.0873` Active CE** vs **`7.1462`** un-thermostatted (**`-0.0589` nats improvement**).


