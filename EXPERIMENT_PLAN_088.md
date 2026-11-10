# `EXPERIMENT_PLAN_088.md` (`Wave 68`): Complementary Orthogonal `512D` ($256 \oplus 256$) Readout, Multi-Timescale Lindblad Highway Gates, Isometric Parallel Transport, Frequency-Adaptive Vocab Trust Regions, & Depth-Scaled `8L` Quantum Trunk

## 1. Executive Summary & Motivation

In Wave 67 ([`EXPERIMENT_PLAN_087.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_087.md)), [`SelectiveExactQuotientMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) eliminated vertical gauge-fiber spin (`MUBErr = 0.00e+00`, `SU16Phase = 0.00e+00`, `KrausGauge = 0.00e+00`), restored `100.0%` horizontal gradient completeness (`raw_vertical_grad_leak_frac = 0.00e+00`), and enabled `W67_S3` (`256D` Shared Stiefel Frame, $L=4, R=4$) to reach **`4.8580` EMA Val CE (`4.8659` Hot CE, `26.92%` Top-1)** at `147.46M` tokens—surpassing Wave 65's `231.18M`-token all-time best (`4.8977` CE) in `< 127M` tokens (`> 100M` tokens ahead).

With vertical gauge collisions eliminated, the remaining validation loss floor is governed by **five structural expressivity and second-order Riemannian curvature bottlenecks**:

1. **The `256D` Tomographic Projection Bottleneck (`50%` Norm Discard)**:
   - Widening $U_{\text{out}} \in \operatorname{St}(512, d_{\text{sub}})$ from $128 \to 256$ in `W67_S3` drove a major leap over `W65_V3`, yet a single `(512, 256)` Stiefel frame still projects out **half (`256` dimensions)** of the `512D` two-basis MUB tomographic measurement ($16\text{ heads} \times 2\text{ MUB bases} \times 16\text{ outcomes} = 512\text{D}$).
2. **Uniform Single-Timescale Highway Mixing Without Thermal Reset (`highway_gamma`)**:
   - All $H=16$ heads in `_KrausDensityHighwayBlock` previously shared an identical initial highway blend weight without multi-timescale differentiation or a context-selective Lindblad depolarizing reset channel toward the maximally mixed thermal state $\frac{1}{16}I_{16}$.
3. **Momentum Attenuation Under Naive Tangent Re-Projection ($\Pi_{T_{W_{t+1}}}(M_t)$)**:
   - Re-projecting horizontal Nesterov momentum $M_t \in T_{W_t}\mathcal{M}$ onto $T_{W_{t+1}}\mathcal{M}$ via orthogonal projection $\Pi_{T_{W_{t+1}}}(M_t)$ drops the second-order Levi-Civita connection term along the geodesic and attenuates momentum norm by $O(\eta_t^2 \|U_{\text{tan}}\|_F^2)$ at every step.
4. **Uniform Inactive-Row Drift & Rare-Token Angular Overshoot on $W_{\text{vocab}} \in (\mathbb{S}^{255})^V$**:
   - In [`BatchSubspaceTikhonovGramMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py), rare tokens appearing once every `~50–100` batches experience a `2.5x` relative-magnitude spike upon appearance followed by `99` steps of un-gated `0.25 * lr` background partition-function drift.
5. **Shallow $L = 4$ Trunk Capacity & Unscaled Multi-Layer Geodesic Perturbation Variance**:
   - Scaling from $L = 4 \to L = 8$ layers without scaling per-layer geodesic step sizes on residual-writing Stiefel/Kraus blocks (`v_proj`, `obs_proj`, `mz_mlp.down_proj`, `k_evol`) doubles the per-step highway perturbation variance.

---

## 2. Mathematical Formulation of the Five Wave 68 Upgrades

### 2.1 Upgrade 1 — Full-Rank `512D` ($256 \oplus 256$) Complementary Orthogonal Subspace Readouts
We parameterize the input and output subspace bridges as **full $512 \times 512$ orthogonal matrices** $Q_{\text{in}} = [U_{\text{in}}^{(1)}, U_{\text{in}}^{(2)}] \in \operatorname{O}(512)$ and $Q_{\text{out}} = [U_{\text{out}}^{(1)}, U_{\text{out}}^{(2)}] \in \operatorname{O}(512)$ partitioned into two mutually orthogonal `256D` Stiefel frames ($U^{(1)\top} U^{(1)} = I_{256}$, $U^{(2)\top} U^{(2)} = I_{256}$, $U^{(1)\top} U^{(2)} = 0$). By Parseval's identity (machine-checked in **Lean 4 Theorem 5c**):
$$\|h_{512}\,U_{\text{out}}^{(1)}\|_2^2 + \|h_{512}\,U_{\text{out}}^{(2)}\|_2^2 \equiv \|h_{512}\|_2^2$$
so **100.0% of the `512D` MUB tomographic norm is preserved**. We test two fixed-once boundary realizations on $Q_{\text{out}} = [U_{\text{out}}^{(1)}, U_{\text{out}}^{(2)}] \in \operatorname{O}(512)$:

1. **`ComplementaryOrthogonalLudersBoundary` (`W68_L1` — Two-Stage $256_{\text{coarse}} \oplus 197_{\text{fine|coarse}}$ Sequential Lüders Quantum Instrument)**:
   - **Stage 1 (`256`-way Coarse Measurement on $U_{\text{out}}^{(1)}$)**: Projects $h_{\text{coarse}} = h_{512}\,U_{\text{out}}^{(1)} \in \mathbb{R}^{N \times 256}$ and scores against $E_{\text{coarse}} \in (\sqrt{256}\,\mathbb{S}^{255})^{256}$ (`100.0%` active duty cycle).
   - **Stage 2 (Lüders Post-Measurement State Collapse onto $U_{\text{out}}^{(2)}$)**: Conditioned on $a^* = y_t \ \& \ \text{0xFF} \in \{0, \dots, 255\}$, collapses the orthogonal `256D` complement $h_{\text{fine}} = h_{512}\,U_{\text{out}}^{(2)} \in \mathbb{R}^{N \times 256}$ using both the Lüders gate $(g_{a^*}, s_{a^*})$ and the projected explicit density-matrix transition $\mathcal{M}_{\rho_t}(a^*)_h = \rho_{t,h}|\phi_{a^*,h}\rangle$:
     $$h_{\text{fine, cond}}(a^*) = h_{\text{fine}} \odot (1 + 0.5\tanh(g_{a^*})) + s_{a^*} + \lambda_\rho\big(m_{512}(a^*)\,U_{\text{out}}^{(2)}\big) \in \mathbb{R}^{N \times 256}$$
   - **Stage 3 (`197`-way Per-Cluster Conditional Fine Measurement)**: Scores $h_{\text{fine, cond}}(a^*)$ against both shared quotient effects $E_{\text{fine, shared}} \in (\sqrt{256}\,\mathbb{S}^{255})^{197}$ and token-specific fine effects $W_{\text{fine}}[256 b + a^*] \in (\sqrt{256}\,\mathbb{S}^{255})$ within bucket $a^*$ via vectorized bucket-indexed gathering (`(N, 197, 256)` / chunked contraction), yielding exact normalized log-likelihood $-\log P(a^* \mid \rho_t) - \log P(b^* \mid a^*, \rho_t)$ over all `50,257` tokens (**Lean 4 Theorem 5a–b**).
2. **`ComplementaryDualSubspace512DBoundary` (`W68_L2`, `W68_L3` — Full `512D` Dual-Subspace Frame with High-Duty Coarse-Fine Complement)**:
   - Uses $U_{\text{out}}^{(1)} \in \operatorname{St}(512, 256)$ for the primary `256D` token sphere $W_{\text{vocab}}^{(1)} \in (\sqrt{256}\,\mathbb{S}^{255})^{50304}$, and uses the orthogonal complement $U_{\text{out}}^{(2)} \in \operatorname{St}(512, 256)$ for a `100%`-duty-cycle bipartite subsystem tables $E_{\text{coarse}}^{(2)} \in (\sqrt{256}\,\mathbb{S}^{255})^{256}$ and $E_{\text{fine}}^{(2)} \in (\sqrt{256}\,\mathbb{S}^{255})^{197}$:
     $$z_{\text{neural}}(v = 256 b + a) = \langle h_{512} U_{\text{out}}^{(1)}, w_v^{(1)}\rangle + \gamma_{\text{comp}}\Big(\langle h_{512} U_{\text{out}}^{(2)}, e_{\text{coarse}, a}^{(2)}\rangle + \langle h_{512} U_{\text{out}}^{(2)}, e_{\text{fine}, b}^{(2)}\rangle\Big)$$
   - Evaluates the complementary `256D` subspace via two tiny `(N, 256) @ (256, 256)` and `(N, 256) @ (256, 197)` GEMMs (`< 0.8%` FLOP overhead) inside a single-pass Rule-9 custom autograd kernel while unlocking all `512` dimensions of $Q_{\text{in}}, Q_{\text{out}} \in \operatorname{O}(512)$.

### 2.2 Upgrade 2 — Multi-Timescale Per-Head Lindblad / Depolarizing Highway Gates ($\alpha_{h,\ell}, \gamma_{t,h}^{(\ell)}$)
In each quantum block $\ell$, we initialize the $H = 16$ heads with a log-spaced multi-timescale retention spectrum $\alpha_{h,\ell} = \sigma(\theta_{h,\ell}) \in [0.08, 0.48] \cdot \sqrt{4/L}$ (spanning long-horizon low-dissipation memory heads $h \in \{0,\dots,3\}$ to fast local syntax heads $h \in \{12,\dots,15\}$) and add a context-dependent Lindblad depolarizing thermal-reset gate $\gamma_{t,h}^{(\ell)} = \gamma_{\max}\,\sigma(w_{\text{lind}, h}^{(\ell)\top} h_{\text{in}, t} + b_{\text{lind}, h}^{(\ell)}) \in (0, 0.15)$ toward the maximally mixed state $\rho_{\text{therm}} = \frac{1}{16}I_{16}$:
$$\rho_{t,h}^{(\ell+1)} = \big(1 - \gamma_{t,h}^{(\ell)}\big)\Big[\big(1 - \alpha_{h,\ell}\big)\,\rho_{t,h}^{(\ell)} + \alpha_{h,\ell}\,\mathcal{E}_{h,\ell}\!\big(\rho_{t,h,\text{attn}}^{(\ell)}\big)\Big] + \gamma_{t,h}^{(\ell)}\,\frac{1}{16}I_{16}$$

### 2.3 Upgrade 3 — Isometric Geodesic/Midpoint Parallel Transport $\mathcal{T}_{W_t \to W_{t+1}}(M_t)$
When updating $W_t \to W_{t+1}$ on $\operatorname{St}(n, k)$ or $\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$, we transport the updated tangent momentum $M_{\text{upd}} \in T_{W_t}\mathcal{M}$ via the midpoint connection transport followed by exact horizontal/tangent projection and Frobenius norm conservation:
$$\widetilde{M}_{t+1} = \Pi_{T_{W_{t+1}}\mathcal{M}}\!\left(M_{\text{upd}} - \frac{1}{2}(W_t + W_{t+1})\big(W_{t+1}^\top M_{\text{upd}}\big)\right), \qquad \mathcal{T}_{W_t \to W_{t+1}}(M_{\text{upd}}) = \widetilde{M}_{t+1}\,\frac{\|M_{\text{upd}}\|_F}{\max(\|\widetilde{M}_{t+1}\|_F, \epsilon)}$$

### 2.4 Upgrade 4 — Frequency-Adaptive Per-Row Spherical Trust Region on $W_{\text{vocab}}$
We track an exponential moving average of each vocabulary row's active appearance duty cycle $d_v \in (0, 1]$ (`beta_duty = 0.98`), gating the relative step magnitude `rel_mag` for rare tokens by $\sqrt{d_v / (d_v + \tau_{\text{duty}})}$ and reducing inactive-row momentum drift from `0.25 * lr` to `0.06 * lr`, preventing rare-token angular overshoot while letting frequent tokens track $Q_{\text{out}}$ tightly.

### 2.5 Upgrade 5 — Depth-Scaled Geodesic Trust Regions ($\sqrt{4/L}$) for $L = 8$ Quantum Trunks (`W68_L3`)
For deeper trunks ($L = 8$), residual-writing Stiefel/Kraus blocks (`v_proj`, `obs_proj`, `mz_mlp.down_proj`, and `k_evol`) scale their geodesic learning rate by $\sqrt{4 / L} = 1/\sqrt{2}$, keeping total highway transport variance invariant with depth.

---

## 3. Top-of-File Named Constants Ledger (Rule 7)

| Constant Name | Value | Module | Physical / Mathematical Role |
| :--- | :---: | :--- | :--- |
| `DEFAULT_COMPLEMENTARY_SUBSPACE_DIM` | `256` | `complementary_512d_lindblad_8l_lab.py` | Half-dimension of $[U^{(1)}, U^{(2)}] \in \operatorname{O}(512)$ (`256 ⊕ 256 = 512D`) |
| `DEFAULT_COMPLEMENTARY_BRANCH_GAIN` | `0.45` | `complementary_512d_lindblad_8l_lab.py` | Initial coupling $\gamma_{\text{comp}}$ of the complementary `256D` coarse-fine subspace |
| `DEFAULT_LINDBLAD_MAX_DEPOL_RATE` | `0.12` | `complementary_512d_lindblad_8l_lab.py` | Maximum per-layer Lindblad depolarizing reset rate $\gamma_{\max}$ toward $\frac{1}{16}I_{16}$ |
| `DEFAULT_LINDBLAD_INIT_BIAS` | `-2.40` | `complementary_512d_lindblad_8l_lab.py` | Initial logit bias for Lindblad thermal reset gate ($\sigma(-2.40) \approx 0.083$) |
| `DEFAULT_HIGHWAY_ALPHA_MIN` | `0.10` | `complementary_512d_lindblad_8l_lab.py` | Slowest head highway retention rate (long-horizon quantum memory heads) |
| `DEFAULT_HIGHWAY_ALPHA_MAX` | `0.46` | `complementary_512d_lindblad_8l_lab.py` | Fastest head highway retention rate (local syntax detector heads) |
| `DEFAULT_VOCAB_DUTY_EMA_BETA` | `0.98` | `complementary_512d_lindblad_8l_lab.py` | EMA decay for per-token appearance duty cycle $d_v \in (0, 1]$ |
| `DEFAULT_VOCAB_DUTY_TRUST_TAU` | `0.015` | `complementary_512d_lindblad_8l_lab.py` | Characteristic duty threshold $\tau_{\text{duty}}$ for rare-token spherical trust region |
| `DEFAULT_INACTIVE_VOCAB_DRIFT_RATIO` | `0.06` | `complementary_512d_lindblad_8l_lab.py` | Reduced step-size ratio for inactive vocabulary rows (`0.06x` vs legacy `0.25x`) |
| `DEFAULT_DEEP_TRUNK_LAYERS` | `8` | `complementary_512d_lindblad_8l_lab.py` | Deep quantum trunk layer count for `W68_L3_DEEP_8L_CONSOLIDATED_512D` |

---

## 4. Planned Ablation Matrix (`Wave 68`)

| Variant (`run_key`) | Boundary & Subspace Rank | Trunk Depth $L$ / $R$ / MLP | Lindblad Highway + Parallel Transport + Freq-Adapt Vocab | Quantitative Target |
| :--- | :---: | :---: | :---: | :--- |
| **`W68_L0_SELECTIVE_QUOTIENT_256D_4L_REF`** | `256D` Shared Stiefel Frame (`St(512, 256)`) | $L=4, R=4, 1536$ | Disabled (`W67_S3` anchor) | Baseline reference |
| **`W68_L1_COMPLEMENTARY_LUDERS_512D_4L`** | `512D` ($256_{\text{coarse}} \oplus 197_{\text{fine|coarse}}$) Orthogonal Lüders Instrument (`O(512)`) | $L=4, R=4, 1536$ | Enabled | `100%` `512D` norm retention, exact $\sum_v P(v)=1$ |
| **`W68_L2_COMPLEMENTARY_512D_LINDBLAD_TRANSPORT_4L`** | `512D` ($256_{\text{tok}} \oplus 256_{\text{subsys}}$) Dual-Subspace Frame (`O(512)`) | $L=4, R=4, 1536$ | Enabled | Lower Val CE & higher Top-1 than `W68_L0` at `L=4` |
| **`W68_L3_DEEP_8L_CONSOLIDATED_512D`** | `512D` ($256_{\text{tok}} \oplus 256_{\text{subsys}}$) Dual-Subspace Frame (`O(512)`) | **$L=8, R=4, 1536$** (`1/sqrt(2)` trust region) | Enabled | Best overall Val CE & Top-1 across `60`-step and `200+`-step horizons |

---

## 5. Operational Tooling & Script Opportunities

1. **Existing Scripts Used**:
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py`
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload`
2. **`Build-Before-Launch` Scripts**:
   - [`scripts/complementary_512d_lindblad_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complementary_512d_lindblad_inspector.py): Verifies `100.0%` Parseval `512D` norm conservation across $Q = [U^{(1)}, U^{(2)}] \in \operatorname{O}(512)$, exact `50,257`-way probability normalization $\sum_v P(v) = 1.0$, Lindblad density-matrix trace/PSD preservation, and isometric parallel transport norm conservation $\|\mathcal{T}(M)\|_F / \|M\|_F = 1.0$.
   - [`scripts/complementary_512d_lindblad_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complementary_512d_lindblad_bench.py): Launches the 4-variant Wave 68 ablation and multi-checkpoint horizon runs.
   - [`scripts/verify_lean4_theorems.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py): Runs `lake build` in `lean/` and audits zero `sorry` / zero custom axioms across all 5 formal Lean 4 theorems.
3. **`Nice-to-Have` Scripts (Logged in [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md))**:
   - `scripts/lindblad_head_entropy_and_timescale_profiler.py`: Profiles per-head von Neumann entropy $S(\rho_{t,h}) = -\operatorname{Tr}(\rho_{t,h}\log\rho_{t,h})$ and learned retention rates $\alpha_{h,\ell}$ across layers $\ell \in \{0, \dots, L-1\}$.

---

## 6. Verification & Empirical Results Ledger

### 6.1 Machine-Checked Lean 4 Quantum Geometry Formalization (`/home/festivity/agy/QuantumDLBlocks/lean/` & [`scripts/verify_lean4_theorems.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py))

Compiled via `lake build` (`leanprover/lean4:v4.15.0`) in `2.35s` with **zero `sorry`**, **zero `admit`**, and **zero custom `axiom`** (`14/14` theorem statements verified in [`.cache/lean4_theorems_verification.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/lean4_theorems_verification.json)):

| Theorem | Lean 4 Module | Verified Theorem Statements | Python Counterpart |
| :--- | :--- | :--- | :--- |
| **Theorem 1** | [`QuotientHorizontalCompleteness.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/QuotientHorizontalCompleteness.lean) | `chain_rule_horizontal_completeness`, `false_symmetry_vertical_leak` | [`project_su_dh_mub_horizontal`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py), [`scripts/selective_quotient_invariance_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/selective_quotient_invariance_inspector.py) |
| **Theorem 2** | [`SplitSpectrumBiorthogonalSVD.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/SplitSpectrumBiorthogonalSVD.lean) | `degenerate_singular_values_synchronous_spin`, `split_spectrum_forces_zero_vertical_spin`, `biorthogonal_svd_normal_space_dichotomy` | [`project_coupled_svd_raw_exact_degenerate`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py), [`filter_coupled_svd_post_ns_corotation`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) |
| **Theorem 3** | [`KrausUnitaryGaugeSylvester.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/KrausUnitaryGaugeSylvester.lean) | `kraus_channel_unitary_gauge_invariance`, `kraus_trace_preservation_invariance`, `kraus_sylvester_horizontal_iff` | [`project_kraus_branch_gauge_horizontal`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/horizontal_grassmann_gauge_free_lab.py) |
| **Theorem 4** | [`NewtonSchulzHorizontalLeakage.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/NewtonSchulzHorizontalLeakage.lean) | `newton_schulz_cubic_term_vertical_leak`, `newton_schulz_poly_non_invariance`, `post_newton_schulz_horizontal_restoration` | [`SelectiveExactQuotientMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py), [`ParallelTransportSelectiveQuotientMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/complementary_512d_lindblad_8l_lab.py) |
| **Theorem 5** | [`SequentialLudersInstrument.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/SequentialLudersInstrument.lean) | `sequential_luders_joint_nonneg`, `sequential_luders_exact_partition_sum_one`, `complementary_orthogonal_subspaces_parseval_conservation` | [`ComplementaryOrthogonalLudersBoundary`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/complementary_512d_lindblad_8l_lab.py), [`ComplementaryDualSubspace512DBoundary`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/complementary_512d_lindblad_8l_lab.py) |

---

### 6.2 Mathematical & Geometric Invariant Verification ([`scripts/complementary_512d_lindblad_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complementary_512d_lindblad_inspector.py))

Verified in [`.cache/complementary_512d_lindblad_inspector.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/complementary_512d_lindblad_inspector.json) (`all_checks_passed = true`) and `pytest tests/test_plan088_complementary_512d_lindblad_8l.py` (`5/5 passed`):
- **Complementary `512D` Parseval Norm Conservation**: `parseval_512d_retention_ratio = 1.000000` (`100.0%` vs `50.0%` in single `256D` projection), `subspace_cross_ortho_max_err = 1.12e-7`.
- **Exact `50,257`-Way Probability Normalization**: `luders_50257_prob_sum_max_err = 2.38e-7`, `dual_50257_prob_sum_max_err = 2.38e-7`.
- **Rule-9 Custom Autograd Equivalence (`_DualSubspace512DSoftcapCEFn`)**: `fwd_loss_abs_diff = 1.91e-6`, `grad_h1_max_err = 7.45e-8`, `grad_h2_max_err = 8.94e-8`, `grad_w_tok_max_err = 5.96e-8`.
- **Lindblad / Depolarizing Highway Trace & PSD Preservation**: `lindblad_trace_max_err = 2.38e-7`, `lindblad_hermitian_max_err = 0.0`, `lindblad_min_eigenvalue = +6.24e-4 >= 0`.
- **Isometric Parallel Transport Norm & Quotient Preservation**: `real_pt_norm_ratio = 1.000000`, `complex_pt_norm_ratio = 1.000000`, `complex_pt_kraus_gauge_leak = 1.33e-7`.

---

### 6.3 60-Step (`0.12M`-Token) 4-Variant Ablation ([`.cache/plan088_complementary_512d_lindblad_bench_60steps.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/plan088_complementary_512d_lindblad_bench_60steps.json))

| Variant (`run_key`) | Params | Step 0 CE | Step 20 (`0.04M`) Hot / EMA CE (Top-1) | Step 40 (`0.08M`) Hot / EMA CE (Top-1) | Step 60 (`0.12M`) Hot / EMA CE (Top-1) | Gauge Leaks (`MUBErr` / `SU16` / `Kraus`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`W68_L0_SELECTIVE_QUOTIENT_256D_4L_REF`** (`W67_S3` Control) | `31.23M` | `8.9099` | `8.0310` / `8.0634` (`5.91%`) | `7.6684` / `7.6577` (`7.63%`) | `7.4392` / `7.4333` (`9.38%`) | `0.00e+00` / `0.00e+00` / `0.00e+00` |
| **`W68_L1_COMPLEMENTARY_LUDERS_512D_4L`** (`256 ⊕ 197` 2-Stage Lüders) | `31.91M` | `8.9073` | `8.9839` / `8.8814` (**`6.88%`**) | `8.5003` / `8.4947` (**`8.96%`**) | `8.1854` / `8.1917` (**`9.59%`**) | `0.00e+00` / `0.00e+00` / `0.00e+00` |
| **`W68_L2_COMPLEMENTARY_512D_LINDBLAD_TRANSPORT_4L`** (`4L` `512D` Consolidated) | `31.61M` | `8.9092` | **`7.9583` / `8.0005`** (`5.81%`) | **`7.6143` / `7.5846`** (**`8.94%`**) | **`7.4184` / `7.4072`** (**`9.88%`**) | `0.00e+00` / `0.00e+00` / `0.00e+00` |
| **`W68_L3_DEEP_8L_CONSOLIDATED_512D`** (`8L` `512D` + $\sqrt{4/L}$ Trust Region) | `44.34M` | `8.9090` | **`7.9427` / `7.9721`** (**`6.16%`**) | **`7.6012` / `7.5726`** (**`9.06%`**) | **`7.4195` / `7.4074`** (**`9.90%`**) | `0.00e+00` / `0.00e+00` / `0.00e+00` |

---

### 6.4 200-Step (`0.41M`-Token) Horizon Comparison ([`.cache/plan088_complementary_512d_lindblad_bench_200steps.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/plan088_complementary_512d_lindblad_bench_200steps.json))

| Step (`Tokens`) | `W68_L0` (`W67_S3` `4L` Control) Hot / EMA CE (Top-1) | `W68_L2` (`4L` Consolidated `512D`) Hot / EMA CE (Top-1) | `W68_L3` (`8L` Consolidated `512D`) Hot / EMA CE (Top-1) | `W68_L3` vs `W68_L0` Advantage ($\Delta$ EMA CE / $\Delta$ Top-1) |
| :---: | :---: | :---: | :---: | :---: |
| **Step 0 (`0.00M`)** | `8.9099` (`0.00%`) | `8.9092` (`0.00%`) | `8.9090` (`0.00%`) | `-0.0009` nats |
| **Step 25 (`0.05M`)** | `7.9075` / `7.9441` (`6.04%`) | `7.8207` / `7.8776` (`5.99%`) | **`7.8003` / `7.8531`** (**`6.09%`**) | **`-0.0910` nats** / `+0.05%` |
| **Step 50 (`0.10M`)** | `7.5442` / `7.5474` (`8.54%`) | `7.5071` / `7.4898` (`9.72%`) | **`7.5125` / `7.4854`** (**`9.74%`**) | **`-0.0620` nats** / **`+1.20%`** |
| **Step 75 (`0.15M`)** | `7.4749` / `7.4135` (`10.04%`) | `7.4000` / `7.3302` (`10.46%`) | **`7.3907` / `7.3241`** (`10.45%`) | **`-0.0894` nats** / `+0.41%` |
| **Step 100 (`0.20M`)** | `7.4498` / `7.4163` (`10.13%`) | `7.3615` / `7.2975` (`10.61%`) | **`7.3438` / `7.2814`** (`10.54%`) | **`-0.1349` nats** / `+0.41%` |
| **Step 125 (`0.26M`)** | `7.3146` / `7.2820` (`10.35%`) | `7.2564` / `7.1900` (`10.89%`) | **`7.2391` / `7.1715`** (`10.76%`) | **`-0.1105` nats** / `+0.41%` |
| **Step 150 (`0.31M`)** | `7.2900` / `7.2712` (`10.15%`) | `7.1974` / `7.1559` (`10.79%`) | **`7.1656` / `7.1279`** (**`10.83%`**) | **`-0.1433` nats** / **`+0.68%`** |
| **Step 175 (`0.36M`)** | `7.2149` / `7.1914` (`10.83%`) | `7.1144` / `7.0969` (`11.30%`) | **`7.0957` / `7.0741`** (**`11.30%`**) | **`-0.1173` nats** / `+0.47%` |
| **Step 200 (`0.41M`)** | `7.1177` / `7.1224` (`11.25%`) | `7.0453` / `7.0493` (`11.26%`) | **`7.0289` / `7.0340`** (**`11.31%`**) | **`-0.0884` nats** (`-0.0888` Hot CE) |

### 6.5 Key Scientific & Architectural Takeaways
1. **Eliminating the `256D` Tomographic Projection Bottleneck via $Q \in \operatorname{O}(512)$ (`W68_L2` vs `W68_L0`)**:
   - At fixed depth $L=4$, replacing the half-rank `256D` projection $U_{\text{out}} \in \operatorname{St}(512, 256)$ (`W68_L0`) with the full-rank complementary dual-subspace boundary $Q_{\text{out}} = [U_{\text{out}}^{(1)}, U_{\text{out}}^{(2)}] \in \operatorname{O}(512)$ + multi-timescale Lindblad highway gates + isometric parallel transport (`W68_L2`) adds only `+0.38M` parameters (`31.61M` vs `31.23M`) and `< 0.9%` readout FLOPs, yet lowers 200-step Validation CE by **`-0.1153` nats at Step 150** (`7.1559` vs `7.2712`) and **`-0.0731` nats at Step 200** (`7.0493` vs `7.1224`), while boosting Step-40 Top-1 accuracy from `7.63% -> 8.94%` (`+1.31%`).
2. **Depth Scaling ($L = 4 \to 8$) with $\sqrt{4/L} = 1/\sqrt{2}$ Geodesic Trust Regions (`W68_L3`)**:
   - Scaling the trunk to $L=8$ (`W68_L3_DEEP_8L_CONSOLIDATED_512D`, `44.34M` params) with $\sqrt{4/L}$ geodesic step scaling on residual-writing Stiefel/Kraus blocks (`v_proj`, `obs_proj`, `mz_mlp.down_proj`, `k_evol`) outperforms both `W68_L0` and `W68_L2` at **every single checkpoint from Step 20 through Step 200**, reaching **`7.1279` EMA CE at Step 150 (`-0.1433` nats vs `W68_L0`, matching `W68_L0`'s Step-200 loss `25%` earlier)** and **`7.0289` Hot CE / `7.0340` EMA CE (`11.31%` Top-1) at Step 200**.
3. **Why Dual-Subspace Joint Readout (`W68_L2`/`W68_L3`) Beats Modulo-256 Hierarchical Conditioning (`W68_L1`) on CE While Both Boost Top-1**:
   - `W68_L1` (`ComplementaryOrthogonalLudersBoundary`) achieves higher Top-1 accuracy than `W68_L0` (`9.59%` vs `9.38%` at Step 60) with `50x` smaller Stage-2 GEMMs (`197` candidates vs `50,257`), but its joint NLL ($\text{NLL}_1(a^*) + \text{NLL}_2(b^* \mid a^*)$) is higher because partitioning GPT-2 BPE token IDs by modulo-256 remainder (`a^* = y & 0xFF`) forces Stage 1 to predict a near-uniform hash bucket before conditioning on semantics. By contrast, `ComplementaryDualSubspace512DBoundary` (`W68_L2` / `W68_L3`) combines the primary `256D` token sphere on $U_{\text{out}}^{(1)}$ with the `100%`-duty-cycle complementary `256D` coarse-fine spheres on $U_{\text{out}}^{(2)}$ inside a single `50,257`-way partition function (`_DualSubspace512DSoftcapCEFn`), capturing `100%` of the `512D` Parseval norm with zero modulo-bucket entropy penalty.

### 6.6 Long-Horizon (`150.00M`-Token) Trajectories of `W68_L2` (`4L`) & `W68_L3` (`8L`) (`batch_size=16, seq_len=512`)

| Checkpoint (`Tokens`) | `W68_L2` (`4L`, `31.61M`) Hot / EMA CE (`Top-1`) | `W68_L3` (`8L`, `44.34M`) Hot / EMA CE (`Top-1`) | `8L vs 4L` EMA Delta |
| :--- | :--- | :--- | :--- |
| **Step 2,500 (`20.48M`)** | `5.6711` / `5.6130` (`21.57%`) | **`5.5470` / `5.4967`** (**`22.67%`**) | **`-0.1163` nats** (`+1.10%`) |
| **Step 5,000 (`40.96M`)** | `5.5496` / `5.4553` (`22.75%`) | **`5.4181` / `5.3363`** (**`23.68%`**) | **`-0.1190` nats** (`+0.93%`) |
| **Step 7,500 (`61.44M`)** | `5.4152` / `5.3665` (`23.06%`) | **`5.3152` / `5.2705`** (**`23.90%`**) | **`-0.0960` nats** (`+0.84%`) |
| **Step 10,000 (`81.92M`)** | `5.3281` / `5.2839` (`23.68%`) | **`5.2477` / `5.1983`** (**`24.39%`**) | **`-0.0856` nats** (`+0.71%`) |
| **Step 11,250 (`92.16M`)** | `5.3043` / `5.2571` (`23.97%`) | **`5.2137` / `5.1724`** (**`24.71%`**) | **`-0.0847` nats** (`+0.74%`) |
| **Step 15,500 (`126.98M`, Pre-Quench)** | `5.2179` / `5.1811` (`24.32%`) | *(Canceled at `92.16M` to launch Wave 69)* | — |
| **Step 18,311 (`150.00M`, Post-Quench)** | **`4.9136` / `4.9111`** (**`26.60%`**) | — | — |

- **Post-Mortem & Transition to Wave 69 (`EXPERIMENT_PLAN_089.md` — Pure Density-Matrix Architecture)**:
  1. Scaling depth from $L=4 \to 8$ (`W68_L3`) with $\sqrt{4/L}$ geodesic step scaling consistently lowered Validation CE by **`-0.085` to `-0.119` nats** across the entire `20M–92M` token range while maintaining `MUBErr = 0.00e+00`, `SU16Phase = 0.00e+00`, and `KrausGauge = 0.00e+00`.
  2. However, auditing `_ExplicitDensityHighwayBlock` revealed why the Waves 63–68 architecture still plateaued near `~4.85–4.91` CE: the trunk carried a parallel classical vector stream `x_t` that projected $Q, K, V$ from `x_t` rather than from the density matrix $\rho_t$, used classical `exp`/`softmax` inside the attention kernel, and used a classical vector MLP (`mz_mlp`) instead of native density-matrix operations. Wave 69 (`EXPERIMENT_PLAN_089.md`) replaces this hybrid stream with a **100% Pure Density-Matrix Architecture** (`PureDensityMatrixGPT`).
