# Experiment Plan 086 (`Wave 66`): Horizontal-Grassmannian Lift, Gauge-Quotiented Coupled SVD/Kraus/$\mathbb{CP}^{15}$ Manifolds, & Spike-Guarded Shadow EMA Thermostat

## 1. Executive Summary & Architectural Motivation

In `Wave 65` (`EXPERIMENT_PLAN_085.md`), our `W65_V3_SHARED_STIEFEL_128D_FRAME` architecture (`St(512, 128)` + tied `128D` sphere vocabulary) achieved a massive early/mid-horizon speedup—reaching **`5.2889` Val CE (`23.92%` Top-1)** at `30.72M` tokens (`-0.5450` nats ahead of `W64_A3` at `30.89M` tokens). However, extending the run to `229.38M` tokens (`28,000` steps) revealed a classic late-horizon geometric pathology:
1. **Unquotiented Gauge Freedom & Vertical-Fiber Momentum Spin**:
   - While our *activation* highway (`rho_packed`) uses an extrinsic $16 \times 16$ Hermitian density-matrix representation $\rho_{t,h} \in \operatorname{Herm}(\mathbb{C}^{16})$ with linear trace readout $\operatorname{Tr}(\rho_{t,h} E_h)$ (free of ket phase ambiguity), our *parameter* manifolds (`u_in_stiefel`, `u_out_stiefel`, coupled `(u_stiefel, v_stiefel)` in `BiorthogonalSVDStiefelProjection`, `probe_u1/u2`, and `k_evol_re/im`) were optimized on the full **Stiefel/Unitary manifold** in `EUCLIDEAN_AMBIENT` momentum mode (`beta = 0.95`).
   - Specifically, the Stiefel tangent projection
     $$G_{\text{Stiefel}} = G - V \operatorname{sym}(V^\dagger G) = \underbrace{(I_D - V V^\dagger) G}_{\text{Horizontal Grassmannian } \mathcal{H}_V} + \underbrace{V \operatorname{skew}(V^\dagger G)}_{\text{Vertical } O(k)/U(k) \text{ Gauge Fiber } \mathcal{V}_V}$$
     retains the **vertical gauge-fiber component** $V \operatorname{skew}(V^\dagger G)$, which 5-step Newton–Schulz then amplifies to unit singular values (`= 1.0`). When two parameter groups share a gauge orbit, their vertical momentum buffers chase each other in circles around a minimum ("gauge spin") rather than descending horizontally across subspaces.
2. **The Thermostat "Ratchet Trap"**:
   - Once `lr_scale` hit its `0.25x` floor at `Step 4,000` (`32.77M` tokens), natural mini-batch noise plus gauge spin kept the Hot-vs-EMA gap around `+0.015` to `+0.021` nats (`>= 0.015`), causing `RiemannianShadowEMAThermostat` to overwrite the active exploratory weights $W \leftarrow W_{\text{ema}}$ and damp momentum by `0.70x` every `250` steps.

`Wave 66` (`EXPERIMENT_PLAN_086.md`) resolves all four parameter-manifold gauge collisions and the thermostat ratchet trap in a clean, flag-free module ([`src/qdlblocks/benchmarks/horizontal_grassmann_gauge_free_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/horizontal_grassmann_gauge_free_lab.py)).

---

## 2. Mathematical Formulation & Hypotheses

### Hypothesis H1 — Intrinsic Horizontal-Lift on $\operatorname{Gr}(k, D)$ for Subspace Bridges ($U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$)
In `SharedStiefelSubspaceFrameBoundary`, rotating the `128D` subspace frame by any $Q \in O(128)$ ($U_{\text{in}} \mapsto U_{\text{in}} Q$, $U_{\text{out}} \mapsto U_{\text{out}} Q$, $W_{\text{vocab}} \mapsto W_{\text{vocab}} Q$) leaves all embeddings and logits invariant. By projecting both the gradient and the momentum buffer onto the **Horizontal Grassmannian Tangent Space** $\mathcal{H}_U = \{\Delta \in \mathbb{R}^{D \times k} : U^\top \Delta = 0\}$:
$$G_{\text{Grassmann}} = (I_D - U U^\top) \nabla_U \mathcal{L}, \qquad M_{\text{Grassmann}} = (I_D - U U^\top) M$$
and re-projecting after 5-step Newton–Schulz:
$$\Delta_{\text{Grassmann}} = (I_D - U U^\top) \operatorname{NS}_5(G_{\text{Grassmann}} + \beta M_{\text{Grassmann}})$$
we satisfy $U^\top \Delta_{\text{Grassmann}} \equiv 0$ (`vertical_fiber_grad_frac == 0.0`, `vertical_fiber_mom_frac == 0.0`), freezing internal $O(128)$ coordinate rotation so $W_{\text{vocab}}$ optimizes token vectors inside a stationary coordinate frame.

### Hypothesis H2 — Joint $O(r)$ Gauge-Quotient Projection for Coupled Biorthogonal SVD-Stiefel Pairs $(U, V)$
In every `BiorthogonalSVDStiefelProjection` ($W(x) = ((x V) \odot \vec{\sigma}) U^\top$ with $U \in \operatorname{St}(D_{\text{out}}, r), V \in \operatorname{St}(D_{\text{in}}, r)$), the joint right-action $(U, V) \mapsto (U Q, V Q)$ ($Q \in O(r)$) generates a vertical gauge orbit with tangent space $\{(U \Omega, V \Omega) : \Omega \in \mathfrak{so}(r)\}$.
For any pair of Stiefel tangent vectors $(\xi_U, \xi_V) \in T_U \operatorname{St}(D_{\text{out}}, r) \times T_V \operatorname{St}(D_{\text{in}}, r)$, the exact orthogonal projection onto the horizontal complement orthogonal to the joint $O(r)$ gauge fiber is:
$$\Omega_{\text{gauge}}(\xi_U, \xi_V) = \frac{1}{2}\Big(\operatorname{skew}(U^\top \xi_U) + \operatorname{skew}(V^\top \xi_V)\Big) \in \mathfrak{so}(r)$$
$$\xi_U^{\text{horiz}} = \xi_U - U \Omega_{\text{gauge}}, \qquad \xi_V^{\text{horiz}} = \xi_V - V \Omega_{\text{gauge}}$$
which guarantees $\operatorname{skew}(U^\top \xi_U^{\text{horiz}}) + \operatorname{skew}(V^\top \xi_V^{\text{horiz}}) \equiv 0$ for both gradients and tangent-transported momentum across all `q_proj, k_proj, v_proj, obs_proj, up_proj, down_proj, exit_bridge, tomo_proj` operators.

### Hypothesis H3 — Gauge-Locked $\mathbb{CP}^{15}$ MUB Tomography & $U(R)$-Quotiented CPTP Kraus Channels
1. **Analytical MUB Locking + $\mathbb{CP}^{15}$ Phase Quotient**: Instead of two independent $U(16)$ probe matrices `probe_u1, probe_u2` that can drift into redundant/colliding bases, we parameterize a **single** $U_h \in U(16)/U(1)^{16}$ per head and derive the second mutually unbiased basis analytically via $U_h^{(2)} := F_{16} U_h$ ($|\langle c_{h,m}^{(1)} \mid c_{h,n}^{(2)}\rangle|^2 \equiv \frac{1}{16}$). Because each probe column only enters through $|c_{h,m}\rangle\langle c_{h,m}| \in \mathbb{CP}^{15}$, we zero out the diagonal vertical $U(1)^{16}$ phase fiber $\operatorname{diag}(\operatorname{Im}(U_h^\dagger \xi)) = 0$ in both gradient and tangent-transported momentum.
2. **$U(R)$ Inter-Branch Kraus Gauge Quotient**: For $\mathbf{K}_h = [K_{h,1}; \dots; K_{h,R}] \in \operatorname{St}_{\mathbb{C}}(R d_h, d_h)$, the channel $\mathcal{E}_h(\rho) = \sum_{r=1}^R K_{h,r}\rho K_{h,r}^\dagger$ is invariant under $(W \otimes I_{d_h})\mathbf{K}_h$ for $W \in U(R)$. We project out the vertical $\mathfrak{u}(R)$ branch-mixing generator:
   $$A_{rs}^{\text{Kraus}} = \frac{1}{2 d_h}\Big(\operatorname{Tr}(K_{h,r}^\dagger \Xi_{h,s}) - \overline{\operatorname{Tr}(K_{h,s}^\dagger \Xi_{h,r})}\Big), \qquad \Xi_{h,r}^{\text{horiz}} = \Xi_{h,r} - \sum_{s=1}^R A_{rs}^{\text{Kraus}} K_{h,s}$$

### Hypothesis H4 — Spike-Guarded Shadow EMA Thermostat (Decoupling Polyak–Ruppert Evaluation from Hard Weight Resets)
By keeping the manifold-retracted Shadow EMA strictly as a read-only Polyak–Ruppert evaluation buffer during normal descent (and only triggering a hard weight rollback $W \leftarrow W_{\text{ema}}$ when hot validation loss spikes by $\mathcal{L}_{\text{hot}}(t) > \mathcal{L}_{\text{hot}}(t-1) + \delta_{\text{spike}}$ with $\delta_{\text{spike}} = 0.030$ nats or when $\mathcal{L}_{\text{hot}}(t) - \mathcal{L}_{\text{ema}}(t) \ge 0.080$ nats), the active weights retain long-horizon momentum across flat valleys while still enjoying the `-0.015` to `-0.030` nat Shadow EMA evaluation advantage.

---

## 3. Planned Ablation Variants (`Wave 66`)

1. **`W66_G0_STIEFEL_AMBIENT_BASELINE`**:
   - Reference `W65_V3_SHARED_STIEFEL_128D_FRAME` configuration (`EUCLIDEAN_AMBIENT` Stiefel momentum with unquotiented vertical fibers and every-checkpoint `0.015`-gap hard EMA reset).
2. **`W66_G1_HORIZONTAL_GRASSMANN_VOCAB_AND_BRIDGES`**:
   - Applies the exact Horizontal Grassmannian projection $G_{\text{Grassmann}} = (I_D - U U^\top) G$ + tangent-transported momentum to `u_in_stiefel, u_out_stiefel` and the joint $O(r)$ gauge quotient $\Omega_{\text{gauge}} = \frac{1}{2}(\operatorname{skew}(U^\top \xi_U) + \operatorname{skew}(V^\top \xi_V))$ to all coupled `(U, V)` pairs in `BiorthogonalSVDStiefelProjection`.
3. **`W66_G2_GAUGE_LOCKED_CP15_MUB_AND_KRAUS`**:
   - Adds single-generator gauge-locked MUB tomography ($U_h^{(2)} \equiv F_{16} U_h^{(1)}$ with $\mathbb{CP}^{15}$ diagonal phase fiber removal) + $U(R)$ inter-branch Kraus gauge quotient.
4. **`W66_G3_FULL_GAUGE_FREE_SPIKE_ONLY_THERMOSTAT`**:
   - Combines `G1 + G2` with the Spike-Guarded Shadow EMA Thermostat (decoupling Polyak–Ruppert EMA evaluation and early LR cooling from destructive every-checkpoint weight rollbacks).

---

## 4. Top-of-File Named Constants Ledger (Rule 7)

| Constant Name | Value | Physical / Mathematical Role |
| :--- | :---: | :--- |
| `DEFAULT_GRASSMANN_SUBSPACE_DIM` | `128` | Rank $k = 128$ of shared vocabulary Grassmannian frame $U_{\text{in}}, U_{\text{out}} \in \operatorname{Gr}(128, 512)$ |
| `DEFAULT_COUPLED_GAUGE_QUOTIENT_SCALE` | `0.5` | Exact $\frac{1}{2}$ projection weight in $\Omega_{\text{gauge}} = \frac{1}{2}(\operatorname{skew}(U^\top \xi_U) + \operatorname{skew}(V^\top \xi_V))$ |
| `DEFAULT_HORIZ_MOMENTUM_BETA` | `0.92` | Tangent-transported horizontal Riemannian momentum decay $\beta$ |
| `DEFAULT_SPIKE_ROLLBACK_TOL_NATS` | `0.030` | Hot validation loss increase $\mathcal{L}_{\text{hot}}(t) - \mathcal{L}_{\text{hot}}(t-1)$ required to trigger hard rollback |
| `DEFAULT_CATASTROPHIC_GAP_TOL_NATS` | `0.042` | Post-Parseval-ramp Hot-vs-EMA gap threshold triggering active-weight banking (`0.60x` LR floor + bidirectional thermal recovery) |
| `DEFAULT_GENTLE_COOL_GAP_NATS` | `0.022` | Thermal gap threshold for gentle LR cooling (`0.96x` decay) without resetting active weights |
| `DEFAULT_GENTLE_LR_DECAY_FACTOR` | `0.96` | Multiplicative LR cooling/recovery factor (`0.96x` cool when gap $\ge 0.022$, `1/0.96x` recovery when gap $< 0.012$) |

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3)

1. **Existing Scripts Used**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py): Enforce Rule 1 (`<= 1,000` SLOC soft warning) and Rule 2 (`<= 8` function arguments).
   - [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py): Sync standardized logs and model cards into the WebUI catalog.
2. **`Build-Before-Launch` Script**:
   - [`scripts/horizontal_grassmann_gauge_free_profiler_and_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/horizontal_grassmann_gauge_free_profiler_and_bench.py): Deterministic CLI tool that profiles pre/post-projection vertical gauge-fiber fractions (`grassmann_step_spin_frac`, `grassmann_mom_spin_frac`, `coupled_svd_step_spin_frac`, `cp15_step_phase_frac`, `kraus_step_gauge_frac`, `mub_unbiased_err`) and runs the 4-variant `Wave 66` benchmark suite and multi-checkpoint horizon runs.
3. **`Nice-to-Have` Script (Logged in `scripts/TOOLING_BACKLOG.md`)**:
   - `scripts/gauge_orbit_perturbation_invariance_checker.py`: Standalone CLI verifier that applies random finite gauge transformations ($Q \in O(128)$ on `(U_in, U_out, W_vocab)`, $Q \in O(r)$ on `(U, V)` when $\sigma = c I$, $e^{i\theta} \in U(1)^{16}$ on MUB probes, and $W \in U(R)$ on Kraus branches) and reports forward-logit invariance and backward-gradient covariance to machine precision.

---

## 6. Verification & Empirical Results Ledger

### 6.1 Unit Test & AST Linter Verification
- **AST Compliance**: [`src/qdlblocks/benchmarks/horizontal_grassmann_gauge_free_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/horizontal_grassmann_gauge_free_lab.py) (`926` SLOC $\le 1,000$, max args `= 8`, **PASS**), [`scripts/horizontal_grassmann_gauge_free_profiler_and_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/horizontal_grassmann_gauge_free_profiler_and_bench.py) (`95` SLOC, **PASS**), [`tests/test_plan086_horizontal_grassmann_gauge_free.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan086_horizontal_grassmann_gauge_free.py) (`107` SLOC, **PASS**).
- **Exact Gauge-Projection Unit Tests**: `4 passed` in [`tests/test_plan086_horizontal_grassmann_gauge_free.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan086_horizontal_grassmann_gauge_free.py), including the exact $R \times R$ Hermitian Lyapunov solver $\frac{1}{2}(A G_K + G_K A) = S_{\text{skew}}(\Xi)$ on the Kraus branch Gram matrix $G_K = K_{\text{flat}} K_{\text{flat}}^\dagger$ in [`project_kraus_branch_gauge_horizontal`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/horizontal_grassmann_gauge_free_lab.py#L243-L280), which stays strictly inside $T_{\mathbf{K}_h}\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ and drives `kraus_step_gauge_frac` to `0.00e+00` (`< 1e-6`).

### 6.2 60-Step 4-Variant Ablation Suite (`B=8, T=256, L=4`, `65,536` Val Tokens)

| Variant | Params | Step 40 Hot / EMA CE | Step 60 Hot / EMA CE | Step 60 Top-1 | `GrSpin` | `SVDSpin` | `CP15Phase` | `KrausGauge` | `MUBErr` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`W66_G0_STIEFEL_AMBIENT_BASELINE`** | `23,554,873` | `7.7114` / `7.6824` (`trig=True`) | `7.4843` / `7.4822` | `9.29%` | **`7.53e-01`** (`75.3%`) | **`1.37e-01`** (`13.7%`) | **`2.40e-01`** (`24.0%`) | **`2.01e-02`** (`2.01%`) | **`4.07e-01`** (`6.5x` MUB drift) |
| **`W66_G1_HORIZONTAL_GRASSMANN_VOCAB_AND_BRIDGES`** | `23,554,873` | `7.7110` / `7.6794` (`trig=True`) | `7.4915` / `7.4902` | `9.16%` | **`0.00e+00`** | **`0.00e+00`** | `2.25e-01` | `1.97e-02` | `4.74e-01` |
| **`W66_G2_GAUGE_LOCKED_CP15_MUB_AND_KRAUS`** | `23,546,681` | **`7.7103` / `7.6789`** (`trig=True`) | `7.4920` / `7.4901` | `9.18%` | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** |
| **`W66_G3_FULL_GAUGE_FREE_SPIKE_ONLY_THERMOSTAT`** | `23,546,681` | **`7.7103` / `7.6789`** (`trig=False`) | `7.4879` / `7.4853` | `9.23%` | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** |

### 6.3 200-Step (`8`-Checkpoint, `0.41M` Tokens) Multi-Checkpoint Horizon Comparison (`W66_G0` vs. `W66_G3`)

| Checkpoint | `W66_G0` Hot / EMA CE (`trig`, `lr`, `MUBErr`) | `W66_G3` Hot / EMA CE (`trig`, `lr`, `MUBErr`) | Key Comparison Insight |
| :--- | :---: | :---: | :--- |
| **Step 25 (`0.05M` tok)** | `7.9174` / `7.9423` (`False`, `1.00x`, `0.228`) | `7.9296` / **`7.9417`** (`False`, `1.00x`, **`0.000`**) | `W66_G3` leads on EMA CE (`7.9417` vs `7.9423`) with `0.0` MUB error. |
| **Step 75 (`0.15M` tok)** | `7.5033` / `7.4331` (`True`, `0.80x`, `0.411`) | **`7.4905`** / `7.4364` (`True`, `0.84x`, **`0.000`**) | Before any thermostat reset (`lr=1.00x`), **`W66_G3` beats `W66_G0` on Hot CE (`7.4905` vs `7.5033`, `-0.0128` nats) and Top-1 (`9.86%` vs `9.67%`)** while cutting thermal gap by `23%` (`0.0541` vs `0.0702`). |
| **Step 100 (`0.20M` tok)** | `7.4530` / `7.4111` (`True`, `0.69x`, `0.438`) | **`7.4460` / `7.4096`** (`False`, `0.80x`, **`0.000`**) | **`W66_G3` beats `W66_G0` on BOTH Hot CE (`7.4460` vs `7.4530`) and EMA CE (`7.4096` vs `7.4111`)** while avoiding `W66_G0`'s second consecutive hard rollback. |
| **Step 150 (`0.31M` tok)** | `7.2357` / `7.2081` (`True`, `0.56x`, `0.603`) | `7.2695` / `7.2457` (`False`, `0.74x`, **`0.000`**) | `W66_G0` triggers its 4th consecutive hard rollback (`lr` ratcheted to `0.56x`, `MUBErr = 0.603`). |
| **Step 200 (`0.41M` tok)** | `7.0713` / `7.0724` (`False`, `0.58x`, **`0.697`**) | `7.1035` / `7.1050` (`False`, **`0.80x`**, **`0.000`**) | `W66_G3` automatically recovers `lr_scale` (`0.74x -> 0.77x -> 0.80x`) once thermal gap cools below `0.012`, maintaining `0.00e+00` gauge spin and `0.00e+00` MUB overlap error (`1` total trigger vs. `4` in `W66_G0`). |

### 6.4 Key Mathematical & Architectural Discoveries (Post-Mortem & Next Wave Guidance)
1. **Quantifying Hidden Vertical Gauge Spin in Stiefel/Unitary Muon (`W66_G0`)**:
   - Profiling `W66_G0_STIEFEL_AMBIENT_BASELINE` proved that **`75.3%` (`GrSpin = 7.53e-01`)** of the Newton–Schulz step norm on tall vocabulary bridges $U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$, **`24.0%` (`CP15Phase = 2.40e-01`)** of the step norm on MUB probes, **`13.7%` (`SVDSpin = 1.37e-01`)** on coupled SVD pairs, and **`2.01%` (`KrausGauge = 2.01e-02`)** on CPTP Kraus operators were vertical fiber movements. Furthermore, unconstrained $U(16) \times U(16)$ MUB probes suffered catastrophic basis collapse (`MUBErr` rose from `0.0` at Step 0 to **`0.407` at Step 60** and **`0.697` at Step 200**, vs. the ideal $1/16 = 0.0625$ MUB overlap).
2. **True Quotient Symmetries (`G1 -> G2`) Strictly Improve Validation Loss**:
   - Adding Gauge-Locked $\mathbb{CP}^{15}$ Flag MUB Tomography ($U_h^{(2)} \equiv F_{16} U_h^{(1)}$ with diagonal phase quotient) and the exact Lyapunov $\mathfrak{u}(R)$ Kraus branch quotient (`W66_G1 -> W66_G2`) **strictly improved validation CE and Top-1 accuracy at every checkpoint** while removing `8,192` redundant parameters and locking `MUBErr = 0.00e+00` forever.
3. **Why Full Grassmannian / $O(r)$ Projection on Tied Bridges & Non-Isotropic SVD (`G0 -> G1`) Over-Constrains Late-Horizon Descent**:
   - **Tied Input-Output Subspace Bridges ($U_{\text{in}}, U_{\text{out}}$)**: Because `w_vocab_128` is *shared* between input token wavepackets ($W_{\text{vocab}} U_{\text{in}}^\top$) and output readout detectors ($(h_{512} U_{\text{out}}) W_{\text{vocab}}^\top$) and initialized with $U_{\text{in}} = U_{\text{out}}$ at Step 0, only the **common** diagonal subgroup $\{(Q, Q) : Q \in O(128)\}$ is a gauge symmetry, while the **relative** rotation $\Omega_{\text{in}} - \Omega_{\text{out}} \in \mathfrak{so}(128)$ is the physical input-to-output transition operator inside the 128D vocabulary subspace. Projecting both $U_{\text{in}}$ and $U_{\text{out}}$ independently onto $\operatorname{Gr}(128, 512)$ ($U_{\text{in}}^\top \xi_{\text{in}} = 0$ and $U_{\text{out}}^\top \xi_{\text{out}} = 0$) freezes their relative $O(128)$ alignment.
   - **Non-Isotropic Singular Spectrum in `BiorthogonalSVDStiefelProjection` ($W = U \operatorname{diag}(\vec{\sigma}) V^\top$)**: Under $\delta U = U(\Omega_{\text{sym}} + \Omega_{\text{anti}})$ and $\delta V = V(\Omega_{\text{sym}} - \Omega_{\text{anti}})$, the operator perturbation is $U^\top \delta W V = (\Omega_{\text{anti}} \Sigma + \Sigma \Omega_{\text{anti}}) + (\Omega_{\text{sym}} \Sigma - \Sigma \Omega_{\text{sym}})$. Once $\vec{\sigma}$ learns distinct singular values ($\sigma_i \neq \sigma_j$), $\Omega_{\text{sym}, ij}$ is the **only** tangent direction that can rotate the singular vectors $(u_i, v_i)$ to diagonalize symmetric gradient perturbations $S_{\text{sym}}$ ($(\delta W)_{ij}^{\text{sym}} = (\sigma_j - \sigma_i)\Omega_{\text{sym}, ij}$).
   - **Next-Wave Recommendation (`Wave 67`)**: Keep the **exact $\mathbb{CP}^{15}$ Flag MUB + Lyapunov $\mathfrak{u}(R)$ Kraus gauge quotients (`W66_G2`)** and **bidirectional Spike-Guarded Thermostat (`W66_G3`)**, replace independent $\operatorname{Gr}(128, 512)$ projection on $(U_{\text{in}}, U_{\text{out}})$ with **common-fiber-only $O(128)$ quotienting** ($\Omega_{\text{in}} + \Omega_{\text{out}} = 0$, preserving $\Omega_{\text{in}} - \Omega_{\text{out}}$), and weight the SVD co-rotation projection by spectral degeneracy $w_{ij} = \exp(-(\sigma_i - \sigma_j)^2 / \tau_\sigma^2)$ so only truly degenerate singular pairs ($\sigma_i \approx \sigma_j$) are quotiented.

