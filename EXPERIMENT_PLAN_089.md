# Experiment Plan 089 (`Wave 69`): Pure Density-Matrix Architecture — Direct $Q/K/V$ Density Projections, `exp`-Free Hilbert–Schmidt Quantum Attention, Cross-Head Quantum "OR + AND" (Schur Product) Logic, Online EMA N-Gram Density Matrices, Stateful Multi-Window Density History, & Unconstrained Born Readout

## 1. Executive Summary & Architectural Diagnosis

Auditing the Waves 63–68 architecture (`_ExplicitDensityHighwayBlock` in [`explicit_density_tomographic_512d_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/explicit_density_tomographic_512d_lab.py)) revealed why validation cross-entropy plateaued near `~4.85–4.91` nats despite exact quotient geometry:
1. **Parallel Classical Vector Stream (`x_t`) Dominated the Trunk**: Every block carried a classical `512D` real vector `x_t` alongside `rho_packed`, projecting $Q, K, V$ from `F.rms_norm(x_t)` rather than from the density matrix $\rho_{t,h}$, and flattening attention outputs back into a classical vector MLP (`mz_mlp`).
2. **Classical `exp`/`softmax` Inside Attention**: `_fused_hermitian_born_doc_swc_fwd_kernel` exponentiated real inner products via `tl.exp(...)` instead of using the intrinsic non-negativity of the quantum Hilbert–Schmidt overlap $\operatorname{Tr}(\rho_t^Q \rho_s^K) \ge 0$.
3. **Lack of Cross-Head Quantum Entanglement & Native Density Logic**: Heads evolved in isolated $\mathbb{C}^{16}$ subspaces during Kraus propagation and only interacted after collapsing to classical vector representations.
4. **Artificial `30.0 * tanh(z / 30.0)` Logit Softcap & Window-Boundary Amnesia**: The readout squashed high-certainty predictions through `tanh`, and every training window reset $\rho_0$ to a fresh state at position $t=0$.

**Wave 69 (`EXPERIMENT_PLAN_089.md`)** introduces [`PureDensityMatrixGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/pure_density_matrix_gpt_lab.py)—a **100% Pure Density-Matrix Architecture** where **the only state flowing through the trunk is a batch of multi-head density matrices** $\boldsymbol{\rho}_t^{(\ell)} = (\rho_{t,1}^{(\ell)}, \dots, \rho_{t,H}^{(\ell)}) \in \mathcal{D}(\mathbb{C}^{16})^H$, parameterized via Rank-$R$ ($R=4$) complex Cholesky/purification factors $L_{t,h}^{(\ell)} \in \mathbb{C}^{16 \times 4}$ ($\rho_{t,h}^{(\ell)} = L_{t,h}^{(\ell)} L_{t,h}^{(\ell)\dagger}$, $\operatorname{Tr}(\rho_{t,h}^{(\ell)}) = \|L_{t,h}^{(\ell)}\|_F^2 = 1$) and their Frobenius-isometric packed Hermitian representations $\tilde{\rho}_{t,h}^{(\ell)} = \tilde{P}(\rho_{t,h}^{(\ell)}) \in \mathbb{R}^{256}$.

---

## 2. Core Mathematical & Architectural Pillars (Machine-Checked in Lean 4 Theorems 6–8)

### Pillar 1 (`Path B` / Lean 4 Theorem 6): Direct $Q/K/V$ Density-Matrix Projections & `exp`-Free Hilbert–Schmidt Quantum Attention
- **Direct Projection from $\rho_{t,h}^{\text{in}} = L_{t,h} L_{t,h}^\dagger$**:
  - Each layer $\ell$ projects the input Cholesky factor $L_{t,h} \in \mathbb{C}^{16 \times R}$ directly through learned horizontal complex unitary/Stiefel operators $W_{Q,h}, W_{K,h}, W_{V,h} \in U(16)$ (with complex RoPE phase rotation $e^{i t \omega_k}$ on $Q, K$ and optional causal lag mixing on $K, V$):
    $$Q_{t,h} = \mathcal{R}_{\text{RoPE}, t}\!\left(W_{Q,h}\, L_{t,h}\right), \qquad K_{s,h} = \mathcal{R}_{\text{RoPE}, s}\!\left(W_{K,h}\, L_{s,h}\right), \qquad V_{s,h} = W_{V,h}\, L_{s,h}$$
    defining the projected Query, Key, and Value density matrices:
    $$\rho_{t,h}^Q = Q_{t,h} Q_{t,h}^\dagger \in \mathcal{D}(\mathbb{C}^{16}), \qquad \rho_{s,h}^K = K_{s,h} K_{s,h}^\dagger \in \mathcal{D}(\mathbb{C}^{16}), \qquad \rho_{s,h}^V = (1 - \gamma_{V,h})\,V_{s,h} V_{s,h}^\dagger + \gamma_{V,h}\,W_{V,h}\rho_{s,h}^{\text{in}}W_{V,h}^\dagger \in \mathcal{D}(\mathbb{C}^{16})$$
- **Flattened Hermitian Outer-Product Hilbert–Schmidt Similarity (`exp`/`softmax`-Free)**:
  - By **Lean 4 Theorem 6**, packing $\rho_{t,h}^Q$ and $\rho_{s,h}^K$ via the Frobenius-isometric map $\tilde{P}: \operatorname{Herm}(\mathbb{C}^{16}) \to \mathbb{R}^{256}$ turns the exact quantum Hilbert–Schmidt overlap into a single Tensor-Core inner product:
    $$S_{t,s}^{(h)} = \big\langle \tilde{P}(\rho_{t,h}^Q),\; \tilde{P}(\rho_{s,h}^K) \big\rangle_{\mathbb{R}^{256}} \equiv \operatorname{Tr}\!\left(\rho_{t,h}^Q \rho_{s,h}^K\right) = \left\| Q_{t,h}^\dagger K_{s,h} \right\|_F^2 = \sum_{r=1}^R \sum_{r'=1}^R \left|\langle q_{t,h}^{(r)} \mid k_{s,h}^{(r')}\rangle\right|^2 \ge 0$$
  - Because $S_{t,s}^{(h)} \ge 0$ identically, applying polynomial $p$-boson coincidence sharpening $\tilde{S}_{t,s}^{(h)} = w_{t,s}^{\text{pos}} \cdot \big(S_{t,s}^{(h)} + \beta_{\text{coinc},h}\,(S_{t,s}^{(h)})^2\big) \ge 0$ and normalizing by the causal partition sum plus positive vacuum floor $\varepsilon_{\text{vac},h} > 0$:
    $$A_{t,s}^{(h)} = \frac{\tilde{S}_{t,s}^{(h)}}{\sum_{j \le t} \tilde{S}_{t,j}^{(h)} + \sum_{m=1}^{M_{\text{hist}}} \tilde{S}_{t,\text{hist},m}^{(h)} + \varepsilon_{\text{vac},h}} \ge 0$$
    produces exact convex mixing weights ($\sum_{s \le t} A_{t,s}^{(h)} + \sum_{m=1}^{M_{\text{hist}}} A_{t,\text{hist},m}^{(h)} + A_{t,\text{vac}}^{(h)} \equiv 1$) with **zero `exp`/`softmax`** and **exact $U(R)$-gauge invariance** ($Q_t \mapsto Q_t U_t$ cancels identically).
- **Convex Value Density Mixing & Residual Density Delta**:
  $$\bar{\rho}_{t,h}^V = \sum_{s \le t} A_{t,s}^{(h)}\,\rho_{s,h}^V + \sum_{m=1}^{M_{\text{hist}}} A_{t,\text{hist},m}^{(h)}\,\rho_{\text{hist},m}^{V,(h)} + A_{t,\text{vac}}^{(h)}\,\frac{1}{16}I_{16}, \qquad \rho_{t,h}^{\text{mid}} = (1 - \alpha_{h,\ell})\,\rho_{t,h}^{\text{in}} + \alpha_{h,\ell}\,\mathcal{E}_{\text{attn},h}\!\left(\bar{\rho}_{t,h}^V\right)$$

### Pillar 2 (`Paths C & D Unified` / Lean 4 Theorem 7): Cross-Head Quantum "OR + AND" (Schur Product) Logic Channel
- Replaces the classical vector Mach–Zehnder MLP (`mz_mlp`) with a native density-matrix logic sub-block operating directly on $\rho_{t,1}^{\text{mid}}, \dots, \rho_{t,H}^{\text{mid}} \in \mathcal{D}(\mathbb{C}^{16})$ and their Cholesky factors $L_{t,h}^{\text{mid}} \in \mathbb{C}^{16 \times R}$:
  1. **Quantum "OR" (Unitary / Orthostochastic Cross-Head Braiding)**:
     - Two learned cross-head orthogonal Stiefel matrices $W_{\text{OR}}^{(A)}, W_{\text{OR}}^{(B)} \in \operatorname{O}(H)$ ($H=16$) and per-head unitary rotations $U_{\text{OR},h}^{(A)}, U_{\text{OR},h}^{(B)} \in U(16)$ coherently braid the Cholesky factors across all $H$ heads:
       $$L_{t,h}^{(A)} = \operatorname{Norm}_F\!\left(\sum_{h'=1}^H W_{\text{OR}, h, h'}^{(A)}\, U_{\text{OR}, h}^{(A)} L_{t,h'}^{\text{mid}}\right), \qquad L_{t,h}^{(B)} = \operatorname{Norm}_F\!\left(\sum_{h'=1}^H W_{\text{OR}, h, h'}^{(B)}\, U_{\text{OR}, h}^{(B)} L_{t,h'}^{\text{mid}}\right)$$
     - Simultaneously, the orthostochastic probabilities $B_{h,h'}^{\text{OR}} = (W_{\text{OR}, h, h'}^{(A)})^2 \ge 0$ ($\sum_{h'=1}^H B_{h,h'}^{\text{OR}} \equiv 1$ since $W_{\text{OR}}^{(A)} \in \operatorname{O}(H)$!) convexly mix the packed density matrices across heads: $\rho_{t,h}^{\text{OR}} = \sum_{h'=1}^H B_{h,h'}^{\text{OR}}\,\rho_{t,h'}^{\text{mid}} \in \mathcal{D}(\mathbb{C}^{16})$.
  2. **Quantum "AND" (Schur / Hadamard Coincidence Product via the Schur Product Theorem)**:
     - By **Lean 4 Theorem 7b (Schur Product Theorem)**, the elementwise Hadamard product of two PSD Hermitian density matrices $\rho_A \circ \rho_B$ is strictly PSD Hermitian, and on the Cholesky factors $L_{t,h}^{(A)}, L_{t,h}^{(B)} \in \mathbb{C}^{16 \times R}$ their complex Hadamard coincidence factor is:
       $$\left(L_{t,h}^{\text{AND,raw}}\right)_{i,r} = \left(L_{t,h}^{(A)}\right)_{i,r} \cdot \left(L_{t,h}^{(B)}\right)_{i,r} \in \mathbb{C}^{16 \times R}, \qquad L_{t,h}^{\text{AND}} = \frac{L_{t,h}^{\text{AND,raw}}}{\left\|L_{t,h}^{\text{AND,raw}}\right\|_F + \varepsilon} \in \mathbb{C}^{16 \times R}$$
       producing the trace-1 PSD coincidence density matrix $\rho_{t,h}^{\text{AND}} = L_{t,h}^{\text{AND}} (L_{t,h}^{\text{AND}})^\dagger \in \mathcal{D}(\mathbb{C}^{16})$.
  3. **Convex "OR + AND" Synthesis & Lindblad Thermal Reset**:
     $$\rho_{t,h}^{\text{logic}} = (1 - \lambda_{\text{and},h})\,\Big(\frac{1}{2}\rho_{t,h}^{\text{OR}} + \frac{1}{2}L_{t,h}^{(A)} L_{t,h}^{(A)\dagger}\Big) + \lambda_{\text{and},h}\,\rho_{t,h}^{\text{AND}} \in \mathcal{D}(\mathbb{C}^{16})$$
     $$\rho_{t,h}^{\text{out}} = (1 - \gamma_{\text{depol},h})\Big[(1 - \beta_{h,\ell})\,\rho_{t,h}^{\text{mid}} + \beta_{h,\ell}\,\rho_{t,h}^{\text{logic}}\Big] + \gamma_{\text{depol},h}\,\frac{1}{16}I_{16} \in \mathcal{D}(\mathbb{C}^{16})$$

### Pillar 3 (`Path A` / Lean 4 Theorem 8a): NanoGPT-Speedrun-Compliant Online EMA Byte/N-Gram Density Matrices
- At Layer 0, instead of offline BPE clustering or a classical Euclidean n-gram table, [`OnlineEmaNgramDensityBank`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/pure_density_matrix_gpt_lab.py) maintains:
  1. Learned Complex Stiefel/Sphere N-Gram Cholesky factors $L_{\text{ngram}}[g] \in \mathbb{C}^{H \times 16}$ (`G = 4096` hash buckets over $(v_t, v_{t-1}, v_{t-2})$ and byte-pairs), and
  2. An **Online Causal EMA Density Buffer** $\rho_{\text{ema}}^{\text{ngram}}[g] \in \mathcal{D}(\mathbb{C}^{16})^H$ initialized at Step 0 to the maximally mixed state $\frac{1}{16}I_{16}$ (zero precomputation!) and updated online during training from the observed token wavepacket density matrices $|\psi_{t,h}^{\text{tok}}\rangle\langle\psi_{t,h}^{\text{tok}}|$:
     $$\rho_{\text{ema}}^{\text{ngram}}[g_t] \;\leftarrow\; (1 - \eta_{\text{ngram}})\,\rho_{\text{ema}}^{\text{ngram}}[g_t] + \eta_{\text{ngram}}\,|\psi_{t,h}^{\text{tok}}\rangle\langle\psi_{t,h}^{\text{tok}}|$$

### Pillar 4 (`Path F` / Lean 4 Theorem 8b): Stateful Multi-Window Quantum Density History Buffer
- Across consecutive training windows, each layer $\ell$ maintains $M_{\text{hist}} = 8$ multi-timescale summary density matrices $\rho_{\text{hist},m}^{(h,\ell)} \in \mathcal{D}(\mathbb{C}^{16})$ (`m = 1..8`, decay rates $\gamma_m \in [0.015, 0.50]$) updated at the end of each batch window by the detached window-average density matrix $\bar{\rho}_{\text{win}}^{(h,\ell)} = \frac{1}{T}\sum_{t=1}^T \rho_{t,h}^{(\ell)}$.
- Current-window Query density matrices $\rho_{t,h}^Q$ attend directly to $\rho_{\text{hist},m}^{(h,\ell)}$ via the exact same Hilbert–Schmidt similarity $\operatorname{Tr}(\rho_{t,h}^Q \rho_{\text{hist},m}^{(h,\ell)}) \ge 0$.

### Pillar 5 (`Path E`): Pure Density-Matrix Tomographic Readout & Unconstrained Exact Log-Likelihood
- At Layer $L$, token scoring depends **exclusively on the final density matrix $\rho_{t,h}^{(L)} \in \mathcal{D}(\mathbb{C}^{16})$** (zero classical trunk vector `h_exit`!).
- Each head $h \in \{1, \dots, 16\}$ extracts a `32D` quantum observable vector combining:
  1. Gauge-locked $\mathfrak{su}(16)$ two-basis MUB expectations $\langle C_h \mid \rho_{t,h}^{(L)} \mid C_h \rangle \in \mathbb{R}^{32}$, and
  2. Direct horizontal Stiefel Hermitian observables $W_{\text{obs},h}^\top \tilde{\rho}_{t,h}^{(L)} \in \mathbb{R}^{32}$ ($W_{\text{obs},h} \in \operatorname{St}(256, 32)$ on the full `256D` operator space $\operatorname{Herm}(\mathbb{C}^{16})$).
- Concatenated into $o_t \in \mathbb{R}^{512}$ and projected through the full-rank orthogonal bridge $Q_{\text{out}} = [U_{\text{out}}^{(1)}, U_{\text{out}}^{(2)}] \in \operatorname{O}(512)$ into [`_UnconstrainedDualSubspace512DCEFn`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/pure_density_matrix_gpt_lab.py), the model evaluates the **unconstrained exact log-likelihood** (with the artificial `30.0 * tanh(z / 30.0)` softcap removed).

---

## 3. Planned Ablation & Scale-Up Variants (`Wave69VariantMode`, Rule 3)

1. **`W69_P0_PURE_DENSITY_HS_ATTN_4L`**:
   - Pure density-matrix trunk ($L=4, H=16, d_h=16, R=4$), direct $Q/K/V$ density projections, `exp`-free Hilbert–Schmidt attention, and Quantum "OR + AND" (Schur product) cross-head logic, with softcap still enabled (`enable_unconstrained_born_ce=False`) and stateless windows (`enable_stateful_ema_and_history=False`) as the clean baseline.
2. **`W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L`**:
   - Adds **Path E** (removes the `30.0 * tanh(...)` softcap in `_UnconstrainedDualSubspace512DCEFn` and enables the full `256D -> 32D` horizontal Stiefel Hermitian observable basis $W_{\text{obs},h} \in \operatorname{St}(256, 32)$).
3. **`W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L`**:
   - Adds **Path A** (`OnlineEmaNgramDensityBank`) + **Path F** (`StatefulQuantumDensityHistoryBuffer` across consecutive stream windows) at $L=4$.
4. **`W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED`**:
   - Scales the consolidated pure density-matrix architecture (`Paths A + B + C/D + E + F`) to **$L = 8$ layers** with $\sqrt{4/L} = 1/\sqrt{2}$ geodesic trust-region scaling on all residual-writing Stiefel/Kraus/Unitary operators.

---

## 4. Verification & Testing Plan

1. **Machine-Checked Lean 4 Formalization (`lean/QuantumDLBlocksLean/` & `scripts/verify_lean4_theorems.py`)**:
   - Verify Theorems 1–8 (`8/8` Lean 4 modules) compile via `lake build` with `0` `sorry`, `0` `admit`, and `0` custom `axiom`.
2. **Numerical & Geometric Invariant Inspector (`scripts/pure_density_matrix_inspector.py`)**:
   - Verify:
     1. Exact Hilbert–Schmidt outer-product identity $\langle \tilde{P}(Q Q^\dagger), \tilde{P}(K K^\dagger) \rangle_{\mathbb{R}^{256}} = \|Q^\dagger K\|_F^2 \ge 0$ (`rel_err < 1e-6`) and right $U(R)$-gauge invariance (`rel_err < 1e-6`).
     2. Exact positive attention weights summing to 1 (`max_sum_err < 1e-6`, `min_weight >= 0`) with zero `exp`/`softmax`.
     3. Quantum "OR + AND" density-matrix closure: orthostochastic row sum $\sum_{h'} (W_{\text{OR}, h, h'})^2 = 1$, Schur product positive semidefiniteness $\lambda_{\min}(\rho_A \circ \rho_B) \ge -10^{-7}$, and unit trace $\operatorname{Tr}(\rho_{t,h}^{(\ell)}) = 1$ across all layers $\ell = 0 \dots L$.
     4. Online EMA n-gram density matrix and stateful history buffer trace-1 and PSD preservation across multi-step updates.
     5. Exact probability normalization $\sum_{v=0}^{50256} P(v \mid \rho_t^{(L)}) = 1.000000$ under unconstrained Born readout.
3. **Unit Tests & Linter**:
   - `pytest tests/test_plan089_pure_density_matrix_gpt.py -v` and `scripts/ast_signature_and_length_linter.py`.

---

## 5. Operational Tooling & Script Opportunities

### 5.1 Existing Scripts Used
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py --json-out /home/festivity/agy/QuantumDLBlocks/.cache/lean4_theorems_verification.json`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/benchmarks/pure_density_matrix_gpt_lab.py scripts/pure_density_matrix_inspector.py scripts/pure_density_matrix_bench.py tests/test_plan089_pure_density_matrix_gpt.py`

### 5.2 Build-Before-Launch Scripts
- [`scripts/pure_density_matrix_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/pure_density_matrix_inspector.py): Audits all 5 Pure Density-Matrix invariants (Theorem 6 Hilbert–Schmidt positivity & $U(R)$ invariance, Theorem 7 Schur "OR + AND" PSD/trace closure, Theorem 8 Online EMA & History density preservation, zero-classical-stream verification, and `50,257`-way normalization).
- [`scripts/pure_density_matrix_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/pure_density_matrix_bench.py): Standardized CLI runner for `W69_P0`..`W69_P3` supporting `150M+`-token trajectories with `StandardizedExperimentLogger`.

### 5.3 Nice-to-Have Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
- `scripts/density_matrix_spectrum_and_entanglement_profiler.py`: Profiles per-layer Von Neumann / Rényi-2 purity $\operatorname{Tr}(\rho_{t,h}^2)$, cross-head mutual coherence, and Schur coincidence activation ratios over long-horizon runs.

---

## 6. Constants Ledger (Rule 7) & Empirical Results

### 6.1 Top-of-File Named Constants (`src/qdlblocks/benchmarks/pure_density_matrix_gpt_lab.py`)

| Constant Name | Value | Mathematical / Physical Rationale |
| :--- | :--- | :--- |
| `DEFAULT_PURE_RHO_HEADS` | `16` | Number of quantum heads $H = 16$ |
| `DEFAULT_PURE_RHO_DIM` | `16` | Per-head Hilbert dimension $d_h = 16$ ($\mathbb{C}^{16}$, $\operatorname{Herm}(\mathbb{C}^{16}) \cong \mathbb{R}^{256}$) |
| `DEFAULT_PURE_CHOLESKY_RANK` | `4` | Rank $R = 4$ of the Cholesky/purification factor $L_{t,h} \in \mathbb{C}^{16 \times 4}$ |
| `DEFAULT_PURE_PACKED_DIM` | `256` | Isometric packed Hermitian dimension $d_h^2 = 256$ |
| `DEFAULT_PURE_OBS_PER_HEAD` | `32` | Per-head observable dimension ($16 \times 32 = 512\text{D}$ total readout state) |
| `DEFAULT_HS_COINCIDENCE_BETA_INIT` | `0.35` | Initial 2-boson Hilbert–Schmidt coincidence sharpening weight $\beta_{\text{coinc}}$ |
| `DEFAULT_HS_VACUUM_FLOOR_INIT` | `0.08` | Positive vacuum no-click denominator floor $\varepsilon_{\text{vac},h} > 0$ in `exp`-free attention |
| `DEFAULT_SCHUR_AND_MIX_INIT` | `0.30` | Initial weight $\lambda_{\text{and}}$ on the Schur coincidence density matrix $\rho^{(A)} \circ \rho^{(B)}$ |
| `DEFAULT_ONLINE_NGRAM_EMA_DECAY` | `0.04` | Online causal EMA update rate $\eta_{\text{ngram}}$ for observed byte/n-gram density matrices |
| `DEFAULT_HISTORY_SLOTS` | `8` | Number of multi-timescale stateful summary density matrices $M_{\text{hist}} = 8$ per head/layer |
| `DEFAULT_HISTORY_MIN_DECAY` | `0.015` | Slowest window EMA update rate $\gamma_1$ (`~67`-window = `34k`-token memory horizon) |
| `DEFAULT_HISTORY_MAX_DECAY` | `0.50` | Fastest window EMA update rate $\gamma_8$ (`2`-window = `1,024`-token memory horizon) |
| `DEFAULT_UNCONSTRAINED_LOGIT_MAX` | `80.0` | Numerical safety clamp (`[-80, 80]`) replacing the `30.0 * tanh(z / 30.0)` softcap |
| `DEFAULT_WIDE_SCHUR_HEADS` | `32` | Wide intermediate $32$-head (`1024D`) Stiefel subsystem lifting $\operatorname{St}(1024, 512)$ inside `apply_quantum_or_and_schur_logic` |
| `PURE_RHO_TRACE_EPS` | `1e-6` | Numerical floor for trace and Frobenius normalization |

### 6.2 Empirical Results (Lean 4 Machine-Checked Proofs, CUDA Inspector, & 60-Step Ablation)

1. **Machine-Checked Lean 4 Formalization (`/home/festivity/agy/QuantumDLBlocks/lean/` & `scripts/verify_lean4_theorems.py`)**:
   - All **8 Lean 4 modules (`22/22` theorems)** compile cleanly in `4.40s` (`lake build` exit code `0`, `0` `sorry`, `0` `admit`, `0` custom `axiom`).
2. **CUDA Numerical & Geometric Invariant Inspector (`scripts/pure_density_matrix_inspector.py`, `.cache/plan089_pure_density_inspector.json`)**:
   - `overall_passed = true`:
     - **Theorem 6 (Hilbert–Schmidt & Right $U(R)$ Invariance)**: `isometric_overlap_rel_err = 2.15e-07`, `ur_gauge_invariance_rel_err = 4.84e-07`, `min_attention_weight = 0.0`, `causal_upper_tri_leak = 0.0`.
     - **Theorem 7 (Quantum "OR + AND" Schur Closure)**: `orthostochastic_row_sum_err = 4.77e-07`, `cholesky_unit_frob_err = 2.38e-07`, `hermitian_symmetry_err = 0.0`, `unit_trace_err = 1.19e-07`, `min_psd_eigenvalue = +0.00194 >= 0`.
     - **Theorem 8 (Online EMA & Stateful History Density Preservation)**: `online_ema_trace_err = 1.19e-07`, `online_ema_min_eigval = +0.05625`, `history_trace_err = 1.19e-07`, `history_min_eigval = +0.04000`.
     - **Unconstrained Born Readout & Gauge-Free Step**: `custom_autograd_max_rel_err = 2.04e-07`, `vocab_50257_prob_sum_err = 1.19e-07`, `su16_step_phase_frac = 4.15e-09`, `kraus_step_gauge_frac = 1.14e-07`.
3. **60-Step Rapid Ablation Across `W69_P0..W69_P3` (`.cache/plan089_wave69_pure_density_60step_summary.json`)**:

| Variant | Layers ($L$) | Params | Step 0 Val CE | Step 20 Hot CE | Step 40 EMA CE | Step 60 EMA CE | Step 60 Top-1 | `MUBErr` | `SU16Phase` / `KrausGauge` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W69_P0_PURE_DENSITY_HS_ATTN_4L` | `4` | `26.90M` | `8.9060` | `7.7800` | `7.3866` | `7.2603` | `10.95%` | `6.00e-08` | `0.00e+00` / `0.00e+00` |
| `W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L` | `4` | `26.90M` | `8.9060` | `7.7719` | `7.3834` | **`7.2569`** | `10.97%` | `6.00e-08` | `0.00e+00` / `0.00e+00` |
| `W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L` | `4` | `26.90M` | `8.9060` | `7.7744` | `7.3840` | `7.2579` | `10.96%` | `5.00e-08` | `0.00e+00` / `0.00e+00` |
| `W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED` | `8` | `37.68M` | `8.9067` | **`7.7453`** | `7.3914` | `7.2680` | **`11.03%`** | `5.00e-08` | `0.00e+00` / `0.00e+00` |

