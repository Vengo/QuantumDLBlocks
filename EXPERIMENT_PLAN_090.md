# Experiment Plan 090 (`Wave 70`): Factored Cross-Head Kraus Superoperator, Attention-Sum Commutation, & Fast LUT-Packed Pure Density-Matrix GPT

## 1. Executive Summary & Architectural Motivation

While `Wave 69` (`W69_P2`, [`EXPERIMENT_PLAN_089.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_089.md)) established the 100% Pure Density-Matrix GPT backbone (`8.9060 -> 5.7575` EMA Val CE in `24.58M` tokens at `3,716 tok/s` with `MUBErr = 6.00e-08`, `SU16Phase = 0.00e+00`, `KrausGauge = 0.00e+00`), two high-leverage questions emerged:
1. **Cross-Head Kraus Superoperator Without $(H, H, M, 16, 16)$ Tensor Explosion**:
   - In `Wave 69`, full cross-head basis-plus-head routing (`w_q/k/v/o_stiefel` $\in \operatorname{O}(512)$ and `w_schur_up/down` $\in \operatorname{St}(1024, 512)$) operated on the Rank-$R$ ($R=4$) Cholesky/purification factors $L_{t,h} \in \mathbb{C}^{16 \times 4}$, whereas the full-rank incoherent attention mixture $\rho_{t,h'}^{\text{attn}} = \sum_{s \le t} A_{t,s}^{(h')} \rho_{s,h'}^V \in \mathcal{D}(\mathbb{C}^{16})$ was only mixed across heads via scalar orthostochastic weights $(W_{\text{OR}})_{h,h'}^2$.
   - A naive pairwise Cross-Head Kraus Superoperator $\rho_{t,h}^{\text{mix}} = \sum_{h'=1}^H \sum_{m=1}^M W_{h,h',m}^2 K_{h,h',m} \rho_{t,h'}^{\text{attn}} K_{h,h',m}^\dagger$ would materialize an $O(B \cdot T \cdot H^2 \cdot M \cdot d_h^2)$ 7D tensor (`1,024` complex $16 \times 16$ congruences per token).
   - **Wave 70 Resolution (`Theorems 9 & 10`)**: By factoring $K_{h,h',m} = K_{\text{out},h,m}\, K_{\text{in},h',m}$ and applying **associative parenthesization (`Trick A`)** and **causal attention-sum commutation (`Trick B`)**, we compute the **exact same pairwise cross-head Kraus superoperator** in $O(H M)$ steps with **zero 7D tensor materialization**, and combine it with a **Multi-Branch Stiefel Kraus Superoperator** ($\mathcal{K}_1, \dots, \mathcal{K}_M \in \operatorname{O}(512)$) that amplifies a single token's Rank-4 state to a **full-rank ($M \cdot R = 16$) mixed density matrix** on $\mathbb{C}^{16}$.
2. **Eliminating Scatter-Indexing Overhead in Hermitian Packing/Unpacking (`Fast Static-LUT Gather`)**:
   - Replacing the 4-way upper/lower triangular index-scatter writes in `unpack_isometric_real_to_hermitian` / `pack_hermitian_to_isometric_real` with precomputed static `256`-element gather LUTs eliminates in-place scatter bottlenecks during forward and activation-checkpointed backward passes.

---

## 2. Machine-Checked Lean 4 Theorems (`lean/QuantumDLBlocksLean/FactoredCrossHeadKrausSuperoperator.lean`)

1. **Theorem 9a (`factored_kraus_associative_parenthesization_eq`)**:
   For any input density matrices $\rho_{h'} \in \operatorname{Herm}(\mathbb{C}^{d_h})$, input-head operators $K_{\text{in},h',m}$, output-head operators $K_{\text{out},h,m}$, and non-negative routing weights $w_{h,h',m} = W_{h,h',m}^2 \ge 0$, the 3-stage factored parenthesization ($O(H M)$ congruences):
   $$\mathcal{E}_{\text{factored}}(\boldsymbol{\rho})_h = \sum_{m=1}^{M} K_{\text{out},h,m} \left( \sum_{h'=1}^{H} w_{h,h',m} \left( K_{\text{in},h',m}\, \rho_{h'}\, K_{\text{in},h',m}^\dagger \right) \right) K_{\text{out},h,m}^\dagger$$
   is identically equal to the naive $O(H^2 M)$ pairwise congruence loop with composite operators $K_{h,h',m} = K_{\text{out},h,m}\, K_{\text{in},h',m}$:
   $$\mathcal{E}_{\text{naive}}(\boldsymbol{\rho})_h = \sum_{h'=1}^{H} \sum_{m=1}^{M} w_{h,h',m}\, \left(K_{\text{out},h,m} K_{\text{in},h',m}\right)\, \rho_{h'}\, \left(K_{\text{out},h,m} K_{\text{in},h',m}\right)^\dagger$$
2. **Theorem 9b (`kraus_commutes_with_causal_attention_sum` & `isometric_packed_routing_linearity`)**:
   - Because $K_{\text{in},h',m}$ does not depend on sequence position $s$, applying $K_{\text{in},h',m}(\cdot)K_{\text{in},h',m}^\dagger$ to the post-attention density matrix $\rho_{t,h'}^{\text{attn}} = \sum_{s \le t} A_{t,s}^{(h')} V_{s,h'} V_{s,h'}^\dagger$ is identically equal to multiplying the Rank-$R$ Cholesky factors $K_{\text{in},h',m} V_{s,h'}$ before the attention sum:
     $$K_{\text{in},h',m} \left(\sum_{s \le t} A_{t,s}^{(h')} V_{s,h'} V_{s,h'}^\dagger\right) K_{\text{in},h',m}^\dagger = \sum_{s \le t} A_{t,s}^{(h')} \big(K_{\text{in},h',m} V_{s,h'}\big)\big(K_{\text{in},h',m} V_{s,h'}\big)^\dagger$$
   - Furthermore, linear routing $\sum_{h'=1}^H w_{h,h',m} \tilde{P}(\sigma_{h',m})$ in the Frobenius-isometric packed space $\mathbb{R}^{d_h^2}$ unpacks to the exact matrix linear combination $\sum_{h'=1}^H w_{h,h',m} \sigma_{h',m}$.
3. **Theorem 10a (`factored_cross_head_kraus_cptp_closure`)**:
   If each $\rho_{h'} \in \mathcal{D}(\mathbb{C}^{d_h})$ is Hermitian, positive semidefinite, and trace-1, $K_{\text{in},h',m}, K_{\text{out},h,m} \in U(d_h)$ are unitary, and $\sum_{m=1}^M \sum_{h'=1}^H w_{h,h',m} = 1$, then $\mathcal{E}_{\text{factored}}(\boldsymbol{\rho})_h \in \mathcal{D}(\mathbb{C}^{d_h})$ is strictly Hermitian, positive semidefinite, and unit-trace.
4. **Theorem 10b (`representational_power_strict_separation_and_rank_amplification`)**:
   - **Strict Expressivity Separation**: Scalar orthostochastic mixing $\rho_h^{\text{scalar}} = \sum_{h'} B_{h,h'} \rho_{h'}$ preserves the common kernel of the input heads ($\forall h',\; v^\dagger \rho_{h'} v = 0 \implies v^\dagger \rho_h^{\text{scalar}} v = 0$), so it can never transfer quantum probability mass into orthogonal basis states. By contrast, `FactoredCrossHeadKrausSuperoperator` can transfer probability mass from inputs supported strictly on $|0\rangle\dots|R-1\rangle$ into all $d_h = 16$ basis states and amplify single-token rank from $R \to \min(d_h, M \cdot R) = 16$.

---

## 3. Planned Module & Variants (`src/qdlblocks/benchmarks/factored_kraus_density_gpt_lab.py`)

| Variant | Layers ($L$) | Cross-Head Kraus Mode | Fast LUT Pack/Unpack | Multi-Branch Stiefel Kraus ($M=4$) | Target Hypothesis Tested |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`W70_K0_LUT_PACKED_BASELINE_4L`** | `4` | Scalar Orthostochastic | **Yes** | No | Verify identical mathematics to `W69_P2` with faster LUT gather packing |
| **`W70_K1_FACTORED_KRAUS_TRICK_A_4L`** | `4` | **Factored Associative (`Trick A`, $M=4$)** | **Yes** | No | Test post-attention $O(H M)$ factored Cross-Head Kraus Superoperator on $\rho_{t,h'}^{\text{attn}}$ |
| **`W70_K2_COMMUTED_AND_MULTIBRANCH_4L`** | `4` | **Attention-Commuted (`Trick B`) + Multi-Branch Stiefel ($M=4$)** | **Yes** | **Yes (`+1.05M`/layer)** | Test Rank-$16$ ($M \cdot R = 16$) cross-head Kraus superoperator + `Trick B` |
| **`W70_K3_DEEP_8L_FACTORED_KRAUS`** | `8` | **Attention-Commuted (`Trick B`) + Multi-Branch Stiefel ($M=4$)** | **Yes** | **Yes (`+1.05M`/layer)** | Scale depth to $L=8$ (`~46.1M` total parameters) with $1/\sqrt{L}$ geodesic step scaling |

---

## 4. Verification Criteria

1. **Machine-Checked Lean 4 Proof (`lean/QuantumDLBlocksLean/FactoredCrossHeadKrausSuperoperator.lean`)**:
   - Compiles via `lake build` and `scripts/verify_lean4_theorems.py` with `0` `sorry`, `0` `admit`, and `0` custom `axiom`.
2. **Numerical Equivalence, Speed, & Expressivity Inspector (`scripts/factored_kraus_density_inspector.py`)**:
   - Verify:
     1. Exact equivalence between `apply_factored_associative_cross_head_kraus` (`Trick A`), `apply_attention_commuted_cross_head_kraus` (`Trick B`), and `apply_naive_pairwise_cross_head_kraus` ($O(H^2 M)$ 7D reference) (`max_rel_err < 1e-6`).
     2. Exact equivalence between `fast_pack_hermitian_to_isometric_real` / `fast_unpack_isometric_real_to_hermitian` and the `Wave 69` reference packing (`max_abs_err == 0.0`).
     3. Strict Representational Power & Rank-Amplification Verification (`Theorem 10b`): when all 16 input heads are Rank-4 states confined to basis indices $\{0, 1, 2, 3\}$, scalar orthostochastic mixing has `0.0` energy on indices $\{4..15\}$ and rank $4$, whereas `FactoredCrossHeadKrausSuperoperator` populates all 16 basis states (`orthogonal_subspace_mass > 0.5`) and achieves full **Rank `16`** (`min_eigval > 1e-3`).
     4. Latency benchmark confirming `Trick A` and `Trick B` achieve `> 8x` speedup over the naive $O(H^2 M)$ congruence loop.
3. **Unit Tests & Linter**:
   - `pytest tests/test_plan090_factored_kraus_density_gpt.py -q` and `scripts/ast_signature_and_length_linter.py`.

---

## 5. Operational Tooling & Script Opportunities

### 5.1 Existing Scripts Used
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py --json-out /home/festivity/agy/QuantumDLBlocks/.cache/lean4_theorems_verification.json`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/benchmarks/factored_kraus_density_gpt_lab.py scripts/factored_kraus_density_inspector.py scripts/factored_kraus_density_bench.py tests/test_plan090_factored_kraus_density_gpt.py`

### 5.2 Build-Before-Launch Scripts
- [`scripts/factored_kraus_density_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/factored_kraus_density_inspector.py): Verifies Theorems 9a, 9b, 10a, and 10b numerically on CPU/CUDA, benchmarks `Naive O(H^2 M)` vs. `Trick A O(H M)` vs. `Trick B Commuted` vs. `Multi-Branch Stiefel Kraus`, and writes `.cache/plan090_factored_kraus_inspector.json`.
- [`scripts/factored_kraus_density_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/factored_kraus_density_bench.py): Standardized CLI runner for `W70_K0..W70_K3`.

### 5.3 Nice-to-Have Scripts
- `scripts/kraus_rank_and_subspace_transfer_profiler.py`: Profiles the empirical effective rank ($\exp(S_{\text{vN}}(\rho_{t,h}))$) and basis-transfer entropy across layers during training.

---

## 6. Constants Ledger (Rule 7) & Empirical Results

### 6.1 Top-of-File Named Constants (`src/qdlblocks/benchmarks/factored_kraus_density_gpt_lab.py`)

| Constant Name | Value | Mathematical / Physical Rationale |
| :--- | :--- | :--- |
| `DEFAULT_FACTORED_KRAUS_BRANCHES` | `4` | Number of cross-head Kraus branches $M = 4$ ($M \cdot R = 4 \times 4 = 16 = d_h$ full-rank capacity) |
| `DEFAULT_FACTORED_KRAUS_BLEND_INIT` | `0.25` | Initial convex blend weight for the Cross-Head Kraus Superoperator attention output |
| `DEFAULT_MULTIBRANCH_KRAUS_BLEND_INIT` | `0.22` | Initial convex blend weight for the $M=4$ Multi-Branch Stiefel Kraus density state |
| `FULL_RANK_EIGVAL_THRESHOLD` | `1e-5` | Eigenvalue threshold for certifying numerical rank on $\mathcal{D}(\mathbb{C}^{16})$ |
| `DEFAULT_RESTORED_MOMENTUM_WARMUP_START` | `0.85` | Initial Riemannian momentum persistence $\beta_0 = 0.85$ at Step 0 |
| `DEFAULT_RESTORED_MOMENTUM_PLATEAU_BETA` | `0.95` | Restored high-persistence Riemannian quotient momentum $\beta_{\text{plateau}} = 0.95$ (`Theorem 11`) |
| `DEFAULT_RESTORED_MOMENTUM_QUENCH_END` | `0.97` | Final cosine-quench Riemannian momentum persistence $\beta_{\text{quench}} = 0.97$ |
| `DEFAULT_MOMENTUM_WARMUP_STEPS` | `300` | Warmup horizon (`300` steps) over which $\beta_t$ ramps linearly from `0.85 -> 0.95` |

### 6.2 Empirical Results (Lean 4 Machine-Checked Modules 9 & 10 + Numerical Inspector)

1. **Machine-Checked Lean 4 Formalization ([`FactoredCrossHeadKrausSuperoperator.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/FactoredCrossHeadKrausSuperoperator.lean) & [`RiemannianMomentumBufferEffectiveness.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/RiemannianMomentumBufferEffectiveness.lean))**:
   - All **10 theorem modules (`31/31` theorems across 13 `.lean` files)** compile via `lake build` and [`scripts/verify_lean4_theorems.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py) with `0` `sorry`, `0` `admit`, and `0` custom `axiom` (`.cache/lean4_theorems_verification.json`).
   - **Theorem 11 (`Theorem 11a, 11b, 11c`)** formally proves that our restored high-persistence Riemannian momentum buffer ($\beta \in [0.85, 0.95]$) maintains:
     1. **Exact Quotient Horizontality (`momentum_buffer_horizontal_invariance`)**: $\Pi_{\mathcal{V}}(M_{\text{clean}}) = 0$, $\Pi_{\mathcal{V}}(M_t) = 0$, $\Pi_{\mathcal{V}}(\Xi_t) = 0$, and $\Pi_{\mathcal{V}}(\Pi_{\mathcal{H}}(\Phi(\Xi_t))) = 0$.
     2. **Opposing-Momentum Cleaning & Strict Descent (`opposing_momentum_cleaning_descent_and_energy_bound` & `uncleaned_momentum_ascent_witness`)**: Projecting out negative gradient alignment when $\langle M_{t,\text{pre}}, G_t\rangle < 0$ guarantees $\langle M_{t,\text{clean}}, G_t\rangle \ge 0$, Pythagorean energy contraction $\|M_{t,\text{clean}}\|^2 \le \|M_{t,\text{pre}}\|^2$, and strict first-order descent $\langle \Xi_t, G_t\rangle \ge (1 + \beta)\|G_t\|^2 = 1.95\|G_t\|^2 > 0$ at $\beta = 0.95$ (whereas uncleaned opposing momentum at $\beta = 0.95$ strictly ascends: $\langle \Xi_{\text{unclean}}, G_t\rangle < 0$).
     3. **Isometric Parallel Transport Conservation (`isometric_parallel_transport_preserves_horizontality_and_norm`)**: Rescaled horizontal midpoint transport preserves both new-point horizontality and kinetic energy $\|\mathcal{T}(M_t)\|_F^2 = \|M_t\|_F^2$.
2. **Numerical Equivalence, Speedup, Representational Power, & Momentum Audit ([`scripts/factored_kraus_density_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/factored_kraus_density_inspector.py), `.cache/plan090_factored_kraus_inspector.json`)**:
   - `overall_passed = true` across all 5 audits (`5/5` unit tests passing in [`tests/test_plan090_factored_kraus_density_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan090_factored_kraus_density_gpt.py)):
     - **Theorem 9a & 9b**: `trick_a_vs_naive_rel_err = 1.12e-07`, `trick_b_vs_naive_rel_err = 1.13e-07` (**`29.75x` speedup** over naive $O(H^2 M)$ congruence loop).
     - **Theorem 10a & 10b**: `scalar_orthogonal_subspace_mass = 0.0` (`Rank 4`) vs. `factored_kraus_orthogonal_subspace_mass = 0.7497` (`Rank 16`) and `multibranch_kraus_orthogonal_subspace_mass = 0.7492` (`Rank 16`).
     - **Theorem 11 (Restored `0.95` Momentum Buffer Audit)**: `plateau_momentum_beta = 0.95`, `uncleaned_opposing_descent_ratio = -0.3062 < 0` (ascent) vs. `cleaned_opposing_descent_ratio = +1.9500` (strict descent), `cleaned_orthogonality_err = 4.82e-07`, `cleaned_pythagorean_err = 2.53e-07`, `transported_norm_rel_err = 0.0`, `max_su16_mom_vertical_leak = 1.19e-08`, `max_kraus_mom_vertical_leak = 3.98e-08`, `active_stiefel_mom_buffers = 42`, `active_su16_mom_buffers = 29`.
3. **Training Trajectory (`W70_K2_COMMUTED_AND_MULTIBRANCH_4L`, `L=4`, `31,361,258` Params, Restored `0.85 -> 0.95` Momentum Buffer)**:
   - Reached **`Step 5,550` (`45.47M` tokens)** on the local RTX 5060 Ti before user-requested cancellation:
     - `Step 0` (`0.00M` tok): `Val CE = 8.9048`
     - `Step 1,500` (`12.29M` tok, `mom = 0.950`): `Hot CE = 6.0330`, `EMA CE = 5.9844`, `Top-1 = 16.87%`
     - `Step 3,000` (`24.58M` tok, `mom = 0.950`): `Hot CE = 5.8044`, `EMA CE = 5.7576`, `Top-1 = 19.18%`
     - `Step 4,500` (`36.86M` tok, `mom = 0.950`): `Hot CE = 5.6662`, `EMA CE = 5.6168`, `Top-1 = 20.15%`
     - `Step 5,400` (`44.24M` tok, `mom = 0.950`): `Hot CE = 5.6369`, **`EMA CE = 5.5588`**, **`Top-1 = 20.78%`**
     - `Step 5,550` (`45.47M` tok, `mom = 0.950`): **`Hot CE = 5.6083`**, `EMA CE = 5.5642`, `Top-1 = 20.71%`, with exact manifold & quotient invariants throughout (`MUBErr = 5.00e-08..6.00e-08`, `SU16Phase = 0.00e+00`, `KrausGauge = 0.00e+00`).


