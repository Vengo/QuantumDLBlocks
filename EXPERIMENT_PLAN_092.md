# EXPERIMENT_PLAN_092: Grouped 4-Qubit ($\mathbb{C}^{16} = (\mathbb{C}^2)^{\otimes 4}$) Kronecker Density Preparation & Multi-Branch $16^4 = 65,536\text{D}$ Coincidence Readout (`Wave 72`)

## 1. Executive Summary & Architectural Motivation

In Waves 69–71 ([`PureDensityMatrixGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/pure_density_matrix_gpt_lab.py), [`FactoredKrausPureDensityGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/factored_kraus_density_gpt_lab.py), and [`AnchoredBipartitePureDensityGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/anchored_bipartite_density_gpt_lab.py)), each head evolves a $16 \times 16$ complex density matrix $\rho_{t,h}^{(\ell)} \in \mathcal{D}(\mathbb{C}^{16})$ across $H = 16$ heads. However:
1. **Unstructured Layer-0 Preparation in $\mathbb{C}^{16}$**: At Layer 0, each head's ket $|\psi_{t,h}^{(0)}\rangle \in \mathbb{C}^{16}$ was prepared by slicing a flat dense projection $Q_{\text{in}} w_v \in \mathbb{R}^{512}$ rather than preparing an explicit **4-qubit Kronecker product state** in $(\mathbb{C}^2)^{\otimes 4} \cong \mathbb{C}^{16}$.
2. **Additive Direct-Sum ($\bigoplus_{h=1}^{16} \mathbb{C}^{16}$) vs. Multiplicative Kronecker ($\bigotimes_{g=1}^4 \mathbb{C}^{16} = \mathbb{C}^{65,536}$) Readout**: Because $2^{16} = (2^4)^4 = 16^4 = 65,536 > 50,257$, grouping 16 qubits into $G = 4$ blocks of $Q = 4$ qubits maps every token into $\mathcal{H}_{\text{joint}} = (\mathbb{C}^{16})^{\otimes 4} = \mathbb{C}^{65,536}$ where all `50,257` tokens can be mutually orthogonal (`0` Welch-bound crowding). Moreover, partitioning our $H = 16$ heads into $B = 4$ branches $\times$ $G = 4$ Kronecker factors forms a **Rank-$B$ entangled mixture across the $65,536\text{D}$ tensor space**:
   $$\boldsymbol{\rho}_t^{\text{joint}} = \sum_{b=1}^{4} w_{t,b} \left(\rho_{t,(b,1)}^{(L)} \otimes \rho_{t,(b,2)}^{(L)} \otimes \rho_{t,(b,3)}^{(L)} \otimes \rho_{t,(b,4)}^{(L)}\right) \in \mathcal{D}\!\left(\mathbb{C}^{65,536}\right),$$
   which completely eliminates the **Mean-Field Superposition Collapse** of single-product states while never materializing a $65,536\text{D}$ tensor in memory.

---

## 2. Machine-Checked Lean 4 Theorems (`lean/QuantumDLBlocksLean/GroupedFourQubitKroneckerDensity.lean` — `Module 12`)

1. **Theorem 13a (`four_qubit_segre_norm_and_intra_head_entanglement`)**:
   - For any four unit-norm single-qubit spinors $|\psi_q\rangle \in \mathbb{C}^2$ ($\|\psi_q\|_2^2 = 1$, $q \in \{1,2,3,4\}$), the 4-qubit Segre product $|\Phi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes |\psi_3\rangle \otimes |\psi_4\rangle \in (\mathbb{C}^2)^{\otimes 4} \cong \mathbb{C}^{16}$ has exact unit norm $\|\Phi\|_2^2 = \prod_{q=1}^4 \|\psi_q\|_2^2 = 1$ and unit-trace outer product $\operatorname{Tr}(|\Phi\rangle\langle\Phi|) = 1$.
   - While any Layer-0 product state has pure reduced qubit marginals ($\operatorname{Tr}(\rho_A^2) = 1$, zero intra-head entanglement), applying an intra-head entangling unitary $U \in SU(16)$ produces an entangled state whose reduced marginal has purity $\operatorname{Tr}(\rho_A^2) = \tfrac{1}{2} < 1$.
2. **Theorem 13b (`mean_field_single_branch_ghz_collapse_vs_multibranch_exactness`)**:
   - On a bimodal target distribution $P_{\text{target}}(0,0,0,0) = \tfrac{1}{2}, P_{\text{target}}(1,1,1,1) = \tfrac{1}{2}$ across $G = 4$ factors, a single-branch ($B=1$) product of marginals $\rho^{\text{MF}} = \bigotimes_{g=1}^4 (\tfrac{1}{2}|0\rangle\langle 0| + \tfrac{1}{2}|1\rangle\langle 1|)$ collapses to $P_{\text{MF}}(0,0,0,0) = P_{\text{MF}}(1,1,1,1) = \tfrac{1}{16}$ (leaking $\tfrac{14}{16} = 87.5\%$ of the probability mass onto invalid cross-terms), whereas a $B \ge 2$ branch Kronecker density mixture $\boldsymbol{\rho}^{\text{joint}} = \sum_{b=1}^B w_b \bigotimes_{g=1}^4 \rho_{(b,g)}$ recovers $P(0,0,0,0) = \tfrac{1}{2}$, $P(1,1,1,1) = \tfrac{1}{2}$, and $P_{\text{cross}} = 0$ identically while preserving $\operatorname{Tr}(\boldsymbol{\rho}^{\text{joint}}) = 1$.
3. **Theorem 13c (`factorized_65536d_hilbert_schmidt_born_coincidence_identity`)**:
   - For $\boldsymbol{\rho}^{\text{joint}} = \sum_{b=1}^B w_b \bigotimes_{g=1}^4 (L_{(b,g)} L_{(b,g)}^\dagger) \in \mathcal{D}(\mathbb{C}^{65,536})$ and $E_v = \bigotimes_{g=1}^4 |\Phi_{v,g}\rangle\langle\Phi_{v,g}|$, the $65,536\text{D}$ Hilbert–Schmidt Born overlap factors without materializing $\mathbb{C}^{65,536}$:
     $$\operatorname{Tr}\!\left(\boldsymbol{\rho}^{\text{joint}} E_v\right) = \sum_{b=1}^{B} w_b \prod_{g=1}^{4} \left\| L_{(b,g)}^\dagger |\Phi_{v,g}\rangle \right\|_2^2 \ge 0.$$

---

## 3. Planned Architectural Variants (`src/qdlblocks/benchmarks/grouped_4qubit_kronecker_density_gpt_lab.py`)

| Variant Mode | Layer-0 Grouped 4-Qubit $(\mathbb{C}^2)^{\otimes 4} \to \mathbb{C}^{16}$ Preparation | $B=4$ Branch $\times$ $G=4$ Factor $65,536\text{D}$ Coincidence Readout | Layers $L$ | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| `W72_Q0_ANCHORED_BIPARTITE_REF_4L` | Off (Wave 71 `W71_A2` baseline) | Off | `4` | Exact Wave 71 reference |
| `W72_Q1_GROUPED_4QUBIT_SEGRE_PREP_4L` | **On** (`16` qubits $\to 4 \times (\mathbb{C}^2)^{\otimes 4} = 4 \times \mathbb{C}^{16}$) | Off | `4` | Isolates Grouped 4-Qubit Segre Layer-0 state preparation |
| `W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L` | **On** (`16` qubits $\to 4 \times (\mathbb{C}^2)^{\otimes 4}$) | **On** ($B=4 \times G=4$ in $\mathcal{D}(\mathbb{C}^{65,536})$) | `4` | Full Wave 72 4L architecture |
| `W72_Q3_GROUPED_4QUBIT_MULTIBRANCH_KRON_8L` | **On** | **On** | `8` | Deep 8-layer grouped 4-qubit Kronecker density trunk |

---

## 4. Verification & Testing Plan

1. **Lean 4 Proof Verification (`scripts/verify_lean4_theorems.py`)**:
   - Compile `lean/QuantumDLBlocksLean/GroupedFourQubitKroneckerDensity.lean` and verify `38/38` machine-checked theorems across `15` `.lean` files with `0` `sorry`, `0` `admit`, and `0` custom `axiom`.
2. **CUDA Numerical Inspector (`scripts/grouped_4qubit_kronecker_inspector.py`)**:
   - Verify Theorem 13a: 4-qubit Segre product unit norm (`segre_norm_err < 1e-6`), Step-0 separable 2-qubit partial-trace purity $\mathcal{P}_{2\text{-qubit}}(\Phi^{(0)}) = 1.000000$, and strictly entangled post-$SU(16)$ purity $\mathcal{P}_{2\text{-qubit}}(U\Phi^{(0)}) < 0.75$, plus exact mutual orthogonality of distinct binary 16-qubit token codewords in $\mathbb{C}^{65,536}$ (`ortho_overlap == 0.0`).
   - Verify Theorem 13b: Mean-field single-branch collapse (`mf_target_prob == 0.0625`, `mf_cross_bleed == 0.875`) vs. $B=2$ multi-branch Kronecker density mixture (`mb_target_prob == 0.5000`, `mb_cross_bleed == 0.0`, `mb_trace_err < 1e-6`).
   - Verify Theorem 13c: Exact numerical equivalence between explicit $65,536\text{D}$ tensor-product trace $\operatorname{Tr}(\boldsymbol{\rho}^{\text{joint}} E_v)$ and the zero-materialization factorized formula $\sum_{b=1}^B w_b \prod_{g=1}^4 \|L_{(b,g)}^\dagger |\Phi_{v,g}\rangle\|_2^2$ (`factorized_65536d_max_err < 1e-6`).
   - Verify `GroupedFourQubitKroneckerDensityGPT` forward/backward and `PureDensityQuotientMuon` step invariants.

---

## 5. Operational Tooling & Script Opportunities

1. **Existing Scripts Used**:
   - [`scripts/verify_lean4_theorems.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py)
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
2. **Build-Before-Launch Scripts**:
   - [`scripts/grouped_4qubit_kronecker_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/grouped_4qubit_kronecker_inspector.py)
   - [`scripts/grouped_4qubit_kronecker_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/grouped_4qubit_kronecker_bench.py)
3. **Nice-to-Have Scripts**:
   - Per-layer 4-qubit entanglement entropy & $65,536\text{D}$ Kronecker branch participation ratio tracker across training checkpoints.

---

## 6. Constants Ledger (Rule 7) & Empirical Verification Results

| Constant Name | Value | Mathematical / Physical Role |
| :--- | :---: | :--- |
| `DEFAULT_KRONECKER_GROUPS` | `4` | Number of 4-qubit Kronecker factors $G = 4$ ($\mathbb{C}^{16} \otimes \mathbb{C}^{16} \otimes \mathbb{C}^{16} \otimes \mathbb{C}^{16} = \mathbb{C}^{65,536}$) |
| `DEFAULT_QUBITS_PER_GROUP` | `4` | Number of qubits $Q = 4$ per head factor ($(\mathbb{C}^2)^{\otimes 4} \cong \mathbb{C}^{16}$, $G \times Q = 16$ qubits total) |
| `DEFAULT_KRONECKER_BRANCHES` | `4` | Number of parallel multi-meaning heads $B = H / G = 16 / 4 = 4$ per Kronecker factor |
| `DEFAULT_SEGRE_QUBIT_BLEND_INIT` | `0.65` | Initial blend weight for the 4-qubit Segre Kronecker product state $|\Phi_{t,g}^{(0)}\rangle \in (\mathbb{C}^2)^{\otimes 4}$ at Layer 0 |
| `DEFAULT_KRON_COINCIDENCE_SCALE_INIT` | `0.45` | Initial scale for the $B=4$ branch $\times$ $G=4$ factor $65,536\text{D}$ Kronecker coincidence readout channel |
| `QUBIT_BINARY_ORTHO_PERTURB_STD` | `0.08` | Smooth Bloch-sphere perturbation around the exact 16-bit Gray/binary orthogonal qubit basis for token $v \in \{0..50256\}$ |

### B. Lean 4 Formalization & Numerical Inspector Verification (`0` Errors)
- **Lean 4 Module 12 ([`lean/QuantumDLBlocksLean/GroupedFourQubitKroneckerDensity.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/GroupedFourQubitKroneckerDensity.lean))**:
  - Verified via [`scripts/verify_lean4_theorems.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py): `15` `.lean` files, **`38/38` machine-checked theorems** (`four_qubit_segre_norm_and_intra_head_entanglement`, `mean_field_single_branch_ghz_collapse_vs_multibranch_exactness`, `factorized_65536d_hilbert_schmidt_born_coincidence_identity`), `0` `sorry`, `0` `admit`, `0` custom `axiom` (`build_time = 6.68s`).
- **Numerical Inspector ([`scripts/grouped_4qubit_kronecker_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/grouped_4qubit_kronecker_inspector.py) — `overall_passed: true`)**:
  - **Theorem 13a (`theorem13a_segre_norm_orthogonality_and_entanglement`)**:
    - 4-qubit Segre product unit norm error: `segre_norm_err = 4.8e-07`.
    - Layer-0 separable 2-qubit partial-trace purity: `separable_purity_mean = 1.000000` (`err = 9.5e-07`).
    - Intra-head post-$SU(16)$ entangled 2-qubit purity: `entangled_post_su16_purity_mean = 0.473151` (`< 1.0`, confirming genuine 4-qubit entanglement inside each head).
    - Exact 16-qubit binary codebook orthogonality in $\mathbb{C}^{65,536}$: `codebook_65536d_ortho_err = 2.15e-06`.
  - **Theorem 13b & 13c (`theorem13bc_multibranch_ghz_and_65536d_factorization`)**:
    - Mean-Field ($B=1$) collapse on bimodal target: `mf_target_0000_prob = 0.0625`, `mf_target_1111_prob = 0.0625`, `mf_cross_bleed_prob = 0.875` (`87.5%` cross-term bleed).
    - Multi-Branch ($B=2$) Kronecker density mixture: `mb_target_0000_prob = 0.500000`, `mb_target_1111_prob = 0.500000`, `mb_cross_bleed_prob = 0.0` (`0%` cross-term bleed).
    - Exact agreement between explicit $65,536\text{D}$ tensor-product trace and factorized Cholesky formula: `factorized_65536d_max_err = 0.0`.
  - **Model & Quotient Optimizer Step (`wave72_model_step_and_quotient`)**:
    - `total_params = 31,533,631` (`31.53M` parameters at $L=4$), `su16_step_phase_frac = 0.0`, `kraus_step_gauge_frac = 1.1e-07`.

### C. `W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L` Training Trajectory (`79.87M` Tokens / `9,750` Steps, Ongoing)
- **Step 0 (`0.00M` tok)**: `Val CE = 8.9096 | Top-1 = 0.46%`
- **Step 150 (`1.23M` tok)**: `Hot CE = 6.9052 | EMA CE = 6.8436 | Top-1 = 12.33%`
- **Step 600 (`4.92M` tok)**: `Hot CE = 6.4716 | EMA CE = 6.4239 | Top-1 = 14.61%`
- **Step 1,500 (`12.29M` tok)**: `Hot CE = 6.1287 | EMA CE = 6.0836 | Top-1 = 16.31%`
- **Step 3,150 (`25.80M` tok)**: `Hot CE = 5.8692 | EMA CE = 5.8132 | Top-1 = 18.03%`
- **Step 4,800 (`39.32M` tok)**: `Hot CE = 5.7446 | EMA CE = 5.6912 | Top-1 = 18.99%`
- **Step 6,000 (`49.15M` tok)**: `Hot CE = 5.6885 | EMA CE = 5.6484 | Top-1 = 19.17%`
- **Step 7,950 (`65.13M` tok)**: `Hot CE = 5.5930 | EMA CE = 5.5524 | Top-1 = 20.17%`
- **Step 9,150 (`74.96M` tok)**: `Hot CE = 5.5562 | EMA CE = 5.5077 | Top-1 = 20.79%`
- **Step 9,450 (`77.41M` tok)**: `Hot CE = 5.5325 | EMA CE = 5.4953 | Top-1 = 20.93% | MUBErr = 5.00e-08 | SU16Phase = 0.00e+00 | KrausGauge = 0.00e+00`
- **Step 9,750 (`79.87M` tok)**: `Hot CE = 5.5420 | EMA CE = 5.4853 | Top-1 = 21.26% | MUBErr = 5.00e-08 | SU16Phase = 0.00e+00 | KrausGauge = 0.00e+00`


