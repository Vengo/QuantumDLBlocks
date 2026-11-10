# Experiment Plan 091 (`Wave 71`): Sensory-Anchored Bipartite Pure Density-Matrix GPT — Dual-Unitary Anchor+Evolved $Q/K/V$ Projections, Bipartite Register Split, & Decoupled Input/Output Frames

## 1. Executive Summary & Architectural Motivation

Auditing the `Waves 69–70` pure density-matrix architecture ([`PureDensityMatrixGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/pure_density_matrix_gpt_lab.py) and [`FactoredKrausPureDensityGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/factored_kraus_density_gpt_lab.py)) revealed three compounding state-collision bottlenecks that explain why `Waves 69–70` slowed down near `5.55` Val CE while `Wave 67` (`W67_S3`, `4.87` Val CE) continued dropping rapidly:

1. **Missing Layer-0 Quantum Sensory Anchor in Layers $1 \dots L$ ("Memoryless State Collision")**:
   - In `PureDensityMatrixGPT.encode_pure_density_trunk`, Layer 0 prepares $\big(L_{t,h}^{(0)}, \tilde{\rho}_{t,h}^{(0)}\big)$ and then Layers $1 \dots L$ overwrite $\big(L_{t,h}^{(\ell-1)}, \tilde{\rho}_{t,h}^{(\ell-1)}\big) \mapsto \big(L_{t,h}^{(\ell)}, \tilde{\rho}_{t,h}^{(\ell)}\big)$ without ever passing $\big(L_{t,h}^{(0)}, \tilde{\rho}_{t,h}^{(0)}\big)$ into deeper blocks.
   - Because `50,257` tokens form an overcomplete frame in $(\mathbb{C}^{16})^{16}$ and quantum channels are memoryless functions of their input matrix, once Layer $\ell$ rotates position $s$'s density matrix $\rho_{s,h}^{(\ell)}$ toward predicting token `B`, Layer $\ell+1$'s $Q/K/V$ projections have **zero mathematical way to distinguish** *"position $s$ originally contained token `B`"* from *"position $s$ contained token `A` and evolved toward predicting `B`"*.
2. **All $R=4$ Cholesky Modes Are Overwritten Together (Erasing Past Keys/Values to Predict Next Tokens)**:
   - Because all $R=4$ columns of $L_{t,h}^{(\ell)} \in \mathbb{C}^{16 \times 4}$ are overwritten together and column `0` (`l_re[..., 0]`) is both initialized to the input token wavepacket `psi0_re` and read out as the dominant next-token ket `dom_re`, rotating $L_{t,h}^{(\ell)}$ toward the next token $x_{t+1}$ actively erases the representation of the current token $x_t$ that future positions $t' > t$ need to attend to.
3. **Self-Copy Attractor from Identical Input & Output Frames ($Q_{\text{in}} = Q_{\text{out}}$ and $Q_{\text{readout}} = I_{512}$)**:
   - Initializing `q_in_ortho` and `q_out_ortho` to the exact same matrix `q_init` while reading out `l_re[..., 0]` through `q_readout_mix = I_{512}` against the tied `w_vocab_256` table causes the Step-0 state $\rho_{t,h}^{(0)}$ to be maximally aligned with predicting **next token = current token ($x_{t+1} = x_t$)**, forcing the trunk to spend capacity erasing the input token just to escape self-copying.

---

## 2. Machine-Checked Lean 4 Theorems (`lean/QuantumDLBlocksLean/SensoryAnchoredBipartiteDensity.lean`)

1. **Theorem 12a (`unanchored_channel_state_collision`)**:
   For any single-argument quantum channel / projection map $\mathcal{E} : \mathcal{D}(\mathcal{H}) \to \mathcal{O}$, if position $s$ with initial token $A$ evolves at Layer $\ell$ into a state $\rho_{s}^{(\ell)}$ that coincides with token $B$'s state $\rho^{(0)}(B)$ ($\rho_{s}^{(\ell)} = \rho^{(0)}(B)$), then $\mathcal{E}(\rho_{s}^{(\ell)}) = \mathcal{E}(\rho^{(0)}(B))$—their $Q/K/V$ projections are identically equal and indistinguishable.
2. **Theorem 12b (`dual_unitary_sensory_anchored_distinguishability` & `dual_unitary_sensory_anchored_cptp`)**:
   For the dual-unitary sensory-anchored channel:
   $$\mathcal{E}_{\text{anchored}}\!\left(\rho_{s}^{(\ell)}, \rho_{s}^{(0)}\right) = w_{\text{evol}}\,\mathcal{E}_{\text{evol}}\!\left(\rho_{s}^{(\ell)}\right) + w_{\text{anc}}\,\mathcal{E}_{\text{anc}}\!\left(\rho_{s}^{(0)}\right)$$
   - **Collision Resolution / Injective Separation**: Even when two positions have the *exact same* evolved state $\mathcal{E}_{\text{evol}}(\rho_{s_1}^{(\ell)}) = \mathcal{E}_{\text{evol}}(\rho_{s_2}^{(\ell)}) = \sigma_*$ (e.g., token $A$ evolved into predicting $B$, matching a position containing token $B$), if $w_{\text{anc}} \neq 0$ and $\mathcal{E}_{\text{anc}}(\rho^{(0)}(A)) \neq \mathcal{E}_{\text{anc}}(\rho^{(0)}(B))$, then $\mathcal{E}_{\text{anchored}}(\rho_*,\, \rho^{(0)}(A)) \neq \mathcal{E}_{\text{anchored}}(\rho_*,\, \rho^{(0)}(B))$.
   - **Exact CPTP Closure**: Whenever $\mathcal{E}_{\text{evol}}(\rho_{s}^{(\ell)})$ and $\mathcal{E}_{\text{anc}}(\rho_{s}^{(0)})$ are unit-trace PSD Hermitian density states and $w_{\text{evol}}, w_{\text{anc}} \ge 0$ with $w_{\text{evol}} + w_{\text{anc}} = 1$, $\mathcal{E}_{\text{anchored}}(\rho_{s}^{(\ell)}, \rho_{s}^{(0)})$ is Hermitian, PSD, and unit-trace.
3. **Theorem 12c (`bipartite_register_trace_and_decoupled_frame_self_copy_free`)**:
   - **Bipartite Register Invariance**: Splitting $\rho = (1 - \lambda)\,\rho_{\text{sens}} + \lambda\,\rho_{\text{pred}}$ preserves exact unit trace $\operatorname{Tr}(\rho) = 1$ and leaves the sensory component $(1 - \lambda)\,\rho_{\text{sens}}$ invariant under any CPTP evolution on $\rho_{\text{pred}}$.
   - **Self-Copy Attractor Elimination**: When $U_{\text{out}} = U_{\text{in}}$ with $\langle u_{\text{in}}, u_{\text{in}}\rangle > 0$, the Step-0 self-copy alignment is strictly positive ($\langle u_{\text{out}}, u_{\text{in}}\rangle > 0$), whereas orthogonal input/output frames ($\langle u_{\text{out}}, u_{\text{in}}\rangle = 0$) reduce Step-0 self-copy alignment to zero.

---

## 3. Planned Module & Variants (`src/qdlblocks/benchmarks/anchored_bipartite_density_gpt_lab.py`)

| Variant | Layers ($L$) | Dual-Unitary Layer-0 Sensory Anchor ($Q/K/V + \rho_V$) | Bipartite Register Split ($2_{\text{sens}} \oplus 2_{\text{pred}}$) | Decoupled $Q_{\text{in}} \neq Q_{\text{out}}$ Frames | Factored Cross-Head Kraus (`Trick B` + Multi-Branch) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`W71_A0_UNANCHORED_W70_REF_4L`** | `4` | No | No | No (`Q_in == Q_out`) | Yes |
| **`W71_A1_SENSORY_ANCHORED_QKV_4L`** | `4` | **Yes (`W_q/k/v_anc` $\in U(16)$)** | No | No | Yes |
| **`W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L`** | `4` | **Yes (`W_q/k/v_anc` $\in U(16)$)** | **Yes ($2_{\text{sens}} \oplus 2_{\text{pred}}$)** | **Yes ($Q_{\text{in}} \perp Q_{\text{out}}$)** | Yes |
| **`W71_A3_DEEP_8L_ANCHORED_BIPARTITE`** | `8` | **Yes (`W_q/k/v_anc` $\in U(16)$)** | **Yes ($2_{\text{sens}} \oplus 2_{\text{pred}}$)** | **Yes ($Q_{\text{in}} \perp Q_{\text{out}}$)** | Yes |

---

## 4. Verification Criteria

1. **Machine-Checked Lean 4 Proof (`lean/QuantumDLBlocksLean/SensoryAnchoredBipartiteDensity.lean`)**:
   - Compiles via `lake build` and `scripts/verify_lean4_theorems.py` with `0` `sorry`, `0` `admit`, and `0` custom `axiom`.
2. **Numerical Inspector (`scripts/anchored_bipartite_density_inspector.py`)**:
   - Verify:
     1. **Theorem 12a & 12b (State-Collision Separation Audit)**: When position $s_1$ (token `A`) and position $s_2$ (token `B`) have identical evolved states $L_{s_1,h}^{(\ell)} \equiv L_{s_2,h}^{(\ell)} = L_*$, an unanchored block produces **zero** $Q/K/V$ separation (`unanchored_k_collision_dist == 0.0`), whereas `SensoryAnchoredBipartiteBlock` produces strictly positive $Q/K/V$ and Value-density separation (`anchored_k_collision_dist > 0.20`, `anchored_v_rho_collision_dist > 0.15`) while preserving exact unit trace (`trace_err < 1e-6`).
     2. **Theorem 12c (Bipartite Sensory Retention & Self-Copy Attractor Elimination)**:
        - Verify that `DecoupledDualSubspace512DBoundary` eliminates the Step-0 self-copy logit spike (`self_copy_logit_advantage` drops by `> 10x` compared to tied `Q_in == Q_out`) and preserves exact orthogonality $Q_{\text{in}}^\top Q_{\text{in}} = I_{512}$, $Q_{\text{out}}^\top Q_{\text{out}} = I_{512}$.
        - Verify that the bipartite sensory sub-register (`cols 0:2`) retains high cosine fidelity with the Layer-0 token state across layers while the predictive sub-register (`cols 2:4`) rotates freely.
     3. **Full `AnchoredBipartitePureDensityGPT` Forward/Backward/Quotient Optimizer Step Audit**: Verify finite loss/gradients, `su16_step_phase_frac < 1e-6`, `kraus_step_gauge_frac < 1e-5`, and `mom_beta = 0.85 -> 0.95`.
3. **Unit Tests & Linter**:
   - `pytest tests/test_plan091_anchored_bipartite_density_gpt.py -q` and `scripts/ast_signature_and_length_linter.py`.

---

## 5. Operational Tooling & Script Opportunities

### 5.1 Existing Scripts Used
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py --json-out /home/festivity/agy/QuantumDLBlocks/.cache/lean4_theorems_verification.json`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/benchmarks/anchored_bipartite_density_gpt_lab.py scripts/anchored_bipartite_density_inspector.py scripts/anchored_bipartite_density_bench.py tests/test_plan091_anchored_bipartite_density_gpt.py`

### 5.2 Build-Before-Launch Scripts
- [`scripts/anchored_bipartite_density_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/anchored_bipartite_density_inspector.py): Audits Theorems 12a, 12b, and 12c numerically on CPU/CUDA and writes `.cache/plan091_anchored_bipartite_inspector.json`.
- [`scripts/anchored_bipartite_density_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/anchored_bipartite_density_bench.py): Standardized CLI runner for `W71_A0..W71_A3`.

### 5.3 Nice-to-Have Scripts
- `scripts/token_collision_and_self_copy_profiler.py`: Measures per-layer $K/V$ cosine similarity with Layer-0 token identity vs. top-1 predicted next token across training checkpoints.

---

## 6. Constants Ledger (Rule 7) & Empirical Results

### 6.1 Top-of-File Named Constants (`src/qdlblocks/benchmarks/anchored_bipartite_density_gpt_lab.py`)

| Constant Name | Value | Mathematical / Physical Rationale |
| :--- | :--- | :--- |
| `DEFAULT_ANCHOR_QKV_MIX_INIT` | `0.30` | Initial weight $\gamma_{Q/K/V,h}^{(\ell)} = 0.30$ for Layer-0 sensory anchor unitary injection into $Q, K, V$ |
| `DEFAULT_ANCHOR_RHO_V_MIX_INIT` | `0.25` | Initial convex blend weight $\gamma_{\rho,h}^{(\ell)} = 0.25$ for Layer-0 sensory density $\tilde{\rho}_{s,h}^{(0)}$ in Value payload |
| `DEFAULT_BIPARTITE_SENSORY_TETHER` | `0.60` | Per-layer tether weight keeping sensory sub-register columns `0:2` anchored to $L_{t,h,:,0:2}^{(0)}$ |
| `DEFAULT_BIPARTITE_SENSORY_MODES` | `2` | Number of sensory-anchored columns ($R_{\text{sens}} = 2$ out of $R = 4$) in $L_{t,h}^{(\ell)} \in \mathbb{C}^{16 \times 4}$ |
| `DECOUPLED_Q_OUT_SEED_OFFSET` | `909` | RNG seed offset ensuring $Q_{\text{out}} \in \operatorname{O}(512)$ is initialized independently of $Q_{\text{in}} \in \operatorname{O}(512)$ |
| `DECOUPLED_READOUT_MIX_SEED_OFFSET` | `1212` | RNG seed offset initializing `q_readout_mix` $\in \operatorname{O}(512)$ independently of $I_{512}$ |

### B. Lean 4 Formalization & CUDA Inspector Verification (`0` Errors)
- **Lean 4 Module 11 ([`lean/QuantumDLBlocksLean/SensoryAnchoredBipartiteDensity.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/SensoryAnchoredBipartiteDensity.lean))**:
  - Verified via [`scripts/verify_lean4_theorems.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py): `14` `.lean` files, **`35/35` machine-checked theorems** (`unanchored_channel_state_collision`, `dual_unitary_sensory_anchored_distinguishability`, `dual_unitary_sensory_anchored_cptp`, `bipartite_register_trace_and_decoupled_frame_self_copy_free`), `0` `sorry`, `0` `admit`, `0` custom `axiom` (`build_time = 6.37s`).
- **CUDA Inspector ([`scripts/anchored_bipartite_density_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/anchored_bipartite_density_inspector.py) — `overall_passed: true`)**:
  - **Theorem 12a & 12b (`theorem12ab_collision_and_anchored_cptp`)**:
    - Unanchored state-collision distance: `unanchored_collision_dist = 0.0`, `unanchored_chol_dist = 0.0` (exact collision when evolved states coincide).
    - Dual-Unitary Sensory-Anchored separation: `anchored_rho_dist = 0.60625`, `anchored_chol_dist = 0.999493`, `frob_unit_err = 2.4e-07`, `trace_err = 2.4e-07`, `min_eigenvalue = -4.0e-08`.
  - **Theorem 12c (`theorem12c_bipartite_and_self_copy_elimination`)**:
    - Step-0 self-copy cosine alignment (`w71_a0_step0_self_copy_cosine` vs. `w71_a2_step0_self_copy_cosine`):
      - `W71_A0` (Wave 70 unanchored baseline, $Q_{\text{in}} \equiv Q_{\text{out}}$, reading `l_re[..., 0]`): **`0.995825` (`99.58%` self-copy alignment $x_{t+1} = x_t$ at Step 0!)**
      - `W71_A2` (Wave 71 decoupled $Q_{\text{in}} \neq Q_{\text{out}}$ + bipartite register reading `l_re[..., 2:4]`): **`0.010141` (`1.01%` — completely eliminating the Step-0 self-copy attractor!)**
    - Sensory sub-register fidelity: `sensory_cosine_fidelity = 0.703082`, `bipartite_frob_err = 3.6e-07`, `q_in_out_frobenius_diff_a2 = 31.949888`.
  - **Model & Quotient Optimizer Step (`wave71_model_step_and_quotient`)**:
    - `total_params = 31,459,882` (`31.46M` parameters at $L=4$), `su16_step_phase_frac = 0.0`, `kraus_step_gauge_frac = 1.0e-07`.

### C. `W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L` Training Trajectory (`49.15M` Tokens / `6,000` Steps)
- **Step 0 (`0.00M` tok)**: `Val CE = 8.9075`
- **Step 150 (`1.23M` tok)**: `Hot CE = 6.8337 | EMA CE = 6.7666 | Top-1 = 12.67%`
- **Step 600 (`4.92M` tok)**: `Hot CE = 6.3971 | EMA CE = 6.3394 | Top-1 = 14.99%`
- **Step 1,500 (`12.29M` tok)**: `Hot CE = 6.0335 | EMA CE = 5.9924 | Top-1 = 17.40%`
- **Step 3,150 (`25.80M` tok)**: `Hot CE = 5.7937 | EMA CE = 5.7468 | Top-1 = 18.82%`
- **Step 4,800 (`39.32M` tok)**: `Hot CE = 5.6929 | EMA CE = 5.6355 | Top-1 = 20.17%`
- **Step 6,000 (`49.15M` tok)**: `Hot CE = 5.6289 | EMA CE = 5.5864 | Top-1 = 20.01% | MUBErr = 5.00e-08 | SU16Phase = 0.00e+00 | KrausGauge = 0.00e+00`


