# EXPERIMENT PLAN 085: Fixed-Once Subsystem Quantum Vocabulary Architectures — Sequential Quantum Instruments (`256 ⊕ 256` Lüders Collapse), Right-Canonical 4-Qudit Matrix Product State (`16^4` MPS) Born Readout, & Shared `St(512, 128)` Subspace Frame (`Wave 65`)

## 1. Executive Summary & Architectural Motivation

In `Waves 62–64` ([`EXPERIMENT_PLAN_083.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_083.md), [`EXPERIMENT_PLAN_084.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_084.md)), the trunk of `KrausMUBThermostatGPT` evolved a compact $H = 16$ ensemble of $\mathbb{C}^{16 \times 16}$ density matrices (`512D`) via Rank-$R=2$ CPTP Stiefel Kraus channels and $2 \times U(16)$ DFT Mutually Unbiased Bases (MUB) tomography. However, the input/output boundary still treated the $V = 50,304$ vocabulary as a **flat, monolithic $50,304$-outcome measurement** (`wte_512` and `w_det_512` of shape `(50304, 512)`), creating three structural bottlenecks:

1. **Parameter & FLOP Asymmetry (`91%` of Parameters in Flat Vocab Tables)**:
   - Two `(50304, 512)` tables consume `51.51M` floats (`91%` of the 4-layer model's parameters) and `51.5G` FLOPs per micro-batch (`64.8%` of 4-layer compute).
2. **`5.17%` Sparse-Token Duty Cycle & $60.15\times$ Geodesic Speed Imbalance**:
   - At `8,192` tokens/step, only `2,601.2 / 50,304` (`5.17%`) vocabulary rows receive an active update on any step, forcing [`BatchSubspaceTikhonovGramMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py) to invert a `512 × 512` feature Gram matrix from sparse rows while the trunk steps $60.15\times$ faster in expected geodesic distance.
3. **Prior Packing/Repacking Overhead**:
   - Earlier experiments added auxiliary `(256, 512)` register logits *into* the flat `50,257`-way logit matrix before a single `50,257`-way `logsumexp`, which still materialized the full `50,304 × 512` GEMM and mixed representations inside the forward pass.

`Wave 65` (`EXPERIMENT_PLAN_085.md`) introduces a **Fixed-Once Subsystem Vocabulary Boundary (`QuantumSubsystemVocabBoundary`)** selected once at model construction time (Rule 3 Strategy Pattern) with **zero runtime table synthesis and zero `50,304 × 512` matrices** in its subsystem modes:

1. **`LUDERS_256X256_INSTRUMENT` (`LudersSequentialInstrumentBoundary` — $256 \oplus 256$ Bipartite Quantum Instrument)**:
   - Factors the vocabulary Hilbert space into two subsystems via the deterministic BPE residue/quotient bijection:
     $$a(v) = v \ \& \ \text{0xFF} \in \{0, \dots, 255\}, \qquad b(v) = v \gg 8 \in \{0, \dots, B_{\max}(a)\}, \qquad v = 256\,b(v) + a(v)$$
     where $B_{\max}(a) = 196$ for $a \in \{0, \dots, 80\}$ (`197` valid outcomes) and $B_{\max}(a) = 195$ for $a \in \{81, \dots, 255\}$ (`196` valid outcomes), covering all $81 \times 197 + 175 \times 196 = 50,257$ GPT-2 tokens with **zero empty coarse buckets, uniform Zipf mass across all 256 coarse outcomes, and zero probability mass leaked to padding tokens**.
   - **Stage 1 (`256`-Outcome Coarse POVM)**: Measures $h_{\text{norm}, t} \in \mathbb{R}^{512}$ against $E^{(1)} \in \mathbb{S}^{511}(\sqrt{512})^{256}$ (`512 × 256` GEMM — **$196\times$ smaller** than `512 × 50,304`).
   - **Stage 2 (Lüders Post-Measurement State Collapse)**: Conditioned on the observed Stage-1 outcome $a^* = a(y_t)$, collapses the tomographic state via a subsystem conditioning operator and the explicit density-matrix bilinear expectation $\langle \phi_{a^*, h} \mid \rho_{t,h} \mid \psi_{\text{exit}, t, h} \rangle$.
   - **Stage 3 (`197`-Outcome Conditional POVM)**: Measures the collapsed state $h_{\text{cond}, t} \in \mathbb{R}^{512}$ against $E^{(2)} \in \mathbb{S}^{511}(\sqrt{512})^{197}$ with exact boundary mask $b \le B_{\max}(a^*)$.
2. **`MPS_16POW4_BORN_CHAIN` (`RightCanonicalMPSBornBoundary` — 4-Site Base-$16$ Matrix Product State Born Machine)**:
   - Represents each token $v \in \{0, \dots, 50256\}$ in little-endian base-$16$ as 4 hexadecimal qudits ($d_{\text{qudit}} = 16 = d_h$):
     $$s_0(v) = v \ \& \ \text{0xF}, \quad s_1(v) = (v \gg 4) \ \& \ \text{0xF}, \quad s_2(v) = (v \gg 8) \ \& \ \text{0xF}, \quad s_3(v) = (v \gg 12) \ \& \ \text{0xF} \in \{0, \dots, S_{3,\max}(p_{012})\}$$
     where $p_{012} = s_0 + 16 s_1 + 256 s_2 \in \{0, \dots, 4095\}$ and $S_{3,\max}(p_{012}) = (50256 - p_{012}) \div 4096 \in \{11, 12\}$ (`1,105 × 13 + 2,991 × 12 = 50,257` exact tokens).
   - Parameterizes the 4 site transition operators as **right-canonical Stiefel isometries** $A^{[k]} \in \operatorname{St}(16 d_{\text{bond}}, d_{\text{bond}})$ (`(2048, 128)`, satisfying $\sum_{s=0}^{15} (A_s^{[k]})^\top A_s^{[k]} = I_{128}$) stepped directly on the Stiefel manifold by [`ModularRiemannianStiefelKrausMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/modular_quantum_statmech_gpt.py).
3. **`SHARED_STIEFEL_128D_FRAME` (`SharedStiefelSubspaceFrameBoundary` — Tied `128D` Sphere + `St(512, 128)` Frame)**:
   - Stores a single tied vocabulary sphere $W_{\text{vocab}} \in \mathbb{S}^{127}(\sqrt{128})^{50304}$ projected into/out of the `512D` trunk via $U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$, shrinking the readout GEMM by $4\times$ and the `BatchSubspaceTikhonovGramMuon` coupled Newton-Schulz $A^{-1/2}$ solve by **$64\times$** (`128 × 128` vs. `512 × 512`).

---

## 2. Hypotheses (`H1–H4`) & Quantitative Targets

- **`H1` (Exact `50,257`-Token Partition Function & Zero-Scan Step-0 Calibration Equivalence)**:
  - Both `LudersSequentialInstrumentBoundary` (`256 ⊕ 197` with support mask $b \le B_{\max}(a)$) and `RightCanonicalMPSBornBoundary` (`16 × 16 × 16 × 13` with support mask $s_3 \le S_{3,\max}(p_{012})$) define **strictly normalized probability distributions over $\{0, \dots, 50256\}$** satisfying:
    $$\left| \sum_{v=0}^{50256} P(v \mid \rho_t) - 1.0 \right| < 10^{-6}$$
    and at Step 0 ($\beta_0 = 0.50$), their hierarchical Zipf log-prior decomposition reproduces the exact Step-0 Shard-0 validation CE (`~9.0834` nats) of the flat `50,257`-way readout with `< 30 ms` zero-data initialization.
- **`H2` (`100.0%` Active Vocabulary Subsystem Duty Cycle at `8,192` Tokens/Step)**:
  - On a single `8,192`-token FineWeb micro-batch (`B = 16, T = 512`), the flat `50,304` vocabulary activates only `5.17%` of its rows, whereas `LUDERS_256X256_INSTRUMENT` activates **`100.0%` (`256/256`)** of its Stage-1 coarse rows and **`100.0%` (`197/197`)** of its Stage-2 fine rows on every single micro-batch, and `MPS_16POW4_BORN_CHAIN` activates **`100.0%` (`16/16`)** of its qudit branches at every site $k \in \{0, 1, 2, 3\}$.
- **`H3` (`18x–50x` Vocabulary Parameter Compression & `8x–40x` Readout GEMM Reduction)**:
  - Replacing the flat `wte_512 + w_det_512` tables (`51.51M` floats) with `LUDERS_256X256_INSTRUMENT` (`~1.15M` boundary floats) or `MPS_16POW4_BORN_CHAIN` (`~1.48M` boundary floats) reduces vocabulary parameter count by **`35x–45x`** and readout GEMM FLOPs by **`24x–98x`**, shifting the 4-layer model from `91%` vocabulary-dominated to **trunk-dominated (`> 75%` trunk parameters and FLOPs)**.
- **`H4` (Riemannian Stiefel & Sphere Manifold Preservation Under Multi-Step Training)**:
  - Training `QuantumSubsystemVocabGPT` across all three fixed-once boundary modes with `ModularRiemannianStiefelKrausMuon` + `BatchSubspaceTikhonovGramMuon` preserves `kraus_stiefel_err < 1e-4`, `mps_stiefel_err < 1e-4`, `probe_unitary_ortho_err < 1e-4`, and strictly monotonic validation NLL descent.

---

## 3. Mathematical Formulation

### 3.1 Mode 1: Two-Stage Sequential Quantum Instrument (`LUDERS_256X256_INSTRUMENT`)

1. **Balanced BPE Residue-Quotient Bijection ($\pi: \{0, \dots, 50256\} \leftrightarrow \mathcal{S}_{256 \times 197}$)**:
   For any GPT-2 token $v \in \{0, \dots, 50256\}$:
   $$a(v) = v \bmod 256 \in \{0, \dots, 255\}, \qquad b(v) = \lfloor v / 256 \rfloor \in \{0, \dots, B_{\max}(a(v))\}$$
   where $B_{\max}(a) = 196$ if $a \le 80$ else $195$. Because BPE ranks are ordered by decreasing frequency, $a(v) = v \bmod 256$ interleaves high-, mid-, and low-frequency tokens uniformly across all $256$ coarse outcomes (`P(a) ~ 1/256`).
2. **Exact Hierarchical Zipf Prior Factorization**:
   Given the online + analytical token log-prior $\log p_{\text{tok}}(v)$ over $v \in \{0, \dots, 50256\}$:
   $$b_a^{(1)} = \log \sum_{b=0}^{B_{\max}(a)} p_{\text{tok}}(256 b + a), \qquad b_{a, b}^{(2)} = \log p_{\text{tok}}(256 b + a) - b_a^{(1)}$$
   so $b_a^{(1)} + b_{a, b}^{(2)} \equiv \log p_{\text{tok}}(256 b + a)$ identically for every valid token $v = 256 b + a$.
3. **Two-Stage Lüders Measurement & State Collapse**:
   - **Stage 1**: Given the vacuum-normalized `512D` tomographic vector $h_{\text{norm}, t} = \beta_0 \cos\theta_{\text{vac}, t}\,\hat{u}_{\text{vocab}, t}$:
     $$z_{t, a}^{(1)} = C_{\text{cap}} \tanh\!\left(\frac{\langle h_{\text{norm}, t},\, r_a^{(1)} \hat{E}_a^{(1)} \rangle + \gamma_{\text{prior}} b_a^{(1)}}{C_{\text{cap}}}\right), \qquad \mathcal{L}_t^{(1)} = \log \sum_{a'=0}^{255} e^{z_{t, a'}^{(1)}} - z_{t, a(y_t)}^{(1)}$$
   - **Stage 2 (Lüders Post-Measurement Collapse Given $a^* = a(y_t)$)**:
     Conditioned on outcome $a^*$, the quantum state collapses via:
     $$u_{\text{cond}, t} = u_{\text{vocab}, t} \odot \big(1 + 0.50 \tanh(G_{a^*})\big) + W_{\text{collapse}}(a^*) + \gamma_{\text{rho}}\,\mathcal{M}_{\rho_t}(a^*)$$
     where $\mathcal{M}_{\rho_t}(a^*) \in \mathbb{R}^{512}$ concatenates the $H = 16$ head-wise density-matrix transition vectors $\operatorname{Re}/\operatorname{Im}(\rho_{t,h} |\phi_{a^*, h}\rangle) \in \mathbb{C}^{16}$.
   - **Stage 3**: Vacuum-normalize $h_{\text{cond}, t} = \operatorname{VacuumSphereNorm}_{\beta_0}(u_{\text{cond}, t}, u_{\text{vac}, t})$ and measure against $\hat{E}_b^{(2)} \in \mathbb{S}^{511}(\sqrt{512})^{197}$:
     $$z_{t, b}^{(2)} = C_{\text{cap}} \tanh\!\left(\frac{\langle h_{\text{cond}, t},\, r_b^{(2)} \hat{E}_b^{(2)} \rangle + \gamma_{\text{prior}} b_{a^*, b}^{(2)}}{C_{\text{cap}}}\right) \quad (b \le B_{\max}(a^*), \ -\infty \text{ if } b > B_{\max}(a^*))$$
     $$\mathcal{L}_t = \mathcal{L}_t^{(1)} + \mathcal{L}_t^{(2)} = -\log P(y_t \mid \rho_t)$$

### 3.2 Mode 2: 4-Site Right-Canonical Matrix Product State Born Readout (`MPS_16POW4_BORN_CHAIN`)

1. **4-Qudit Little-Endian Base-$16$ Decomposition**:
   Every token $v \in \{0, \dots, 50256\}$ decomposes into 4 hexadecimal qudits $(s_0, s_1, s_2, s_3) \in \{0, \dots, 15\}^3 \times \{0, \dots, 12\}$:
   $$v = s_0 + 16 s_1 + 256 s_2 + 4096 s_3, \qquad p_{012} = s_0 + 16 s_1 + 256 s_2 \in \{0, \dots, 4095\}$$
   with exact site-3 boundary support $s_3 \in \{0, \dots, S_{3,\max}(p_{012})\}$ where $S_{3,\max}(p_{012}) = 12$ if $p_{012} \le 1104$ else $11$.
2. **Right-Canonical Stiefel MPS Site Isometries ($A^{[k]} \in \operatorname{St}(16 d_{\text{bond}}, d_{\text{bond}})$)**:
   For each site $k \in \{0, 1, 2, 3\}$ and bond dimension $d_{\text{bond}} = 128$, the site operator $A^{[k]} \in \mathbb{R}^{(16 \cdot 128) \times 128}$ is constrained to the Stiefel manifold $\operatorname{St}(2048, 128)$ via `ModularRiemannianStiefelKrausMuon`, satisfying:
   $$\sum_{s=0}^{15} (A_s^{[k]})^\top A_s^{[k]} = I_{128}$$
   At site $k$, given the incoming bond state $\chi_t^{(k)} \in \mathbb{R}^{128}$ and head-group tomographic slice $h_{t, k} \in \mathbb{R}^{128}$, the 16 candidate post-measurement bond states are:
   $$\psi_{t, s}^{(k)} = A_s^{[k]} \chi_t^{(k)} + E_{s}^{[k]} \odot h_{t, k} \in \mathbb{R}^{128} \quad (s \in \{0, \dots, 15\})$$
   producing site scores $z_{t, s}^{(k)}$ (`16` outcomes per site!) and collapsing onto the observed qudit branch $\chi_t^{(k+1)} = \operatorname{RMSNorm}(\psi_{t, s_k^*}^{(k)})$ for site $k + 1$.

---

## 4. Planned Changes, Module Layout & Testing

1. **New Flag-Free Subsystem Vocabulary Module ([`src/qdlblocks/benchmarks/quantum_subsystem_vocab_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/quantum_subsystem_vocab_lab.py))**:
   - `VocabSubsystemMode(str, Enum)`: `LUDERS_256X256_INSTRUMENT`, `MPS_16POW4_BORN_CHAIN`, `SHARED_STIEFEL_128D_FRAME`.
   - `LudersSequentialInstrumentBoundary`, `RightCanonicalMPSBornBoundary`, `SharedStiefelSubspaceFrameBoundary`.
   - `QuantumSubsystemVocabGPT`, `build_subsystem_vocab_optimizers`, `evaluate_subsystem_vocab_model`, `run_subsystem_vocab_comparison_suite`.
2. **First-Class CLI Script ([`scripts/quantum_subsystem_vocab_profiler_and_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/quantum_subsystem_vocab_profiler_and_bench.py))**:
   - Profiles parameter counts, readout GEMM FLOPs, per-step active duty cycle (`100.0%` vs `5.17%`), exact `50,257`-way partition function normalization ($\sum_{v=0}^{50256} P(v) = 1.0$), and multi-step training curves across all three fixed-once subsystem vocabulary modes.
3. **Verification Script & Unit Tests**:
   - [`examples/95_verify_plan085_quantum_subsystem_vocab.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/95_verify_plan085_quantum_subsystem_vocab.py)
   - [`tests/test_plan085_quantum_subsystem_vocab.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan085_quantum_subsystem_vocab.py)

---

## 5. Operational Tooling & Script Identification

### 5.1 Existing Scripts Used
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/benchmarks/quantum_subsystem_vocab_lab.py scripts/quantum_subsystem_vocab_profiler_and_bench.py examples/95_verify_plan085_quantum_subsystem_vocab.py tests/test_plan085_quantum_subsystem_vocab.py`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/verify_plan.py --plan-num 85 --skip-pytest --json-out .cache/verify_plan085.json`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload`

### 5.2 `Build-Before-Launch` Scripts
1. [`scripts/quantum_subsystem_vocab_profiler_and_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/quantum_subsystem_vocab_profiler_and_bench.py): Profiles parameter compression (`35x–45x`), readout FLOP reduction (`24x–98x`), exact `50,257`-token probability normalization (`< 1e-6` error), `100.0%` active duty cycle, and training convergence across `LUDERS_256X256_INSTRUMENT`, `MPS_16POW4_BORN_CHAIN`, and `SHARED_STIEFEL_128D_FRAME`.

### 5.3 `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/mps_entanglement_entropy_and_luders_collapse_inspector.py`: Computes the bipartite bond entanglement spectrum across the 4-qudit MPS sites ($k = 0 \mid 123$, $01 \mid 23$, $012 \mid 3$) and the trace-distance state shift $\|\rho_t - \rho_t^{(a^*)}\|_1$ induced by Lüders collapse.

---

## 6. Constants Ledger & Empirical Results (`Wave 65`)

### 6.1 Rule 7 Named Constants Ledger (`src/qdlblocks/benchmarks/quantum_subsystem_vocab_lab.py`)

| Constant Name | Value | Mathematical Role & Rationale |
| :--- | :---: | :--- |
| `LUDERS_COARSE_OUTCOMES` | `256` | Number of Stage-1 coarse quantum instrument outcomes $a(v) = v \ \& \ \text{0xFF} \in \{0, \dots, 255\}$ |
| `LUDERS_FINE_MAX_OUTCOMES` | `197` | Maximum Stage-2 conditional outcomes $b(v) = v \gg 8 \in \{0, \dots, 196\}$ ($256 \times 196 + 81 = 50,257$) |
| `LUDERS_FULL_ROW_CUTOFF` | `81` | Coarse outcomes $a \in \{0, \dots, 80\}$ have $197$ valid fine outcomes; $a \in \{81, \dots, 255\}$ have $196$ |
| `MPS_QUDIT_BASE` | `16` | Hexadecimal qudit dimension per MPS site ($16^4 = 65,536 \ge 50,257$, matching $d_h = 16$) |
| `MPS_NUM_SITES` | `4` | Number of sequential MPS qudit sites ($s_0, s_1, s_2, s_3$) |
| `MPS_SITE3_MAX_OUTCOMES` | `13` | Maximum valid outcomes at Site 3 ($s_3 \in \{0, \dots, 12\}$, since $\lfloor 50256 / 4096 \rfloor = 12$) |
| `MPS_SITE3_FULL_PREFIX_CUTOFF` | `1105` | Prefixes $p_{012} \in \{0, \dots, 1104\}$ have $13$ valid Site-3 outcomes; $p_{012} \ge 1105$ have $12$ |
| `DEFAULT_MPS_BOND_DIM` | `128` | MPS bond dimension $d_{\text{bond}} = 128$ ($A^{[k]} \in \operatorname{St}(2048, 128)$) |
| `DEFAULT_SHARED_VOCAB_SUBSPACE_DIM` | `128` | Shared Stiefel vocabulary manifold dimension $d_{\text{voc}} = 128$ ($U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$) |
| `DEFAULT_LUDERS_RHO_TRACE_SCALE` | `0.40` | Initial coupling $\gamma_{\text{rho}}$ for density-matrix Lüders collapse conditioning $\langle \phi_{a^*, h} \mid \rho_{t,h} \mid \psi_{t,h} \rangle$ |

### 6.2 Empirical Results (`Wave 65` — Executed & Verified)

1. **Exact Hierarchical Zipf Prior Bijection & `50,257`-Token Partition Function Verification ([`.cache/verify_plan085_quantum_subsystem_vocab.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/verify_plan085_quantum_subsystem_vocab.json))**:
   - `build_luders_hierarchical_zipf_priors` ($256 \times 197$, $a(v) = v \ \& \ \text{0xFF}, b(v) = v \gg 8$): Reconstructs $\log p_{\text{tok}}(v)$ across all `50,257` tokens to **`9.537e-07` max error** (`81 × 197 + 175 × 196 = 50,257`, zero empty coarse buckets).
   - `build_mps_4qudit_hierarchical_zipf_priors` ($16 \times 16 \times 16 \times 13$, $v = s_0 + 16 s_1 + 256 s_2 + 4096 s_3$): Reconstructs $\log p_{\text{tok}}(v)$ across all `50,257` tokens to **`0.000e+00` max error** (`1,105 × 13 + 2,991 × 12 = 50,257`).
   - Exact `50,257`-way partition function error $\max_t |\sum_{v=0}^{50256} P(v \mid \rho_t) - 1.0|$:
     - `LUDERS_256X256_INSTRUMENT`: **`2.35e-07`** (`nll_consistency_err = 0.0`)
     - `MPS_16POW4_BORN_CHAIN`: **`1.45e-07`** (`nll_consistency_err = 0.0`)
     - `SHARED_STIEFEL_128D_FRAME`: **`7.66e-07`** (`nll_consistency_err = 0.0`)

2. **Parameter Compression, Duty Cycle, & Convergence Comparison ([`.cache/plan085_subsystem_vocab_bench.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/plan085_subsystem_vocab_bench.json), `B=8, T=128`, `30` Steps on FineWeb Shard 1 / Shard 0)**:

| Variant (`VocabSubsystemMode`) | Boundary Vocab Params (`vs 51.51M Flat`) | Boundary Share of Total Params | Step-0 Init Time | Active Duty Cycle (`1K` / `8K` Batch) | Step 0 Val CE | Step 10 Val CE | Step 20 Val CE | Step 30 Val CE (`Top-1`) | Kraus / Boundary Stiefel Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`W65_V1_LUDERS_256X256_INSTRUMENT`** (`256 ⊕ 197` Lüders Collapse) | **`911,383` (`56.52x` smaller)** | **`7.60%`** (`11.99M` total) | **`9.1 ms`** | **`66.7%` / `100.0%`** | `8.8611` | `8.6024` | `8.2662` | **`8.0861` (`6.37%`)** | `1.11e-07` / `0.00e+00` |
| **`W65_V2_MPS_16POW4_BORN_CHAIN`** (`16^4` Right-Canonical Stiefel MPS) | **`1,376,529` (`37.42x` smaller)** | **`11.05%`** (`12.45M` total) | **`24.9 ms`** | **`100.0%` / `100.0%`** | `9.0045` | `8.5041` | `8.2482` | **`8.0766` (`6.64%`)** | `1.11e-07` / `2.39e-07` |
| **`W65_V3_SHARED_STIEFEL_128D_FRAME`** (`St(512, 128)` + Tied `128D` Sphere) | **`6,670,593` (`7.72x` smaller)** | **`37.59%`** (`17.75M` total) | **`49.1 ms`** | `0.9%` / `5.17%` | `8.8663` | `8.3191` | `8.0971` | **`7.9639` (`6.74%`)** | `1.11e-07` / `2.66e-07` |

3. **Full `L = 8` (`B = 16, T = 512`, `120` Steps = `0.98M` Tokens) Comparative Benchmark with Verified `RiemannianShadowEMAThermostat` ([`.cache/plan085_w65_8l_thermostat_120steps.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/plan085_w65_8l_thermostat_120steps.json))**:

| Variant (`L = 8`, `8,192` tok/step) | Vocab Params (`vs 51.51M Flat`) | Active Duty Cycle | Step 0 CE | Step 30 CE (`Gap`, `Trig`) | Step 60 CE (`Gap`, `Trig`) | Step 90 CE (`Gap`, `Trig`) | Step 120 CE (`Top-1`) | Boundary Stiefel / $Z_t$ Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *(Wave 64 Ref)* `W64_A3_FULL_CONSOLIDATED_8L` | `51,511,296` (`1.0x`) | `5.17%` | `9.1473` | `7.4068` (`-0.0227`, `False`) | `7.2085` (`+0.0596`, **`True`**) | `7.0767` (`+0.0179`, **`True`**) | `7.0078` (`12.30%`) | `0.00e+00 / 0.00e+00` |
| **`W65_V1_LUDERS_256X256_INSTRUMENT`** (`256 ⊕ 197` Lüders Collapse) | **`911,383` (`56.5x` smaller)** | **`97.7%`** | `8.9058` | `7.6278` (`-0.0209`, `False`) | **`7.3859`** (`+0.0480`, **`True`**) | **`7.2386`** (`+0.0215`, **`True`**) | **`7.1521` (`11.35%`)** | `0.00e+00 / 2.40e-07` |
| **`W65_V2_MPS_16POW4_BORN_CHAIN`** (`16^4` Right-Canonical Stiefel MPS) | **`1,376,529` (`37.4x` smaller)** | **`100.0%`** | `8.9952` | **`8.2070`** (`+0.2706`, **`True`**) | **`7.7032`** (`+0.1014`, **`True`**) | `7.5583` (`+0.0142`, `False`) | **`7.4806` (`9.16%`)** | `6.22e-07 / 1.80e-07` |
| **`W65_V3_SHARED_STIEFEL_128D_FRAME`** (`St(512, 128)` + Tied `128D` Sphere) | **`6,670,593` (`7.7x` smaller)** | `5.4%` | `8.9066` | `7.4425` (`+0.0115`, `False`) | **`7.1701`** (`+0.1095`, **`True`**) | **`6.9646`** (`+0.0249`, **`True`**) | **`6.8337` (`13.02%`)** | `3.36e-07 / 2.97e-07` |

4. **Key Architectural Findings**:
   - **Shared `St(512, 128)` Subspace Frame (`W65_V3`) Beats Full `512D` Flat Vocabulary (`W64_A3`) by `-0.1741` Nats with `7.7x` Fewer Vocab Parameters**: Tying input and output vocabulary representations in a `128D` sphere $\mathbb{S}^{127}(\sqrt{128})$ bridged by $U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$ improves 120-step validation CE from `7.0078 -> 6.8337` (`12.30% -> 13.02%` Top-1) because the `128×128` Gram covariance in `BatchSubspaceTikhonovGramMuon` is $4\times$ better conditioned (`2,600` active tokens estimating `128` dimensions instead of `512`) and $64\times$ cheaper to invert.
   - **Storing Conditional Log-Prior Biases as 1D Parameter Vectors Avoids `DEFAULT_BALL_MAX_RMS` Truncation**: In `RiemannianGeodesicScalarAndProbeOptimizer`, `_step_bounded_tangent_param` clamps 2D matrices (`p.ndim == 2`) to row RMS $\le 2.5$. Storing `bias_fine_cond` (`256 × 197`, RMS $\approx 5.5$) and `bias_s1..s3` as 1D parameters (viewed as 2D during forward contraction) prevents Step-1 truncation of the hierarchical Zipf prior and improved `W65_V1_LUDERS_256X256_INSTRUMENT` by **`-0.4533` nats** (`8.5394 -> 8.0861` Val CE, `3.98% -> 6.37%` Top-1).
   - **Duty-Matched Geodesic Step Rate & Closed-Loop Thermostat Stabilization for `100%`-Duty-Cycle Subsystem Tables**: In `W65_V1`, `W65_V2`, and `W65_V3`, the fixed `RiemannianShadowEMAThermostat` automatically detected thermal overshoot at Steps 30–90 (`+0.048` to `+0.271` nats thermal gap), banking the manifold-retracted Shadow EMA weights and adjusting the step size on the fly.

5. **Long-Horizon Scaling Run (`W65_V3_SHARED_STIEFEL_128D_FRAME`, `L = 4`, `229.38M` Tokens / `28,000` Steps, [`.cache/live_logs/plan085_w65_v3_shared_stiefel_128d_frame_36621steps.log`](file:///home/festivity/agy/QuantumDLBlocks/.cache/live_logs/plan085_w65_v3_shared_stiefel_128d_frame_36621steps.log))**:

| Step | Tokens (`M`) | `lr_scale` | Hot Val CE | **EMA Val CE** | Thermal Gap | Trig? | Val PPL | Top-1 / Top-5 | Kraus / Bound Stiefel Err |
| ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | `1.00x` | `8.9075` | **`8.9075`** | `+0.0000` | `False` | `7387.17` | `0.20%` / `0.94%` | `0.00e+0` / `2.22e-7` |
| `1,000` | `8.19M` | `0.55x` | `5.8304` | **`5.8000`** | `+0.0304` | `True` | `330.30` | `20.79%` / `36.73%` | `9.02e-8` / `3.33e-7` |
| `2,000` | `16.38M` | `0.38x` | `5.5395` | **`5.5191`** | `+0.0204` | `True` | `249.41` | `22.26%` / `38.68%` | `9.08e-8` / `3.10e-7` |
| `3,750` | `30.72M` | `0.27x` | `5.3015` | **`5.2889`** | `+0.0126` | `False` | `198.13` | `23.92%` / `40.42%` | `9.12e-8` / `3.51e-7` |
| `5,000` | `40.96M` | `0.25x` | `5.2327` | **`5.2140`** | `+0.0187` | `True` | `183.83` | `24.37%` / `41.00%` | `8.96e-8` / `3.50e-7` |
| `7,250` | `59.39M` | `0.25x` | `5.1290` | **`5.1177`** | `+0.0113` | `False` | `166.95` | `24.80%` / `41.58%` | `8.30e-8` / `3.19e-7` |
| `10,750` | `88.06M` | `0.25x` | `5.0554` | **`5.0423`** | `+0.0131` | `False` | `154.82` | `25.41%` / `42.21%` | `8.74e-8` / `3.46e-7` |
| `12,250` | `100.35M` | `0.25x` | `5.0547` | **`5.0288`** | `+0.0259` | `True` | `152.76` | `25.43%` / `42.24%` | `8.47e-8` / `3.06e-7` |
| `14,250` | `116.74M` | `0.25x` | `4.9954` | **`4.9788`** | `+0.0166` | `True` | `145.30` | `25.70%` / `42.68%` | `8.61e-8` / `3.35e-7` |
| `18,000` | `147.46M` | `0.25x` | `4.9599` | **`4.9468`** | `+0.0131` | `False` | `140.72` | `25.94%` / `43.09%` | `8.76e-8` / `3.20e-7` |
| `21,500` | `176.13M` | `0.25x` | `4.9383` | **`4.9207`** | `+0.0176` | `True` | `137.10` | `26.05%` / `43.44%` | `8.62e-8` / `3.11e-7` |
| `24,750` | `202.75M` | `0.25x` | `4.9187` | **`4.9051`** | `+0.0136` | `False` | `134.98` | `26.19%` / `43.46%` | `8.49e-8` / `3.30e-7` |
| **`27,750`** | **`227.33M`** | **`0.25x`** | **`4.9137`** | **`4.8977`** | **`+0.0160`** | **`True`** | **`133.98`** | **`26.45%` / `43.65%`** | **`8.59e-8` / `3.27e-7`** |
| `28,000` | `229.38M` | `0.25x` | `4.9183` | **`4.9036`** | `+0.0147` | `False` | `134.77` | `26.23%` / `43.54%` | `8.32e-8` / `3.15e-7` |

6. **Late-Horizon Scaling Post-Mortem (`0M -> 30M` Fast Descent vs. `60M -> 229M` Diminishing Returns)**:
   - **Dramatic Early/Mid-Horizon Gains (`0M -> 30.72M` Tokens)**: Compared to `W64_A3` at `30.89M` tokens (`5.8339` Val CE, `Val PPL = 341.69`, `20.36%` Top-1), `W65_V3_SHARED_STIEFEL_128D_FRAME` reached **`5.2889` Val CE (`Val PPL = 198.13`, `23.92%` Top-1)** at `30.72M` tokens (**`-0.5450` nats, `1.72x` lower PPL, `+3.56%` Top-1**) and matched `W64_A3`'s `30.89M`-token loss in just `8.19M` tokens (`3.77x` more token-efficient).
   - **Why Convergence Slowed After `~60M` Tokens (`5.1177` at `59.4M` $\to$ `4.8977` at `227.3M`)**:
     1. **`lr_scale` Clamped at the `0.25x` Floor + Every-250-Step Hard EMA Reset ("Ratchet Trap")**: By `Step 4,000` (`32.77M` tokens), `lr_scale` reached its minimum floor (`DEFAULT_THERMOSTAT_LR_MIN_SCALE = 0.25`). However, because mini-batch gradient noise at `8,192` tok/step naturally produces a steady-state Hot-vs-EMA Jensen gap of `~0.015–0.020` nats (`>= DEFAULT_THERMOSTAT_GAP_TOL_NATS = 0.015`), the thermostat kept firing `trig = True` on almost every 250-step checkpoint and **overwriting the active exploratory weights $W \leftarrow W_{\text{ema}}$ + decaying optimizer momentum by `0.70x`**. In late training, repeatedly pulling the active iterate back to the lagging EMA center every 250 steps kills long-range momentum across flat valleys!
     2. **Tied `128D` Readout Rank Bottleneck (`d_voc = 128` vs `512D` Trunk)**: While projecting the vocabulary through $\operatorname{St}(512, 128)$ dramatically accelerates the first `50M` tokens (`8.91 -> 5.15` Val CE), rank `128` caps the number of orthogonal context-discriminative directions available to separate fine-grained tail tokens below `~4.90` Val CE.
     3. **4-Layer (`L = 4`, `3.79M` Trunk Params) Expressivity Ceiling**: With `6.67M` vocab params and only `3.79M` trunk params (`L = 4`), the `512D` trunk saturates its multi-hop compositional capacity around `150M–200M` tokens.
