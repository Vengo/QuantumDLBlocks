# Experiment Plan 094 (`Wave 74`): Fused `bfloat16` Tensor-Core 1-Layer & 2-Layer 4-Compartment Quantum Statistical Mechanics Baton LM on `300M` FineWeb Tokens

## 1. Executive Summary & Experimental Intent

In `Wave 73` ([`EXPERIMENT_PLAN_093.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_093.md)), we validated the **4-Compartment Quantum Statistical Mechanics Baton Architecture** (`Packet 0` Online Unigram Thermal Bath + `Packets 1–2` Vectorized $K=4, \chi=8$ MPS & Online Lindblad/Bigram Transition Baton + `Packets 3–6` Exterior-Masked Motif-Preconditioned Born-Attention + `Packet 7` 64D Walsh–Hadamard Token-ID Register), demonstrating that:
- Entrywise Schur subspace intersection ($\rho_t^{\text{seq}} \odot \rho_t^{\text{ngram}}$) beats additive mixing by `-0.0481` nats (`+28.6%` relative Top-1 accuracy) in 1 layer at identical parameter count (`W73_B1` vs. `W73_B0`),
- Adding the **2-Hop Triangle Closure** ($A_{\text{2hop}} = P_{\text{attn}} P_{\text{local}}$) inside a single layer (`W73_B3_FOUR_COMPARTMENT_2HOP_TRIANGLE_1L`) captures `43%` of the 2-layer gain at 1-layer speed, and
- Adding a second highway layer (`W73_B2_FOUR_COMPARTMENT_2L`) enables 4-way Schur conjunction across routed 2-way states, achieving the highest Top-1 accuracy (`+68.0%` relative) and output purity (`0.4738`).

However, the reference `Wave 73` implementation materialized explicit `(B, T, 8, 64, 64)` `float32` density matrices at every token position ($32,768$ complex entries/token) and executed $O(B H T^2 d^2)$ attention contractions (`einsum("bhts,bshde->bthde")`), limiting throughput to `9.5k tok/s` (`1L`) and `5.9k tok/s` (`2L`).

**Goal of `Wave 74` (`EXPERIMENT_PLAN_094.md`)**:
1. **Prove and Implement the Exact Factorized Schur-Beam & Diagonal-Conjugated Thermal Bath Identities (`Lean 4 Module 14`, `Theorems 15a–15c`)**:
   - Eliminate all `(B, T, H, 64, 64)` intermediate tensor allocations by keeping token-varying states in low-rank purified factors $Z_t \in \mathbb{C}^{R \times 64}$ ($R = 4$) while keeping only the global thermal bath $\rho_{\text{uni}} \in \mathbb{C}^{64 \times 64}$ and streaming bigram operator $T_{\text{bi}}^{\text{ema}} \in \mathbb{C}^{64 \times 64}$ as shared $64 \times 64$ buffers.
2. **Execute a Full `bfloat16` + Kernel-Fusion Optimization Pass ([`fused_bf16_four_compartment_baton_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/fused_bf16_four_compartment_baton_lab.py))**:
   - Fuse complex linear projections into single real-block `bfloat16` Tensor Core GEMMs (`(B*T, 128) @ (128, ...)`),
   - Replace `.unfold()` in the $K=4, \chi=8$ MPS transfer chain with zero-copy causal slice shifts and packed `(B*T, 256) @ (256, 16)` `bfloat16` Tensor Core GEMMs,
   - Compute key-state MPS purity from the $\chi \times \chi$ ($8 \times 8$) Gram matrix $G_t = B_t B_t^\dagger$ (`64` entries instead of $4,096$, **Theorem 15c**),
   - Execute a single fused `(B*T, 128) @ (128, 50257)` `bfloat16` Tensor Core vocabulary readout GEMM, and compile the highway step via `@torch.compile`.
3. **Head-to-Head `1L` vs. `2L` 2-Hop Triangle Sprint & `300,000,000`-Token (`300M` Token) Multi-Shard FineWeb Run**:
   - Benchmark **`W74_F1_BF16_FUSED_2HOP_1L`** (1-Layer 4-Compartment Baton + 2-Hop Triangle) against **`W74_F2_BF16_FUSED_2HOP_2L`** (2-Layer 4-Compartment Baton + 2-Hop Triangle + 4-Way Schur Conjunction), and
   - Train a full **`300,000,000`-token (`300M` token)** run across `FineWeb` Shards 1, 2, and 3 (`fineweb_train_000001.bin`..`000003.bin`, validating on `fineweb_val_000000.bin`) on the local 16 GB NVIDIA RTX 5060 Ti GPU with 100% online thermal bath & n-gram transition ingestion from cold Step 0.

---

## 2. Mathematical Foundations & Lean 4 Formal Verification (`Module 14: FusedFactorizedSchurBaton.lean`)

### 2.1 Theorem 15a — Factorized Ket/Beam Schur Intersection Identity (`theorem15a_factorized_ket_schur_action_identity`)
For any two rank-1 density operators $\rho_A = |a\rangle\langle a|$ and $\rho_B = |b\rangle\langle b|$ on $\mathbb{C}^d$, their entrywise Schur product satisfies:
$$\big(|a\rangle\langle a| \odot |b\rangle\langle b|\big)_{ij} = (a_i \bar{a}_j)(b_i \bar{b}_j) = (a_i b_i)\,\overline{(a_j b_j)} = \big(|a \odot b\rangle\langle a \odot b|\big)_{ij},$$
so for any probe vector $|\psi\rangle \in \mathbb{C}^d$:
$$\boxed{\big(|a\rangle\langle a| \odot |b\rangle\langle b|\big)|\psi\rangle \;\equiv\; |a \odot b\rangle\langle a \odot b \mid \psi\rangle, \qquad \operatorname{Tr}\!\big(|a\rangle\langle a| \odot |b\rangle\langle b|\big) \equiv \|a \odot b\|_2^2.}$$
More generally, for rank-$R$ Cholesky factors $Z_A, Z_B \in \mathbb{C}^{R \times d}$, elementwise multiplication of paired beams $Z_{A \odot B} = Z_A \odot Z_B \in \mathbb{C}^{R \times d}$ forms a positive-semidefinite Schur intersection $\rho_{A \odot B} = \frac{Z_{A \odot B}^\dagger Z_{A \odot B}}{\|Z_{A \odot B}\|_F^2 + \varepsilon}$ in $O(R d)$ without ever materializing a $d \times d$ matrix!

### 2.2 Theorem 15b — Diagonal-Conjugated Thermal Bath Schur Action Identity (`theorem15b_diagonal_conjugated_bath_schur_action_identity`)
For any token ket $|a_t\rangle \in \mathbb{C}^d$ and the global $d \times d$ online unigram thermal bath $\rho_{\text{uni}} \in \mathcal{D}(\mathbb{C}^d)$, the entrywise Schur product $(|a_t\rangle\langle a_t|) \odot \rho_{\text{uni}} = \operatorname{diag}(a_t)\,\rho_{\text{uni}}\,\operatorname{diag}(\bar{a}_t)$ acting on any probe vector $|\psi_t\rangle \in \mathbb{C}^d$ satisfies:
$$\boxed{\Big(\big(|a_t\rangle\langle a_t|\big) \odot \rho_{\text{uni}}\Big)|\psi_t\rangle \;\equiv\; a_t \odot \Big(\rho_{\text{uni}}\big(\bar{a}_t \odot |\psi_t\rangle\big)\Big), \qquad \operatorname{Tr}\!\Big(\big(|a_t\rangle\langle a_t|\big) \odot \rho_{\text{uni}}\Big) \equiv \big\langle |a_t|^2,\; \operatorname{diag}(\rho_{\text{uni}}) \big\rangle.}$$
Because $\rho_{\text{uni}} \in \mathbb{C}^{64 \times 64}$ is shared across the batch `(B, T)`, computing $\rho_{\text{uni}}(\bar{a}_t \odot |\psi_t\rangle)$ for all $B \times T$ tokens simultaneously is a single `bfloat16` Tensor Core GEMM `(B*T, 64) @ (64, 64)`!

### 2.3 Theorem 15c — Gram-Matrix Purity Cyclicity Identity (`theorem15c_gram_purity_cyclicity_identity`)
For the rank-$\chi$ ($\chi = 8$) MPS beam factor $B_t \in \mathbb{C}^{\chi \times d}$ ($d = 64$), the purity of $\rho_t^{\text{MPS}} = \frac{B_t^\dagger B_t}{\|B_t\|_F^2}$ satisfies:
$$\boxed{\operatorname{Tr}\!\big((\rho_t^{\text{MPS}})^2\big) = \frac{\operatorname{Tr}\!\big((B_t^\dagger B_t)^2\big)}{\|B_t\|_F^4} \;\equiv\; \frac{\operatorname{Tr}\!\big((B_t B_t^\dagger)^2\big)}{\|B_t\|_F^4} = \frac{\|G_t\|_F^2}{\|B_t\|_F^4}, \qquad G_t = B_t B_t^\dagger \in \mathbb{C}^{8 \times 8},}$$
reducing key-state MPS purity evaluation from $64 \times 64 = 4,096$ entries/token to $8 \times 8 = 64$ entries/token (**`64×` reduction**).

---

## 3. Fused `bfloat16` Tensor-Core Architecture (`FusedBF16FourCompartmentBatonLM`)

1. **Packed Real-Block Complex Representation (`128D` Real Representation of $\mathbb{C}^{64}$)**:
   - Represents a complex ket $|\psi\rangle = \psi_{\text{re}} + i \psi_{\text{im}} \in \mathbb{C}^{64}$ as a contiguous `bfloat16` vector $[\psi_{\text{re}}, \psi_{\text{im}}] \in \mathbb{R}^{128}$ so that inner products $\operatorname{Re}\langle \phi \mid \psi \rangle = \langle \phi_{\text{re}}, \psi_{\text{re}} \rangle + \langle \phi_{\text{im}}, \psi_{\text{im}} \rangle$ and vocabulary projections `(B*T, 128) @ (128, 50257)` execute in a **single** `bfloat16` Tensor Core GEMM.
2. **Shift-Fused $K=4, \chi=8$ MPS Baton (`FusedBF16MPSBatonModule`)**:
   - Eliminates `.unfold()` allocation via zero-copy causal slice shifts and contracts the bond-input outer product `(B*T, 256)` against packed complex transfer cores `(256, 16)` in `bfloat16` Tensor Cores.
3. **2-Hop Triangle Exterior Born-Attention (`use_two_hop_triangle = True` in Both `1L` and `2L`)**:
   - Applies exterior causal masking ($s \le t - K$ on Heads `0, 1, 2` and full causal $s \le t$ on Head `3`), key-state Gram MPS purity weighting, and the 2-hop triangle attention closure $A_{\text{eff}} = (1 - \alpha_{\text{2hop}}) P_{\text{attn}} + \alpha_{\text{2hop}} (P_{\text{attn}} P_{\text{local}})$ inside **every** layer.
   - In the **2-Layer (`W74_F2_BF16_FUSED_2HOP_2L`)** variant, Layer 1 builds 2-way Schur-intersected states $Z_t^{\text{deep},(1)} = \operatorname{Norm}_F(Z_t^{\text{seq},(1)} \odot Z_t^{\text{ngram},(1)})$ with 2-hop triangle attention, and Layer 2 routes and intersects them into a **4-way Schur conjunction** $Z_t^{\text{4way}} = \operatorname{Norm}_F(Z_t^{\text{deep},(1)} \odot Z_t^{\text{deep},(2)})$.

---

## 4. Verification & Experimental Matrix

| Run Key | Layers | 2-Hop Triangle | Schur Intersection | Token Budget | Shards Used | Purpose |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`w74_f1_bf16_fused_2hop_1l_sprint`** | `1` | `True` | `2-Way` ($\rho^{\text{seq}} \odot \rho^{\text{ngram}}$) | `6.55M` (`400` steps) | Shard 1 | Fast head-to-head comparison of `1L` 2-Hop Triangle in fused `bfloat16` |
| **`w74_f2_bf16_fused_2hop_2l_sprint`** | `2` | `True` | `4-Way` ($\rho^{\text{deep},(1)} \odot \rho^{\text{deep},(2)}$) | `6.55M` (`400` steps) | Shard 1 | Fast head-to-head comparison of `2L` 2-Hop Triangle + 4-Way Schur in fused `bfloat16` |
| **`w74_bf16_fused_2hop_300m_spin`** | `2` (`+1L` companion) | `True` | `4-Way` / `2-Way` | **`300.0M`** (`300,000,000` tokens) | Shards 1, 2, 3 (`fineweb_train_000001..000003.bin`) | Full **`300M`-token** multi-shard FineWeb training run with 100% online bath & n-gram ingestion |

---

## 5. Operational Tooling & Script Opportunities (Mandatory Section 5 Gate)

### 5.1 Existing Scripts Used
- [`scripts/verify_lean4_theorems.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py): Verifies all 17 Lean 4 files (`44/44` theorems, `0` `sorry`).
- [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py): Verifies Rule 1 (`<= 1,000` SLOC soft limit) and Rule 2 (`<= 8` function arguments).
- [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py): Syncs `Wave 74` telemetry to `.cache/benchmarks.db` and reloads the WebUI.

### 5.2 `Build-Before-Launch` Scripts Created in `Wave 74`
1. [`scripts/fused_bf16_baton_speed_and_parity_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fused_bf16_baton_speed_and_parity_inspector.py): Verifies exact numerical equivalence of Theorems 15a–15c (`< 1e-6` error) and measures the throughput (`tok/s`) and VRAM (`MiB`) speedup of `FusedBF16FourCompartmentBatonLM` vs. `Wave 73`.
2. [`scripts/run_wave74_300m_four_compartment_spin.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/run_wave74_300m_four_compartment_spin.py): Executes the `1L` vs. `2L` 2-Hop Triangle comparison and the **`300,000,000`-token (`300M` token)** multi-shard FineWeb training run with live `StandardizedExperimentLogger` checkpoints.

### 5.3 `Nice-to-Have` Candidate Scripts (Logged in [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md))
1. `scripts/multi_shard_online_bath_drift_and_entropy_tracker.py`: Tracks the von Neumann / Rényi-2 entropy $-\ln \operatorname{Tr}(\rho_{\text{uni}}^2)$ of `Packet 0` (`online_uni_bath`) and singular-value spectrum of `Packet 2` (`online_trans_ema`) across `100M -> 300M -> 900M` shard transitions.

---

## 6. Top-of-File Constants Ledger (Rule 7) & Empirical Results

### 6.1 Top-of-File Constants Ledger ([`src/qdlblocks/benchmarks/fused_bf16_four_compartment_baton_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/fused_bf16_four_compartment_baton_lab.py))

| Constant Name | Value | Mathematical Role & Rationale |
| :--- | :---: | :--- |
| `DEFAULT_VOCAB_SIZE` | `50257` | Official GPT-2 BPE vocabulary size (`FineWeb` `kjj0/fineweb10B-gpt2`). |
| `DEFAULT_HILBERT_DIM` | `64` | Per-compartment complex Hilbert dimension $d = 64$ (`128` packed real+imag `bfloat16` features). |
| `DEFAULT_SCHUR_BEAM_RANK` | `4` | Factorized Cholesky beam rank $R = 4$ per compartment (`(B, T, 4, 64)` complex beams). |
| `DEFAULT_MPS_WINDOW_K` | `4` | Causal $K$-gram window size $K = 4$ for the shift-fused 1D MPS baton and exterior attention mask. |
| `DEFAULT_MPS_BOND_DIM` | `8` | Virtual bond dimension $\chi = 8$ for the MPS transfer chain and $8 \times 8$ Gram purity matrix. |
| `DEFAULT_TWO_HOP_MIX_ALPHA` | `0.30` | Weight $\alpha_{\text{2hop}} = 0.30$ on the 2-hop triangle attention closure $P_{\text{attn}} P_{\text{local}}$. |
| `DEFAULT_ONLINE_BATH_EMA_DECAY` | `0.02` | Online ingestion EMA step rate $\eta_{\text{uni}} = 0.02$ for multi-shard `300M`-token streaming. |
| `DEFAULT_ONLINE_TRANS_EMA_DECAY` | `0.02` | Online ingestion EMA step rate $\eta_{\text{trans}} = 0.02$ for streaming bigram transition matrix $T_{\text{bi}}^{\text{ema}}$. |
| `DEFAULT_ONLINE_HASH_EMA_DECAY` | `0.05` | Online ingestion EMA step rate $\eta_{\text{hash}} = 0.05$ for the 8,192-bucket causal bigram/trigram hash table. |
| `DEFAULT_ONLINE_NGRAM_HASH_BUCKETS` | `8192` | Number of causal bigram/trigram hash buckets in `Packet 2`. |
| `DEFAULT_HASH_PRIME_PREV1` | `65537` | Prime multiplier for lag-1 token $x_{t-1}$ in causal n-gram hashing. |
| `DEFAULT_HASH_PRIME_PREV2` | `1000003` | Prime multiplier for lag-2 token $x_{t-2}$ in causal n-gram hashing. |

### 6.2 Empirical Results (`Wave 74` — Executed on Local 16 GB NVIDIA GeForce RTX 5060 Ti)

#### 1. Lean 4 Formal Verification ([`lean/QuantumDLBlocksLean/FusedFactorizedSchurBaton.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/FusedFactorizedSchurBaton.lean) via [`scripts/verify_lean4_theorems.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_lean4_theorems.py))
- **Status**: **`overall_passed = True`** (`17/17` Lean 4 files, **`44/44` machine-checked theorems**, **`0` `sorry`/`admit`/custom `axiom`**, `build_time = 6.4728s`, saved to [`.cache/lean4_theorems_verification.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/lean4_theorems_verification.json)).
- **New `Wave 74` Theorems Verified**:
  1. [`theorem15a_factorized_ket_schur_action_identity`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/FusedFactorizedSchurBaton.lean#L85-L96): Proves $\big(|a\rangle\langle a| \odot |b\rangle\langle b|\big)|\psi\rangle = |a \odot b\rangle\langle a \odot b \mid \psi\rangle$ and $\operatorname{Tr}(|a\rangle\langle a| \odot |b\rangle\langle b|) = \|a \odot b\|_2^2$.
  2. [`theorem15b_diagonal_conjugated_bath_schur_action_identity`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/FusedFactorizedSchurBaton.lean#L126-L137): Proves $\big((|a\rangle\langle a|) \odot \rho_{\text{uni}}\big)|\psi\rangle = a \odot \big(\rho_{\text{uni}}(a \odot |\psi\rangle)\big)$ and $\operatorname{Tr}((|a\rangle\langle a|) \odot \rho_{\text{uni}}) = \sum_i a_i^2 (\rho_{\text{uni}})_{ii}$.
  3. [`theorem15c_gram_purity_cyclicity_identity`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/FusedFactorizedSchurBaton.lean#L166-L176): Proves $\operatorname{Tr}((B^\top B)^2) = \operatorname{Tr}((B B^\top)^2) = \|G\|_F^2$.

#### 2. Theorem 15a–15c Numerical Parity & Kernel-Fused `bfloat16` Speedup ([`scripts/fused_bf16_baton_speed_and_parity_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fused_bf16_baton_speed_and_parity_inspector.py), [`.cache/wave74_fused_bf16_parity_and_speed.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/wave74_fused_bf16_parity_and_speed.json))
- **Numerical Parity vs. Explicit $64 \times 64$ Density Matrices (`passed = True`)**:
  - `theorem15a_factorized_schur_action_max_err = 2.19e-8`, `theorem15a_factorized_schur_trace_max_err = 1.86e-9`
  - `theorem15b_diag_conjugated_bath_action_max_err = 5.94e-9`, `theorem15b_diag_conjugated_bath_trace_max_err = 5.96e-8`
  - `theorem15c_gram_purity_max_err = 1.19e-7`
- **Throughput & Peak VRAM Benchmark (`B=64, T=256` on RTX 5060 Ti)**:

| Variant | Precision & Kernel Mode | Layers | Step Throughput (`tok/s`) | Speedup vs. `W73_B3` (`1L` Step / Loop) | Speedup vs. `W73_B2` (`2L` Loop) | Peak VRAM (`MiB`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `W73_B3_EXPLICIT_FP32_1L` | Explicit `(B, T, 8, 64, 64)` `float32` | `1L` | `12,326.7` (`9,529.4` loop) | `1.00x` | — | `> 12,000` (at `B=64`) |
| **`W74_F1_FUSED_BF16_2HOP_1L`** | Factorized `(B, T, 4, 64)` `bfloat16` + `@torch.compile` | `1L` | **`162,185.1`** | **`13.16x` (`17.02x` loop)** | — | **`3,682.0 MiB`** |
| **`W74_F2_FUSED_BF16_2HOP_2L`** | Factorized `(B, T, 4, 64)` `bfloat16` + `@torch.compile` | `2L` | **`128,775.1`** | **`10.45x` (`13.51x` loop)** | **`21.88x`** (vs. `5,886.6`) | **`4,368.0 MiB`** |

#### 3. Head-to-Head `1L` vs. `2L` Sprint (`8.19M` Tokens) & Full `300,000,000`-Token (`300M` Token) Multi-Shard FineWeb Run ([`scripts/run_wave74_300m_four_compartment_spin.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/run_wave74_300m_four_compartment_spin.py), [`.cache/wave74_300m_spin_summary.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/wave74_300m_spin_summary.json))

| Run Key | Layers | Params (Trunk / Total) | Steps | Online Ingested Tokens | Step 0 Val CE | Best Val CE | Final Val CE | Final Val PPL | Top-1 Acc | Output Purity $\operatorname{Tr}(\rho^2)$ | Final Gates $(w_{\text{id}}, w_{\text{deep}}, w_{\text{ngram}}, w_{\text{uni}})$ | Sustained `tok/s` (Wall Time) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w74_f1_bf16_fused_2hop_1l_sprint` | `1L` | `199K / 6.63M` | `500` | `8,192,000` | `12.8410` | `6.3700` | `6.3700` | `584.04` | `17.32%` | `0.5002` | `(0.111, 0.600, 0.177, 0.112)` | `104,775.7` (`78.2s` incl. compile) |
| **`w74_f2_bf16_fused_2hop_2l_sprint`** | **`2L`** | **`389K / 6.82M`** | `500` | `8,192,000` | `12.0887` | **`6.1138`** | **`6.1138`** | **`452.07`** | **`18.51%`** | `0.4668` | `(0.141, 0.535, 0.191, 0.132)` | **`123,049.3`** (`66.6s`) |
| **`w74_bf16_fused_2hop_2l_300m_spin`** | **`2L`** | **`389K / 6.82M`** | **`18,311`** | **`300,007,424`** | `12.0887` | **`5.2158`** | **`5.2172`** | **`184.42`** | **`23.26%`** | `0.3417` | `(0.203, 0.792, 0.002, 0.003)` | **`129,684.7`** (`2,313.4s` / `38.6m`) |

#### 4. Multi-Shard `300M`-Token Progression Across `FineWeb` Shards 1, 2, and 3 (`w74_bf16_fused_2hop_2l_300m_spin`)

| Step | Tokens Ingested (`M`) | Active Shard | Train CE | Val CE (`fineweb_val_000000.bin`) | Val PPL | Top-1 Acc | Output Purity $\operatorname{Tr}(\rho^2)$ | Online Bath Purity $\operatorname{Tr}(\rho_{\text{uni}}^2)$ | Gates $(w_{\text{id}}, w_{\text{deep}}, w_{\text{ngram}}, w_{\text{uni}})$ | Sustained `tok/s` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | Shard 1 (Cold) | — | `12.0887` | `177,858.45` | `1.93%` | `0.3558` | `0.015625` ($1/64$) | `(0.183, 0.326, 0.283, 0.209)` | — |
| `1,500` | `24.58M` | Shard 1 | `5.4736` | `5.5898` | `267.67` | `20.99%` | `0.3728` | `0.048677` | `(0.109, 0.844, 0.017, 0.028)` | `130,089.1` |
| `3,000` | `49.15M` | Shard 1 | `5.4860` | `5.4321` | `228.63` | `21.77%` | `0.3511` | `0.062123` | `(0.111, 0.880, 0.002, 0.006)` | `129,967.3` |
| `4,500` | `73.73M` | Shard 1 | `5.2097` | `5.3692` | `214.69` | `22.20%` | `0.3464` | `0.064194` | `(0.133, 0.863, 0.002, 0.003)` | `129,922.1` |
| `6,000` | `98.30M` | Shard 1 | `5.5092` | `5.3341` | `207.30` | `22.48%` | `0.3442` | `0.067232` | `(0.150, 0.844, 0.002, 0.003)` | `129,900.7` |
| `7,500` | `122.88M` | Shard 2 | `5.2693` | `5.3117` | `202.69` | `22.51%` | `0.3410` | `0.067509` | `(0.156, 0.840, 0.002, 0.003)` | `129,888.2` |
| `9,000` | `147.46M` | Shard 2 | `5.4016` | `5.2863` | `197.61` | `22.69%` | `0.3397` | `0.067920` | `(0.168, 0.828, 0.002, 0.003)` | `129,879.2` |
| `10,500` | `172.03M` | Shard 2 | `5.2182` | `5.2665` | `193.74` | `22.85%` | `0.3396` | `0.068809` | `(0.177, 0.818, 0.002, 0.003)` | `129,873.8` |
| `12,000` | `196.61M` | Shard 2 | `5.3602` | `5.2528` | `191.09` | `23.01%` | `0.3390` | `0.068814` | `(0.184, 0.812, 0.002, 0.003)` | `129,871.7` |
| `13,500` | `221.18M` | Shard 3 | `5.1125` | `5.2390` | `188.48` | `23.05%` | `0.3397` | `0.068776` | `(0.188, 0.807, 0.002, 0.003)` | `129,867.5` |
| `15,000` | `245.76M` | Shard 3 | `4.8568` | `5.2256` | `185.97` | `23.17%` | `0.3403` | `0.069967` | `(0.193, 0.801, 0.002, 0.003)` | `129,864.2` |
| `18,000` | `294.91M` | Shard 3 | `5.1538` | **`5.2158`** | **`184.17`** | `23.25%` | `0.3415` | `0.070669` | `(0.200, 0.795, 0.002, 0.003)` | `129,861.0` |
| `18,311` | `300.01M` | Shard 3 End | `5.4691` | `5.2172` | `184.42` | **`23.26%`** | `0.3417` | `0.071173` | `(0.203, 0.792, 0.002, 0.003)` | `129,782.7` |

#### 5. Key Physical & Architectural Takeaways for Next Waves
1. **`2L` 4-Way Schur Intersection Outperforms `1L` by `-0.2562` Nats at `129.7k tok/s`**:
   - At `8.19M` tokens (`500` steps), the 2-layer 2-Hop Triangle model (`w74_f2_bf16_fused_2hop_2l_sprint`, `6.1138` Val CE, `18.51%` top-1) beats the 1-layer model (`w74_f1_bf16_fused_2hop_1l_sprint`, `6.3700` Val CE, `17.32%` top-1) by **`-0.2562` nats** (`+1.19%` top-1 accuracy) while running **`21.88x` faster** than `Wave 73`'s `2L` FP32 implementation (`128,775 tok/s` vs. `5,886.6 tok/s`).
2. **Spontaneous Two-Phase Regime Shift in 4-Compartment POVM Gates (`0M -> 8M -> 300M` Tokens)**:
   - During the first `0M -> 8.2M` tokens, the model relies heavily on standalone n-gram backoff (`w_ngram = 0.283 -> 0.191`) and the online unigram thermal bath (`w_uni = 0.209 -> 0.132`).
   - Beyond `25M` tokens, because $\rho_{\text{deep}} = \rho_{\text{seq}} \odot \rho_{\text{ngram}}$ already incorporates the matured $K=4, \chi=8$ MPS baton and Lindblad transition operators via Schur intersection, the standalone backoff gates naturally anneal (`w_ngram -> 0.002, w_uni -> 0.003` while the unigram log-prior remains active at `DEFAULT_UNIGRAM_PRIOR_WEIGHT = 0.25`), and probability mass concentrates into a clean **80/20 division of labor** between the **4-way Schur Deep+N-Gram Intersection (`w_deep = 0.792`)** and **Walsh–Hadamard Token-ID In-Context Copying (`w_id = 0.203`, rising steadily from `0.109` at `24.6M` to `0.203` at `300.0M` tokens)**.
3. **Why Validation CE Asymptotes Near `5.2158` (`389K` Trunk Parameters at $d = 64$)**:
   - Out of `6,821,512` total parameters in `w74_bf16_fused_2hop_2l_300m_spin`, `6,432,896` (`94.3%`) reside in the $d=64$ vocabulary embedding table (`token_ket_re`, `token_ket_im`), leaving only **`388,616` (`389K`) parameters in the entire 2-layer trunk** (`194K` parameters/layer) and restricting the vocabulary readout to a rank-$128$ real bottleneck (`packed_readout: (B, T, 128)`).
   - Scaling the per-compartment Hilbert dimension from $d = 64 \to 192$ or $256$ (or reading out from the concatenated 4-beam tensor `(B, T, R=4, d=64)` $\in \mathbb{C}^{256}$ = `512` real features instead of collapsing the 4 beams to a single $d=64$ ket before vocabulary projection) is the immediate high-leverage step to unlock `< 3.5` Val CE while retaining the `100k+ tok/s` factorized `bfloat16` speed.

