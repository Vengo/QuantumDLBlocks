# EXPERIMENT PLAN 082: Two 8-Qubit Registers ("Qubytes") $(\mathbb{C}^2)^{\otimes 8}_{\text{hi}} \otimes (\mathbb{C}^2)^{\otimes 8}_{\text{lo}}$ with 8-Qubit Walsh-Hadamard Pauli-$Z$ Basis, Step-1 Multi-Mode Qubyte Distribution, Step-3 Latent `512D` Contraction, & Rank-$R$ Schmidt Entanglement (`Wave 62`)

## 1. Executive Summary & Architectural Motivation

In [`EXPERIMENT_PLAN_081.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_081.md) (`Wave 61`), we established a strict `modded-nanogpt` speedrun-compliant (`0` offline pre-scanned tokens) **Deterministic `uint16` Radix Byte-Pair `(b_hi, b_lo)` Decomposition**:
$$b_{\text{hi}}(v) \;=\; v \gg 8 \;\in\; \{0, \dots, 196\}, \qquad b_{\text{lo}}(v) \;=\; v \ \& \ 255 \;\in\; \{0, \dots, 255\}, \qquad v \;=\; 256\,b_{\text{hi}}(v) + b_{\text{lo}}(v)$$
Factoring token wavepackets and readout detectors over `(b_hi, b_lo)` (`Z2`) improved 300-step Validation CE from `6.2710` (`Z0`) to **`6.1292` (`459.07` PPL, `18.37%` Top-1)** and scaled to **`5.2807` Val CE (`196.51` PPL, `24.34%` Top-1)** at 1,200 steps (`Z5`), closing `94.4%` of the cold-start gap against `900M`-token offline PPMI calibration (`N4`).

However, `EXPERIMENT_PLAN_081` exposed two fundamental architectural questions:
1. **Why was the multi-mode Cholesky readout (`Z2`/`Z4`/`Z5`) compute-intensive (`885s` at $R=8$, `1,264s` at $R=12$), and are Step 1 and Step 3 useful representations for language modeling?**
   - In `Z2`/`Z4`/`Z5`, `_VacuumFourMode512DWavepacketCEFn` evaluated **$2R + 1$ full `50,304`-token vocabulary matmuls in forward** and **$4R + 2$ in backward** (`51` full-vocab GEMMs/step at $R=8$; `75` full-vocab GEMMs/step at $R=12$), plus $R$ full `50,304`-way `tanh` + `log_softmax` reductions per token.
   - **Step 3 (Contracting the $R$ Cholesky Modes in `512D` Latent Space First)**: Summing the vacuum-weighted Cholesky modes inside `512D` Hilbert space before multiplying by $W_{\text{vocab}} \in \mathbb{R}^{50304 \times 512}$ reduces the `50,304`-token vocabulary GEMMs from $2R + 1$ down to **`1` in forward and `2` in backward** (`~4x–5x` speedup). However, because a single linear dot product $\langle w_v \mid \sum_r \pi_{t,r} h_{t,r} \rangle$ is linear in $h$, **pure Step 3 collapses the $R$ Cholesky modes to a Rank-1 mean vector at the readout**, losing multimodal branching unless paired with a nonlinear multi-mode projection.
   - **Step 1 (Distributing the $R$ Cholesky Modes Over the `(b_hi, b_lo)` Basis Before Vocabulary Expansion)**: Projecting all $R$ Cholesky modes onto the compact `197` high-byte and `256` low-byte basis states requires only **$197 + 256 = 453$ dot products per mode** (**`111x` fewer multi-mode FLOPs** than `50,304`). Combining **Step 1** (all $R$ Cholesky modes act nonlinearly on the `453` byte-pair states) with **Step 3** (the contracted `512D` vector queries the `50,304` fine residual $\Delta W[v]$ **only once**) preserves full Rank-$R$ multi-modality across all $197 \times 256$ byte-pair cells while keeping the `50,304`-vocabulary cost at **1 GEMM**!
2. **Can we represent the `50,257` vocabulary as Two 8-Qubit Registers ("Qubytes": $\mathcal{H}_{\text{hi}} \otimes \mathcal{H}_{\text{lo}} = (\mathbb{C}^2)^{\otimes 8} \otimes (\mathbb{C}^2)^{\otimes 8}$)?**
   - In `Z2`, `b_hi` (`0..196`) and `b_lo` (`0..255`) were treated as unstructured categorical lookup tables where adjacent byte values (e.g. `12 = 00001100_2` and `13 = 00001101_2`) shared zero internal qubit structure.
   - Representing `b_hi` and `b_lo` as **Two 8-Qubit Registers ("Qubytes")** (`16` qubits total: $|q_{15}\dots q_8\rangle_{\text{hi}} \otimes |q_7\dots q_0\rangle_{\text{lo}}$) unlocks two quantum-mechanical mechanisms:
     1. **Exact 8-Qubit Walsh-Hadamard / Pauli-$Z^{\otimes S}$ Tensor Basis ($H^{\otimes 8} \in \mathbb{R}^{256 \times 256}$) & 16-Qubit Bloch-Ising Hamiltonian**: Every state $|b\rangle$ of an 8-qubit register is coupled across all 256 states via the exact 8-qubit Hadamard unitary $(H^{\otimes 8})_{b,S} = \frac{1}{16}(-1)^{\operatorname{popcount}(b\,\&\,S)}$ ordered by Pauli $k$-body Hamming weight $|S| = \operatorname{popcount}(S) \in \{0,\dots,8\}$, plus direct 16-qubit single-spin Pauli-$Z_j$ fields and cross-register 2-qubit Ising couplings $Z_i^{\text{hi}} Z_j^{\text{lo}}$.
     2. **Rank-$R$ Bipartite Schmidt Entanglement Across the Two 8-Qubit Registers**: In a bipartite two-register quantum system $\mathcal{H}_{\text{hi}} \otimes \mathcal{H}_{\text{lo}}$, our $R$ Cholesky wavepacket modes naturally form a **Rank-$R$ Schmidt decomposition**:
        $$|\Psi_t\rangle_{\text{hi},\text{lo}} \;=\; \sum_{r=0}^{R-1} \pi_{t,r}\, |\alpha_{t,r}\rangle_{\text{hi}} \otimes |\beta_{t,r}\rangle_{\text{lo}} \;\in\; \mathbb{R}^{197 \times 256}$$
        evaluated via a single batched matrix multiply `(N, 197, R) @ (N, R, 256)` (`inner dim R = 8` or `12`), allowing each Cholesky mode $r$ to entangle a specific high-byte frequency tier $b_{\text{hi}}^{(r)}$ with its corresponding low-byte distribution $b_{\text{lo}}^{(r)}$ without cross-term ghosting!

---

## 2. Quantitative Hypotheses (`H1–H4`)

- **`H1` (16-Qubit Pauli-$Z$ / Walsh-Hadamard Structure & Step-1 / Step-3 Algebraic Identity)**:
  - The 8-qubit Walsh-Hadamard transform $H^{\otimes 8} \in \mathbb{R}^{256 \times 256}$ satisfies exact orthogonality $\|(H^{\otimes 8})^T H^{\otimes 8} - I_{256}\|_{\max} < 10^{-6}$.
  - Low-order Pauli-$Z^{\otimes S}$ observables ($k$-body weight $|S| \le 3$, comprising $1 + 8 + 28 + 56 = 93$ of $256$ Pauli strings) capture **`> 70%`** of the variance of the empirical FineWeb log-frequency landscape across `b_hi`.
  - Distributing $R$ Cholesky modes over the `(b_hi, b_lo)` basis (`453` vectors, **Step 1**) + contracting the fine residual in `512D` (**Step 3**) reduces multi-mode vocabulary projection FLOPs by **`111.0x`** (`453` vs. `50,304` dot products/mode) and reduces full `50,304` vocabulary GEMMs from $2R + 1$ (`17` at $R=8$, `25` at $R=12$) down to **`1`**.
- **`H2` (Step 3 Alone `Q0` vs. Step 1 + Step 3 `Q1` at 300 Steps)**:
  - **Step 3 Alone (`Q0_STEP3_ONLY_LATENT_512D_CONTRACTION`, $R=8$)** cuts 300-step wall-clock training + validation time by **`>= 3.0x`** compared to `Z2` (`<= 280s` vs. `885.3s`), while achieving `< 6.20` Val CE.
  - **Step 1 + Step 3 (`Q1_STEP1_PLUS_STEP3_FACTORED_MULTIMODE`, $R=8$)** preserves the **`>= 3.0x` wall-clock speedup** while beating Step 3 alone (`Q0`) in Validation CE (`<= 6.12` Val CE) and maintaining high active Cholesky mode diversity via multi-mode EM posterior responsibilities on the `(197, 256)` byte-pair grid.
- **`H3` (Two 8-Qubit Registers with Walsh-Hadamard Pauli-$Z$ Basis `Q2` & Rank-$R$ Schmidt Entanglement `Q3`/`Q4` at 300 Steps)**:
  - Representing `(b_hi, b_lo)` as **Two 8-Qubit Registers** with the 8-qubit Walsh-Hadamard Pauli-$Z^{\otimes S}$ spectral basis and 16-qubit Bloch-Ising couplings (`Q2`), combined with **Rank-$R$ Bipartite Schmidt Entanglement** $\sum_{r=0}^{R-1} \pi_{t,r} |\alpha_{t,r}\rangle_{\text{hi}} \otimes |\beta_{t,r}\rangle_{\text{lo}}$ (`Q3` at $R=8$, `Q4` at $R=12$), beats `Z2` (`6.1292` Val CE) and `Z4` (`6.1490` Val CE) on the official `10,485,760`-token FineWeb Shard-0 benchmark while running **`3.2x–4.5x` faster** (`< 300s` even at $R=12$ vs. `1,264.2s` for `Z4`).
- **`H4` (1,200-Step / `9.83M`-Token Strict Speedrun-Compliant Two-Qubyte Schmidt Scaling `Q5`)**:
  - Scaling the $R=12$ Two 8-Qubit Register Schmidt-Entangled architecture (`Q5_TWO_8QUBIT_SCHMIDT_SCALING_1200`) to `1,200` steps (`9.83M` training tokens, `0` pre-scanned tokens) matches or beats `Z5` (`5.2807` Val CE / `196.51` PPL) in **`< 1,400s`** (**`> 3.5x` faster** than `Z5`'s `5,050s`).

---

## 3. Mathematical Formulation

### 3.1 Two 8-Qubit Registers ("Qubytes"): $\mathcal{H}_{\text{hi}} \otimes \mathcal{H}_{\text{lo}} \cong (\mathbb{C}^2)^{\otimes 8} \otimes (\mathbb{C}^2)^{\otimes 8}$

Every GPT-2 token ID $v \in \{0, \dots, 50256\}$ is a 16-bit integer whose binary expansion defines a computational basis state of **two 8-qubit registers**:
$$|v\rangle \;=\; |b_{\text{hi}}(v)\rangle_A \otimes |b_{\text{lo}}(v)\rangle_B \;=\; \bigotimes_{i=0}^{7} |q_{8+i}(v)\rangle_A \;\otimes\; \bigotimes_{j=0}^{7} |q_j(v)\rangle_B, \qquad q_k(v) = (v \gg k) \,\&\, 1 \in \{0, 1\}$$
Each qubit $j \in \{0, \dots, 7\}$ of byte $b \in \{0, \dots, 255\}$ has Pauli-$Z_j$ spin eigenvalue:
$$s_j(b) \;=\; (-1)^{q_j(b)} \;=\; 1 - 2\big((b \gg j)\,\&\,1\big) \;\in\; \{+1, -1\}$$

#### Exact 8-Qubit Walsh-Hadamard / Pauli-$Z^{\otimes S}$ Unitary Basis ($H^{\otimes 8}$)
The $2^8 = 256$ tensor-product Pauli-$Z$ operators $Z^S = \bigotimes_{j=0}^{7} Z_j^{S_j}$ ($S \in \{0, \dots, 255\}$) are diagonal in the computational basis with eigenvalues:
$$H^{\otimes 8}_{b, S} \;=\; \frac{1}{\sqrt{256}}\,(-1)^{\operatorname{popcount}(b \,\&\, S)}, \qquad (H^{\otimes 8})^T H^{\otimes 8} \;=\; I_{256}$$
where the Hamming weight $|S| = \operatorname{popcount}(S) \in \{0, \dots, 8\}$ is the **Pauli $k$-body interaction order**:
- $|S| = 0$ (`1` state): Global register identity $I^{\otimes 8}$.
- $|S| = 1$ (`8` states): Single-qubit Bloch-$Z_j$ magnetizations (coarse binary dyadic splits of BPE rank in Register A and ASCII/byte sub-ranges in Register B).
- $|S| = 2$ (`28` states): Pairwise 2-qubit Ising couplings $Z_i Z_j$.
- $|S| \in \{3, \dots, 8\}$ (`219` states): Multi-qubit parity correlations.

We parameterize the High 8-Qubit Register ($197$ active states) and Low 8-Qubit Register ($256$ states) detector and wavepacket bases as a superposition of computational-basis tables and Pauli-$Z^{\otimes S}$ spectral basis tables gated by interaction order $|S|$:
$$D^{\text{qubyte}}_{\text{hi}} \;=\; D^{\text{comp}}_{\text{hi}} \;+\; \gamma_{\text{walsh}}\, H^{\otimes 8}_{[0:197, :]}\,\operatorname{diag}\!\big(\sigma(\eta_{\text{hi}})\big)\, V^{\text{pauli}}_{\text{hi}} \;\in\; \mathbb{R}^{197 \times 512}$$
$$D^{\text{qubyte}}_{\text{lo}} \;=\; D^{\text{comp}}_{\text{lo}} \;+\; \gamma_{\text{walsh}}\, H^{\otimes 8}\,\operatorname{diag}\!\big(\sigma(\eta_{\text{lo}})\big)\, V^{\text{pauli}}_{\text{lo}} \;\in\; \mathbb{R}^{256 \times 512}$$
with Step-0 locality prior $\eta_S^{(0)} = \eta_{\text{base}} - \eta_{\text{decay}}\operatorname{popcount}(S)$ (`DEFAULT_PAULI_BASE_LOGIT = 0.80`, `DEFAULT_PAULI_HAMMING_DECAY = 0.35`).

### 3.2 Step 3 (Latent `512D` Contraction) + Step 1 (Multi-Mode Qubyte Projection) + Rank-$R$ Schmidt Entanglement

Given the trunk anchor semantic state $h_{t,\text{sem},0} \in \mathbb{R}^{448}$, the $R$ vacuum-normalized Cholesky branches $(h_{t,\text{sem},r}, h_{t,\text{id},r}) \in \mathbb{R}^{448} \times \mathbb{R}^{64}$, and branch controls $(\beta_{t,r}, \tau_{t,r}, \alpha_{t,r})$ with $\pi_{t,r} = \operatorname{softmax}_r(\alpha_{t,r})$:

1. **Step 3 — Latent `512D` Mode Contraction for the Single `50,304` Vocabulary GEMM**:
   $$\bar{h}_{t,\text{sem}} \;=\; h_{t,\text{sem},0} \;+\; \sum_{r=0}^{R-1} \pi_{t,r}\,\beta_{t,r}\,h_{t,\text{sem},r} \;\in\; \mathbb{R}^{448}, \qquad \bar{h}_{t,\text{id}} \;=\; \sum_{r=0}^{R-1} \pi_{t,r}\,\tau_{t,r}\,h_{t,\text{id},r} \;\in\; \mathbb{R}^{64}$$
   $$\bar{h}_{t,\text{aug}} \;=\; \big[\,\bar{h}_{t,\text{sem}},\;\; \bar{h}_{t,\text{id}},\;\; h_{t,\text{car}}\,\big] \;\in\; \mathbb{R}^{528}, \qquad z^{\text{step3}}_{t,v} \;=\; \langle W_{\text{aug}}[v] \mid \bar{h}_{t,\text{aug}} \rangle$$
   This requires **only 1** `(N, 528) @ (528, 50304)` GEMM in forward and **2** in backward.

2. **Step 1 — Multi-Mode Cholesky Projection onto the Two 8-Qubit Registers (`453` Basis Vectors, `111x` Fewer Multi-Mode FLOPs)**:
   Each individual Cholesky mode $r \in \{0, \dots, R-1\}$ projects onto the High-Qubyte (`197`) and Low-Qubyte (`256`) detector bases:
   $$u^{\text{hi}}_{t,r} \;=\; \langle \hat{D}^{\text{qubyte}}_{\text{hi},\text{sem}} \mid h_{t,\text{sem},0} + \beta_{t,r} h_{t,\text{sem},r} \rangle \;+\; \tau_{t,r} \langle \hat{D}^{\text{qubyte}}_{\text{hi},\text{id}} \mid h_{t,\text{id},r} \rangle \;\in\; \mathbb{R}^{197}$$
   $$u^{\text{lo}}_{t,r} \;=\; \langle \hat{D}^{\text{qubyte}}_{\text{lo},\text{sem}} \mid h_{t,\text{sem},0} + \beta_{t,r} h_{t,\text{sem},r} \rangle \;+\; \tau_{t,r} \langle \hat{D}^{\text{qubyte}}_{\text{lo},\text{id}} \mid h_{t,\text{id},r} \rangle \;\in\; \mathbb{R}^{256}$$
   With mode-wise normalized qubyte distributions $P^{\text{hi}}_{t,r} = \operatorname{softmax}(u^{\text{hi}}_{t,r} + \ln P_0(b_{\text{hi}}))$ and $P^{\text{lo}}_{t,r} = \operatorname{softmax}(u^{\text{lo}}_{t,r})$, the **multi-mode log-density innovation** over the Rank-1 mean product is:
   $$\Delta \Phi^{\text{step1}}_t[b_{\text{hi}}, b_{\text{lo}}] \;=\; \ln\!\left(\sum_{r=0}^{R-1} \pi_{t,r}\, P^{\text{hi}}_{t,r}[b_{\text{hi}}]\, P^{\text{lo}}_{t,r}[b_{\text{lo}}] + \varepsilon\right) \;-\; \ln\!\left(\bar{P}^{\text{hi}}_t[b_{\text{hi}}]\,\bar{P}^{\text{lo}}_t[b_{\text{lo}}] + \varepsilon\right)$$

3. **Rank-$R$ Bipartite Schmidt Entanglement & 16-Qubit Bloch-Ising Coupling Across the Two 8-Qubit Registers (`Q3..Q5`)**:
   Each Cholesky mode $r \in \{0, \dots, R-1\}$ generates bounded Schmidt factor states $\alpha_{t,r} = \tanh(u^{\text{hi}}_{t,r} / \tau_{\text{schmidt}}) \in \mathbb{R}^{197}$ and $\beta_{t,r} = \tanh(u^{\text{lo}}_{t,r} / \tau_{\text{schmidt}}) \in \mathbb{R}^{256}$, forming the **Rank-$R$ entangled two-qubyte operator**:
   $$\Omega^{\text{Schmidt}}_t[b_{\text{hi}}, b_{\text{lo}}] \;=\; \gamma_{\text{schmidt}} \sum_{r=0}^{R-1} \pi_{t,r}\, \alpha_{t,r}[b_{\text{hi}}]\, \beta_{t,r}[b_{\text{lo}}] \;+\; \gamma_{\text{ising}} \big(\langle m^{\text{hi}}_t \mid s_{\text{hi}}(b_{\text{hi}}) \rangle + \langle m^{\text{lo}}_t \mid s_{\text{lo}}(b_{\text{lo}}) \rangle + s_{\text{hi}}(b_{\text{hi}})^T J_t s_{\text{lo}}(b_{\text{lo}})\big)$$
   Both $\Delta \Phi^{\text{step1}}_t$ and $\Omega^{\text{Schmidt}}_t$ are `(N, 197, 256)` tensors computed via tiny `(N, 197, R) @ (N, R, 256)` batched matmuls and added directly to $z^{\text{step3}}_{t,v}$ inside the single-pass chunked autograd kernel `_SingleGemmStep3QubyteCEFn`.

---

## 4. Planned Changes & Verification Architecture

1. **New Flag-Free Benchmark Module ([`src/qdlblocks/benchmarks/two_qubyte_register_cholesky_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/two_qubyte_register_cholesky_lab.py))**:
   - `build_8qubit_walsh_hadamard_and_spins`: Exact $256 \times 256$ 8-qubit Walsh-Hadamard Pauli-$Z^{\otimes S}$ matrix $H^{\otimes 8}$, Pauli Hamming weight vector $|S| \in \{0 \dots 8\}$, and $\{\pm 1\}^8$ spin table.
   - `Two8QubitRegisterSynthesizer`: Parameterizes the two 8-qubit registers $(\mathbb{C}^2)^{\otimes 8}_{\text{hi}} \otimes (\mathbb{C}^2)^{\otimes 8}_{\text{lo}}$ with Pauli-$Z^{\otimes S}$ spectral basis tables, 16-qubit Bloch-Ising Hamiltonian projections, Step-1 multi-mode `(197, 256)` log-density innovation, and Rank-$R$ Schmidt entanglement.
   - `_SingleGemmStep3QubyteCEFn`: Rule-9 single-pass chunked custom autograd function executing **1 forward and 2 backward `50,304` GEMMs** per token chunk while propagating exact gradients to both `h_step3_528` and `qubyte_grid_197x256`.
   - `TwoQubyteRegisterCholeskyGPT`: Inherits `ZeroPPMIBytePairCholeskyGPT` (`init_tokens_scanned == 0`, strict `modded-nanogpt` compliance) and dispatches `Q0..Q5`.
2. **Diagnostic & Benchmark CLI Scripts**:
   - [`scripts/two_qubyte_register_and_step1_step3_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/two_qubyte_register_and_step1_step3_profiler.py)
   - [`scripts/two_qubyte_register_cholesky_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/two_qubyte_register_cholesky_bench.py)
3. **Verification Script & Unit Tests**:
   - [`examples/92_verify_plan082_two_qubyte_register_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/92_verify_plan082_two_qubyte_register_cholesky.py)
   - [`tests/test_plan082_two_qubyte_register_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan082_two_qubyte_register_cholesky.py)

---

## 5. Operational Tooling & Script Identification

### 5.1 Existing Scripts Used
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py --paths src/qdlblocks/benchmarks/two_qubyte_register_cholesky_lab.py scripts/two_qubyte_register_and_step1_step3_profiler.py scripts/two_qubyte_register_cholesky_bench.py examples/92_verify_plan082_two_qubyte_register_cholesky.py tests/test_plan082_two_qubyte_register_cholesky.py`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/verify_plan.py --plan-num 82 --json-out .cache/verify_plan082.json`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload`

### 5.2 `Build-Before-Launch` Scripts
1. [`scripts/two_qubyte_register_and_step1_step3_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/two_qubyte_register_and_step1_step3_profiler.py): Audits the 16-qubit binary/Pauli-$Z$ marginal entropies, 8-qubit Walsh-Hadamard orthogonality (`< 1e-6`), Pauli Hamming-weight cumulative variance capture on FineWeb Shard 0, Step 1 algebraic equivalence, and wall-clock readout throughput (`Z2` vs. `Q0` Step 3 vs. `Q1` Step 1+3 vs. `Q3` Two-Qubyte Schmidt).
2. [`scripts/two_qubyte_register_cholesky_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/two_qubyte_register_cholesky_bench.py): Runs the 300-step (`2.46M` tokens) `Q0..Q4` ablation matrix and the 1,200-step (`9.83M` tokens) `Q5` scaling run with `StandardizedExperimentLogger`.

### 5.3 `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/qubyte_pauli_spectrum_and_schmidt_entanglement_inspector.py`: Visualizes the learned 8-qubit Pauli-$Z^{\otimes S}$ Hamming-weight spectrum $\sigma(\eta_S)$, the $8 \times 8$ cross-register Ising coupling matrix $J$, and the Schmidt entanglement entropy $S(\rho_A)$ across high-byte vs. low-byte registers.

---

## 6. Constants Ledger & Empirical Results (`Wave 62`)

### 6.1 Rule 7 Named Constants Ledger (`src/qdlblocks/benchmarks/two_qubyte_register_cholesky_lab.py`)

| Constant Name | Value | Mathematical Role & Rationale |
| :--- | :---: | :--- |
| `QUBYTE_NUM_QUBITS` | `8` | Qubits per register ($2^8 = 256$ states per qubyte; `16` qubits total across $(\mathbb{C}^2)^{\otimes 8}_{\text{hi}} \otimes (\mathbb{C}^2)^{\otimes 8}_{\text{lo}}$). |
| `QUBYTE_HILBERT_DIM` | `256` | Dimension $2^8 = 256$ of each 8-qubit register Walsh-Hadamard unitary $H^{\otimes 8}$. |
| `DEFAULT_PAULI_BASE_LOGIT` | `0.80` | Step-0 logit offset $\eta_{\text{base}}$ for low-weight Pauli-$Z^{\otimes S}$ observables. |
| `DEFAULT_PAULI_HAMMING_DECAY` | `0.35` | Step-0 decay per Hamming weight $\operatorname{popcount}(S)$ favoring 1-qubit and 2-qubit Ising terms. |
| `DEFAULT_WALSH_BASIS_SCALE` | `0.25` | Blending scale $\gamma_{\text{walsh}}$ for the 8-qubit Walsh-Hadamard Pauli-$Z^{\otimes S}$ basis table. |
| `DEFAULT_STEP1_INNOVATION_SCALE` | `0.35` | Coupling scale for the Step-1 multi-mode `(197, 256)` log-density innovation $\Delta \Phi^{\text{step1}}_t$. |
| `DEFAULT_SCHMIDT_ENTANGLE_SCALE` | `0.40` | Coupling scale $\gamma_{\text{schmidt}}$ for the Rank-$R$ bipartite Schmidt entanglement tensor $\sum_r \pi_{t,r} \alpha_{t,r} \otimes \beta_{t,r}$. |
| `DEFAULT_SCHMIDT_TEMP` | `2.50` | Temperature $\tau_{\text{schmidt}}$ bounding the per-mode qubyte Schmidt factors $\alpha_{t,r}, \beta_{t,r}$. |
| `DEFAULT_ISING_COUPLING_SCALE` | `0.15` | Scale $\gamma_{\text{ising}}$ for the 16-qubit Bloch-magnetization and cross-register Pauli-$ZZ$ Ising Hamiltonian. |
| `DEFAULT_QUBYTE_AUX_LOSS_WEIGHT` | `0.08` | Weight $\lambda_{\text{aux}}$ for the exact multi-mode `(197, 256)` two-qubyte EM responsibility auxiliary NLL. |

### 6.2 Empirical Results & Architectural Post-Mortem (`Wave 62`)

1. **Two 8-Qubit Registers ("Qubytes") & Step-1/Step-3 Profiler ([`.cache/two_qubyte_step1_step3_profile.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/two_qubyte_step1_step3_profile.json))**:
   - **Exact 8-Qubit Walsh-Hadamard Unitarity**: $\|(H^{\otimes 8})^T H^{\otimes 8} - I_{256}\|_{\max} = 0.0$.
   - **Pauli-$Z^{\otimes S}$ Hamming-Weight Energy Capture on FineWeb Shard 0 (`10,485,760` tokens)**:
     - In the **High 8-Qubit Register (`b_hi`, qubits $q_{15}\dots q_8$)**, the 8 single-qubit Pauli-$Z_j$ magnetizations (`weight_1`) alone capture **`80.00%`** of the log-frequency variance; adding the 28 pairwise Ising couplings $Z_i Z_j$ (`weight_2`) captures **`94.33%`**, and `weight <= 3` captures **`96.77%`**.
     - In the **Low 8-Qubit Register (`b_lo`, qubits $q_7\dots q_0$)**, single-qubit marginal entropy is near-maximal (`7.8670 / 8.0000` bits total, `0.9391–1.0000` bits/qubit), and log-frequency variance is distributed across higher-order multi-qubit Pauli strings (`15.52%` at $|S|=1$, `26.33%` cumulative at $|S|\le 2$, `45.05%` at $|S|\le 3$, `80.18%` at $|S|\le 5$).
   - **Step-1 Algebraic Equivalence & Step-3 Throughput (`B=16, T=512`, `8,192` tok/step)**:
     - Projecting $R$ Cholesky modes onto `(197 + 256 = 453)` byte-pair basis states matches direct `50,257`-token projection to **`1.14e-5`** max error while reducing multi-mode dot products by **`110.94x`** (`453` vs. `50,257`).
     - Training step time (`forward + backward`):
       - `Z2` ($R=8$, $1+2R$ full `50,304` GEMMs): `2,786.57 ms/step` (`2,939.8 tok/s`, `1.00x`)
       - `Q0` (**Step 3 Only**, $R=8$, `1` full `50,304` GEMM): **`879.36 ms/step` (`9,315.9 tok/s`, `3.17x` faster)**
       - `Q1` (**Step 1 + Step 3**, $R=8$): **`963.20 ms/step` (`8,505.0 tok/s`, `2.89x` faster)**
       - `Q3` (**Two 8-Qubit Registers + Schmidt Entanglement**, $R=8$): **`976.63 ms/step` (`8,388.0 tok/s`, `2.85x` faster)**
       - `Q4` (**Two 8-Qubit Registers + Schmidt Entanglement**, $R=12$): **`1,237.15 ms/step` (`6,621.7 tok/s`, `2.25x` faster than $R=8$ `Z2`)**

2. **300-Step Empirical Run (`Q0` Completed, `Q1` Partial Before Cancellation for Architectural Consolidation)**:
   - **`Q0_STEP3_ONLY_LATENT_512D_CONTRACTION` ($R=8$, `0` tokens pre-scanned)**:
     - Step 0: `9.0617` Val CE (`8618.97` PPL, `Top1: 0.58%`, `Modes: 5.43/8`)
     - Step 100 (`0.82M` tok): `6.5939` Val CE (`730.63` PPL, `Top1: 14.35%`, `Top5: 28.69%`, `Modes: 2.73/8`, `p_noclick: 0.288`)
     - Step 200 (`1.64M` tok): `6.2598` Val CE (`523.12` PPL, `Top1: 16.29%`, `Top5: 31.61%`, `Modes: 1.59/8`, `p_noclick: 0.281`)
     - Step 300 (`2.46M` tok): **`6.0794` Val CE (`436.76` PPL, `Top1: 17.55%`, `Top5: 33.41%`, `Modes: 1.50/8`, `p_noclick: 0.276`)** — **beating `Z2` (`6.1292` Val CE / `459.07` PPL) by `-0.0498` nats (`-22.3` PPL)** while using only **1** `50,304` vocabulary GEMM per token!
   - **`Q1_STEP1_PLUS_STEP3_FACTORED_MULTIMODE` ($R=8$, partial before cancellation at Step 200)**:
     - Step 0 `9.0623` $\to$ Step 100 `6.5960` $\to$ Step 200 **`6.2415` Val CE (`513.64` PPL, `-0.0183` nats ahead of `Q0` at Step 200)**.

3. **Post-Mortem & Rationale for Stopping to Consolidate**:
   - Why wall-clock run time during full `10.48M`-token validation remained higher than expected: `build_multibranch_readout_tensors` re-synthesized the full `(50304, 512)` detector table `w_det_synth` from `(u_hi, u_lo, schur, walsh)` on **every one of the `1,280` validation batches** (`1,280 × 50,304 × 512` operations per validation checkpoint) instead of caching it once per evaluation or eliminating the `50,304` detector synthesis via Step 1.
   - More fundamentally, across Waves 58–62 the model accumulated too many overlapping subsystems (`448D/64D` split + `16D` carrier + `OnlineCausalZipfTracker` + `ZeroPPMIBytePairNGramProjector` + `BytePairSchurWavepacketSynthesizer` + `Two8QubitRegisterSynthesizer` + 4-group optimizer split). See [`docs/SESSION_WRAPUP_VACUUM_CHOLESKY_AND_ZERO_PPMI_BYTE_PAIR_CONSOLIDATION.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SESSION_WRAPUP_VACUUM_CHOLESKY_AND_ZERO_PPMI_BYTE_PAIR_CONSOLIDATION.md) for the complete consolidation analysis and next-step simplification plan.

