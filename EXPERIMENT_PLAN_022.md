# Experiment Plan 022: 64-Qubit Sparse Fock Excitation Space ($\mathcal{H}_{64}^{(\le 2)}$), Sparse Hashed 1/2/3-Gram Qubit Encodings, & 4M-Parameter Internal Density Matrix Layers

---

## 1. Project Context, Problem Statement, & Motivation

### 1.1 Project Overview & Evolution Across Waves 1–13
In **QuantumDLBlocks**, we are developing quantum-geometric sequence transducers (**`QuantumBornGPT`**) that evolve unit-norm pure states $|\psi\rangle \in S^{d-1}$ or positive semi-definite (PSD) unit-trace density matrices $\rho \in \mathcal{D}(\mathcal{H})$ through isometric Stiefel projections, unitary rotary position embeddings (`UnitaryRoPE`), coherence-thresholded purified Born-rule self-attention (`flash_purified_born_attention`, $A_{t,s} \propto |\langle q_t | k_s \rangle|^2$), overcomplete two-beam interferometer frames, and projective Positive Operator-Valued Measure (`ProjectivePOVMHead`) readouts $P(w \mid \psi) \propto \beta |\langle M_w | \psi \rangle|^2$. We benchmark all models on the `100M`-token (`1`-shard) and `900M`-token (`9`-shard) **FineWeb-Edu** language modeling benchmark against a strictly held-out `4,999,488`-token validation shard at fixed sequence length `T_val = 512` alongside an exact reproduction of the **`KellerJordan/modded-nanogpt`** leaderboard architecture (**`ClassicalModdedGPT`**).

Across Waves 1–13, our 900M-token FineWeb-Edu validation cross-entropy (`Val CE`, lower is better) progressed as follows:
- **Waves 1–6 (Readout Rank Uncapping & Overcomplete Frames)**: `4.3821 -> 3.8514` Val CE (`QuantumBornGPT`).
- **Waves 7–10 (Early-Epoch Decoupling, Layer-0 $V_0$ Residual, & GPU-Staged Shards)**: `3.8514 -> 3.6926` Val CE (`QuantumBornGPT`) and `3.4280` Val CE (`ClassicalModdedGPT`).
- **Waves 11–12 (`modded-nanogpt` Leaderboard Parity & Step-Budget Scaling)**:
  - `ClassicalModdedGPT` reached **`3.3362` Val CE** (`28.1` PPL, `w12_900m_classical_leaderboard_exact_deep_L16_b262k`).
  - `QuantumBornGPT` reached **`3.6621` Val CE** (`39.0` PPL, `w12_900m_qborn_speedrun_v5_deep_L12_H8_d128_lr035`).
- **Wave 13 (Plan 021: 9-Shard `900M`-Token PPMI-SVD Co-Occurrence Spectral Initialization + Hashed Bigram+Trigram Embeddings + `ANVIL` Categories A/B/C)**:
  - Initializing `dm_seed` and `povm_head.effects` with a 9-shard (`900M`-token) Shifted Positive Pointwise Mutual Information (SPPMI) + Randomized SVD spectral sketch (`5.05s` on GPU) and adding a `262,144`-row hashed bigram+trigram table (`HashedNgramEmbedding`) with `ANVIL`/`PolarExpress` + `NorMuon` + `TwinRail` + `TailWeightAverager` + `MUDDDynamicSkip` produced a **`-0.1500` nats (`-5.5` PPL)** breakthrough:
    - `w13_900m_qborn_v6_9shard_ppmi_svd_deep_L12_H8_d128` (`L=12`): **`3.5121` Val CE** (`33.5` PPL, `37.35%` Top-1, `58.91%` Top-5 — **All-Time `QuantumBornGPT` Record**).
    - `w13_900m_qborn_v6_9shard_ppmi_svd_ngram_anvil_L8` (`L=8` with N-Gram): **`3.6182` Val CE** (`37.3` PPL, `36.51%` Top-1).
    - `w13_900m_qborn_v6_9shard_ppmi_svd_ablation_no_ngram` (`L=8` without N-Gram): **`3.6686` Val CE** (`39.2` PPL, `35.82%` Top-1).
    - `w13_900m_classical_anvil2_full_catABC_b131k` (`L=12` Classical): **`3.3583` Val CE** (`28.7` PPL, `38.66%` Top-1).

### 1.2 The Next Frontier: Scaling from 10 Dense Qubits ($d=1024$) to **64 Qubits in a Sparse Excitation Space** with a ~4M-Parameter Internal Density Matrix Layer
Despite Wave 13's success, our current `QuantumBornGPT` architecture has two structural limitations:
1. **Only 10 Effective Qubits ($d = 2^{10} = 1024$) vs. Massive Dense Lookup Tables (`320M` Embedding Parameters)**:
   - A dense state vector $|\psi\rangle \in \mathbb{R}^{1024}$ corresponds to a full tensor-product register of only **$N_q = 10$ qubits** ($2^{10} = 1024$).
   - Meanwhile, to inject unigram, bigram, and trigram context into that 10-qubit register, Wave 13 allocated a **dense** `50,304 × 1024` unigram table (`51.5M` parameters) plus a **dense** `262,144 × 1024` bigram+trigram table (`268.4M` parameters) — **320 million parameters** in dense lookup tables, even though each individual token, 2-gram, or 3-gram is a discrete, sparse linguistic event!
2. **How a 64-Qubit Register in a Sparse Excitation (Fock) Subspace Achieves Massive Expressivity with ~4 Million Parameters per Density Matrix Layer**:
   - A full 64-qubit Hilbert space $(\mathbb{C}^2)^{\otimes 64}$ has dimension $2^{64} \approx 1.84 \times 10^{19}$, which cannot be represented densely.
   - However, in quantum many-body physics and quantum optics, weakly-excited multi-qubit systems live in the **sparse $k$-excitation Fock subspace** $\mathcal{H}_{64}^{(\le 2)}$ spanned by computational basis states $|b_0 b_1 \dots b_{63}\rangle$ of **Hamming weight $\le 2$** (at most 2 simultaneously excited qubits):
     - **0-excitation vacuum $|0\rangle^{\otimes 64}$**: $\binom{64}{0} = 1$ state.
     - **1-qubit single excitations $|1_i\rangle$ ($0 \le i < 64$)**: $\binom{64}{1} = \mathbf{64}$ states.
     - **2-qubit pair excitations $|1_i, 1_j\rangle$ ($0 \le i < j < 64$)**: $\binom{64}{2} = \frac{64 \times 63}{2} = \mathbf{2,016}$ states.
     - **Total basis dimension of the 64-qubit $\le 2$-excitation space**:
       $$D_{\text{Fock}\le 2}(64) = 1 + 64 + 2,016 = \mathbf{2,081} \quad (\text{or } D = \mathbf{2,048} = 64 \text{ single-qubit modes} + 1,984 \text{ two-qubit pair modes})$$
   - Look at the parameter count of a **density matrix / Hermitian operator / unitary transition channel** $W_{\rho} \in \mathbb{R}^{D \times D}$ on this sparse 64-qubit space:
     - On the $\binom{64}{2} = 2,016$ two-qubit pair space: $2,016 \times 2,016 = \mathbf{4,064,256}$ parameters (**4.06 million parameters**)!
     - On the hardware-aligned $D = 2,048$ space ($64$ single-qubit + $1,984$ pair modes): $2,048 \times 2,048 = \mathbf{4,194,304}$ parameters (**4.19 million parameters**)!
     - On the full $1 + 64 + 2,016 = 2,081$ space: $2,081 \times 2,081 = \mathbf{4,330,561}$ parameters (**4.33 million parameters**)!
   - Even better, look at how **unigrams (1-grams)**, **bigrams (2-grams)**, and **trigrams (3-grams)** map physically into this 64-qubit sparse excitation space:
     1. **Unigrams ($x_t$) $\to$ 64 Single-Qubit Amplitudes + Sparse Pair Excitations**:
        - Each vocabulary token $w \in \{0, \dots, V-1\}$ stores a **64-qubit single-excitation amplitude vector** $u(w) \in \mathbb{R}^{64}$ (requiring only $50,304 \times 64 = \mathbf{3.22\text{M}}$ parameters instead of $51.5\text{M}$!) plus $K_{\text{uni}}$ sparse hashed pair indices in Sector 2 (`64:2048`).
     2. **Bigrams $(x_{t-1}, x_t)$ $\to$ Sparse 2-Qubit Pair Excitations + Direct Two-Qubit Outer Product $u(x_{t-1}) \otimes u(x_t)$**:
        - Physically, a bigram $(x_{t-1}, x_t)$ is a **two-body correlation** between the qubits excited at $t-1$ and the qubits excited at $t$:
          $$\Psi_{\text{pair}}^{(2)}(i, j) = u_i(x_{t-1}) u_j(x_t) + u_j(x_{t-1}) u_i(x_t) \quad (0 \le i < j < 64)$$
        - Simultaneously, a **Sparse Hashed N-Gram Qubit Table** maps each bigram $(x_{t-1}, x_t)$ to **only $K_{\text{sparse}} = 16$ active pair excitations** out of the $2,048$ basis states (using a compact $262,144 \times 16 = \mathbf{4.19\text{M}}$-parameter amplitude table with deterministic $\pm 1$ Weinberger signs, reducing n-gram table memory by **64x** compared to a dense $262,144 \times 1024$ table!).
     3. **Trigrams $(x_{t-2}, x_{t-1}, x_t)$ $\to$ 3-Qubit Partial-Trace Reduced Density State + Sparse Hashed Excitations**:
        - In a 64-qubit system, a trigram $(x_{t-2}, x_{t-1}, x_t)$ excites a 3-qubit state across $u(x_{t-2}), u(x_{t-1}), u(x_t) \in \mathbb{R}^{64}$. Tracing out any one of the 3 qubits projects the 3-qubit state onto the **three constituent 2-qubit pairs** in $\binom{64}{2}$:
          $$\Psi_{\text{pair}}^{(3)}(i, j) = u_i(x_{t-2}) u_j(x_t) + u_i(x_{t-2}) u_j(x_{t-1}) + u_i(x_{t-1}) u_j(x_t)$$
          augmented by $K_{\text{sparse}} = 16$ sparse hashed trigram excitations scattered into the $2,048$-D Fock vector!
     4. **Internal 64-Qubit Density Matrix Layer (`Sparse64QubitDensityLayer` — $4.19\text{M}$ Parameters) & Rank-$R$ Mixed Density Matrix Evolution**:
        - Inside each block (or as a dedicated 64-qubit internal layer), the state evolves via:
          - **Two-Body Qubit Entangler ($64 \leftrightarrow 1,984$ Fock Coupler)**: Couples the $64$ single-qubit amplitudes $q \in \mathbb{R}^{64}$ and the $1,984$ two-qubit pair amplitudes $P_{ij}$ via symmetric two-body creation ($q_i \tilde{q}_j \to P_{ij}$) and annihilation/partial-trace ($\sum_j P_{ij} q_j \to q_i$), followed by
          - **The $2,048 \times 2,048$ ($4,194,304$-parameter) Internal Density Matrix / Isometric Channel Operator $W_{\rho} \in \mathbb{R}^{2048 \times 2048}$**, which mixes all 64 single-qubit and 1,984 pair-excitation modes, and
          - **Rank-$R$ Mixed-State Density Matrix Representation ($\rho_t = \sum_{r=1}^R p_{t,r} |\psi_{t,r}\rangle\langle\psi_{t,r}|$)**: Represents each token's internal quantum state as a genuine rank-$R$ PSD unit-trace density matrix ($R \in \{1, 4\}$) whose readout probabilities follow the exact Born trace rule $P(w \mid \rho_t) \propto \text{Tr}(\rho_t M_w) = \sum_{r=1}^R p_{t,r} |\langle M_w | \psi_{t,r} \rangle|^2$.

---

## 2. Observable Hypotheses & Quantitative Predictions

### Hypothesis `H1`: Parameter Efficiency & Expressivity of Sparse Hashed 1/2/3-Gram 64-Qubit Fock Encoding (`Sparse64QubitNgramEmbedder`)
- **Claim**: Replacing dense $1024$-D unigram + n-gram tables ($320\text{M}$ total embedding parameters) with `Sparse64QubitNgramEmbedder` ($N_q = 64$ qubits, $D_{\text{Fock}} = 2048$ with $64$ single-qubit modes + $1984$ pair modes, $50,304 \times 64 = 3.22\text{M}$ unigram qubit amplitudes, and $K_{\text{sparse}} = 16$ active excitations per 2/3-gram bucket $= 4.19\text{M}$ parameters) reduces total embedding + n-gram table parameters by **`> 15x`** while achieving:
  1. Exact zero-collision sign cancellation in expectation ($\mathbb{E}[\langle h(a) | h(b) \rangle] \approx 0$ for distinct n-grams $a \neq b$, empirical mean $|\cos| < 0.08$),
  2. When initialized with a 64-qubit PPMI-SVD co-occurrence sketch, a Step-0 collocation coherence ratio $\ge 3.0\times$ over random pairs in the $2048$-D 64-qubit Fock space, and
  3. Higher Logit Matrix Effective Rank $R_\sigma(L)$ and lower validation cross-entropy (`Val CE`) after 150 FineWeb-Edu training steps than an iso-parameter dense baseline.

### Hypothesis `H2`: Expressivity of the $4.19\text{M}$-Parameter ($2048 \times 2048$) 64-Qubit Internal Density Matrix Layer (`Sparse64QubitDensityLayer`) & Two-Body Fock Coupler
- **Claim**: Adding the 64-qubit internal density-matrix layer (`Sparse64QubitDensityLayer`, containing the $2048 \times 2048 = 4,194,304$-parameter operator $W_\rho$ plus the explicit $64 \leftrightarrow 1984$ two-body qubit pair coupler) and **Rank-$R=4$ Mixed Density Matrix evolution** ($\rho_t = \sum_{r=1}^4 p_{t,r} |\psi_{t,r}\rangle\langle\psi_{t,r}|$) strictly improves FineWeb-Edu validation cross-entropy (`Val CE`) and Top-1 accuracy over:
  1. The 10-qubit ($d=1024$) baseline `QuantumBornGPT` at equal training steps, and
  2. An ablation without the 64-qubit two-body Fock coupler / mixed-state density matrix rank ($R=1$).
- **Quantitative Pass Criteria**:
  - `Sparse64QubitDensityLayer` primary operator parameter count is within `[4,000,000, 4,400,000]` (`4,194,304` exact for $2048 \times 2048$).
  - On a controlled multi-hundred-step FineWeb-Edu GPU benchmark on our local RTX 5060 Ti (`T_val = 512`), the 64-qubit sparse Fock + $4.19\text{M}$ density-matrix model achieves at least **`0.10` nats lower `Val CE`** than the 10-qubit ($d=1024$) dense baseline.

### Hypothesis `H3`: Purity Dynamics ($\text{Tr}(\rho_t^2)$) & Sparsity Scaling ($K_{\text{sparse}} \in \{8, 16, 32\}$)
- **Claim**: In the Rank-$R=4$ 64-qubit density-matrix model, the learnable branch mixture weights $p_{t,r}$ (`Tr(rho_t) = 1`, purity $\mathcal{P}_t = \sum_{r=1}^R p_{t,r}^2 \in [1/R, 1]$) adaptively modulate state purity across token positions (with non-degenerate purity $0.28 \le \overline{\mathcal{P}} \le 0.95$, confirming genuine mixed-state density-matrix utilization rather than rank-1 collapse), and $K_{\text{sparse}} = 16$ achieves within `0.02` nats of $K_{\text{sparse}} = 32$ at half the n-gram parameter budget (`4.19M` vs `8.39M` parameters).

---

## 3. Planned Architectural & Code Changes

### 3.1 Component 1: `src/qdlblocks/layers/sparse_qubit_fock.py` (New Reusable Module)
Create [`src/qdlblocks/layers/sparse_qubit_fock.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/sparse_qubit_fock.py) and export all public classes/functions in `src/qdlblocks/layers/__init__.py` and `src/qdlblocks/__init__.py`:

1. **`build_64qubit_pair_index_tables(num_qubits: int = 64, max_pair_modes: int = 1984) -> tuple[torch.Tensor, torch.Tensor]`**:
   - Constructs deterministic buffer tables `pair_i: (max_pair_modes,)` and `pair_j: (max_pair_modes,)` (`0 <= i < j < num_qubits`) mapping the $\binom{64}{2} = 2016$ two-qubit pairs (truncated or wrapped to `max_pair_modes = fock_dim - num_qubits = 1984` when `fock_dim = 2048`, or full `2016` when `fock_dim = 2080`) so single-qubit amplitudes $q \in \mathbb{R}^{B \times T \times 64}$ can be vectorized-gathered into two-qubit pair amplitudes $q_i \cdot \tilde{q}_j \in \mathbb{R}^{B \times T \times \text{max\_pair\_modes}}$ in $O(B T D_{\text{Fock}})$ without materializing a $B \times T \times 64 \times 64$ tensor.

2. **`Sparse64QubitNgramEmbedder(nn.Module)`**:
   - Encodes token IDs `idx: (B, T)` directly into a unit-normalized 64-qubit sparse Fock state $|\psi_0\rangle \in S^{D_{\text{Fock}}-1}$ (`fock_dim: int = 2048`, `num_qubits: int = 64`, `sparse_k: int = 16`, `ngram_vocab_size: int = 262144`):
     - **64-Qubit Unigram Table (`qubit_wte: nn.Embedding(vocab_size, num_qubits)`)**: Only $50,304 \times 64 = 3,219,456$ (`3.22M`) parameters.
     - **Sparse Unigram Pair Hash (`unigram_sparse_vals: nn.Embedding(vocab_size, sparse_k)`)**: Scatters $K = 16$ signed amplitudes per token into deterministic hashed coordinates of the pair sector (`64:2048`), giving each token unique high-dimensional pair-excitation coordinates without a $50,304 \times 2048$ table.
     - **Sparse Bigram + Trigram Qubit Excitation Table (`ngram_sparse_vals: nn.Embedding(ngram_vocab_size, sparse_k)`)**:
       - Stores only $K = 16$ floats per hash bucket ($262,144 \times 16 = \mathbf{4,194,304}$ parameters).
       - For each position $(b, t)$, hashes `bigram_ids[b, t]` and `trigram_ids[b, t]` to $K = 16$ deterministic target indices in `[0, fock_dim)` via multiplicative-XOR primes `(offset_k * PRIME_K) & (fock_dim - 1)` and $\pm 1$ Weinberger signs, accumulating into the $2048$-D Fock state via `scatter_add_`.
     - **Explicit 2-Qubit & 3-Qubit Partial-Trace Pair Excitation**:
       - From $u_t = \text{qubit\_wte}(x_t) \in \mathbb{R}^{B \times T \times 64}$, computes shifted histories $u_{t-1}$ and $u_{t-2}$ and injects the symmetric two-qubit and three-qubit partial-trace correlation:
         $$P_{t, m} = u_{t, i_m} \big(\gamma_2 u_{t-1, j_m} + \gamma_3 u_{t-2, j_m}\big) + u_{t-1, i_m} \big(\gamma_2 u_{t, j_m} + \gamma_3 u_{t-2, j_m}\big)$$
         directly into Sector 2 (`64:2048`)!

3. **`Sparse64QubitDensityLayer(nn.Module)`**:
   - Implements the **64-qubit internal density-matrix layer** operating on $\mathcal{H}_{64}^{(\le 2)}$ (`fock_dim: int = 2048`, `num_qubits: int = 64`, `density_rank: int = 4`, `use_fock_coupler: bool = True`):
     - **Core $4.19\text{M}$-Parameter Density Matrix / Channel Operator (`density_operator: nn.Linear(fock_dim, fock_dim, bias=False)`)**:
       - Shape `(2048, 2048)` = **exact `4,194,304` (`4.19M`) parameters**!
       - Initialized on the Stiefel manifold $O(2048)$ (and trained with `StiefelMuon` + `PolarExpress`), with a learnable Hermitian PSD spectral filter `eig_weights: nn.Parameter(torch.ones(fock_dim))` so $W_\rho = U \, \text{diag}(\sigma(w)) \, U^\top$ (or asymmetric channel $U \, \text{diag}(w)$) acts as a trace-preserving quantum channel on the 64-qubit Fock state.
     - **Two-Body 64-Qubit Creation / Annihilation Coupler (`use_fock_coupler=True`)**:
       - Extracts Sector 1 single-qubit amplitudes $q \in \mathbb{R}^{B \times T \times 64}$ and Sector 2 pair amplitudes $P \in \mathbb{R}^{B \times T \times 1984}$.
       - Applies a compact $64 \times 64$ single-qubit unitary $U_1 \in \mathbb{R}^{64 \times 64}$ (`4,096` params) to form $\tilde{q} = U_1 q$, excites two-qubit pairs via $\Delta P_m = \frac{1}{\sqrt{64}}(q_{i_m} \tilde{q}_{j_m} + q_{j_m} \tilde{q}_{i_m})$, and feeds pair correlations back into single qubits via partial trace $\Delta q_{i} = \sum_{m: i_m=i} P_m \tilde{q}_{j_m}$.
     - **Rank-$R$ Mixed Density Matrix Branch Expansion & Purity Gate (`density_rank: int = 4`)**:
       - Produces $R$ pure eigen-branches $|\psi_{t,1}\rangle, \dots, |\psi_{t,R}\rangle \in S^{D-1}$ and context-dependent mixture probabilities $p_{t,r} = \text{Softmax}(G_\rho(\psi_t))_r$ ($\sum_{r=1}^R p_{t,r} = 1$), defining the rank-$R$ density matrix:
         $$\rho_t = \sum_{r=1}^R p_{t,r} |\psi_{t,r}\rangle\langle\psi_{t,r}|, \qquad \text{Purity}(\rho_t) = \text{Tr}(\rho_t^2) = \sum_{r=1}^R p_{t,r}^2 \in [1/R, 1]$$
       - Exposes `last_purity_mean: float` and `last_pair_sector_energy: float` for live telemetry!

4. **`Sparse64QubitPOVMReadout(nn.Module)`**:
   - Parameter-efficient POVM readout from the 64-qubit Fock state $\rho_t = \sum_{r=1}^R p_{t,r} |\psi_{t,r}\rangle\langle\psi_{t,r}|$ (`fock_dim = 2048`) to `vocab_size = 50,304` logits:
     - Supports both:
       1. **Factorized 64-Qubit Detector Readout**: Measures the $R$ branches via $R$ orthogonal $64$-qubit sub-projections $P_r: \mathbb{R}^{2048} \to \mathbb{R}^{d_{\text{det}}}$ (`d_det = 256` or `512`) against detector effects $M_w \in S^{d_{\text{det}}-1}$ via the exact **Born density-matrix trace rule**:
          $$\text{logit}_w(\rho_t) = \beta_t \Big(\text{Tr}(\rho_t M_w) - \tau_{\text{floor}}\Big) + \alpha_{\text{prior}} \log \pi_w = \beta_t \sum_{r=1}^R p_{t,r} \langle M_{w,r} | \psi_{t,r} \rangle$$
       2. **Full-Fock Detector Readout** (`d_det = fock_dim`).

5. **`initialize_sparse_64qubit_with_ppmi_svd`**:
   - Initializes `Sparse64QubitNgramEmbedder.qubit_wte` (`50,304 × 64`) and detector effects using the top-64 (and top-$d_{\text{det}}$) singular vectors of the Shifted PPMI co-occurrence matrix across FineWeb shards, with even/odd coordinate orthogonal decoupling so Step-0 self-repetition is strictly zero ($\max_w |\langle M_w | \psi_0(w) \rangle| < 10^{-6}$).

### 3.2 Component 2: Integration into `QuantumBornGPT`, `model_card.py`, `inspect_model.py`, & `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`
- Update [`src/qdlblocks/models/born_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/born_gpt.py) to accept:
  - `use_sparse_64qubit: bool = False`
  - `num_qubits: int = 64`
  - `sparse_ngram_k: int = 16`
  - `use_sparse_density_layer: bool = False`
  - `density_matrix_rank: int = 4`
  - `use_fock_coupler: bool = True`
  When `use_sparse_64qubit=True`, `QuantumBornGPT` uses `Sparse64QubitNgramEmbedder` for input token + sparse 2/3-gram encoding in $\mathcal{H}_{64}^{(\le 2)}$ ($D = 2048$ or `embed_dim`), inserts `Sparse64QubitDensityLayer` (`4.19M` params for $D=2048$) at the middle of the stack (and/or per-block Fock coupling), and evaluates multi-branch density-matrix trace readout when `density_matrix_rank > 1`.
- Update [`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py), [`src/qdlblocks/benchmarks/model_card.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/model_card.py), and [`scripts/inspect_model.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py) to support CLI flags `--sparse-64qubit`, `--num-qubits 64`, `--sparse-ngram-k 16`, `--sparse-density-layer`, `--density-matrix-rank 4`, and `--no-fock-coupler`.

---

## 4. Verification & Local GPU Experimental Suite

1. **Unit Test Suite (`tests/test_hypothesis_022_sparse_64qubit_density_matrix.py`)**:
   - `test_64qubit_fock_pair_tables_and_sparse_ngram_embedder`: Verifies `build_64qubit_pair_index_tables`, `Sparse64QubitNgramEmbedder` shapes `(B, T, 2048)`, unit norm $\|\psi_0\|_2 = 1$, parameter budget (`qubit_wte` $= 3.22\text{M}$, `ngram_sparse_vals` $= 4.19\text{M}$ at $K=16$), and Weinberger sign orthogonality.
   - `test_sparse_64qubit_density_layer_4m_param_budget_and_purity`: Verifies `Sparse64QubitDensityLayer(fock_dim=2048, num_qubits=64, density_rank=4)` has `density_operator` with exact `2048 * 2048 = 4,194,304` parameters, preserves unit norms on all $R=4$ eigen-branches, produces valid mixture weights $\sum_r p_{t,r} = 1$ with purity $\text{Tr}(\rho_t^2) \in [0.25, 1.0]$, and backpropagates finite gradients to all parameters.
   - `test_sparse_64qubit_ppmi_svd_init_and_zero_self_overlap`: Verifies `initialize_sparse_64qubit_with_ppmi_svd` achieves $\ge 3.0\times$ Step-0 collocation similarity ratio in the $2048$-D 64-qubit Fock space while maintaining $< 10^{-6}$ Step-0 input-output self-overlap.
   - `test_quantum_born_gpt_sparse_64qubit_integration_and_model_card`: Verifies end-to-end forward/backward of `QuantumBornGPT(embed_dim=2048, use_sparse_64qubit=True, use_sparse_density_layer=True, density_matrix_rank=4)` and Model Card CLI flag parsing.

2. **Local GPU Experimental Script (`examples/32_verify_plan022_sparse_64qubit_experiments.py`)**:
   - Runs on our local **NVIDIA GeForce RTX 5060 Ti (16GB)** using real FineWeb-Edu tokens (`.cache/fineweb_100m/train_100M.bin` and held-out `val_5M.bin` at `T_val = 512`), testing `H1`, `H2`, and `H3` across **5 controlled multi-hundred-step local GPU experiments** (`Wave 14 / Local 64-Qubit Study`) and ingesting all telemetry and Model Cards into [`.cache/benchmarks.db`](file:///home/festivity/agy/QuantumDLBlocks/.cache/benchmarks.db) and the live WebUI (`http://127.0.0.1:8765`):
     1. `w14_local_qborn_10qubit_d1024_baseline`: Standard 10-qubit ($d=1024$) dense `QuantumBornGPT` control.
     2. `w14_local_64qubit_sparse_fock2048_4m_density_rank1`: 64-Qubit Sparse Fock ($D=2048$) + Sparse Hashed 1/2/3-Gram ($K=16$) + $4.19\text{M}$-param ($2048 \times 2048$) Internal Density Matrix Layer (`Rank-1` pure state).
     3. `w14_local_64qubit_sparse_fock2048_4m_density_rank4_mixed`: 64-Qubit Sparse Fock ($D=2048$) + Sparse Hashed 1/2/3-Gram ($K=16$) + $4.19\text{M}$-param Internal Density Matrix Layer + **Rank-4 Mixed Density Matrix ($\rho_t = \sum_{r=1}^4 p_{t,r} |\psi_{t,r}\rangle\langle\psi_{t,r}|$)** + Two-Body Fock Coupler.
     4. `w14_local_64qubit_sparse_fock2048_no_coupler_ablation`: Ablation disabling the $64 \leftrightarrow 1984$ two-body Fock coupler (`use_fock_coupler=False`) to isolate the contribution of explicit 1-qubit $\leftrightarrow$ 2-qubit entanglement coupling.
     5. `w14_local_64qubit_sparse_fock2048_k32_vs_k8_ablation`: Sparsity scaling comparison ($K_{\text{sparse}} = 8$ vs $16$ vs $32$).

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Gate)

1. **Existing Scripts Used**:
   - `scripts/inspect_model.py`: Extended with `--sparse-64qubit`, `--sparse-density-layer`, and `--density-matrix-rank` to inspect exact parameter breakdowns (`qubit_wte`, `ngram_sparse_vals`, `density_operator` $= 4,194,304$ params) and run 2-step CUDA smoke tests.
   - `scripts/monitor_colab_runs.py`: Ingests all `w14_local_*` and `verify022_*` runs into `.cache/benchmarks.db` and serves them at `http://127.0.0.1:8765`.
2. **`Build-Before-Launch` Scripts**:
   - Update `scripts/inspect_model.py` and `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` with the new 64-qubit sparse Fock CLI flags.
3. **`Nice-to-Have` Scripts (Logged in `scripts/BACKLOG.md`)**:
   - `scripts/qubit_fock_spectrum_probe.py`: A diagnostic CLI tool that loads a trained `Sparse64QubitDensityLayer` checkpoint and plots the $64 \times 64$ two-qubit pair-excitation correlation heatmap $C_{ij} = \mathbb{E}[|\Psi_{\text{pair}}(i, j)|^2]$ and the rank-$R$ density-matrix purity histogram $\text{Tr}(\rho_t^2)$ across FineWeb-Edu validation documents.
