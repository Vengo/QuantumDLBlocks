# Experiment Plan 023 — 16-Qubit Multi-Register Unigram Amplitude Table (`50,304 × 16 × 32`) & 256-State Interacting Quantum Hierarchy (`256 × 16 × 32` + `16 × 256 × 256` Unitary Coupling)

---

## 1. Standalone Project Context, Diagnosis of Wave 14, & Motivation for Plan 023

### 1.1 What `QuantumDLBlocks` Is Building and Where the Leaderboard Stands
**QuantumDLBlocks** is a dual-backend (`PyTorch + Triton` on GPU and `JAX + Pallas` on TPU) deep-learning library benchmarking **Quantum-Geometric Language Models** (`QuantumBornGPT`) head-to-head against **`ClassicalModdedGPT`** (our exact single-GPU port of the `KellerJordan/modded-nanogpt` speedrun architecture) on the **FineWeb-Edu `900M`-token (`9 × 100M` shard) streaming benchmark** evaluated on a held-out `5M`-token validation shard at fixed context length $T_{\text{val}} = 512$.

Across Waves 1–13:
- **`ClassicalModdedGPT` All-Time Record**: `w12_900m_classical_leaderboard_exact_3ve_skip7_b131k` achieved **`3.3095` Val CE** (`27.4` PPL, `39.21%` Top-1, `132.4m`), and in Wave 13 `w13_900m_classical_anvil2_full_catABC_b131k` achieved **`3.3583` Val CE**.
- **`QuantumBornGPT` All-Time Record (Wave 13)**: `w13_900m_qborn_v6_9shard_ppmi_svd_deep_L12_H8_d128` achieved **`3.5121` Val CE** (`33.5` PPL, `37.35%` Top-1, `58.91%` Top-5, `192.6m`, **`-0.1500` nats** ahead of Wave 12's `3.6621`), and the 8-layer `w13_900m_qborn_v6_9shard_ppmi_svd_ngram_anvil_L8` achieved **`3.6182` Val CE** (`37.3` PPL, `163.7m`, **`-0.1262` nats** ahead of Wave 12's `3.7444`).
  - At the **`100M`-token (Shard 1) milestone**, Wave 13's `L=12` leader reached **`3.9531` Val CE** (and `4.1973` at `50M` tokens), while Wave 13's `L=8` model reached **`4.0612` Val CE** at `100M` tokens.

### 1.2 Quantitative Post-Mortem of Wave 14 at `50M–100M` Tokens
In Wave 14 (Plan 022), we tested a `64`-qubit sparse Fock excitation space ($\mathcal{H}_{64}^{(\le 2)} \cong \mathbb{R}^{2048}$) where:
1. Each of the `50,304` vocabulary tokens was compressed down to a **single `64`-dimensional unigram vector** (`qubit_wte: 50,304 × 64 = 3.22M` params) plus `K=16` scalar pair excitations (`unigram_sparse_vals: 50,304 × 16 = 0.80M` params), and
2. The `4d-TwoBeam` bilinear interference MLP (`OvercompleteTwoBeamInterference`) inside each block was replaced by a single `2048 × 2048` (`4.19M`-parameter) linear density operator (`Sparse64QubitDensityLayer`).

At the `50M–100M` token mark on our 5 Colab GPUs, the Wave 14 runs recorded:
- `w14_900m_qborn_64qubit_fock2048_4m_density_rank4_L8`: **`4.4410` Val CE** at `100M` tokens (`41,821 tok/s`, $\beta = 58.63$ saturated near the `60.0` clamp),
- `w14_900m_qborn_64qubit_fock2048_4m_density_rank1_ablation_L8`: **`4.4341` Val CE** at `100M` tokens (`46,797 tok/s`),
- `w14_900m_qborn_v6_hybrid_64qubit_density_coupler_L12_d1024`: **`4.4078` Val CE** at `75M` tokens (`36,437 tok/s`),
- `w14_900m_qborn_64qubit_fock2048_4m_density_rank4_deep_L12`: **`4.4949` Val CE** at `75M` tokens (`31,183 tok/s`),
- `w14_900m_qborn_64qubit_fock2048_k32_symdual64_L8`: **`4.8004` Val CE** at `50M` tokens (`28,558 tok/s`).

**Why Wave 14 lagged Wave 13 by `+0.38` to `+0.48` nats at `100M` tokens**:
1. **Unigram Capacity Bottleneck (`64` floats/token vs `512` floats/token)**: Compressing `50,304` vocabulary tokens into a single `64`-D vector (`3.22M` params) starved individual tokens of lexical specificity, forcing the POVM inverse temperature $\beta$ to saturate to `58.6` trying to pull nearly-collinear token states apart.
2. **Loss of Bilinear Two-Beam Interference (`OvercompleteTwoBeamInterference`)**: Replacing the `4d-TwoBeam` interference MLP ($\psi_{\text{mlp}} \propto W_{\text{down}}(\phi_1 \odot \phi_2)$) with a single $2048 \times 2048$ linear operator removed the multiplicative two-beam interference that gave `QuantumBornGPT` its largest expressivity jump in Waves 6–13.
3. **Lack of Hierarchical Concept Mediation**: In natural language, `50,304` surface tokens organize into a hierarchy of higher-level semantic/syntactic concepts. When tokens only interact through flat pairwise attention or hash buckets, rare or compositional tokens cannot share statistical strength through higher-level quantum concept states.

All 5 Wave 14 Colab VMs have now been stopped (`0.00 CU/hr` burn rate) so we can implement and locally validate **Plan 023** on our local **NVIDIA GeForce RTX 5060 Ti (16GB)** before launching Wave 15.

---

## 2. Mathematical & Architectural Specification of the `50,304 × 16 × 32` Qubit Table & `256`-State Quantum Hierarchy

### 2.1 The `50,304 × 16 × 32` Multi-Qubit Unigram Amplitude Table
Instead of a flat vector or an over-compressed `64`-D vector, each vocabulary token $w \in \{0, \dots, V-1\}$ ($V = 50,304$) is parameterized by a **3D Qubit Amplitude Tensor**:
$$\mathcal{Q}_{\text{tok}} \in \mathbb{R}^{50,304 \times 16 \times 32}$$
- **Exact Parameter Count**:
  $$50,304 \times 16 \times 32 = 25,755,648 \text{ parameters } (\approx 25.76\text{M parameters})$$
- **Physical & Quantum-Geometric Interpretation**:
  - Each unigram $w$ is associated with **$Q = 16$ qubit sub-registers (or 16 5-qubit subsystems, since $2^5 = 32$)**, where each sub-register $q \in \{0, \dots, 15\}$ holds a normalized state vector on the 32-dimensional unit sphere $S^{31} \subset \mathbb{R}^{32}$:
    $$\psi_{w, q}^{\text{low}} = \frac{\mathcal{Q}_{\text{tok}}[w, q, :]}{\|\mathcal{Q}_{\text{tok}}[w, q, :]\|_2 + \epsilon} \in S^{31} \subset \mathbb{R}^{32}$$
  - Normalized across the 16 sub-registers by $1/\sqrt{16}$, each token lives on the **16-fold product of unit spheres**:
    $$\psi_w^{\text{low}} = \frac{1}{\sqrt{16}} \bigoplus_{q=0}^{15} \psi_{w, q}^{\text{low}} \in (S^{31})^{16} \subset S^{511} \subset \mathbb{R}^{512}$$
  - Notice how $16 \times 32 = 512$ is **exactly half of our $d = 1024$ Hilbert space dimension**! Under even/odd parity orthogonal decoupling (`--ortho-decouple`), the `512` even coordinates `0::2` of $\mathbb{R}^{1024}$ correspond 1-to-1 with the flattened `(16, 32)` input qubit registers, while the `512` odd coordinates `1::2` of $\mathbb{R}^{1024}$ correspond 1-to-1 with the output POVM measurement detectors!

### 2.2 The `256` Higher-Level Quantum States (`256 × 16 × 32`) & Their `16 × 256 × 256` Interaction Operator
To allow the `50,304` lower-level vocabulary tokens to interact with **`256` higher-level quantum states** that interact with each other and connect lower-level tokens in a hierarchy, we introduce:

1. **Higher-Level Quantum State Bank (`higher_states`)**:
   $$\Phi \in \mathbb{R}^{256 \times 16 \times 32} \quad (256 \times 16 \times 32 = 131,072 \text{ parameters})$$
   where each higher-level state $m \in \{0, \dots, 255\}$ has $Q = 16$ normalized sub-register states $\hat{\Phi}_{m, q} = \text{Normalize}_{S^{31}}(\Phi_{m, q}) \in S^{31} \subset \mathbb{R}^{32}$.

2. **Higher-Level State-to-State Interaction Operator (`higher_interaction` & `higher_sub_unitary`)**:
   - The `256` higher-level quantum states interact **with each other** through a per-register orthogonal/unitary $256 \times 256$ coupling operator:
     $$H_{\text{high}} \in \mathbb{R}^{16 \times 256 \times 256} \quad (16 \times 256 \times 256 = 1,048,576 \text{ parameters } \approx 1.05\text{M})$$
     and a per-register intra-state $32 \times 32$ unitary rotation:
     $$W_{\text{sub}} \in \mathbb{R}^{16 \times 32 \times 32} \quad (16 \times 32 \times 32 = 16,384 \text{ parameters})$$
   - Both $H_{\text{high}}[q] \in O(256)$ and $W_{\text{sub}}[q] \in O(32)$ are initialized on the orthogonal group (with near-identity + off-diagonal coupling so higher-level states immediately exchange amplitude while preserving distinct concept identities) and updated by batched `StiefelMuon`!
   - The **interacted higher-level quantum states** $\tilde{\Phi} \in \mathbb{R}^{256 \times 16 \times 32}$ are computed by evolving $\hat{\Phi}$ through $(H_{\text{high}}, W_{\text{sub}})$:
     $$\tilde{\Phi}_{m, q} = \text{Normalize}_{S^{31}}\!\left(\sum_{m'=0}^{255} H_{\text{high}}[q, m, m'] \, (\hat{\Phi}_{m', q} W_{\text{sub}}[q]^{\top})\right) \in S^{31} \subset \mathbb{R}^{32}$$

3. **Bottom-Up $\to$ Higher-Level Interaction $\to$ Top-Down Hierarchical Coupling**:
   Given any sequence of 16-qubit token states $\psi_{t, q} \in \mathbb{R}^{B \times T \times 16 \times 32}$ ($t \in \{0, \dots, T-1\}$, $q \in \{0, \dots, 15\}$):
   - **Step A — Bottom-Up Lower-to-Higher Born Resonance (`50,304 -> 256`)**:
     Each qubit register $q \in \{0, \dots, 15\}$ computes its inner-product overlap with the $256$ higher-level states $\hat{\Phi}_{:, q} \in \mathbb{R}^{256 \times 32}$:
     $$s_{t, q, m} = \gamma_q \langle \psi_{t, q} \mid \hat{\Phi}_{m, q} \rangle = \gamma_q \sum_{a=0}^{31} \psi_{t, q, a} \hat{\Phi}_{m, q, a} \in \mathbb{R}^{B \times T \times 16 \times 256}$$
     where $\gamma_q \in \mathbb{R}^{16}$ is a learnable per-register inverse temperature (initialized to $\sqrt{32} \approx 5.66$).
     The bottom-up excitation probabilities over the 256 higher-level states are given by purified Born resonance:
     $$w_{t, q, m} = \text{ReLU}(s_{t, q, m} - \tau_q)^2 + \epsilon_0 \text{ReLU}(s_{t, q, m}), \qquad p_{t, q, m} = \frac{w_{t, q, m}}{\sum_{m'=0}^{255} w_{t, q, m'} + \epsilon} \in \Delta^{255}$$
   - **Step B — Higher-Level State-to-State Propagation & Causal Sequence Memory (`256 <-> 256`)**:
     - First, the excited higher-level distribution $p_{t, q} \in \mathbb{R}^{B \times T \times 16 \times 256}$ propagates across the higher-level interaction graph $H_{\text{high}}[q] \in \mathbb{R}^{256 \times 256}$:
       $$c_{t, q} = \text{einsum}(\text{"btqm,qnm->btqn"}, p_{t, q}, H_{\text{high}}) \in \mathbb{R}^{B \times T \times 16 \times 256}$$
     - Second, to allow the 256 higher-level quantum states to **connect lower-level tokens across time $t - \Delta \to t$**, we apply a causal multi-horizon exponential moving superposition over the sequence in the 256-state space ($t-1, t-2, t-3, t-4$):
       $$\bar{c}_{t, q} = (1 - \alpha_{\text{mem}, q}) c_{t, q} + \alpha_{\text{mem}, q} \sum_{\Delta=1}^{4} \omega_{q, \Delta} \, c_{t - \Delta, q} \in \mathbb{R}^{B \times T \times 16 \times 256}$$
       where $c_{t-\Delta, q} = 0$ for $t < \Delta$ (strictly causal!), and $\omega_{q, \Delta} = \text{Softmax}_\Delta(\theta_{q, \Delta})$.
   - **Step C — Top-Down Higher-to-Lower Reconstruction (`256 -> 16 × 32`)**:
     The interacted higher-level states $\tilde{\Phi}_{m, q} \in \mathbb{R}^{256 \times 16 \times 32}$ are projected back down into the 16 qubit registers of token $t$:
     $$\psi_{t, q}^{\text{hier}} = \text{Normalize}_{S^{31}}\!\left(\sum_{m=0}^{255} \bar{c}_{t, q, m} \, \tilde{\Phi}_{m, q}\right) \in \mathbb{R}^{B \times T \times 16 \times 32}$$
     and combined coherently with $\psi_{t, q}$ on $(S^{31})^{16}$:
     $$\psi_{t, q}^{\text{out}} = \text{Normalize}_{S^{31}}\!\Big(\psi_{t, q} + g_{\text{hier}, q} \, \psi_{t, q}^{\text{hier}}\Big)$$

### 2.3 Where the 2-Level Quantum Hierarchy Operates in `QuantumBornGPT`
As selected by the user (**Full 2-Level Quantum Hierarchy preserving the fast `d=1024` `4d-TwoBeam` backbone**), the `50,304 × 16 × 32` + `256 × 16 × 32` hierarchy operates at three levels simultaneously:

1. **Level 1 — Hierarchical 16-Qubit Input Lift (`Hierarchical16QubitEmbedder`, $50,304 \times 16 \times 32 \leftrightarrow 256 \times 16 \times 32 \to d = 1024$)**:
   - Looks up $\psi_{t}^{\text{low}} = \mathcal{Q}_{\text{tok}}[x_t] \in \mathbb{R}^{B \times T \times 16 \times 32}$,
   - Passes $\psi_{t}^{\text{low}}$ through the `256`-state quantum hierarchy (`Step A -> Step B -> Step C`) to obtain $\psi_t^{\text{out}} \in \mathbb{R}^{B \times T \times 16 \times 32}$,
   - Flattens `(16, 32) -> 512` and lifts into $d = 1024$:
     - Under `--ortho-decouple`: places the `512`-D hierarchical 16-qubit state on the **even coordinates `0::2`** of $\psi_0(t) \in S^{1023}$ (while an orthogonal `512 × 512` Stiefel lift $W_{\text{even}} \in O(512)$ mixes across the 16 registers if enabled, initialized to $I_{512}$ so Step-0 PPMI-SVD geometry is preserved 100%!).
     - Also supports optional hashed 2/3-gram (`--ngram-hash`) injection on top of $\psi_0(t)$.

2. **Level 2 — Per-Block Hierarchical Quantum Bridge (`QuantumBlockHierarchyAdapter` inside `QuantumBornBlock`)**:
   - Inside each `QuantumBornBlock` (which retains `SubspaceProjectorBornAttention` and `OvercompleteTwoBeamInterference` (`4d-TwoBeam`) at $d = 1024$), after the block's attention + `4d-TwoBeam` MLP update:
   - Extracts the 16-qubit register view $\psi_{\text{reg}}^{(l)} \in \mathbb{R}^{B \times T \times 16 \times 32}$ from the $d = 1024$ state $\psi^{(l)}$ (combining even `0::2` and odd `1::2` parity channels via learnable per-layer parity weights $\beta_{\text{even}}^{(l)}, \beta_{\text{odd}}^{(l)}$),
   - Routes $\psi_{\text{reg}}^{(l)}$ through the **shared `256 × 16 × 32` Higher-Level Quantum Hierarchy** (modulated by per-layer register gates $g_{\text{layer}}^{(l)} \in \mathbb{R}^{16}$) to produce $\Delta \psi_{\text{hier}}^{(l)} \in \mathbb{R}^{B \times T \times 16 \times 32}$,
   - Embeds $\Delta \psi_{\text{hier}}^{(l)}$ back into $\mathbb{R}^{1024}$ (updating both even `0::2` and odd `1::2` coordinates so the hierarchy actively rotates input states on `0::2` into output POVM coordinates on `1::2`!) and normalizes on $S^{1023}$.

3. **Level 3 — Two-Level Coarse-to-Fine Hierarchical POVM Readout (`Hierarchical16QubitPOVMReadout`)**:
   - At the output readout, we want every vocabulary token $w \in \{0, \dots, 50,303\}$ to be predicted by **both** its fine-grained detector state AND its membership in the `256` higher-level quantum states, **without materializing an extra `(B, T, 50304)` tensor**:
   - We parameterize:
     1. A fine-grained odd-coordinate `16 × 32` detector table $\mathcal{M}_{\text{tok}} \in \mathbb{R}^{50,304 \times 16 \times 32}$ (`25.76M` params, or full `povm_head` in `asymmetric` mode), AND
     2. A vocabulary-to-hierarchy affinity matrix $A_{\text{vocab}\leftarrow\text{high}} \in \mathbb{R}^{50,304 \times 256}$ (`12.88M` params, initialized from the Step-0 inner product between $\mathcal{M}_{\text{tok}}$ and the 256 higher-level states $\tilde{\Phi}$!).
   - Once per step (before calling `direct_chunked_povm_ce`), we construct the **hierarchically-coupled POVM effect matrix**:
     $$\mathcal{M}_{\text{hier\_delta}} = \text{reshape}\!\Big(A_{\text{vocab}\leftarrow\text{high}} \cdot \text{reshape}(\tilde{\Phi}, [256, 512]),\, [50304, 512]\Big)$$
     and add $g_{\text{readout}} \, \mathcal{M}_{\text{hier\_delta}}$ into the odd coordinates `1::2` (and coupled `0::2` coordinates) of the POVM effect matrix $M_{\text{seed}} \in \mathbb{R}^{50,304 \times 1024}$!
   - **Why this is fast and expressive**:
     - Constructing $\mathcal{M}_{\text{hier\_delta}}$ is a **single `(50304, 256) @ (256, 512)` matrix multiply once per optimizer step** (`< 0.4 ms` on GPU!), independent of batch size $B$ and sequence length $T$!
     - Yet mathematically, by linearity of the inner product $\langle \phi_{t} \mid M_{\text{seed}}(w) + g_{\text{readout}} \sum_{m=1}^{256} A_{w, m} \tilde{\Phi}_m \rangle = \langle \phi_t \mid M_{\text{seed}}(w) \rangle + g_{\text{readout}} \sum_{m=1}^{256} A_{w, m} \langle \phi_t \mid \tilde{\Phi}_m \rangle$, **every token prediction at every position $(b, t)$ is simultaneously the sum of its direct token measurement AND its two-level hierarchical measurement through the 256 interacting higher-level quantum states**, while running inside `direct_chunked_povm_ce` at full speed (`~70k–90k tok/s`)!

### 2.4 9-Shard PPMI-SVD Spectral + Centroid Initialization of the 2-Level Hierarchy
In `initialize_hierarchical_16qubit_with_ppmi_svd`:
1. Computes (or loads from disk cache) the 9-shard (`900M`-token) Shifted PPMI + Randomized SVD spectral embeddings $E_{\text{ppmi}} \in \mathbb{R}^{50,304 \times 512}$.
2. Reshapes $E_{\text{ppmi}}$ into `(50,304, 16, 32)`, blends with isotropic sub-register vectors `(1 - alpha) * Z + alpha * E_ppmi`, and normalizes each of the 16 sub-registers on $S^{31}$ to initialize:
   - `unigram_qubits` (`50,304 × 16 × 32`, placed on even coordinates `0::2`), and
   - `povm_odd_qubits` / `povm_head[:, 1::2]` (`50,304 × 16 × 32`, placed on odd coordinates `1::2`) so Step-0 self-repetition overlap is identically zero ($\max_w |\langle M_w \mid \psi_0(w)\rangle| < 10^{-6}$).
3. **Spectral Centroid Initialization of the `256` Higher-Level Quantum States (`256 × 16 × 32`)**:
   - Runs 8 iterations of fast vectorized GPU spherical k-means / stratified spectral partitioning over the active vocabulary's `(V_active, 512)` PPMI-SVD states to initialize `higher_states` (`256 × 16 × 32`) as 256 well-separated semantic cluster centroids on $(S^{31})^{16}$!
   - Initializes $H_{\text{high}}[q] \in \mathbb{R}^{256 \times 256}$ as the orthogonalized affinity graph between those 256 centroids ($\text{polar}(I_{256} + 0.20 \, \text{Sim}(\Phi_q, \Phi_q))$) so related higher-level states are immediately connected at Step 0!
   - Initializes $A_{\text{vocab}\leftarrow\text{high}} \in \mathbb{R}^{50,304 \times 256}$ from the token-to-centroid cosine similarities so the 2-level readout immediately boosts semantically related tokens at Step 0!

---

## 3. Observable Hypotheses & Quantitative Pass Criteria

### Hypothesis H1 (`50,304 × 16 × 32` Multi-Qubit Table + `256`-State Quantum Hierarchy Geometry & Step-0 Spectral Bridging)
- **Claim**:
  1. `Hierarchical16QubitEmbedder` stores exact shape `(50304, 16, 32)` (`25,755,648` parameters) in `unigram_qubits`, `(256, 16, 32)` (`131,072` parameters) in `higher_states`, and `(16, 256, 256)` (`1,048,576` parameters) in `higher_interaction`.
  2. Every sub-register state has unit norm after scaling ($\|\psi_{t, q}\|_2 = 1/\sqrt{16} \pm 10^{-5}$, total state norm $\|\psi_0(t)\|_2 = 1.0 \pm 10^{-5}$).
  3. After `initialize_hierarchical_16qubit_with_ppmi_svd`:
     - Step-0 self-repetition overlap is strictly zero ($\max_w |\langle M_w \mid \psi_0(w)\rangle| < 10^{-6}$),
     - Frequent collocation pairs $(u, v)$ share significantly higher **2-level hierarchical concept similarity** $\langle p(u) \mid H_{\text{high}} p(v) \rangle$ across the 256 higher-level quantum states than random token pairs (ratio $\ge 2.5\times$).

### Hypothesis H2 (Controlled Local RTX 5060 Ti Training: `16 × 32` + `256`-State Hierarchy + `4d-TwoBeam` vs. Wave 14 `64`-Qubit Sparse Fock & Flat `d=1024` Baselines)
- **Claim**: On a controlled 200-step (`1.64M` token, $B=16, T=512$, real FineWeb-Edu tokens) local RTX 5060 Ti benchmark:
  1. Preserving the `4d-TwoBeam` (`d=1024`) backbone while equipping `QuantumBornGPT` with the **`50,304 × 16 × 32` unigram qubit table + `256 × 16 × 32` interacting higher-level quantum hierarchy (`--hierarchical-16qubit --num-higher-states 256`)** achieves **both lower Validation Cross-Entropy (`Val CE`) AND higher training throughput (`tok/s`)** than Wave 14's `D_Fock=2048` `Sparse64QubitDensityLayer` configuration, and beats the flat 10-qubit `d=1024` baseline by **`>= 0.15` nats**.
  2. Ablating the `256`-state higher-level interaction (`--no-higher-hierarchy` or zeroing `higher_interaction`) degrades validation CE compared to the full interacting `256`-state hierarchy, confirming that the `256` higher-level quantum states actively help connect lower-level tokens.

---

## 4. Planned File Changes

1. **Create [`src/qdlblocks/layers/hierarchical_qubit.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/hierarchical_qubit.py)**:
   - `Hierarchical16QubitEmbedder(nn.Module)`:
     - `self.unigram_qubits = nn.Parameter` of shape `(vocab_size, 16, 32)` (`25,755,648` params at `V=50,304`).
     - `self.higher_states = nn.Parameter` of shape `(num_higher_states, 16, 32)` (`131,072` params at `M=256`).
     - `self.higher_interaction = nn.Parameter` of shape `(16, num_higher_states, num_higher_states)` (`1,048,576` params at `M=256`).
     - `self.higher_sub_unitary = nn.Parameter` of shape `(16, 32, 32)` (`16,384` params).
     - `self.register_gamma = nn.Parameter` (`(16,)`), `self.register_tau = nn.Parameter` (`(16,)`), `self.causal_mem_gate = nn.Parameter` (`(16,)`), `self.causal_mem_weights = nn.Parameter` (`(16, 4)`), `self.hier_mix_gain = nn.Parameter` (`(16,)`).
     - Methods:
       - `get_interacted_higher_states() -> torch.Tensor` returning $\tilde{\Phi} \in \mathbb{R}^{M \times 16 \times 32}$,
       - `route_through_hierarchy(psi_reg: torch.Tensor, layer_gate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]` mapping `(B, T, 16, 32) -> (psi_hier: (B, T, 16, 32), p_high: (B, T, 16, M))`,
       - `forward_block_bridge(psi_1024: torch.Tensor, layer_idx: int) -> torch.Tensor` applying the per-block 256-state hierarchical bridge on $S^{1023}$ (including even $\leftrightarrow$ odd parity coupling),
       - `forward(idx: torch.Tensor) -> torch.Tensor` returning $\psi_0 \in \mathbb{R}^{B \times T \times 1024}$ on $S^{1023}$.
   - `Hierarchical16QubitPOVMReadout(nn.Module)`:
     - `self.povm_odd_qubits = nn.Parameter` of shape `(vocab_size, 16, 32)` (`25.76M` params) and `self.vocab_higher_affinity = nn.Parameter` of shape `(vocab_size, num_higher_states)` (`12.88M` params at `M=256`), plus `self.hier_readout_gain` and `self.inv_temp`.
     - `get_coupled_povm_matrix(embedder: Hierarchical16QubitEmbedder, base_povm_head: torch.Tensor | None = None) -> torch.Tensor` returning the normalized `(vocab_size, embed_dim)` POVM effect matrix combining fine-grained `16 × 32` token detectors and the `256`-state hierarchical readout bridge.
   - `initialize_hierarchical_16qubit_with_ppmi_svd(...)`:
     - Initializes `unigram_qubits` (`50,304 × 16 × 32`), `povm_odd_qubits` (`50,304 × 16 × 32`), `higher_states` (`256 × 16 × 32` via spherical spectral k-means centroids), `higher_interaction` (`16 × 256 × 256` via orthogonalized centroid affinity graph), and `vocab_higher_affinity` (`50,304 × 256`).
2. **Integrate into [`QuantumBornGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/born_gpt.py), [`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py), [`src/qdlblocks/benchmarks/model_card.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/model_card.py), and [`scripts/inspect_model.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py)**:
   - Add `--hierarchical-16qubit`, `--qubit-registers 16`, `--qubit-amplitudes 32`, `--num-higher-states 256`, and `--no-block-hierarchy-bridge`.
   - Ensure `split_muon_adamw_params` sends `higher_interaction` (`16 × 256 × 256`) and `higher_sub_unitary` (`16 × 32 × 32`) to `StiefelMuon` (3D batched orthogonalization!) while sending `unigram_qubits`, `higher_states`, `povm_odd_qubits`, and `vocab_higher_affinity` to `AdamW` (`weight_decay=0.0`, `lr=base_adamw_lr * 2.5`).
3. **Create Unit Tests & Controlled Local RTX 5060 Ti Benchmark**:
   - [`tests/test_hypothesis_023_hierarchical_16qubit_256states.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_023_hierarchical_16qubit_256states.py)
   - [`examples/33_verify_plan023_hierarchical_16qubit_experiments.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/33_verify_plan023_hierarchical_16qubit_experiments.py)

---

## 5. Operational Tooling & Script Identification Gate

### 5.1 Existing Scripts Reused
- [`scripts/colab_fleet.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py): Used to harvest Wave 14 logs and register Model Cards.
- [`scripts/monitor_colab_runs.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py): Zero-downtime SQLite + HTTP dashboard server (`http://127.0.0.1:8765`).
- [`scripts/inspect_model.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py): Extended with `--hierarchical-16qubit`, `--qubit-registers`, `--qubit-amplitudes`, and `--num-higher-states`.

### 5.2 Nice-to-Have Script Logged to `scripts/BACKLOG.md`
- `scripts/inspect_hierarchy_clusters.py` (`Nice-to-Have`, logged to `scripts/BACKLOG.md`): Post-training CLI diagnostic that prints the top-10 vocabulary tokens most strongly coupled to each of the `256` higher-level quantum states (`256 × 16 × 32`) and the strongest off-diagonal concept-to-concept transitions in $H_{\text{high}} \in \mathbb{R}^{16 \times 256 \times 256}$.
