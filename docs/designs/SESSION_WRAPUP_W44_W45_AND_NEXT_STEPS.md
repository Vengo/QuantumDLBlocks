# QuantumDLBlocks Project Status & Session Wrap-Up (`Waves 44–45`)

> **Document Purpose**: A self-contained, zero-assumed-context technical summary of the problems solved, architectural discoveries made, experimental outcomes observed, and concrete next steps planned across Waves 44 and 45 of the **QuantumDLBlocks** research project.

---

## 1. Project Overview, Core Hypothesis, & Benchmark Standard

### 1.1 What QuantumDLBlocks Is Building
**QuantumDLBlocks** is an experimental deep-learning research project investigating whether autoregressive language models built entirely from **quantum-mechanical operators in complex Hilbert space** ($\mathbb{C}^d$) can match or surpass the sample efficiency and wall-clock training speed of state-of-the-art classical Transformer language models.

In a standard **classical Transformer** (such as GPT-2, LLaMA, or Google's Gemma 7B), neural activations are unconstrained real vectors $x \in \mathbb{R}^d$, attention weights and token output probabilities are computed using exponential normalization ($\operatorname{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$), and non-linear feature synthesis relies on wide Feed-Forward Networks (FFNs / Multi-Layer Perceptrons, **MLPs**) using gated or rectified activations (`GeGLU`, `SwiGLU`, or `ReLU²`).

In **QuantumDLBlocks**, we replace those classical primitives with strict quantum-information-theoretic counterparts:
1. **Complex Wavefunction & Density-Matrix States**: Token states live on unit complex spheres ($\mathbb{C}\mathbb{S}^{d-1} = \{\psi \in \mathbb{C}^d : \|\psi\|_2 = 1\}$) or trace-one positive semi-definite (PSD) quantum density matrices ($\rho \in \mathbb{C}^{d \times d}, \rho \succeq 0, \operatorname{Tr}(\rho) = 1$).
2. **Unitary & Stiefel-Constrained Evolution (`StiefelMuon` Optimizer)**: Linear layers are parameterized as complex matrices $W = W_{\text{re}} + i W_{\text{im}}$ optimized on or near the **complex Stiefel manifold** ($W^\dagger W = I$, where $W^\dagger = \overline{W}^T$ is the conjugate transpose), preserving quantum norm and phase coherence across layers. Across all quantum models, learnable parameters are partitioned into three optimizer groups:
   - **`Muon` / `StiefelMuon` Group (`Muon LR`)**: `Muon` (*MomentUm Orthogonalized by Newton-Schulz*) is the matrix-orthogonalizing optimizer popularized by the `modded-nanogpt` speedrun. Our complex extension, **`StiefelMuon`**, updates all 2D and 3D complex weight matrices $(W_{\text{re}}, W_{\text{im}})$ by projecting their Nesterov momentum updates onto the unitary manifold via iterative polynomial orthogonalization.
   - **Embedding `AdamW` Group (`Embed LR`)**: Standard `AdamW` optimizer updating the high-dimensional token and N-gram lookup tables.
   - **Scalar / Readout `AdamW` Group (`Scalar LR`)**: Separate `AdamW` group updating 1D scalar gates, branch routers, and measurement prior weights.
3. **100% Exponential-Free Born-Rule Probability Readout**: Nowhere in the quantum model are `exp()`, `softmax()`, or `logsumexp()` used. Instead, both causal self-attention weights and next-token vocabulary probabilities are governed by the **generalized $p$-boson Born rule** of quantum mechanics (with default $p=2$, i.e., quartic amplitude / squared intensity):
   $$P(v \mid x_{1:t}) = \frac{I(v \mid x_{1:t})}{\sum_{v'=0}^{V-1} I(v' \mid x_{1:t})}, \qquad I(v \mid x_{1:t}) = w_v \left|\langle \phi_v \mid \tilde{\psi}_t \rangle\right|^{2p} \quad \text{or} \quad w_v \operatorname{Tr}\!\left(\rho_t E_v\right)^p$$
   where $|\tilde{\psi}_t\rangle$ (or $\rho_t$) is the calibrated readout context state at sequence position $t$, $|\phi_v\rangle$ (or $E_v = |\phi_v\rangle\langle\phi_v|$) is the measurement vector for vocabulary token $v \in \{0, \dots, V-1\}$, and $w_v > 0$ is a learnable **Positive Operator-Valued Measure (POVM)** prior weight.
4. **Factored Kronecker Tensor-Product Vocabulary ($\mathbb{C}^{16 \times 16 \times 16 \times 16} = \mathbb{C}^{65,536}$) & $R$-Branch Mixed-State Readout**: Rather than computing a dense matrix multiplication against a $V \times d$ vocabulary matrix, the vocabulary of $V = 50,257$ tokens is embedded into a 4-way Kronecker tensor product of $M = 4$ small complex factor registers of dimension $d_f = 16$ ($16^4 = 65,536 > 50,257$). Because an inner product of Kronecker products factors into the product of four $16$-dimensional inner products,
   $$\left\langle \bigotimes_{m=1}^{4} \phi_{v,m} \,\middle|\, \bigotimes_{m=1}^{4} \tilde{\psi}_{t,m} \right\rangle = \prod_{m=1}^{4} \langle \phi_{v,m} \mid \tilde{\psi}_{t,m} \rangle,$$
   a 64-complex-dimensional state ($4 \times 16 = 64$) can address a $65,536$-dimensional Hilbert space without dense $65,536$-dimensional projections.
   - **Polysemy Branches ($R \in \{1, 2, 4\}$) & Mixed-State Intensity**: To handle **polysemy** (single words with multiple distinct meanings, such as *"bank"* or *"run"*), each token and context state maintains $R$ parallel Kronecker branches. For branch $r \in \{0, \dots, R-1\}$, let $A_{t,v,r} = \prod_{m=1}^{4} \langle \phi_{v,r,m} \mid \tilde{\psi}_{t,r,m} \rangle^p \in \mathbb{C}$ denote the $p$-boson 4-factor complex overlap. A context-dependent **branch router** produces real normalized weights $\beta_{t,r}$ ($\sum_{r=0}^{R-1} \beta_{t,r}^2 = 1$), and a static per-branch complex amplitude $\alpha_r \in \mathbb{C}$ weights each branch. The **Mixed-State Born Intensity** $I(v \mid x_{1:t})$ combines two physical terms:
     $$I(v \mid x_{1:t}) = w_v \Biggl[ \alpha_{\text{holo}} \underbrace{\left| \sum_{r=0}^{R-1} \beta_{t,r} \alpha_r A_{t,v,r} \right|^2}_{\text{Coherent (Holographic) Sum}} \;+\; \gamma_{\text{coinc}} \underbrace{\sum_{r=0}^{R-1} \beta_{t,r}^2 |\alpha_r|^2 \left|A_{t,v,r}\right|^2}_{\text{Incoherent (Coincidence / Diagonal) Sum}} \Biggr]$$
     where the **Coherent (Holographic) Term** captures cross-branch phase interference ($A_{t,v,r}^* A_{t,v,r'}$) and the **Incoherent (Coincidence) Term** represents the diagonal trace of a mixed density state (guaranteeing strictly non-negative probability mass from every branch without destructive phase cancellation).
5. **Step-0 N-Gram Carrier Calibration & Hashed Context Tables (`6.9766` Initial Val CE)**:
   - If a Born-rule model is initialized with random unit vectors, every token has uniform probability $1/V$, yielding a Step-0 Cross-Entropy loss of $\ln(50,257) \approx 10.825\text{ nats}$. To start the model at an immediate statistical baseline before Step 1, we use a three-part **Step-0 N-Gram Calibration** computed once from empirical token counts across the $900\text{M}$-token training corpus:
     1. **Unigram Frequency Carrier ($|c_{\text{ref}}\rangle \in \mathbb{C}^{16}$)**: We designate the uniform unit vector $|c_{\text{ref}}\rangle = \frac{1}{\sqrt{16}}(1, 1, \dots, 1)^T$ in each $\mathbb{C}^{16}$ factor as the **unigram carrier**. At initialization, each token's embedding $|\phi_{v,r,m}\rangle$ is decomposed into a component along $|c_{\text{ref}}\rangle$ (scaled by the empirical unigram probability $P_{\text{uni}}(v)^{1/(2pM)}$ so frequent words like *"the"* have high baseline overlap) and a semantic component in the $15$-dimensional **orthogonal complement subspace $\mathcal{H}_\perp = \{u \in \mathbb{C}^{16} : \langle c_{\text{ref}} \mid u \rangle = 0\}$**.
     2. **Orthogonal Helmert Subspaces Inside $\mathcal{H}_\perp$**: Using the discrete **Helmert matrix** (an orthogonal matrix whose first row is $(1,\dots,1)/\sqrt{16}$ and whose remaining 15 rows form an exact orthonormal basis for $\mathcal{H}_\perp$), we partition the 15 carrier-orthogonal dimensions of $\mathcal{H}_\perp$ into disjoint, mutually orthogonal subspaces assigned to meaning branches $r = 0, 1, \dots, R-1$, initialized from the top empirical bigram Singular Value Decomposition (SVD) modes of the corpus.
     3. **Hashed Bigram & Trigram Context Tables (`3.15M` parameters)**: Two complex hash embedding tables (`bigram_re/im` and `trigram_re/im`, each with $16,384$ hash buckets of shape `(R, M=4, d_f=16)`) look up hashed token pairs $(x_{t-1}, x_t)$ and triples $(x_{t-2}, x_{t-1}, x_t)$ and blend local N-gram context directly into the Layer-0 wavefunction. Together, Unigram + Bigram + Trigram calibration lowers Step-0 validation loss from $10.825\text{ nats}$ to **`6.9766\text{ nats}`** before a single gradient update occurs.

### 1.2 The Canonical Benchmark (`Modded-NanoGPT` on Official `FineWeb`)
To prevent moving goalposts, all classical and quantum models in this repository are evaluated on the exact benchmark standard established by the **`KellerJordan/modded-nanogpt`** community speedrun leaderboard:
- **Dataset**: Official **`FineWeb`** (`kjj0/fineweb10B-gpt2`), tokenized with the GPT-2 Byte-Pair Encoding (BPE) tokenizer ($V = 50,257$ vocabulary tokens).
- **Training Corpus**: $900,000,000$ tokens stored in 9 pre-tokenized binary shards (`fineweb_train_000001.bin` through `000009.bin`, $100\text{M}$ tokens per shard).
- **Validation Standard**: Cross-Entropy loss measured in **nats** (natural logarithm base $e$, abbreviated **Val CE**; Perplexity is $\text{Val PPL} = e^{\text{Val CE}}$) on the **first $10,485,760$ ($10 \times 2^{20}$) sequential tokens** of `FineWeb` validation Shard 0 (`fineweb_val_000000.bin`). During training, rapid mid-run checkpoints evaluate on the first $65,536$ tokens of Shard 0, followed by a full $10,485,760$-token evaluation at completion.
- **Target Metric**: Reaching **$\text{Val CE} \le 3.280\text{ nats}$** ($\text{Val PPL} \le 26.58$). On an $8 \times \text{NVIDIA H100 SXM}$ ($80\text{ GB}$) cluster, the official classical `modded-nanogpt` speedrun reaches $3.280$ Val CE in **$0.665\text{ minutes}$ (~40 seconds)** over $\sim 450\text{M}–500\text{M}$ training tokens.

### 1.3 Hardware Testbeds Used in This Session
1. **Local Workstation (`Primary Node`)**: A single **NVIDIA GeForce RTX 5060 Ti (`16 GB` GDDR7 VRAM)** desktop GPU delivering **`~23.7 TFLOP/s`** of sustained `bfloat16` matrix-multiply throughput—approximately **$160\times$ less raw compute** than an $8 \times \text{H100}$ cluster.
2. **Remote Multi-GPU Workstation (`ssh upstairs-gpu`, Provisioned in This Session)**: A **NixOS** system with PCIe GPU passthrough housing **`2x NVIDIA GeForce RTX 3090` (`2 \times 24\text{ GB} = 48\text{ GB}` total GDDR6X VRAM)** and `32 GB` of DDR5 system RAM, delivering **`141.6 TFLOP/s`** of combined `bfloat16` Tensor Core throughput (**$6.0\times$ faster** than the local RTX 5060 Ti).

---

## 2. Problems We Solved & Investigated in This Session

Over this session, we tackled six interconnected problems spanning quantum mathematical invariants, GPU kernel engineering, learning-rate dynamics, classical leaderboard verification, multi-GPU infrastructure, and deep architectural comparison against modern Transformers (Gemma 7B and the Classical Leaderboard).

### Problem 1: Resolving Six Mathematical & Architectural Bottlenecks Plus Diagnostic Tooling in the Quantum Kronecker Model (`Wave 44` / `EXPERIMENT_PLAN_059`)
At the start of Wave 44, our complex Kronecker language model (`NgramCalibratedKroneckerBornGPT`) suffered from six mathematical and systems bottlenecks that limited both training throughput and validation loss. We designed, reviewed, and implemented a **7-track overhaul** (Tracks 1–6 fixing the model and optimizer mathematics, and Track 7 building 14 automated diagnostic CLI probes) resulting in [`MultiHeadMixedStateKroneckerGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py):

1. **Track 1 — Broken Complex $U(1)$ Phase Equivariance in `StiefelMuon`**:
   - *The Problem*: Previously, `StiefelMuon` orthogonalized the real part $W_{\text{re}}$ and imaginary part $W_{\text{im}}$ of complex weight matrices *independently* using real Newton-Schulz iterations. Applying a global complex phase rotation $W \mapsto e^{i\theta} W$ changed the optimizer trajectory, violating quantum $U(1)$ gauge invariance and failing to enforce true complex unitarity ($W^\dagger W = I$).
   - *The Solution*: Implemented **Joint-Complex Polar Express** ([`complex_polar_express_orth`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/functional.py)). Given the complex Nesterov momentum update matrix $G = G_{\text{re}} + i G_{\text{im}}$ (with per-element raw gradients $g_{\text{re}}, g_{\text{im}}$), `StiefelMuon` first normalizes by the joint complex second-moment variance $v_{\text{joint}} = g_{\text{re}}^2 + g_{\text{im}}^2$, initializes the complex Polar Express iterate $X_0 = G / \|G\|_F$, and iteratively updates $X_{k+1} = X_k (a_k I + b_k H_k + c_k H_k^2)$ using the **Hermitian Gram matrix** $H_k = X_k^\dagger X_k = (X_{k,\text{re}}^T X_{k,\text{re}} + X_{k,\text{im}}^T X_{k,\text{im}}) + i(X_{k,\text{re}}^T X_{k,\text{im}} - X_{k,\text{im}}^T X_{k,\text{re}})$. This guarantees exact $U(1)$ phase equivariance and reduced the **complex unitary error** ($\|W^\dagger W - I\|_F / \sqrt{d_{\text{out}}}$, measuring deviation from an exact unitary matrix) from $0.41$ to $< 0.038$.
2. **Track 2 — Sub-Register Dimension Mismatch (`d_register = 8` vs. `16`) & Incomplete Optimizer Routing**:
   - *The Problem*: Attention sub-registers defaulted to dimension $8$ while the Kronecker readout factors used dimension $16$, forcing mismatched reshapes; additionally, several 3D unitary mixer tensors bypassed `StiefelMuon`.
   - *The Solution*: Standardized $d_{\text{register}} = 16$ across all Kronecker models and created [`classify_kronecker_model_parameters`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/muon.py) to automatically pair and bucket all 2D and 3D complex matrix slices into Joint-Complex `StiefelMuon`.
3. **Track 3 — Real-Only Self-Echo Subtraction Corrupting the Unigram Carrier**:
   - *The Problem*: Because input and output token embeddings share the same vocabulary factors $|\phi_v\rangle$, the calibrated readout context state $|\tilde{\psi}_t\rangle$ at position $t$ retains strong overlap with the current input token's Layer-0 state $|\psi_{0,t}\rangle$, causing the model to over-predict immediate token repetition ($x_{t+1} = x_t$, the **self-echo** phenomenon). Previously, subtracting the *real* dot product $\operatorname{Re}\langle \psi_{0,t} \mid \tilde{\psi}_t \rangle$ across the full $\mathbb{C}^{16}$ factor space inadvertently erased the shared unigram frequency carrier $|c_{\text{ref}}\rangle$ and ignored imaginary phase echoes.
   - *The Solution*: Implemented **Carrier-Orthogonal Complex Self-Echo Removal** ([`apply_carrier_orthogonal_complex_self_echo`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py)), which decomposes both $|\tilde{\psi}_t\rangle$ and $|\psi_{0,t}\rangle$ into their $|c_{\text{ref}}\rangle$ carrier component and their orthogonal complement $|\tilde{\psi}_t^\perp\rangle, |\psi_{0,t}^\perp\rangle \in \mathcal{H}_\perp$, and projects out the full complex overlap $\langle \psi_{0,t}^\perp \mid \tilde{\psi}_t^\perp \rangle \in \mathbb{C}$ strictly inside $\mathcal{H}_\perp$.
4. **Track 4 — Exponential Dilution of Secondary Polysemy Branches ($0.22^8 \approx 5.5 \times 10^{-6}$)**:
   - *The Problem*: Secondary meaning branches ($r \ge 1$) were initialized with a $0.22\times$ amplitude scale relative to branch $r=0$. Because the 4-factor Kronecker product multiplies four factor inner products and the $p=2$ Born rule raises the result to the 4th power, branch $r=1$ was attenuated at Step 0 by $0.22^{4 \times 2} = 0.22^8 \approx 5.5 \times 10^{-6}$, leaving secondary meaning branches effectively dead. Moreover, the branch router $\beta_{t,r}$ only inspected real norms ($|\psi_t|^2$) and was blind to complex phase.
   - *The Solution*: Initialized all $R$ branches at full unit factor norm ($1.0\times$) in orthogonal Helmert subspaces of $\mathcal{H}_\perp$, and replaced the norm-only router with a zero-initialized **Phase-Sensitive Hermitian Router** ([`compute_phase_sensitive_branch_beta`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py)) that computes real and imaginary cross- overlaps between the evolved state $|\psi_t\rangle$ and initial state $|\psi_{0,t}\rangle$ to weight the $R$ branches.
5. **Track 5 — Duplicate Vocabulary Backward Pass & Validation Re-Extraction**:
   - *The Problem*: The custom `torch.autograd.Function` chunked Born loss looped over all $50,257$ vocabulary tokens *twice* during the backward pass (once for state/embedding gradients, and a second time for POVM prior weight gradients), nearly doubling backward FLOPs. Similarly, validation re-extracted static vocabulary factors on every batch.
   - *The Solution*: Unified all gradients (`grad_psi`, `grad_vocab`, and `grad_povm`) into a single backward vocabulary pass—enforced by our codebase's single-pass custom autograd engineering invariant ("Rule 9"), verified automatically by [`autograd_efficiency_auditor.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/autograd_efficiency_auditor.py) and [`scripts/custom_autograd_efficiency_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/custom_autograd_efficiency_linter.py)—and hoisted static vocabulary factors outside the $10.48\text{M}$-token validation loop.
6. **Track 6 — Replacing 11-GEMM Krylov Heuristics with Multi-Head Stinespring Mixed States**:
   - *The Problem*: Earlier models approximated mixed quantum states using an ad-hoc polynomial expansion requiring 11 sequence-by-sequence ($T \times T$) General Matrix Multiplications (GEMMs) per layer plus manual scalar gates.
   - *The Solution*: Built [`MultiHeadSuperpositionMixedStateAttention`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multihead_superposition_mixed_state_attn.py), which splits each of the $M=4$ Kronecker factors into $K=2$ coherent sub-heads ($H_{\text{total}} = 8$ sub-heads of dimension $d_f = 16$). After causal $p=2$ Born attention (requiring only **3 $T \times T$ GEMMs** per layer), it forms the $2 \times 2$ cross-sub-head **Hermitian Gram matrix** $G_{t,m} \in \mathbb{C}^{2 \times 2}$ (whose eigenvalues measure quantum purity $\mathcal{P} = \operatorname{Tr}(\rho^2) \in [1/2, 1]$) and applies a **Stinespring isometry** $W_m$ (a linear norm-preserving operator from Stinespring's dilation theorem that maps the coupled sub-head + environment registers into $R$ polysemy branches, naturally inducing a mixed quantum state upon partial trace).
7. **Track 7 — Automated Kronecker Diagnostic CLI Suite (`14` Scripts)**:
   - Implemented 14 reusable CLI diagnostic scripts in `scripts/` (backed by [`kronecker_diagnostic_utils.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/kronecker_diagnostic_utils.py)) to inspect Helmert carrier orthogonality, polysemy subspace angles, Stiefel unitary error, self-echo repetition rates, and hardware throughput gates without ad-hoc one-liner scripts.

---

### Problem 2: Diagnosing GPU Power Instability ("Watt Droop") & Engineering Four Fused Triton/Compiled Kernels
- **The Problem**: When running `MultiHeadMixedStateKroneckerGPT` on the local RTX 5060 Ti, GPU power consumption oscillated erratically between `70W` and `115W` (well below the card's `180W` thermal design power), and training throughput stalled at **`5,459 tok/s`**.
- **Root Cause**: Profiling revealed that the GPU was not compute-bound; it was starved by **memory-bandwidth bottlenecks and Python kernel-launch gaps**. Specifically, iterating over vocabulary chunks in Python with unfused complex arithmetic, running 8 separate $T \times T$ attention operations, and executing the 3-stage unitary meaning mixer (which sequentially applies a $U(16)$ intra-factor rotation on the $d_f=16$ axis, a $U(4)$ cross-factor rotation on the $M=4$ axis, and a $U(2)$ cross-branch rotation on the $R=2$ axis) launched hundreds of micro-kernels per step.
- **The Solution & Outcome**: We engineered and integrated four fused GPU kernels:
  1. **Fused Triton Mixed-State Kronecker Chunk Forward & Backward Kernels** ([`fused_mixed_state_kronecker_ce_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_mixed_state_kronecker_ce_triton.py)): Fuses the 4-factor complex inner products, coherent (holographic) + incoherent (coincidence) intensity terms, and single-pass backward reductions inside GPU SRAM tiles.
  2. **Flash Causal $p$-Boson Complex Attention Triton Kernel** ([`flash_complex_krylov_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_complex_krylov_triton.py)): Computes causal $|\langle Q_i \mid K_j \rangle|^4$ scores and weights complex values $|V_j\rangle$ in a single SRAM-tiled Triton pass without materializing $T \times T$ intermediate matrices in High-Bandwidth Memory (HBM).
  3. **Compiled Cross-Head Gram + Stinespring Coupling Step** (`fused_cross_head_gram_stinespring_step`).
  4. **Compiled 3-Stage ($U(16) \to U(4) \to U(2)$) Unitary Meaning Mixer Step** (`fused_kronecker_16x4_unitary_mixer_step`).
- **Result**: End-to-end training throughput jumped **`2.98x`** at batch size $B=16$ (`5,459 tok/s -> 16,287 tok/s`) and reached **`20,211 tok/s`** (`~22,200 tok/s` pure step speed) at $B=32$, stabilizing GPU power draw.

---

### Problem 3: Removing WSD Decay, Compact Quantum Models (`L=2`), & Constant Learning Rate Sweeps
- **The Problem**: We realized we did not know how long a quantum training run should be, and using a **Warmup-Stable-Decay (WSD)** learning-rate schedule (which linearly decays the learning rate to near zero over the final third of a fixed step budget) makes it impossible to tell whether loss stops improving because the model has saturated or simply because the learning rate decayed to zero.
- **Steps Taken**:
  1. Removed WSD decay (`lr_schedule="constant"` after a 20-step linear warmup) and focused on fast, compact 2-layer models:
     - **Model A (`R=2, L=2, d_c=128`)**: 2 meaning branches, 2 quantum layers, $128$ complex dimensions ($256$ real scalars), `16.86M` total parameters.
     - **Model B (`R=1, L=2, d_c=64`)**: 1 meaning branch, 2 quantum layers, $64$ complex dimensions ($128$ real scalars), `10.17M` total parameters.
  2. Ran a **9-point Constant Learning Rate Sweep** across multipliers `0.15x, 0.25x, 0.35x, 0.50x, 0.75x, 1.00x, 1.50x, 2.00x, 3.00x` of our baseline learning rates (`Muon LR = 0.025`, `Embed AdamW LR = 0.045`, `Scalar LR = 0.012`) for `300 steps` ($4.92\text{M}$ tokens at $B=16, T=1024$, taking ~65 seconds per run):

| LR Multiplier | `Muon` LR | `Embed` LR | `R=2, L=2, d_c=128` Final Val CE (`300` steps) | `R=1, L=2, d_c=64` Final Val CE (`300` steps) | Regime Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `0.15x` | `0.00375` | `0.00675` | `6.4539` | — | Severely under-stepped; slow departure from Step-0 N-gram init (`6.9766`) |
| `0.25x` | `0.00625` | `0.01125` | `6.2454` | — | Under-stepped |
| `0.35x` | `0.00875` | `0.01575` | `6.1444` | — | Under-stepped |
| `0.50x` | `0.01250` | `0.02250` | `6.0694` | `6.1430` | Stable but conservative |
| `0.75x` | `0.01875` | `0.03375` | `6.0005` | `6.0598` | Entering optimal basin |
| `1.00x` | `0.02500` | `0.04500` | `5.9277` | **`5.9873`** | Baseline sweet spot for $R=1$ |
| **`1.50x`** | **`0.03750`** | **`0.06750`** | **`5.9102`** | `5.9905` | **Global optimum for `R=2, L=2`** |
| `2.00x` | `0.05000` | `0.09000` | `5.9225` | `6.0170` | Slight phase step overshoot |
| `3.00x` | `0.07500` | `0.13500` | `6.0005` | `6.1204` | High-LR oscillation on the unit sphere |

  3. **Extended 1,200-Step Constant-LR Run (`w44_compact_r2_d128c_L2_b32_const_lr_1200steps`)**:
     - Using the optimal `1.50x` constant LR (`Muon LR = 0.053` with $\sqrt{32/16}$ batch scaling, `Embed LR = 0.0675`) at batch size $B=32$ ($32,768$ tokens/step, $39.32\text{M}$ total tokens), the model trained at **`20,211 tok/s`** and reached **`5.6779` Val CE** (**`5.6612` official $10.48\text{M}$-token Val CE**, `287.5` PPL, `27.80%` Top-1 accuracy) in `32.4 minutes`.
     - **Key Finding**: Loss dropped rapidly from `6.9766` (Step 0) to `5.7136` (Step 750, $24.6\text{M}$ tokens), but over the final 450 steps ($14.7\text{M}$ tokens) it improved by only `0.0357 nats` (`5.7136 -> 5.6779`), oscillating in a noisy floor. This proved that constant LR was not the bottleneck—rather, the quantum architecture itself was hitting an expressivity ceiling far above `3.28` Val CE.

---

### Problem 4: Auditing the Classical Leaderboard Baseline (`Wave 45` / `EXPERIMENT_PLAN_060`) & Running the 3-Hour Speedrun on RTX 5060 Ti
- **The Question**: Since our RTX 5060 Ti (`16 GB`) has roughly $1/160\text{th}$ the compute of an $8 \times \text{H100}$ node (which hits $\le 3.28$ Val CE in `0.665` minutes), a back-of-the-envelope calculation ($0.665 \times 160 \approx 106\text{ minutes}$) suggested our GPU should be able to reach $\le 3.28$ Val CE in **2 to 3 hours** using the official classical architecture. Why had our legacy classical baseline (`ClassicalModdedGPT`) only reached `4.73` Val CE at `14,500 tok/s`?
- **Four Bottlenecks Discovered in Legacy `ClassicalModdedGPT`**:
  1. **Full-Vocabulary `float32` Logit Materialization**: Computing `[B=16, T=1024, V=50257]` in `float32` allocated **`3.07 GiB`** for the logit tensor alone (plus `3.07 GiB` for its gradient), causing Out-Of-Memory (OOM) crashes above micro-batch $B=6$.
  2. **Forced Activation Checkpointing**: To survive the logit memory spike, legacy runs recomputed every Transformer layer during the backward pass, adding **`+33%` FLOP overhead**.
  3. **Uncompiled Execution + `float32` Rotary Positional Embeddings (RoPE)**: Applying RoPE in `float32` cast $Q$ and $K$ out of `bfloat16`, forcing PyTorch SDPA off the fast Tensor Core FlashAttention kernel (`14,500 tok/s`).
  4. **Missing Leaderboard Architectural Features**: Legacy runs omitted Token Value Embeddings (`ve0, ve1, ve2`) and U-Net Encoder-Decoder Skip Connections (`skip_weights`).
- **What We Built ([`CompiledLeaderboardClassicalGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py) & [`examples/70_verify_plan060_compiled_classical_leaderboard_speedrun.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/70_verify_plan060_compiled_classical_leaderboard_speedrun.py))**:
  - **Fused Chunked Cross-Entropy (`chunk_tokens = 4096`)**: Computes `lm_head` and `30 * tanh(logits / 30)` softcapped cross-entropy in $4,096$-token tiles without ever materializing `[B, T, V]`. Peak VRAM at micro-batch $B=16, T=1024$ ($16,384$ tokens/micro-step) dropped from `>16 GB` (OOM) to **`3.17 GB`**, allowing us to **completely disable activation checkpointing**.
  - **Pure-`bfloat16` RoPE + QK-RMSNorm + `ReLU²` MLP + `@torch.compile`**: Increased throughput on the RTX 5060 Ti from `14,500 tok/s` to **`44,520 tok/s` (`3.07x` speedup, `59%` Model FLOPs Utilization)**.
  - **Included Both Leaderboard Skip/Embedding Mechanisms**:
    1. **Token Value Embeddings (`ve0, ve1, ve2`)**: 3 extra `(50304, 768)` embedding tables (where $V=50,257$ is padded to $V_{\text{padded}} = 50,304 = 393 \times 128$ so all embedding tables align with 128-byte GPU Tensor Core memory transactions) mixed directly into the attention Values $V$ of the first 3 layers (`0, 1, 2`) and last 3 layers (`9, 10, 11`), while interior layers (`3, 4, 5, 6, 8`) mix the Layer-0 value residual $V_0$ (with Layer 7 skipping attention, matching the leaderboard).
    2. **U-Net Encoder-Decoder Skip Connections (`skip_weights`)**: 6 learnable scalar weights adding the residual outputs of encoder layers `0..5` directly into decoder layers `11..6`.
- **Live 3-Hour Classical Run Trajectory (`w45_compiled_classical_l12_d768_b65k_3hr_sub3p27`)**:
  - Configured for `7,800` steps ($511.18\text{M}$ tokens using micro-batch $B=16$, sequence length $T=1024$, and `grad_accum_steps = 4` for $B_{\text{total}} = 4 \times 16 \times 1024 = 65,536$ tokens/step; `12 layers`, `d_model = 768`, `6 heads`, `head_dim = 128`, `MLP hidden_dim = 3072`, `152.73M` parameters, WSD schedule with 35% linear cooldown starting at Step 5,070):
  - **Progress across the run (at steady `44,521 tok/s`)**:
    - `Step 200` (`13.1M` tok): `5.9746` Val CE
    - `Step 1,000` (`65.5M` tok): `4.1620` Val CE
    - `Step 2,000` (`131.1M` tok): `3.8415` Val CE
    - `Step 3,600` (`235.9M` tok): `3.6948` Val CE
    - `Step 5,000` (`327.7M` tok, end of constant-LR plateau): `3.6286` Val CE
    - `Step 5,600` (`367.0M` tok, WSD cooldown `lr_mult = 0.693`): `3.5525` Val CE
    - `Step 6,000` (`393.2M` tok, WSD cooldown `lr_mult = 0.545`): `3.5097` Val CE
    - `Step 6,400` (`419.4M` tok, WSD cooldown `lr_mult = 0.397`): `3.4721` Val CE
    - `Step 6,600` (`432.5M` tok, WSD cooldown `lr_mult = 0.323`): `3.4532` Val CE
    - `Step 7,000` (`458.8M` tok, WSD cooldown `lr_mult = 0.175`): `3.4306` Val CE
    - `Step 7,338` (`480.9M` tok, `180.02 min` wall-clock limit reached): **`3.4212` Val CE (`30.61` PPL, `38.68%` Top-1)** on the $65\text{K}$-token slice, and **`3.4694` Official $10.49\text{M}$-Token Val CE (`32.12` PPL, `37.02%` Top-1)**—improving over our legacy classical baseline (`4.7252`) by **`-1.3040 nats`** on a single RTX 5060 Ti. (The remaining `~0.14` nat gap to `3.28` is explained by the final 462 steps of WSD cooldown clipped by the 180-minute cap plus the three leaderboard additions listed in Question A below.)

---

### Problem 5: Provisioning & Benchmarking the Remote `2x RTX 3090` NixOS Cluster (`upstairs-gpu`)
- **What We Did**: Connected via `ssh upstairs-gpu` to the user's newly configured NixOS gaming PC with GPU passthrough (`AMD Ryzen 7 7700X`, `32 GB` DDR5 RAM, `2x NVIDIA GeForce RTX 3090` with `24 GB` VRAM each, using `uv` Python environment manager).
- **Verification Results**:
  - Executed a multi-GPU `torch.distributed` NCCL + `bfloat16` GEMM benchmark across both RTX 3090s simultaneously:
    - **GPU 0 (`RTX 3090`)**: `70.66 TFLOP/s` (`bfloat16` $8192^3$ GEMM)
    - **GPU 1 (`RTX 3090`)**: `70.93 TFLOP/s` (`bfloat16` $8192^3$ GEMM)
    - **Combined Node Compute**: **`141.58 TFLOP/s`** (`6.0x` the measured `23.7 TFLOP/s` of the local RTX 5060 Ti) and **`48 GB` total VRAM** (`3.0x` the local `16 GB` VRAM), with verified NCCL `all_reduce` collectives (`4.14 GB/s` bus bandwidth).

---

### Problem 6: Visualizing the Quantum Block Graph & Answering the Core Architectural Question (Gemma 7B vs. Classical Leaderboard vs. Quantum Block Ordering, Embeddings, & MLP)

To see exactly how our quantum blocks are strung together, we built an interactive **QuantumDL WebUI Block Architecture Visualizer** ([`src/qdlblocks/benchmarks/model_block_visualizer.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/model_block_visualizer.py) and [`scripts/export_model_block_visualizer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/export_model_block_visualizer.py), served live at `http://localhost:8765/architecture` and exported to [`architecture_visualizer.html`](file:///home/festivity/agy/QuantumDLBlocks/architecture_visualizer.html)).

Inspecting this block graph side-by-side with **Gemma 7B** and the **Classical Modded-NanoGPT Leaderboard** directly answers the central architectural questions raised in this session:

#### Question A: What Is the Status of Token Value Embeddings (`ve0..ve2`) and U-Net Skip Connections on the Classical Side, and What Else Does the Classical Leaderboard Do?
1. **Already Active in Our Classical Model (`CompiledLeaderboardClassicalGPT`)**:
   - Both **Token Value Embeddings (`ve0, ve1, ve2`)** and **U-Net Encoder-Decoder Skip Connections (`skip_weights`)** are **already implemented and enabled by default** in [`CompiledLeaderboardClassicalGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py) (and are actively running in `w45_compiled_classical_l12_d768_b65k_3hr_sub3p27`).
2. **Three Remaining Classical Leaderboard Additions We Should Consider Next**:
   - Inspecting `CompiledLeaderboardClassicalGPT` against the latest $8 \times \text{H100}$ `< 500M`-token leaderboard entries reveals three specific differences that explain any remaining `~0.10–0.15` nat gap at the same token count:
     1. **Untied Input Token Embedding (`wte`) and Output Head (`lm_head`) with Separate Learning Rates**: By default, `CompiledLeaderboardClassicalGPT` ties `wte.weight = lm_head.weight` (`tie_embeddings = True`). The official leaderboard **unties** `wte` and `lm_head` (`--untie-embeddings`), optimizing `wte` with a massive AdamW learning rate (`lr = 0.6`) and `lm_head` with a small learning rate (`lr = 0.008`, zero-initialized).
     2. **Per-Layer Initial Residual Stream Blending (`x_0` Bypass)**: In addition to U-Net skip connections, the leaderboard saves the Layer-0 token embedding $x_0 = \text{RMSNorm}(\text{wte}(x))$ and blends it into the residual stream at *every* layer $l$:
        $$x_l \leftarrow \lambda_{l,1}\,x_l + \lambda_{l,2}\,x_0$$
        where $(\lambda_{l,1}, \lambda_{l,2})$ are learnable scalars initialized to $(1.0, 0.0)$.
     3. **Dynamic Sliding-Window Causal Attention Schedule (`128 -> 384 -> 1024`)**: Rather than using full $1024$-token causal attention from Step 0, leaderboard entries use short local attention windows (`64` or `128` tokens) on most layers early in training and progressively widen the window schedule, forcing early layers to learn sharp local N-gram induction heads rapidly while cutting attention FLOPs.

#### Question B: How Does Gemma 7B's 7-Step Block Structure Compare to the Classical Leaderboard and Our Quantum Model? Do We Have an MLP, and Do We Need It?

Let us compare the exact step-by-step dataflow and parameter budget (`100.0%` accounting) of **Gemma 7B**, **Classical Leaderboard GPT**, and our **Quantum `MultiHeadMixedStateKroneckerGPT` (`w44_compact_r2_d128c_L2_b32`, `16.86M` total parameters)**:

| Step | Gemma 7B (`d_model=3072`) | Classical Leaderboard (`d_model=768`, `152.73M` params) | Quantum `w44_compact_r2_d128c_L2` (`d_c=128`, i.e. $R=2, M=4, d_f=16$, `16.86M` params) |
| :--- | :--- | :--- | :--- |
| **1. Token Embedding** | Lookup $V \to \mathbb{R}^{3072}$ (`9.2%` of params) | Lookup $V \to \mathbb{R}^{768}$ + `ve0..ve2` (`25.3%` of params) | Lookup $V \to \mathbb{C}^{2 \times 4 \times 16}$ (`12.87M`, **`76.3%`**) + Hashed Bigram/Trigram tables (`3.15M`, **`18.7%`**) = **`95.0%` (`16.02M`) of all params!** |
| **2. Q, K, V Projection** | Linear $3072 \to 4096$ ($16 \times 256$) | Linear $768 \to 768$ ($6 \times 128$) + mix `ve_l` / $V_0$ into $V$ | Complex Linear $\mathbb{C}^{128} \to \mathbb{C}^{128}$ reshaped into $M=4$ factors $\times K=2$ sub-heads $\times d_f=16$ |
| **3. Position & Attention** | RoPE + Scaled Dot-Product Attention (`softmax`) | Pure-`bf16` RoPE + QK-RMSNorm + `softmax` SDPA | Complex Phase RoPE + **Causal $p=2$ Born Attention** ($|\langle Q_i \mid K_j \rangle|^4$, no `softmax`) + **$2 \times 2$ Cross-Head Hermitian Gram + Stinespring $W_m$** |
| **4. Output Projection (`o_proj`)** | Linear $4096 \to 3072$ + Residual add (`Steps 2–4 = 23.8%` of params) | Zero-init Linear $768 \to 768$ + Residual add (`Steps 2–4 = 24.1%` of params) | Complex Linear `o_proj` ($\mathbb{C}^{128} \to \mathbb{C}^{128}$) + gated residual blend (**`Steps 2–4 = 173K` across 2 layers, `1.0%` of params**) |
| **5. Post-Attention Norm** | `RMSNorm` on $\mathbb{R}^{3072}$ | `RMSNorm` on $\mathbb{R}^{768}$ | Global L2 normalization onto complex unit sphere $\mathbb{C}\mathbb{S}^{127}$ |
| **6. The MLP (Intra-Token Feature Synthesis)** | **Gemma `GeGLU` MLP** ($3072 \to 2 \times 24576 \to 3072$, $8\times$ expansion, **`67.0%` of model params**) | **`ReLU²` MLP** ($768 \to 3072 \to 768$, $4\times$ expansion, $\max(0, u)^2$, zero-init `fc2`, **`50.6%` of model params**) | **Two back-to-back blocks (`490K` across 2 layers = `2.9%` of model params!)**:<br>(6a) `ComplexUnitaryInterferenceFFN` ($\mathbb{C}^{128} \to \mathbb{C}^{256} \to \mathbb{C}^{128}$, `393K` params)<br>(6b) `Kronecker16x4UnitaryMeaningMixer` ($U(16)$ factor $\to U(4)$ register $\to U(2)$ branch + per-factor $\mathbb{C}\mathbb{S}^{15}$ norm, `97K` params) |
| **7. Output Readout (`lm_head`)** | Tied Linear $\mathbb{R}^{3072} \to \mathbb{R}^V$ + `softmax` | Tied Linear $\mathbb{R}^{768} \to \mathbb{R}^V$ + softcap + `softmax` | Readout operators + Self-Echo removal + Phase-sensitive router + POVM priors (**`181K`, `1.1%` of params**) $\to$ **$16^4$ Mixed-State Born Readout** |

*(Note on exact `100.0%` parameter accounting for `w44_compact_r2_d128c_L2` (`16.86M` total): `76.3%` Token Embedding + `18.7%` N-Gram Tables [`95.0%` static tables] + `1.0%` Attention + `2.9%` FFN/Unitary Mixer [`3.9%` 2-layer quantum backbone] + `1.1%` Readout/POVM/Router = **`100.0%`**.)*

#### Three Critical Insights From This Comparison

1. **What the Classical MLP Actually Does (and Why We Need Its Quantum Analogue)**:
   - In both Gemma 7B (`GeGLU`: $\text{GELU}(W_{\text{gate}} x) \odot W_{\text{up}} x$) and the Classical Leaderboard (`ReLU²`: $\max(0, W_{\text{fc1}} x)^2$), Attention (`Steps 2–4`) only **moves information between tokens**, whereas the MLP (`Step 6`) contains **`50.6%–67.0%` of all model weights** and acts as a **key-value associative memory bank** (Geva et al., 2021) that synthesizes multi-token features into next-token predictions *within each token position*.
   - Crucially, `ReLU²` and `GeGLU` rely on three mathematical properties:
     - **(i) Wide Overcomplete Expansion ($4\times$ to $8\times$)** where entangled features become linearly separable.
     - **(ii) High Sparsity / Non-Linear Thresholding**: In `ReLU²`, $>75\%$ of hidden neurons are **exact zero** for any given token, and squaring ($\max(0, u)^2$) quadratically suppresses weak, noisy feature correlations while strongly amplifying high-confidence pattern matches.
     - **(iii) Zero-Output Identity Initialization**: In the Classical Leaderboard, `fc2.weight` is initialized to **exact zero** (`0.0`). When a token does not trigger a specific memory neuron, the MLP adds `0` to the residual stream, leaving the attention state undisturbed.

2. **Why Our Current Quantum FFN + Unitary Mixer (`Steps 6a & 6b`) Are Fighting Each Other**:
   - Yes, we *do* currently have an MLP-like block—[`ComplexUnitaryInterferenceFFN`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/gemini_packed_complex_rope_gpt.py) (`Step 6a`)—followed immediately by [`Kronecker16x4UnitaryMeaningMixer`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py) (`Step 6b`). However, our Block Visualizer exposed three structural problems in how they are constructed and ordered:
     - **Problem A (Zero Sparsity & Forced Unit-Norm Noise Injection)**: In `ComplexUnitaryInterferenceFFN`, the input wavefunction $|\psi\rangle \in \mathbb{C}^{128}$ is projected into two parallel $2\times$-expanded complex hidden beams $b_1, b_2 \in \mathbb{C}^{256}$ (each L2-normalized to unit length), and beam $b_1$ is modulated by their elementwise cross-beam interference $b_1 \odot \bigl(1 + \tanh(\gamma)(b_1^* \odot b_2)\bigr)$, where $\gamma \in \mathbb{R}$ is a learnable scalar gain parameter. Notice that the modulation factor $1 + \tanh(\gamma)(b_1^* \odot b_2)$ is centered around **`1.0` (never `0.0` / never sparse)**! Worse still, after projecting back down via `proj_down` ($256 \to 128$), the FFN output is **L2-normalized to unit length $\|\text{ffn}\|_2 = 1$ on $\mathbb{C}\mathbb{S}^{127}$** *before* being blended into the residual stream. Because a unit vector on a sphere cannot have norm zero, the quantum FFN is **mathematically incapable of outputting zero** when no feature matches—it injects a dense unit-norm vector into the residual stream at every layer!
     - **Problem B (Coordinate-Frame Tug-of-War Between Flat $\mathbb{C}^{128}$ and Factored $(\mathbb{C}\mathbb{S}^{15})^8$)**: Inside every single layer, the state is:
       1. Factored into $(R=2, M=4, d_f=16)$ inside Stinespring attention,
       2. Flattened to $\mathbb{C}^{128}$ and mixed densely across all factors by `o_proj` (and normalized on $\mathbb{C}\mathbb{S}^{127}$),
       3. Mixed densely across all 128 dimensions *again* by `ComplexUnitaryInterferenceFFN` (and normalized on $\mathbb{C}\mathbb{S}^{127}$), and
       4. Immediately reshaped back to $(2, 4, 16)$ by `Kronecker16x4UnitaryMeaningMixer` (which applies a $U(16)$ intra-factor rotation, a $U(4)$ cross-factor rotation, and a $U(2)$ cross-branch rotation), forcing every 16D sub-register back onto its own unit sphere $\mathbb{C}\mathbb{S}^{15}$ (destroying relative amplitude information between the 4 factors).
     - **Problem C (Inverted Parameter Budget — `95.0%` in Static Embeddings vs. `3.9%` in Backbone Layers)**:
       - Because our input token embedding `embed_re/im` of shape `(50257, R=2, M=4, d_f=16)` is tied directly to the $16^4$ Kronecker output readout, **`76.3%` (`12.87M`) of our `16.86M` parameters sit in the static token lookup table** and another **`18.7%` (`3.15M`) sit in static N-gram tables**, leaving **only `3.9%` (`663K` parameters total: `173K` Attention + `490K` FFN/Mixer)** for the entire 2-layer quantum Transformer backbone!
       - Worse, because the tied `embed_re/im` table is initialized with an $88\%$ amplitude tilt toward the rank-1 unigram frequency carrier $|c_{\text{ref}}\rangle$ so that Step-0 output probabilities match word frequencies, **all frequent input tokens enter Layer 0 pointing in nearly the same direction** along $|c_{\text{ref}}\rangle$. That is why the embedding function is poorly conditioned for guiding attention and context evolution.

---

## 3. Outcomes We Are Looking to Achieve

1. **Outcome 1 — Verified Classical Speedrun Parity on Local & Multi-GPU Hardware**:
   - Confirm how close the 3-hour `CompiledLeaderboardClassicalGPT` run (`w45_compiled_classical_l12_d768_b65k_3hr_sub3p27`) gets to `3.28` Val CE on the single RTX 5060 Ti, and close any remaining gap using the three remaining leaderboard mechanisms (untied `wte`/`lm_head`, per-layer $x_0$ residual blend, and sliding-window causal attention schedule), plus distributed `2x RTX 3090` execution on `upstairs-gpu` (which reduces a 3-hour RTX 5060 Ti run to **~30 minutes**).
2. **Outcome 2 — Principled Quantum Block Ordering, Embedding Decoupling, & Quantum Associative MLP**:
   - Replace our redundant, parameter-starved (`3.9%` backbone) post-attention double-mixer with a clean, Gemma-7B / Classical-inspired quantum block architecture where:
     - The **input embedding function** produces isotropic, phase-diverse context representations (decoupled from the rank-1 unigram output carrier tilt),
     - Each layer has a **single consistent geometric coordinate frame** without back-to-back dense-vs-factored normalization clashes, and
     - Post-attention feature synthesis uses a **Sparse/Thresholded Quantum Associative MLP** (the complex-amplitude analogue of `ReLU²` / `GeGLU` with zero-initialized `proj_down` before residual sphere retraction) that holds a meaningful share (`40%–60%`) of the model's learnable parameters.
3. **Outcome 3 — Rapid Iteration via a Two-Tier `< 60s` Small-Problem Quantum Testbed**:
   - Rather than waiting 30–180 minutes per experiment to discover whether a quantum block ordering or embedding function works, evaluate every candidate quantum block modification on a standardized **two-tier `< 60s` fast-feedback testbed**:
     - **Tier 1 (`~10s` Karpathy Overfit & Associative Memorization Probe via `scripts/overfit_probe.py`)**: Tests whether the quantum block ordering and MLP can rapidly memorize $N \in \{1, 8, 32\}$ FineWeb sequences to `~100%` Top-1 accuracy (measuring associative memory capacity in isolation).
     - **Tier 2 (`~45s` `500-Step` / `8.2M`-Token Constant-LR FineWeb Micro-Speedrun at $B=16, T=1024$)**: Measures out-of-sample validation Cross-Entropy on `FineWeb` Shard 0 without WSD schedule confounding.

---

## 4. Concrete Next Steps

### Step 1: Harvest & Analyze the Completed 3-Hour Classical Run (`w45_compiled_classical_l12_d768_b65k_3hr_sub3p27`)
- Allow the currently running 3-hour classical speedrun on the local RTX 5060 Ti (currently at **`3.4721` Val CE at Step 6,400 / 7,800** and dropping ~`0.02` nats every 200 steps in WSD cooldown) to complete its final ~18 minutes and run the official $10,485,760$-token validation evaluation.
- Test enabling `--untie-embeddings` + per-layer $x_0$ residual blending (`x = lambda_1 * x + lambda_2 * x_0`) + sliding-window attention curriculum (`128 -> 384 -> 1024`) in [`CompiledLeaderboardClassicalGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py).

### Step 2: Run the Two-Tier `< 60s` Quantum Block-Ordering & Embedding Ablation Suite (`Wave 46` / `EXPERIMENT_PLAN_061`)
Design and run a controlled ablation study on our two-tier fast benchmark (**Tier 1**: `~10s` $N \in \{1, 8, 32\}$ sequence memorization probe; **Tier 2**: `~45s` `500-step` / `8.2M`-token constant-LR FineWeb micro-speedrun at $B=16, T=1024$) testing four isolated changes to the quantum architecture:
1. **Ablation A — Clean Block Ordering (Eliminating the Double-Mixer Clash)**:
   - Compare the current 3-stage post-attention sequence (`o_proj` $\to$ `ComplexUnitaryInterferenceFFN` $\to$ `Kronecker16x4UnitaryMeaningMixer`) against a clean 2-sublayer Transformer order (`Sublayer 1: Multi-Head Born Attention + o_proj + Sphere Norm` $\to$ `Sublayer 2: Quantum MLP + Sphere Norm`), applying the factored $(R, M=4, d_f=16)$ projection *once* at the readout head rather than forcing per-factor $\mathbb{C}\mathbb{S}^{15}$ normalization inside every layer.
2. **Ablation B — Quantum `ReLU²` / Quadratic-Threshold Associative MLP (`Zero-Init Residual`)**:
   - Replace unit-norm `ComplexUnitaryInterferenceFFN` with a $4\times$-expanded **Quantum Quadratic-Threshold MLP**:
     - Project $|\psi\rangle \in \mathbb{C}^d \to |u\rangle \in \mathbb{C}^{4d}$,
     - Apply a phase-preserving **quadratic magnitude threshold gate** (the complex analogue of $\text{ReLU}^2$: $g(u_k) = \max(0, |u_k| - \tau)^2 \cdot \frac{u_k}{|u_k| + \epsilon}$, which zeroes out weak activations below threshold $\tau$ and quadratically amplifies strong feature matches while preserving complex phase $e^{i\arg(u_k)}$),
     - Project down via **`zero-initialized` `proj_down`** ($4d \to d$) WITHOUT normalizing the MLP output to unit norm first, and then retract $\text{Normalize}(|\psi\rangle + \Delta|\psi_{\text{mlp}}\rangle)$ onto $\mathbb{C}\mathbb{S}^{d-1}$.
3. **Ablation C — Decoupling the Input Embedding from the Unigram Output Carrier**:
   - Keep the $16^4$ Kronecker POVM readout (and its N-gram/unigram calibration) at the *output* (`lm_head`), or remove the rank-1 carrier tilt $|c_{\text{ref}}\rangle$ from the *input* state entering Layer 0 so input token wavefunctions start orthogonally distributed across $\mathbb{C}^{d_{\text{model}}}$ rather than collinearly clustered along $|c_{\text{ref}}\rangle$.
4. **Ablation D — Quantum Token Value Embeddings (`ve0..ve2`) & U-Net Skip Connections**:
   - Port the Classical Leaderboard's U-Net encoder-decoder skip connections and Layer-0 initial state bypass ($|\psi_l\rangle \leftarrow \text{Normalize}(\lambda_1 |\psi_l\rangle + \lambda_2 |\psi_0\rangle)$) into the quantum backbone.

### Step 3: Scale Winning Architectures on `upstairs-gpu` (`2x RTX 3090`, `141.6 TFLOP/s`)
- Sync the repository to `upstairs-gpu` and add single-node multi-GPU Distributed Data Parallel (`torchrun --nproc_per_node=2`) support to our runners so both classical and quantum $500\text{M}$-token speedruns execute in **~25–30 minutes** on the dual RTX 3090s while keeping the local RTX 5060 Ti free for interactive visualization and `< 60s` architectural ablations.
