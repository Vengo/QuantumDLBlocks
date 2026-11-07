# Session Wrap-Up: Fiber-Bundle Wavepackets, Born-Schur Density Matrices, and Uncollapsed Multi-Mode Cholesky Language Modeling

**Date**: November 3, 2026  
**Project**: `QuantumDLBlocks` — Quantum-Inspired Deep Learning Building Blocks for Language Modeling  
**Hardware Environment**: Single Local Workstation GPU (`NVIDIA GeForce RTX 5060 Ti`, `16 GB` VRAM)  

---

## 1. Executive Summary & Project Context

This document provides a self-contained technical summary of our architectural research session in `QuantumDLBlocks`. It is written so that a reader with **no prior context of our conversation** and **no access to repository files** can understand the mathematical problems we set out to solve, the target outcomes, the exact equations and empirical results of every step we took, and our next research steps.

Across seven sequential experimental phases—progressing from token embedding geometry (`I1..I8`) and complex product-sphere tradeoffs (`I9..I14`) to `900M`-token statistical initialization (`I15..I17`, `S1..S4`), multi-branch superposition readouts (`M0..M4`), Born-Schur density-matrix attention (`D0..D4`), phase-invariant outer-product inputs (`W0..W3`), cone-free native Cholesky readouts (`D5..D6`), and finally **uncollapsed 4-mode Cholesky inter-layer highways (`D7..D9`)**—we systematically simplified the architecture while improving every empirical benchmark:

1. **On `50,304`-Vocabulary Sequence Reversal** (a diagnostic benchmark testing exact token identity preservation across causal attention layers), our **Fiber-Bundle Split Embedding + Born-Schur Density-Matrix Highway (`D3` / `W2`)** achieved **`100.00%` Top-1 validation accuracy** across both frequent and rare/unseen tokens, cutting Step-100 validation cross-entropy (CE) from `0.0981` nats (`D0` baseline) to **`0.0527` nats**, and Step-200 validation CE to **`0.0179` nats**.
2. **On Official `FineWeb` Autoregressive Language Modeling (`L = 4` layers, total real dimension $d = 512$, sequence length $T = 512$, batch size $B = 16$)** at the `300`-step (`2.46M`-token) screening horizon:
   - Eliminating a redundant `99,000`-parameter $R = 4$ Geodesic Cone readout head and reading out the 4 Cholesky modes directly from the final layer's density factor (`D5_NATIVE_CHOLESKY_DENSITY_NO_CONE`) improved **Pure Neural Validation CE** from `5.8592` (`D4`) and `5.8299` (`M1`) down to **`5.7528` (`Perplexity = 315.08`, `Top-1 = 20.95%`)**.
   - Propagating all **4 Cholesky modes $(L_{\text{re}}^{(\ell)}, L_{\text{im}}^{(\ell)}) \in \mathbb{C}^{B \times 8 \times T \times 4 \times 32}$ across all layers $\ell = 1 \to 2 \to 3 \to 4$ without collapsing** to a rank-1 wavevector between layers, combined with a joint Semantic (`448D`) + Identity (`64D`) Cholesky readout (`D8_UNCOLLAPSED_CHOLESKY_SEM_ID_READOUT`), dropped 300-step **Pure Neural Validation CE** another **`-0.0658` nats** to **`5.6870` (`Perplexity = 295.00`, `Top-1 = 21.00%`, `Top-5 = 37.32%`)**—beating our pure single-branch wavepacket baseline (`S2`, `5.9522`) by **`-0.2652` nats (`-89.62` PPL)**, beating our $n$-gram-augmented single-branch baseline (`S3 / M0`, `5.8400`) by **`-0.1530` nats (`-48.79` PPL)**, and surpassing the 4-layer classical Transformer baseline (`S1`, `5.7316`) by **`-0.0446` nats (`-13.45` PPL)**.
3. **On a Longer `1,200`-Step (`9.83M`-Token) `FineWeb` Scaling Run (`D9_UNCOLLAPSED_CHOLESKY_LONG_SCALING_1200`)**, **Pure Neural Validation CE** dropped monotonically by **`-1.9743` nats (`7.1549 -> 5.1806`, `Neural Perplexity: 1,280.36 -> 177.79`)**, while **Combined KN-Synergy Validation CE** (interpolating the neural Cholesky distribution with an unseen-shard-safe Kneser-Ney bigram prior built strictly from training Shards `1..9`) broke below `5.00` at Step 750 (`6.14M` tokens) and reached **`4.9162` (`Combined Perplexity = 136.48`, `Combined Top-1 = 26.12%`, `Combined Top-5 = 43.93%`)** at Step 1,200, with the model self-organizing to actively utilize **`3.04` out of `4` Cholesky eigenmodes**.

---

## 2. Glossary, Layer-Level Architecture, and Terminology

All mathematical objects, layer equations, tasks, evaluation modes, and optimizers referenced in this document are defined below:

- **Vocabulary & Tokenization ($V = 50,304$)**:
  Standard GPT-2 byte-pair encoding (BPE) vocabulary ($50,257$ real tokens padded with unused dummy indices to $V = 50,304 = 393 \times 128$ for GPU Tensor-Core tile alignment).
- **Training vs. Validation Data Hygiene (`FineWeb` Shards `1..9` vs. `Shard 0`)**:
  - **Training Corpus (`900,000,000` tokens)**: Stored in 9 binary shard files `fineweb_train_000001.bin`..`fineweb_train_000009.bin` ($100\text{M}$ tokens/shard from `kjj0/fineweb10B-gpt2`). Every statistical initialization (unigram frequencies, `CountSketch` co-occurrence matrices, Kneser-Ney bigram tables, ridge-regression alignment) and every training batch is drawn **exclusively** from Shards `1..9`.
  - **Validation Corpus (`Shard 0`)**: Stored in `fineweb_val_000000.bin`. Validation metrics are computed deterministically on the first `65,536` sequential tokens of `Shard 0` ($8$ batches of $B = 16 \times T = 512$). **Strict Invariant**: `Shard 0` is never read during initialization, calibration, or training.
- **Benchmark Task 1 — `50,304`-Vocabulary Span Reversal (`FineWebSpanReversalDataset`)**:
  - Each sequence consists of a span of real `FineWeb` tokens $[w_1, w_2, \dots, w_L]$ (span length $L = 31$ in Phase 1–3; $L = 8$ in Phase 5–6), followed by a separator token (`ID = 50,257`) and the reversed target span $[w_L, w_{L-1}, \dots, w_1]$. Cross-entropy loss and Top-1 accuracy are evaluated on the reversed target span across all $V = 50,304$ classes, partitioned into `512` frequent tokens and `512` rare/unseen tokens.
  - Because any token can appear at any position without semantic predictability from neighbors, this task tests whether the embedding geometry and attention highway preserve **exact lexical identity** across all $50,304$ tokens.
- **Benchmark Task 2 — `FineWeb` Autoregressive Language Modeling (`LongSeqFineWebStream`)**:
  - Causal next-token prediction over natural web text. In Phase 1–2 (`I2..I17`), a fast micro-screening setup ($L = 2$ layers, $T = 64$, $B = 32$, $300$ steps) compared 16 embedding initializers; from Phase 3 onward (`S1..S4`, `M0..M4`, `D4..D9`), we use the long-sequence setup ($L = 4$ layers, $d = 512$, $T = 512$, $B = 16$, $8,192$ tokens/step).
  - **Pure Neural Val CE (`neural_val_ce`)**: Cross-entropy loss (in nats, base $e$; $\text{Perplexity} = e^{\text{CE}}$) of the neural model's own probability distribution $P_{\text{neural}}(y \mid x_{\le t})$ on unseen `Shard 0`.
  - **Combined KN-Synergy Val CE (`val_ce` when `enable_kn_synergy_eval=True`)**: Cross-entropy of a fixed mixture $P_{\text{comb}}(y \mid x_{\le t}) = 0.45\, P_{\text{neural}}(y \mid x_{\le t}) + 0.55\, P_{\text{KN}}(y \mid x_t)$, where $P_{\text{KN}}(y \mid x_t)$ is an interpolated Kneser-Ney bigram/induction distribution precomputed strictly from training Shards `1..9`. Both `Pure Neural Val CE` and `Combined KN-Synergy Val CE` are explicitly labeled whenever KN synergy is enabled (`S4`, `M4`, `D9`).
- **Product-of-Spheres Wavepacket Embedding `(V, M, 2, d_c)`**:
  - Each token $v \in \{0, \dots, V-1\}$ is represented as $M$ complex wavepackets $\psi_{v, m} \in \mathbb{C}^{d_c}$ ($m \in \{0, \dots, M-1\}$), stored as a real tensor of shape `(V, M, 2, d_c)` where the `2` axis holds `(real, imag)` parts and $2 M d_c = 512$. Each packet is independently RMS-normalized onto its own sphere of radius $\sqrt{2 d_c}$ ($\|\psi_{v,m}\|_2^2 = 2 d_c$).
- **Full Per-Layer Forward Pass (`_ShallowQuantumWavepacketBlock` / `_NativeCholeskySchurWavepacketBlock`)**:
  Each layer $\ell \in \{1, \dots, L\}$ maps the incoming hidden state $x_t^{(\ell-1)} \in \mathbb{R}^{512}$ and the initial layer-0 anchor state $x_t^{(0)} \in \mathbb{R}^{512}$ to $x_t^{(\ell)} \in \mathbb{R}^{512}$ via two sublayers:
  1. **Anchor Residual Highway & Biorthogonal SVD-Stiefel $Q, K, V$ Projections**:
     - First, the block forms the anchored input $x_{\text{in}, t}^{(\ell)} = \lambda_0^{(\ell)} x_t^{(\ell-1)} + \lambda_1^{(\ell)} x_t^{(0)}$ (initialized at $\lambda_0 = 1.0, \lambda_1 = 0.15$) and normalizes $h_t^{(\ell)} = \text{RMSNorm}(x_{\text{in}, t}^{(\ell)}) \in \mathbb{R}^{512}$.
     - Every $512 \times 512$ linear projection ($Q, K, V$, observation projection $W_{\text{obs}}$, and pre-readout `exit_bridge`) is a **`BiorthogonalSVDStiefelProjection`** parameterized as $W = U \operatorname{diag}(\sigma) V^\top$, where $U, V \in \mathbb{R}^{512 \times 512}$ are maintained on the orthogonal Stiefel manifold ($U^\top U = V^\top V = I_{512}$) and $\sigma \in \mathbb{R}^{512}$ is a learned singular-value spectrum.
     - Applying $W_Q, W_K, W_V$ to $h_t^{(\ell)}$ yields raw query, key, and value tensors reshaped into $H = 8$ complex heads/packets of dimension $d_c = 32$ ($\mathbb{C}^{32} \cong \mathbb{R}^{64}$).
     - The value state blends the projected value with the gated layer-0 anchor: $|v_{t,h}^{(\ell)}\rangle = s_0^{(\ell)} |v_{\text{raw}, t,h}^{(\ell)}\rangle + s_1^{(\ell)}\big(2\sigma(W_{\text{ve}} h_{t, :16}^{(\ell)})_h \cdot |x_{t,h}^{(0)}\rangle\big)$.
     - Queries and keys are RMS-normalized per head and rotated by complex Rotary Position Embeddings (`RoPE`) to form $|q_{t,h}^{(\ell)}\rangle, |k_{t,h}^{(\ell)}\rangle \in \mathbb{C}^{32}$.
     - **Definition of the McWeeny Target Ket $|\text{tar}_{t,h}^{(\ell)}\rangle$**: The local target ket modulated by the density operator in Phase 5–7 is explicitly the sum of the rotary query and key kets at position $t$:
       $$|\text{tar}_{t,h}^{(\ell)}\rangle \;=\; |q_{t,h}^{(\ell)}\rangle + |k_{t,h}^{(\ell)}\rangle \;\in\; \mathbb{C}^{32}$$
     - After causal Born attention and density-matrix / Cholesky McWeeny modulation produce the head outputs $|\text{out}_{t,h}^{(\ell)}\rangle \in \mathbb{C}^{32}$ (flattened to $\text{out}_t^{(\ell)} \in \mathbb{R}^{512}$), the first sublayer updates the residual stream via a per-head sigmoid gate and Stiefel observation projection $W_{\text{obs}}^{(\ell)}$:
       $$x_{\text{mid}, t}^{(\ell)} \;=\; x_{\text{in}, t}^{(\ell)} + \gamma_{\text{obs}}^{(\ell)}\, W_{\text{obs}}^{(\ell)}\!\Big(2\sigma(W_{\text{attn\_gate}} h_{t, :16}^{(\ell)}) \odot \text{out}_t^{(\ell)}\Big)$$
  2. **Mach-Zehnder Interferometer MLP Sublayer (`SVDStiefelMachZehnderInterferometer`)**:
     - Instead of a standard unconstrained MLP, the second sublayer models a two-beam optical Mach-Zehnder interferometer over $h_{\text{mlp}, t}^{(\ell)} = \text{RMSNorm}(x_{\text{mid}, t}^{(\ell)}) \in \mathbb{R}^{512}$ with hidden dimension $d_{\text{ff}} = 1,536$:
       - Two biorthogonal Stiefel up-projections split the state into a signal beam $u_1 = W_{\text{up}, 1} h_{\text{mlp}, t}^{(\ell)} \in \mathbb{R}^{1536}$ and a phase/control beam $u_2 = W_{\text{up}, 2} h_{\text{mlp}, t}^{(\ell)} \in \mathbb{R}^{1536}$.
       - Constructive/destructive two-beam interference gates the squared-ReLU intensity via $\phi(u_1, u_2) = \operatorname{ReLU}(u_1)^2 \odot (2\sigma(u_2))$, which is projected back to $\mathbb{R}^{512}$ via Stiefel down-projection $W_{\text{down}} \in \mathbb{R}^{512 \times 1536}$:
         $$x_t^{(\ell)} \;=\; x_{\text{mid}, t}^{(\ell)} + \gamma_{\text{mz}}^{(\ell)}\, W_{\text{down}}\,\phi\!\big(W_{\text{up}, 1} h_{\text{mlp}, t}^{(\ell)},\, W_{\text{up}, 2} h_{\text{mlp}, t}^{(\ell)}\big)$$
- **Density Matrix ($\rho$), Low-Rank Cholesky Factor ($L$), Schur Product ($\odot$), McWeeny Purification, and POVM Gate**:
  - **Density Matrix & Cholesky Factor**: A rank-$k$ mixed state $\rho = \sum_{r=0}^{k-1} p_r |\psi_r\rangle\langle\psi_r| \in \mathbb{C}^{d_c \times d_c}$ ($\rho^\dagger = \rho \succeq 0, \operatorname{Tr}(\rho) = 1$) is factored as $\rho = L^\dagger L$, where $L \in \mathbb{C}^{k \times d_c}$ ($k = 4$) has rows $l_r = \sqrt{p_r}\langle\psi_r|$ with mode energy $\|l_r\|_2^2 = p_r$.
  - **Schur Product ($\odot$)**: Entrywise product $(A \odot B)_{ij} = A_{ij} B_{ij}$. By the Schur Product Theorem, $\rho_a, \rho_b \succeq 0 \implies \rho_a \odot \rho_b \succeq 0$ (acting as a cross-packet coincidence detector).
  - **McWeeny Purification**: $\rho \leftarrow \rho^2 / \operatorname{Tr}(\rho^2)$, which squares eigenvalues ($\lambda_r \mapsto \lambda_r^2 / \sum_j \lambda_j^2$). For $\rho = L^\dagger L$, $\rho^2 = L^\dagger G L$ where $G = L L^\dagger \in \mathbb{C}^{4 \times 4}$ is the $4 \times 4$ Gram matrix and $\operatorname{Tr}(\rho^2) = \|G\|_F^2$.
  - **POVM Born-Trace Gate**: For unit vector $|m\rangle \in \mathbb{C}^{d_c}$, $\operatorname{Tr}(\rho |m\rangle\langle m|) = \|L |m\rangle\|_2^2 \in [0, 1]$.
- **Riemannian Manifold Optimizers & Learning-Rate Schedule (`lr_mult`)**:
  - `ModularRiemannianStiefelKrausMuon` ($\text{lr} = 0.028$) updates Stiefel matrices ($U, V$ in $Q, K, V, W_{\text{obs}}, \text{exit\_bridge}, \text{Mach-Zehnder MLP}$) and unitary matrices $U_{\text{evol}} \in U(32)^8$ via Riemannian tangent projection + Newton-Schulz retraction.
  - `BatchSubspaceTikhonovGramMuon` updates product-sphere embeddings `wte_3d` ($\text{lr} = 0.045$) and readout detectors `w_det_512` ($\text{lr} = 0.038$) via Tikhonov-regularized $64 \times 64$ packet Gram preconditioning $(G + \lambda I)^{-1/2}$ + sphere retraction.
  - `AdamW` ($\text{lr} = 0.004$; temperatures at $\text{lr} = 0.06$) updates scalar gates and singular value spectra $\sigma$.
  - **Cosine Schedule (`lr_mult`)**: Linear warmup over the first $8\%$ of total `steps` ($0.20 \to 1.00$), followed by cosine annealing to `lr_mult = 0.12` at `step == steps`.

---

## 3. The Core Problems We Have Been Solving & Discussing

### Problem 1: Semantic Similarity vs. Lexical Identity Collision on a Single Sphere
In standard single-sphere embeddings, each token $v$ has one vector $e_v \in \mathbb{R}^d$. Training on natural text pulls synonyms and co-occurring words toward high cosine overlap ($\langle e_u, e_v\rangle \to 1$) so trunk layers can share representations across related concepts. However, at the output readout, high cosine overlap between distinct tokens causes **lexical crowding**: high-frequency words overshadow rare synonyms, and exact token identification tasks (`50,304`-vocab reversal) fail on tail tokens.

### Problem 2: Choosing the Complex Product-Sphere Geometry `(50304, M, 2, d_c)`
To interpret embeddings as multi-packet complex wavefunctions while decoupling meaning from lexical identity, we structured the embedding table as `(50304, M, 2, d_c)` with $2 M d_c = 512$ real dimensions, partitioning the $M$ complex packets into $M_{\text{sem}}$ **Semantic Packets** and $M_{\text{id}} = M - M_{\text{sem}}$ **Identity Packets**:
- If $M$ is too small (`(50304, 4, 2, 64)`), each complex packet has $2 d_c = 128$ real dimensions, so dedicating even $1$ packet to identity sacrifices $25\%$ ($128 / 512$) of semantic capacity.
- If $M$ is too large (`(50304, 16, 2, 16)`), each packet has only $2 d_c = 32$ real dimensions, raising the random pairwise cosine noise floor ($\text{RMS}(\cos\theta) = 1/\sqrt{32} \approx 0.177$) and limiting per-packet expressivity.

### Problem 3: Data-Driven Corpus Initialization Without Validation Leakage
Random Gaussian initialization (`I1`) starts a $50,304$-vocab model at uniform cross-entropy $\ln(50,304) = \mathbf{10.8258\text{ nats}}$. We wanted to determine how much zero-shot structure we could pack into the semantic packets at Step 0 by streaming the `100M` tokens of Shard 1 or all `900M` tokens of Shards `1..9` (strictly excluding validation `Shard 0`) using multi-horizon co-occurrence statistics and closed-form linear algebra.

### Problem 4: The $K_{\max} \approx 2.96$ Superposition Capacity Ceiling & Orthogonal Branch Dilution
In `64D` real identity space ($1$ complex packet of $\mathbb{C}^{32}$), random unit vectors have root-mean-square pairwise cosine $1/\sqrt{64} = 0.125$ and extreme-value crosstalk $\approx \sqrt{2\ln V / d_{\text{id}}}$ across $V = 50,304$ tokens.
- If an ambiguous context represents a superposition of $K$ plausible next tokens with a single unit query vector $h_{\text{id}} = \frac{1}{\sqrt{K}}\sum_{k=1}^K w_{\text{id}, v_k}$, its signal overlap with each true target drops to $1/\sqrt{K}$. Equating signal to the extreme-value noise floor gives the **Johnson-Lindenstrauss Superposition Capacity Ceiling**:
  $$K_{\max}(d_{\text{id}}, V) \;=\; \frac{d_{\text{id}}}{2\ln V} \;=\; \frac{64}{2\ln(50,304)} \;\approx\; \mathbf{2.956\text{ simultaneous tokens}}$$
- Thus, a single `64D` identity query vector cannot cleanly address more than $\sim 3$ candidate tokens at once.
- Yet naively introducing $R = 4$ **strictly orthogonal ($90^\circ$)** readout branches hurts unambiguous ($K = 1$) predictions because $3$ of the $4$ branches point $90^\circ$ away from the target token and contribute near-zero probability, costing a $-\ln \pi_0$ mixture-weight dilution penalty.

### Problem 5: Global Phase Gauge Collisions & Norm Drift in Wavevector Attention vs. Density-Matrix Schur Algebra
When attention aggregates value wavepackets as a linear sum of vectors $\sum_{j \le t} w_{tj} |v_{j,h}\rangle \in \mathbb{C}^{32}$:
1. **Arbitrary Global Phase Collisions**: Multiplying token $j$'s wavevector by a global phase $e^{i\theta}$ represents the same physical quantum state, but flips $|v_1\rangle + e^{i\theta}|v_2\rangle$ between constructive addition and destructive cancellation.
2. **Non-Linear Norm Drift**: A weighted sum of unit vectors does not have unit norm, requiring $\ell_2$ renormalization.
- Lifting wavevectors to **outer-product density matrices** $\rho_{j,h} = |v_{j,h}\rangle\langle v_{j,h}| = (e^{i\theta}|v_{j,h}\rangle)(\langle v_{j,h}|e^{-i\theta})$ cancels global phase noise identically, preserves exact unit trace $\operatorname{Tr}(\bar{\rho}_{t,h}) = 1$ under Born attention weights $P_{tj}^{(h)} \ge 0$, and enables **cross-packet Schur-product ($\bar{\rho}_{t,a} \odot \bar{\rho}_{t,b}$) coincidence binding**.

### Problem 6: Eliminating the Rank-4 $\to$ Rank-1 $\to$ Rank-4 Bottleneck & Passing Uncollapsed Cholesky Modes Across Layers
Inspecting our combined Born-Schur + Geodesic Cone model (`D4_BORN_SCHUR_POVM_CONE_R4`) revealed a structural redundancy:
- Inside every attention block, the model already constructed a **rank-4 Cholesky density factor** $L_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$, **collapsed** it to a rank-1 wavevector at the end of the block, and then used a `99,000`-parameter $R = 4$ Geodesic Cone readout head (`mb_head`) at the final layer to **re-inflate** the rank-1 vector back into 4 branches.
- This motivated our next two questions:
  1. Can we remove the cone readout head (`mb_head = None`) and read out the 4 branches directly from the density factor's own 4 Cholesky modes (`D5, D6`)?
  2. Can we pass all **4 Cholesky modes $L_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$ across all layers $\ell = 1 \to 2 \to 3 \to 4$ without collapsing** (`D7, D8, D9`) while keeping their row phases and packet bases aligned?

### Problem 7: Forced Unit-Norm Readouts on $\mathbb{S}^{d-1}$ Cannot Say "I Don't Know" (The Need for an Orthogonal Vacuum Subspace $\mathcal{H}_{\text{vac}}$ & Pure-Neural Step-0 Calibration)
When a readout branch or Cholesky mode is strictly `RMSNorm`-normalized inside the `64D` vocabulary identity subspace $\mathbb{S}^{63}$, **it is forced to point somewhere on the vocabulary sphere with full unit length**, even when the context is ambiguous or a Cholesky mode ($r \in \{1, 2, 3\}$) carries near-zero energy:
- In `64D` identity space with $V = 50,304$ tokens, pointing in a random unit direction produces spurious extreme-value cosine overlaps up to $\sqrt{2\ln(50,304)/64} \approx 0.58$.
- At Step 0, `SemanticSubspaceNGramProjector` encodes context certainty into the pre-normalized vector norm $\|\text{readout\_id\_64}\|_2$ (large on deterministic contexts like `"United"` $\to$ `"States"`, near zero on high-entropy contexts like `"the"` $\to$ `?`). Wrapping that vector in `F.rms_norm(..., (64,))` erases the norm/confidence signal and amplifies uncertain `64D` vectors back to full unit norm—forcing the Step-0 calibration to pick a tiny global `temp_id_scale` (`0.6–0.8`) and leaving Pure Neural Step-0 CE at `7.1549` (which previously motivated an inelegant external `0.45 Neural + 0.55 KN` weighted average).
- Crucially, adding an extra $(V+1)$-th "null token" class to `softmax` hurts Cross-Entropy (any probability mass assigned to $V+1$ directly penalizes $-\ln P(y_{\text{true}})$). By contrast, embedding the readout sphere in $\mathbb{S}^{(d + d_{\text{vac}}) - 1} = \mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$—where all $V = 50,304$ token detectors have **identically zero support** on the $d_{\text{vac}}$-dimensional **Orthogonal Vacuum Subspace $\mathcal{H}_{\text{vac}}$** ($\langle w_v \mid h^{\text{vac}}\rangle \equiv 0$)—gives every readout branch and Cholesky mode a geometric **confidence/abstention angle** $\cos\theta_{\text{vac}} \in [0, 1]$ while staying strictly on the unit sphere:
  $$|h_{t,r}\rangle \;=\; \cos\theta_{\text{vac}}(t,r)\, |h_{t,r}^{\text{vocab}}\rangle \;+\; \sin\theta_{\text{vac}}(t,r)\, |h_{t,r}^{\text{vac}}\rangle, \qquad z_{t,r,v} \;=\; \big(\tau\cos\theta_{\text{vac}}(t,r)\big)\,\langle w_v \mid h_{t,r}^{\text{vocab}}\rangle$$

---

## 4. Outcomes We Are Looking to Achieve

1. **A Unified, Parameter-Lean Quantum Density-Matrix Architecture**:
   - Store token embeddings as `(50304, 8, 2, 32)` product-sphere wavefunctions split into a **Semantic Base Space** (`Packets 0..6`, `448D` real) and an **Identity Fiber** (`Packet 7`, `64D` real).
   - Propagate **uncollapsed rank-4 Cholesky density factors $L_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$** across all layers using Born attention, $8 \times 8$ cross-packet Schur coincidence products, $U(1)^4$ phase-locked POVM highways, and $4 \times 4$ Gram McWeeny purification, reading out next-token probabilities directly from the final Cholesky modes with zero redundant cone heads.
2. **`100.00%` Exact Token Discrimination on `50,304`-Vocab Reversal (`Val CE < 0.02` nats)**.
3. **Monotonic `FineWeb` Validation Improvements Breaking Below `5.0` (and Toward `< 4.0`) Val CE on Unseen `Shard 0` on a Single `16 GB` GPU**.

---

## 5. Chronological Steps Taken, Mathematical Formulations, and Empirical Results

### Phase 1: Support-Preserving CountSketch & Fiber-Bundle Split (`I1..I8`)
- **Code**: `src/qdlblocks/benchmarks/support_preserving_countsketch_lab.py`
- **Context on `I1` vs. `I2..I8`**:
  - `I1` denotes the uncalibrated zero-bias random Gaussian sphere baseline (`Step-0 Val CE = \ln(50,304) = 10.8258` nats). Starting with `I2_UNIGRAM_PRIOR_RANDOM_SPHERE`, all variants (`I2..I17`) include the empirical training-shard unigram log-prior bias $b_{\text{unigram}, v} = \ln P_{\text{unigram}}(v)$ at readout (`Step-0 Val CE \approx 7.99` nats) so all 16 initializers compare embedding vector geometry on equal footing.
- **Mathematical Formulation**:
  - **PPMI CountSketch (`I3`)**: Stream training tokens to estimate bigram co-occurrence counts $C_{u,v}$ and Positive Pointwise Mutual Information $\text{PPMI}(u,v) = \max\!\big(0, \ln \frac{C_{u,v} N}{c_u c_v}\big)$, projected into packet dimension via signed sparse random hash matrices.
  - **Support-Preserving Iterative Tangent-Space Gram-Schmidt (`I5..I8`)**: Raw `CountSketch` (`I3`) collapses rare/unseen tokens onto a shared mean vector (`rare_max_cos = 0.9997`). To fix this without destroying semantic correlations among frequent tokens, we center only active vocabulary rows ($c_v > 0$), optionally apply ZCA covariance whitening $C^{-1/2}$ (`I7`), and blend each token's semantic sketch vector $\hat{s}_v$ with a strictly orthogonalized identity vector $\hat{z}_v \perp \hat{s}_v$ via an empirical-Bayes count gate:
    $$\psi_v \;=\; \alpha_v\, \hat{s}_v + \sqrt{1 - \alpha_v^2}\, \hat{z}_v, \qquad \alpha_v \;=\; \alpha_{\max}\sqrt{\frac{c_v}{c_v + k_0}} \quad (k_0 = 32.0)$$
    so unseen/rare tokens ($c_v \ll k_0$) automatically receive $\alpha_v \to 0$ (pure orthogonal identity $\hat{z}_v$), while frequent tokens receive rich co-occurrence semantics.
  - **Fiber-Bundle Packet Split (`I6`) & Adaptive Packet Gate (`I8`)**: Instead of blending meaning and identity inside every packet (`I5`), `I6` and `I8` partition the $M = 8$ packets into dedicated **Semantic Packets** (`0..5`) and dedicated **Identity Packets** (`6..7`, orthogonalized via iterative Gram-Schmidt on the sphere so `rare_max_cos = 0.2982`).
  - **Why `I8` Jumps in `Semantic Gram Corr` (`0.5311 -> 0.8301`)**:
    - **Definition of `Semantic Gram Corr`**: The Pearson correlation coefficient $\operatorname{Corr}\big(G_{\text{emb}}^{\text{off-diag}}, G_{\text{PPMI}}^{\text{off-diag}}\big)$ between the upper-triangle off-diagonal pairwise cosines $G_{\text{emb}}(u,v) = \frac{\langle w_{\text{sem}, u}, w_{\text{sem}, v}\rangle}{\|w_{\text{sem}, u}\|_2 \|w_{\text{sem}, v}\|_2}$ of the initialized semantic packets and the exact empirical PPMI cosine similarity matrix $G_{\text{PPMI}}(u,v)$ over the top `512` frequent tokens.
    - In `I6`, all 6 semantic packets used single-horizon random hash projections with equal weighting (`1.0`). In `I8`, each packet $m \in \{0, \dots, M-1\}$ is scaled at readout by a learned sigmoid gate $g_m = 2\sigma(\theta_m)$ (initialized higher on semantic packets for LM and higher on identity packets for reversal) combined with multi-band variance normalization across packets, raising `Semantic Gram Corr` from `0.5311` to `0.8301`.

#### Table 1A: Phase 1 (`I2..I8`) `50,304`-Vocab Span Reversal & Micro-LM Screening ($d = 256, 8 \times 32, 300\text{ Steps}$)

| Variant | Embedding Initialization Strategy | Rare Max $\cos\theta$ | Semantic Gram Corr | Reversal Step 300 Val CE | Reversal Top-1 Acc | Micro-LM ($T=64$) Val CE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `I2_UNIGRAM_PRIOR_RANDOM_SPHERE` | Random Gaussian spheres + unigram bias | `0.2692` | `-0.0076` | `0.0181` | `100.00%` | `7.1380` |
| `I3_RAW_COUNTSKETCH_SPHERE` | Raw PPMI CountSketch (uncentered tail collapse) | `0.9997` | `0.9883` | `0.0860` | `99.07%` | `6.8060` |
| `I4_LOWRANK_SVD_SPHERE` | Top-$d$ truncated SVD of bigram PPMI | `0.2942` | `0.5463` | `0.1043` | `97.98%` | `6.8515` |
| `I5_SUPPORT_GATED_COUNTSKETCH` | Active-only centering + Bayes count gate $\alpha_v$ | `0.2982` | `0.5368` | `0.0139` | `100.00%` | `6.9592` |
| `I6_SPLIT_MEANING_ID_WAVEPACKET` | Split packets (`0..5` Semantic, `6..7` Identity) | `0.2982` | `0.5311` | `0.0141` | `100.00%` | `6.9775` |
| `I7_WHITENED_SUPPORT_COUNTSKETCH` | ZCA covariance whitening $C^{-1/2}$ + Bayes gate | `0.2982` | `0.4841` | `0.0168` | `100.00%` | `6.9728` |
| **`I8_SPLIT_MEANING_ID_ADAPTIVE_PKT_GATE`** | **Split Semantic/Identity packets + Adaptive Gate $g_m = 2\sigma(\theta_m)$** | **`0.2982`** | **`0.8301`** | **`0.0132`** | **`100.00%`** | **`6.9098`** |

---

### Phase 2: Complex Product-Sphere Shape Tradeoff (`I9..I14`, `(50304, M, 2, d_c)`)
- **Code**: `src/qdlblocks/benchmarks/support_preserving_countsketch_lab.py`, `src/qdlblocks/benchmarks/shallow_longseq_wavepacket_lm_lab.py`
- **What We Evaluated**:
  - Tested 4D complex wavefunction tables `(50304, M, 2, d_c)` across $M \in \{4, 8, 16\}$ and $d_c \in \{16, 32, 64\}$, varying the number of complex identity packets (`ID1`, `ID2`, `ID4`).
  - **How Semantic vs. Identity Subspaces Are Handled Differently Throughout Training**:
    1. **Initialization**: Semantic packets (`0 .. M - M_id - 1`) receive multi-horizon PPMI `CountSketch` + tangent Gram-Schmidt; Identity packets (`M - M_id .. M - 1`) receive strictly orthogonalized token IDs.
    2. **Causal Wavepacket Dispersion (Semantic-Only)**: At the trunk input, causal complex dispersion (`apply_causal_wavepacket_dispersion`, rotating packet $m$ of token $t-1$ by $\cos(\theta_m) + i\sin(\theta_m)$ and adding it to token $t$) and gated temporal smearing ($x_{t,\text{sem}} \leftarrow x_{t,\text{sem}} + \lambda_{\text{smear}}\sigma(W_{\text{smear}} x_{t, :16}) x_{t-1,\text{sem}}$) act **strictly on the semantic packets (`0..6`, `448D`)**. The identity packet's dispersion angle is clamped to `0.0` after every optimizer step (`clamp_identity_dispersion_and_scalars_`) so token $t$'s identity fiber is never blurred by neighboring tokens.
    3. **Factored Readout Scaling**: Readout logits sum a temperature-scaled semantic cosine inner product ($\frac{\tau_{\text{sem}}}{448}\langle h_{\text{sem}}, w_{\text{sem}, v}\rangle$) and a temperature-scaled identity cosine inner product ($\frac{\tau_{\text{id}}}{64}\langle h_{\text{id}}, w_{\text{id}, v}\rangle$).

#### Table 2A: Phase 2 (`I9..I14`) Complex Wavefunction Shape Tradeoff (`50,304` Vocab, $300$ Steps)

| Variant | Tensor Shape `(V, M, 2, d_c)` | Real $d$ | Semantic / ID Split (Real Dim) | Rare Max $\cos\theta$ | Sem Gram Corr | Reversal Step 300 CE (`Top-1`) | Micro-LM ($T=64$) Val CE (`Top-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `I9_COMPLEX_WAVEFUNCTION_4x2x32` | `(50304, 4, 2, 32)` | `256` | `3 Sem (192D) / 1 ID (64D)` | `0.2660` | `0.8393` | `0.0113` (`100.0%`) | `6.8996` (`12.91%`) |
| `I10_COMPLEX_WAVEFUNCTION_4x2x64` | `(50304, 4, 2, 64)` | `512` | `3 Sem (384D) / 1 ID (128D)` | `0.1906` | `0.8872` | `0.0087` (`100.0%`) | `6.8446` (`13.36%`) |
| `I11_COMPLEX_WAVEFUNCTION_8x2x32_ID2` | `(50304, 8, 2, 32)` | `512` | `6 Sem (384D) / 2 ID (128D)` | `0.1947` | `0.8465` | `0.0097` (`100.0%`) | `6.8647` (`13.04%`) |
| **`I12_COMPLEX_WAVEFUNCTION_8x2x32_ID1`** | **`(50304, 8, 2, 32)`** | **`512`** | **`7 Sem (448D) / 1 ID (64D)`** | **`0.1947`** | **`0.9151`** | **`0.0097` (`100.0%`)** | **`6.8536` (`13.39%`)** |
| `I13_COMPLEX_WAVEFUNCTION_16x2x16_ID4` | `(50304, 16, 2, 16)` | `512` | `12 Sem (384D) / 4 ID (128D)` | `0.1934` | `0.8763` | `0.0094` (`100.0%`) | `6.8796` (`13.41%`) |
| `I14_COMPLEX_WAVEFUNCTION_16x2x16_ID2` | `(50304, 16, 2, 16)` | `512` | `14 Sem (448D) / 2 ID (64D)` | `0.1934` | `0.9270` | `0.0103` (`100.0%`) | `6.8749` (`13.17%`) |

- **Why `I12` (`(50304, 8, 2, 32)` with `7` Semantic + `1` Identity Packet = `448D + 64D`) Was Selected**:
  - Compared to `(4, 2, 64)` (`I10`), `(8, 2, 32)` (`I12`) needs only `64D` for its 1 identity packet (preserving `448D` vs. `384D` for semantics and raising semantic Gram correlation from `0.8872` to `0.9151`) while matching `100.0%` reversal accuracy and `13.39%` LM Top-1 accuracy.
  - Compared to `(16, 2, 16)` (`I13, I14`), `(8, 2, 32)` has twice the complex Hilbert dimension per packet ($\mathbb{C}^{32}$ vs. $\mathbb{C}^{16}$) and maps cleanly onto $H = 8$ attention heads (`1` complex packet per head).

---

### Phase 3: `900M`-Token Multi-Horizon Semantic Initialization & Shallow Long-Seq Baselines (`I15..I17`, `S1..S4`)
- **Code**: `src/qdlblocks/benchmarks/support_preserving_countsketch_lab.py`, `src/qdlblocks/benchmarks/shallow_longseq_wavepacket_lm_lab.py`
- **Mathematical Formulation of Phase 3 Initializers & Modules**:
  1. **Multi-Lag (`[1, 2, 4, 8]`) Bilateral PPMI Sketch over Shard 1 (`100M` tokens, `I15`) and Shards `1..9` (`900M` tokens, `I17`)**:
     - Streams `900M` training tokens on GPU in `3.83` seconds (`50,039 / 50,304 = 99.57%` active vocabulary coverage), accumulating weighted co-occurrence statistics at token offsets $\Delta \in \{1, 2, 4, 8\}$ into the 7 semantic packets (`448D`).
  2. **Step-0 Closed-Form Next-Token Ridge Alignment (`I16, I17`)**:
     - Over the active vocabulary tokens, let $S \in \mathbb{R}^{V_{\text{act}} \times 448}$ be the normalized semantic embedding matrix of context tokens $u$, and let $P_{\text{next}} \in \mathbb{R}^{V_{\text{act}} \times 448}$ be the empirical conditional expectation of the next token's semantic vector: $(P_{\text{next}})_u = \sum_{v} P_{\text{bigram}}(v \mid u)\, S_v$.
     - At Step 0, we solve the $448 \times 448$ Tikhonov-regularized system $W_{\text{ridge}} = (S^\top D_p S + \lambda_{\text{ridge}} I_{448})^{-1} (S^\top D_p P_{\text{next}})$ ($\lambda_{\text{ridge}} = 0.08$, where $D_p = \operatorname{diag}(P_{\text{unigram}})$) and align the semantic slice via $S \leftarrow \text{SphereNorm}((1 - \eta) S + \eta\, S W_{\text{ridge}})$ so that $\langle w_{\text{sem}, x_t}, w_{\text{det}, \text{sem}, v}\rangle$ immediately predicts empirical next-token transitions at Step 0 (`7.9899 -> 7.3498` Step-0 CE in Table 3A).
  3. **`SemanticSubspaceNGramProjector` (Introduced in `S3` and Retained in `S4, M0..M4, D4..D9`)**:
     - For each token position $t$, causal bigrams $h_{\text{bi}}(t) = (100003 x_{t-1} + x_t) \bmod B_{\text{hash}}$, trigrams $h_{\text{tri}}(t) = (10000019 x_{t-2} + 100003 x_{t-1} + x_t) \bmod B_{\text{hash}}$, and skip-bigrams $h_{\text{skip}}(t) = (100003 x_{t-2} + x_t) \bmod B_{\text{hash}}$ ($B_{\text{hash}} = 65,536$) look up vectors in learned hash tables:
       - An input rank-$32$ table $E_{\text{in}} \in \mathbb{R}^{B_{\text{hash}} \times 32}$ projected into the `448D` semantic input via semi-orthogonal matrix $U_{\text{sem}} \in \mathbb{R}^{32 \times 448}$: $\Delta x_{t,\text{sem}}^{(0)} = \gamma_{\text{in}}\big(E_{\text{in}}[h_{\text{bi}}(t)]\, U_{\text{sem}}\big)$ (padded with $64$ zeros on the identity slice so Packet 7 is untouched).
       - Direct readout $n$-gram tables producing `readout_sem_448` $\in \mathbb{R}^{448}$ and `readout_id_64` $\in \mathbb{R}^{64}$ (initialized at Step 0 from empirical Shard-1 next-token semantic/identity averages per hash bucket), which blend with the trunk output via learned gates `trunk_sem_gate` and `trunk_id_gate`:
         $$h_{\text{sem}, 0} = \text{RMSNorm}\big(g_{\text{sem}}\, h_{\text{sem}, \text{trunk}} + \text{readout\_sem\_448}\big), \qquad h_{\text{id}, 0} = \text{RMSNorm}\big(g_{\text{id}}\, h_{\text{id}, \text{trunk}} + \text{readout\_id\_64}\big)$$
  4. **Step-0 Post-Trunk Ridge-Procrustes Calibration of `exit_bridge` (`S2..S4`, `M0..M4`, `D4..D9`)**:
     - In `initialize_and_calibrate_shallow_model_`, before Step 1 we run `16` calibration batches (`131,072` tokens from Shard 1) forward through the 4-layer trunk (`no_grad`) to collect pre-bridge hidden states $H \in \mathbb{R}^{131072 \times 512}$ and target detector embeddings $Y = W_{\text{det}}[y] \in \mathbb{R}^{131072 \times 512}$.
     - We solve $W_* = (\frac{1}{N}H^\top H + 0.08 I_{512})^{-1} (\frac{1}{N}H^\top Y) \in \mathbb{R}^{512 \times 512}$, take its singular value decomposition $W_*^\top = U \Sigma V^\top$, and initialize `exit_bridge` (`u_stiefel = U, v_stiefel = V, sigma = \Sigma / \text{RMS}(\Sigma)`).

#### Table 3A: Phase 3 Step-0 & 300-Step Scaling from `5M` Tokens (`I12`) $\to$ `100M` Tokens (`I15, I16`) $\to$ `900M` Tokens (`I17`)

| Variant | Corpus Streamed | Zero-Step Alignment | Micro-LM Step 0 Val CE (`Top-1`) | Micro-LM Step 300 Val CE (`Top-1`) |
| :--- | :---: | :---: | :---: | :---: |
| `I12_COMPLEX_WAVEFUNCTION_8x2x32_ID1` | `5M` tokens | Unigram bias only | `7.9899` (`2.82%`) | `6.8536` (`13.39%`) |
| `I15_SHARD1_100M_MULTILAG_SKETCH` | Shard 1 (`100M` tokens) | Unigram bias only | `8.1149` (`2.48%`) | `6.7255` (`13.79%`) |
| `I16_SHARD1_100M_STEP0_CALIBRATED` | Shard 1 (`100M` tokens) | + Closed-form Ridge Alignment | `7.3702` (`11.46%`) | `6.6759` (`14.19%`) |
| **`I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED`** | **Shards `1..9` (`900M` tokens)** | **+ Closed-form Ridge Alignment** | **`7.3498` (`11.50%`)** | **`6.6615` (`14.32%`)** |

#### Table 3B: Transition to Long-Sequence FineWeb (`L = 4` Layers, $d = 512$, $T = 512$, $B = 16$, $300$ Steps = $2.46\text{M}$ Tokens, `S1..S4`)

| Variant | Architecture Description (`L = 4, d = 512, T = 512`) | Step 0 Pure Neural CE | Step 100 Pure Neural CE | Step 200 Pure Neural CE | **Step 300 Pure Neural CE (`PPL`)** | Step 300 Combined KN CE (`PPL`) | Top-1 Acc | Top-5 Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `S1_CLASSICAL_4L_BASELINE` | Standard 4-layer Transformer (RoPE + SwiGLU + softmax) | `7.3549` | `6.2891` | `5.9102` | `5.7316` (`308.45`) | — | `20.61%` | `36.01%` |
| `S2_QUANTUM_WAVEPACKET_4L_PURE` | 4-layer `(8, 2, 32)` wavepacket trunk (`I17` init, no `sem_ngram`) | `7.1420` | `6.4794` | `6.1328` | `5.9522` (`384.62`) | — | `18.03%` | `33.70%` |
| `S3_QUANTUM_WAVEPACKET_4L_SEMANTIC_NGRAM` (`= M0`) | `S2` + `SemanticSubspaceNGramProjector` (`448D` hash lift) | `7.1634` | `6.3300` | `6.0167` | `5.8400` (`343.79`) | — | `18.92%` | `35.37%` |
| `S4_QUANTUM_WAVEPACKET_4L_CALIBRATED_SYNERGY` | `S3` + Shard `1..9` KN bigram synergy eval (`0.45 Neural + 0.55 KN`) | `7.1634` | `6.3298` | `6.0163` | `5.8398` (`343.71`) | `5.1461` (`171.76`) | `24.58%`* | `41.94%`* |

*(\*In `S4`, Top-1/Top-5 are evaluated on the Combined KN-Synergy distribution; in `S1..S3`, they are Pure Neural. Note also that in `MultiBranchWavepacketGPT`, `M0_SINGLE_BRANCH_BASELINE` with `num_branches = 1` delegates directly to `S3_QUANTUM_WAVEPACKET_4L_SEMANTIC_NGRAM`, so `S3` and `M0` are the exact same `5.8400` Val CE single-branch model.)*

---

### Phase 4: Superposition Capacity Verification & Multi-Branch Geodesic Cone Readout (`M0..M4`)
- **Code**: `src/qdlblocks/benchmarks/multibranch_wavepacket_readout_lab.py`
- **Synthetic $K$-Target Superposition Probe Setup (`run_superposition_capacity_probe`, Table 4A)**:
  - We construct a dictionary of $N_{\text{pool}} = 4,096$ unit-norm identity vectors $W_{\text{id}} \in \mathbb{R}^{4096 \times 64}$ (taken from Packet 7 of `I17`, with inverse temperature $\tau_{\text{id}} = 10.5$).
  - For each target multiplicity $K \in \{1, 2, 4, 8, 16\}$ across $32$ random trials, we sample $K$ equiprobable target tokens $\{v_1, \dots, v_K\}$ and partition them evenly across $R \in \{1, 4, 8\}$ unit-norm readout branch vectors $h_r = \text{Normalize}\!\big(\sum_{k \in \mathcal{B}_r} w_{\text{id}, v_k}\big) \in \mathbb{S}^{63}$.
  - Each branch computes log-probabilities $\ln P_r(v) = \text{log\_softmax}_{v'}(\tau_{\text{id}}\langle h_r, w_{\text{id}, v'}\rangle)_v$, combined via uniform mixture $\ln P_{\text{mix}}(v) = \operatorname{logsumexp}_{r=0}^{R-1}(\ln P_r(v) - \ln R)$.
  - We report (a) the mean cross-entropy $-\frac{1}{K}\sum_{k=1}^K \ln P_{\text{mix}}(v_k)$ vs. the theoretical Shannon entropy floor $\ln K$, and (b) **`Top-K Recall`**: the fraction of the $K$ true targets $\{v_1, \dots, v_K\}$ that appear among the top-$K$ highest-probability indices of $\ln P_{\text{mix}}$.
- **Mathematical Formulation of `MultiBranchWavepacketReadoutHead` (`mb_head`) and `_FactoredMultiBranchWavepacketCEFn`**:
  - Given final-layer trunk state $h_{\text{pre}} = \text{RMSNorm}(x^{(L)}) \in \mathbb{R}^{512}$ and mid-layer state $h_{\text{mid}} = \text{RMSNorm}(x^{(\lfloor L/2\rfloor)}) \in \mathbb{R}^{512}$, `mb_head` forms multi-depth source vectors $s_{\text{sem}} = \text{RMSNorm}(0.70 h_{\text{pre}} + 0.30 h_{\text{mid}})$ and $s_{\text{id}} = \text{RMSNorm}(0.75 h_{\text{pre}} + 0.25 h_{\text{mid}})$:
    - **Secondary Semantic Direction ($h_{\text{sem}, 1} \in \mathbb{R}^{448}$)**: In coupled semantic modes (`M2, M3, M4`), a biorthogonal Stiefel bridge `sem1_bridge` maps $s_{\text{sem}} \mapsto u_{\text{sem}, 1} \in \mathbb{R}^{448}$, which is Gram-Schmidt orthogonalized against Anchor Branch 0's semantic vector $h_{\text{sem}, 0}$ so $h_{\text{sem}, 1} \perp h_{\text{sem}, 0}$ (in `M1`, $h_{\text{sem}, 1} = 0$).
    - **Excited Identity Branches ($h_{\text{id}, r} \in \mathbb{R}^{64}$ for $r = 1 \dots R-1$)**: $R - 1$ semi-orthogonal Stiefel matrices `id_branch_stiefel` $\in \mathbb{R}^{(R-1) \times 64 \times 512}$ map $s_{\text{id}} \mapsto \tilde{u}_{\text{id}, r} \in \mathbb{R}^{64}$, which are Gram-Schmidt orthogonalized against Anchor Branch 0's identity vector $h_{\text{id}, 0}$ to yield unit-RMS orthogonal vectors $u_{\text{id}, r} \perp h_{\text{id}, 0}$.
    - **$90^\circ$ Orthogonal Branches (`M1..M4` initial) vs. $30^\circ$ Geodesic Tangent Cone (`M1_CONE_FIBER_R4`)**:
      In $90^\circ$ mode, $h_{\text{id}, r} = u_{\text{id}, r}$. In **$30^\circ$ Geodesic Cone mode (`M1_CONE_FIBER_R4` / `D4`)**, each excited branch stays within $30^\circ$ ($\cos 30^\circ \approx 0.866, \sin 30^\circ = 0.50$) of Anchor Branch 0:
      $$h_{\text{id}, r} \;=\; \cos(30^\circ)\, h_{\text{id}, 0} + \sin(30^\circ)\, u_{\text{id}, r}, \qquad u_{\text{id}, r} \perp h_{\text{id}, 0}$$
    - **Branch Control Gates ($\beta_{t,r}, \gamma_{t,r}, \alpha_{t,r}$)**: Computed from $s_{\text{sem}, :16}$ via small linear projections: $\beta_{t,r} = 0.35\tanh(W_{\text{sem1}} s_{\text{sem}, :16})_r$ ($\beta_{t,0} = 0$), $\gamma_{t,r} = 2\sigma(W_{\text{id\_scale}} s_{\text{sem}, :16})_r$ ($\gamma_{t,0} = 1$), and mixture logits $\alpha_{t,r} = (W_{\text{gate}} s_{\text{sem}, :16} + b_{\text{gate}})_r$.
  - **Factored Multi-Branch Log-Sum-Exp Loss (`_FactoredMultiBranchWavepacketCEFn`)**:
    Evaluates **only 1 or 2 semantic GEMMs (`448D`) + $R$ small identity GEMMs (`64D`)**:
    $$z_{\text{base}, v} = \frac{\tau_{\text{sem}}}{448}\langle h_{\text{sem}, 0}, w_{\text{sem}, v}\rangle + b_{\text{unigram}, v}, \quad z_{\text{sem1}, v} = \frac{\tau_{\text{sem}}}{448}\langle h_{\text{sem}, 1}, w_{\text{sem}, v}\rangle, \quad z_{\text{id}, r, v} = \frac{\tau_{\text{id}}}{64}\langle h_{\text{id}, r}, w_{\text{id}, v}\rangle$$
    $$\text{raw}_{t, r, v} = 30.0\tanh\!\left(\frac{z_{\text{base}, v} + \beta_{t,r}\, z_{\text{sem1}, v} + \gamma_{t,r}\, z_{\text{id}, r, v}}{30.0}\right), \qquad \ln P_{\text{neural}}(v \mid x_{\le t}) = \underset{r=0}{\overset{R-1}{\operatorname{logsumexp}}}\!\Big(\ln \pi_{t,r} + \text{raw}_{t,r,v} - \underset{v'}{\operatorname{logsumexp}}(\text{raw}_{t,r,v'})\Big)$$
    where $\pi_{t,r} = \operatorname{softmax}_r(\alpha_{t,r})$.
  - **Step-0 Multi-Branch SVD Residual Calibration (`calibrate_multibranch_residual_step0_` in `M1..M4, D4`)**:
    After `exit_bridge` is calibrated for Anchor Branch 0, we run `8` Shard-1 batches forward (`no_grad`), project each target token's embedding $y = (y_{\text{sem}}, y_{\text{id}})$ orthogonal to Anchor Branch 0's prediction ($r_{\text{sem}} = y_{\text{sem}} - \frac{\langle y_{\text{sem}}, h_{\text{sem}, 0}\rangle}{\|h_{\text{sem}, 0}\|_2^2} h_{\text{sem}, 0}$ and $r_{\text{id}} = y_{\text{id}} - \frac{\langle y_{\text{id}}, h_{\text{id}, 0}\rangle}{\|h_{\text{id}, 0}\|_2^2} h_{\text{id}, 0}$), solve Tikhonov ridge regressions from $(s_{\text{sem}}, s_{\text{id}})$ to $(r_{\text{sem}}, r_{\text{id}})$, and initialize `sem1_bridge` and `id_branch_stiefel` via SVD of the ridge solutions.
  - **Definition of `Active Branches` / `Active Modes` (`effective_active_branches`)**:
    For any multi-branch or multi-mode model, let $H(\pi_t) = -\sum_{r=0}^{R-1} \pi_{t,r}\ln \pi_{t,r}$ be the Shannon entropy of the branch mixture probabilities at token $t$, averaged over all $N_{\text{val}} = 65,536$ validation tokens: $\bar{H} = \frac{1}{N_{\text{val}}}\sum_{t=1}^{N_{\text{val}}} H(\pi_t)$. Then:
    $$\text{Active Branches / Active Modes} \;=\; \exp(\bar{H}) \;\in\; [1.0,\, R]$$

#### Table 4A: Synthetic $K$-Target Superposition Probe (`64D` Identity Sphere, $V = 50,304$, Theoretical $K_{\max} = 2.956$)

| Simultaneous Targets $K$ | Shannon Entropy Floor ($\ln K$) | $R = 1$ Single Branch CE (`Top-K Recall`) | $R = 4$ Branches CE (`Top-K Recall`) | $R = 8$ Branches CE (`Top-K Recall`) |
| :---: | :---: | :---: | :---: | :---: |
| $K = 1$ | `0.0000` | `0.2486` (`100.0%`) | `0.2486` (`100.0%`) | `0.2486` (`100.0%`) |
| $K = 2$ | `0.6931` | `2.0326` (`100.0%`) | **`1.0913` (`100.0%`)** | `1.3606` (`100.0%`) |
| $K = 4$ ($> K_{\max}$) | `1.3863` | `4.0323` (**`78.91%`**) | **`1.6385` (`100.0%`)** | `1.9291` (`100.0%`) |
| $K = 8$ | `2.0794` | `5.5728` (**`44.53%`**) | `3.4542` (`100.0%`) | **`2.3270` (`100.0%`)** |
| $K = 16$ | `2.7726` | `6.6593` (**`22.46%`**) | `5.3779` (`81.05%`) | **`4.1781` (`100.0%`)** |

#### Table 4B: Phase 4 (`M0..M4`) 300-Step FineWeb Multi-Branch Readout Ablation (`L = 4, d = 512, T = 512`)

| Variant | Branches $R$ | Branch Geometry & Coupling | Step 100 Pure Neural CE | Step 200 Pure Neural CE | **Step 300 Pure Neural CE (`PPL`)** | Step 300 Combined KN CE (`PPL`) | Top-1 Acc | Active Branches |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `S3 / M0_SINGLE_BRANCH_BASELINE` | `1` | Single wavepacket readout (`num_branches = 1`) | `6.3300` | `6.0167` | `5.8400` (`343.79`) | — | `18.92%` | `1.00` |
| `M1_ORTHO_90DEG_R4` (initial) | `4` | $90^\circ$ orthogonal Stiefel branches (`id` only) | `6.3385` | `6.0379` | `5.8665` (`353.00`) | — | `19.12%` | `3.12 / 4` |
| `M2_COUPLED_SEM_ID_R4` ($90^\circ$) | `4` | $90^\circ$ orthogonal branches + `sem1_bridge` | `6.3393` | `6.0460` | `5.8764` (`356.51`) | — | `18.87%` | `3.18 / 4` |
| `M3_COUPLED_SEM_ID_R8` ($90^\circ$) | `8` | $90^\circ$ orthogonal $R=8$ branches + `sem1_bridge` | `6.3336` | `6.0373` | `5.8683` (`353.65`) | — | `19.01%` | `5.62 / 8` |
| `M4_COUPLED_R8_CALIBRATED_SYNERGY` | `8` | `M3` + Shard `1..9` KN bigram synergy eval | `6.3336` | `6.0368` | `5.8680` (`353.54`) | `5.1524` (`172.84`) | `24.52%`* | `5.60 / 8` |
| **`M1_CONE_FIBER_R4` ($30^\circ$ Cone)** | **`4`** | **$30^\circ$ Geodesic Tangent Cone (`id_branch_stiefel`)** | **`6.3273`** | **`6.0060`** | **`5.8299` (`340.31`)** | — | **`19.12%`** | **`2.35 / 4`** |

*(\*In `M4`, Top-1 is evaluated on the Combined KN-Synergy distribution; all other rows report Pure Neural Top-1.)*

---

### Phase 5: Born-Schur Density-Matrix Attention & Full Density-Matrix Highway (`D0..D4`)
- **Code**: `src/qdlblocks/benchmarks/born_schur_density_lab.py`, `src/qdlblocks/kernels/fused_quantum_wavepacket_attn.py`
- **Mathematical Formulation of Explicit $32 \times 32$ Density-Matrix Models (`D0..D3` on Reversal) vs. Low-Rank Cholesky Factor $L_{t,h} \in \mathbb{C}^{4 \times 32}$ (`D4..D9` on FineWeb)**:
  1. **Explicit $32 \times 32$ Density-Matrix Ablation on Reversal (`D0 -> D1 -> D2 -> D3`)**:
     On short sequence reversal ($L_{\text{span}} = 8, T = 17$), each of the $H = 8$ packets has complex state dimension $d_c = 32$, and we materialize explicit $32 \times 32$ Hermitian density matrices $\bar{\rho}_{t,h} = \sum_{j \le t} P_{tj}^{(h)} |v_{j,h}\rangle\langle v_{j,h}| \in \mathbb{C}^{32 \times 32}$ (via real triangle packing $M = \rho_{\text{re}} + \rho_{\text{im}}$, since $\rho_{\text{re}}$ is symmetric and $\rho_{\text{im}}$ is skew-symmetric):
     - **`D0_MCWEENY_MATMUL_BASELINE`**: Uses standard matrix multiplication McWeeny purification $\rho_{t,h} \leftarrow \bar{\rho}_{t,h}^2 / \operatorname{Tr}(\bar{\rho}_{t,h}^2)$ within each block without cross-packet Schur products, projecting back to a rank-1 ket $|\psi_{\text{out}, t,h}\rangle = \text{Normalize}(\rho_{t,h} |\psi_{\text{in}, t,h}\rangle)$ at the end of each block.
     - **`D1_BORN_SCHUR_STATIC_ATTN`**: Adds the **dimension-calibrated cross-packet entrywise Schur product** across heads $a, b \in \{0, \dots, 7\}$, where the scalar multiplier $d_c = 32$ (written as $D = 32$ in Table 5A) restores unit expected trace $\mathbb{E}[\operatorname{Tr}(d_c(\bar{\rho}_{t,a} \odot \bar{\rho}_{t,b}))] = 1.0$:
       $$S_{t,h} \;=\; d_c \cdot \left(\sum_{a=0}^7 U_{ha}^2\, \bar{\rho}_{t,a}\right) \odot \left(\sum_{b=0}^7 V_{hb}^2\, \bar{\rho}_{t,b}\right) \;\in\; \mathbb{C}^{32 \times 32}, \qquad \tilde{\rho}_{t,h} \;=\; \frac{\sum_a W_{ha}^2 \bar{\rho}_{t,a} + S_{t,h}}{\operatorname{Tr}\!\big(\sum_a W_{ha}^2 \bar{\rho}_{t,a} + S_{t,h}\big) + \varepsilon}$$
     - **`D2_BORN_SCHUR_DYNAMIC_POVM`**: Replaces the static 1:1 addition in `D1` with dynamic **Born-Trace POVM gates** $g_{\text{lin}, t,h} = \operatorname{Tr}(\bar{\rho}_{t,h} |m_{\text{lin}, h}\rangle\langle m_{\text{lin}, h}|)$ and $g_{\text{schur}, t,h} = \operatorname{Tr}(\bar{\rho}_{t,h} |m_{\text{schur}, h}\rangle\langle m_{\text{schur}, h}|)$ before projecting back to a rank-1 ket between layers.
     - **`D3_FULL_DENSITY_MATRIX_HIGHWAY`**: Eliminates inter-layer ket collapse on reversal by maintaining a **persistent inter-layer density-matrix highway $\rho_{t,h}^{(\ell)} \in \mathbb{C}^{32 \times 32}$** initialized at Layer 0 as $\rho_{t,h}^{(0)} = |\psi_{t,h}^{(0)}\rangle\langle\psi_{t,h}^{(0)}|$ and updated across layers via POVM highway gate $\alpha_{t,h} = \operatorname{Tr}(\rho_{t,h}^{(\ell-1)} |m_{\text{hwy}, h}\rangle\langle m_{\text{hwy}, h}|)$ and 1-step McWeeny purification:
       $$\hat{\rho}_{t,h}^{(\ell)} \;=\; (1 - \alpha_{t,h})\, \rho_{t,h}^{(\ell-1)} + \alpha_{t,h}\, \tilde{\rho}_{t,h}^{(\ell)}, \qquad \rho_{t,h}^{(\ell)} \;=\; \frac{\big(\hat{\rho}_{t,h}^{(\ell)}\big)^2}{\operatorname{Tr}\!\big((\hat{\rho}_{t,h}^{(\ell)})^2\big) + \varepsilon}$$
  2. **Scaling to $T = 512$ `FineWeb` via Rank-4 Cholesky Factors $L_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$ (`D4..D9`)**:
     - Materializing $32 \times 32$ matrices at $T = 512$ costs $O(B H T d_c^2)$ memory. Instead, `fused_low_rank_cholesky_ket_attention` constructs a **4-row pre-attention local factor** $F_{j,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$ at each position $j$ from the 4 local kets of layer $\ell$:
       $$F_{j,h}^{(\ell)} \;=\; \frac{\widetilde{F}_{j,h}^{(\ell)}}{\big\|\widetilde{F}_{j,h}^{(\ell)}\big\|_F}, \qquad \widetilde{F}_{j,h}^{(\ell)} \;=\; \begin{bmatrix} \langle \hat{v}_{j,h}^{(\ell)} | \\ \langle v_{j,h}^{(\ell)} | \\ \langle k_{j,h}^{(\ell)} | \\ \langle q_{j,h}^{(\ell)} | \end{bmatrix} \;\in\; \mathbb{C}^{4 \times 32}$$
       where $|\hat{v}_{j,h}^{(\ell)}\rangle = |v_{j,h}^{(\ell)}\rangle / \|v_{j,h}^{(\ell)}\|_2$ is the unit-normalized value ket (row 0, norm $1$) and rows $1, 2, 3$ are the unnormalized value ket $|v_{j,h}^{(\ell)}\rangle$ and RMS-normalized rotary key/query kets $|k_{j,h}^{(\ell)}\rangle, |q_{j,h}^{(\ell)}\rangle$ (each of norm $\approx \sqrt{64} = 8$).
     - In a single fused causal pass (`_FusedHermitianBornRhoKetAttnFn`) under Hermitian-Born attention weights $P_{tj}^{(h)} = \operatorname{softmax}_{j \le t}\!\big(\sigma_{\text{sdpa}}\operatorname{Re}(e^{i\phi_h}\langle q_{t,h}\mid k_{j,h}\rangle) + \beta_{\text{born}, h}|\langle q_{t,h}\mid k_{j,h}\rangle|^2\big)$, the kernel aggregates both $|\bar{v}_{t,h}^{(\ell)}\rangle = \sum_{j \le t} P_{tj}^{(h)} |v_{j,h}^{(\ell)}\rangle \in \mathbb{C}^{32}$ and the **rank-4 Cholesky factor** $L_{t,h}^{(\ell)} = \sum_{j \le t} P_{tj}^{(h)} F_{j,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$ (representing $\bar{\rho}_{t,h}^{(\ell)} = (L_{t,h}^{(\ell)})^\dagger L_{t,h}^{(\ell)}$ in $4 \times 32$ memory instead of $32 \times 32$).
  3. **Mode-Aligned ($r \leftrightarrow r$) Low-Rank Cholesky Cross-Packet Schur Step (`_cholesky_schur_step`)**:
     - *Linear-Algebraic Clarification*: For two rank-$k$ density matrices $\rho_a = L_a^\dagger L_a$ and $\rho_b = L_b^\dagger L_b$ ($k = 4$), the exact factor of the full $32 \times 32$ Schur product $\rho_a \odot \rho_b$ is the row-wise Kronecker (Khatri-Rao) product of all $(r, r')$ mode pairs, which has $k^2 = 16$ rows (and would explode as $4 \to 16 \to 256$ across layers). To preserve fixed rank $k = 4$ in $O(k d_c)$ memory, `_cholesky_schur_step` performs **mode-aligned ($r \leftrightarrow r$) entrywise Schur binding** between corresponding Cholesky rows $r \in \{0, 1, 2, 3\}$ of $L_{t,h}$ and the cross-packet partner $P_{t,h} = \sum_{b=0}^7 W_{hb} L_{t,b} \in \mathbb{C}^{4 \times 32}$ ($W_{hb} = A_{hb}^2 \ge 0$):
       $$S_{t,h}^{\text{chol}} \;=\; \sqrt{d_c}\, \big(L_{t,h} \odot P_{t,h}\big) \;\in\; \mathbb{C}^{4 \times 32} \qquad (\sqrt{d_c} = \sqrt{32})$$
     - $L_{t,h}$ and $S_{t,h}^{\text{chol}}$ are then combined via Born-Trace POVM gates $g_{\text{lin}} = \|L_{t,h}|m_{\text{lin}, h}\rangle\|_2$ and $g_{\text{schur}} = s_h \|L_{t,h}|m_{\text{schur}, h}\rangle\|_2$ and Frobenius-normalized so $\|\tilde{L}_{t,h}\|_F = 1$ ($\operatorname{Tr}(\tilde{\rho}_{t,h}) = 1$):
       $$\tilde{L}_{t,h} \;=\; \frac{(g_{\text{lin}} + \varepsilon) L_{t,h} + g_{\text{schur}} S_{t,h}^{\text{chol}}}{\big\|(g_{\text{lin}} + \varepsilon) L_{t,h} + g_{\text{schur}} S_{t,h}^{\text{chol}}\big\|_F} \;\in\; \mathbb{C}^{4 \times 32}$$
  4. **Inversion-Free $4 \times 4$ Gram McWeeny Purification (`associative_low_rank_cholesky_gram_modulation`)**:
     Using the $4 \times 4$ Gram matrix $G_{t,h} = \tilde{L}_{t,h} \tilde{L}_{t,h}^\dagger \in \mathbb{C}^{4 \times 4}$ (where $\|G_{t,h}\|_F^2 = \operatorname{Tr}(\tilde{\rho}_{t,h}^2)$) and unitary evolution matrix $U_{\text{evol}, h} \in U(32)$, the block modulates $|w_{0,t,h}\rangle = U_{\text{evol}, h}^\dagger |\text{tar}_{t,h}\rangle$ (where $|\text{tar}_{t,h}\rangle = |q_{t,h}\rangle + |k_{t,h}\rangle$) in $O(k^2 d_c)$ ops:
     $$|w_{2,t,h}\rangle \;=\; \tilde{L}_{t,h}^\dagger \Big(G_{t,h} \big(\tilde{L}_{t,h} |w_{0,t,h}\rangle\big)\Big), \qquad |\text{out}_{t,h}\rangle \;=\; U_{\text{evol}, h}\!\left(|\bar{v}_{t,h}\rangle + \beta_{\rho, h}\left(\frac{|w_{2,t,h}\rangle}{\|G_{t,h}\|_F^2} - \frac{1}{d_c}|w_{0,t,h}\rangle\right)\right)$$

#### Table 5A: Phase 5 (`D0..D3`) `50,304`-Vocab Sequence Reversal ($L_{\text{span}} = 8, 200\text{ Steps}$) & `D4` FineWeb ($300\text{ Steps}$)

| Variant | Task | Architecture Description | Step 50 Val CE (`Top-1`) | Step 100 Val CE (`Top-1`) | Step 200 / 300 Val CE (`Top-1`) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `D0_MCWEENY_MATMUL_BASELINE` | Reversal | Rank-1 ket trunk + matrix-product McWeeny only | `5.5768` (`33.56%`) | `0.0981` (`100.00%`) | `0.0233` (`100.00%` @ Step 200) |
| `D1_BORN_SCHUR_STATIC_ATTN` | Reversal | + Cross-packet entrywise Schur product $d_c(\bar{\rho}_a \odot \bar{\rho}_b)$ | `5.2542` (`36.59%`) | `0.1281` (`99.96%`) | `0.0232` (`100.00%` @ Step 200) |
| `D2_BORN_SCHUR_DYNAMIC_POVM` | Reversal | + Dynamic Born-Trace POVM gate $\operatorname{Tr}(\bar{\rho}_{t,h} M_h)$ | `5.1959` (`37.80%`) | `0.0946` (`99.99%`) | `0.0226` (`100.00%` @ Step 200) |
| **`D3_FULL_DENSITY_MATRIX_HIGHWAY`** | Reversal | **Full inter-layer $\rho_{t,h}^{(\ell)} \in \mathbb{C}^{32 \times 32}$ residual highway** | **`4.3994` (`46.77%`)** | **`0.0576` (`100.00%`)** | **`0.0183` (`100.00%` @ Step 200)** |
| `D4_BORN_SCHUR_POVM_CONE_R4` | FineWeb | Low-rank Cholesky Schur block (collapsed $4\to 1$ between layers) + $30^\circ$ Cone `mb_head` | — | `6.3319` (`15.40%`) | `5.8592` (`18.86%` @ Step 300) |

---

### Phase 6: Phase-Invariant Outer-Product Input (`W0..W3`) & Cone-Free Native Cholesky Readout (`D5, D6`)
- **Code**: `src/qdlblocks/benchmarks/wavefunction_outer_density_lab.py`
- **Explicit Layer-0 Input Equations for `W0, W1, W2, W3` (`apply_phase_invariant_outer_density_input_lift`)**:
  Let $|\psi_{\text{tok}, t, h}\rangle \in \mathbb{C}^{32}$ be token $t$'s wavepacket from `wte_3d` and $|\text{pos}_{t,h}\rangle \in \mathbb{C}^{32}$ be the positional wavepacket from `wpe_3d`.
  - In `W0`, `W1`, and `W2`, the base wavepacket is $|\psi_{t,h}\rangle = |\psi_{\text{tok}, t, h}\rangle + \gamma_{\text{pos}}|\text{pos}_{t,h}\rangle$, and the causal neighbor state is $|\tilde{\psi}_{t-1, h}\rangle = e^{i\theta_{\text{disp}, h}}|\psi_{t-1, h}\rangle$ (clamped to $0$ on the identity packet). Let $|u_{t,h}\rangle = |\psi_{t,h}\rangle / \|\psi_{t,h}\|_2$ and $|\tilde{u}_{t-1,h}\rangle = |\tilde{\psi}_{t-1,h}\rangle / \|\tilde{\psi}_{t-1,h}\|_2$:
    - **`W0_VECTOR_SUM_THEN_OUTER` (Baseline)**: Adds vectors directly ($|\psi_{t,h}\rangle + |\tilde{\psi}_{t-1,h}\rangle$) before forming Layer 0's density state, which is vulnerable to global phase flips on $|\tilde{\psi}_{t-1,h}\rangle$.
    - **`W1_OUTER_PHASE_ALIGNED_MIXTURE`**: Acts on $|u_{t,h}\rangle$ with the rank-1 outer-product operator $\rho_{t-1,h} = |\tilde{u}_{t-1,h}\rangle\langle\tilde{u}_{t-1,h}|$ gated by Born-Trace POVM probability $g_{\text{mix}, t,h} = \sigma(\eta_{\text{mix}, h})|\langle m_{\text{mix}, h} \mid u_{t,h}\rangle|^2$:
      $$|u_{t,h}^{\text{W1}}\rangle \;=\; (1 - g_{\text{mix}, t,h})\, |u_{t,h}\rangle + g_{\text{mix}, t,h}\, |\tilde{u}_{t-1,h}\rangle\langle\tilde{u}_{t-1,h} \mid u_{t,h}\rangle$$
    - **`W2_OUTER_PLUS_BIGRAM_SCHUR_INPUT`**: Adds the **Bigram Schur Coincidence** modulation—the diagonal of the entrywise Schur outer product $d_c\big(|u_{t,h}\rangle\langle u_{t,h}| \odot |\tilde{u}_{t-1,h}\rangle\langle\tilde{u}_{t-1,h}|\big)$ acting on $|u_{t,h}\rangle$, which scales each basis coordinate $d \in \{0, \dots, 31\}$ by token $t-1$'s phase-invariant Born probability density $d_c |\tilde{u}_{t-1,h,d}|^2$ gated by $g_{\text{schur}, t,h} = \sigma(\eta_{\text{schur}, h})|\langle m_{\text{schur}, h} \mid u_{t,h}\rangle|^2$:
      $$|u_{t,h}^{\text{W2}}\rangle \;=\; |u_{t,h}^{\text{W1}}\rangle + g_{\text{schur}, t,h}\, d_c\Big(|\tilde{u}_{t-1,h}|^2 \odot |u_{t,h}\rangle\Big)$$
      Under any phase rotation $|\tilde{u}_{t-1,h}\rangle \mapsto e^{i\theta_{b,t,h}}|\tilde{u}_{t-1,h}\rangle$, both `W1` and `W2` are **100% invariant** ($0.0$ numerical change, verified in `test_wavefunction_outer_density_input_and_cone_free_readout`).
    - **`W3_UNITARY_POS_OUTER_DENSITY_INPUT` (Why Multiplicative Positional Phase Failed)**: In `W3`, we replaced additive positional anchoring on $|\psi_{\text{tok}, t, h}\rangle$ with coordinate-wise multiplicative complex phase rotation $e^{i\phi_{\text{pos}}(t)} \odot |\psi_{\text{tok}, t, h}\rangle$. On sequence reversal (`Table 6A`), `W3` dropped to `94.60%` Top-1 (`0.5184` CE) because multiplying token $t$'s identity packet (`Packet 7`) by a position-dependent phase $e^{i\phi_{\text{pos}}(t)}$ rotates the token's lexical identity vector differently at every position $t$, scrambling its alignment with the static output detector table `w_det_512`.
- **Eliminating the `D4` Rank-4 $\to$ Rank-1 $\to$ Rank-4 Bottleneck (`D5_NATIVE_CHOLESKY_DENSITY_NO_CONE`)**:
  - In `D4`, each layer formed $\tilde{L}_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$, collapsed it into the rank-1 vector $|\text{out}_{t,h}\rangle$, and used `mb_head` (`99,000` parameters + $30^\circ$ cone + Step-0 SVD residual calibration) to re-create 4 branches.
  - In `D5` and `D6`, we set `mb_head = None` (`-99,000` parameters, zero Step-0 SVD residual calibration) and read out the 4 branches directly from the final layer's evolved identity Cholesky factor $L_{\text{out}, t, 7}^{(L)} = \tilde{L}_{t,7}^{(L)} U_{\text{evol}, 7}^\dagger \in \mathbb{C}^{4 \times 32}$.
  - **Exact Born Probability Conservation of the 4 Cholesky Mode Energies**: Because $\|\tilde{L}_{t,7}^{(L)}\|_F = 1$ and $U_{\text{evol}, 7}^\dagger \in U(32)$ is unitary, the 4 row energies $E_{t,r} = \|l_{\text{out}, t, 7, r}^{(L)}\|_2^2$ satisfy $\sum_{r=0}^3 E_{t,r} = \|L_{\text{out}, t, 7}^{(L)}\|_F^2 = 1$, forming an exact Born probability distribution over the 4 Cholesky modes:
    $$\ln \tilde{\pi}_{t,r} \;=\; \ln E_{t,r} + b_{\text{prior}, r}, \qquad r \in \{0, 1, 2, 3\}$$

#### Table 6A: Phase 6 (`W0..W3`) `50,304`-Vocab Reversal Input Outer-Product Ablation ($200$ Steps)

| Variant | Input Combination Mechanism at Layer 0 | Step 100 Val CE (`Top-1`) | Step 200 Val CE (`Top-1`) |
| :--- | :--- | :---: | :---: |
| `W0_VECTOR_SUM_THEN_OUTER` | Vector sum $|\psi_t\rangle + |\text{pos}_t\rangle + |\tilde{\psi}_{t-1}\rangle$ before outer product | `0.0576` (`100.00%`) | `0.0183` (`100.00%`) |
| `W1_OUTER_PHASE_ALIGNED_MIXTURE` | Phase-invariant outer-product operator $(|\tilde{u}_{t-1}\rangle\langle\tilde{u}_{t-1}|)|u_t\rangle$ + POVM gate | `0.0536` (`100.00%`) | **`0.0179` (`100.00%`)** |
| **`W2_OUTER_PLUS_BIGRAM_SCHUR_INPUT`** | **Outer-product mixture + Input Bigram Schur coincidence $d_c(|\tilde{u}_{t-1}|^2 \odot |u_t\rangle)$** | **`0.0527` (`100.00%`)** | **`0.0179` (`100.00%`)** |
| `W3_UNITARY_POS_OUTER_DENSITY_INPUT` | Multiplicative positional phase $e^{i\phi_{\text{pos}}(t)} \odot |\psi_{\text{tok}, t}\rangle$ (rotates ID fiber) | `3.0279` (`61.21%`) | `0.5184` (`94.60%`) |

*(Note on `D5` vs. `D6`: On `FineWeb` (`Table 7A` below), `D5` (`use_outer_product_input=False`, `5.7528` Val CE) and `D6` (`use_outer_product_input=True`, `5.7612` Val CE) performed within `0.008` nats of each other because `FineWeb` models already include `SemanticSubspaceNGramProjector` at the input; therefore, `D7..D9` use `use_outer_product_input=False` to isolate the uncollapsed inter-layer Cholesky highway cleanly.)*

---

### Phase 7: Uncollapsed 4-Mode Cholesky Inter-Layer Highway (`D7, D8`) & `1,200`-Step FineWeb Scaling (`D9`)
- **Code**: `src/qdlblocks/benchmarks/wavefunction_outer_density_lab.py`, `scripts/wavefunction_outer_density_bench.py`
- **Mathematical Formulation of the Uncollapsed 4-Mode Cholesky Highway (`D7..D9`)**:
  1. **$U(1)^4$ Row Phase-Locking (`phase_lock_cholesky_modes`)**:
     Let $L_{\text{in}, t,h}^{(\ell-1)} \in \mathbb{C}^{4 \times 32}$ be the incoming 4-mode Cholesky factor from layer $\ell - 1$ and $\tilde{L}_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$ be layer $\ell$'s post-Schur Cholesky factor (both in the canonical pre-$U_{\text{evol}}$ packet basis).
     For each mode $r \in \{0, 1, 2, 3\}$, we compute the complex inner product between incoming row $l_{\text{in}, r}$ and current row $\tilde{l}_{\text{cur}, r}$:
     $$\langle l_{\text{in}, r} \mid \tilde{l}_{\text{cur}, r}\rangle \;=\; \text{ip}_{\text{re}, r} + i\,\text{ip}_{\text{im}, r}, \qquad e^{i\phi_r} \;=\; \frac{\text{ip}_{\text{re}, r} + i\,\text{ip}_{\text{im}, r}}{\sqrt{\text{ip}_{\text{re}, r}^2 + \text{ip}_{\text{im}, r}^2 + \varepsilon}}$$
     Rotating each incoming row via $l_{\text{in}, r}^{\text{aligned}} = e^{i\phi_r} l_{\text{in}, r}$ leaves the previous layer's density matrix $\rho^{(\ell-1)} = (L_{\text{in}}^{(\ell-1)})^\dagger L_{\text{in}}^{(\ell-1)} = \sum_{r=0}^3 |l_{\text{in}, r}\rangle\langle l_{\text{in}, r}|$ **100% invariant** while guaranteeing constructive real overlap $\operatorname{Re}\langle \tilde{l}_{\text{cur}, r} \mid l_{\text{in}, r}^{\text{aligned}}\rangle \ge 0$.
  2. **Per-Mode Born-Trace POVM Highway Merge (`_merge_uncollapsed_cholesky_highway`)**:
     Each mode $r \in \{0, 1, 2, 3\}$ of head $h$ computes its Born overlap with a learned highway POVM vector $|m_{\text{hwy}, h}\rangle \in \mathbb{C}^{32}$:
     $$p_{t,h,r} \;=\; \big|\langle \tilde{l}_{\text{cur}, t,h,r} \mid m_{\text{hwy}, h}\rangle\big|^2 \in [0, 1], \qquad \alpha_{t,h,r} \;=\; \sigma\!\big(b_{\text{hwy}, h, r} + s_{\text{hwy}, h, r}\, p_{t,h,r}\big)$$
     $$L_{t,h,r}^{(\ell)} \;=\; \frac{\sqrt{\alpha_{t,h,r}}\, \tilde{l}_{\text{cur}, t,h,r}^{(\ell)} + \sqrt{1 - \alpha_{t,h,r}}\, l_{\text{in}, t,h,r}^{\text{aligned}, (\ell-1)}}{\Big\|\sqrt{\alpha_{t,h}}\, \tilde{L}_{\text{cur}, t,h}^{(\ell)} + \sqrt{1 - \alpha_{t,h}}\, L_{\text{in}, t,h}^{\text{aligned}, (\ell-1)}\Big\|_F}$$
     $L_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$ is then passed into `associative_low_rank_cholesky_gram_modulation` AND passed **uncollapsed** as `in_cholesky` to layer $\ell + 1$.
  3. **Why `D7` (`5.7522`) Needed `D8`'s Joint Semantic (`448D`) + Identity (`64D`) Cholesky Readout (`5.6870`)**:
     - In `D7_UNCOLLAPSED_CHOLESKY_4MODE_HIGHWAY`, all 8 packets (`0..6` semantic + `7` identity) propagated 4 uncollapsed Cholesky modes across layers $\ell = 1 \to 2 \to 3 \to 4$. However, `D7` kept `D5`'s identity-only Cholesky readout, which inspected **only Packet 7 (`64D`, $12.5\%$ of the model's width)** and discarded the 3 excited Cholesky modes ($r = 1, 2, 3$) of **Semantic Packets `0..6` (`448D`, $87.5\%$ of the model's width)**! Consequently, `D7` (`5.7522`) improved over `D5` (`5.7528`) by only `-0.0006` nats (`+0.19%` Top-5).
     - In **`D8_UNCOLLAPSED_CHOLESKY_SEM_ID_READOUT`**, we connected the final layer's evolved Cholesky modes $L_{\text{out}, t,h}^{(L)} = L_{t,h}^{(L)} U_{\text{evol}, h}^\dagger \in \mathbb{C}^{4 \times 32}$ across **all 8 packets**:
       - **Identity Branches (`4 x 64D`)**: Packet 7's 4 Cholesky rows $l_{\text{out}, t, 7, r}^{(L)} \in \mathbb{R}^{64}$ ($r = 0, 1, 2, 3$) form the 4 identity branches (with mode 0 anchored to $h_{\text{id}, 0}$ and modes $1, 2, 3$ coherence-blended via $\sigma(\text{chol\_mode\_coherence}) h_{\text{id}, 0} + (1 - \sigma(\text{chol\_mode\_coherence})) \text{RMSNorm}(l_{\text{out}, t, 7, r}^{(L)})$).
       - **Excited Semantic Cholesky Direction (`448D`)**: From Semantic Packets `0..6`, we take the 3 excited Cholesky rows ($r \in \{1, 2, 3\}$), form their learned softmax superposition $u_{\text{sem}, t} = \sum_{r=1}^3 w_{\text{sem}, r}\, l_{\text{out}, t, 0..6, r}^{(L)} \in \mathbb{R}^{448}$, project out the component parallel to the base semantic vector $h_{\text{sem}, 0}$ so $h_{\text{sem}, 1} \perp h_{\text{sem}, 0}$, and assign each excited branch $r \in \{1, 2, 3\}$ its own learned gate $\beta_r = 0.35\tanh(g_{\text{sem}, r})$ in `_FactoredMultiBranchWavepacketCEFn`.
       - **Combined Semantic + Identity Mode Energy Weights**:
         $$E_{t,r} \;=\; (1 - w_{\text{sem}})\,\big\|l_{\text{out}, t, 7, r}^{(L)}\big\|_2^2 + w_{\text{sem}}\,\frac{1}{7}\sum_{h=0}^6 \big\|l_{\text{out}, t, h, r}^{(L)}\big\|_2^2, \qquad \ln \tilde{\pi}_{t,r} \;=\; \ln E_{t,r} + b_{\text{prior}, r}$$
       - Unlocking those `448D` semantic Cholesky modes in `D8` immediately dropped 300-step validation CE by **`-0.0652` nats (`5.7522 -> 5.6870`, `-19.87` PPL)**!

#### Table 7A: Complete 300-Step (`2.46M`-Token) FineWeb Architectural Progression (`Pure Neural Val CE`, `300`-Step Cosine Horizon)

| Variant | Inter-Layer Cholesky | Readout Mechanism | Extra Params | Step 100 CE | Step 200 CE | **Step 300 Val CE** | **Step 300 PPL** | **Pure Neural Top-1** | **Pure Neural Top-5** | Active Modes |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `S2_QUANTUM_WAVEPACKET_4L_PURE` | Rank-1 ket | $R=1$ Wavepacket (no `sem_ngram`) | `0` | `6.4794` | `6.1328` | `5.9522` | `384.62` | `18.03%` | `33.70%` | `1.00` |
| `D4_BORN_SCHUR_POVM_CONE_R4` | Collapsed $4 \to 1 \to 4$ | $30^\circ$ Geodesic Cone (`mb_head`) | `+99K` | `6.3319` | `6.0325` | `5.8592` | `350.43` | `18.86%` | `35.31%` | `2.29 / 4` |
| `S3 / M0_SINGLE_BRANCH_BASELINE` | Rank-1 ket | $R=1$ Wavepacket (with `sem_ngram`) | `0` | `6.3300` | `6.0167` | `5.8400` | `343.79` | `18.92%` | `35.37%` | `1.00` |
| `M1_CONE_FIBER_R4` | Rank-1 ket | $30^\circ$ Geodesic Cone (`mb_head`) | `+99K` | `6.3273` | `6.0060` | `5.8299` | `340.31` | `19.12%` | `35.68%` | `2.35 / 4` |
| `D5_NATIVE_CHOLESKY_DENSITY_NO_CONE` | Collapsed $4 \to 1$ | Cone-Free Identity Cholesky | **`0`** | `6.2418` | `5.9209` | `5.7528` | `315.08` | `20.95%` | `36.64%` | `2.24 / 4` |
| `D6_OUTER_INPUT_NATIVE_CHOLESKY_NO_CONE` | Collapsed $4 \to 1$ | Outer Input + Identity Cholesky | **`0`** | `6.2427` | `5.9289` | `5.7612` | `317.73` | `20.92%` | `36.68%` | `2.26 / 4` |
| **`D7_UNCOLLAPSED_CHOLESKY_4MODE_HIGHWAY`** | **Uncollapsed $4 \times 32$** | Cone-Free Identity Cholesky (`64D`) | **`0`** | `6.2307` | `5.9129` | **`5.7522`** | **`314.87`** | `20.92%` | `36.83%` | `2.27 / 4` |
| `S1_CLASSICAL_4L_BASELINE` | Standard Transformer | Standard Linear + Softmax | `0` | `6.2891` | `5.9102` | `5.7316` | `308.45` | `20.61%` | `36.01%` | `1.00` |
| **`D8_UNCOLLAPSED_CHOLESKY_SEM_ID_READOUT`** | **Uncollapsed $4 \times 32$** | **Cone-Free Semantic (`448D`) + Identity (`64D`) Cholesky** | **`0`** | **`6.1946`** | **`5.8559`** | **`5.6870`** | **`295.00`** | **`21.00%`** | **`37.32%`** | **`2.44 / 4`** |
| **`V2_VACUUM_PLUS_LAYER0_INPUT_CHOLESKY`** | **Layer-0 $L^{(0)}$ + Uncollapsed $4 \times 32$** | **Vacuum Sphere $\mathcal{H}_{\text{vac}}$ ($d_{\text{vac}}=4$) + Cholesky** | **`+0.3K`** | `6.1982` | `5.8445` | **`5.6796`** | **`292.83`** | `20.86%` | **`37.47%`** | **`2.63 / 4`** |
| **`V1_VACUUM_SUBSPACE_STEP0_PURE_NEURAL`** | **Uncollapsed $4 \times 32$** | **Vacuum Sphere $\mathcal{H}_{\text{vac}}$ ($d_{\text{vac}}=4$) + Cholesky** | **`+0.3K`** | `6.1984` | **`5.8444`** | **`5.6795`** | **`292.81`** | **`21.02%`** | **`37.58%`** | **`2.66 / 4`** |
| **`R3_RANK6_VACUUM_CHOLESKY` ($R = 6$)** | **Layer-0 $L^{(0)}$ + Uncollapsed $6 \times 32$** | **Vacuum Sphere $\mathcal{H}_{\text{vac}}$ + $R=6$ Cholesky** | **`+0.4K`** | `6.2015` | `5.8420` | **`5.6786`** | **`292.53`** | `20.77%` | **`37.56%`** | **`3.12 / 6`** |
| **`R5_RANK12_VACUUM_CHOLESKY` ($R = 12$)** | **Layer-0 $L^{(0)}$ + Uncollapsed $12 \times 32$** | **Vacuum Sphere $\mathcal{H}_{\text{vac}}$ + $R=12$ Cholesky** | **`+0.6K`** | `6.2006` | **`5.8318`** | **`5.6739`** | **`291.18`** | **`20.92%`** | **`37.54%`** | **`4.53 / 12`** |
| **`B2_SOFTMAX_FREE_VACUUM_BORN_R12` ($R = 12$)** | **Uncollapsed $12 \times 32$ + Trunk No-Click** | **100% `softmax`-Free $2p$-Boson Coincidence Born on $\mathcal{H}_{\text{vac}}$** | **`+0.8K`** | `6.2198` | `5.8787` | **`5.7093`** | **`301.65`** | `20.50%` | `36.93%` | `2.92 / 12` |
| **`N1_RANK8_NOCLICK_VACUUM_POVM` ($R = 8$)** | **Uncollapsed $8 \times 32$ + Trunk No-Click** | **Vacuum Sphere $\mathcal{H}_{\text{vac}}$ + Trunk No-Click POVM** | **`+0.7K`** | `6.1968` | `5.8385` | **`5.6743`** | **`291.28`** | `20.67%` | `37.54%` | `3.52 / 8` |
| **`N2_RANK12_NOCLICK_VACUUM_POVM` ($R = 12$)** | **Uncollapsed $12 \times 32$ + Trunk No-Click** | **Vacuum Sphere $\mathcal{H}_{\text{vac}}$ + Trunk No-Click POVM** | **`+0.8K`** | `6.1980` | **`5.8289`** | **`5.6699`** 🏆 | **`290.02`** 🏆 | `20.78%` | **`37.63%`** 🏆 | **`4.63 / 12`** 🏆 |

#### Table 7B: Longer `1,200`-Step (`9.83M`-Token) FineWeb Scaling Run (`D9_UNCOLLAPSED_CHOLESKY_LONG_SCALING_1200`)

- **Why `D8` at Step 300 (`5.6870`) and `D9` at Step 300 (`5.8405` Pure Neural CE) Differ**:
  Both `D8` and `D9` run the **exact same neural architecture** (`uncollapsed_cholesky_highway=True, use_semantic_cholesky_readout=True`), but with different total step horizons (`steps = 300` in `D8` vs. `steps = 1,200` in `D9`):
  - Because our cosine learning-rate schedule (`lr_mult`, defined in Section 2) scales warmup (`0.08 * steps`) and cosine cooldown to the total run length `steps`, at Step 300 **`D8` has finished $100\%$ of its cosine cooldown (`lr_mult = 0.12`, fully annealed)**, whereas at Step 300 **`D9` is only $25\%$ into its 1,200-step schedule (`lr_mult = 0.892`, still near peak learning rate `1.00`)**.
  - Once `D9` progresses through its cosine cooldown, its Pure Neural Val CE drops past `D8` at Step 450 (`5.6445`) and reaches **`5.1806` (`Neural PPL = 177.79`)** at Step 1,200.
- **Note on Accuracy Columns in Table 7B**: Because `D9` sets `enable_kn_synergy_eval=True`, the evaluator computes `Combined Top-1 Acc` and `Combined Top-5 Acc` from the KN-Synergy combined distribution (`0.45 Neural + 0.55 Shard 1..9 KN Bigram`), while reporting both **Pure Neural Val CE** and **Combined KN-Synergy Val CE** side by side:

| Step | Tokens | `lr_mult` | Train CE | **Pure Neural Val CE** (`Neural PPL`) | **Combined KN-Synergy Val CE** (`Combined PPL`) | **Combined Top-1** | **Combined Top-5** | **Active Modes** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | `0.200` | — | `7.1549` (`1280.36`) | `5.5316` (`252.54`) | `21.54%` | `39.00%` | `3.20 / 4` |
| `150` | `1.23M` | `0.989` | `6.0576` | `6.0995` (`445.64`) | `5.2792` (`196.21`) | `23.92%` | `41.17%` | `1.89 / 4` |
| `300` | `2.46M` | `0.892` | `5.9361` | `5.8405` (`343.95`) | `5.1944` (`180.25`) | `24.33%` | `41.54%` | `2.53 / 4` |
| `450` | `3.69M` | `0.728` | `5.6189` | `5.6445` (`282.73`) | `5.1144` (`166.40`) | `24.85%` | `42.12%` | `2.99 / 4` |
| `600` | `4.92M` | `0.525` | `5.3696` | `5.4995` (`244.57`) | `5.0498` (`156.00`) | `25.27%` | `42.71%` | `3.12 / 4` |
| `750` | `6.14M` | `0.328` | `5.4424` | `5.3725` (`215.40`) | **`5.0030` (`148.85`)** | `25.43%` | `43.06%` | `3.18 / 4` |
| `900` | `7.37M` | `0.181` | `5.1137` | `5.2945` (`199.24`) | **`4.9709` (`144.16`)** | `25.78%` | `43.37%` | `3.09 / 4` |
| `1050` | `8.60M` | `0.160` | `5.3722` | `5.2207` (`185.06`) | **`4.9309` (`138.50`)** | `26.06%` | `43.86%` | `3.01 / 4` |
| **`1200`** | **`9.83M`** | **`0.120`** | **`5.1181`** | **`5.1806` (`177.79`)** | **`4.9162` (`136.48`)** | **`26.12%`** | **`43.93%`** | **`3.04 / 4`** |

---

### Experiment 7 (`V0..V4` / [`EXPERIMENT_PLAN_078.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_078.md)) — Orthogonal Vacuum Subspace ($\mathcal{H}_{\text{vac}}$) on the Readout Sphere, Layer-0 Cholesky Initialization, and 1,200-Step Pure-Neural 4-Mode $512\text{D}$ Scaling ([`vacuum_subspace_cholesky_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/vacuum_subspace_cholesky_lab.py))

- **Hypothesis & Motivation**:
  - Eliminate the external Kneser-Ney interpolation (`enable_kn_synergy_eval=False`, **100% Pure Neural** evaluation at every step) and test whether embedding the readout sphere in $\mathbb{S}^{(d + d_{\text{vac}}) - 1} = \mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$ ($d_{\text{vac}} = 4$, where all $V = 50,304$ token detectors have identically zero support $w_{v, \text{vac}} \equiv 0$) gives both the `64D` identity anchor ($r = 0$) and the excited Cholesky modes ($r = 1, 2, 3$) a geometric abstention / dark-parking angle $\cos\theta_{\text{vac}}(t, r) \in [0, 1]$.
  - Simultaneously test **Layer-0 Multi-Horizon Input Cholesky Initialization** ($L_{t,h}^{(0)} \in \mathbb{C}^{4 \times 32}$ built from $[|\psi_{t,h}\rangle,\, |\tilde{\psi}_{t-1,h}\rangle,\, |\tilde{\psi}_{t-2,h}\rangle,\, \sqrt{d_c}(|\psi_{t,h}\rangle \odot |\tilde{\psi}_{t-1,h}\rangle)]$) and **True 4-Mode Independent $512\text{D}$ ($448\text{D}$ Semantic + $64\text{D}$ Identity) Cholesky Readout** (`_VacuumFourMode512DWavepacketCEFn`).
- **Key Architectural & Scaling Discoveries**:
  1. **Unit-Frobenius to Unit-RMS Conversion (`CHOLESKY_UNIT_RMS_SCALE = 16.0`) in Native Wavepacket Space**:
     - Because each 4-mode Cholesky factor $L_{t,h}^{(L)} \in \mathbb{C}^{4 \times 32}$ is normalized to unit **Frobenius norm** ($\sum_{r=0}^3 \|L_{t,h,r}^{(L)}\|_2^2 = 1$), an average mode ($E_{t,r} = 1/4$) has coordinate mean-square $\operatorname{MS} = \frac{1/4}{64} = \frac{1}{256}$ ($\text{RMS} = 1/16$). Scaling $L_{t,h}^{(L)}$ by $\sqrt{4 \times 64} = 16.0$ in the native wavepacket basis puts active modes at unit coordinate RMS ($\operatorname{MS} = 4 E_{t,r} \approx 1.0$) while low-energy modes ($E_{t,r} \ll 1/4$) naturally fall below the vacuum rest-mass $u_{\text{chol\_vac}}$ and rotate into $\mathcal{H}_{\text{vac}}$.
  2. **`V1` & `V2` Beat `D8` at 300 Steps (`5.6795` vs. `5.6870` Pure Neural Val CE, `37.58%` Top-5, `2.66` Active Modes)**:
     - Adding $\mathcal{H}_{\text{vac}}$ ($d_{\text{vac}} = 4$) in **`V1_VACUUM_SUBSPACE_STEP0_PURE_NEURAL`** improves 300-step Pure Neural Val CE from `5.6870` (`D8`) to **`5.6795` (`292.81` PPL)**, Pure Neural Top-1 from `21.00%` to **`21.02%`**, Pure Neural Top-5 from `37.32%` to **`37.58%`**, and active Cholesky modes from `2.44` to **`2.66 / 4`**.
     - By Step 300, Anchor Mode 0 rotates to $\cos\theta_{\text{vac}, 0} = 0.823$ ($\theta_{\text{vac}, 0} \approx 34.6^\circ$), while Excited Cholesky Modes $1, 2, 3$ rotate to $\cos\theta_{\text{vac}, r} = 0.655$ ($\theta_{\text{vac}, r} \approx 49.1^\circ$ into the Dark Parking Vacuum Subspace $\mathcal{H}_{\text{vac}}$).
  3. **`V4` Beats `D9` Across the Full `1,200`-Step (`9.83M`-Token) Pure-Neural Scaling Run (`5.1771` Pure Neural Val CE, `177.17` PPL, `24.97%` Pure Neural Top-1, `42.20%` Pure Neural Top-5, `2.86` Active Modes)**:
     - Over `1,200` steps (`9.83M` tokens), the full `V4` model ($\mathcal{H}_{\text{vac}} + L_{t,h}^{(0)} + \text{4-Mode Independent } 512\text{D}$ Cholesky readout) outperforms `D9`'s Pure Neural Val CE (`5.1806`) at every checkpoint from Step 150 onward, reaching **`5.1771` Pure Neural Val CE (`177.17` PPL)**, **`24.97%` Pure Neural Top-1** (vs. `24.92%`), **`42.20%` Pure Neural Top-5** (vs. `42.06%`), and **`2.86 / 4` active Cholesky modes** (vs. `2.68` in `D9`), with Excited Modes parking at $\cos\theta_{\text{vac}, r} = 0.538$ ($\theta_{\text{vac}, r} \approx 57.5^\circ$, placing $71.0\%$ of their squared norm in $\mathcal{H}_{\text{vac}}$).

#### Table 8A: 300-Step (`2.46M`-Token) Vacuum-Subspace & 4-Mode Cholesky Ablation (`V0..V3`, 100% Pure Neural)

| Variant | Step 0 Pure Neural CE (`PPL`) | Step 0 Top-1 / Top-5 | Step 100 CE | Step 200 CE | **Step 300 Pure Neural CE (`PPL`)** | **Pure Neural Top-1 / Top-5** | **Active Modes** | **Final $\cos\theta_{\text{vac}}$ (Anchor / Excited)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `V0_D8_UNIT_SPHERE_BASELINE` ($d_{\text{vac}} = 0$) | `7.1549` (`1280.32`) | `12.10%` / `23.22%` | `6.1946` | `5.8559` | `5.6870` (`295.01`) | `21.00%` / `37.32%` | `2.44 / 4` | `1.000` / `1.000` (Forced $\mathbb{S}^{63}$) |
| **`V1_VACUUM_SUBSPACE_STEP0_PURE_NEURAL`** ($d_{\text{vac}} = 4$) | `7.1622` (`1289.78`) | **`12.28%` / `23.36%`** | `6.1984` | **`5.8444`** | **`5.6795` (`292.81`)** | **`21.02%` / `37.58%`** | **`2.66 / 4`** | **`0.823` ($\mathbf{34.6^\circ}$) / `0.655` ($\mathbf{49.1^\circ}$)** |
| **`V2_VACUUM_PLUS_LAYER0_INPUT_CHOLESKY`** ($d_{\text{vac}} = 4$ + $L^{(0)}$) | `7.1622` (`1289.78`) | **`12.28%` / `23.35%`** | `6.1982` | `5.8445` | **`5.6796` (`292.83`)** | `20.86%` / **`37.47%`** | **`2.63 / 4`** | **`0.824` ($\mathbf{34.5^\circ}$) / `0.653` ($\mathbf{49.3^\circ}$)** |
| **`V3_VACUUM_4MODE_INDEPENDENT_512D_CHOLESKY`** ($4 \times 512\text{D}$) | `7.1622` (`1289.80`) | **`12.28%` / `23.36%`** | `6.2072` | `5.8597` | `5.6970` (`297.99`) | `20.83%` / `37.31%` | `2.30 / 4` | **`0.823` ($\mathbf{34.6^\circ}$) / `0.646` ($\mathbf{49.8^\circ}$)** |

#### Table 8B: 1,200-Step (`9.83M`-Token) 100% Pure-Neural Scaling (`V4_PURE_NEURAL_VACUUM_CHOLESKY_SCALING_1200` vs. `D9`)

| Step | Tokens | `lr_mult` | `V4` Train CE | `D9` Pure Neural CE (`PPL`) | **`V4` Pure Neural CE (`PPL`)** | **`V4` Pure Neural Top-1** | **`V4` Pure Neural Top-5** | **`V4` Active Modes** | **`V4` $\cos\theta_{\text{vac}}$ (Anchor / Excited)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | `1.000` | — | `7.1549` (`1280.32`) | `7.1622` (`1289.74`) | **`12.29%`** | **`23.36%`** | **`3.25 / 4`** | `0.996` / `0.981` |
| `150` | `1.23M` | `0.995` | `5.9994` | `6.1436` (`465.72`) | **`6.1285` (`458.76`)** | **`17.15%`** | **`33.10%`** | `1.73 / 4` | `0.837` / `0.730` |
| `300` | `2.46M` | `0.928` | `5.8178` | `5.8229` (`337.95`) | `5.8234` (`338.11`) | **`20.15%`** | **`35.82%`** | `2.34 / 4` | `0.816` / `0.607` |
| `450` | `3.69M` | `0.795` | `5.5372` | `5.6227` (`276.63`) | **`5.6209` (`276.14`)** | **`21.28%`** | **`37.73%`** | **`2.55 / 4`** | `0.777` / `0.544` |
| `600` | `4.92M` | `0.620` | `5.3085` | `5.4751` (`238.67`) | **`5.4697` (`237.39`)** | **`23.06%`** | **`39.48%`** | **`2.71 / 4`** | `0.753` / `0.550` |
| `750` | `6.14M` | `0.434` | `5.3760` | `5.3607` (`212.87`) | **`5.3590` (`212.52`)** | **`23.57%`** | **`40.30%`** | **`2.81 / 4`** | `0.751` / `0.548` |
| `900` | `7.37M` | `0.271` | `5.0671` | `5.2912` (`198.58`) | **`5.2843` (`197.22`)** | **`24.13%`** | **`40.87%`** | **`2.87 / 4`** | `0.718` / `0.532` |
| `1050` | `8.60M` | `0.160` | `5.3177` | `5.2191` (`184.77`) | **`5.2148` (`183.98`)** | `24.56%` | **`41.88%`** | **`2.82 / 4`** | `0.724` / `0.541` |
| **`1200`** | **`9.83M`** | **`0.120`** | **`5.0779`** | `5.1806` (`177.79`) | **`5.1771` (`177.17`)** | **`24.97%`** | **`42.20%`** | **`2.86 / 4`** | **`0.720` ($\mathbf{43.9^\circ}$) / `0.538` ($\mathbf{57.5^\circ}$)** |

---

### Experiment 8 (`R0..R6` / [`EXPERIMENT_PLAN_079.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_079.md)) — Lifting the $R=4$ Restriction: Unrestricted Variable-Rank $R$-Mode Cholesky Highway & Progressive Vacuum Subspace Parking ([`variable_rank_cholesky_vacuum_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/variable_rank_cholesky_vacuum_lab.py))

- **Hypothesis & Motivation**:
  - Previously, `fused_low_rank_cholesky_ket_attention` hardcoded `k_chol = 4` because it only stacked the 4 single-position kets `[|r>, |v>, |k>, |q>]` and because on the fixed unit sphere $\mathbb{S}^{63}$, adding modes $R > 4$ caused **Anchor Trace Dilution** ($E_{t,0} \propto 1/R$) and **Unit-Sphere False-Positive Noise** from unused modes.
  - Now that the Vacuum Subspace $\mathcal{H}_{\text{vac}}$ gives unused modes a dark parking state, we lifted the $R=4$ restriction to support **arbitrary Cholesky rank $R \ge 1$** ($R \in \{1, 2, 4, 6, 8, 12, 16, \dots\}$) via three mathematical invariants:
    1. **Multi-Horizon & Schur Coincidence Attention Cholesky Basis (`fused_variable_rank_cholesky_ket_attention`)**: Populates modes $r = 0..3$ with local kets $(|\hat{v}\rangle, |v\rangle, |k\rangle, |q\rangle)$, modes $r = 4, 5$ with **Value-Key Schur Coincidence** $\sqrt{d_c}|\hat{v} \odot \hat{k}\rangle$ and **Value-Query Conjugate Dipole** $\sqrt{d_c}|\hat{v} \odot \hat{q}^*\rangle$, modes $r = 6, 7$ with **Attended Causal Lag-1** $|\hat{v}_{s-1}\rangle$ and **Causal Bigram Schur Coincidence** $\sqrt{d_c}|\hat{v}_s \odot \hat{v}_{s-1}\rangle$, and modes $r = 8..11$ with **Causal Lag-2, Skip-Bigram, Query-Key Born, and Trigram Schur Coincidences**.
    2. **Rank-Invariant Excited Trace Budget ($\gamma_{\text{trace}}(R) = \sqrt{3 / \max(1, R - 1)}$)**: Scales the $R-1$ excited rows so the total excited Frobenius trace $\sum_{r=1}^{R-1} E_{t,r}$ relative to anchor energy $E_{t,0}$ is **strictly invariant across all $R \ge 2$** (`< 1e-5` variance across $R \in \{2, 4, 6, 8, 12, 16\}$).
    3. **Rank-Adaptive Coordinate Unit-RMS Scaling ($\alpha_{\text{chol}}(R) = \sqrt{R \cdot d_{\text{id}}} = 8\sqrt{R}$) + Progressive Vacuum Rest-Mass ($\nu_{\text{vac}, r} = \nu_{\text{vac}, 1}(1 + 0.35\frac{r-1}{R-2})$)**: Ensures $\operatorname{MS}(\alpha_{\text{chol}}(R) L_{t,7,r}^{(L)}) = R \cdot E_{t,r}$ while parking higher-order modes ($r \ge 4$) deeper in $\mathcal{H}_{\text{vac}}$ unless strongly excited by context.

#### Table 9A: 300-Step (`2.46M`-Token) Unrestricted Cholesky Rank Sweep ($R \in \{2, 4, 6, 8, 12\}$ & Vacuum Ablation, 100% Pure Neural)

| Variant | Rank $R$ | Vacuum Dim $d_{\text{vac}}$ | Step 0 Pure Neural CE (`PPL`) | Step 200 CE | **Step 300 Pure Neural CE (`PPL`)** | **Pure Neural Top-1 / Top-5** | **Active Modes ($\exp(H(\pi))$)** | **Final $\cos\theta_{\text{vac}}$ `[Anchor / Core 1..3 / High 4..R-1]`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R0_RANK8_NO_VACUUM_SPHERE` | `8` | `0` (Forced $\mathbb{S}^{63}$) | `7.1642` (`1292.29`) | `5.8589` | `5.6942` (`297.15`) | `20.62%` / `37.35%` | `3.42 / 8` (`42.8%`) | `1.000 / 1.000 / 1.000` |
| `R1_RANK2_VACUUM_CHOLESKY` | `2` | `4` | `7.1654` (`1293.91`) | `5.8713` | `5.7164` (`303.82`) | `20.36%` / `36.92%` | `1.03 / 2` (`51.5%`) | `0.847 / 0.698 / 0.698` |
| `R2_RANK4_VACUUM_CHOLESKY_REF` (`V2`) | `4` | `4` | `7.1649` (`1293.26`) | `5.8445` | `5.6808` (`293.19`) | `20.89%` / `37.43%` | `2.65 / 4` (`66.3%`) | `0.854 / 0.728 / —` |
| **`R3_RANK6_VACUUM_CHOLESKY`** | **`6`** | **`4` + Prog.** | `7.1638` (`1291.81`) | `5.8420` | **`5.6786` (`292.53`)** | `20.77%` / **`37.56%`** | **`3.12 / 6` (`52.0%`)** | **`0.852 / 0.721 / 0.663`** |
| **`R4_RANK8_VACUUM_CHOLESKY`** | **`8`** | **`4` + Prog.** | `7.1640` (`1292.02`) | `5.8430` | **`5.6804` (`293.07`)** | `20.71%` / `37.51%` | **`3.55 / 8` (`44.4%`)** | **`0.851 / 0.736 / 0.668`** |
| **`R5_RANK12_VACUUM_CHOLESKY`** | **`12`** | **`4` + Prog.** | `7.1640` (`1292.05`) | **`5.8318`** | **`5.6739` (`291.18`)** 🏆 | **`20.92%`** / `37.54%` | **`4.53 / 12` (`37.8%`)** 🏆 | **`0.853 / 0.762 / 0.711`** |

#### Table 9B: 1,200-Step (`9.83M`-Token) 100% Pure-Neural Scaling at Unrestricted Rank $R = 8$ (`R6` vs. `V4` and `D9`)

| Step | Tokens | `R6` Train CE | `D9` ($R=4$, No Vac) CE (`PPL`) | `V4` ($R=4$, Vac) CE (`PPL`) | **`R6` ($R=8$, Vac) Pure Neural CE (`PPL`)** | **`R6` Pure Neural Top-1 / Top-5** | **`R6` Active Modes (`R=8`)** | **`R6` $\cos\theta_{\text{vac}}$ `[Anchor / Core 1..3 / High 4..7]`** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | — | `7.1549` (`1280.32`) | `7.1622` (`1289.74`) | **`7.1639` (`1291.98`)** | `12.32%` / `23.35%` | `5.31 / 8` | `0.996 / 0.986 / 0.949` |
| `150` | `1.23M` | `5.9917` | `6.1436` (`465.72`) | `6.1285` (`458.76`) | **`6.1033` (`447.32`)** | **`17.80%` / `33.29%`** | `2.69 / 8` | `0.845 / 0.753 / 0.663` |
| `300` | `2.46M` | `5.8108` | `5.8229` (`337.95`) | `5.8234` (`338.11`) | **`5.8049` (`331.91`)** | `19.67%` / **`35.91%`** | **`5.02 / 8`** | `0.821 / 0.659 / 0.481` |
| `450` | `3.69M` | `5.5368` | `5.6227` (`276.63`) | `5.6209` (`276.14`) | **`5.6117` (`273.61`)** | **`21.46%` / `37.97%`** | **`6.27 / 8`** | `0.791 / 0.613 / 0.419` |
| `600` | `4.92M` | `5.3207` | `5.4751` (`238.67`) | `5.4697` (`237.39`) | `5.4778` (`239.31`) | `22.82%` / `39.22%` | **`6.35 / 8`** | `0.766 / 0.619 / 0.393` |
| `750` | `6.14M` | `5.3577` | `5.3607` (`212.87`) | `5.3590` (`212.52`) | **`5.3566` (`212.01`)** | **`23.64%`** / `40.19%` | **`6.42 / 8`** | `0.777 / 0.630 / 0.398` |
| `900` | `7.37M` | `5.2912` (`198.58`) | `5.2843` (`197.22`) | **`5.2836` (`197.08`)** | `23.94%` / **`40.95%`** | **`6.40 / 8`** | `0.757 / 0.623 / 0.379` |
| `1050` | `8.60M` | `5.3219` | `5.2191` (`184.77`) | `5.2148` (`183.98`) | **`5.2093` (`182.97`)** | `24.54%` / **`41.93%`** | **`6.43 / 8`** | `0.766 / 0.647 / 0.381` |
| **`1200`** | **`9.83M`** | **`5.0712`** | `5.1806` (`177.79`) | `5.1771` (`177.17`) | **`5.1717` (`176.22`)** 🏆 | `24.89%` / **`42.21%`** | **`6.50 / 8` (`81.3%`)** 🏆 | **`0.771 / 0.648 / 0.383`** |

---

### Experiment 9 (`N1..N4, B0..B2, S0..S1` / [`EXPERIMENT_PLAN_080.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_080.md)) — Trunk Quantum "No-Click" POVM Vacuum Sink, Per-Mode Cholesky Spectrum Profiler, 100% `softmax`-Free Vacuum-Born, Leslie Smith $\beta = 0$ SGD, and $R = 12$ `1,200`-Step Scaling ([`noclick_vacuum_and_smith_sgd_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/noclick_vacuum_and_smith_sgd_lab.py), [`softmax_free_vacuum_born_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/softmax_free_vacuum_born_lab.py))

- **Key Discoveries Across Five Tracks**:
  1. **Trunk Quantum "No-Click" POVM Vacuum Sink (`N1 = 5.6743` at $R = 8$; `N2 = 5.6699` at $R = 12$, New 300-Step Record)**:
     - Equipping each wavepacket head $h \in \{0, \dots, 7\}$ inside causal Hermitian-Born Cholesky attention (`_FusedNoClickVacuumHermitianBornAttnFn`) with a learned Vacuum Key $|\hat{k}_{\text{vac}, h}\rangle \in \mathbb{C}^{32}$ paired with zero value excitation ($|v_{\text{vac}}\rangle = \mathbf{0}$) and zero excited Cholesky excitation ($F_{\text{vac}, r \ge 1} = \mathbf{0}$) improves $R = 8$ by **`-0.0037` nats** (`5.6780 -> 5.6743`) and improves $R = 12$ by **`-0.0040` nats** (`5.6739 -> 5.6699`, `290.02` PPL, `20.78%` Top-1, `37.63%` Top-5).
     - **Emergent Identity vs. Semantic No-Click Asymmetry**: Without head-specific supervision, the Identity Packet Head ($h = 7, \omega = 0$) learns to route **`36.6%` (`p_noclick_id = 0.366`)** of its causal attention mass into the No-Click Vacuum Sink at Step 300 and **`67.5%` (`p_noclick_id = 0.675` vs. `0.269` for semantic heads)** at Step 1,200 (`N4`), shielding exact lexical identity from uninformative prior window tokens while the 7 semantic heads integrate topical context.
  2. **Per-Mode Cholesky Spectrum & Tri-Modal Zipf Specialization (`scripts/cholesky_mode_spectrum_and_vacuum_angle_profiler.py`)**:
     - Profiling all individual modes $r = 0 \dots R-1$ on `65,536` unseen Shard-0 validation tokens ([`.cache/cholesky_mode_spectrum_profile.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/cholesky_mode_spectrum_profile.json)) reveals a crisp three-way Zipf specialization:
       - **Decile 0 (`Top 5,025` frequent function/boundary tokens, `81.0%` of corpus)**: Won by **Mode $r = 2$ (Rotary Key Ket $\lvert k_{t,h}\rangle$)** (`4.9217` vs. `5.2249` Mode-0 CE, **`-0.3032` nat advantage**).
       - **Deciles 1..3 (Core content words, ranks `5,026..20,101`)**: Won by **Mode $r = 1$ (Local Value Ket $\lvert v_{t,h}\rangle$)** (**`-0.4003` to `-0.5402` nat advantage** over Mode 0).
       - **Deciles 4..9 (Rare & multi-token tail words, ranks `20,102..50,256`)**: Won unanimously across all six rare deciles by **Mode $r = 6$ (Attended Causal Lag-1 Ket $\lvert r_{t-1,h}\rangle$)** (**`+0.3314` nat mean tail advantage** over Mode 0), which parks deepest in the vacuum on ordinary tokens ($\cos\theta_{\text{vac}} = 0.5595$ mean, `0.3685` at p10) and swings out to `0.9141` (p90) on rare collocations.
       - **Why $R = 12$ Beats $R = 8$**: At $R = 12$, **Mode $r = 10$ (Query-Key Born Dipole $\lvert q_{t,h}\rangle \odot \lvert k_{t,h}^*\rangle$)** captures **`12.05%` of total Cholesky energy (`11.09%` mixture weight)**—the 4th largest mode in the entire spectrum (`r = 1, 3, 2, 10`), which is truncated away when $R \le 8$.
  3. **1,200-Step (`9.83M`-Token) Scaling at $R = 12$ + Trunk No-Click POVM (`N4 = 5.1697` Pure Neural Val CE / `175.86` PPL / `24.91%` Top-1 / `8.11 / 12` Active Modes)**:
     - `N4_RANK12_NOCLICK_VACUUM_SCALING_1200` overtakes `R6` ($R = 8$, `5.1717`) and `V4` ($R = 4$, `5.1771`) over the second half of the 1,200-step run, reaching **`5.1697` Pure Neural Val CE (`175.86` PPL, `24.91%` Top-1, `42.19%` Top-5)** and expanding its active Cholesky spectrum from `4.63 / 12` (at 300 steps) to **`8.11 / 12` active modes** (`67.6%` utilization) at `9.83M` tokens.
  4. **Ripping Out `softmax` — 100% `softmax`-Free Vacuum-Sharpened Coincidence Born (`B0 -> B1 -> B2`)**:
     - Replacing `softmax`, `logsumexp`, and `tanh` softcapping across the entire architecture with $2p$-boson Fock Born polynomials $\Phi_{2p}(z) = \big(1 + \frac{z}{2p}\big)_+^{2p}$ (`_VacuumCoincidenceBornCEFn`) starts **`-0.0390` nats better at Step 0** (`7.1249` vs. `7.1639`) and converges cleanly to **`5.7093` Val CE (`301.65` PPL, `20.50%` Top-1)** at Step 300 (`B2`).
     - Adding $\mathcal{H}_{\text{vac}}$ ($d_{\text{vac}} = 4$) + Trunk No-Click Sink inside 100% `softmax`-free Born improves 300-step Val CE by **`-0.0080` nats at $R = 8$ (`B0 = 5.7197 -> B1 = 5.7117`)** and **`-0.0104` nats at $R = 12$ (`B0 = 5.7197 -> B2 = 5.7093`, `+0.18%` Top-1)**—larger than the vacuum gain in `softmax` (`-0.0052` nats), while our telemetry pinpoints why `B2` (`5.7093`, `2.92/12` modes) trails `N2` (`5.6699`, `4.63/12` modes): summing *unnormalized* polynomial mode intensities before a single global denominator $Z_t$ starves modes that rotate partially into $\mathcal{H}_{\text{vac}}$, motivating **per-mode normalized polynomial Fock mixtures** $\sum_r \pi_{t,r} \frac{\Phi_r(v)}{\sum_{v'} \Phi_r(v')}$.
  5. **Leslie Smith Momentum-Free ($\beta = 0$) SGD LR Range Test & Why Spectral/Riemannian Whitening Requires Temporal Covariance Integration**:
     - In the 80-step exponential LR Range Test, $\beta = 0.0$ exhibits a **`1.9114x` higher stability ceiling before divergence** ($\mu_{\text{min-loss}}(\beta = 0) = 1.8636\times$ vs. $\mu_{\text{min-loss}}(\beta = 0.70) = 0.9750\times$), confirming Smith's geodesic no-overshoot prediction.
     - However, at 300 steps, momentum-free $\beta = 0.0$ (`S0 = 5.8178`, `S1 = 5.8722`) trails $\beta = 0.70$ (`N1 = 5.6743`) by `0.1435` nats because our Polar-Express Stiefel and Tikhonov-Gram optimizers apply a **0-homogeneous spectral normalizer** $\Phi_{\text{polar}}(c M_t) \equiv \Phi_{\text{polar}}(M_t)$ that drives all singular values of $M_t$ toward `1.0`. Without multi-step momentum ($\beta = 0.70$), $M_t = G_t$ is a single `8,192`-token mini-batch gradient touching only `~2,100 / 50,257` token types, so $\Phi_{\text{polar}}(G_t)$ whitens rank-deficient mini-batch sampling noise to full unit singular value.

#### Table 10A: `EXPERIMENT_PLAN_080` Summary Table (`N1..N4, B0..B2, S0..S1`)

| Variant | Rank $R$ | Readout Family | Trunk No-Click | Mom. $\beta$ | Step-0 Val CE | **Final Pure Neural Val CE (`PPL`)** | **Top-1 / Top-5 Acc** | **Active Modes** | **Trunk $p_{\text{noclick}}$ (`mean / id / sem`)** |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`N1_RANK8_NOCLICK_VACUUM_POVM`** (`300` st) | `8` | Vacuum Cholesky (`softmax`) | `True` | `0.70` | `7.1639` | **`5.6743` (`291.28`)** | `20.67% / 37.54%` | `3.52 / 8` | `0.272 / 0.346 / 0.261` |
| **`N2_RANK12_NOCLICK_VACUUM_POVM`** (`300` st) | **`12`** | Vacuum Cholesky (`softmax`) | **`True`** | **`0.70`** | `7.1643` | **`5.6699` (`290.02`)** 🏆 | **`20.78% / 37.63%`** | **`4.63 / 12`** | **`0.272 / 0.366 / 0.259`** |
| `B0_SOFTMAX_FREE_BORN_NO_VACUUM_R8` (`300` st) | `8` | 100% `softmax`-Free Born ($d_{\text{vac}}=0$) | `False` | `0.70` | `7.1275` | `5.7197` (`304.81`) | `20.32% / 36.84%` | `2.38 / 8` | `0.000 / 0.000 / 0.000` |
| `B1_SOFTMAX_FREE_VACUUM_BORN_R8` (`300` st) | `8` | 100% `softmax`-Free Born ($d_{\text{vac}}=4$) | `True` | `0.70` | `7.1249` | `5.7117` (`302.40`) | `20.46% / 36.82%` | `2.37 / 8` | `0.258 / 0.362 / 0.244` |
| **`B2_SOFTMAX_FREE_VACUUM_BORN_R12`** (`300` st) | **`12`** | **100% `softmax`-Free Born ($d_{\text{vac}}=4$)** | **`True`** | **`0.70`** | **`7.1254`** | **`5.7093` (`301.65`)** | **`20.50% / 36.93%`** | **`2.92 / 12`** | **`0.275 / 0.380 / 0.260`** |
| `S0_MOMENTUM_FREE_SGD_BASELINE_LR` (`300` st) | `8` | Vacuum Cholesky (`softmax`) | `True` | `0.00` | `7.1639` | `5.8178` (`336.25`) | `19.71% / 36.11%` | `2.99 / 8` | `0.325 / 0.428 / 0.310` |
| `S1_MOMENTUM_FREE_SGD_SMITH_1CYCLE` (`300` st) | `8` | Vacuum Cholesky (`softmax`) | `True` | `0.00` | `7.1639` | `5.8722` (`355.02`) | `19.45% / 35.65%` | `2.65 / 8` | `0.297 / 0.300 / 0.296` |
| **`N4_RANK12_NOCLICK_VACUUM_SCALING_1200`** (`1200` st) | **`12`** | **Vacuum Cholesky (`softmax`)** | **`True`** | **`0.70`** | `7.1643` | **`5.1697` (`175.86`)** 🏆 | **`24.91% / 42.19%`** 🏆 | **`8.11 / 12`** 🏆 | **`0.320 / 0.675 / 0.269`** |

---

### Experiment 10 (`Z0..Z5` / [`EXPERIMENT_PLAN_081.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_081.md)) — Strict Speedrun-Compliant Zero-PPMI Deterministic Byte-Pair (`b_hi, b_lo`) & Chain-Rule Quantum Cholesky Architecture ([`zero_ppmi_byte_pair_cholesky_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/zero_ppmi_byte_pair_cholesky_lab.py))

- **Hypothesis & Motivation**:
  - Eliminate the offline `900M`-token PPMI / unigram / n-gram pre-scan (`0` training tokens touched before Step 1, `< 20 ms` Step-0 initialization vs. `58,500 ms` 9-shard PPMI scan) to achieve **100% compliance with strict `modded-nanogpt` speedrun rules**, while keeping the external model signature `idx: (B, T) -> CE loss over 50,257 valid GPT-2 tokens` unmodified via the **bijective `uint16` radix byte-pair decomposition**:
    $$b_{\text{hi}}(v) = v \gg 8 \in \{0, \dots, 196\}, \qquad b_{\text{lo}}(v) = v \ \& \ 255 \in \{0, \dots, 255\}, \qquad v = 256\,b_{\text{hi}}(v) + b_{\text{lo}}(v)$$
- **Key Architectural & Scaling Discoveries**:
  1. **Why $b_{\text{hi}}(v) = v \gg 8$ Is a Built-In BPE Merge-Rank & Zipf Coordinate (`scripts/byte_pair_radix_and_zero_ppmi_cold_start_profiler.py`)**:
     - On unseen FineWeb Shard 0, total unigram entropy $H(v) = 7.3067\text{ nats}$ splits cleanly into $H(b_{\text{hi}}) = 3.3991\text{ nats}$ (`46.52%`) + $H(b_{\text{lo}} \mid b_{\text{hi}}) = 3.9076\text{ nats}$ (`53.48%`), with buckets `b_hi in {0..3}` (`2.0%` of vocabulary) covering **`53.18%` of all token occurrences**.
     - Closed-form analytical BPE-rank Zipf initialization (`build_zero_data_bpe_rank_zipf_prior`, `0` tokens scanned) drops Step-0 validation CE from `10.8258` (uniform) to **`8.9050`**, and `OnlineCausalZipfTracker` pulls it down to **`7.9329`** within the first `50` training batches (`val_bhi_ce = 3.4104` vs. `3.3991` entropy floor).
  2. **`100.0%` Step-1 Vocabulary Detector Coverage via Factored `(b_hi, b_lo)` + Schur Coincidence (`Z2 = 6.1292` Val CE vs. `Z0 = 6.2710`, `-0.1418` Nats, `+1.61%` Top-1)**:
     - In an unfactored `50,304` table (`Z0`), a single `8,192`-token training batch updates only **`5.36%`** of vocabulary wavepackets on Step 1. With `BytePairSchurWavepacketSynthesizer` (`Z2`), shared high-byte (`197 x 512`), low-byte (`256 x 512`), and Schur coincidence ($\hat{D}_{\text{hi}}[b_{\text{hi}}] \odot \hat{D}_{\text{lo}}[b_{\text{lo}}]$) basis vectors give **`100.0%` of all `50,257` token wavepackets and readout detectors dense gradients from Step 1**, improving 300-step Zero-PPMI Val CE from `6.2710` (`Z0`) and `6.2271` (`Z1`) to **`6.1292` (`459.07` PPL, `18.37%` Top-1, `32.92%` Top-5)**.
  3. **Exact Chain-Rule Two-Byte Autoregressive Readout (`Z3`: `3.46x` Faster Training, `1.19e-7` Normalization Error, `17.07%` Top-1)**:
     - `ExactChainRuleTwoByteReadout` (`197 + 256 = 453` outcomes, zero `50,304`-way matmul during training) achieves exact probability conservation ($\sum_{v=0}^{50,256} P_{\text{chain}}(v \mid h_t) \equiv 1.000000$), the best Step-0 Zero-PPMI Val CE (**`8.9229`**), **`3.46x` faster training throughput (`255.8s` vs `885.3s`)**, and higher Top-1 accuracy (`17.07%`) than `Z0` (`16.76%`) and `Z1` (`16.96%`). Its higher conditional tail CE (`6.7474` vs. `6.1292` on `Z2`) pinpoints that while $b_{\text{hi}} = v \gg 8$ is a global frequency coordinate, $b_{\text{lo}} = v \ \& \ 255$ requires gathering the `256` token-specific detector rows $W_{\text{det}}[256\,b_{\text{hi}}(y_t) + b_{\text{lo}}]$ for bucket $b_{\text{hi}}(y_t)$.
  4. **1,200-Step (`9.83M`-Token) Zero-PPMI Scaling (`Z5 = 5.2807` Val CE / `196.51` PPL / `24.34%` Top-1 / `41.04%` Top-5)**:
     - Starting with **`0` pre-scanned tokens (`18.3 ms` init, `9.1344` Step-0 Val CE)**—nearly `2.0` nats behind `N4`'s `900M`-token PPMI start (`7.1643`)—`Z5` closes **`92.0%` of the initial cold-start deficit online**, reaching **`5.2807` Val CE (`196.51` PPL, `24.34%` Top-1, `41.04%` Top-5, `3.61 / 12` active modes, `p_noclick = 0.380`)** at Step 1,200.

#### Table 11A: `EXPERIMENT_PLAN_081` Zero-PPMI Byte-Pair Cholesky Summary (`Z0..Z5`, `0` Offline Tokens Scanned)

| Variant | Rank $R$ | Pre-Scan Tokens (`init_ms`) | Step-0 Val CE | **Final Zero-PPMI Val CE (`PPL`)** | **Top-1 / Top-5 Acc** | **Active Modes** | **Trunk $p_{\text{noclick}}$ (`mean / id / sem`)** | **Wall Time (`s`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Z0_ZERO_PPMI_COLD_START_BASELINE` (`300` st) | `8` | `0` (`230.8 ms`) | `9.1425` | `6.2710` (`529.00`) | `16.76% / 32.16%` | `2.07 / 8` | `0.319 / 0.193 / 0.337` | `879.4s` |
| `Z1_ZERO_PPMI_ONLINE_EMA_ZIPF` (`300` st) | `8` | `0` (`18.9 ms`) | `9.1425` | `6.2271` (`506.27`) | `16.96% / 32.59%` | `2.08 / 8` | `0.300 / 0.147 / 0.322` | `880.6s` |
| **`Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR`** (`300` st) | **`8`** | **`0` (`18.9 ms`)** | `9.1356` | **`6.1292` (`459.07`)** 🏆 | **`18.37% / 32.92%`** 🏆 | `1.70 / 8` | `0.279 / 0.145 / 0.298` | `885.3s` |
| **`Z3_ZERO_PPMI_EXACT_CHAIN_RULE_TWOBYTE`** (`300` st) | `8` | `0` (`19.5 ms`) | **`8.9229`** | `6.7474` (`851.87`) | `17.07% / 30.89%` | `2.19 / 8` | `0.300 / 0.201 / 0.314` | **`255.8s`** (`3.46x` faster) |
| `Z4_ZERO_PPMI_HIERARCHICAL_BYTE_PAIR_R12` (`300` st) | `12` | `0` (`18.4 ms`) | `9.1344` | `6.1490` (`468.24`) | `17.81% / 32.80%` | `1.64 / 12` | `0.293 / 0.194 / 0.308` | `1264.2s` |
| **`Z5_ZERO_PPMI_BYTE_PAIR_SCALING_1200`** (`1200` st) | **`12`** | **`0` (`18.3 ms`)** | `9.1344` | **`5.2807` (`196.51`)** 🏆 | **`24.34% / 41.04%`** 🏆 | **`3.61 / 12`** | **`0.380 / 0.335 / 0.386`** | `5004.0s` |

---

## 6. Steps We Could Take Next (Prioritized Pure-Neural Research Roadmap)

With `EXPERIMENT_PLAN_078.md` (`V0..V4`), `EXPERIMENT_PLAN_079.md` (`R0..R6`), `EXPERIMENT_PLAN_080.md` (`N1..N4, B0..B2, S0..S1`), and `EXPERIMENT_PLAN_081.md` (`Z0..Z5`) completed and validated on `FineWeb`:

1. **Bucket-Gathered Exact Chain-Rule Two-Byte Readout (`Combining Z2's 6.1292 Accuracy with Z3's 3.46x Speedup`)**:
   - In `ExactChainRuleTwoByteReadout`, replace the shared `256 x 512` low-byte table with a bucket-indexed gather of the `256` token-specific detector rows $W_{\text{det}}[256\,b_{\text{hi}}(y_t) + b_{\text{lo}}]$ synthesized by `BytePairSchurWavepacketSynthesizer` (`197` high-byte outcomes + `256` bucket-specific low-byte outcomes = `453` dot products per token instead of `50,304`). This gives every $(b_{\text{hi}}, b_{\text{lo}})$ token its own dedicated `512D` detector while retaining `Z3`'s **`3.46x` training speedup** and exact $\sum_{v=0}^{50,256} P(v \mid h_t) \equiv 1.0$ normalization.
2. **Per-Mode Normalized `softmax`-Free Vacuum-Born Mixture (`Closing the 0.039-Nat Gap Between B2 and N2`)**:
   - Normalize each Cholesky mode's $2p$-boson Fock intensity across the vocabulary ($P_{\text{Born}, t, r}(v) = \frac{I_{\text{sem}, t, v}\Phi_{2p_{\text{id}}}(z_{\text{id}, t, r, v})}{\sum_{v'} I_{\text{sem}, t, v'}\Phi_{2p_{\text{id}}}(z_{\text{id}, t, r, v')}}$) *before* mixing via $\sum_{r=0}^{R-1} \pi_{t,r} P_{\text{Born}, t, r}(v)$, eliminating "unnormalized mode sum starvation" when excited modes rotate partially into $\mathcal{H}_{\text{vac}}$.
3. **`4,800`-Step (`39.3M`-Token) Strict Speedrun-Compliant Zero-PPMI Scaling Toward `< 4.4` Val CE**:
   - Scale the Bucket-Gathered Zero-PPMI Byte-Pair + $R = 12$ No-Click Vacuum Cholesky architecture to `4,800` steps (`39.3M` tokens) on Shards `1..9`.
