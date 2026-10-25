# QuantumDLBlocks: Project Synthesis, Mathematical Root-Cause Analysis, Experimental Outcomes, and Next-Step Roadmap

**Date**: October 23, 2026  
**Project**: `QuantumDLBlocks` (Quantum-Inspired Deep Learning Building Blocks for Language & Vision)  
**Hardware Context**: Local `NVIDIA GeForce RTX 5060 Ti` (strictly capped at `<= 8.0 GB` VRAM via `torch.cuda.set_per_process_memory_fraction(0.5, device=0)`), `16`-core CPU, `125 GiB` System RAM (with remote Colab GPU/TPU fleet tooling paused while credits are replenished).

---

## 1. Executive Summary & Project Vision

**`QuantumDLBlocks`** investigates whether core mathematical structures from quantum mechanics and harmonic analysis can serve as first-class, parameter-efficient building blocks for modern deep learning architectures—and whether their optimization and calibration failure modes can be diagnosed and solved from first principles rather than brute-force parameter scaling.

Instead of treating a neural network as unconstrained Euclidean vector transformations terminated by an independent dense linear classification head ($W_{\text{out}} \in \mathbb{R}^{V \times d}$), our architectures represent token or image states as **unit-normalized wavefunctions** $\psi \in \mathbb{S}^{d-1} \subset \mathbb{R}^d$ ($\|\psi\|_2 = 1$) that evolve and are measured through five core quantum-inspired building blocks:

### 1.1 Multi-Scale Dyadic Wavelet-Cholesky Hierarchies (`Tied2Pow16WaveletCholeskyPyramid` & `TiedDyadicWaveletClassPyramid`)
- **Motivation**: A standard Transformer with hidden dimension $d = 256$ and GPT-2 vocabulary size $V_{\text{valid}} = 50,257$ (padded to $V = 50,304 = 786 \times 64$ for GPU tensor-core alignment, with indices $50,257 \dots 50,303$ masked out during loss computation) requires $50,304 \times 256 = 12.88\text{M}$ parameters just for a flat lookup table.
- **Spectral Dyadic Tree Assignment (`perm_v_to_leaf`)**: During initialization (`initialize_radical_2pow16_with_ppmi_svd`), we compute 256-dimensional Positive Pointwise Mutual Information (PPMI) SVD co-occurrence embeddings from `FineWeb-Edu` and recursively partition the vocabulary via top-down spectral bisection (`build_dyadic_2pow16_spectral_tree`) into the leaves of a **binary dyadic tree** of size $N = 2^{16} = 65,536$. Each token $v \in \{0, \dots, V-1\}$ is assigned a unique leaf index $\ell = \text{perm}[v] \in \{0, \dots, 65,535\}$ (stored as buffer `perm_v_to_leaf`), so semantically related words share parent nodes across dyadic scales.
- **Explicit Coarse-to-Fine Wavelet Synthesis Equation**:
  In our `1-bank` `Radical2Pow16WaveletCholeskyGPT` (`Waves 22–25`, `10.40M` total parameters), a token's leaf wavefunction $\psi_{\text{self}}(v) \in \mathbb{S}^{d-1}$ ($d = 256$) is synthesized from four dyadic wavelet tables and four block-Cholesky SPD operators ($C_s = L_s L_s^\top$):
  1. **Coarse root scale ($256$ nodes)**: $W_{256} \in \mathbb{R}^{256 \times 256}$, indexed by $\ell_0 = \lfloor \ell / 2^8 \rfloor = \ell \gg 8 \in \{0, \dots, 255\}$.
  2. **Coarse-mid residual scale ($2,048$ nodes)**: $W_{2048} \in \mathbb{R}^{2048 \times 256}$, indexed by $\ell_1 = \lfloor \ell / 2^5 \rfloor = \ell \gg 5 \in \{0, \dots, 2047\}$.
  3. **Mid detail subspace ($8,192$ nodes, rank $d_{\text{mid}} = 64$)**: $W_{8192} \in \mathbb{R}^{8192 \times 64}$ lifted into $\mathbb{R}^{256}$ via row-orthonormal Stiefel basis $U_{8192} \in \mathbb{R}^{64 \times 256}$ ($U_{8192} U_{8192}^\top = I_{64}$), indexed by $\ell_2 = \lfloor \ell / 2^3 \rfloor = \ell \gg 3 \in \{0, \dots, 8191\}$.
  4. **Fine leaf detail subspace ($65,536$ leaves, rank $d_{\text{fine}} = 64$)**: $W_{65536} \in \mathbb{R}^{65536 \times 64}$ lifted into $\mathbb{R}^{256}$ via row-orthonormal Stiefel basis $U_{65536} \in \mathbb{R}^{64 \times 256}$ ($U_{65536} U_{65536}^\top = I_{64}$), indexed directly by $\ell \in \{0, \dots, 65535\}$.
  With learnable scale gains $g = (g_0, g_1, g_2, g_3) \in \mathbb{R}^4$, the leaf vector is:
  $$y(\ell) = g_0 \, C_{256}\!\left(W_{256}[\ell \gg 8]\right) + g_1 \, C_{2048}\!\left(W_{2048}[\ell \gg 5]\right) + g_2 \, C_{8192}\!\left(W_{8192}[\ell \gg 3]\right) U_{8192} + g_3 \, C_{65536}\!\left(W_{65536}[\ell]\right) U_{65536}$$
  $$\psi_{\text{self}}(v) = \frac{y(\text{perm}[v])}{\|y(\text{perm}[v])\|_2}$$
  *(Note on `16-bank` vs. `1-bank`*: In `Wave 21` (`15.59M` params), $U_{65536}$ was partitioned into $16$ separate Stiefel banks of shape $(16, 64, 256)$—one per $4,096$-leaf subtree—and the model also stored a $(2048, 256)$ predecessor bigram table `pred_wav_2048`. In `Waves 22–25` (`10.40M` params), we unified $U_{65536}$ into a single shared $(64, 256)$ Stiefel basis and gathered $\ell = \text{perm}[0:V]$ *before* multiplying by $U_{65536}$, saving `5.19M` parameters and `23.3%` synthesis FLOPs.)*

### 1.2 Sweldens Lifting Wavelet & Block-Cholesky SPD Operators (`WaveletCholeskyMultiscaleBranch` & `BlockCholeskyLinear`)
- **Block-Cholesky SPD Operator (`BlockCholeskyLinear`)**:
  Partitions a $d$-dimensional vector into $G = d / B_{\text{blk}}$ blocks of size $B_{\text{blk}}$ (e.g., $d = 256, B_{\text{blk}} = 32 \implies G = 8$ blocks). Each block $g \in \{1, \dots, G\}$ is parameterized by a lower-triangular Cholesky factor $L_g \in \mathbb{R}^{B_{\text{blk}} \times B_{\text{blk}}}$ with strictly positive diagonal $(L_g)_{ii} = \operatorname{softplus}(\text{raw\_diag}_{g,i}) + 10^{-4}$ and strictly lower-triangular entries $\operatorname{tril}(\text{raw\_tril}_g, -1)$. The layer applies the Symmetric Positive-Definite (SPD) operator $A_g = L_g L_g^\top \succ 0$ to an input batch $X \in \mathbb{R}^{N \times d}$ (where $N = B \cdot T$ is total tokens in the batch) via two fused batched matrix multiplications (`torch.bmm`) on reshaped $(G, N, B_{\text{blk}})$ tensors:
  $$Y_g = (X_g L_g) L_g^\top \quad \Longleftrightarrow \quad y = A x, \quad A = \operatorname{diag}(L_1 L_1^\top, \dots, L_G L_G^\top) \succ 0$$
- **Sweldens Lifting Wavelet Transform (`WaveletCholeskyMultiscaleBranch`)**:
  At each scale level $\ell \in \{1, \dots, L_{\text{wav}}\}$, the current coarse representation $s^{(\ell-1)} \in \mathbb{R}^{N \times d_\ell}$ is split into even channels $x_e = s^{(\ell-1)}_{:, 0::2}$ and odd channels $x_o = s^{(\ell-1)}_{:, 1::2} \in \mathbb{R}^{N \times (d_\ell / 2)}$ and transformed via Sweldens lifting:
  $$\text{Detail (Predict): } d^{(\ell)} = C_{\text{det}}^{(\ell)}\!\left(x_o - P^{(\ell)}(x_e)\right), \qquad \text{Coarse (Update): } s^{(\ell)} = C_{\text{coarse}}^{(\ell)}\!\left(x_e + U^{(\ell)}(d^{(\ell)})\right)$$
  After $L_{\text{wav}}$ levels, the multi-scale wavelet coefficients $[s^{(L_{\text{wav}})}, d^{(L_{\text{wav}})}, \dots, d^{(1)}] \in \mathbb{R}^{N \times d}$ are concatenated and inverted back to the signal domain through inverse lifting steps.

### 1.3 Mach-Zehnder Two-Beam Phase Interferometers (`OvercompleteTwoBeamInterference`)
- Given two unit-normalized pathway states $\psi^{(1)}, \psi^{(2)} \in \mathbb{S}^{d-1}$ (for example, a causal attention / spatial-mixed pathway and a multi-scale wavelet pathway), the interferometer optionally lifts them into an overcomplete frame of dimension $d_{\text{frame}} = \lfloor \alpha_{\text{exp}} d \rfloor$ (where $\alpha_{\text{exp}} \ge 1.0$ is the frame expansion ratio, hence the name `OvercompleteTwoBeamInterference`) and applies a unitary $2 \times 2$ Mach-Zehnder beamsplitter with learnable per-channel mixing angles $\theta \in \mathbb{R}^d$ and relative phase gate $\phi(\psi^{(1)}, \psi^{(2)}) = \cos(\Delta \varphi)$:
  $$\begin{pmatrix} u^{(1)} \\ u^{(2)} \end{pmatrix} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \psi^{(1)} \\ \psi^{(2)} \end{pmatrix}, \qquad \psi_{\text{out}} = \operatorname{Norm}\!\left(u^{(1)} + \cos(\Delta \varphi) \odot u^{(2)}\right) \in \mathbb{S}^{d-1}$$

### 1.4 Layer Dataflow in Language (`Radical2Pow16WaveletCholeskyGPT`) and Vision (`WaveletCholeskyVisionClassifier`)
- **`Radical2Pow16WaveletCholeskyGPT` (`10.40M` params, $d = 256$, $L = 6$ layers, $H = 8$ heads)**:
  Organizes its $6$ layers in a `2 + 2 + 2` Kronecker/U-Net hierarchy:
  - *Layers 0–1 (Front) & Layers 4–5 (Back)* (`RadicalWaveletCholeskyBlock`): Input wavefunction $\psi_t \in \mathbb{S}^{256-1}$ passes through causal `SubspaceProjectorBornAttention` (giving $\psi_{\text{attn}}$) and `WaveletCholeskyMultiscaleBranch` (giving $\psi_{\text{wav}}$), which are recombined via `OvercompleteTwoBeamInterference` to update both the primary sharp state $\psi_t$ and a co-propagating diffuse bath state $\psi_{\text{diff}, t}$.
  - *Layers 2–3 (Middle)* (`WaveletCholeskyUNetBlock`): Decomposes $\psi_t$ into coarse ($d/8 = 32$), mid ($d/4 = 64$), and fine subbands, applies causal shifted `BlockCholeskyLinear` mixing on the low-frequency subbands, upsamples via inverse lifting, and interferes the result with attention via `OvercompleteTwoBeamInterference`.
- **`WaveletCholeskyVisionClassifier` ($d = 128$, $L = 3$ layers, $4 \times 4$ patches $\to 64$ spatial tokens)**:
  Each `WaveletCholeskyVisionBlock` applies `RMSNorm` $\to$ `WaveletCholeskyMultiscaleBranch` + Stiefel channel mixer ($h_{\text{wav}}$) $\to$ orthogonal $64 \times 64$ spatial token mixer ($h_{\text{spatial}}$) $\to$ Mach-Zehnder two-beam phase-interference gate ($\cos\theta \, h_{\text{wav}} + \sin\theta \, h_{\text{spatial}} + \gamma_{\text{int}} (h_{\text{wav}} \odot \tanh(h_{\text{spatial}}))$) $\to$ residual SwiGLU MLP.

### 1.5 Two-Beam Born-Rule & Stiefel Kraus Bath Readouts
- Output probabilities are governed by a two-channel quantum measurement on the final hidden state $\psi_t \in \mathbb{S}^{d-1}$:
  1. **Sharp (Rank-1) Channel**: Measures the centered alignment $\hat{c}_{t,v} = c_{t,v} - \mu_t$ where $c_{t,v} = \langle \psi_t \mid m_{\text{sharp}, v} \rangle$ and $\mu_t = \frac{1}{V} \sum_{u=1}^V c_{t,u}$.
  2. **Diffuse (Rank-$R$) Stiefel Kraus Bath Channel**: Parameterized by $R$ orthogonal Stiefel matrices $K_1, \dots, K_R \in \mathbb{R}^{d \times d}$ ($K_r^\top K_r = I_d$; $R = 16$ in language, $R = 8$ in vision) and spectral-floor-regularized probability weights $w_1, \dots, w_R > 0$ ($\sum_{r=1}^R w_r = 1$). To avoid materializing $d \times d$ density matrices $\rho_t = \psi_t \psi_t^\top$ at every token, we form the coherent effective Kraus operator $\bar{K} = \sum_{r=1}^R \sqrt{w_r} K_r \in \mathbb{R}^{d \times d}$, compute the diffuse inner products $u_{t,v} = \langle \psi_{\text{diff}, t} \mid \operatorname{Norm}(\bar{K} m_{\text{sharp}, v}) \rangle$, and evaluate the rectified-quadratic diffuse fidelity:
     $$f_{\text{diff}, t, v} = u_{t,v} + \alpha_{\text{quad}} \max(u_{t,v}, 0)^2 \quad (\alpha_{\text{quad}} = 0.5), \qquad \hat{f}_{\text{diff}, t, v} = f_{\text{diff}, t, v} - \frac{1}{V} \sum_{u=1}^V f_{\text{diff}, t, u}$$

### 1.6 Riemannian Stiefel-Manifold Optimization (`StiefelMuon`)
- Orthogonal basis matrices (`up_8192`, `up_65536`, `kraus_generators`, attention projections) are optimized using **`StiefelMuon`**:
  1. **Update Orthogonalization (`polar_express`)**: The Nesterov momentum update matrix $G_t$ is orthogonalized via a 5-step Newton-Schulz polynomial iteration (`polar_express`) so the step direction itself has unit singular values.
  2. **Continuous Riemannian Stiefel Landing Penalty (`stiefel_penalty = 0.02`)**: Before applying the step, a smooth Riemannian landing gradient $\gamma_{\text{stiefel}} W (W^\top W - I)$ is added to pull the weight matrix $W$ itself toward the Stiefel manifold $W^\top W = I$ continuously at every step.
  3. **Norm-Damped Stiefel Retraction on `kraus_generators`**: Every 2 steps, `retract_kraus_generators()` applies a norm-damped landing update $K_r \leftarrow K_r - \eta_{\text{land}} K_r (K_r^\top K_r - I) / (1 + \|K_r^\top K_r - I\|_F / \sqrt{d})$ (falling back to exact Polar SVD $U V^\top$ only if the mean Stiefel error $\text{kraus\_stiefel\_err} = \frac{1}{R}\sum_{r=1}^R \|K_r^\top K_r - I_d\|_F / \sqrt{d}$ exceeds `0.03`).

### What Outcomes We Are Seeking
1. **Breaking the `FineWeb-Edu` Cross-Entropy Wall**:
   - Our long-term project benchmark reference is the `FineWeb-Edu` `100M`-token Slowrun (`3.183` Val CE) and Modded-NanoGPT Speedrun (`3.280` Val CE) on the official `4,999,488`-token validation shard (`fineweb_val_000000.bin`).
   - For fast local iteration under our strict `<= 8.0 GB` VRAM cap on a single RTX 5060 Ti, we run standardized `300`-step (`2.46M` tokens, `~1.7 min`), `900`-step (`7.37M` tokens, `~4.9 min`), and `3,600`-step (`29.49M` tokens, `~19.2 min`) scaling horizons where all models see the exact same token stream (`fineweb_train_000001.bin`, $B=16, T=512$). Prior to that session, our all-time `3,600`-step local record was `Wave 21` at `5.0211` Val CE (`40.3 min`).
2. **Resolving the "High Top-1 Accuracy, Bad Cross-Entropy" Paradox**:
   - Understanding from first principles why our quantum models achieved high Top-1 accuracy (matching or beating much larger models) while suffering inflated Cross-Entropy (`> 5.14` nats in `Wave 23`), and eliminating the root cause so Cross-Entropy drops in lockstep with Top-1 accuracy.
3. **Using Fast-Loop Multi-Class Vision Benchmarks (`CIFAR-100` & `ILSVRC ImageNet-1K`) as Controlled Ablation Laboratories**:
   - Reproducing the high-Top-1 / bad-CE pathology in `< 12 seconds` per training run on $100$-class and $1,000$-class classification ($K > d = 128$) and running leave-one-out ablations to measure the exact contribution of each quantum building block.

---

## 2. Core Problems Investigated & Mathematical Root Causes (`Waves 23–25`)

### Problem 1: Why Does High Top-1 Accuracy Coexist with Poor Cross-Entropy When $V \gg d$? (The Parseval Frame Floor & Anti-Alignment Squaring Paradox)
- For any collection of $V$ unit vectors $\{m_v\}_{v=1}^V \subset \mathbb{S}^{d-1}$ with Gram matrix $G = M^\top M \in \mathbb{R}^{d \times d}$ ($\operatorname{Tr}(G) = V$), when $\{m_v\}$ forms a tight frame ($M^\top M = \frac{V}{d} I_d$), for **every** unit query state $\psi_t \in \mathbb{S}^{d-1}$:
  $$\sum_{v=1}^V \langle \psi_t \mid m_v \rangle^2 = \frac{V}{d} \implies \mathbb{E}_{v \ne y}\!\left[\langle \psi_t \mid m_v \rangle^2\right] = \frac{V/d - 1}{V - 1} \approx \frac{1}{d}$$
- Under a linear pure-state Born rule $P(v \mid \psi_t) = \langle \psi_t \mid m_v \rangle^2 / (V / d)$, even when $\langle \psi_t \mid m_y \rangle^2 = 1$, $P(y \mid \psi_t) \le d / V \implies \text{CE}_{\text{linear Born}} \ge \ln(V / d)$.
- Worse, squaring $c_{t,v}^2$ rectifies anti-aligned words ($c_{t,v} < 0$) into positive excitations with a vanishing gradient $2c_{t,v} \to 0$ at $c_{t,v} = 0$.

### Problem 2: Concave $k$-Boson Saturation vs. Convex-Peak Traceless Born (`Waves 23–25`)
- Removing the concave $k$-boson wrapper $\ell_{t,v}^{\text{W23}} = k \ln(1 + \tilde{r}_t \widehat{O}_{t,v} + \lambda_{\text{diff}} f_{\text{diff}, t, v}) + b_v$ in favor of the $k \to \infty$ coherent limit (`convex_traceless_born`) and 256-root / superclass conditional factorization (`dyadic_hierarchical_born`) eliminated peak-gradient compression and cut `1,000`-Class Test CE by **`169x`** (`2.4688` $\to$ `0.0146`) and `ILSVRC ImageNet-1K` Val CE by **`-1.1268` nats** (`6.4035` $\to$ `5.2767`).

---

## 3. Summary of `Waves 23–25` Experimental Outcomes

| Wave / Model (`FineWeb-Edu`, `<= 8.0 GB` VRAM) | Params | Step 300 CE (`2.46M` Tok) | Step 900 CE (`7.37M` Tok) | Step 3,600 CE (`29.49M` Tok) | Step 3,600 Top-1 / Top-50 | Throughput (`tok/s`) | 3,600-Step Wall Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Wave 25` (`w25_8gb_convex_traceless_born`)** | **`10.40M`** | **`6.1478`** | **`5.4465`** | **`5.0162`** | **`25.25%` / `64.65%`** | **`25,616`** | **`19.19 min`** |
| **`Wave 25` (`w25_8gb_last_layer_meas_convex_traceless`)** | **`10.40M`** | `6.1610` | **`5.4637`** | — | — | **`25,247`** | — |
| `Wave 21` (`w21_8gb_2pow16_16bank` — Prev Record) | `15.59M` | `6.1668` | `5.4734` | `5.0211` | `25.19%` / `64.53%` | `12,195` | `40.30 min` |
| `Wave 22` (`w22_8gb_radical_2pow16_1bank_fast`) | `10.40M` | `6.2375` | `5.5613` | `5.0688` | `24.83%` / `64.08%` | `21,980` | `22.36 min` |
| `Wave 20` (`w20_8gb_wavelet_cholesky_19p24M`) | `19.24M` | `6.1868` | `5.5253` | `5.0858` | `24.66%` / `63.85%` | `19,350` | `25.40 min` |
| `Wave 19` (`Parity-Split QuantumBornGPT`) | `38.28M` | `6.2910` | `5.5840` | `5.1157` | `24.21%` / `63.20%` | `16,800` | `29.25 min` |
| `Wave 23` (`w23_8gb_last_layer_meas_traceless` Concave) | `10.40M` | `6.2989` | `5.6818` | `5.1479` | `25.17%` / `64.48%` | `23,350` | `21.05 min` |
| `Classical Baseline` (`30.95M`–`53.54M` Transformer) | `30.95M` | `6.5447` | `5.9152` | `5.5202` | `21.32%` / `58.45%` | `19,100` | `25.73 min` |

*(See [`PROJECT_SUMMARY_WAVES_26_TO_32.md`](file:///home/festivity/agy/QuantumDLBlocks/PROJECT_SUMMARY_WAVES_26_TO_32.md) for the subsequent synthesis covering `Waves 26–32` / `Experiment Plans 041–047`.)*
