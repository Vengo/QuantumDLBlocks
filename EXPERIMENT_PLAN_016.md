# Experiment Plan 016: Project Synthesis, Closing the `3.28` NanoGPT Speedrun Gap, Quantum-Classical Systems/Architectural Unification, and Optimizer Ablations

---

## 1. Executive Project Overview: What We Are Building & The Outcomes We Seek

### 1.1 What Is `QuantumDLBlocks`? (`QuantumBornGPT` vs. `ClassicalModdedGPT`)
`QuantumDLBlocks` is a dual-backend (`PyTorch + Triton` on CUDA GPUs and `JAX + Equinox + Haliax + Pallas` on Google Cloud TPUs) deep learning research framework investigating **Quantum-Geometric Language Modeling (`QuantumBornGPT`)** alongside a modernized **Single-GPU Classical Speedrun Transformer (`ClassicalModdedGPT`)**.

In a standard classical Transformer (such as GPT-2 or `KellerJordan/modded-nanogpt`), hidden states live in unconstrained Euclidean space $\mathbb{R}^d$, causal self-attention weights are computed via exponentiated dot products ($\text{softmax}(q_i^\top k_j / \sqrt{d_h})$), feed-forward layers apply unconstrained affine maps with $\text{ReLU}^2$ or $\text{GELU}$ activations, and final vocabulary logits are unconstrained linear projections $\ell = W_{\text{lm}} h \in \mathbb{R}^V$.

By contrast, **`QuantumBornGPT`** formulates sequence modeling strictly on **unit-norm quantum state manifolds** — real unit spheres $\mathbb{S}^{d-1} = \{\psi \in \mathbb{R}^d : \|\psi\|_2 = 1\}$ — and **Stiefel manifolds** $\text{St}(d, k) = \{W \in \mathbb{R}^{d \times k} : W^\top W = I_k\}$:

1. **Pure State Ket Representation**: Every token at every layer $l \in \{0, \dots, L\}$ is represented as a unit-norm pure state ket $|\psi_{b,t}^{(l)}\rangle \in \mathbb{S}^{d-1}$ ($\||\psi_{b,t}^{(l)}\rangle\|_2 = 1$), maintained via spherical normalization $\mathcal{N}(x) = x / (\|x\|_2 + \epsilon)$ (with $\epsilon = 10^{-8}$). For our default Hilbert space dimension $d = 1024 = 2^{10}$, each token state corresponds to a **10-qubit pure quantum state**.
2. **Isometric Hilbert Lift from a Compact Seed Space**: Rather than allocating an unconstrained $V \times d$ embedding table, each vocabulary token $v \in \{0, \dots, V-1\}$ ($V = 50,304$ for GPT-2 / FineWeb-Edu) is prepared as a unit vector $|s_v\rangle \in \mathbb{S}^{D_{\text{seed}}-1}$ in a lower-dimensional **seed space** ($D_{\text{seed}} \le d$, e.g., $D_{\text{seed}} \in \{128, 256, 384, 512, 1024\}$) and lifted into the full Hilbert space $\mathbb{S}^{d-1}$ via a semi-orthogonal Stiefel lift matrix $W_{\text{lift}} \in \mathbb{R}^{d \times D_{\text{seed}}}$ ($W_{\text{lift}}^\top W_{\text{lift}} = I_{D_{\text{seed}}}$):
   $$|\psi_0(v)\rangle = \mathcal{N}(W_{\text{lift}} |s_v\rangle) \in \mathbb{S}^{d-1}$$
3. **Born-Rule Quantum Interference Attention (`SubspaceProjectorBornAttention`)**: Instead of exponentiating dot products via Softmax, causal self-attention probabilities in each head $h \in \{1, \dots, H\}$ (with per-head dimension $d_h = d / H$) are governed by the **quantum Born rule** — the squared overlap between unit-norm Unitary-RoPE-rotated query and key kets $|q_i\rangle, |k_j\rangle \in \mathbb{S}^{d_h - 1}$ — followed by L1 probability normalization over causal positions $j \le i$ and spherical L2 normalization of the aggregated value kets $|v_j\rangle \in \mathbb{S}^{d_h - 1}$:
   $$w_{i,j} = \gamma_h \,\big(\max(0, \langle q_i | k_j \rangle - \tau_h)\big)^2 \cdot \mathbf{1}[j \le i], \qquad A_{i,j} = \frac{w_{i,j}}{\sum_{t \le i} w_{i,t} + \epsilon_{\text{pur}}}, \qquad |o_i\rangle = \mathcal{N}\!\left(\sum_{j \le i} A_{i,j} |v_j\rangle\right) \in \mathbb{S}^{d_h - 1}$$
   where $\gamma_h > 0$ is a learnable per-head sharpness scalar, $\tau_h \ge 0$ is a learnable coherence threshold, and $\epsilon_{\text{pur}} \ge 0$ is a maximally-mixed purification regularizer.
4. **Quantum Wave Interference MLP (`1-Beam` vs. `2-Beam` Overcomplete Tight Frames)**: Instead of a standard Euclidean feed-forward network, each block applies a wave-interference transformation using frame matrices of shape $M \times d$, where $M = \rho \cdot d$ is the intermediate frame dimension and $\rho \in \{1.5, 2.0, 3.0, 4.0\}$ is the **frame expansion factor**:
   - **1-Beam Quadratic Born MLP (`OvercompleteFrameInterference`)**: Uses a single frame matrix $U \in \mathbb{R}^{M \times d}$ with centered quadratic Born probabilities:
     $$m_{1\text{beam}}(\psi) = W_{\text{down}}\!\left((U \psi)^{\odot 2} - \frac{1}{d}\mathbf{1}_M\right)$$
   - **2-Beam Bilinear Interference MLP (`OvercompleteTwoBeamInterference`, Notation `M×2`)**: Uses a pair of row-wise mutually orthogonal frame matrices $U_1, U_2 \in \mathbb{R}^{M \times d}$ (denoted `M×2`, e.g., `1536×2` for $\rho=1.5$, `2048×2` for $\rho=2.0$, `3072×2` for $\rho=3.0$, `4096×2` for $\rho=4.0$) to compute phase-sensitive bilinear interference:
     $$m_{2\text{beam}}(\psi) = W_{\text{down}}\!\left((U_1 \psi) \odot \text{SiLU}(U_2 \psi)\right)$$
   Both combine with the backbone via a gated spherical residual step: $|\psi^{(l)}\rangle = \mathcal{N}(|\psi^{(l-1)}\rangle + g_{\text{res}}^{(l)} \odot \mathcal{N}(m(\psi^{(l-1)})))$.
5. **POVM (Positive Operator-Valued Measure) Measurement Collapse (Readout)**: At the final layer $L$, the contextual Hilbert ket $|\psi_{b,t}^{(L)}\rangle \in \mathbb{S}^{d-1}$ is mapped to a unit-norm readout ket $|\phi_{b,t}\rangle \in \mathbb{S}^{D_{\text{out}}-1}$ (after an orthogonal rotation $|\psi_{\text{final}}\rangle = \mathcal{N}(W_{\text{rot}} |\psi^{(L)}\rangle)$) and measured against unit-norm vocabulary measurement vectors $|m_v\rangle \in \mathbb{S}^{D_{\text{out}}-1}$ ($v \in \{0, \dots, V-1\}$) scaled by an inverse temperature $\beta_{b,t}$ and optional log-unigram prior bias $b_{\text{prior}, v}$:
   $$\ell_{b,t,v} = \beta_{b,t} \langle \phi_{b,t} | m_v \rangle + b_{\text{prior}, v} \quad \Longrightarrow \quad P(v \mid \phi_{b,t}) = \frac{\exp(\ell_{b,t,v})}{\sum_{v'=0}^{V-1} \exp(\ell_{b,t,v'})}$$
   We support four POVM readout modes:
   - **`tied_stiefel`**: $D_{\text{out}} = D_{\text{seed}}$. Collapses $|\psi_{\text{final}}\rangle \in \mathbb{S}^{d-1}$ into seed space $\mathbb{S}^{D_{\text{seed}}-1}$ via $|\phi_{\text{lin}}\rangle = \mathcal{N}(W_{\text{lift}}^\top |\psi_{\text{final}}\rangle)$ (refined by a nonlinear Seed Readout Adapter) and measures against vocabulary vectors with a learned seed-space offset $\Delta_v \in \mathbb{R}^{D_{\text{seed}}}$: $|m_v\rangle = \mathcal{N}(|s_v\rangle + \Delta_v) \in \mathbb{S}^{D_{\text{seed}}-1}$.
   - **`tied_vocab`**: $D_{\text{out}} = D_{\text{seed}}$. Same seed collapse as `tied_stiefel`, but sets $\Delta_v \equiv 0$ so output measurement vectors are strictly identical to input seed kets ($|m_v\rangle = |s_v\rangle$).
   - **`symmetric_dual`**: $D_{\text{out}} = D_{\text{seed}}$. Directly collapses $|\psi^{(L)}\rangle$ via $W_{\text{lift}}^\top$ without $W_{\text{rot}}$ or $\Delta_v$.
   - **`asymmetric` (Full-Rank Hilbert Readout)**: $D_{\text{out}} = d$. Bypasses seed-space collapse at readout entirely ($|\phi_{b,t}\rangle = |\psi_{b,t}^{(L)}\rangle \in \mathbb{S}^{d-1}$) and measures against an independent full-rank POVM frame $M_{\text{full}} \in \mathbb{R}^{V \times d}$ ($|m_v\rangle \in \mathbb{S}^{d-1}$).
6. **Riemannian `StiefelMuon` Optimization**: All 2D internal projection matrices $W \in \mathbb{R}^{M \times N}$ are updated using Nesterov momentum followed by 5-step quintic Newton-Schulz polar orthogonalization (`zeropower_via_newtonschulz5` with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$) and optional continuous Stiefel retraction $W \leftarrow W - \lambda_{\text{st}} W (W^\top W - I)$, while token embeddings, POVM heads, and 1D/scalar gates are trained with decoupled `AdamW`.

---

### 1.2 Quantum Geometric & Systems Telemetry Definitions
Across all runs, we track four quantum-geometric observables alongside standard language modeling metrics (`Train CE`, `Val CE` in natural units / nats, `Val PPL = exp(Val CE)`, `Top-1 / Top-5 Accuracy`, `MedRank`, and `tok/s`):
1. **Readout Effective Quantum Rank ($R_{\text{eff}}(\rho_{\text{seed}})$)**: Given $N = B \times T$ unit-norm readout kets $\{|\phi_i\rangle\}_{i=1}^N \subset \mathbb{S}^{D_{\text{out}}-1}$, we form their empirical density matrix $\rho = \frac{1}{N} \sum_{i=1}^N |\phi_i\rangle\langle\phi_i| \in \mathbb{R}^{D_{\text{out}} \times D_{\text{out}}}$, compute its von Neumann entropy $S_{\text{vN}}(\rho) = -\sum_{k=1}^{D_{\text{out}}} \lambda_k \log \lambda_k$ from the eigenvalues $\{\lambda_k\}$ of $\rho$, and define $R_{\text{eff}}(\rho) = \exp(S_{\text{vN}}(\rho)) \in [1, D_{\text{out}}]$.
2. **10-Qubit Bipartite Entanglement Entropy ($S(\rho_A)$)**: Reshapes each 1024D ket $|\psi\rangle \in \mathbb{S}^{1023}$ into a $2^5 \times 2^5 = 32 \times 32$ bipartite amplitude matrix $\Psi_{A,B}$, traces out subsystem $B$ to form $\rho_A = \Psi_{A,B} \Psi_{A,B}^\top \in \mathbb{R}^{32 \times 32}$, and computes $S(\rho_A) = -\text{Tr}(\rho_A \log \rho_A) \in [0, \ln 32 \approx 3.466\text{ nats}]$.
3. **Mean Stiefel / Frame Isometry Deviation (`IsoErr`)**: Averaged across all 2D internal matrices $W \in \mathbb{R}^{r \times c}$, measures deviation of the smaller dimension's Gram matrix from the identity: $\text{IsoErr}(W) = \|W^\top W - I_c\|_F / \sqrt{c}$ (for $r \ge c$) or $\|W W^\top - I_r\|_F / \sqrt{r}$ (for $r < c$).
4. **Measurement Inverse Temperature ($\beta$)**: The learned (or context-modulated) POVM sharpness $\beta$ scaling directional cosine similarities $\langle \phi_{b,t} | m_v \rangle$.

---

### 1.3 Outcomes We Are Looking to Achieve
1. **Match or Beat the `3.28` Val CE NanoGPT Speedrun Leaderboard Target on a Single GPU (`ClassicalModdedGPT`)**:
   - The `KellerJordan/modded-nanogpt` speedrun benchmark targets **`Val CE <= 3.28`** on the standard `FineWeb10B` GPT-2 validation shard (`fineweb_val_000000.bin`).
   - Within our **`900M`-token (`899.48M` tokens across 9 unique 100M-token shards) FineWeb-Edu training budget**, we want our single-GPU `ClassicalModdedGPT` benchmark to reach or exceed **`3.28` Val CE**, while strictly preserving our **`block_size = 512` (`T_val = 512`) validation invariant** so every historical and future run remains 100% comparable.
2. **Co-Design High-Leverage Speedrun Primitives into `QuantumBornGPT`**:
   - Adapt every effective algorithmic and systems primitive from the classical speedrun leaderboard—Block-Sparse / Sliding-Window + Document-Level Masking (`<|endoftext|>`), Single-Pass Direct Chunked Cross-Entropy, 3-Bank Token Value Embeddings (`ve`), Scalar U-Net Skip Connections (`u_lambda`), Interior Attention Layer Skipping, Half-Truncated Zero-Frequency RoPE, Fused `QKV` Projections + Batched 3D Newton-Schulz, and On-GPU `uint16` Shard Staging—into mathematically principled unit-sphere counterparts for `QuantumBornGPT`, pushing quantum language modeling from `3.8514` down toward `< 3.65` Val CE.
3. **Isolate the Role of Orthogonalized Updates via Controlled Optimizer Ablations**:
   - Test our best architecture across three distinct optimization regimes—`StiefelMuon + AdamW` (with momentum warmup `0.85 -> 0.95`), `Pure AdamW`, and `Mini-Batch SGD + Nesterov Momentum`—to quantify how much of the speedrun advantage comes from Newton-Schulz semi-orthogonal updates versus architectural design.

---

## 2. Chronological Summary of Problems Solved & Steps Taken (`Waves 1–8` / `Plans 001–015`)

### 2.1 Overview of Waves 1–5 (`Plans 001–012`: Unlocking Quantum Geometric Capacity)
- **Wave 1 (`Plans 001–005`, 300M Tokens = 3 Epochs × 100M Shard)**: Established the initial `D_seed = 128` Stiefel-tied `QuantumBornGPT` baseline (`4.3488` Val CE, $R_{\text{eff}} = 27.38$) and built Multi-Token POVM Prediction (`Plan 001`), Seed Dimension Scaling (`Plan 002`), Tangent-Space Stiefel Muon (`Plan 003`), Multi-Beam Rank-$K$ Kraus Readout (`Plan 004`), and Fused Tiled POVM Cross-Entropy (`Plan 005`).
- **Wave 2 (`Plans 006–007`, 300M Tokens)**: Solved strict Stiefel orthogonality rigidity ($W^\top W = I$) by introducing **Overcomplete Tight Frames** (`Plan 006`, regularizing $U^\top U \approx \frac{M}{d} I_d$ at the Welch bound) and the **1-Beam Quadratic Seed Readout Adapter** (`Plan 007`, recovering nullspace energy lost during $d \to D_{\text{seed}}$ collapse), improving Val CE to `4.2367`.
- **Wave 3 (`Plans 008–010`, 300M Tokens)**: Diagnosed why $D_{\text{seed}} = 512\text{–}1024$ suffered subspace starvation under 2-Beam MLPs (`Plan 008`), introduced **Overcomplete Two-Beam Interference Frames (`2048×2`)** (`Plan 009`), and built the **Two-Beam Phase-Sensitive Seed Readout Adapter** (`Plan 010`, adding an odd-parity cubic term $(U_1 \psi)^{\odot 2} \odot \text{SiLU}(U_2 \psi)$ that breaks antipodal sign degeneracy $\pm \psi_\perp \in \ker(W_{\text{lift}}^\top)$), reaching `4.1846` Val CE on 300M tokens.
- **Wave 4 (`Plan 011`, 900M Unique Tokens = 1 Epoch × 9 Shards)**: Replaced 3–9 epoch looping over a single 100M-token shard (which caused a `0.38+ nat` memorization gap) with **`FineWebMultiShardStream`**, streaming 900M unique tokens across 9 shards with zero-latency background thread prefetching (`0.0 ms` wait). Validated Warmup-Stable-Decay (`WSD`) linear cooldown (`-0.345 nats` drop during cooldown) and broke `4.00` Val CE for the first time (`w4_900m_full_synergy512_cosine` at **`3.9223` Val CE**).
- **Wave 5 (`Plan 012`, 900M Unique Tokens, `120M–175M` Parameter Scaling)**: Scaled `QuantumBornGPT` from `66M` up to `120M–175M` parameters across 5 orthogonal axes:
  - **Axis A (`4d` Overcomplete Frame `4096×2` + `D_seed=512` + `2BeamAdapter`, `162.2M` params)**: `w5_900m_frame4d_synergy512` achieved **`3.8765` Val CE (`48.3` PPL, `212,020 tok/s`)**.
  - **Axis B (Full-Rank `50304×1024` Asymmetric POVM Readout + `L=9`, `155.6M` params)**: `w5_900m_asym1024_deep9` jumped $R_{\text{eff}}$ to **`72.62`** and achieved **`3.8825` Val CE (`48.5` PPL)**.

---

### 2.2 Wave 6 (`Plan 013`): Classical Baseline, Karpathy's Overfitting Test, & Dynamic Bloch-Purity Sharpness
- **Problems Investigated**:
  1. How does `QuantumBornGPT` compare head-to-head against a single-GPU `ClassicalModdedGPT` (`RMSNorm`, QK-RMSNorm Causal Softmax Attention with RoPE, $\text{ReLU}^2$ MLP, U-Net skips, softcapped `lm_head` $30 \tanh(z/30)$, Muon + AdamW) on the exact same 900M-token stream?
  2. Can `QuantumBornGPT` pass Karpathy's small-batch overfitting test ($N \in \{1, 8, 32\}$ 512-token records = up to `16,384` tokens), or do unit-sphere constraints limit final-layer probability sharpness?
- **Key Findings & Results**:
  - `QuantumBornGPT` (`D_seed >= 384` with `SeedAdapter`) overfits 32 full 512-token records (`16,384` tokens) to **`99.90%` Top-1 accuracy (`0.0096` Train CE)**, matching `ClassicalModdedGPT` (`99.74%` Top-1, `0.0130` CE).
  - However, we proved the **Constant-Logit-Norm Theorem**: with unit-norm state $|\phi_{b,t}\rangle \in \mathbb{S}^{D_{\text{out}}-1}$, an approximately tight vocabulary frame $M^\top M \approx \frac{V}{D_{\text{out}}} I$, and a single global scalar $\beta$, the logit vector norm $\|\ell_{b,t}\|_2 \approx \beta \sqrt{V / D_{\text{out}}}$ is locked to a constant across all tokens, preventing simultaneous high sharpness on deterministic tokens (`"San " -> "Francisco"`) and low sharpness on high-entropy contexts.
  - Implementing **`ContextualBlochPurityGate` (`--dynamic-povm-beta`)**, which modulates per-token inverse temperature $\beta_{b,t} = \beta_0 \cdot r(\psi_{b,t}) \in [0.25\beta_0, 4.0\beta_0]$, propelled `w6_900m_asym1024_frame3d_dynbeta` to a new all-time `QuantumBornGPT` record of **`3.8514` Val CE (`47.1` PPL)**, while our initial Wave-6 classical baselines reached **`3.6003` Val CE** (`w6_900m_classical_modded_124m`) and **`3.5444` Val CE** (`w6_900m_classical_modded_162m`).

---

### 2.3 Wave 7 (`Plan 014`) & Wave 8 Phase 1 (`Plan 015`): Solving Early-Epoch Quantum Stagnation, Classical Speedrun Tuning, & Cross-Layer Value Residuals
- **Problems Investigated**:
  1. **Why did `QuantumBornGPT` validation loss drop slowly in early epochs (`0–100M` tokens) compared to `ClassicalModdedGPT`?**
     - We identified four compounding early-epoch bottlenecks in `QuantumBornGPT`:
       - **(a) Self-Repetition Attention Collapse (`56.9%` Diagonal Weight)**: At initialization, query and key projections share components along the token's own embedding $|\psi_0\rangle$, causing Born attention $|\langle q_i | k_i \rangle|^2$ to place `56.9%` of its weight on the diagonal $j = i$ (acting as an identity copy instead of gathering context).
       - **(b) Missing Log-Unigram Prior Bias at $\beta_0 = 20$**: Without a log-unigram bias $b_{\text{prior}}$, the initial spherical readout predicts a uniform distribution ($\text{CE} \approx \ln 50304 = 10.83$), spending the first `50M` tokens rotating `50,304` unit vectors just to learn unigram token frequencies.
       - **(c) Ungated $1\text{:}1$ Spherical Residual Mixing**: Adding random unit-norm branch outputs with weight `1.0` at step 0 rotated the token ket by $45^\circ$ per sublayer ($12 \times 45^\circ$ across 6 blocks), scrambling token identity before features formed.
       - **(d) Quadratic Born Attention Tail Noise**: $|\langle q_i | k_j \rangle|^2$ lacked a coherence threshold $\tau_h$ and purification floor $\epsilon_{\text{pur}}$ to suppress $O(1/\sqrt{d_h})$ random overlap noise across long contexts $T = 512$.
  2. **Why did our Wave-6 `ClassicalModdedGPT` reach `3.5444` instead of `3.28`?**
     - Wave-6 used a shallow `L = 6, d = 1024` shape, Gaussian-initialized residual projections with tied embeddings, and a single learning rate (`0.004`) for both embeddings and `lm_head`.
     - In `Plan 014` (Wave-7), we introduced `--zero-init-classical` (untied zero-initialized `lm_head`, zero-initialized `out_proj` and `fc2`, Nesterov `StiefelMuon` with rectangular aspect-ratio scaling $\sqrt{\max(1, d_{\text{out}}/d_{\text{in}})}$, decoupled 3-group `AdamW` with `lr_wte = 0.20, lr_head = 0.008`, and `L = 12, d = 768` GPT-2 Small depth).
  3. **Colinearity of Same-Layer `v_proj(x_0)` vs. True Layer-0 Value Residual $V^{(0)}$ (`Plan 015` / Wave-8)**:
     - Previously, value residual mixing computed $(1-\alpha_l) W_V^{(l)} x_{\text{in}}^{(l)} + \alpha_l W_V^{(l)} x_0 = W_V^{(l)}((1-\alpha_l)x_{\text{in}}^{(l)} + \alpha_l x_0)$, which was 100% colinear with the input highway gate and wasted $(L-1)$ `v_proj` GEMMs per forward pass. In `Plan 015`, we replaced this with true cross-layer $V^{(0)} = W_V^{(0)} x_{\text{in}}^{(0)}$ passed from Layer 0 to Layers $1 \dots L-1$, and enabled our fused SRAM-tiled Triton kernel `flash_purified_born_attention` (`use_flash_born=True`, giving a **$6.41\times$ attention speedup**).
- **Empirical Outcomes from Wave-7 & Wave-8 Phase 1**:
  1. **Wave-7 Classical Speedrun (`w7_900m_classical_speedrun_12L_t1024`) Completed in `67.97 min` (`221,490 tok/s`) at `Val CE = 3.4604` (`31.8` PPL, `37.51%` Top-1)**:
     - Dropped **`-0.2238 nats`** during its 25% WSD cooldown (`3.6842 -> 3.4604`), beating Wave-6 `124M` (`3.6003`) by **`-0.1399 nats`** and Wave-6 `162M` (`3.5444`) by **`-0.0840 nats`** (with unseen streaming `Train CE = 3.4339`).
  2. **Wave-7 `QuantumBornGPT` Early-Epoch Fix Validated (`-0.2016 nats` Faster Early Drop)**:
     - `w7_900m_qborn_earlyfix_frame4d_synergy512` (`--ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn`) dropped `100M`-token Val CE from `4.6328` to **`4.4312` (`-0.2016 nats`)**, and at `600M` tokens (`Step 9150`, *before* its `675M -> 900M` WSD cooldown even begins!) has already reached **`3.9365` Val CE (`51.2` PPL)** — **`-0.20 nats` ahead of Wave-6 at the same token mark**.
  3. **Critical Empirical Discovery from Wave-8 Phase 1 (`--classical-stiefel-penalty 0.005`)**:
     - When we tested `--classical-stiefel-penalty 0.005` on `ClassicalModdedGPT` (`w8_900m_classical_speedrun_12L_t512_layer0v_norm`), clamping 2D weight singular values near `1.0` (`IsoErr = 1.74` vs. `78.31`) *degraded* `200M`-token Val CE from `3.9610` to `4.2312` (`+0.2702 nats`).
     - **Why?** Because `ClassicalModdedBlock` uses **parameterless `F.rms_norm(x)`** (which has no learnable $\gamma$ weight vector). Consequently, the singular values of the 2D weight matrices $(W_1, W_2, W_V, W_O)$ *are* the network's only learnable sublayer gains! Forcing $W^\top W \approx I$ via Stiefel retraction starves classical sublayers of amplitude scaling. Instead, increasing the global batch size via `grad_accum = 8` (`524,288` tok/step, `1,716` total steps for `900M` tokens, matching `modded-nanogpt`'s step count) naturally bounds weight norm accumulation without clamping layer gains.

---

### 2.4 Summary Table of Milestone Runs Across Waves 1–7

| Wave | Run Key | Architecture & Key Innovations | Tokens | Params | `100M` Val CE | `450M` Val CE | Final `900M` Val CE | Val PPL | Top-1 | Speed (`tok/s`) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W7** | **`w7_900m_classical_speedrun_12L_t1024`** | **`ClassicalModdedGPT` (`L=12, d=768`, Zero-Init, WSD)** | **900M** | **`152.7M`** | **`4.1927`** | **`3.7571`** | **`3.4604`** 🥇 | **`31.8`** | **`37.51%`** | **`221,490`** |
| **W6** | `w6_900m_classical_modded_162m` | `ClassicalModdedGPT` (`L=6, d=1024, d_ff=5632`, Cosine) | 900M | `162.8M` | `4.2158` | `3.7668` | `3.5444` | `34.6` | `36.73%` | `233,952` |
| **W7** | `w7_900m_classical_speedrun_6L_zeroinit` | `ClassicalModdedGPT` (`L=6, d=1024`, Zero-Init, WSD) | 900M | `189.4M` | `4.2492` | `3.8810` | `3.5573` | `35.1` | `36.62%` | `165,507` |
| **W6** | `w6_900m_classical_modded_124m` | `ClassicalModdedGPT` (`L=6, d=1024, d_ff=3840`, Cosine) | 900M | `128.1M` | `4.2407` | `3.8178` | `3.6003` | `36.6` | `36.26%` | `185,381` |
| **W6** | **`w6_900m_asym1024_frame3d_dynbeta`** | **`QuantumBornGPT` (`L=8, 3d` Frame, Asym1024, Dyn-$\beta$)** | **900M** | **`180.4M`** | `4.5459` | `4.1229` | **`3.8514`** 🏆 | **`47.1`** | **`33.76%`** | `181,941` |
| **W5** | `w5_900m_frame4d_synergy512` | `QuantumBornGPT` (`L=6, 4d` Frame `4096×2`, `D_seed=512`) | 900M | `162.2M` | `4.6110` | `4.1812` | `3.8765` | `48.3` | `33.50%` | `212,020` |
| **W5** | `w5_900m_asym1024_deep9` | `QuantumBornGPT` (`L=9`, Full-Rank `50304×1024` Asym POVM) | 900M | `155.6M` | `4.5820` | `4.1490` | `3.8825` | `48.5` | `33.34%` | `135,011` |
| **W7** | **`w7_900m_qborn_earlyfix_frame4d_synergy512`** | **`QuantumBornGPT` (`L=6, 4d` Frame + 4-Part EarlyFix, WSD)** | *600M / 900M* | **`162.3M`** | **`4.4312`** | **`4.0592`** | *`3.9365` @ 600M (Pre-Cooldown)* | *`51.2`* | *`32.89%`* | `79,398` |
| **W7** | **`w7_900m_qborn_earlyfix_asym1024_L8`** | **`QuantumBornGPT` (`L=8, 3d` Asym1024 + EarlyFix, WSD)** | *475M / 900M* | **`180.5M`** | **`4.4027`** | **`4.0019`** | *Running (Pre-Cooldown)* | *`54.7`* | *`32.14%`* | `62,075` |
| **W1** | `gpu_slowrun_hyp` (Initial Baseline) | `QuantumBornGPT` (`L=6, 1.5d` 1-Beam, `D_seed=128`) | 300M | `66.1M` | `4.6812` | — | `4.3488` (at 300M) | `77.4` | `29.66%` | `198,850` |

---

## 3. Deep-Dive Technical Analysis of the `3.28` NanoGPT Speedrun Gap & Direct Answers to Key Questions

### 3.1 Why Did Wave-7 Classical Reach `3.4604` Instead of `3.28`? (Missing Systems & Algorithmic Primitives)
Comparing our `ClassicalModdedGPT` (`src/qdlblocks/models/classical_modded_gpt.py`) and `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` against the `KellerJordan/modded-nanogpt` PR #360-era single-node speedrun reveals two categories of missing primitives:

#### Category A: Five Systems & Kernel Throughput Bottlenecks (Slowing Down Wall-Clock `tok/s`)
1. **Double `lm_head` / POVM Readout GEMM from Activation Checkpointing**:
   - In `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`, `compute_step_loss` wraps `compiled_classical_chunk_ce` (and `compiled_chunk_ce`) in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` over `16,384`-token chunks to save VRAM. Because of `checkpoint(...)`, PyTorch executes the massive `(65536, d) @ (d, 50304)` vocabulary projection **twice per step** (once in forward to compute loss, and a second time during backward recomputation). Writing a direct single-pass custom `autograd.Function` that computes chunk logits, accumulates the scalar loss, and writes analytical chunk gradients $(\nabla x_c, \nabla W_{\text{head}})$ in a single forward loop eliminates the second `50,304`-vocab GEMM completely while keeping the exact same `16,384`-token peak memory footprint.
2. **Dense Full-Window Attention Across Document Boundaries vs. Block-Sparse Sliding-Window + `<|endoftext|>` Document Masking**:
   - Currently, every layer runs dense causal attention across the full sequence (`T = 512` or `1024`) even across `<|endoftext|>` (`token 50256`) document boundaries. In `modded-nanogpt`, precomputing a `FlexAttention` `BlockMask` (or Triton tile mask) that prevents cross-document attention (`doc_id[q] == doc_id[kv]`) and uses short sliding windows ($W \in \{128, 384\}$ on most layers, keeping full window on anchor layers) skips $50\%\text{–}75\%$ of attention tiles while eliminating cross-document contamination.
3. **Dense $d \times d$ `unet_gates` Matrices Instead of Scalar Skip Weights (`u_lambda`)**:
   - Currently, `ClassicalModdedGPT` allocates $L/2$ full $d \times d$ matrices (`self.unet_gates`) and runs `F.linear(skip_x, self.unet_gates[enc_idx])` on every decoder block, adding $L/2$ extra $d \times d$ GEMMs per step and $L/2$ extra Newton-Schulz polar decompositions per optimizer step. In `modded-nanogpt`, U-Net skip connections are simple learnable scalar/1D weights (`x = x + u_lambda * skip_x`), costing zero matrix multiplications.
4. **Unfused `Q, K, V` Projections & Sequential Python Loop Over 72+ Matrices in `StiefelMuon`**:
   - Fusing `q_proj, k_proj, v_proj` into a single `qkv_proj = nn.Linear(d, 3*d, bias=False)` merges three kernel launches into one GEMM. Furthermore, stacking all same-shape 2D weight updates across layers into a 3D tensor `(K, M, N)` inside `StiefelMuon.step()` allows the 5-step Newton-Schulz iteration to execute as a single batched `torch.bmm` call per unique matrix shape instead of 72+ sequential calls.
5. **Per-Step CPU `uint16 -> int64` Conversion & Unpinned Host-to-Device Copies**:
   - See Section 3.2 below for a complete precision and memory-bandwidth breakdown.

#### Category B: Four Algorithmic & Architectural Primitives (Improving Sample Efficiency Toward `<= 3.28` Val CE)
1. **3-Bank Dedicated Token Value Embeddings (`ve`) in First 3 & Last 3 Layers**:
   - In `modded-nanogpt`, a bank of 3 token-level embedding tables `ve = nn.ModuleList([nn.Embedding(vocab_size, d) for _ in range(3)])` (optimized by `AdamW` at `lr = 0.20`) is mixed directly into the attention value tensor $V^{(l)}$ of the first 3 layers ($l \in \{0, 1, 2\}$) and last 3 layers ($l \in \{L-3, L-2, L-1\}$). This gives early and late attention heads a direct token-specific value bypass without consuming backbone `v_proj` capacity.
2. **Skipping an Interior Attention Layer (e.g., Layer 6 in a 12-Layer Stack)**:
   - Skipping the attention sublayer in one interior block (`skip_attn_layers = (6,)`, executing only the MLP / two-beam interferometer in that block) saves $\sim 8.3\%$ of attention FLOPs and reduces over-smoothing across middle layers once `ve` and U-Net skips are active.
3. **Half-Truncated / Zero-Frequency RoPE (`zero_half_freq=True`)**:
   - See Section 3.3 below for the exact mathematical formulation and why it benefits both Classical Softmax and Quantum Born attention.
4. **Muon Momentum Warmup (`0.85 -> 0.95` over 300 Steps) & Matched `524,288`-Token Batch Size (`grad_accum = 8`)**:
   - In early steps (`1..300`), gradient directions rotate rapidly; warming up Muon's Nesterov momentum $\mu_t$ from `0.85` to `0.95` over 300 steps prevents stale early gradients from polluting the Newton-Schulz orthogonalizer. Coupling this with `grad_accum = 8` (`524,288` tokens/step = `1,716` steps for `900M` tokens) matches the low-noise large-batch regime of `modded-nanogpt`.

---

### 3.2 Precision Analysis: `uint16 -> int64` Token IDs vs. `bfloat16` Weights
1. **Are we casting `uint16` token IDs to `bfloat16`?**
   - **No—never.** FineWeb-Edu token IDs (`0 .. 50256`) are discrete integer vocabulary indices stored on disk as 2-byte unsigned integers (`np.uint16`). `bfloat16` is a 16-bit floating-point format (1 sign bit, 8 exponent bits, 7 mantissa bits) that can only represent every integer exactly up to $2^8 = 256$; for token IDs in `[32768, 50256]`, adjacent `bfloat16` values are spaced by `256`, which would corrupt vocabulary lookups completely.
   - Instead, PyTorch's `nn.Embedding` and `F.cross_entropy` require 8-byte signed integer indices (`torch.int64` / `long`). Currently, `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` calls `active_shard[start:end].astype(np.int64)` **on the CPU every step**, which allocates an unpinned temporary CPU buffer (forcing `.to(device, non_blocking=True)` to fall back to a synchronous blocking copy) and transfers $4\times$ more bytes (`8 B/tok` vs. `2 B/tok`) over PCIe.
   - **Our Fix**: Transfer each `100M`-token shard (`200 MB` in `uint16`) to GPU HBM **once** when the shard becomes active (`active_shard_gpu` of dtype `torch.uint16`), and slice + cast `active_shard_gpu[start:end].to(torch.int64)` directly inside GPU HBM (`3 TB/s`) every micro-step. This is **100% lossless** (every `uint16` integer in `0..65535` maps bit-for-bit identically to `int64`) and eliminates per-step CPU/PCIe overhead.
2. **Do we experience precision loss with weights in `bfloat16`?**
   - If master weights are stored in pure `bfloat16`, small late-cooldown parameter updates $|\Delta w| < 2^{-8} |w| \approx 0.0039 |w|$ round to zero (`bf16(w + dw) == bf16(w)`, known as **update underflow**).
   - Therefore, our models keep all **master weights in `float32`** so `StiefelMuon` and `AdamW` accumulate updates in full `float32` precision, while `torch.amp.autocast(dtype=torch.bfloat16)` dynamically casts weights to `bfloat16` for TensorCore matrix multiplications.

---

### 3.3 Mathematical Analysis: What Does Half-Truncated / Zero-Frequency RoPE (`zero_half_freq=True`) Do?
In standard `UnitaryRoPE` (`src/qdlblocks/layers/isometric.py`) with per-head dimension $d_h = 64$ ($d_h / 2 = 32$ 2D rotation planes $m \in \{0, \dots, 31\}$), each 2D coordinate pair $(2m, 2m+1)$ of a query $q_i$ or key $k_j$ is rotated by angle $t \theta_m$, where $\theta_m = 10000^{-m / 32}$. Consequently, the inner product between query position $i$ and key position $j$ is:
$$\langle \text{RoPE}(q_i, i), \text{RoPE}(k_j, j) \rangle = \sum_{m=0}^{31} \langle q_{i,m}, R((i-j)\theta_m) k_{j,m} \rangle$$
Under full RoPE ($\theta_m > 0$ for all $m$), **every single coordinate pair rotates with relative distance $(i - j)$**; there is zero position-invariant subspace inside any attention head.

When **`zero_half_freq = True` (Half-Truncated RoPE)** is enabled, we assign log-spaced inverse frequencies $\theta_m = \text{base}^{-m / 16}$ to the first half of the rotation planes ($m \in \{0, \dots, 15\}$, covering `32` dimensions) and set **$\theta_m = 0$** ($\cos 0 = 1, \sin 0 = 0$) on the second half ($m \in \{16, \dots, 31\}$, covering the remaining `32` dimensions):
$$\mathcal{H}_{\text{head}} = \mathcal{H}_{\text{pos}} \oplus \mathcal{H}_{\text{sem}}, \qquad \langle \text{RoPE}(q_i, i), \text{RoPE}(k_j, j) \rangle = \underbrace{\sum_{m=0}^{15} \langle q_{i,m}, R((i-j)\theta_m) k_{j,m} \rangle}_{\text{Position-Sensitive Subspace }\mathcal{H}_{\text{pos}}} + \underbrace{\sum_{m=16}^{31} \langle q_{i,m}, k_{j,m} \rangle}_{\text{Position-Invariant Semantic Subspace }\mathcal{H}_{\text{sem}}}$$
- **Strict Unitarity Preserved**: Because $R(0) = I_2$ is the exact $2 \times 2$ identity matrix, Half-Truncated RoPE remains strictly orthogonal/unitary ($\|\text{RoPE}(\psi)\|_2 = \|\psi\|_2 = 1$).
- **Why This Is Especially Important for `QuantumBornGPT`**: In `SubspaceProjectorBornAttention`, attention weights are proportional to $\max(0, \langle q_i | k_j \rangle - \tau_h)^2$. Under full RoPE, oscillating negative cosines $\cos((i-j)\theta_m) < 0$ at intermediate distances $(i - j)$ frequently drive the total inner product $\langle q_i | k_j \rangle$ below $\tau_h$ (into the `ReLU` dead zone with zero weight and zero gradient) even when tokens $i$ and $j$ have high semantic similarity. By keeping $\mathcal{H}_{\text{sem}}$ unrotated ($\theta_m = 0$), semantically matching tokens maintain a distance-invariant positive carrier amplitude $\sum_{m=16}^{31} \langle q_{i,m}, k_{j,m} \rangle > 0$ across arbitrary context distances!

---

## 4. Observable Hypotheses & Quantitative Pass/Fail Criteria

### Hypothesis 1 (`H1` — Systems & Kernel Roofline Speedup: Single-Pass Chunked CE, Batched 3D Newton-Schulz, Fused `QKV`, Scalar U-Net Skips, On-GPU `uint16` Staging, & Sliding-Window Tile Skipping):
- **Claim**:
  1. Replacing `torch.utils.checkpoint.checkpoint` around the chunked vocabulary cross-entropy with `_DirectChunkedSoftcapLinearCEFn` (`direct_chunked_softcap_linear_ce`) and `_DirectChunkedPOVMCEFn` (`direct_chunked_povm_ce`) computes exact analytical losses and gradients while reducing vocabulary readout GEMM invocations from `2` per step to `1` per step.
  2. Stacking same-shape 2D parameter updates into 3D tensors `(K, M, N)` via `zeropower_via_newtonschulz5_batched` in `StiefelMuon` matches sequential 2D Newton-Schulz outputs while reducing optimizer step latency.
  3. Combining `direct_chunked_ce`, batched 3D `StiefelMuon`, fused `QKV`, scalar U-Net skips (`unet_skip_mode="scalar"`), on-GPU `uint16` shard staging, and sliding-window + document-masked attention delivers a significant end-to-end training throughput (`tok/s`) speedup without loss of numerical accuracy.
- **Observable Metrics & Pass Criteria**:
  1. **Exact Readout Gradient & Single-GEMM Verification**: On random inputs $(N_{\text{tok}} = 4096, d = 256, V = 4096, \text{chunk\_size} = 1024)$, `max_abs_diff(loss_direct, loss_ref) < 1e-5`, gradient cosine similarity `cos(dx_direct, dx_ref) > 0.99999` and `cos(dw_direct, dw_ref) > 0.99999` for both `direct_chunked_softcap_linear_ce` and `direct_chunked_povm_ce`, and the number of forward vocabulary GEMM calls per forward+backward step is `1` (vs. `2` under `checkpoint`).
  2. **Batched 3D Newton-Schulz Parity**: For a stack of $K = 12$ matrices of shape $(256, 256)$, `max_abs_diff(zeropower_via_newtonschulz5_batched(G_3d), stack([zeropower_via_newtonschulz5(G_k)])) < 1e-5`.
  3. **End-to-End Step Throughput Gain**: On a 12-layer model benchmark ($B = 16, T = 512, d = 512, V = 16384$), enabling the Plan 016 systems stack increases end-to-end training step throughput (`tok/s`) by **$\ge 1.30\times$** over the unoptimized baseline.

### Hypothesis 2 (`H2` — Architectural Sample Efficiency: 3-Bank Value Embeddings `ve`, Half-Truncated RoPE, Interior Attention Skip, & Muon Momentum Warmup):
- **Claim**:
  1. Enabling Half-Truncated RoPE (`zero_half_freq=True`) preserves unit norm $\|\text{RoPE}(\psi)\|_2 = \|\psi\|_2$ while keeping the upper $50\%$ of head dimensions invariant across sequence positions $t \in [0, T)$ and increasing positive-cone Born overlap survival $\mathbb{P}(\langle q_i | k_j \rangle > 0)$ for semantically matched tokens across distances $|i - j| \in [16, 256]$.
  2. Combining 3-bank token Value Embeddings (`num_value_embeddings = 3` mixed into layers $\{0, 1, 2\}$ and $\{L-3, L-2, L-1\}$), Half-Truncated RoPE (`zero_half_freq = True`), Scalar U-Net Skips (`unet_skip_mode = "scalar"`), Interior Attention Layer Skipping (`skip_attn_layers = (L // 2,)`), Document-Masked Sliding-Window Attention, and Muon Momentum Warmup (`0.85 -> 0.95`) lowers validation cross-entropy at matched token budgets for **both** `ClassicalModdedGPT` and `QuantumBornGPT`.
- **Observable Metrics & Pass Criteria**:
  1. **Half-Truncated RoPE Invariance & Positive-Cone Survival**: On unit vectors with shared semantic signal across positions $t \in [0, 128)$, the upper half of coordinates $x[..., d_h/2:]$ has position variance $< 10^{-12}$, unit norm error $|\|x_{\text{rope}}\|_2 - 1| < 10^{-6}$, and positive-cone Born energy $\mathbb{E}[\max(0, \langle q_i | k_j \rangle)^2]$ across distances $|i - j| \ge 16$ is **$\ge 1.50\times$** higher than under full RoPE (`zero_half_freq=False`).
  2. **Controlled Multi-Document Language Modeling Improvement**: On a 150-step controlled multi-document language modeling task ($V = 512, d = 128, L = 6, H = 4, T = 64$ with `<|endoftext|>` document boundaries and contextual token dependencies), enabling the Plan 016 architectural stack reduces validation cross-entropy by **$\ge 0.08\text{ nats}$** for both `ClassicalModdedGPT` and `QuantumBornGPT` compared to their respective baselines.

### Hypothesis 3 (`H3` — Controlled Optimizer Ablation: `StiefelMuon + AdamW` vs. `Pure AdamW` vs. `Mini-Batch SGD + Nesterov`):
- **Claim**: On our speedrun-upgraded `ClassicalModdedGPT` and `QuantumBornGPT` architectures at matched token budgets, `StiefelMuon + AdamW` (using batched 5-step Newton-Schulz orthogonalization with momentum warmup `0.85 -> 0.95` on 2D weights) achieves significantly faster convergence and lower validation cross-entropy than both `Pure AdamW` and `Mini-Batch SGD + Nesterov` (which lacks Newton-Schulz singular-value equalization), while `Mini-Batch SGD + Nesterov` and `Pure AdamW` provide clean empirical baselines quantifying the exact nats contributed by Newton-Schulz orthogonalization.
- **Observable Metrics & Pass Criteria**:
  1. **Three-Optimizer Execution & Convergence Ordering**: In a controlled 150-step ablation ($V = 512, d = 128, L = 6, H = 4, T = 64$) across `optimizer_mode in ("muon_adamw", "pure_adamw", "sgd_nesterov")`, all three modes converge stably without NaNs, and `muon_adamw` achieves the lowest final validation cross-entropy — beating `sgd_nesterov` by **$\ge 0.10\text{ nats}$** and `pure_adamw` by **$\ge 0.04\text{ nats}$** while maintaining mean 2D weight condition number $\kappa(W)$ closer to `1.0`.

---

## 5. Planned Code & Architectural Changes

### 5.1 Operational Scripting Classification (`Build-Before-Launch` vs. `Nice-to-Have`)
- **`Build-Before-Launch`**:
  - Existing `scripts/colab_fleet.py` (`ensure-sessions`, `push`, `launch`, `status`, `harvest`, `compare`) and `scripts/monitor_colab_runs.py` already automate fleet deployment and wave comparison.
- **`Nice-to-Have` (Logged to `scripts/BACKLOG.md`)**:
  - Add `--optimizer-filter` to `scripts/colab_fleet.py compare` so optimizer ablation runs can be grouped side-by-side in CLI tables.

---

### 5.2 Component 1: Single-Pass Direct Chunked Cross-Entropy (`src/qdlblocks/layers/direct_chunked_ce.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py` (and export in `src/qdlblocks/layers/__init__.py` and `src/qdlblocks/__init__.py`) containing:
1. `direct_chunked_softcap_linear_ce(x_2d: torch.Tensor, w_lm: torch.Tensor, targets: torch.Tensor, softcap: float = 30.0, chunk_size: int = 16384) -> torch.Tensor`:
   - Custom `torch.autograd.Function` (`_DirectChunkedSoftcapLinearCEFn`) that loops over `start in range(0, N_tok, chunk_size)` in `forward`:
     - Computes `raw = F.linear(xc, w_cast)` (`(chunk, V)`).
     - Applies softcap `u = raw.float() / softcap; tanh_u = torch.tanh(u); logits = softcap * tanh_u` and derivative factor `d_softcap = 1.0 - tanh_u.square()` (or `logits = raw.float(), d_softcap = 1.0` when `softcap <= 0`).
     - Computes `log_probs = F.log_softmax(logits, dim=-1)` and accumulates `F.nll_loss(log_probs, yc, reduction="sum")` into `total_loss`.
     - Computes chunk logit gradient `g_raw = ((log_probs.exp() - one_hot(yc)) * d_softcap / N_tok).to(x_2d.dtype)` in-place via `scatter_add_`.
     - Computes `dx[start:end] = g_raw @ w_cast` and accumulates `dw += g_raw.T @ xc` in `float32`.
   - In `backward(ctx, grad_out)`, simply returns `(dx * grad_out, dw * grad_out, None, None, None)`—executing **zero GEMMs in backward**!
2. `direct_chunked_povm_ce(phi_2d: torch.Tensor, m_seed: torch.Tensor, inv_t: torch.Tensor, targets: torch.Tensor, log_prior: torch.Tensor | None = None, chunk_size: int = 16384) -> torch.Tensor`:
   - Custom `torch.autograd.Function` (`_DirectChunkedPOVMCEFn`) for `QuantumBornGPT` POVM readout:
     - Normalizes `m_norm = l2_normalize(m_seed, dim=-1)` once before the chunk loop.
     - For each chunk `start:end`, computes `cos_sim = F.linear(phi_c, m_norm.to(phi_c.dtype)).float()`, scales by `inv_t_c` (supporting both scalar `(1,)` and per-token `(N_tok, 1)` dynamic-$\beta$ tensors!), adds optional `log_prior`, computes `log_probs = F.log_softmax(logits, dim=-1)`, and accumulates exact analytical gradients `dphi`, `dm_norm`, `d_inv_t`, and `d_log_prior` in a single pass.
     - Applies the exact spherical Jacobian pullback `dm_seed = (dm_norm - m_norm * (dm_norm * m_norm).sum(dim=-1, keepdim=True)) / (||m_seed||_2 + eps)` once on the `(V, D_out)` vocabulary matrix after the chunk loop!

---

### 5.3 Component 2: Block-Sparse Document Masking (`<|endoftext|>`) & Sliding-Window Attention (`src/qdlblocks/kernels/flash_born.py` & `src/qdlblocks/layers/attention.py`)
1. **Document ID Helper (`compute_document_ids`)**:
   - In `src/qdlblocks/layers/attention.py`, add `compute_document_ids(idx: torch.Tensor, eot_token_id: int = 50256) -> torch.Tensor`:
     - Given token indices `idx` of shape `(B, T)`, defines document IDs such that an `<|endoftext|>` token at position $t$ closes the current document (or starts a new segment after $t$):
       ```python
       is_eot = (idx == eot_token_id).to(torch.int32)
       doc_ids = torch.cumsum(is_eot, dim=-1, dtype=torch.int32) - is_eot
       ```
       (so `<|endoftext|>` belongs to the document it terminates, and position $t+1$ increments to the next document ID).
2. **Triton `flash_born_attention` & `flash_purified_born_attention` Sliding-Window + Document Masking (`src/qdlblocks/kernels/flash_born.py`)**:
   - Update `flash_born_attention` and `flash_purified_born_attention` to accept `window_size: int = 0` (`0` = full causal window) and `doc_ids: torch.Tensor | None = None` (`(B, T)` `int32`):
     - In the Triton forward and backward kernels, tighten the inner loop lower bound to `lo = 0 if WINDOW_SIZE <= 0 else tl.maximum(0, (start_m - WINDOW_SIZE + 1) & ~(BLOCK_N - 1))`, skipping all key tiles completely outside the sliding window!
     - When `HAS_DOC_IDS: tl.constexpr` is `True`, load `doc_m = tl.load(DOC_IDS + off_b * stride_doc_b + offs_m)` and `doc_n = tl.load(DOC_IDS + off_b * stride_doc_b + offs_n)` and mask out cross-document pairs (`doc_m[:, None] == doc_n[None, :]`) and out-of-window pairs (`(offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE`).
3. **Reference & Classical Attention Support (`src/qdlblocks/layers/attention.py` & `src/qdlblocks/models/classical_modded_gpt.py`)**:
   - Update `SubspaceProjectorBornAttention.forward` and `ClassicalModdedBlock.forward` to accept `window_size: int = 0` and `doc_ids: torch.Tensor | None = None`.
   - In `ClassicalModdedBlock`, when `window_size > 0` or `doc_ids is not None`, enforce causal + sliding-window (`i - j < window_size`) + document-boundary (`doc_ids[:, i] == doc_ids[:, j]`) masking via `FlexAttention` `BlockMask` (when `use_flex_doc_sliding_attn=True` on CUDA) or an additive boolean attention mask passed into `F.scaled_dot_product_attention(..., attn_mask=mask)`.

---

### 5.4 Component 3: Batched 3D Newton-Schulz, Muon Momentum Warmup, & Multi-Optimizer Support (`src/qdlblocks/functional.py` & `src/qdlblocks/optim/muon.py`)
1. **`zeropower_via_newtonschulz5_batched` (`src/qdlblocks/functional.py`)**:
   - Accepts 2D `(M, N)` or 3D `(K, M, N)` tensors, normalizes each matrix by its Frobenius norm `torch.linalg.matrix_norm(X, ord="fro", keepdim=True) + eps`, transposes via `.mT` if `M > N`, runs 5 steps of batched quintic iteration using `torch.bmm`, and transposes back.
2. **`StiefelMuon` Upgrades (`src/qdlblocks/optim/muon.py`)**:
   - Add `momentum_warmup_steps: int = 0`, `momentum_init: float = 0.85`, and `batched_ns: bool = True` to `StiefelMuon.__init__`.
   - At step $t \ge 1$, compute effective momentum:
     $$\mu_t = \begin{cases} \mu_{\text{init}} + (\mu - \mu_{\text{init}}) \cdot \min\!\left(1.0, \frac{t - 1}{\text{momentum\_warmup\_steps}}\right) & \text{if } \text{momentum\_warmup\_steps} > 0 \\ \mu & \text{otherwise} \end{cases}$$
   - When `batched_ns=True`, group 2D parameters within each param group by `(p.shape, p.device, p.dtype)`, stack their Nesterov-extrapolated gradient buffers `g_in` into a single 3D tensor `(K, M, N)`, call `zeropower_via_newtonschulz5_batched` once per shape group, and apply the aspect-ratio-scaled updates back to each parameter.

---

### 5.5 Component 4: Half-Truncated RoPE, 3-Bank Value Embeddings (`ve`), Scalar U-Net Skips, Fused `QKV`, & Interior Attention Layer Skipping
1. **`UnitaryRoPE(zero_half_freq: bool = False)` (`src/qdlblocks/layers/isometric.py`)**:
   - Add `zero_half_freq: bool = False` to `UnitaryRoPE.__init__`. When `True`, set the first `rot_d = (head_dim // 2) // 2` frequencies to `1.0 / (base ** (arange(0, rot_d) / max(1, rot_d)))` and the remaining `(head_dim // 2) - rot_d` frequencies to `0.0`.
2. **Unified Upgrades to `ClassicalModdedBlock` / `ClassicalModdedGPT` (`src/qdlblocks/models/classical_modded_gpt.py`) and `SubspaceProjectorBornAttention` / `QuantumBornBlock` / `QuantumBornGPT` (`src/qdlblocks/layers/attention.py`, `src/qdlblocks/layers/born_block.py`, `src/qdlblocks/models/born_gpt.py`)**:
   - **`unet_skip_mode: Literal["matrix", "scalar"] = "matrix"`**:
     - When `"scalar"`, omit `self.unet_gates` (saving $L/2$ dense $d \times d$ matrices) and instead allocate `self.unet_lambdas = nn.ParameterList([nn.Parameter(torch.full((embed_dim,), 0.10)) for _ in range(n_half)])`, blending encoder skip states via `x = x + self.unet_lambdas[enc_idx].to(x.dtype) * skip_x` (and followed by `l2_normalize` in `QuantumBornGPT`).
   - **`use_fused_qkv: bool = False`**:
     - When `True`, replace separate `q_proj, k_proj, v_proj` with a single `self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)` initialized as 3 stacked orthogonal `(embed_dim, embed_dim)` blocks and split via `q, k, v = self.qkv_proj(h).chunk(3, dim=-1)`.
   - **`num_value_embeddings: int = 0`**:
     - When `num_value_embeddings > 0` (e.g. `3`), allocate `self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(num_value_embeddings)])`.
     - Map layer index `l` to value embedding bank index: for $k = \min(\text{num\_value\_embeddings}, L // 2)$, if $l < k$, bank index is $l$; if $l \ge L - k$, bank index is $l - (L - k)$; otherwise no `ve` is applied at layer $l$.
     - Pass `ve_l = self.value_embeds[bank_idx](idx)` into block $l$, which mixes `ve_l` (reshaped to `(B, H, T, D_h)`, and L2-normalized per head in `QuantumBornGPT`) into $V^{(l)}$ via a learnable gate $\alpha_{\text{ve}}^{(l)} = \sigma(\text{ve\_mix\_logit}^{(l)})$:
       $$V^{(l)} \leftarrow (1 - \alpha_{\text{ve}}^{(l)}) V^{(l)} + \alpha_{\text{ve}}^{(l)} V_{\text{ve}}^{(l)}$$
   - **`skip_attn_layers: tuple[int, ...] = ()`**:
     - When `layer_idx in skip_attn_layers`, set `block.skip_attention = True` so `block.forward` skips the attention sublayer and runs only the MLP / two-beam interferometer sublayer (while Layer 0 still returns `v_0` if `return_v0=True`).
   - **`zero_half_rope: bool = False`**:
     - Passes `zero_half_freq=zero_half_rope` into `UnitaryRoPE`.

---

### 5.6 Component 5: JAX Mirror (`src/qdlblocks_jax/speedrun_primitives.py`) & CLI Benchmark Wiring (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)
1. **JAX Mirror (`src/qdlblocks_jax/speedrun_primitives.py`)**:
   - Implement `zeropower_via_newtonschulz5_batched_jax`, `half_truncated_rope_cos_sin_jax`, and `compute_document_ids_jax`, and export in `src/qdlblocks_jax/__init__.py`.
2. **CLI Benchmark (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)**:
   - Preserve fixed validation at `block_size = min(512, T_ep)` (`512`).
   - Add `--stage-shard-on-gpu` (default `True` when CUDA device total memory $\ge 20\text{ GB}$, staging the `200 MB` `uint16` shard on GPU once per shard and slicing `.to(torch.int64)` on-GPU).
   - Add `--direct-chunked-ce` (default `True`, routing `compute_step_loss` through `direct_chunked_softcap_linear_ce` and `direct_chunked_povm_ce`).
   - Add `--unet-skip-mode {matrix,scalar}`, `--fused-qkv`, `--num-value-embeddings` (int, default `0`), `--skip-attn-layers` (str, default `""`, e.g. `"6"`), `--zero-half-rope`, `--doc-mask-attn`, `--sliding-window` (int, default `0`), `--muon-momentum-warmup` (int, default `0`), and `--optimizer-mode {muon_adamw,pure_adamw,sgd_nesterov}`.
   - Ensure `value_embeds` parameters are routed to the `wte` parameter group in `AdamW` (`lr = 0.20` when `zero_init_classical` is active) and excluded from `StiefelMuon`.

---

## 6. Verification Plan, Wave-9 Fleet Launch, & Future Steps

### 6.1 Unit Test Suite (`tests/test_hypothesis_016_speedrun_systems_and_arch.py`)
1. `test_direct_chunked_ce_exact_gradients_and_single_gemm`: Verifies loss and gradient parity (`< 1e-5` loss diff, `> 0.99999` grad cosine) of `direct_chunked_softcap_linear_ce` and `direct_chunked_povm_ce` (both scalar and dynamic per-token $\beta_{b,t}$) against standard PyTorch autograd and verifies that the vocabulary projection GEMM is called only once per forward+backward pass.
2. `test_half_truncated_rope_norm_preservation_and_position_invariant_subspace`: Verifies unit norm preservation $\|\text{RoPE}(\psi)\|_2 = 1$, zero position variance on the upper half of head channels, and $\ge 1.50\times$ higher positive-cone Born overlap survival at long distances.
3. `test_batched_3d_newton_schulz_and_muon_momentum_warmup`: Verifies 3D `(K, M, N)` `zeropower_via_newtonschulz5_batched` parity (`< 1e-5`) against sequential 2D calls and verifies `StiefelMuon` momentum warmup schedule `0.85 -> 0.95`.
4. `test_value_embeddings_scalar_unet_fused_qkv_and_skip_attn_layer`: Verifies forward/backward execution and parameter shapes for both `ClassicalModdedGPT` and `QuantumBornGPT` with `num_value_embeddings=3`, `unet_skip_mode="scalar"`, `use_fused_qkv=True`, `skip_attn_layers=(2,)`, and `zero_half_rope=True`.
5. `test_triton_flash_born_sliding_window_and_doc_mask_parity`: Verifies CUDA forward/backward parity of `flash_born_attention` and `flash_purified_born_attention` with `window_size=32` and `doc_ids` against reference masked attention.
6. `test_on_gpu_uint16_shard_staging_and_jax_parity_016`: Verifies bit-exact `uint16 -> int64` on-GPU slice casting across `[0, 50304)` and verifies PyTorch-JAX parity for `src/qdlblocks_jax/speedrun_primitives.py`.

### 6.2 Controlled Hypothesis & Optimizer Ablation Script (`examples/26_hypothesis_speedrun_systems_arch_and_optimizers.py`)
- Executes controlled experiments `H1` (Systems Roofline Speedup), `H2` (Architectural Sample Efficiency for Classical & Quantum), and `H3` (Three-Optimizer Ablation: `muon_adamw` vs. `pure_adamw` vs. `sgd_nesterov`), asserts all quantitative pass criteria, and saves structured telemetry to `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_016.json`.

### 6.3 Wave-9 900M-Token Fleet Plan & Future Directions We Could Take Next
1. **Immediate Wave-9 Fleet Launch (Starting on Freed GPUs `gpu_g4_w6_2` & `gpu_a100_w6_2`)**:
   - **`w9_900m_classical_speedrun_v2_pr360`** (`L=12, d=768, H=12, d_ff=3072, T=1024, grad_accum=4, T_val=512`): Full PR #360-era single-GPU classical speedrun combining `zero_init_classical`, `true_layer0_v_residual`, `num_value_embeddings=3`, `unet_skip_mode="scalar"`, `use_fused_qkv=True`, `skip_attn_layers=(6,)`, `zero_half_rope=True`, `doc_mask_attn=True`, `sliding_window=384`, `muon_momentum_warmup=300`, `direct_chunked_ce=True`, `stage_shard_on_gpu=True`, and `wsd_min_lr_ratio=0.0` — targeting **`<= 3.28` Val CE**.
   - **`w9_900m_qborn_speedrun_v2_asym1024_L8`** & **`w9_900m_qborn_speedrun_v2_frame4d_L6`**: Full Speedrun-V2 `QuantumBornGPT` uniting the validated 4-part Early-Epoch Fix (`-0.20 nats`) with fused Triton `flash_purified_born_attention` (`6.41x` faster), `true_layer0_v_residual`, `num_value_embeddings=3`, `unet_skip_mode="scalar"`, `zero_half_rope=True`, `doc_mask_attn=True`, `muon_momentum_warmup=300`, `direct_chunked_ce=True`, and `wsd_min_lr_ratio=0.0`.
   - **`w9_900m_classical_opt_ablation_adamw`** & **`w9_900m_classical_opt_ablation_sgd`**: Full 900M-token optimizer ablations (`pure_adamw` and `sgd_nesterov`) on the exact same Speedrun-V2 architecture as additional GPU slots free up from Wave-7.
2. **Subsequent Research Steps Beyond Wave-9**:
   - **Progressive Window-Length & Sequence-Length Curriculum (`T = 256 -> 512 -> 1024`)**: Dynamically ramping training sequence length and sliding-window radius over the 900M-token run so the first 400M tokens execute at $2\times$ higher step frequency while the final 500M tokens sharpen long-range `T = 512` validation context.
   - **Compact Seed-Space Bigram Hash Superposition**: Injecting a lightweight `< 100 MB` local bigram `[t-1, t]` phase shift directly into $|\psi_0\rangle$ before `W_lift`, capturing the sample-efficiency benefit of `modded-nanogpt`'s n-gram lookup table without its `130 GB` 8-GPU memory footprint.
   - **FP8 `torch._scaled_mm` Readout on Hopper/Blackwell (`RTX PRO 6000` / `H100`)**: Quantizing the single-pass chunked `lm_head` / POVM forward and backward GEMMs to `float8_e4m3fn` on compute capability $\ge 8.9$ GPUs to halve readout memory bandwidth.
