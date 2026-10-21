# Experiment Plan 018: Comprehensive Project Synthesis (Waves 1–9), Unpinned Leaderboard Sorting, Persistent "Mark Obsolete" Lifecycle, & Wave-10 900M-Token Speedrun V3

## 1. Executive Summary & Standalone Project Overview

### 1.1 What Is `QuantumDLBlocks` / `QuantumBornGPT`?
`QuantumDLBlocks` is a dual-backend (`PyTorch + Triton` on CUDA GPUs and `JAX + Equinox + Haliax + Pallas` on Google Cloud TPUs) deep learning research framework investigating **Quantum-Geometric Language Modeling (`QuantumBornGPT`)** and benchmarking it head-to-head against modernized **Classical `Modded-NanoGPT` (`ClassicalModdedGPT`)** architectures on the standard `FineWeb10B` (`FineWeb-Edu`, GPT-2 tokenizer, vocabulary size $V = 50,304$) benchmark.

In a standard Euclidean Transformer (`GPT-2` or `ClassicalModdedGPT`), hidden states $x_{b,t}^{(l)} \in \mathbb{R}^d$ evolve in unconstrained Euclidean space, causal self-attention weights are computed via exponentiated dot products:
$$A_{i,j}^{\text{classical}} = \frac{\exp(q_i^\top k_j / \sqrt{d_h}) \cdot \mathbf{1}[j \le i]}{\sum_{t \le i} \exp(q_i^\top k_t / \sqrt{d_h})}$$
feed-forward blocks apply an elementwise nonlinearity such as $\text{ReLU}^2(x) = (\max(0, x))^2$, and vocabulary logits are produced by an unconstrained linear projection $\ell_{b,t} = W_{\text{lm}} x_{b,t}^{(L)} \in \mathbb{R}^V$ (optionally softcapped via $30 \tanh(\ell / 30)$).

By contrast, **`QuantumBornGPT`** constrains representations and operators strictly to **real quantum state manifolds (unit spheres $\mathbb{S}^{d-1} = \{\psi \in \mathbb{R}^d : \|\psi\|_2 = 1\}$) and Stiefel / tight-frame manifolds ($\text{St}(d, k) = \{W \in \mathbb{R}^{d \times k} : W^\top W = I_k\}$)**:

1. **Pure Quantum State Ket Backbone**: Every token $t \in \{0, \dots, T-1\}$ in batch item $b \in \{0, \dots, B-1\}$ at layer $l \in \{0, \dots, L\}$ is a unit-norm pure state ket $|\psi_{b,t}^{(l)}\rangle \in \mathbb{S}^{d-1}$, maintained via spherical normalization $\text{Normalize}(x) = x / \sqrt{\|x\|_2^2 + \epsilon}$ ($\epsilon = 10^{-6}$). At our default Hilbert dimension $d = 1024 = 2^{10}$, each token ket represents a **10-qubit real pure quantum state**.
2. **Isometric Hilbert Lift from a Seed Subspace**: Rather than requiring a full $V \times d$ embedding table, each vocabulary token $v \in \{0, \dots, V-1\}$ is prepared as a unit seed vector $|s_v\rangle \in \mathbb{S}^{D_{\text{seed}}-1}$ in a lower-dimensional **seed Hilbert space** ($D_{\text{seed}} \in \{128, 256, 384, 512, 1024\}$) and lifted into $\mathbb{S}^{d-1}$ via a semi-orthogonal Stiefel lift matrix $W_{\text{lift}} \in \mathbb{R}^{d \times D_{\text{seed}}}$ ($W_{\text{lift}}^\top W_{\text{lift}} = I_{D_{\text{seed}}}$):
   $$|\psi_0(v)\rangle = \text{Normalize}\!\big(W_{\text{lift}} |s_v\rangle\big) \in \mathbb{S}^{d-1}$$
3. **Born-Rule Quantum Interference Attention (`FlashBorn` & `FlashPurifiedBorn`)**: In each attention head ($H$ heads, per-head dimension $d_h = d / H$), queries $|q_i\rangle \in \mathbb{S}^{d_h-1}$, keys $|k_j\rangle \in \mathbb{S}^{d_h-1}$, and values $|v_j\rangle \in \mathbb{S}^{d_h-1}$ are unit-normalized kets rotated by `UnitaryRoPE`. Instead of exponential Softmax, causal transition probabilities follow the **quantum Born rule** (squared inner-product overlap $|\langle q_i | k_j \rangle|^2$), or its trace-purified generalization (`FlashPurifiedBorn`, introduced in Plans 014–015):
   $$w_{i,j}^{\text{Born}} = \gamma_h |\langle q_i | k_j \rangle|^2 \cdot \mathbf{1}[j \le i], \qquad A_{i,j} = \frac{w_{i,j}}{\sum_{t \le i} w_{i,t} + \epsilon}, \qquad |o_i\rangle = \text{Normalize}\!\left(\sum_{j \le i} A_{i,j} |v_j\rangle\right) \in \mathbb{S}^{d_h-1}$$
   where $\gamma_h > 0$ is a learnable per-head sharpness parameter.
4. **Two-Beam Overcomplete Tight-Frame Wave Interference MLP (`M×2`)**: Instead of a standard Euclidean MLP, each layer applies a two-beam interference block using a pair of overcomplete frame matrices $U_1, U_2 \in \mathbb{R}^{M \times d}$ where $M = \rho \cdot d$ is the intermediate frame dimension and $\rho \in \{1.5, 2.0, 3.0, 4.0\}$ is the **frame expansion factor** (denoted `1536×2`, `2048×2`, `3072×2`, or `4096×2` for $d=1024$):
   $$m_{\text{2beam}}(\psi) = W_{\text{down}}\!\big((U_1 \psi) \odot \text{SiLU}(U_2 \psi)\big), \qquad |\psi^{(l)}\rangle = \text{Normalize}\!\big(|\psi^{(l-1)}\rangle + \alpha_{\text{res}} \, \text{Normalize}(m_{\text{2beam}}(\psi^{(l-1)}))\big)$$
5. **POVM (Positive Operator-Valued Measure) Readout Collapse**: At the final layer $L$, the contextual Hilbert ket $|\psi_{b,t}^{(L)}\rangle \in \mathbb{S}^{d-1}$ is mapped to a unit readout ket $|\phi_{b,t}\rangle \in \mathbb{S}^{D_{\text{out}}-1}$ and measured against unit-norm vocabulary measurement vectors $|m_v\rangle \in \mathbb{S}^{D_{\text{out}}-1}$ ($v \in \{0, \dots, V-1\}$) scaled by an inverse measurement temperature $\beta_{b,t} > 0$:
   $$\ell_{b,t,v} = \beta_{b,t} \langle \phi_{b,t} | m_v \rangle + b_{\text{prior}, v}, \qquad P(v \mid \phi_{b,t}) = \frac{\exp(\ell_{b,t,v})}{\sum_{v'=0}^{V-1} \exp(\ell_{b,t,v'})}$$
   We support two primary high-performing readout modes:
   - **`tied_stiefel` + `TwoBeamSeedReadoutAdapter` (`D_out = D_seed`, e.g. `384` or `512`)**: Projects $|\psi_{\text{final}}\rangle \in \mathbb{S}^{d-1}$ to $\mathbb{S}^{D_{\text{seed}}-1}$ via $W_{\text{lift}}^\top$ plus a nonlinear two-beam phase-sensitive adapter that recovers orthogonal nullspace features from $\ker(W_{\text{lift}}^\top)$, and measures against $|m_v\rangle = \text{Normalize}(|s_v\rangle + \Delta_v) \in \mathbb{S}^{D_{\text{seed}}-1}$.
   - **`asymmetric` Full-Rank Hilbert Readout (`D_out = d = 1024`)**: Bypasses seed-space collapse at readout altogether ($|\phi_{b,t}\rangle = \text{Normalize}(W_{\text{rot}} |\psi_{b,t}^{(L)}\rangle) \in \mathbb{S}^{d-1}$) and measures directly against an un-bottlenecked full-rank POVM measurement frame $M_{\text{full}} \in \mathbb{R}^{V \times d}$ ($|m_v\rangle \in \mathbb{S}^{d-1}$).
6. **Riemannian Stiefel Muon Optimization (`StiefelMuon`)**: All 2D internal projection matrices are optimized using 5-step Newton-Schulz orthogonalization (`zeropower_via_newtonschulz5_batched`) with Nesterov momentum warmup, cautious weight decay, and optional continuous Stiefel retraction penalty $\frac{1}{4}\|W^\top W - I\|_F^2$, while token embeddings, POVM heads, and scalar gates are optimized with fused `AdamW`.

---

### 1.2 Core Quantum-Geometric & Systems Telemetry Definitions
Every benchmark run logs both standard language modeling metrics (evaluated strictly at the NanoGPT speedrun validation context length **$T_{\text{val}} = \text{block\_size} = 512$** across the 5,000,000-token `fineweb_val_000000.bin` split) and four quantum-geometric observables:
1. **Validation Cross-Entropy (`Val CE`, in nats) & Perplexity (`Val PPL = exp(Val CE)`)**: Token-level negative log-likelihood on `fineweb_val_000000.bin` at $T_{\text{val}} = 512$, accompanied by `Top-1 %`, `Top-5 %`, and median target token rank (`MedRank`).
2. **Readout Effective Quantum Rank ($R_{\text{eff}}(\rho_{\text{seed}})$)**: Given $N$ unit-norm readout kets $\{|\phi_i\rangle\}_{i=1}^N \subset \mathbb{S}^{D_{\text{out}}-1}$, we form their empirical density matrix $\rho = \frac{1}{N}\sum_{i=1}^N |\phi_i\rangle\langle\phi_i| \in \mathbb{R}^{D_{\text{out}} \times D_{\text{out}}}$, compute its von Neumann entropy $S_{\text{vN}}(\rho) = -\sum_{k=1}^{D_{\text{out}}} \lambda_k \log \lambda_k$ from the eigenvalues $\{\lambda_k\}$ of $\rho$, and define:
   $$R_{\text{eff}}(\rho_{\text{seed}}) = \exp\!\big(S_{\text{vN}}(\rho)\big) \in [1, D_{\text{out}}]$$
   which measures the effective number of orthogonal quantum state dimensions actively populated at readout.
3. **10-Qubit Bipartite Entanglement Entropy ($S(\rho_A)$, in nats)**: Reshaping each 1024D ket $|\psi\rangle \in \mathbb{S}^{1023}$ into a $2^5 \times 2^5 = 32 \times 32$ bipartite amplitude matrix $\Psi_{A,B}$ across two 5-qubit subsystems $A$ and $B$, the reduced density matrix is $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|) = \Psi_{A,B}\Psi_{A,B}^\top \in \mathbb{R}^{32 \times 32}$, with von Neumann entanglement entropy $S(\rho_A) = -\text{Tr}(\rho_A \log \rho_A) \in [0, \ln 32 \approx 3.466\text{ nats}]$.
4. **Stiefel / Frame Isometry Deviation (`IsoErr`) & POVM Sharpness ($\beta$)**:
   $$\text{IsoErr}(W) = \frac{\|W^\top W - I_c\|_F}{\sqrt{c}} \;\; (r \ge c) \quad \text{or} \quad \frac{\|W W^\top - I_r\|_F}{\sqrt{r}} \;\; (r < c)$$
   averaged across internal 2D backbone matrices, alongside the mean POVM inverse temperature $\beta = \mathbb{E}[\beta_{b,t}]$.

---

## 2. Problems Solved & Discussed Across Waves 1–9 (`Plans 001–017`)

### 2.1 Chronological Synthesis of Problems, Solutions, and Breakthroughs

| Wave & Plans | Core Problem Investigated | Architectural / Systems Solution Built | Quantitative Outcome (`T_val = 512`) |
| :---: | :--- | :--- | :--- |
| **Baseline & Wave 1** (`Plans 001–005`, 300M tok = 3 ep × 100M shard) | Can a pure unit-sphere / Stiefel network train stably at 100M+ token scale with fast POVM readout? | Built `QuantumBornGPT` (`L=6, d=1024, D_seed=128`), `FlashBorn` Triton kernel, `TangentStiefelMuon`, Strided Decoupled Multi-Token Prediction (`MTP`), `MultiBeamKrausReadout`, and `fused_tiled_povm_cross_entropy`. | Reduced 300M-token Val CE from **`4.3488` (`77.4` PPL)** (`gpu_3ep` baseline) to **`4.2181` (`67.9` PPL)** (`hyp_gpu_exp03`, `D_seed=256`). |
| **Wave 2** (`Plans 006–007`, 300M tok) | Strict $W^\top W = I$ orthogonality limits MLP feature capacity, and linear collapse $W_{\text{lift}}^\top : \mathbb{R}^{1024} \to \mathbb{R}^{D_{\text{seed}}}$ discards $(1024 - D_{\text{seed}})$ orthogonal dimensions. | Relaxed Stiefel orthogonality via **Overcomplete Tight Frames** (`OvercompleteFrameInterference`, governed by the Welch bound) and built `OvercompleteSeedReadoutAdapter` to recover nullspace energy from $\ker(W_{\text{lift}}^\top)$. | Improved 300M-token Val CE to **`4.2367`** and revealed the **Quantum State Capacity Law**: $\text{Val CE}$ scales linearly with $-\log R_{\text{eff}}(\rho_{\text{seed}})$. |
| **Wave 3** (`Plans 008–010`, 300M tok) | Why did expanding $D_{\text{seed}} \to 512\text{–}1024$ without a phase-sensitive adapter cause subspace starvation, and why was the 1-beam quadratic adapter blind to antipodal sign flips $\pm \psi_\perp \in \ker(W_{\text{lift}}^\top)$? | Built `OvercompleteTwoBeamInterference` (`2048×2`) and `TwoBeamSeedReadoutAdapter` (combining even quadratic $(U_1\psi)^{\odot 2}$ and odd cubic $(U_1\psi)^{\odot 2} \odot \text{SiLU}(U_2\psi)$ terms to break antipodal sign degeneracy). | Reached **`4.1846` Val CE (`65.7` PPL)** on 300M tokens (`w3_gpu_adapter384_2beam`) and boosted $R_{\text{eff}}(\rho_{\text{seed}})$ from `27.4` to **`48.2`**. |
| **Wave 4** (`Plan 011`, 900M unique tok = 1 ep × 9 shards) | Repeating a single 100M-token shard for 9 epochs caused a `0.38+` nat memorization gap that ruined late-stage WSD cooldown. | Built `FineWebMultiShardStream` with asynchronous background shard prefetching (`0.0 ms` I/O stall across `fineweb_train_000001..000009.bin`) and validated Warmup-Stable-Decay (`WSD`) cooldown on non-repeating data. | Eliminated memorization (`Train - Val CE ≈ 0.02` nats) and broke the **`4.00` Val CE** barrier: **`3.9223` Val CE (`50.5` PPL)** (`w4_900m_full_synergy512_cosine`). |
| **Wave 5** (`Plan 012`, 900M unique tok, `120M–211M` params) | How should we scale `QuantumBornGPT` parameters from `66M` to `120M–211M` across Frame Expansion (`3d/4d`), Full-Rank Asymmetric Readout (`D_out=1024`), Width (`d=1536`), Depth (`L=12`), and Tied Vocab? | Built multi-axis scaling (`parameter_scaling.py`) with Welch-bound tight-frame targets and geodesic residual scaling $\alpha_{\text{res}}(L) = \sqrt{6/L}$. | `4d` Frame (`4096×2`) + `D_seed=512` hit **`3.8765` Val CE (`48.3` PPL)** (`w5_900m_frame4d_synergy512`), and Full-Rank `Asym-1024` POVM + `L=9` hit **`3.8825` Val CE** ($R_{\text{eff}} = 72.62$). |
| **Wave 6** (`Plan 013`, 900M unique tok) | **Constant-Logit-Norm Theorem**: Unit-norm kets $|\phi_{b,t}\rangle \in \mathbb{S}^{D_{\text{out}}-1}$ with a single scalar $\beta$ lock $\|\ell_{b,t}\|_2 \approx \beta\sqrt{V/D_{\text{out}}} = \text{const}$, preventing context-dependent sharpness on heteroscedastic language contexts. Also needed an apples-to-apples **Classical `Modded-NanoGPT`** baseline. | Built `ContextualBlochPurityGate` (`--dynamic-povm-beta`, modulating per-token Bloch purity radius $r(\psi_{b,t}) \in [0.25, 4.0]$) and `ClassicalModdedGPT` (`--architecture classical_modded`). | `w6_900m_asym1024_frame3d_dynbeta` set a new quantum record of **`3.8514` Val CE (`47.1` PPL)**, while `ClassicalModdedGPT` (`6L, d=1024`) reached **`3.5444` Val CE (`34.6` PPL)** (`162M`) and **`3.6003` Val CE** (`124M`). |
| **Wave 7 & Wave 8** (`Plans 014–015`, 900M unique tok) | Why did `QuantumBornGPT` exhibit a slower early-epoch (`0–100M` tok) validation drop than `ClassicalModdedGPT`? Why did a naive Python-loop `PurifiedBorn` cut GPU throughput from `181k` to `62k tok/s`? And why did Classical `L=12` stall at `4.2312` when `StiefelMuon` applied a `0.005` Stiefel penalty to Euclidean MLPs? | Diagnosed 4 early-epoch quantum bottlenecks (Quadratic Born SNR attenuation, input-highway token echo, unigram prior cold-start, and uniform spherical mixing); built fused Triton `flash_purified_born_attention` (`2.79×` kernel speedup), true cross-layer Layer-0 $V^{(0)}$ residual sharing (`--true-layer0-v-residual`), decoupled `--classical-stiefel-penalty 0.0`, and `WSD-0` (`min_lr_ratio=0.0`). | Improved `50M`-token Quantum Val CE by **`-0.1480` nats** (`4.7461 -> 4.5981`), while `w7_900m_classical_speedrun_12L_t1024` (`L=12, d=768, ZeroInit + WSD`) reached **`3.4604` Val CE (`31.8` PPL, `221,490 tok/s`)**. |
| **Wave 9 & Permanent Server** (`Plans 016–017`, 900M unique tok — **Just Completed!**) | How do we close the remaining sample-efficiency and systems gap to the `KellerJordan/modded-nanogpt` PR #360 speedrun (`3.28` Val CE), apply those primitives to `QuantumBornGPT`, run an apples-to-apples Optimizer Ablation (`Muon` vs `Pure AdamW`), and run the dashboard as a **zero-downtime permanent server** with per-run **Model Cards**? | Implemented `direct_chunked_softcap_linear_ce` & `direct_chunked_povm_ce` (`0` backward GEMMs), 3D batched Newton-Schulz (`7.91×` faster), `num_value_embeddings=3` (`ve` in layers `0,1,2` & `L-3,L-2,L-1`), `unet_skip_mode="scalar"`, `use_fused_qkv=True`, `skip_attn_layers=(6,)`, `zero_half_rope=True`, sliding-window + `<\|endoftext\|>` masking, and `DynamicWaveRegistry` + `RunModelCard` hot-reload server (`0` restarts). | **Massive Double Breakthrough**:<br>• **Quantum #1 All-Time Record**: `w9_900m_qborn_speedrun_v2_asym1024_L8` hit **`3.7203` Val CE (`41.3` PPL, `34.93%` Top-1)** — beating Wave-6 (`3.8514`) by **`-0.1311` nats (`-5.8` PPL)**!<br>• **Quantum #2 All-Time Record**: `w9_900m_qborn_speedrun_v2_frame4d_synergy512` hit **`3.7495` Val CE (`42.5` PPL, `126,338 tok/s`)** — beating Wave-5 (`3.8765`) by **`-0.1270` nats**!<br>• **Classical #1 All-Time Record**: `w9_900m_classical_speedrun_v2_pr360` hit **`3.4280` Val CE (`30.8` PPL, `37.96%` Top-1, `Train CE = 3.4045`)**!<br>• **Optimizer Ablation**: `StiefelMuon` (`3.4280`) beat `Pure AdamW` (`3.5025`) by **`-0.0745` nats (`-2.4` PPL)** on the exact same architecture! |

---

### 2.2 Deep-Dive Analysis of the Completed Wave-9 Results & What They Reveal for Wave-10

All five Wave-9 `900M`-token FineWeb-Edu runs completed 100% on our 5-GPU Colab fleet (`3× G4 RTX PRO 6000 Blackwell` + `2× A100-SXM4-40GB`), evaluated every `50M` tokens at fixed validation block size $T_{\text{val}} = 512$:

| Wave-9 Run Key | Architecture & Key Primitives | Global Batch & Steps | `50M` CE | `100M` CE | `200M` CE | `450M` CE | `650M` CE (Pre-Decay) | **`900M` Final Val CE (PPL)** | Top-1 / Top-5 | Throughput (`tok/s`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`w9_900m_classical_speedrun_v2_pr360`** | `ClassicalModdedGPT` (`L=12, d=768, H=12, d_ff=3072`), `ve=3`, `scalar` U-Net, `fused_qkv`, `skip_attn=6`, `zero_half_rope`, `win=384+doc_mask`, `WSD-0` (`Muon+AdamW`) | `262,144` tok/step (`3,429` steps) | `4.8834` | `4.2521` | `3.9375` | `3.7068` | `3.6114` | **`3.4280` (`30.8` PPL)** 🥇 | **`37.96%` / `59.77%`** | `136,754` |
| **`w9_900m_classical_speedrun_v2_t512_b524k`** | Same PR #360 V2 architecture at $2\times$ larger batch (`B=256, G_acc=4`) | `524,288` tok/step (`1,710` steps) | `5.4751` | `4.5934` | `4.0505` | `3.7323` | `3.6352` | **`3.4500` (`31.5` PPL)** 🥈 | `37.65%` / `59.43%` | `148,884` |
| **`w9_900m_classical_opt_ablation_pure_adamw`** | Exact same PR #360 V2 architecture with **`--optimizer-mode pure_adamw`** (`lr=0.003` backbone) | `262,144` tok/step (`3,429` steps) | `5.2333` | `4.4529` | `4.0570` | `3.7854` | `3.6815` | **`3.5025` (`33.2` PPL)** | `37.07%` / `58.85%` | `110,906` |
| **`w9_900m_qborn_speedrun_v2_asym1024_L8`** | `QuantumBornGPT` (`L=8, d=1024, 3d-Frame 3072×2, Asym1024`), `ve=3`, `scalar` U-Net, `fused_qkv`, `zero_half_rope`, `FlashPurifiedBorn win=384+doc_mask`, `WSD-0` | `65,536` tok/step (`13,725` steps) | **`4.5753`** | **`4.3590`** | **`4.1800`** | **`4.0336`** | **`3.9905`** | **`3.7203` (`41.3` PPL)** 🏆 | **`34.93%` / `56.30%`** | `95,615` |
| **`w9_900m_qborn_speedrun_v2_frame4d_synergy512`** | `QuantumBornGPT` (`L=6, d=1024, 4d-Frame 4096×2, D_seed=512, 2BeamAdapter`), `ve=3`, `scalar` U-Net, `fused_qkv`, `zero_half_rope`, `FlashPurifiedBorn win=384+doc_mask`, `WSD-0` | `65,536` tok/step (`13,725` steps) | `4.5997` | `4.4123` | `4.2594` | `4.1314` | `4.0699` | **`3.7495` (`42.5` PPL)** 🥈 | `34.81%` / `55.99%` | `126,338` |

Three critical scientific insights emerge from this table:

1. **Why `QuantumBornGPT` Jumped by `-0.1311` Nats (`3.8514 -> 3.7203` Val CE, `-5.8` Perplexity Points)**:
   - Both Wave-9 quantum models kept the fine-grained `65,536` tok/step batch size (`13,725` StiefelMuon steps over `900M` tokens) while integrating Plan 016's PR #360 primitives (`num_value_embeddings=3`, `unet_skip_mode="scalar"`, `use_fused_qkv=True`, `zero_half_rope=True`, `true_layer0_v_residual=True`, `flash_purified_born_attention` with `window_size=384` + `doc_ids`, and `WSD-0` linear cooldown to zero).
   - At `50M` tokens, `w9_900m_qborn_speedrun_v2_asym1024_L8` reached `4.5753` Val CE (beating Wave-6's `4.7461` by **`-0.1708` nats**), and during the final 25% `WSD-0` cooldown (`675M -> 900M` tokens), its validation loss plunged by **`-0.2702` nats** (`3.9905 -> 3.7203`), while `w9_900m_qborn_speedrun_v2_frame4d_synergy512` plunged by **`-0.3204` nats** (`4.0699 -> 3.7495`)!
   - Furthermore, Wave-9 tested `L=8 + 3d-Frame + Asym1024` (`3.7203`) and `L=6 + 4d-Frame + Synergy512` (`3.7495`) *separately*. Unifying **`L=8` + `4d-Frame (4096×2)` + `Asym-1024`** (`w10_900m_qborn_speedrun_v3_asym1024_frame4d_L8`) and scaling **`Asym-1024` to `L=12` layers with `skip_attn_layers=(6,)`** (`w10_900m_qborn_speedrun_v3_asym1024_L12_deep`) are our next logical steps to break **`3.65` Val CE (`< 38.5` PPL)**.

2. **Why `w9_900m_classical_speedrun_v2_pr360` Reached `3.4280` (`Train CE = 3.4045`) and What Held It Back From `<= 3.32`**:
   - **Finding A — Step-Count Starvation at Large Global Batch (`3,429` Steps vs. `6,858` or `13,725` Steps)**:
     Compare `w9_900m_classical_speedrun_v2_pr360` (`B_global = 262,144` tok/step = `3,429` steps) against Wave-7's `w7_900m_classical_speedrun_12L_t1024` (`B_global = 131,072` tok/step = `6,858` steps) at **matched optimizer step counts**:
     - At `~763` optimizer steps (`200M` tok for `w9` vs. `100M` tok for `w7`), `w9` achieved **`3.9375` Val CE** vs. `w7`'s `4.1927` (**`-0.2552` nats better per step**).
     - At `~3,429` optimizer steps (`900M` tok for `w9` vs. `450M` tok for `w7`), `w9` achieved **`3.6114` pre-cooldown / `3.4280` post-cooldown Val CE** vs. `w7`'s `3.7571` (**`-0.1457` nats better pre-cooldown at the same step count**).
     - However, because `w9` used $B_{\text{global}} = 262,144$ tokens/step (and `w9_b524k` used $524,288$ tokens/step = only `1,710` steps!), `w9` executed only `3,429` gradient steps across the entire `900M`-token stream and used a conservative Muon learning rate of `lr = 0.025` (whereas `modded-nanogpt` PR #360 uses `lr = 0.04–0.05` with aspect scaling). Running the PR #360 architecture at **`131,072` tok/step (`6,858` steps, `2.0×` more updates)** with `--muon-lr 0.04` and at **`65,536` tok/step (`13,725` steps, `4.0×` more updates)** with `--muon-lr 0.035` directly eliminates step-count starvation!
   - **Finding B — 4D Boolean `attn_mask` Fallback in PyTorch SDPA (`136.8k tok/s` vs. `221.5k tok/s`) & Train/Val Mask Mismatch**:
     In `ClassicalModdedBlock.forward` (`src/qdlblocks/models/classical_modded_gpt.py`), when `window_size > 0` or `doc_ids is not None`, PyTorch constructs a dense 4D boolean tensor `mask_4d` of shape `(B, 1, T, T)` and calls `F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d, is_causal=False)`. On PyTorch CUDA, passing an explicit 4D `attn_mask` bypasses the fused FlashAttention-2 `is_causal=True` kernel and falls back to the slower memory-efficient/math SDPA kernel (reducing G4 Blackwell throughput from `221,490 tok/s` in Wave-7 to `136,754 tok/s` in Wave-9). Worse, `evaluate_validation` evaluates at $T_{\text{val}} = 512$ with `window_size=0` (full 512-token causal attention), creating a train/validation receptive-field mismatch whenever training uses `window_size=384` at $T_{\text{train}} = 512$. Running `ClassicalModdedGPT` at $T_{\text{train}} = T_{\text{val}} = 512$ with native FlashAttention-2 (`is_causal=True`, `window_size=0, doc_mask_attn=False`) restores **`> 220,000 tok/s`** throughput and guarantees zero train/validation context mismatch!

3. **What the Optimizer Ablation (`Muon` vs. `Pure AdamW` vs. `SGD+Nesterov`) Proved**:
   - Holding the Wave-9 PR #360 architecture (`228.23M` params, `L=12, d=768, ve=3, scalar U-Net, fused_qkv, skip_attn=6, zero_half_rope, WSD-0`) and batch schedule (`3,429` steps) completely identical:
     - **`StiefelMuon + AdamW`** (`w9_900m_classical_speedrun_v2_pr360`) achieved **`3.4280` Val CE (`30.8` PPL, `37.96%` Top-1, `136,754 tok/s`)**.
     - **`Pure AdamW`** (`w9_900m_classical_opt_ablation_pure_adamw`) achieved **`3.5025` Val CE (`33.2` PPL, `37.07%` Top-1, `110,906 tok/s`)**.
   - This proves two things simultaneously: (a) Newton-Schulz orthogonalized `StiefelMuon` provides a decisive **`-0.0745` nat (`-2.4` PPL, `+0.89%` Top-1) sample-efficiency advantage** and **`+23.3%` throughput advantage** over `Pure AdamW` at 900M-token scale, and (b) our Wave-9 PR #360 architectural upgrades are so strong that even with `Pure AdamW` (`3.5025`), the Wave-9 model beats all Wave-6 Muon classical baselines (`3.5444` and `3.6003`)! In Wave-10, we complete this 3-way study by running **`--optimizer-mode sgd_nesterov`** (`w10_900m_classical_opt_ablation_sgd_nesterov`) on the exact same `900M`-token configuration.

---

### 2.3 New Dashboard UX & Lifecycle Problems to Solve Now (`Plan 018`)

With **61+ Model Cards** across Baseline and Waves 1–9 now served by our permanent zero-downtime benchmarking server (`scripts/monitor_colab_runs.py` on `0.0.0.0:8765`), the user identified two important usability issues on the live dashboard:

1. **Problem A — Baseline Models Appearing at the Top of the Sorted Leaderboard List**:
   Inspection of `scripts/monitor_colab_runs.py` and `.cache/live_logs/latest_snapshot.json` revealed three compounding causes:
   - **Cause 1 (Duplicate `Baseline` Entries & Duplicate React Keys in `collect_snapshot()`)**: In `collect_snapshot()`, the 5 single-shard multi-epoch control runs (`gpu_5ep`, `gpu_3ep`, `gpu_wsd_mtp`, `tpu_3ep`, `tpu_wsd_mtp` with `wave_id = "Baseline"`) were manually constructed and appended at indices `0..4` of `runs_catalog`, AND saved to `.cache/model_cards/<key>.json`. Then `DynamicWaveRegistry.discover_all_specs()` scanned `.cache/model_cards/*.json` and appended those exact same 5 keys *a second time* at indices `61..65`! Having two entries with identical `key` (`gpu_5ep`, `gpu_3ep`, etc.) in `runs_catalog` caused React 18's `<tr key={r.key}>` reconciliation to glitch when reordering rows on sort and left the duplicate `Baseline` rows at the top of the DOM table (as well as hardcoded at the top of `format_terminal_report`).
   - **Cause 2 (String-Based `wave_id` & `short_label` Sorting)**: When sorting by the `Wave` column (`wave_id`), `filteredAndSortedRuns` used JavaScript string `.localeCompare()`, which sorts `"Baseline"` (`"B"`) ahead of `"W1".."W9"` (`"W"`) and sorts `"W10"` between `"W1"` and `"W2"`. Similarly, when sorting by `Run / Session` (`short_label`), `"Baseline G4..."` lacked a `[session]` prefix whereas wave runs began with `"[gpu_...]"`, causing inconsistent ordering.
   - **Cause 3 (Classical Baseline Dominance in Default `Best Val CE` Sort)**: When sorting by `Best Val CE` ascending, the 7 Euclidean `ClassicalModdedGPT` baseline runs (`3.4280` to `3.6003` Val CE) occupy rows `1..7` ahead of all `QuantumBornGPT` runs (`3.7203+` Val CE), with no one-click way to filter the leaderboard by architecture family (`Quantum` vs. `Classical`) or mark superseded baseline runs obsolete.

2. **Problem B — Need for a Persistent "Mark Obsolete" / "Show Obsolete" Workflow**:
   As we progress into Wave 10 (`65+` total runs), early single-shard 100M-token baselines (`gpu_5ep`, `gpu_3ep`, `gpu_wsd_mtp`, `tpu_3ep`, `tpu_wsd_mtp`), crashed OOM experiments (`w5_900m_deep12_adapter384` OOM attempt), and superseded early ablations clutter the active Leaderboard Table. We need a first-class **"Mark Obsolete"** lifecycle feature that:
   - Hides marked-obsolete runs from the Leaderboard Table (and charts) by default,
   - Provides an explicit opt-in toggle (**`🗄️ Show Obsolete (N)`** and status pill **`🗄️ Obsolete Only (N)`**) to view and inspect marked-obsolete records at any time,
   - Lets the user mark or unmark (`♻️ Restore`) any individual run (via a button on each table row and inside the Model Card Modal) or bulk-mark filtered runs, and
   - Persists the `obsolete` state durably on the server across browser sessions and hot-reloads (in both `.cache/live_logs/ui_prefs.json` and `.cache/model_cards/<run_key>.json` via a zero-downtime `POST /api/obsolete` endpoint).

---

## 3. Outcomes We Are Looking to Achieve in Plan 018 & Wave-10

1. **Strict, Unpinned Leaderboard Sorting & Zero Duplicate Catalog Keys**:
   - Every `run_key` in `runs_catalog` is strictly unique (`0` duplicate keys).
   - `Baseline` runs (`wave_id = "Baseline"`) are never pinned to the top of any sorted list or terminal report; sorting by `Best Val CE`, `Val PPL`, `Top-1 %`, `Tok/s`, `Params`, `D_seed`, `Wave` (using numeric `wave_num`: `Baseline=0, W1=1, ..., W9=9, W10=10`), or `Run / Session` (using normalized label strings) orders all visible rows strictly by the selected column and sort direction.
2. **Durable "Mark Obsolete" / "Show Obsolete (`N`)" Filtering & Architecture Family Toggle**:
   - Every run and `RunModelCard` has a persistent `obsolete: bool` attribute (plus `obsolete_reason: str` and `obsolete_at: str`), controllable via `POST /api/obsolete` in `< 10 ms` with zero server downtime.
   - By default, the 5 legacy single-shard `Baseline` runs (`gpu_5ep`, `gpu_3ep`, `gpu_wsd_mtp`, `tpu_3ep`, `tpu_wsd_mtp`) and historical `OOM_CRASH` / `STOPPED` incomplete runs are pre-marked `obsolete: true` so the default Leaderboard Table cleanly displays active/completed modern multi-shard experiments, while clicking **`🗄️ Show Obsolete (N)`** immediately reveals all obsolete records with one-click **`♻️ Restore`** buttons.
   - An **Architecture Family Filter** (`All Families` | `⚛️ Quantum Born` | `🏛️ Classical Baseline`) in the table toolbar lets the user view `QuantumBornGPT` runs alone, `ClassicalModdedGPT` baselines alone, or both together in a single click.
   - The 4 Top KPI Summary Cards at the top of the dashboard dynamically compute the live #1 Classical record (`3.4280` CE), #1 Quantum record (`3.7203` CE), and #2 Quantum record (`3.7495` CE) directly from `runs_catalog`.
3. **Wave-10 `900M`-Token Speedrun V3 Breakthroughs on All 5 Colab GPUs**:
   - Push `ClassicalModdedGPT` past Wave-9's `3.4280` Val CE toward **`<= 3.35` Val CE** (and closer to the `3.280` NanoGPT speedrun target at $T_{\text{val}} = 512$) by restoring `6,858` (`B=131k`) and `13,725` (`B=65k`) StiefelMuon steps, `--muon-lr 0.04` / `0.035`, and fast native FlashAttention-2 causal SDPA (`> 220,000 tok/s`).
   - Complete the 3-way `900M`-token Optimizer Ablation (`StiefelMuon` `3.4280` vs. `Pure AdamW` `3.5025` vs. `SGD + Nesterov`).
   - Push `QuantumBornGPT` past Wave-9's `3.7203` Val CE (`41.3` PPL) toward **`< 3.66` Val CE (`< 38.8` PPL)** by unifying Wave-9's #1 (`Asym-1024 + L=8`) and #2 (`4d-Frame 4096×2`) quantum winners and scaling to `L=12` layers with `skip_attn_layers=(6,)`.

---

## 4. Testable Hypotheses & Quantitative Predictions

### Hypothesis H1 (Unpinned Comparator Sorting & Catalog Deduplication)
- **Claim**: Deduplicating `baseline_defs` against `DynamicWaveRegistry.discover_all_specs()` in `collect_snapshot()` eliminates all duplicate `run_key` entries in `runs_catalog`, and sorting `runs_catalog` (and React's `filteredAndSortedRuns`) using strict column comparators with numeric `wave_num` (`Baseline = 0`, `W1 = 1`, `W2 = 2`, `W2B = 2.5`, ..., `W9 = 9`, `W10 = 10`) ensures that `Baseline` models never appear at the top of `best_val_ce` ascending, `wave_id` descending, or `format_terminal_report(snap)`.
- **Quantitative Pass Criteria**:
  1. `len(runs_catalog) == len({r["key"] for r in runs_catalog})` (`0` duplicate keys across all `>= 61` runs).
  2. When sorted by `best_val_ce` ascending (including when `showObsolete=True`), the top 10 rows contain `0` `"Baseline"` wave runs (since `"Baseline"` runs have `best_val_ce` in `[4.2676, 4.5035]`, placing them at rank `>= 40`), and #1 is `w9_900m_classical_speedrun_v2_pr360` (`3.4280` CE) overall or `w9_900m_qborn_speedrun_v2_asym1024_L8` (`3.7203` CE) when `filterFamily == "quantum_born"`.
  3. When sorted by `wave_id` descending, `"W10"` ( `wave_num = 10`) sorts ahead of `"W9"` (`wave_num = 9`), which sorts ahead of `"W8" .. "W1"`, with `"Baseline"` (`wave_num = 0`) at the very bottom.

### Hypothesis H2 (Zero-Downtime Persistent "Mark Obsolete" / "Show Obsolete" Lifecycle)
- **Claim**: Adding `obsolete: bool`, `obsolete_reason: str`, and `obsolete_at: str` to `RunModelCard` and `POST /api/obsolete` to `create_benchmark_http_server` allows any single run or batch of runs to be marked or unmarked obsolete atomically in `< 15 ms` with `0` server restarts, hiding obsolete runs from default table/chart queries (`showObsolete = false`) while preserving 100% of their telemetry and Model Card data when `showObsolete = true`.
- **Quantitative Pass Criteria**:
  1. `POST /api/obsolete` with `{"run_keys": ["gpu_5ep", "gpu_3ep"], "obsolete": true, "reason": "Superseded baseline"}` completes in `< 15.0 ms` while `server_uptime_s` increases monotonically (`0` restarts).
  2. Both `.cache/model_cards/gpu_5ep.json` (`card["obsolete"] is True`) and `.cache/live_logs/ui_prefs.json` (`prefs["obsoleteMap"]["gpu_5ep"] is True`) are persisted atomically on disk and survive `POST /api/reload`.
  3. Toggling `"obsolete": false` via `POST /api/obsolete` restores the run to the active non-obsolete catalog view in `< 15.0 ms` with zero loss of `evals`, `steps`, or `model_card` fields.

### Hypothesis H3 (Fast Causal SDPA & Step-Budget Scaling for `ClassicalModdedGPT` V3)
- **Claim**: At fixed sequence length $T_{\text{train}} = T_{\text{val}} = 512$ on `ClassicalModdedGPT`, running native causal SDPA (`window_size=0, doc_mask_attn=False`, dispatching to fused FlashAttention-2 `is_causal=True`) increases step throughput by **`>= 1.25×`** over materialized 4D boolean mask SDPA (`window_size=384, doc_mask_attn=True`), and increasing the optimizer step budget by `2.0×` at a matched token budget with `--muon-lr 0.04` achieves strictly lower validation Cross-Entropy than `3,429`-step scaling (`lr = 0.025`).
- **Quantitative Pass Criteria**:
  1. In controlled GPU micro-benchmarks at $T = 512$, native causal SDPA (`is_causal=True`) achieves **`>= 1.25×`** higher step throughput (`tok/s`) than 4D boolean `attn_mask` SDPA and `0.0` train/eval attention mask discrepancy.
  2. Over a matched token budget on a controlled synthetic FineWeb shard, `ClassicalModdedGPT` V3 with `2×` step frequency (`B` vs `2B` tokens/step) and `muon_lr = 0.04` reduces validation Cross-Entropy by **`>= 0.05` nats** compared to the `1×` step frequency (`muon_lr = 0.025`) baseline.

---

## 5. Detailed Technical Specification of Code Changes

### 5.1 Operational Scripting Classification (`Build-Before-Launch` vs. `Nice-to-Have`)
Following our operational discipline:
- **Build-Before-Launch (Implemented in Plan 018)**:
  1. Catalog deduplication, numeric `wave_num` sorting, and dynamic Top KPI cards in `scripts/monitor_colab_runs.py` and `src/qdlblocks/benchmarks/model_card.py`.
  2. Persistent `obsolete` lifecycle (`RunModelCard` schema fields, `POST /api/obsolete` endpoint, React `"🗄️ Show Obsolete (N)"` toggle, per-row/modal `"🗄️ Obsolete"` / `"♻️ Restore"` buttons, and Architecture Family filter `All | Quantum | Classical`).
  3. `--muon-lr` CLI flag in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` and `src/qdlblocks/benchmarks/model_card.py`.
  4. Wave-10 declarative fleet manifest `configs/wave10_fleet.json`.
- **Nice-to-Have (Deferred to `scripts/BACKLOG.md`)**:
  - CLI subcommand `scripts/colab_fleet.py obsolete --waves w1,w2` for batch-marking entire waves obsolete from the terminal (logged to `scripts/BACKLOG.md`; the web UI and HTTP API `POST /api/obsolete` already support bulk marking).

---

### 5.2 Component 1 & 2: `src/qdlblocks/benchmarks/model_card.py` Updates
1. **Wave Registry Metadata for Wave 10 & Numeric `wave_num`**:
   - Add `"W10": "Wave 10 (Plan 018: Speedrun V3 Step-Budget Scaling, Fast SDPA, SGD Ablation & Deep L=12 Quantum)"` to `WAVE_DEFAULT_LABELS` and `"W10": "EXPERIMENT_PLAN_018"` to `WAVE_DEFAULT_PLANS`.
   - In `infer_wave_id_and_num(run_key: str, raw_wave: str = "") -> tuple[str, int]`, ensure `"Baseline"` returns `("Baseline", 0)`, `"W2B"` returns `("W2B", 2)`, and any `"W<N>"` returns `(f"W{N}", N)`.
2. **`RunModelCard` Obsolete Lifecycle Fields & `--muon-lr` Parsing**:
   - Add three optional top-level fields to `RunModelCard` (and `build_model_card_from_manifest_entry` / `validate_model_card`):
     - `obsolete: bool = False`
     - `obsolete_reason: str = ""`
     - `obsolete_at: str = ""`
   - Add `set_model_card_obsolete(card: dict[str, Any], obsolete: bool, reason: str = "") -> dict[str, Any]` in `src/qdlblocks/benchmarks/model_card.py`, which returns a copy of `card` with `obsolete = bool(obsolete)`, `obsolete_reason = str(reason) if obsolete else ""`, and `obsolete_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ") if obsolete else ""`.
   - In `_build_cli_arg_parser()` and `parse_cli_args_to_card_sections(cli_args: str)`:
     - Add `parser.add_argument("--muon-lr", type=float, default=None)`.
     - Set `optimization["muon_lr"] = float(ns.muon_lr) if ns.muon_lr is not None else (0.003 if opt_mode == "pure_adamw" else (0.05 if opt_mode == "sgd_nesterov" else 0.025))`.
3. **Prevent Duplicate `Baseline` Specs in `DynamicWaveRegistry`**:
   - Define `BASELINE_RUN_KEYS = frozenset({"gpu_5ep", "gpu_3ep", "gpu_wsd_mtp", "tpu_3ep", "tpu_wsd_mtp"})`.
   - Ensure that when `collect_snapshot(include_historical_specs=True)` constructs `baseline_defs`, `DynamicWaveRegistry.discover_all_specs()` does not emit a second duplicate spec for keys already in `BASELINE_RUN_KEYS` (or `collect_snapshot` merges them by `key` so each `key` appears in `runs_catalog` **at most once**).

---

### 5.3 Component 1 & 2: `scripts/monitor_colab_runs.py` Updates
1. **Deduplicated & Pre-Sorted `runs_catalog` + Default Obsolete Set**:
   - Define `DEFAULT_OBSOLETE_KEYS: dict[str, str] = {`:
     - `"gpu_5ep": "Superseded single-shard 100M×5ep baseline (Plan 001)"`,
     - `"gpu_3ep": "Superseded single-shard 100M×3ep baseline (Plan 001)"`,
     - `"gpu_wsd_mtp": "Superseded single-shard 100M×5ep WSD+MTP baseline (Plan 001)"`,
     - `"tpu_3ep": "Superseded single-shard TPU 100M×3ep baseline (Plan 001)"`,
     - `"tpu_wsd_mtp": "Superseded single-shard TPU 100M×3ep WSD+MTP baseline (Plan 001)"`,
     - `"w5_900m_deep12_adapter384_oom": "Historical OOM crash prior to gradient checkpointing"`,
     - `"w8_900m_classical_speedrun_12L_t512_layer0": "Stopped early at 200M tok (superseded by Wave-9/10 PR#360 runs)"`
     `}`
   - In `collect_snapshot()`:
     - Load `ui_prefs.json` *before* building `catalog`, reading `obsolete_map: dict[str, bool] = saved_prefs.get("obsoleteMap", {})` and `obsolete_reasons: dict[str, str] = saved_prefs.get("obsoleteReasons", {})`.
     - For any `key` not yet explicitly present in `obsolete_map`:
       - If `card.get("obsolete") is not None` and `key in saved_prefs.get("explicitObsoleteKeys", [])`, use `bool(card["obsolete"])`.
       - Otherwise, default `is_obsolete = (key in DEFAULT_OBSOLETE_KEYS) or (run_state == "OOM_CRASH")`.
     - Deduplicate `catalog` by `key` so every `r["key"]` is 100% unique.
     - Include on every `_build_catalog_entry(...)` dict:
       - `"wave_num": float(wave_num)` (`0.0` for `Baseline`, `2.5` for `W2B`, `float(num)` for `W1..W10`),
       - `"arch_family": str(mc_arch.get("family", "classical_modded" if "classical" in key else "quantum_born"))`,
       - `"obsolete": bool(is_obsolete)`,
       - `"obsolete_reason": str(obsolete_reason)`,
       - `"obsolete_at": str(obsolete_at)`.
     - Sort `catalog` on the backend by `(r["obsolete"], r["best_val_ce"] is None, r["best_val_ce"] if r["best_val_ce"] is not None else 999.0)` ascending so that both the initial JSON snapshot and `format_terminal_report(snap)` list the best active models first and never pin `Baseline` models to the top.
2. **HTTP Endpoint `POST /api/obsolete` (and `POST /api/model-cards/<run_key>/obsolete`)**:
   - Accepts JSON payload:
     ```json
     {
       "run_key": "gpu_5ep",
       "run_keys": ["gpu_5ep", "gpu_3ep"],
       "obsolete": true,
       "reason": "Marked obsolete from dashboard"
     }
     ```
   - Updates `ui_prefs.json` (`obsoleteMap`, `obsoleteReasons`, `explicitObsoleteKeys`), updates each corresponding `.cache/model_cards/<key>.json` on disk via `set_model_card_obsolete` + `save_model_card_atomic`, and updates `state["latest_snap"]["runs_catalog"]` and `state["cards_by_key"]` in memory under `state_lock` in `< 15 ms` with `0` server restarts.
   - Returns:
     ```json
     {
       "ok": true,
       "updated_keys": ["gpu_5ep", "gpu_3ep"],
       "obsolete": true,
       "obsolete_count": 7,
       "active_count": 54,
       "server_uptime_s": 123.45
     }
     ```
3. **Interactive React Frontend Upgrades in `scripts/monitor_colab_runs.py`**:
   - **Strict Unpinned Column Sorting (`filteredAndSortedRuns`)**:
     - When `sortField === "wave_id"`, compare numeric `(a.wave_num ?? 0) - (b.wave_num ?? 0)` (with tie-breaking on `a.best_val_ce ?? 999`).
     - When `sortField === "short_label"`, compare normalized strings with leading `[session] ` stripped (`a.short_label.replace(/^\[[^\]]+\]\s*/, "")`) so `"Baseline ..."` and `"[gpu_...] W9 ..."` sort purely alphabetically by run title without bracket-prefix distortion.
     - For all numeric columns (`best_val_ce`, `best_val_ppl`, `best_top1`, `best_top5`, `ep05_ce`, `ep10_ce`, `params_M`, `seed_dim`, `final_r_eff`, `final_iso_err`, `avg_tok_per_s`, `wall_min`), sort strictly by the numeric value (`null` values always placed at the bottom regardless of `sortAsc`).
   - **Persistent "Mark Obsolete" / "Show Obsolete (`N`)" & Architecture Family Controls**:
     - Add state `showObsolete` (default `false`), `filterFamily` (`"ALL"` | `"quantum_born"` | `"classical_modded"`), and `obsoleteMap` (initialized from `snap.runs_catalog` and `saved_ui_prefs.obsoleteMap`).
     - Add `toggleRunObsolete(runKeys, targetObsolete, reason)` which optimistically updates React state (`obsoleteMap`, and when marking obsolete, unchecks the run from `visibleMap` so it doesn't clutter the chart) and sends `POST /api/obsolete`.
     - In the Table Filter Bar:
       - Add **Architecture Family Filter Pills**: `All Families`, `⚛️ Quantum Born`, `🏛️ Classical Baseline`.
       - Add **Obsolete Visibility Toggle Button**: **`🗄️ Show Obsolete (${obsoleteCount})`** (highlighted when active) and status pill **`🗄️ Obsolete Only (${obsoleteCount})`**.
       - Add bulk action buttons: **`🗄️ Mark Filtered Obsolete`** and (when `showObsolete` or `filterStatus === "OBSOLETE"`) **`♻️ Restore Filtered`**.
     - In each Leaderboard Table row (and inside `ModelCardModal`):
       - Render a compact **`🗄️`** button (`title="Mark this run as obsolete and hide from default table"`) when `!isObsolete`, or a **`♻️ Restore`** button (`title="Restore this obsolete run to active leaderboard"`) + **`OBSOLETE`** pill badge when `isObsolete`.
   - **Dynamic Top KPI Summary Cards & Chart Target Lines**:
     - Compute `bestClassicalRun`, `bestQuantumRun1`, and `bestQuantumRun2` dynamically from non-obsolete runs in `catalog` so the KPI cards immediately display Wave-9's `3.4280 CE (30.8 PPL)` Classical record and `3.7203 CE (41.3 PPL)` / `3.7495 CE (42.5 PPL)` Quantum records (and automatically update as Wave-10 achieves new records!).

---

### 5.4 Component 3: `--muon-lr` in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` & `configs/wave10_fleet.json`
1. **`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`**:
   - Add `--muon-lr` (`type=float, default=0.025`) to `parse_args()`, `run_cuda_benchmark(..., muon_lr: float = 0.025)`, `run_tpu_benchmark(..., muon_lr: float = 0.025)`, and `main()`.
   - In `run_cuda_benchmark`: set `base_muon_lr = float(muon_lr)` when `optimizer_mode == "muon_adamw"`, logging `MuonLR={base_muon_lr:g}` in the architecture/schedule header.
2. **Wave-10 Declarative Fleet Manifest (`configs/wave10_fleet.json`)**:
   Create `/home/festivity/agy/QuantumDLBlocks/configs/wave10_fleet.json` specifying our 5-GPU `900M`-token Wave-10 experiments:

| Slot / Session | Hardware | Run Key (`wave10_fleet.json`) | Architecture & Scientific Objective (`900M` Unique FineWeb Tokens, `T_val = 512`) |
| :---: | :---: | :--- | :--- |
| **`gpu_g4_w6_1`** | G4 RTX PRO 6000 Blackwell (96GB) | **`w10_900m_classical_speedrun_v3_fastsdpa_b131k`** | **`ClassicalModdedGPT` Speedrun V3 (`L=12, d=768, H=12, d_ff=3072`)**: `B=128, G_acc=2, T=512` (**`131,072` tok/step = `6,858` steps**, `2.0×` Wave-9's steps), **`--muon-lr 0.04`**, `--muon-momentum-warmup 300`, `ve=3`, `scalar` U-Net, `fused_qkv`, `skip_attn=6`, `zero_half_rope`, `true_layer0_v_residual`, `zero_init_classical`, fast FlashAttention-2 causal SDPA (`window=0, doc_mask=False`), `direct_chunked_ce`, `stage_shard_on_gpu`, `WSD-0`. Tests `2×` step resolution + higher Muon LR + fast aligned causal SDPA to push past `3.4280` toward **`<= 3.35` Val CE**. |
| **`gpu_g4_w6_2`** | G4 RTX PRO 6000 Blackwell (96GB) | **`w10_900m_classical_speedrun_v3_b65k_13725steps`** | **`ClassicalModdedGPT` Fine-Grained `13,725`-Step Speedrun V3 (`L=12, d=768, H=12, d_ff=3072`)**: `B=128, G_acc=1, T=512` (**`65,536` tok/step = `13,725` steps**, `4.0×` Wave-9's steps!), **`--muon-lr 0.035`**, `--muon-momentum-warmup 300`, `ve=3`, `scalar` U-Net, `fused_qkv`, `skip_attn=6`, `zero_half_rope`, `true_layer0_v_residual`, `zero_init_classical`, fast FlashAttention-2 causal SDPA, `direct_chunked_ce`, `stage_shard_on_gpu`, `WSD-0`. Directly tests whether full `13,725`-step StiefelMuon trajectory resolution closes the remaining single-GPU gap toward **`3.28` Val CE**. |
| **`gpu_g4_w6_3`** | G4 RTX PRO 6000 Blackwell (96GB) | **`w10_900m_qborn_speedrun_v3_asym1024_frame4d_L8`** | **`QuantumBornGPT` Speedrun V3 Unification (`L=8, d=1024, 4d-Frame 4096×2, Asym-1024 POVM`)**: Unifies Wave-9's **#1 All-Time Quantum Winner** (`L=8 + Asym-1024 + DynamicBeta`, `3.7203` CE) with Wave-9's **#2 All-Time Quantum Winner** (`4d-Frame 4096×2`, `3.7495` CE) at `65,536` tok/step (`13,725` steps) + `ve=3`, `scalar` U-Net, `fused_qkv`, `zero_half_rope`, `true_layer0_v_residual`, `FlashPurifiedBorn` (`window=384 + doc_mask`), `ortho_decouple`, `povm_log_prior`, `gated_spherical_residual`, `WSD-0`. Target: **`< 3.66` Val CE (`< 38.8` PPL)**. |
| **`gpu_a100_w6_1`** | NVIDIA A100-SXM4-40GB | **`w10_900m_qborn_speedrun_v3_asym1024_L12_deep`** | **`QuantumBornGPT` Deep `L=12` Speedrun V3 (`L=12, d=1024, 3d-Frame 3072×2, Asym-1024 POVM, skip_attn=6`)**: Scales Wave-9's `3.7203` winner from `L=8` to **`L=12` layers** with `depth_scaled_residual=True`, interior attention skip `--skip-attn-layers 6`, `ve=3`, `scalar` U-Net, `fused_qkv`, `zero_half_rope`, `true_layer0_v_residual`, `FlashPurifiedBorn` (`window=384 + doc_mask`), `DynamicBeta`, `WSD-0` at `65,536` tok/step (`13,725` steps). Target: **`< 3.65` Val CE (`< 38.5` PPL)**. |
| **`gpu_a100_w6_2`** | NVIDIA A100-SXM4-40GB | **`w10_900m_classical_opt_ablation_sgd_nesterov`** | **900M-Token Optimizer Ablation #3 (`--optimizer-mode sgd_nesterov`)**: Exact same `ClassicalModdedGPT` PR #360 V2 architecture (`L=12, d=768, ve=3, scalar U-Net, fused_qkv, skip_attn=6, zero_half_rope, WSD-0`) trained with **Mini-Batch SGD + Nesterov Momentum (`lr=0.05, momentum=0.95`)** on the 2D backbone matrices to complete our 3-way 900M-token comparison (`StiefelMuon` `3.4280` vs. `Pure AdamW` `3.5025` vs. `SGD+Nesterov`). |

---

## 6. Verification & Unit Testing Plan

1. **Unit Test Suite (`tests/test_hypothesis_018_dashboard_obsolete_and_wave10.py`)**:
   - `test_catalog_deduplication_and_unpinned_baseline_sorting`: Calls `collect_snapshot(skip_remote_poll=True)` and verifies:
     - Every `r["key"]` in `runs_catalog` is unique (`0` duplicates of `gpu_5ep`, `gpu_3ep`, `gpu_wsd_mtp`, `tpu_3ep`, `tpu_wsd_mtp`).
     - Every entry has numeric `wave_num`, `arch_family` in `("quantum_born", "classical_modded")`, and boolean `obsolete`.
     - Sorting `runs_catalog` by `best_val_ce` ascending places `w9_900m_classical_speedrun_v2_pr360` (`3.4280`) #1 overall and `w9_900m_qborn_speedrun_v2_asym1024_L8` (`3.7203`) #1 among `quantum_born` runs, with all `"Baseline"` runs (`wave_num == 0`) at rank `>= 40` (never pinned to the top).
     - Sorting by `wave_num` descending places `W10 > W9 > W8 > ... > W1 > Baseline`.
   - `test_zero_downtime_obsolete_endpoint_and_persistence`: Boots `create_benchmark_http_server` on an ephemeral port, sends `POST /api/obsolete` to mark `["w6_900m_classical_modded_124m", "gpu_5ep"]` obsolete, verifies `< 25 ms` latency, `0` server restarts, disk persistence in both `.cache/model_cards/*.json` and `ui_prefs.json`, and then restores `"w6_900m_classical_modded_124m"` with `"obsolete": false` and verifies immediate restoration.
   - `test_wave10_manifest_and_muon_lr_card_registration`: Registers `configs/wave10_fleet.json` via `DynamicWaveRegistry`, validates all 5 Wave-10 `RunModelCard` JSON files (`validate_model_card`), and verifies `--muon-lr 0.04` / `0.035` and `sgd_nesterov` are accurately parsed into `card["optimization"]["muon_lr"]` and `card["optimization"]["optimizer_mode"]`.
   - `test_html_dashboard_contains_obsolete_and_family_controls`: Renders `render_html_dashboard(snap)` and verifies the React bundle includes the `POST /api/obsolete` handler, `showObsolete` toggle (`Show Obsolete`), `filterFamily` (`Quantum Born` / `Classical Baseline`), numeric `wave_num` sorting, and dynamic Top KPI card computation.
2. **Controlled Verification & Benchmark Script (`examples/28_verify_plan018_dashboard_and_wave10.py`)**:
   - Executes all H1, H2, and H3 empirical benchmarks (catalog deduplication/sorting audit, live zero-downtime `POST /api/obsolete` latency benchmark, and GPU causal FlashAttention-2 SDPA vs. 4D-mask SDPA throughput + `2×` step-budget convergence experiment) and writes `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_018.json`.
