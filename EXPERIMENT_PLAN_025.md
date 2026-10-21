# EXPERIMENT PLAN 025: Controlled 8 GB VRAM Classical vs. Quantum Training Dynamics, Target-Rank Cross-Entropy Decomposition, & Wave 13 + Wave 15 Mechanism Unification on RTX 5060 Ti

**Status:** Drafted for Independent Blind Review  
**Date:** 2026-10-20  
**Hardware Target:** Local NVIDIA GeForce RTX 5060 Ti (`16,311 MiB` total VRAM, strictly capped to **$\le 8.0\text{ GB}$ (`50%` VRAM)** via `torch.cuda.set_per_process_memory_fraction(0.5)`) while 5 Wave 15 `900M`-token runs execute concurrently on Google Colab (`gpu_g4_w6_1..3`, `gpu_a100_w6_1..2`)  
**Predecessor Plans:** `EXPERIMENT_PLAN_019.md` (Leaderboard-Exact `ClassicalModdedGPT`), `EXPERIMENT_PLAN_021.md` (9-Shard PPMI-SVD Init, Hashed N-Grams, & ANVIL Cat A/B/C), `EXPERIMENT_PLAN_023.md` (`50,304 × 16 × 32` Unigram Qubit Table & `256`-State Interacting Quantum Hierarchy), `EXPERIMENT_PLAN_024.md` (Calibrated Last-Layer Softmax Readout & Selective 4-Session Wave 15 Hotfix)

---

## 1. Standalone Project Context, Empirical Puzzle Across Waves 11–15, & Motivation

### 1.1 Project Overview: What We Are Building and Why
In **QuantumDLBlocks**, we are developing and benchmarking quantum-inspired deep learning architectures (**`QuantumBornGPT`**) against state-of-the-art classical speedrun transformers (**`ClassicalModdedGPT`**, modeled on the official `KellerJordan/modded-nanogpt` speedrun leaderboard) on the `900M`-token (`9 × 100M`-token shard) **FineWeb-Edu** pretraining benchmark (`vocab_size = 50,304` padded from the `50,257` GPT-2 tokenizer) and evaluated on the exact `10,485,760`-token (`10 × 2^20`) `KellerJordan/modded-nanogpt` validation shard (`fineweb_val_000000.bin`).

In `QuantumBornGPT`:
1. **Hyperspherical State Manifold ($S^{d-1}$)**: Every token state is a unit-norm real amplitude ket $|\psi_t^{(l)}\rangle \in S^{d-1}$ ($\|\psi_t^{(l)}\|_2 = 1$).
2. **Stiefel-Manifold Orthogonal Operators ($O(d)$)**: Linear transformations (`qkv_proj`, `out_proj`, two-beam overcomplete interference frames, and 3D higher-state interaction operators) are evolved on the Stiefel manifold via **`StiefelMuon`** using 6-stage quintic **`PolarExpress`** orthogonalization, **`NorMuon`** row-variance equalization, and **Twin-Rail Nesterov Momentum**.
3. **16-Qubit Multi-Register Amplitudes & 256-State Interacting Quantum Hierarchy (Plans 023 & 024 / Wave 15)**:
   - Each of the `50,304` vocabulary tokens is parameterized by `Q = 16` qubit sub-registers of `A = 32` amplitudes (`50,304 × 16 × 32`, initialized from 9-shard `900M`-token Shifted PPMI + Randomized SVD collocation eigenvectors).
   - `M = 256` higher-level quantum states $\Phi \in \mathbb{R}^{256 \times 16 \times 32}$ interact via a per-register unitary tensor $U_{\text{high}} \in \mathbb{R}^{16 \times 256 \times 256}$ ($U_{\text{high}, q}^\top U_{\text{high}, q} = I_{256}$) to route token representations bottom-up (`50,304 -> 256`), laterally (`256 <-> 256`), and top-down (`256 -> 50,304`).
   - In Plan 024 (`--last-layer-softmax --logit-softcap 15.0`), we decoupled the unit-direction readout vectors $\hat{w}_v \in S^{d-1}$ from learnable per-token radial norms $r_v > 0$ and applied soft-capped linear softmax readout $z_{t,v} = 15 \tanh((\langle W_{\text{soft}, v}, \text{RMSNorm}(\psi_t) \odot g_{\text{rms}} \rangle + b_v)/15)$.

### 1.2 The Empirical Puzzle Across Waves 12, 13, and 15
Comparing our best completed and live `900M`-token runs across **Classical (`ClassicalModdedGPT`)**, **Flat Quantum (`Wave 13 QuantumBornGPT`)**, and **Hierarchical 16-Qubit Quantum (`Wave 15 QuantumBornGPT`)** reveals a striking divergence in training behavior:

| Model Family & Run Key | Architecture & Key Mechanisms | Tokens Seen | Held-Out Val CE | Val PPL | Top-1 Acc (%) | Top-5 Acc (%) | `Top-5 - Top-1` Gap (%) | Median Target Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ClassicalModdedGPT` (`Wave 12`)**<br>`w12_900m_classical_leaderboard_exact_deep_L16_b262k` | Euclidean residual (`L=16, d=768, H=6`), `VE=3`, `SkipAttn`, `ZeroHalfRoPE`, `Softcap=15` | `898.9M` | **`3.3362`** | **`28.11`** | `38.80%` | **`60.93%`** | **`22.13%`** | `3` |
| **`ClassicalModdedGPT` (`Wave 13`)**<br>`w13_900m_classical_anvil2_full_catABC_b131k` | Euclidean residual (`L=12, d=768, H=6`), ANVIL Cat A/B/C, Hashed N-Gram, `Softcap=15` | `898.9M` | **`3.3583`** | **`28.74`** | `38.66%` | **`60.73%`** | **`22.07%`** | `3` |
| **`QuantumBornGPT` Flat (`Wave 13`)**<br>`w13_900m_qborn_v6_9shard_ppmi_svd_deep_L12_H8_d128` | Flat $S^{1023}$ (`L=12, d=1024, H=8`), `VE=3`, `ZeroHalfRoPE`, `Win=384`, `DepthScaledRes`, `Ngram` | `899.5M` | **`3.5121`** | **`33.52`** | `37.35%` | `58.91%` | **`21.56%`** | `3` |
| **`QuantumBornGPT` 16-Qubit Hierarchy (`Wave 15`)**<br>`w15_900m_qborn_16x32_qubit_256hier_flagship_L8` | `16×32` Qubits + `256` States (`L=8, d=1024, H=8`), Pure POVM *(No `VE`, No `ZeroHalfRoPE`, No `Win=384`)* | `724.6M` | `3.6627` | `38.97` | **`40.78%`** | `59.10%` | `18.32%` | `3` |
| **`QuantumBornGPT` 16-Qubit + Softmax (`Wave 15 Live`)**<br>`w15_900m_qborn_16x32_qubit_256hier_ngram_softmax_L8` | `16×32` Qubits + `256` States + Hashed N-Gram + `--last-layer-softmax` (`cap=15`) | `49.9M` *(Step 762)* | **`4.3424`** *(vs `4.3631` POVM)* | `76.89` | **`33.79%`** | **`50.71%`** *(+0.45% vs POVM)* | `16.92%` | `6` |

Look closely at what this table reveals:
1. **Why does Wave 15 (`16×32` Qubits + `256` States) jump to `40.78%` Top-1 accuracy (surpassing `ClassicalModdedGPT`'s `38.80%` Top-1 by `+1.98%`), yet its `Top-5 - Top-1` gap is `18.32%` (vs `21.56%` in Wave 13 Quantum and `22.13%` in Wave 12 Classical) and its Val CE is `3.6627` (vs `3.5121` in Wave 13 Quantum and `3.3362` in Wave 12 Classical)?**
2. **Two Distinct Factors Are Compounding Between Wave 13, Wave 15, and Classical**:
   - **Factor A — Target-Rank Probability Calibration & Tail NLL Penalty**:
     Cross-Entropy $\text{CE} = \mathbb{E}[-\log P(y_t \mid x_{\le t})]$ is the probability-weighted sum of conditional negative log-likelihoods across target ranks:
     $$\text{CE} = \sum_{b \in \{1,\; 2..5,\; 6..50,\; 51..1000,\; >1000\}} \underbrace{p(r_t \in b)}_{\text{Rank Bucket Share}} \cdot \underbrace{\mathbb{E}\!\left[-\log P(y_t \mid x_{\le t}) \;\middle|\; r_t \in b\right]}_{\text{Conditional Bucket NLL}}$$
     Even when a model wins `Rank 1` on `40.78%` of tokens, if its output distribution is over-peaked on Rank 1 or assigns excessive mass to the 256-cluster average while under-allocating mass to Ranks `2–50` or over-penalizing surprising targets ($r_t > 1000$), the conditional NLL on misses ($r_t \ge 2$) inflates overall Cross-Entropy by `0.20–0.35 nats`!
   - **Factor B — Omitted Wave 13 Backbone Primitives & Over-Repeated Per-Block Hierarchy Causal Smearing in Wave 15**:
     Comparing the Wave 13 CLI config (`configs/wave13_fleet.json`) against the Wave 15 CLI config (`configs/wave15_fleet.json`), Wave 13 (`3.5121` Val CE, `21.56%` `Top-5 - Top-1` gap) included **four crucial backbone primitives** that were omitted from Wave 15's `L=8` models:
     1. `--num-value-embeddings 3` (`VE=3` token-identity value embeddings injected into early and late attention layers),
     2. `--zero-half-rope` (preserving half of each head's channels as position-invariant semantic subspace),
     3. `--depth-scaled-residual` ($1/\sqrt{2L}$ residual branch scaling),
     4. `--sliding-window 384` (preventing long-range attention dilution across distant context),
     AND in Wave 15, `Hierarchical16QubitEmbedder.forward_block_bridge` calls `route_through_hierarchy(..., include_causal_memory=True)` at **every single transformer block** ($L = 8$ times per forward pass in addition to the embedding layer!). Applying a 4-token exponential moving average (`causal_mem` over $t-1 \dots t-4$) inside *every* intermediate transformer block (after self-attention has already mixed context) over-smooths the mid-layer representations toward local 4-gram templates—boosting immediate next-token `Top-1` hits on repetitive local phrases while blurring subtle contextual runner-ups (`Ranks 2–50`)!

---

## 2. Goals & Outcomes of Experiment Plan 025 (Local 8 GB VRAM Investigation on RTX 5060 Ti)

While our 5 Wave 15 `900M`-token runs continue on Google Colab, we will use **at most `8.0 GB` of the `16.0 GB` VRAM** on our local **NVIDIA GeForce RTX 5060 Ti** (`torch.cuda.set_per_process_memory_fraction(0.5)`) to conduct a rigorous, controlled experimental study of **Classical vs. Quantum training behavior**:

1. **Exact Target-Rank Cross-Entropy Decomposition & Calibration Probe (`src/qdlblocks/benchmarks/calibration_probe.py` & `scripts/calibration_rank_histogram_probe.py`)**:
   - Build a fast, GPU-vectorized diagnostic analyzer that decomposes Validation Cross-Entropy into the 5 canonical target-rank buckets (`Rank 1`, `Ranks 2–5`, `Ranks 6–50`, `Ranks 51–1000`, `Ranks > 1000`), measuring each bucket's token fraction $p(r_t \in b)$, conditional NLL $\mathbb{E}[-\log P(y_t) \mid r_t \in b]$, contribution to total CE $p(r_t \in b) \cdot \text{NLL}(b)$, cumulative top-$k$ probability mass ($\sum_{k=1}^1 P_{(k)}$, $\sum_{k=2}^5 P_{(k)}$, $\sum_{k=6}^{50} P_{(k)}$), prediction Shannon entropy $H(P_t)$, and Expected Calibration Error (`ECE`).
2. **Layerwise Training Dynamics & Representation Geometry Tracker (Classical Euclidean vs. Quantum Hyperspherical)**:
   - Track how Classical (`ClassicalModdedGPT`) and Quantum (`QuantumBornGPT`) networks evolve across training steps (`Step 0 -> Step 300`):
     - Layerwise angular state velocity $\theta_l = \arccos(\langle \hat{h}_l, \hat{h}_{l-1} \rangle)$ across blocks $l = 1 \dots L$,
     - Layerwise spectral effective rank $R_\sigma(h_l)$ and participation ratio $R_{\text{PR}}(h_l)$ from embedding ($l=0$) to readout ($l=L$),
     - Gradient norm distribution between embeddings/readout (`AdamW`) and internal backbone matrices (`StiefelMuon`),
     - Stiefel manifold orthogonality error $\|W^\top W - I\|_F / \sqrt{d}$.
3. **Controlled 5-Model FineWeb-Edu Experimental Suite Strictly Within $\le 8.0\text{ GB}$ VRAM (`examples/35_verify_plan025_8gb_classical_vs_quantum_dynamics.py`)**:
   - Train 5 matched-budget models (`300` optimization steps each on genuine FineWeb-Edu tokens from `.cache/fineweb_100M/train_full_100M.bin`, `B_micro = 8, grad_accum = 2, T = 512` = `8,192` tokens/step = `2.46M` tokens/run, evaluated on `.cache/fineweb_100M/val_slice.bin` using the exact `KellerJordan/modded-nanogpt` sequential validation protocol) under a hard `8.0 GB` VRAM limit (`torch.cuda.set_per_process_memory_fraction(0.5)`):
     1. **`w15_8gb_classical_leaderboard_exact_L6_d512`**: Leaderboard-exact `ClassicalModdedGPT` (`L=6, d=512, H=8, d_ff=2048`, `VE=3`, `SkipAttn`, `Softcap=15.0`).
     2. **`w15_8gb_qborn_flat_w13_v6_L6_d512`**: Wave 13 style flat `QuantumBornGPT` (`L=6, d=512, H=8`, `4d-TwoBeam`, `VE=3`, `ZeroHalfRoPE`, `DepthScaledRes`, `Win=384`, 9-Shard PPMI-SVD Init, Asymmetric POVM).
     3. **`w15_8gb_qborn_16qubit_256hier_povm_L6_d512`**: Wave 15 style `16×16` Qubit + `256`-State Hierarchical `QuantumBornGPT` with pure unit-sphere POVM readout (`--last-layer-softmax` OFF).
     4. **`w15_8gb_qborn_16qubit_256hier_softmax_L6_d512`**: Wave 15 + Plan 024 `16×16` Qubit + `256`-State Hierarchical `QuantumBornGPT` with Calibrated Last-Layer Softmax (`--last-layer-softmax --logit-softcap 15.0`).
     5. **`w15_8gb_qborn_16qubit_256hier_w13w15_unified_L6_d512`**: **Unified Wave 13 + Wave 15 + Plan 024 Quantum Architecture** — combining `16×16` Unigram Qubits + `256`-State Hierarchy + Calibrated Last-Layer Softmax (`softcap=15.0`) + Hashed Bigram/Trigram Embeddings (`ngram_hash`) + Wave 13's `VE=3`, `ZeroHalfRoPE`, `DepthScaledRes`, `Win=384`, **and restricting `include_causal_memory=True` to the embedding layer (`embed_only_causal_memory`) so intermediate block bridges (`forward_block_bridge`) route pure contextual token states without re-smearing $[t-1 \dots t-4]$ at every block!**

---

## 3. Observable Hypotheses & Quantitative Predictions

### Hypothesis H1: Target-Rank Cross-Entropy Decomposition & Calibration Mechanism (Classical vs. Quantum POVM vs. Quantum Calibrated Softmax)
- **Claim**: Decomposing held-out validation Cross-Entropy into the 5 target-rank buckets (`Rank 1`, `Ranks 2–5`, `Ranks 6–50`, `Ranks 51–1000`, `Ranks > 1000`) will prove that:
  1. Pure unit-sphere POVM (`w15_8gb_qborn_16qubit_256hier_povm_L6_d512`) suffers a higher conditional NLL on non-Rank-1 targets ($\mathbb{E}[-\log P(y_t) \mid r_t \in [2, 50]]$ and $\mathbb{E}[-\log P(y_t) \mid r_t > 1000]$) and a narrower `Top-5 - Top-1` gap than both `ClassicalModdedGPT` and Calibrated Softmax `QuantumBornGPT`.
  2. Enabling Calibrated Last-Layer Softmax (`w15_8gb_qborn_16qubit_256hier_softmax_L6_d512`) reduces the conditional NLL on `Ranks 2–50` and `Ranks > 1000`, lowers Expected Calibration Error (`ECE`), widens the `Top-5 - Top-1` gap, and lowers overall Validation Cross-Entropy by **$\ge 0.04\text{ nats}$** compared to pure POVM at Step 300.
- **Quantitative Pass Criteria**:
  1. The 5-bucket CE decomposition identity $\left|\sum_{b} p(r_t \in b) \cdot \text{NLL}(b) - \text{Val CE}\right| < 10^{-4}$ holds on all evaluated models.
  2. `w15_8gb_qborn_16qubit_256hier_softmax_L6_d512` achieves **$\ge 0.04\text{ nats}$** lower Val CE and a wider `Top-5 - Top-1` gap than `w15_8gb_qborn_16qubit_256hier_povm_L6_d512`.

### Hypothesis H2: Unifying Wave 13 Backbone Primitives (`VE=3`, `ZeroHalfRoPE`, `DepthScaledRes`, `Win=384`) + Embedding-Only Causal Hierarchy Memory With Wave 15 (`16-Qubit + 256-State + Softmax`)
- **Claim**: In Wave 15, calling `include_causal_memory=True` inside `forward_block_bridge` at every transformer block $l = 0 \dots L-1$ compounds 4-step temporal smoothing $L+1$ times across depth, while omitting Wave 13's `VE=3`, `ZeroHalfRoPE`, `DepthScaledRes`, and `Win=384` weakens mid-layer contextual disambiguation (`Ranks 2–50`). Adding `bridge_causal_memory: bool = False` (so 4-step causal hierarchy memory is applied once at the embedding lift while per-block bridges route pure contextual states) AND restoring Wave 13's `VE=3 + ZeroHalfRoPE + DepthScaledRes + Win=384 + Hashed N-Gram` in `w15_8gb_qborn_16qubit_256hier_w13w15_unified_L6_d512` will outperform both the Wave 13 flat quantum model (`w15_8gb_qborn_flat_w13_v6_L6_d512`), the Wave 15 softmax model (`w15_8gb_qborn_16qubit_256hier_softmax_L6_d512`), and the matched Classical baseline (`w15_8gb_classical_leaderboard_exact_L6_d512`)!
- **Quantitative Pass Criteria**:
  1. `w15_8gb_qborn_16qubit_256hier_w13w15_unified_L6_d512` achieves the **lowest held-out Validation Cross-Entropy** among all 4 Quantum models and beats `w15_8gb_qborn_16qubit_256hier_povm_L6_d512` by **$\ge 0.10\text{ nats}$** at Step 300.
  2. Layerwise representation geometry tracking confirms that `w15_8gb_qborn_16qubit_256hier_w13w15_unified_L6_d512` maintains higher final-layer spectral effective rank $R_\sigma(\psi_L)$ and lower Stiefel orthogonality error (`IsoErr < 0.02`) than uncalibrated POVM.

### Hypothesis H3: Strict $\le 8.0\text{ GB}$ VRAM Hardware Envelope & Full-Scale `d=1024, L=8` (`255M`-Param) Memory-Capped Training on RTX 5060 Ti
- **Claim**: By enforcing `torch.cuda.set_per_process_memory_fraction(0.5)` (`<= 8.0 GB` hard cap on the `16 GB` RTX 5060 Ti), all 5 `L=6, d=512` (`V=50,304`) models will train and evaluate in **$\le 5.5\text{ GB}$ Peak VRAM** (`0` OOM errors). Furthermore, even the full-scale Wave 15 flagship architecture (`L=8, d=1024, H=8, V=50,304`, `16×32` unigram qubits + `256` higher states + `--last-layer-softmax`, **`255.33M` parameters** — which used `33.36 GB` on Colab at `B=32, T=512` without activation checkpointing) can be trained and evaluated on our local RTX 5060 Ti in **strictly $\le 8.0\text{ GB}$ Peak VRAM** using `B_micro = 4, T = 512`, per-block activation checkpointing, and chunked soft-cap cross-entropy (`direct_chunked_softcap_linear_ce`).
- **Quantitative Pass Criteria**:
  1. Every experiment in the suite records `peak_vram_gb <= 8.0` (`torch.cuda.max_memory_allocated() / 1e9 <= 8.0`) under `torch.cuda.set_per_process_memory_fraction(0.5)`.
  2. A 30-step local memory-envelope verification of the full `255.33M`-parameter `L=8, d=1024` `16×32`-Qubit + `256`-State + Softmax flagship completes with decreasing loss and `peak_vram_gb <= 8.0` on the RTX 5060 Ti.

---

## 4. Planned Code Changes

### 4.1 `src/qdlblocks/benchmarks/calibration_probe.py` (New Module)
Create `src/qdlblocks/benchmarks/calibration_probe.py` (and export in `src/qdlblocks/benchmarks/__init__.py`) providing:
1. `RANK_BUCKETS: tuple[tuple[str, int, int], ...]`:
   - `("rank_1", 1, 1)`
   - `("ranks_2_5", 2, 5)`
   - `("ranks_6_50", 6, 50)`
   - `("ranks_51_1000", 51, 1000)`
   - `("ranks_gt_1000", 1001, 1_000_000_000)`
2. `compute_calibration_rank_decomposition(logits_2d: torch.Tensor, targets_1d: torch.Tensor, valid_vocab_size: int = 50257, num_ece_bins: int = 10) -> dict[str, Any]`:
   - Computes exact `float32` log-probabilities $\log P(v \mid x_t)$ and probabilities $P(v \mid x_t)$ over `0 .. valid_vocab_size - 1`.
   - Computes the exact rank $r_t = 1 + \sum_{v \ne y_t} \mathbb{I}[\text{logit}_{t,v} > \text{logit}_{t, y_t}]$ of the ground-truth token $y_t$ for each token $t$.
   - For each bucket $b \in \text{RANK\_BUCKETS}$, computes:
     - `fraction`: $p(r_t \in b) \in [0, 1]$,
     - `mean_nll`: $\mathbb{E}[-\log P(y_t) \mid r_t \in b]$,
     - `ce_contribution`: $p(r_t \in b) \cdot \mathbb{E}[-\log P(y_t) \mid r_t \in b]$ (satisfying $\sum_b \text{ce\_contribution}_b = \text{total\_ce}$ to within $10^{-5}$),
     - `mean_target_prob`: $\mathbb{E}[P(y_t) \mid r_t \in b]$.
   - Computes sorted candidate probability mass allocation across every token prediction:
     - `mean_top1_prob_mass`: $\mathbb{E}[P_{(1)}]$,
     - `mean_ranks_2_5_prob_mass`: $\mathbb{E}[\sum_{k=2}^5 P_{(k)}]$,
     - `mean_ranks_6_50_prob_mass`: $\mathbb{E}[\sum_{k=6}^{50} P_{(k)}]$,
     - `mean_tail_gt_50_prob_mass`: $\mathbb{E}[\sum_{k>50} P_{(k)}]$,
     - `mean_prediction_entropy_nats`: $\mathbb{E}[-\sum_v P_t(v) \log P_t(v)]$,
     - `logit_std`: mean per-token standard deviation of logits $\mathbb{E}[\text{std}_v(z_{t,v})]$,
     - `ece`: 10-bin Expected Calibration Error on top-1 confidence $P_{(1)}$ vs empirical top-1 accuracy $\mathbb{I}[\arg\max_v z_{t,v} = y_t]$.
3. `compute_layerwise_dynamics_profile(model: nn.Module, x: torch.Tensor, y: torch.Tensor, valid_vocab_size: int = 50257) -> dict[str, Any]`:
   - Runs a forward + backward probe pass on `(x, y)` to extract:
     - `layer_angular_velocity_deg`: list of mean angular rotations $\frac{180}{\pi} \arccos(\text{clamp}(\langle \hat{h}_l, \hat{h}_{l-1} \rangle, -1, 1))$ for $l = 1 \dots L$,
     - `layer_effective_ranks`: list of Shannon spectral effective ranks $R_\sigma(h_l)$ for $l = 1 \dots L$,
     - `layer_participation_ratios`: list of participation ratios $R_{\text{PR}}(h_l)$ for $l = 1 \dots L$,
     - `grad_norm_embed_readout`: L2 norm of gradients on embedding + readout parameters,
     - `grad_norm_backbone_muon`: L2 norm of gradients on internal 2D/3D backbone matrices,
     - `embed_to_backbone_grad_ratio`: ratio of embedding/readout gradient norm to backbone gradient norm.

### 4.2 `src/qdlblocks/layers/hierarchical_qubit.py` & `src/qdlblocks/models/born_gpt.py`
- Add optional boolean attribute `bridge_causal_memory: bool = True` (default `True` for backward compatibility, configurable via `bridge_causal_memory=False` in `Hierarchical16QubitEmbedder` and `QuantumBornGPT`) so `forward_block_bridge` passes `include_causal_memory=self.bridge_causal_memory` to `route_through_hierarchy(...)`. When `bridge_causal_memory=False`, the 4-tap causal hierarchy memory ($t-1 \dots t-4$) is applied once at the initial token embedding lift (`embedder.forward`), while the per-block bridges (`forward_block_bridge` at layers $l = 0 \dots L-1$) route pure contextual token states through the `256` higher-level states without redundant 4-gram smearing at every block.

### 4.3 `scripts/calibration_rank_histogram_probe.py` (Promoted from `scripts/BACKLOG.md`)
- Implement the standalone CLI & programmatic tool `scripts/calibration_rank_histogram_probe.py` (promoting the item added to `scripts/BACKLOG.md` in Plan 024!) that compares Classical (`ClassicalModdedGPT`), Uncalibrated POVM `QuantumBornGPT`, and Calibrated Softmax `QuantumBornGPT` on target-rank cross-entropy decomposition, top-$k$ probability mass allocation, and layerwise dynamics, printing a formatted Markdown table and optionally writing JSON output.

### 4.4 Verification Tests & 8 GB Local GPU Experimental Suite
1. **`tests/test_hypothesis_025_8gb_classical_vs_quantum_dynamics.py`**:
   - Unit Test 1: Verifies `compute_calibration_rank_decomposition` exact partition identity ($\sum_b p(b) = 1.0$, $\sum_b \text{ce\_contribution}_b = \text{total\_ce}$ within $10^{-5}$, $\sum \text{prob\_mass} = 1.0$ within $10^{-4}$, valid `ece`).
   - Unit Test 2: Verifies `compute_layerwise_dynamics_profile` on both `ClassicalModdedGPT` and `QuantumBornGPT`, returning finite angular velocities, spectral effective ranks, and non-zero gradient norms.
   - Unit Test 3: Verifies `bridge_causal_memory=False` vs `bridge_causal_memory=True` in `Hierarchical16QubitEmbedder` and `QuantumBornGPT` (confirming strict causality `0.0` leakage on prefix tokens and distinct intermediate bridge states).
   - Unit Test 4: Verifies `scripts/calibration_rank_histogram_probe.py --help` and `--quick-test` CLI execution.
2. **`examples/35_verify_plan025_8gb_classical_vs_quantum_dynamics.py`**:
   - Enforces `torch.cuda.set_per_process_memory_fraction(0.5)` at startup so PyTorch can never allocate more than `8.0 GB` of the RTX 5060 Ti's `16 GB` VRAM.
   - Runs the 5 controlled `300`-step FineWeb-Edu models (`w15_8gb_classical_leaderboard_exact_L6_d512`, `w15_8gb_qborn_flat_w13_v6_L6_d512`, `w15_8gb_qborn_16qubit_256hier_povm_L6_d512`, `w15_8gb_qborn_16qubit_256hier_softmax_L6_d512`, `w15_8gb_qborn_16qubit_256hier_w13w15_unified_L6_d512`) + the 30-step full `255.33M`-parameter `L=8, d=1024` 8 GB memory-envelope stress test (`verify025_8gb_full_255m_d1024_L8_envelope`).
   - Records full target-rank CE decompositions and layerwise training dynamics at Step 0, Step 100, Step 200, and Step 300.
   - Saves `.cache/hypothesis_experiments_025.json`, `.cache/live_logs/w15_8gb_*.log`, `.cache/model_cards/w15_8gb_*.json`, ingests all runs into `.cache/benchmarks.db`, and triggers `POST http://127.0.0.1:8765/api/reload`.

---

## 5. Operational Tooling & Script Opportunities (Mandatory Gate)

### 5.1 Must-Have (Implemented in This Plan)
- **`scripts/calibration_rank_histogram_probe.py`** & **`src/qdlblocks/benchmarks/calibration_probe.py`**:
  - **Justification**: Promoted directly from `scripts/BACKLOG.md` (identified in Plan 024). Provides a single-command diagnostic probe to decompose validation Cross-Entropy across `Rank 1`, `Ranks 2–5`, `Ranks 6–50`, `Ranks 51–1000`, and `Ranks > 1000`, alongside layerwise angular velocity, spectral rank $R_\sigma$, and gradient norm ratios for both Classical and Quantum models.

### 5.2 Nice-to-Have (Appended to `scripts/BACKLOG.md`)
- **`scripts/vram_envelope_autotuner.py`**:
  - **Purpose**: Given a hardware VRAM cap (e.g., `--max-vram-gb 8.0` or `16.0`) and a `QuantumBornGPT` or `ClassicalModdedGPT` architecture config, binary-searches `(micro_batch_size, grad_accum, checkpoint_blocks, chunk_tokens)` under `torch.cuda.set_per_process_memory_fraction` to find the highest-throughput configuration that guarantees zero OOMs.
