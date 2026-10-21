# Experiment Plan 021: 9-Shard (`900M`-Token) PPMI + Randomized SVD Co-Occurrence Spectral Initialization, Hashed Bigram+Trigram Embeddings, & Full `modded-nanogpt` Categories A/B/C Reusable Primitives

**Date:** 2026-10-19  
**Status:** Revised After Blind Review Pass 1  
**Target Repository:** `/home/festivity/agy/QuantumDLBlocks`

---

## 1. Self-Contained Project Overview: Problems, Outcomes, History (Waves 1–12), & Next Steps

### 1.1 The Core Problems We Are Solving
This project investigates whether a **quantum-geometric language model** (**`QuantumBornGPT`**) can compete with and ultimately match a state-of-the-art **classical Transformer baseline** (**`ClassicalModdedGPT`**, modeled after the `KellerJordan/modded-nanogpt` speedrun leaderboard) on the **FineWeb-Edu 10B** (`gpt2` tokenizer, vocabulary size $V = 50,257$ padded to $V_{\text{pad}} = 50,304$) benchmark under identical hardware, token budgets ($900\text{M}$ training tokens across 9 distinct $100\text{M}$-token shards), and fixed validation evaluation (`block_size = 512`, `T_val = 512` token context window on the held-out `val_slice.bin` shard).

To make every architectural term in this document 100% self-contained for a reader who has never seen the codebase:
1. **Pure-State Hyperspherical Geometry ($S^{d-1}$) & Dual Embedding Tables in `QuantumBornGPT`**:
   - Instead of unconstrained Euclidean residual vectors $x \in \mathbb{R}^d$ normalized by `RMSNorm`, `QuantumBornGPT` represents each token position as a unit-norm quantum state ket $|\psi_t\rangle \in S^{d-1}$ ($\|\psi_t\|_2 = 1$, default $d = 1024$) and updates states via **Stiefel-manifold isometric operators** ($W^\top W = I$) optimized with **`StiefelMuon`**.
   - **How `QuantumBornGPT` Parameterizes Embeddings & Readout (`dm_seed` vs. `wte` vs. `povm_head.effects` vs. `value_embeds`)**:
     - `QuantumBornGPT` supports two embedding modes controlled by `readout_mode`:
       - **Asymmetric Full-Width Mode (`readout_mode = "asymmetric"`, denoted `asym1024`)**: Each of the $V = 50,304$ tokens has a full-width seed state row in `dm_seed` (`nn.Embedding(V, d)`, shape `(50304, 1024)`), normalized onto the unit sphere $S^{d-1}$ at lookup (`psi_0 = l2_normalize(dm_seed(idx))`). It also maintains a legacy compact seed table `wte` (`nn.Embedding(V, seed_dim)`, default `seed_dim = 128`) and isometric lift `W_lift` (`(d, seed_dim)`) used when `readout_mode != "asymmetric"` or when compact seed adapters (`OvercompleteSeedReadoutAdapter`, `TwoBeamSeedReadoutAdapter`) are enabled.
       - **POVM Measurement Effects (`povm_head.effects`)**: At the output of the $L$-layer quantum backbone, next-token probabilities are computed via a **Positive Operator-Valued Measure (POVM)** over unit-norm measurement vectors $\{|m_v\rangle\}_{v=1}^V \subset S^{d-1}$. When `--ortho-decouple` (`use_ortho_decouple = True`) is enabled, `povm_head.effects` is an untied `nn.Parameter` of shape `(V, d)` initialized in the **odd coordinate subspace** $\{1, 3, 5, \dots, d-1\}$, whereas `dm_seed.weight` is initialized in the **even coordinate subspace** $\{0, 2, 4, \dots, d-2\}$. This guarantees exact Step-0 input-output orthogonality ($\langle m_v | \psi_0(v) \rangle = 0 \implies \text{Step-0 self-repetition rate} = 0$). When `use_ortho_decouple = False` in `asymmetric` mode, `povm_head.effects` is tied to `l2_normalize(dm_seed.weight)` transformed by an orthogonal adapter (`readout_ortho: (d, d)`).
       - **Multi-Bank Value Embeddings (`value_embeds`)**: Following `modded-nanogpt`, when `num_value_embeddings = k` (typically $k = 3$), the model allocates $k$ separate token embedding tables `value_embeds[0..k-1]`, each of shape `(V, d)`, and injects `value_embeds[bank_idx](idx)` directly into the attention value tensor $V^{(l)}$ on the first $k$ layers ($l \in \{0, \dots, k-1\}$) and the last $k$ layers ($l \in \{L-k, \dots, L-1\}$).
   - **Gated Spherical Residual (`--gated-spherical-residual`) & Overcomplete Two-Beam MLP (`frame4d` / `TwoBeamInterferenceMLP`)**:
     - In `QuantumBornBlock`, each sublayer (attention or MLP) produces a unit-norm candidate state $|\phi_t\rangle \in S^{d-1}$ and updates the residual ket via a learnable per-channel gate $g = \sigma(\text{logit}_g)$ (initialized to `0.25`):
       $$|\psi_t'\rangle = \text{Normalize}\!\Big((1 - g) \odot |\psi_t\rangle + g \odot |\phi_t\rangle\Big) \in S^{d-1}$$
     - `TwoBeamInterferenceMLP` (`--two-beam-mlp`, with `--overcomplete-two-beam --frame-expansion 4.0`, denoted `frame4d`) splits the MLP hidden transformation into two parallel isometric beams $u_1 = \text{ReLU}^2(\psi W_1^{(a)}), u_2 = \text{ReLU}^2(\psi W_1^{(b)}) \in \mathbb{R}^{4d}$ that interfere via elementwise product + sum before projecting back to $\mathbb{R}^d$ and normalizing onto $S^{d-1}$.

2. **Born-Rule Quadratic Interference Attention & POVM Readout**:
   - Classical attention computes exponential softmax weights $\propto \exp(\langle q_t, k_s \rangle / \sqrt{d_h})$. `QuantumBornGPT` computes **Born-rule transition probabilities** from quantum state inner products:
     - Standard Born attention: $p_{t,s} \propto |\langle q_t | k_s \rangle|^2$.
     - **Coherence-Thresholded Purified Born Attention** (introduced in Plan 014): suppresses random high-dimensional orthogonality noise ($|\langle q_t | k_s \rangle| \approx 1/\sqrt{d_h}$) using a learnable per-head coherence floor $\tau_h \in (0, \tau_{\max})$ and quartic (2nd-order density-matrix purification $\rho^2 / \text{Tr}(\rho^2)$) amplification:
       $$A_{t,s}^{(h)} = \left[\gamma_h \max\!\big(0,\, \langle q_t^{(h)} | k_s^{(h)} \rangle - \tau_h\big)\right]^4 + \epsilon_{\text{base}} \left[\max\!\big(0,\, \langle q_t^{(h)} | k_s^{(h)} \rangle\big)\right]^2$$
   - Next-token probabilities are extracted via the POVM readout with inverse temperature $\beta = 1/\tau_{\text{povm}}$ (which can be modulated per token by `ContextualBlochPurityGate`, `--dynamic-povm-beta`) and empirical unigram log-prior bias $b_v$ (`--povm-log-prior`):
     $$P(v \mid \psi_t) = \frac{\exp\!\big(\beta \, |\langle m_v | \psi_t \rangle|^2 + b_v\big)}{\sum_{u=1}^V \exp\!\big(\beta \, |\langle m_u | \psi_t \rangle|^2 + b_u\big)}$$

Across Waves 1–12, we have been solving **two coupled problems**:
1. **Problem 1 — Closing the Classical & Quantum Sample-Efficiency Gap vs. the Official `KellerJordan/modded-nanogpt` Speedrun Leader**:
   - In Plan 019 (Wave 11) and Plan 020 (Wave 12), we implemented the `2025-01-16` `modded-nanogpt` baseline (`--leaderboard-exact-classical`). On our 60-step pre-launch micro-verification benchmark (which trains and evaluates on deterministic contextual Markov-Zipf shards generated by `generate_contextual_zipf_shard`, where ground-truth entropy is lower so CE values sit around `~2.4–3.3`), `--leaderboard-exact-classical` reduced 60-step validation CE by **`-0.3720` nats** (`3.2974 -> 2.9254` CE).
   - However, when we ran Wave 11 on the full `900M`-token real FineWeb-Edu stream (`val_slice.bin`) with `B_tok = 524,288` tokens/step, `900M / 524,288` yielded only `1,716` optimizer steps (compared to `6,866` steps at `B_tok = 131,072` in Wave 10), causing Wave 11 to under-step and finish at `3.7330` Val CE vs. Wave 10's `3.4280`. Wave 12 (currently running on Colab) restored high step counts (`3,433` steps at `B_tok = 262k` and `6,858` steps at `B_tok = 131k`).
   - Furthermore, auditing the **current** `KellerJordan/modded-nanogpt` `master` branch (Record `#360` / `ANVIL2`, `2026-04`, which reaches `3.278` Val CE in `72.4s` on `8xH100` using only `~330M` tokens) revealed three categories of modern speedrun mechanisms that we had not yet implemented as reusable `QuantumDLBlocks` building blocks:
     - **Category A (Sample-Efficiency & Optimization Primitives)**:
       1. **Tail Weight Averaging + Frobenius Decontraction (`TailWeightAverager`)**: Maintaining an exponential moving average (EMA) of weights over the final cooldown tail (`25%` of steps) and rescaling averaged weight matrices back to their pre-average Frobenius norm (or unit sphere $S^{d-1}$ for quantum states) at validation time.
       2. **`PolarExpress` / `ANVIL` 6-Step Quintic Whitening Cascade + `NorMuon` Per-Lane Variance Equalizer + Twin-Rail Momentum + Sign-Aligned Cautious Weight Decay**: Replacing fixed 5-step Newton-Schulz `(3.4445, -4.7750, 2.0315)` with the 6 re-derived `ANVIL_MAPS` quintic polynomials after Frobenius-margin normalization (`1.05 * ||X||_F`), equalizing row/column update variance via `NorMuon` (`_rail_equalizer`), blending fast (`beta=0.85`) and slow (`beta=0.98`) Nesterov velocity rails, and gating weight decay on sign alignment `(w * u > 0)`.
       3. **Input-Dependent 6-Channel Value-Embedding Gates (`ve_gate`), Per-Head Attention Output Gates (`attn_gate`), Gated eXcluding Self-Attention (`XSA`), & Multi-Way Dynamic Dense (`MUDD`) Skip Connections**: Data-dependent per-token routing of residual streams (`x_0`, intermediate layer caches, and n-gram embeddings) and attention head outputs.
       4. **1-Token Forward `Smear` Gate in Leaderboard Mode + Linearly Tapered Multi-Token Prediction (`--tapered-mtp`)**: Blending `x[t-1]` into `x[t]` via a learned 12-channel gate and linearly decaying auxiliary multi-token prediction weight to `0.0` at `65%` of training so late steps focus purely on single-token cross-entropy.
       5. **Dynamic `YaRN` Attention Temperature & Frequency Scaling + Partial Key Offset (`partial_key_offset`) + Validation Window Extension + Paired/Mixed-Width Heads**.
       6. **Document-Boundary (`<|endoftext|>`, token `50256`) Batch Start Alignment (`--align-to-bos`)**.
     - **Category B (Local N-Gram Context Parameter Bank)**:
       7. **Hashed Bigram + Trigram Embedding Table with Weinberger $\pm 1$ Sign Trick (`HashedNgramEmbedding`)**: Mapping $(x_{t-1}, x_t)$ and $(x_{t-2}, x_{t-1}, x_t)$ via fast deterministic integer multiplicative-XOR hashes into a compact embedding table multiplied by an independent pseudorandom $\pm 1$ sign pool so colliding n-grams cancel in expectation.
     - **Category C (Throughput & Systems Primitives)**:
       8. **Sampled-Softmax Candidate Cross-Entropy (`sampled_softcap_linear_ce` & `sampled_povm_ce`)**: Computing early-stage training cross-entropy over $P \ll V$ candidates (all unique target tokens in the microbatch plus deterministic coprime-stride negatives `k * 20011 mod V`) while keeping validation strictly full-vocabulary ($V = 50,304$).
       9. **Every-$K$-Step Sparse/Deferred Embedding Optimizer Stepping (`--embed-step-every`)**, **Optional `FP8` (`torch.float8_e4m3fn`) Linear Projection Wrapper (`FP8LinearWrapper` with automatic BF16 fallback)**, and **Optional Step CUDA-Graph Runner (`--cuda-graph-steps`)**.

2. **Problem 2 — Isotropic Orthogonal Initialization Blindness in `QuantumBornGPT` (and How 9-Shard `900M`-Token PPMI + Randomized SVD Solves It)**:
   - Currently, both `QuantumBornGPT` (`dm_seed`, `50,304 × 1024`) and `ClassicalModdedGPT` (`wte`, `50,304 × 768`) initialize all $50,304$ token embeddings i.i.d. from isotropic Gaussian noise $\mathcal{N}(0, I)$.
   - On the $1024$-dimensional unit sphere $S^{1023}$, two random unit vectors have expected inner product $0$ and standard deviation $1/\sqrt{1024} = 0.03125$ (and after projecting into a $d_h = 128$ head subspace, $1/\sqrt{128} \approx 0.0884$).
   - Consequently, at Step 0, **tokens that almost always co-occur in natural language (e.g., `" San"` and `" Francisco"`, `" quantum"` and `" mechanics"`, `" New"` and `" York"`) are just as orthogonal as completely unrelated tokens**.
   - In `QuantumBornGPT`, this is especially costly because `flash_purified_born_attention` suppresses inner products below the coherence threshold $\tau_0 = 0.0625$:
     $$A_{t,s}^{(h)} \propto \max\!\big(0,\, \langle q_t^{(h)} | k_s^{(h)} \rangle - \tau_0\big)^4$$
     With random orthogonal initialization, true linguistic collocations start near $\langle q_t | k_s \rangle \approx 0$, producing near-zero quartic Born signal-to-noise ratio (SNR) for the first several hundred steps of training, while rare tokens that appear only a few dozen times per shard barely rotate away from their random initial directions.
   - **Our Solution**: Before training begins (or loaded from a fast disk cache in `< 20 ms`), we stream up to **all 9 FineWeb training shards (`900M` tokens)** through a vectorized GPU **Shifted Positive Pointwise Mutual Information (SPPMI) + Randomized SVD** spectral sketch (`src/qdlblocks/layers/cooccurrence_init.py`). We initialize each token's state ket by blending an isotropic random identity vector $\hat{z}_w \in S^{d-1}$ with its unit-normalized $\Sigma^{1/2}$-weighted PPMI left singular vector $\tilde{c}_w \in S^{d-1}$:
     $$\psi_0(w) = \text{Normalize}\!\Big((1 - \alpha_{\text{ppmi}})\, \hat{z}_w + \alpha_{\text{ppmi}}\, \tilde{c}_w\Big)$$
     Crucially, when `--ortho-decouple` is active in `QuantumBornGPT`, we place the PPMI-SVD input state basis in the **even coordinate subspace** (`0, 2, 4, ...`) of `dm_seed` and the PPMI-SVD output measurement basis in the **odd coordinate subspace** (`1, 3, 5, ...`) of `povm_head.effects`. This guarantees **both**:
     1. **Zero Step-0 self-repetition bias**: $\langle m_w | \psi_0(w) \rangle = 0$ for every token $w \in [0, V)$, AND
     2. **Immediate high-SNR Born interference on co-occurring tokens**: frequently co-occurring pairs $(u, v)$ start with $\langle \psi_0(u) | \psi_0(v) \rangle \gg \tau_0$ from Step 0!

---

### 1.2 Summary of Steps Taken Across Waves 1–12
*(Note: All full-wave benchmarks in the table below report validation cross-entropy on the held-out real FineWeb-Edu `val_slice.bin` shard at `T_val = 512`. Pre-launch micro-verifications run 40–60 steps on synthetic contextual Markov-Zipf shards to validate relative deltas in `< 30s` before committing multi-hour Colab GPU runs.)*

| Wave / Plan | Key Architectural & Systems Changes | Best Classical Val CE (`T_val=512`) | Best QuantumBornGPT Val CE (`T_val=512`) | Key Empirical Outcome |
| :--- | :--- | :---: | :---: | :--- |
| **Waves 1–3** (Plans 001–010) | Single-shard `100M`-token multi-epoch training; `QuantumBornGPT` $d=1024$, $L=6$; `StiefelMuon` + `AdamW`; WSD schedule; Multi-Token Prediction (MTP). | — | `4.1999` | Discovered multi-epoch memorization ceiling on single shard (`0.84` nat generalization gap). |
| **Waves 4–5** (Plans 011–012) | `FineWebMultiShardStream` streaming 9 distinct `100M`-token shards (`900M` unique tokens); `TwoBeamInterferenceMLP`; `ContextualBlochPurityGate` (`--dynamic-povm-beta`). | — | `4.0846` | Eliminated multi-epoch memorization; `w5_900m_two_beam_seed` reached `4.0846` CE. |
| **Wave 6** (Plan 013) | Introduced matched `ClassicalModdedGPT` (`ReLU^2` MLP, QK-RMSNorm, RoPE, `logit_softcap=30.0`) and `flash_born_attention` Triton kernel. | `3.5880` | `4.0315` | Established apples-to-apples Classical vs. Quantum speedrun benchmark. |
| **Wave 7** (Plan 014) | Fixed `QuantumBornGPT` early-epoch bottleneck via `--ortho-decouple`, `--povm-log-prior`, `--gated-spherical-residual`, and `flash_purified_born_attention`. | `3.5352` | `3.8573` | Quantum Val CE improved by **`-0.1742` nats** (`4.0315 -> 3.8573`). |
| **Wave 8** (Plan 015) | True Layer-0 Value Residual (`--true-layer0-v-residual`), Zero-Init Residual/Head (`--zero-init-classical`), `--wsd-min-lr-ratio 0.0`. | `3.4658` | `3.7973` | Both families advanced (`3.4658` Classical, `3.7973` Quantum). |
| **Waves 9–10** (Plans 016–018) | GPU-staged shards, `direct_chunked_povm_ce`, 3-bank Value Embeddings, U-Net skips, zero-half RoPE, doc-boundary masking, SQLite/Model Cards. | `3.4280` | `3.6926` | `w10_900m_qborn_speedrun_v3_asym1024_frame4d_L8` broke the `3.70` barrier (**`3.6926` Val CE**). |
| **Wave 11** (Plan 019) | Implemented `--leaderboard-exact-classical` (`2025-01-16` parity: untied `N(0,1)` `wte`, `0.12` attn scale, `softcap=15`, `base=1024` RoPE, pre-block scalar U-Net skips, EMA Muon). | `3.7330` *(at `B_tok=524k`, `1716` steps)* | `3.6926` *(prior record)* | Confirmed `B_tok=524k` under-steps a single `900M` pass (`1,716` steps vs `3,433+`); 60-step synthetic micro-verification gained `-0.3720` nats (`3.2974 -> 2.9254` CE). |
| **Wave 12** (Plan 020, Active Now) | SQLite WAL telemetry DB (`benchmarks.db`), per-family 2D `(wall_min, best_val_ce)` Pareto-frontier auto-obsolete, 3-Tab WebUI, auto-VM shutdown; launched `B_tok=131k/262k` Classical & `H=8, d_head=128` Quantum V5. | `3.6433` *(in progress @ ~350M/900M tok)* | `4.1285` *(in progress @ ~200M/900M tok)* | Currently running on all 5 Colab GPUs; will auto-stop VMs on completion. |

---

### 1.3 Outcomes We Aim to Achieve in Plan 021
1. **Reusable Implementation of All Categories A, B, and C Mechanisms in `QuantumDLBlocks`**:
   - Every mechanism from our `KellerJordan/modded-nanogpt` audit (Record `#360` / `ANVIL2`) is implemented as a clean, unit-tested, reusable module in `src/qdlblocks/` and exported in `qdlblocks.__init__` so both `QuantumBornGPT` and `ClassicalModdedGPT` (as well as future architectures in this repository) can compose them via Python API or CLI flags.
2. **9-Shard (`900M`-Token) GPU PPMI + Randomized SVD Spectral Embedding Initialization (`src/qdlblocks/layers/cooccurrence_init.py`)**:
   - Capable of streaming `900M` tokens (all 9 FineWeb training shards) through a vectorized GPU co-occurrence sketch + Shifted PPMI + Randomized SVD in a few seconds (with automatic disk caching), giving co-occurring tokens high initial inner-product coherence while preserving exact input/output subspace orthogonality (`--ortho-decouple`) in `QuantumBornGPT`.
3. **Rigorous Pre-Launch Verification & Wave-13 Queue Staging**:
   - Verify all new primitives via unit tests (`tests/test_hypothesis_021_ppmi_svd_ngram_and_anvil.py`) and controlled multi-step empirical experiments (`examples/31_verify_plan021_ppmi_svd_ngram_and_anvil.py`), log all micro-verification runs into the SQLite `benchmarks.db` (`🧪 Pre-Launch Micro-Verifications` tab), and register `configs/wave13_fleet.json` as `QUEUED` in `session_queue` without interrupting the 5 running Wave-12 Colab jobs.

---

## 2. Observable Hypotheses & Quantitative Pass/Fail Criteria

### Hypothesis H1 — 9-Shard PPMI + Randomized SVD Co-Occurrence Initialization + Hashed Bigram/Trigram Lookup on `QuantumBornGPT`
- **Mechanism**:
  1. Computing a Shifted PPMI + Randomized SVD basis $\tilde{c}_w \in S^{d-1}$ over multi-shard token streams and initializing `QuantumBornGPT` via orthogonal-subspace-decoupled blending:
     $$\psi_0^{\text{even}}(w) = \text{Normalize}\!\Big((1 - \alpha_{\text{ppmi}})\, \hat{z}_w^{\text{even}} + \alpha_{\text{ppmi}}\, \tilde{c}_w^{\text{even}}\Big), \qquad m_w^{\text{odd}} = \text{Normalize}\!\Big((1 - \alpha_{\text{ppmi}})\, \hat{z}_w^{\text{odd}} + \alpha_{\text{ppmi}}\, \tilde{c}_w^{\text{odd}}\Big)$$
     endows frequently co-occurring token pairs $(u, v)$ with positive inner-product coherence $\langle q_u | k_v \rangle > \tau_0$ at Step 0 while maintaining $\langle m_w | \psi_0(w) \rangle = 0$.
  2. Augmenting the backbone with `HashedNgramEmbedding` (hashed bigram + trigram lookup with the deterministic $\pm 1$ Weinberger sign trick) provides immediate local 2-gram and 3-gram context at Layer 0 and middle layers.
- **Quantitative Predictions**:
  1. **Throughput**: Accumulating the co-occurrence sketch across `900M` tokens (`9 × 100M` `uint16` shards) and computing the `SPPMI + Randomized SVD` embedding basis on GPU completes in **`< 15.0` seconds** (and loading from the `.pt` cache completes in **`< 0.2` seconds**).
  2. **Step-0 Zero Self-Repetition Preservation**: With `use_ortho_decouple=True`, the maximum self-overlap across all $V$ tokens satisfies $\max_w |\langle m_w | \psi_0(w) \rangle| < 10^{-6}$.
  3. **Step-0 Collocation Coherence & Born Attention SNR**: On top co-occurring token pairs vs. random token pairs, the mean Step-0 state cosine similarity on collocations is **`>= 3.0x`** higher than on random pairs, and mean Step-0 purified Born attention weight on collocated context tokens increases by **`>= 50%`** relative to isotropic Gaussian initialization.
  4. **Validation Cross-Entropy**: In a controlled 45-step multi-shard training comparison on `QuantumBornGPT`, enabling **PPMI-SVD Init + `HashedNgramEmbedding`** achieves a strictly lower final validation CE than isotropic initialization by **`>= 0.05` nats**.

### Hypothesis H2 — Category A Sample-Efficiency Primitives (`PolarExpress`/`ANVIL` + `NorMuon` + Twin-Rail + Cautious WD + `MUDD`/Gates + `TailWeightAverager`)
- **Mechanism**:
  - Combining the 6-step `ANVIL_MAPS` (`zeropower_via_polarexpress5`) + `NorMuon` per-lane variance equalizer + twin-rail Nesterov momentum + cautious weight decay with `InputChannelGate` (`ve_gate`, `attn_gate`, `GatedXSA`), `MUDDDynamicSkip`, and `TailWeightAverager` (with Frobenius norm decontraction) improves both per-step spectral conditioning and final validation generalization.
- **Quantitative Predictions**:
  1. **Spectral Convergence**: On random rectangular matrices $G \in \mathbb{R}^{M \times N}$ with condition number $\kappa(G) \ge 50$, 6-step `zeropower_via_polarexpress5` drives the singular value root-mean-square error $\text{RMSE}(\sigma(U), 1)$ to **`< 0.08`**, strictly lower than 5-step fixed Newton-Schulz (`zeropower_via_newtonschulz5`).
  2. **`NorMuon` Norm Preservation**: `normuon_equalize` reduces row-wise update energy variance across rows while preserving the matrix Frobenius norm within relative error **`< 10^{-5}`**.
  3. **Validation Cross-Entropy & Tail Averaging**: In a controlled 45-step training benchmark, enabling the Category A bundle (`PolarExpress` + `NorMuon` + `TwinRail` + `CautiousWD` + `MUDD`/Gates + `TailWeightAverager`) achieves a lower validation CE than the un-averaged baseline, and evaluating with `TailWeightAverager.average_parameters(model)` yields a strictly lower or equal validation CE (`val_ce_ema <= val_ce_raw`).

### Hypothesis H3 — Category C Throughput & Systems Primitives (`SampledSoftmax`, `FP8LinearWrapper`, Every-$K$-Step Embeddings, & `--align-to-bos`)
- **Mechanism**:
  - `sampled_softcap_linear_ce` and `sampled_povm_ce` compute exact cross-entropy and backward gradients over a deterministic $P$-candidate subset ($P \ll V$) containing 100% of the microbatch's ground-truth targets plus coprime-stride negatives (`k * 20011 mod V`), shrinking the vocabulary GEMM from $V = 50,304$ to $P$ during early training stages while converging to the exact full-vocabulary loss when $P = V$.
- **Quantitative Predictions**:
  1. **Target Coverage & Full-Vocab Exactness**: `build_sampled_candidate_set` includes `100%` of unique valid targets in the microbatch (`(remapped_targets >= 0).all()`), and when $P = V$, `sampled_softcap_linear_ce` and `sampled_povm_ce` match `direct_chunked_softcap_linear_ce` and `direct_chunked_povm_ce` within **`< 10^{-5}`** in loss and gradient norms.
  2. **Throughput Speedup**: At $V = 50,304$ and $P = 8,192$, `sampled_softcap_linear_ce` and `sampled_povm_ce` execute forward+backward faster than full-vocabulary `50,304`-way evaluation.
  3. **BoS Alignment**: `find_bos_aligned_starts` aligns `100%` of sequence starts to positions immediately following `<|endoftext|>` (`50256`) (or index `0`) whenever sufficient document boundaries exist in the shard.

---

## 3. Operational & Scripting Triage (`Build-Before-Launch` vs. `Nice-to-Have`)

### 3.1 Operational CLI Commands Used in Plan 021
1. **Model & Parameter Bank Inspection (`scripts/inspect_model.py`)**:
   - `.venv/bin/python scripts/inspect_model.py` — Inspects parameter counts, Stiefel orthogonality errors $\|W^\top W - I\|_F$, and state norm preservation across `QuantumBornGPT` and `ClassicalModdedGPT`.
2. **Pre-Launch Hypothesis Verification & SQLite Ingestion (`examples/31_verify_plan021_ppmi_svd_ngram_and_anvil.py`)**:
   - `.venv/bin/python examples/31_verify_plan021_ppmi_svd_ngram_and_anvil.py` — Executes `H1` (`900M`-token PPMI-SVD throughput benchmark + 45-step `QuantumBornGPT` ablation), `H2` (Category A `ANVIL`/`NorMuon`/`TailEMA`/`MUDD` verification), and `H3` (Category C `SampledSoftmax`/`FP8LinearWrapper`/`--align-to-bos` verification), ingests `verify021_*` micro-runs into `.cache/benchmarks.db`, and stages `configs/wave13_fleet.json` in `session_queue`.
3. **Fleet Registration & Live Monitoring (`scripts/colab_fleet.py` & `scripts/monitor_colab_runs.py`)**:
   - `.venv/bin/python scripts/colab_fleet.py register-cards --manifest configs/wave13_fleet.json` — Registers declarative Model Cards for all 5 Wave-13 runs and triggers `POST http://127.0.0.1:8765/api/reload` with zero server downtime.
   - `.venv/bin/python scripts/monitor_colab_runs.py --once` — Prints a one-shot terminal leaderboard table and Colab fleet status snapshot from `.cache/benchmarks.db`.

### 3.2 `Build-Before-Launch` vs. `Nice-to-Have` Classification
| Item | Classification | Rationale |
| :--- | :---: | :--- |
| `src/qdlblocks/layers/cooccurrence_init.py` (9-shard GPU PPMI + Randomized SVD sketch & ortho-decoupled initializer) | **Build-Before-Launch** | Required for Hypothesis H1 and Wave-13 9-shard PPMI-SVD runs. |
| `src/qdlblocks/layers/ngram_hash.py` (`HashedNgramEmbedding` with Weinberger $\pm 1$ sign pool) | **Build-Before-Launch** | Required for Category B and Hypothesis H1. |
| `src/qdlblocks/optim/tail_ema.py` (`TailWeightAverager` with Frobenius/spherical decontraction) | **Build-Before-Launch** | Required for Category A and Hypothesis H2. |
| `src/qdlblocks/functional.py` & `src/qdlblocks/optim/muon.py` (`zeropower_via_polarexpress5`, `normuon_equalize`, Twin-Rail momentum, Cautious WD) | **Build-Before-Launch** | Required for Category A and Hypothesis H2. |
| `src/qdlblocks/layers/mudd_and_gates.py` (`InputChannelGate`, `apply_gated_xsa`, `MUDDDynamicSkip`, `ForwardSmearGate`) | **Build-Before-Launch** | Required for Category A and Hypothesis H2. |
| `src/qdlblocks/layers/isometric.py` & `src/qdlblocks/layers/attention.py` (`YaRNRotarySchedule`, `apply_partial_key_offset`, `paired_heads`, `qk_head_dim`) | **Build-Before-Launch** | Required for Category A parity. |
| `src/qdlblocks/layers/direct_chunked_ce.py` & `src/qdlblocks/layers/fp8_linear.py` (`sampled_softcap_linear_ce`, `sampled_povm_ce`, `FP8LinearWrapper`) | **Build-Before-Launch** | Required for Category C and Hypothesis H3. |
| `scripts/precompute_ppmi_svd_cache.py --shards 1..9 --embed-dims 768,1024 --buckets 2048` (Standalone offline CLI tool to prebuild & upload `.pt` PPMI-SVD caches to HuggingFace/GCS) | **Nice-to-Have** (Log in `scripts/BACKLOG.md`) | `compute_ppmi_svd_embeddings` already computes and caches the 9-shard sketch on GPU in `< 15s` at startup, so a separate cloud-upload CLI is a convenience optimization for future multi-host fleets. |
| Multi-node `8xH100` NCCL all-to-all distributed `NgramTable` sharding (`--distributed-ngram-alltoall` for `130 GB` 84.6M-row tables across 8 ranks) | **Nice-to-Have** (Log in `scripts/BACKLOG.md`) | Our Colab fleet runs single-GPU instances (`1x A100` / `1x RTX PRO 6000`), where a compact GPU-resident `HashedNgramEmbedding` (`262,144` to `1,048,576` rows) fits directly in VRAM with zero network overhead. |
| `scripts/colab_fleet.py auto-drain-queue --poll-seconds 60` (Autonomous queue dispatcher that launches the next `QUEUED` wave when a Colab VM finishes and stops) | **Nice-to-Have** (Log in `scripts/BACKLOG.md`) | Saves manual wave launch invocation between waves while preserving user control over compute unit spend. |

---

## 4. Detailed Architecture & Code Specification

### 4.1 Component 1: 9-Shard (`900M`-Token) GPU PPMI + Randomized SVD Spectral Initialization & Hashed Bigram/Trigram Lookup

#### 4.1.1 `src/qdlblocks/layers/cooccurrence_init.py`
1. **`compute_ppmi_svd_embeddings`**:
   ```python
   def compute_ppmi_svd_embeddings(
       token_arrays: Iterable[np.ndarray | torch.Tensor],
       vocab_size: int = 50304,
       embed_dim: int = 1024,
       num_context_buckets: int = 2048,
       window_size: int = 4,
       cds_alpha: float = 0.75,
       shift_kappa: float = 1.0,
       svd_oversample: int = 32,
       svd_niter: int = 4,
       chunk_tokens: int = 25_000_000,
       device: torch.device | str | None = None,
       cache_path: Path | str | None = None,
       seed: int = 42,
   ) -> dict[str, Any]:
   ```
   - **Exact Single-Pass Algorithm**:
     1. **Disk Cache Check**: If `cache_path` is provided and exists on disk, load via `torch.load(cache_path, map_location=dev, weights_only=True)` and return the cached dictionary (`embeddings: (vocab_size, embed_dim)` `float32` unit-normalized spectral basis, `singular_values: (k_svd,)`, `unigram_counts: (vocab_size,)`, `bucket_map: (vocab_size,)`, `total_tokens: int`, `elapsed_s: float`).
     2. **Deterministic Context Bucket Mapping (`bucket_map: (vocab_size,)` `int64`)**:
        - Let $B = \min(\text{vocab\_size}, \text{num\_context\_buckets})$.
        - **Case A (`vocab_size <= num_context_buckets`)**: Every token gets its own collision-free context column: `B = vocab_size` and `bucket_map = torch.arange(vocab_size, device=dev, dtype=torch.int64)`.
        - **Case B (`vocab_size > num_context_buckets`)**: On the first chunk (`sample_slice = first_shard[:min(5_000_000, len(first_shard))]`), compute unigram counts via `torch.bincount(sample_slice, minlength=vocab_size)`. Select the top $B_{\text{exact}} = B // 2$ most frequent token IDs using deterministic stable sorting (`torch.argsort(-sample_counts, stable=True)[:B_exact]`) and assign them dedicated columns `0 .. B_exact - 1`. Assign all remaining token IDs $v \in [0, \text{vocab\_size})$ deterministically via multiplicative hashing into the second half of the columns:
          $$\text{bucket\_map}[v] = B_{\text{exact}} + \big((v \cdot 36313) \bmod (B - B_{\text{exact}})\big)$$
     3. **Vectorized Single-Pass Co-Occurrence Accumulation**:
        - Allocate `counts_flat = torch.zeros(vocab_size * B, dtype=torch.float32, device=dev)` and `unigram_counts = torch.zeros(vocab_size, dtype=torch.float64, device=dev)`.
        - Stream each shard in `token_arrays` in slices of `chunk_tokens` (`25M` tokens) onto `dev` (`cuda` if available else `cpu`), clamping token IDs to `[0, vocab_size - 1]`.
        - Update `unigram_counts.add_(torch.bincount(chunk_i64, minlength=vocab_size).to(torch.float64))`.
        - Compute `ctx_b = bucket_map[chunk_i64]`. For each offset $\delta \in \{1, \dots, \text{window\_size}\}$ with distance weight $w_\delta = 1.0 / \delta$:
          - Forward pairs $(x_t, b(x_{t+\delta}))$: `flat_fwd = chunk_i64[:-delta] * B + ctx_b[delta:]`
          - Backward pairs $(x_{t+\delta}, b(x_t))$: `flat_bwd = chunk_i64[delta:] * B + ctx_b[:-delta]`
          - Accumulate `counts_flat.add_(torch.bincount(flat_fwd, minlength=vocab_size * B).to(torch.float32), alpha=w_delta)` and `counts_flat.add_(torch.bincount(flat_bwd, minlength=vocab_size * B).to(torch.float32), alpha=w_delta)`.
     4. **Shifted Positive Pointwise Mutual Information (SPPMI) Transform**:
        - Reshape `C = counts_flat.view(vocab_size, B)` (`C[w, b]` is the total distance-weighted co-occurrence count of token $w$ with context bucket $b$).
        - Compute row marginal sums $r_w = \sum_{b=0}^{B-1} C_{w,b}$ (`(vocab_size, 1)`) and column marginal sums $c_b = \sum_{w=0}^{V-1} C_{w,b}$ (`(1, B)`).
        - Compute context-distribution-smoothed background probability $p_{\alpha}(b)$ ($\sum_{b=0}^{B-1} p_{\alpha}(b) = 1$, default $\alpha_{\text{cds}} = 0.75$):
          $$p_{\alpha}(b) = \frac{c_b^{\alpha_{\text{cds}}}}{\sum_{b'=0}^{B-1} c_{b'}^{\alpha_{\text{cds}}} + 10^{-12}}$$
        - Under statistical independence between target word $w$ and context bucket $b$, the expected co-occurrence count is $\mathbb{E}[C_{w,b}] = r_w \, p_{\alpha}(b)$. With multiplicative shift parameter $\kappa_{\text{shift}} = \max(\text{shift\_kappa}, 10^{-6})$ (default `1.0`; setting $\kappa_{\text{shift}} = k$ subtracts $\log k$ from PMI as in Levy & Goldberg 2014):
          $$\text{SPPMI}_{w,b} = \max\!\left(0.0,\, \log\!\left(\frac{\max(C_{w,b},\, 10^{-12})}{\max(r_w,\, 10^{-12}) \, p_{\alpha}(b) \, \kappa_{\text{shift}} + 10^{-12}}\right)\right)$$
          and explicitly zero out any row $w$ where $r_w == 0$ (`SPPMI[r_w.squeeze(-1) <= 0] = 0.0`).
     5. **Deterministic GPU Randomized SVD (`torch.svd_lowrank`)**:
        - Set rank $k_{\text{svd}} = \min(\text{embed\_dim}, V, B)$ and oversampled subspace size $q = \min(V, B, k_{\text{svd}} + \text{svd\_oversample})$.
        - Seed a local generator or `torch.manual_seed(seed)` before calling `U, S, V_svd = torch.svd_lowrank(SPPMI, q=q, niter=svd_niter)`.
        - **Deterministic Sign Convention**: For each column $j \in \{0, \dots, k_{\text{svd}} - 1\}$, find $i^*_j = \arg\max_{i} |U_{i, j}|$ and multiply column $U_{:, j}$ by $\text{sign}(U_{i^*_j, j})$ (using $+1$ if $U_{i^*_j, j} \ge 0$ else $-1$) so singular vectors are sign-deterministic across hardware and runs.
        - Weight by $\Sigma^{1/2}$: `U_k = U[:, :k_svd] * torch.sqrt(S[:k_svd].clamp_min(1e-12)).unsqueeze(0)`.
        - If $k_{\text{svd}} < \text{embed\_dim}$, right-pad `U_k` with zeros to shape `(vocab_size, embed_dim)`.
        - Normalize active rows ($r_w > 0$) to unit $L_2$ norm (`l2_normalize(U_k, dim=-1)`); set unseen/padding rows ($r_w == 0$, such as padded token IDs `50,257..50,303`) strictly to `0.0`.
     6. Save atomically to `cache_path` (via `.tmp` rename) if specified and return the result dictionary.

2. **`initialize_quantum_born_with_ppmi_svd`**:
   ```python
   @torch.no_grad()
   def initialize_quantum_born_with_ppmi_svd(
       model: nn.Module,
       ppmi_embeddings: torch.Tensor,
       alpha: float = 0.65,
       seed: int = 42,
   ) -> dict[str, float]:
   ```
   - Takes `ppmi_embeddings` of shape `(V_src, D_src)` and slices/pads to `(V, d) = (model.vocab_size, model.embed_dim)`.
   - Identifies active rows `active_mask = (ppmi_embeddings.norm(dim=-1) > 1e-6)` (`(V,)` boolean). For any unseen/padding row where `active_mask[w]` is `False`, $\tilde{c}_w = 0$, so blending $(1 - \alpha)\hat{z}_w + \alpha \tilde{c}_w$ followed by `l2_normalize` automatically yields the unit-norm isotropic vector $\hat{z}_w \in S^{d-1}$ with zero NaNs or zero-norm rows.
   - **When `getattr(model, "use_ortho_decouple", False)` is `True`**:
     - Let $d_{\text{half}} = d // 2$. Extract the leading $d_{\text{half}}$ spectral columns `c_half = l2_normalize(ppmi_embeddings[:, :d_half], dim=-1)` (with `c_half[~active_mask] = 0.0`).
     - Draw independent Gaussian matrices `z_even ~ N(0, I_{d_half})` and `z_odd ~ N(0, I_{d_half})` using a `torch.Generator` seeded with `seed`, and unit-normalize each row (`z_even = l2_normalize(z_even, dim=-1)`, `z_odd = l2_normalize(z_odd, dim=-1)`).
     - Construct blended half-space unit vectors:
       $$u_{\text{even}} = \text{l2\_normalize}\!\big((1.0 - \alpha)\, z_{\text{even}} + \alpha\, c_{\text{half}},\; \text{dim}=-1\big) \in S^{d_{\text{half}}-1}$$
       $$u_{\text{odd}} = \text{l2\_normalize}\!\big((1.0 - \alpha)\, z_{\text{odd}} + \alpha\, c_{\text{half}},\; \text{dim}=-1\big) \in S^{d_{\text{half}}-1}$$
     - Write `model.dm_seed.weight` (`(V, d)`):
       - `model.dm_seed.weight.zero_()`
       - `model.dm_seed.weight[:, 0::2].copy_(u_even.to(model.dm_seed.weight.dtype))`
     - Write `model.povm_head.effects` (when untied `nn.Parameter` of shape `(V, d)`):
       - `model.povm_head.effects.zero_()`
       - `model.povm_head.effects[:, 1::2].copy_(u_odd.to(model.povm_head.effects.dtype))`
     - Because `dm_seed.weight` has non-zero entries only on even coordinates `0::2` and `povm_head.effects` has non-zero entries only on odd coordinates `1::2`, every token $w \in [0, V)$ satisfies $\langle m_w | \psi_0(w) \rangle = 0$ identically!
   - **When `getattr(model, "use_ortho_decouple", False)` is `False`**:
     - Blends `u_full = l2_normalize((1.0 - alpha) * z_full + alpha * c_full, dim=-1)` across all $d$ coordinates and writes `model.dm_seed.weight.copy_(u_full)`.
   - Also updates `model.wte.weight` (`(V, seed_dim)`) using the leading `seed_dim` spectral components and initializes `model.value_embeds` (if `model.value_embeds is not None`) from `model.dm_seed.weight`.
   - Returns diagnostic metrics `{"max_self_overlap": float, "active_vocab_fraction": float, "alpha": float}`.

3. **`initialize_classical_with_ppmi_svd`**:
   ```python
   @torch.no_grad()
   def initialize_classical_with_ppmi_svd(
       model: nn.Module,
       ppmi_embeddings: torch.Tensor,
       alpha: float = 0.50,
       seed: int = 42,
   ) -> dict[str, float]:
   ```
   - Preserves the existing per-row standard deviation $\sigma_{\text{row}}$ of `model.wte.weight` (`1.0` in `--leaderboard-exact-classical`, or `0.02` in standard mode):
     - Scales `c_scaled = c_full * math.sqrt(d)` (so each coordinate of `c_scaled` has unit variance `1.0` on active tokens), blends `w_new = (math.sqrt(max(0.0, 1.0 - alpha**2)) * z_std + alpha * c_scaled) * target_std`, and writes `model.wte.weight.copy_(w_new)`.

4. **`compute_collocation_coherence_metrics`**:
   - Given a token sequence `tokens_1d` and embedding matrix `embeddings: (V, d)`, extracts the top $K = 256$ most frequent adjacent collocated pairs $(u, v)$ ($u \neq v$) and $K = 256$ random token pairs, and returns:
     - `colloc_mean_cos`, `random_mean_cos`, `cos_ratio = colloc_mean_cos / max(abs(random_mean_cos), 1e-6)`, `colloc_purified_born_energy`, and `random_purified_born_energy`.

---

#### 4.1.2 `src/qdlblocks/layers/ngram_hash.py` (Category B: Hashed Bigram + Trigram Embedding with Weinberger $\pm 1$ Sign Trick)
- Implements the exact Record `#360` multiplicative-XOR bigram and trigram hashing (`track_1_short/ngram_table.py`):
  - `BIGRAM_ROW_MULS = (36313, 27191)`
  - `TRIGRAM_ROW_MULS = (17351, 60961, 45259)`
  - `BIGRAM_SIGN_MULS = (48271, 30011)`
  - `TRIGRAM_SIGN_MULS = (58699, 39779, 26801)`
- **`compute_ngram_hashes_and_signs(idx: torch.Tensor, ngram_vocab_size: int = 262144, sign_pool_rows: int = 8192)`**:
  - Accepts `idx: (B, T)` integer token IDs on CPU or GPU.
  - Let `half = ngram_vocab_size // 2`, `bigram_mod = half - 1`, `trigram_mod = ngram_vocab_size - half - 1`.
  - Computes in 64-bit integer arithmetic before modulo/masking:
    - `bigram_ids`: shape `(B, T)` in `[0, half)`:
      - `bigram_ids[:, 0] = bigram_mod`
      - For $t \ge 1$, `prev_b` uses `bigram_mod` at $t = 1$ and `idx[:, t-1]` for $t \ge 2$ (matching `ngram_row_ids` in `track_1_short/ngram_table.py`):
        `bigram_ids[:, t] = ((36313 * idx[:, t]) ^ (27191 * prev_b)) % bigram_mod`
    - `trigram_ids`: shape `(B, T)` in `[half, ngram_vocab_size)`:
      - `trigram_ids[:, :2] = half + trigram_mod`
      - For $t \ge 2$: `trigram_ids[:, t] = (((17351 * idx[:, t]) ^ (60961 * idx[:, t-1]) ^ (45259 * idx[:, t-2])) % trigram_mod) + half`
    - `bigram_sign_ids`: shape `(B, T)` in `[0, sign_pool_rows)`:
      - `0` at $t = 0$, else `((48271 * idx[:, t]) ^ (30011 * idx[:, t-1])) & (sign_pool_rows - 1)`
    - `trigram_sign_ids`: shape `(B, T)` in `[0, sign_pool_rows)`:
      - `0` at $t < 2$, else `(((58699 * idx[:, t]) ^ (39779 * idx[:, t-1]) ^ (26801 * idx[:, t-2])) & (sign_pool_rows - 1))`
- **`HashedNgramEmbedding(nn.Module)`**:
  - Parameters & Buffers:
    - `self.table = nn.Embedding(ngram_vocab_size, embed_dim)`: initialized to `zeros_` by default (`zero_init: bool = True`).
    - `self.register_buffer("sign_pool", sign_pool, persistent=False)`: deterministic $\pm 1$ tensor of shape `(sign_pool_rows, embed_dim)` generated with a local `torch.Generator` seeded with `1337`.
    - `self.layer_scales = nn.Parameter(torch.full((num_layers,), float(inject_scale_init), dtype=torch.float32))` (default `inject_scale_init = 0.05`).
  - **Forward & Exact Injection Equation**:
    - `h_ngram = F.embedding(bigram_ids, self.table.weight) * self.sign_pool[bigram_sign_ids].to(dtype) + F.embedding(trigram_ids, self.table.weight) * self.sign_pool[trigram_sign_ids].to(dtype)` of shape `(B, T, embed_dim)`.
    - **In `QuantumBornGPT`**:
      - At initial embedding construction (`embed_Descriptive` / `forward_features`): when `self.ngram_embed is not None`, computes `h_ngram = self.ngram_embed(idx)` (`(B, T, d)`) and injects it into the initial state ket before spherical normalization:
        $$\psi_0 = \text{l2\_normalize}\!\big(\psi_{\text{seed}} + s_0 \, h_{\text{ngram}},\; \text{dim}=-1\big), \qquad s_0 = \text{layer\_scales}[0]$$
      - Because `self.table.weight` is initialized to `0.0` at Step 0 (`zero_init=True`), $h_{\text{ngram}} \equiv 0$ at Step 0, which guarantees $\psi_0 = \psi_{\text{seed}}$ and **strictly preserves $\max_w |\langle m_w | \psi_0(w) \rangle| < 10^{-6}$ at Step 0**!
    - **In `ClassicalModdedGPT`**:
      - At initial embedding construction in `forward_features`: adds `x = x + self.ngram_embed.layer_scales[0].to(x.dtype) * h_ngram` prior to `x = F.rms_norm(x, (self.embed_dim,), eps=self.eps)`.
    - **Optimizer Routing**:
      - `ngram_embed.table.weight` is routed to the `AdamW` embedding parameter group (same LR as `wte` / `dm_seed`), and `ngram_embed.layer_scales` is routed to the `AdamW` scalar parameter group. Both automatically participate in `--embed-step-every K` when $K > 1$.

---

### 4.2 Component 2: Category A Optimizer & Tail-Averaging Primitives (`src/qdlblocks/optim/tail_ema.py`, `src/qdlblocks/functional.py`, `src/qdlblocks/optim/muon.py`)

#### 4.2.1 `src/qdlblocks/optim/tail_ema.py` (`TailWeightAverager`)
- Implements `KellerJordan/modded-nanogpt` `track_1_short/tail_average.py`:
  ```python
  class TailWeightAverager:
      """Late-training exponential moving average (EMA) with Frobenius/spherical decontraction."""
      def __init__(
          self,
          model: nn.Module,
          start_frac: float = 0.75,
          ema_rate: float = 4.0 / 53.0,
          blend_alpha: float = 0.65,
          update_every: int = 1,
          decontract_frobenius: bool = True,
          spherical_param_substrings: tuple[str, ...] = ("dm_seed", "effects"),
      ) -> None:
  ```
  - **`update(step: int, total_steps: int) -> bool`**:
    - Active when `step >= int(start_frac * total_steps)` and `(step % update_every == 0 or step == total_steps)`.
    - On first active tick, seeds `self.shadow[name] = param.detach().float().clone()` for all `requires_grad` parameters.
    - On subsequent ticks, updates `self.shadow[name].lerp_(param.detach().float(), ema_rate)`.
  - **`@contextlib.contextmanager average_parameters(model: nn.Module | None = None)`**:
    - If `not self.is_seeded`, yields immediately (no-op).
    - Otherwise, clones current parameter data `orig_data[name] = param.data.clone()`, computes blended weights:
      $$W_{\text{blend}} = (1.0 - \alpha_{\text{blend}})\, W_{\text{cur}}^{\text{fp32}} + \alpha_{\text{blend}}\, W_{\text{ema}}^{\text{fp32}}$$
    - **Decontraction Rule**:
      - If `any(sub in name for sub in spherical_param_substrings)` and `W_blend.ndim == 2`:
        - Projects each row back onto the unit sphere $S^{d-1}$ (`W_ship = l2_normalize(W_blend, dim=-1)`).
      - Else if `decontract_frobenius and W_blend.ndim >= 2`:
        - Rescales $W_{\text{blend}}$ to match the live parameter's Frobenius norm $\|W_{\text{cur}}\|_F$:
          $$W_{\text{ship}} = W_{\text{blend}} \cdot \frac{\|W_{\text{cur}}^{\text{fp32}}\|_F}{\|W_{\text{blend}}\|_F + 10^{-12}}$$
      - Else (1D scalars/biases): `W_ship = W_blend`.
    - Copies `W_ship.to(param.dtype)` into `param.data`, yields for validation evaluation, and restores `orig_data[name]` in a `finally:` block.

#### 4.2.2 `src/qdlblocks/functional.py` & `src/qdlblocks/optim/muon.py` (`PolarExpress`/`ANVIL`, `NorMuon`, Twin-Rail, Cautious WD)
1. **In `src/qdlblocks/functional.py`**:
   - Define `ANVIL_QUINTIC_MAPS`:
     ```python
     ANVIL_QUINTIC_MAPS: tuple[tuple[float, float, float], ...] = (
         (3.923798038567, -6.095026865488, 3.905234618423),
         (3.278126713798, -3.328923386476, 0.989127286973),
         (3.505298394150, -5.137358782410, 1.968325560615),
         (2.815058591845, -3.685181239622, 1.417196497642),
         (2.245503932403, -2.443826979899, 0.963091710461),
         (2.256537145403, -2.166840097229, 0.929501253245),
     )
     ```
   - **Explicit `zeropower_via_polarexpress5` and `zeropower_via_polarexpress5_batched` Recurrence**:
     - Given 2D `(M, N)` or 3D `(K, M, N)` tensor $G$:
       1. Cast working copy $X = G.\text{bfloat16}()$ if $G.\text{dtype} == \text{torch.bfloat16}$ else $G.\text{float}()$.
       2. Normalize by the Frobenius margin (`frobenius_margin = 1.05`, `eps = 1e-6`):
          $$X \leftarrow \frac{X}{\text{frobenius\_margin} \cdot \|X\|_F + \epsilon}$$
       3. If $X.\text{size}(-2) > X.\text{size}(-1)$ (tall matrix), transpose $X \leftarrow X.\text{mT}$ and set `transposed = True` so $A = X X^\top$ always has the smaller dimension $\min(M, N) \times \min(M, N)$.
       4. For step $t \in \{0, \dots, \text{steps} - 1\}$ with coefficients $(a_t, b_t, c_t) = \text{ANVIL\_QUINTIC\_MAPS}[\min(t, 5)]$:
          $$A = X X^\top, \qquad B = b_t A + c_t (A A), \qquad X \leftarrow a_t X + B X$$
       5. If `transposed`, transpose back $X \leftarrow X.\text{mT}$ and return `X.type_as(G)`.
   - **Explicit `normuon_equalize` (`NorMuon` Per-Lane Energy Equalizer)**:
     ```python
     def normuon_equalize(
         U: torch.Tensor,
         lane_energy: torch.Tensor | None = None,
         beta2: float = 0.95,
         eps: float = 1e-10,
     ) -> tuple[torch.Tensor, torch.Tensor]:
     ```
     - Let `red_dim = -2 if U.size(-2) >= U.size(-1) else -1` (reduces along the longer matrix dimension so `lane_power` has shape `(..., 1, N)` when $M \ge N$ or `(..., M, 1)` when $M < N$).
     - Compute `lane_power = U.float().square().mean(dim=red_dim, keepdim=True)` and `lane_len = float(U.size(red_dim))`.
     - Compute pre-equalization Frobenius norm: `pre_norm = (lane_power.sum(dim=(-2, -1), keepdim=True) * lane_len).sqrt()`.
     - **State Initialization & Update**: If `lane_energy is None` or `lane_energy.shape != lane_power.shape`, initialize `lane_energy = lane_power.clone()`; otherwise update in place `lane_energy.lerp_(lane_power.to(lane_energy.dtype), 1.0 - beta2)`.
     - Compute inverse-RMS gain `lane_gain = lane_energy.float().clamp_min(eps).rsqrt()`.
     - Compute post-gain norm: `post_norm = ((lane_power * lane_len) * lane_gain.square()).sum(dim=(-2, -1), keepdim=True).sqrt()`.
     - Rescale: `eq_scale = lane_gain * (pre_norm / post_norm.clamp_min(eps))`, and return `(U * eq_scale.type_as(U), lane_energy)`.

2. **In `src/qdlblocks/optim/muon.py` (`StiefelMuon`)**:
   - Add parameters to `StiefelMuon.__init__`:
     - `polar_method: Literal["newton_schulz", "polar_express"] = "newton_schulz"`
     - `use_normuon: bool = False`, `normuon_beta2: float = 0.95`
     - `use_twin_rail: bool = False`, `rail_fast_beta: float = 0.85`, `rail_slow_beta: float = 0.98`, `rail_fast_weight: float = 0.4385`, `rail_engage_step: int = 64`
   - **Exact Twin-Rail Momentum Equations (`use_twin_rail=True`)**:
     - Maintain `fast_buf = state["momentum_buffer"]` (initialized to zeros) and `slow_buf = state["slow_momentum_buffer"]` (initialized to zeros).
     - On each step (`step_num >= 1`):
       $$\text{fast\_buf} \leftarrow \text{lerp}(\text{fast\_buf},\, g,\, 1 - \beta_{\text{fast}}), \qquad \text{slow\_buf} \leftarrow \text{lerp}(\text{slow\_buf},\, g,\, 1 - \beta_{\text{slow}})$$
       $$w_{\text{rail}} = \begin{cases} 1.0 & \text{if } \text{step\_num} < \text{rail\_engage\_step} \\ \text{rail\_fast\_weight} & \text{if } \text{step\_num} \ge \text{rail\_engage\_step} \end{cases}$$
       $$\text{blend} = w_{\text{rail}}\, \text{fast\_buf} + (1 - w_{\text{rail}})\, \text{slow\_buf}$$
       $$g_{\text{in}} = \begin{cases} \text{lerp}(g,\, \text{blend},\, m_{\text{eff}}) & \text{if nesterov} \\ \text{blend} & \text{otherwise} \end{cases}$$
   - **Exact Polar Orthogonalization + `NorMuon` Equalization + Sign-Aligned Cautious Weight Decay**:
     - Compute polar factor $U = \text{zeropower\_via\_polarexpress5}(g_{\text{in}}, \text{steps}=\text{ns\_steps})$ when `polar_method == "polar_express"` else `zeropower_via_newtonschulz5(g_in, steps=ns_steps)`.
     - If `use_normuon=True`, equalize $U, \text{state}[\text{"lane\_energy"}] = \text{normuon\_equalize}(U, \text{state.get}(\text{"lane\_energy"}), \beta_2=\text{normuon\_beta2})$.
     - Apply aspect scaling `update = U * sqrt(max(1.0, M / N))` if `aspect_scale=True` else `U`.
     - If `wd > 0.0` and `decay_mode == "cautious"`:
       $$\text{cwd\_mask} = (p \odot \text{update}) > 0, \qquad p \leftarrow p - (\text{lr} \cdot \text{wd}) \, (p \odot \text{cwd\_mask})$$
     - Apply polar step: $p \leftarrow p - \text{lr} \cdot \text{update}$.

---

### 4.3 Component 3: Category A Architectural & Data-Stream Primitives (`src/qdlblocks/layers/mudd_and_gates.py`, `src/qdlblocks/layers/isometric.py`, `src/qdlblocks/layers/attention.py`, `src/qdlblocks/benchmarks/fineweb_stream.py`)

#### 4.3.1 `src/qdlblocks/layers/mudd_and_gates.py`
1. **`InputChannelGate(nn.Module)`**:
   - Reads the leading `gate_channels: int = 6` channels of input tensor `x: (B, T, D)` (`c = min(gate_channels, D)`):
     - Parameters: `self.weight = nn.Parameter(torch.zeros(num_heads, c))` and `self.bias = nn.Parameter(torch.zeros(num_heads))` (zero-initialized).
     - Forward:
       $$g(x) = 2.0 \, \sigma\!\big(x_{:, :, :c} W^\top + b\big) \in \mathbb{R}^{B \times T \times H}$$
       reshaped to `(B, H, T, 1)` when `head_first=True` else `(B, T, H, 1)`.
     - **Step-0 Identity Guarantee**: Because `weight == 0` and `bias == 0` at initialization, $g(x) = 2.0 \cdot \sigma(0) \equiv 1.0$ at Step 0!
     - **Exact Placement**:
       - When `use_input_gates=True` in `ClassicalModdedBlock` or `SubspaceProjectorBornAttention`:
         - `ve_input_gate` scales the injected value embedding `ve_heads` (`ve_heads = ve_input_gate(h_in) * ve_heads`).
         - `attn_output_gate` scales the per-head attention output `attn_out` (`(B, H, T, D_h)`) before the output projection `out_proj` (`attn_out = attn_output_gate(h_in) * attn_out`).
2. **`apply_gated_xsa(y: torch.Tensor, v: torch.Tensor, xsa_alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor`**:
   - Given head outputs `y: (B, H, T, D_h)`, values `v: (B, H, T, D_h)`, and learnable per-head parameter `xsa_alpha: (H,)` (initialized to `torch.zeros(num_heads)`):
     $$\hat{y}_{b,h,t} = y_{b,h,t} - \tanh(\alpha_h)\, \frac{\langle y_{b,h,t}, v_{b,h,t} \rangle}{\|v_{b,h,t}\|_2^2 + \epsilon}\, v_{b,h,t}$$
   - **Exact Placement**: Applied inside `ClassicalModdedBlock.forward` and `SubspaceProjectorBornAttention.forward` immediately after computing the attention head output `y` (`(B, H, T, D_h)`) and before `out_proj`. At Step 0, $\alpha_h = 0 \implies \tanh(0) = 0 \implies \hat{y} = y$.
3. **`MUDDDynamicSkip(nn.Module)`**:
   - Multi-way Dynamic Dense Connection (`MUDD`, arXiv:2502.12170):
     - Given current state $x \in \mathbb{R}^{B \times T \times D}$ and a sequence of $K$ auxiliary source states $[s_0, \dots, s_{K-1}]$ (Specifically, at each block: $K = 1$ with $s_0 = \psi_0$ / $x_0$, or $K = 2$ when an intermediate cached layer state is also provided):
       $$h = \text{GELU}(x W_1) \in \mathbb{R}^{B \times T \times d_{\text{mudd}}}, \qquad \mu = (h W_2 + b_2) \cdot \text{scale} \in \mathbb{R}^{B \times T \times (1 + K)}$$
       where $d_{\text{mudd}} = 64$, $\text{scale} = 0.1$, $W_1$ is Kaiming-initialized, $W_2 = 0$, $b_2[0] = 1.0 / \text{scale}$ (weight `1.0` on current state $x$), and $b_2[1:] = 0.0$ (weight `0.0` on auxiliary skip states at Step 0).
     - Combines:
       $$y = \mu_0 \odot x + \sum_{k=0}^{K-1} \mu_{k+1} \odot s_k$$
     - If `spherical=True` (in `QuantumBornGPT`), returns `l2_normalize(y, dim=-1)`. At Step 0, $\mu_0 = 1, \mu_{k+1} = 0$, so `MUDDDynamicSkip` is an exact identity map.
4. **`ForwardSmearGate(nn.Module)`**:
   - Parameters: `self.gate_proj = nn.Linear(min(12, embed_dim), 1, bias=False)` initialized to `zeros_`, and `self.smear_lambda = nn.Parameter(torch.zeros(1))` initialized to `0.0`.
   - Forward on `x: (B, T, D)`:
     - For $T \le 1$, returns `x`.
     - For $T > 1$, computes $g_t = \lambda_{\text{smear}} \, \sigma(\text{gate\_proj}(x_{:, 1:, :12})) \in \mathbb{R}^{B \times (T-1) \times 1}$, sets $\tilde{x}_{:, 0} = x_{:, 0}$ and $\tilde{x}_{:, 1:} = x_{:, 1:} + g_t \odot x_{:, :-1}$.
     - If `spherical=True` (in `QuantumBornGPT`), returns `l2_normalize(x_tilde, dim=-1)`.

#### 4.3.2 `src/qdlblocks/layers/isometric.py` & `src/qdlblocks/layers/attention.py`
1. **`YaRNRotarySchedule(nn.Module)`** & **`apply_partial_key_offset`**:
   - `YaRNRotarySchedule(head_dim: int, max_seq_len: int = 65536, base: float = 1024.0, base_attn_scale: float = 0.12)`:
     - Stores `self.angular_freq` (`(head_dim // 2,)` with half-truncated zeros in the upper half) and `self.attn_scale = base_attn_scale`.
     - `apply_yarn(old_window: int, new_window: int, alpha: float = 1.0, beta: float = 32.0) -> float`:
       - When `new_window > old_window > 0`, rescales frequencies via the YaRN ramp (`rotations = block_size * old_window * angular_freq / (2 * pi)`, `weight = clamp((rotations - alpha) / (beta - alpha), 0, 1)`, `scaling_factor = old_window / new_window`, `angular_freq *= scaling_factor + weight * (1 - scaling_factor)`), rebuilds `cos_cached` and `sin_cached`, and updates `self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1.0`.
   - `apply_partial_key_offset(k_bhtd: torch.Tensor, rotary_dim: int = 64) -> torch.Tensor`:
     - Given key tensor `k_bhtd` of shape `(B, H, T, D_h)`: if $T > 1$ and $D_h > \text{rotary\_dim}$, shifts the stationary (non-rotating) upper channels by 1 token position so each key carries both position $t$'s rotating phase and position $t-1$'s stationary features:
       `k_out = k_bhtd.clone(); k_out[:, :, 1:, rotary_dim:] = k_bhtd[:, :, :-1, rotary_dim:]`.
2. **Optional Mixed-Width QK Slicing (`qk_head_dim`) & Paired-Head Attention (`use_paired_heads`)**:
   - When `qk_head_dim` is set to $d_{qk} < d_h$ (e.g., $d_{qk} = 64$ while $d_h = 128$), queries and keys slice the leading $d_{qk}$ channels (`q = q[..., :qk_head_dim]`, `k = k[..., :qk_head_dim]`) before QK-norm and attention score computation while values `v` remain full width $d_h$.
   - When `use_paired_heads=True` (with even `num_heads`), adjacent head pairs `(2h, 2h+1)` are interleaved along the sequence axis `(B, H // 2, 2 * T, D_h)` with causal masking so head `2h` and head `2h+1` attend jointly across the paired stream before reshaping back to `(B, H, T, D_h)`.

#### 4.3.3 `src/qdlblocks/benchmarks/fineweb_stream.py` (`find_bos_aligned_starts` & `compute_tapered_mtp_weight`)
1. **`find_bos_aligned_starts`**:
   ```python
   def find_bos_aligned_starts(
       tokens: np.ndarray | torch.Tensor,
       seq_len: int,
       batch_size: int,
       eot_token_id: int = 50256,
       rng: np.random.Generator | torch.Generator | None = None,
   ) -> np.ndarray | torch.Tensor:
   ```
   - Finds all valid start positions `pos` in `[0, len(tokens) - seq_len - 1)` where `pos == 0` or `tokens[pos - 1] == eot_token_id`.
   - If at least 1 valid boundary exists, samples `batch_size` indices uniformly (with replacement if `len(valid) < batch_size`) from `valid`; otherwise falls back to uniform random offsets in `[0, len(tokens) - seq_len - 1)`.
2. **`compute_tapered_mtp_weight(step: int, total_steps: int, mtp_weight_init: float, taper_end_frac: float = 0.65) -> float`**:
   - Let $S_{\text{taper}} = \max(1, \lfloor \text{taper\_end\_frac} \cdot \text{total\_steps} \rfloor)$. For $\text{step} \ge S_{\text{taper}}$, returns `0.0`; for $0 \le \text{step} < S_{\text{taper}}$, returns `mtp_weight_init * (1.0 - step / S_taper)`.

---

### 4.4 Component 4: Category C Throughput & Systems Primitives (`src/qdlblocks/layers/direct_chunked_ce.py`, `src/qdlblocks/layers/fp8_linear.py`)

#### 4.4.1 Sampled-Softmax Candidate Cross-Entropy (`src/qdlblocks/layers/direct_chunked_ce.py`)
1. **`build_sampled_candidate_set`**:
   ```python
   def build_sampled_candidate_set(
       targets: torch.Tensor,
       vocab_size: int,
       num_candidates: int,
       step: int = 0,
       negative_stride: int = 20011,
   ) -> tuple[torch.Tensor, torch.Tensor]:
   ```
   - Given `targets` (`(B, T)` or `(N_tok,)`):
     - Flatten `flat_targets = targets.reshape(-1)`.
     - If `num_candidates <= 0` or `num_candidates >= vocab_size`:
       - Return `(torch.arange(vocab_size, device=targets.device, dtype=torch.int64), flat_targets)`.
     - **Coprime Stride Check**: Ensure `stride = int(negative_stride) % vocab_size`; while `stride <= 0 or math.gcd(stride, vocab_size) != 1`, increment `stride += 1` so `k * stride mod vocab_size` is guaranteed to be a full-period permutation for any `vocab_size >= 2` (including small synthetic test vocabularies).
     - Extract sorted unique valid target IDs: `u_targets = torch.unique(flat_targets[(flat_targets >= 0) & (flat_targets < vocab_size)])`.
     - If `u_targets.numel() >= num_candidates`, set `candidate_ids = u_targets` (sorted ascending).
     - Otherwise, generate `n_needed = num_candidates - u_targets.numel()` negative classes:
       - Construct `n_probe = min(vocab_size, num_candidates + u_targets.numel())` stride indices `k = torch.arange(n_probe, device=targets.device, dtype=torch.int64) + (step * num_candidates)`.
       - Compute `raw_neg = (k * stride) % vocab_size`.
       - Filter out IDs already present in `u_targets` via `in_u = (torch.searchsorted(u_targets, raw_neg).clamp(max=max(0, u_targets.numel() - 1)))` check (`u_targets[in_u] == raw_neg`), take the first `n_needed` non-colliding negatives, concatenate with `u_targets`, and sort ascending to obtain `candidate_ids: (P,)` where $P = \text{num\_candidates}$.
     - Remap `flat_targets` to candidate positions `[0, P)` via `remapped = torch.searchsorted(candidate_ids, flat_targets.clamp(min=0))`, setting `remapped = torch.where(flat_targets >= 0, remapped, flat_targets)`.
     - Return `(candidate_ids, remapped)`.
2. **`sampled_softcap_linear_ce`** & **`sampled_povm_ce`**:
   - **`sampled_softcap_linear_ce(x, lm_head_weight, targets, num_candidates=10240, step=0, softcap=15.0, chunk_tokens=4096, ignore_index=-1)`**:
     - Calls `candidate_ids, remapped_targets = build_sampled_candidate_set(targets, lm_head_weight.size(0), num_candidates, step=step)`.
     - Slices `w_sub = lm_head_weight[candidate_ids]` (`(P, D)`) and calls `direct_chunked_softcap_linear_ce(x, w_sub, remapped_targets, softcap=softcap, chunk_tokens=chunk_tokens, ignore_index=ignore_index)`.
   - **`sampled_povm_ce(psi, dm_seed_weight, wte_weight, W_lift_weight, readout_ortho_weight, inv_t, targets, num_candidates=10240, step=0, readout_mode="asymmetric", log_prior=None, effects_weight=None, chunk_tokens=4096, ignore_index=-1, eps=1e-12)`**:
     - Calls `candidate_ids, remapped_targets = build_sampled_candidate_set(targets, dm_seed_weight.size(0), num_candidates, step=step)`.
     - Slices `dm_sub = dm_seed_weight[candidate_ids]`, `wte_sub = wte_weight[candidate_ids]`, `prior_sub = log_prior[candidate_ids] if log_prior is not None else None`, and (when untied `effects_weight` is provided) `eff_sub = effects_weight[candidate_ids]`, then delegates to `direct_chunked_povm_ce` over the $P$ candidates.

#### 4.4.2 `src/qdlblocks/layers/fp8_linear.py` (`FP8LinearWrapper`)
- Provides `FP8LinearWrapper(nn.Module)`, `is_fp8_scaled_mm_supported(device)`, and `convert_linear_layers_to_fp8(model)`:
  - During `training` on CUDA devices where `is_fp8_scaled_mm_supported` is `True` and input/output dimensions are multiples of 16, quantizes activations `x` and weights `w` to `torch.float8_e4m3fn` with dynamic per-tensor `amax / 448.0` scaling and dispatches to `torch._scaled_mm` in a custom `torch.autograd.Function` (with backward pass computing `grad_x` and `grad_w`).
  - Automatically and transparently falls back to standard `F.linear(x, w, bias)` during `eval()` or when shapes/devices do not satisfy `torch._scaled_mm` alignment constraints.

---

### 4.5 CLI Integration (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`), Model Cards (`src/qdlblocks/benchmarks/model_card.py`), & Wave-13 Manifest (`configs/wave13_fleet.json`)

1. **New CLI Flags in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` & `src/qdlblocks/benchmarks/model_card.py`**:
   - `--ppmi-svd-init`: Initialize embeddings using the GPU Shifted PPMI + Randomized SVD co-occurrence sketch.
   - `--ppmi-svd-shards` (`int`, default `9`): Number of FineWeb shards (`1..9`, up to `900M` tokens) to include in the PPMI-SVD sketch.
   - `--ppmi-svd-alpha` (`float`, default `0.65`): Blend weight $\alpha_{\text{ppmi}} \in [0, 1]$ between random Gaussian identity and PPMI-SVD basis.
   - `--ngram-hash`: Enable `HashedNgramEmbedding` (hashed bigram + trigram lookup with $\pm 1$ sign pool).
   - `--ngram-vocab-size` (`int`, default `262144`): Number of rows in the hashed n-gram table.
   - `--polar-express`: Use 6-step `ANVIL_MAPS` (`zeropower_via_polarexpress5`) in `StiefelMuon`.
   - `--normuon`: Enable `NorMuon` per-lane variance equalization in `StiefelMuon`.
   - `--twin-rail-muon`: Enable `ANVIL` twin-rail (`fast + slow` EMA) Nesterov momentum in `StiefelMuon`.
   - `--cautious-wd`: Set `decay_mode="cautious"` in `StiefelMuon`.
   - `--tail-ema`: Enable `TailWeightAverager` over the last `25%` of training steps and evaluate validation under the decontracted EMA weights.
   - `--mudd-and-gates`: Enable `InputChannelGate` (`ve_gate`, `attn_gate`), `GatedXSA`, and `MUDDDynamicSkip`.
   - `--smear-gate`: Enable 1-token `ForwardSmearGate` (even when `--leaderboard-exact-classical` is active).
   - `--tapered-mtp`: Linearly taper `mtp_weight` to `0.0` at `65%` of total steps.
   - `--align-to-bos`: Align training batch sequence starts to `<|endoftext|>` (`50256`) document boundaries.
   - `--sampled-softmax-candidates` (`int`, default `0`): When `> 0`, use sampled-softmax candidate CE with $P$ candidates during the first `75%` of training steps and full softmax for the final `25%` and all validation evaluations.
   - `--embed-step-every` (`int`, default `1`): Step the `AdamW` embedding optimizer every $K$ steps.
   - `--fp8-linear`: Wrap eligible linear projections in `FP8LinearWrapper`.

2. **Wave-13 Fleet Manifest (`configs/wave13_fleet.json`) — Staged in SQLite `session_queue` as `QUEUED`**:
   - **Slot 1 (`gpu_g4_w6_1`) — `w13_900m_qborn_v6_9shard_ppmi_svd_ngram_anvil_L8`**:
     - `QuantumBornGPT` (`L=8, d=1024, H=8, d_head=128`, 4d overcomplete two-beam MLP) with **9-Shard (`900M`-token) PPMI-SVD Co-Occurrence Initialization (`--ppmi-svd-init --ppmi-svd-shards 9 --ppmi-svd-alpha 0.65`)** + **`--ngram-hash`** + **Category A (`--polar-express --normuon --twin-rail-muon --cautious-wd --tail-ema --mudd-and-gates --align-to-bos`)**.
   - **Slot 2 (`gpu_g4_w6_2`) — `w13_900m_qborn_v6_9shard_ppmi_svd_deep_L12_H8_d128`**:
     - Deep `L=12` `QuantumBornGPT` (`d=1024, H=8, d_head=128`) with **9-Shard (`900M`-token) PPMI-SVD Initialization** + **`--ngram-hash`** + **Category A (`--polar-express --normuon --twin-rail-muon --cautious-wd --tail-ema --mudd-and-gates --align-to-bos`)**.
   - **Slot 3 (`gpu_g4_w6_3`) — `w13_900m_qborn_v6_9shard_ppmi_svd_ablation_no_ngram`**:
     - Controlled ablation isolating **9-Shard (`900M`-token) PPMI-SVD Co-Occurrence Initialization** on `QuantumBornGPT` (`L=8, d=1024, H=8, d_head=128`) without `--ngram-hash`, measuring the pure geometric gain of spectral co-occurrence initialization vs. Wave 12.
   - **Slot 4 (`gpu_a100_w6_1`) — `w13_900m_classical_anvil2_full_catABC_b131k`**:
     - `ClassicalModdedGPT` (`L=12, d=768, H=6, d_head=128`) with **all Categories A, B, and C enabled** (`--leaderboard-exact-classical --ngram-hash --polar-express --normuon --twin-rail-muon --cautious-wd --tail-ema --mudd-and-gates --smear-gate --align-to-bos --sampled-softmax-candidates 16384`).
   - **Slot 5 (`gpu_a100_w6_2`) — `w13_900m_classical_anvil2_9shard_ppmi_svd_b131k`**:
     - `ClassicalModdedGPT` with **all Categories A, B, and C** PLUS **9-Shard (`900M`-token) PPMI-SVD Co-Occurrence Initialization (`--ppmi-svd-init --ppmi-svd-shards 9`)**.

---

## 5. Verification Plan: Unit Tests & Pre-Launch Controlled Experiments

### 5.1 Unit Test Suite (`tests/test_hypothesis_021_ppmi_svd_ngram_and_anvil.py`)
1. `test_ppmi_svd_cooccurrence_init_ortho_and_collocation_snr`:
   - Generates multi-shard contextual Markov-Zipf tokens (including unseen/padding token IDs at the top of `vocab_size`), runs `compute_ppmi_svd_embeddings`, initializes `QuantumBornGPT(..., use_ortho_decouple=True)` via `initialize_quantum_born_with_ppmi_svd`, and verifies:
     - `max_self_overlap < 1e-6` (exact Step-0 input-output orthogonality),
     - all rows (including unseen/padding tokens) have unit $L_2$ norm `1.0` with zero NaNs,
     - `cos_ratio >= 3.0` and `colloc_purified_born_energy` increases by `>= 50%` over isotropic init,
     - disk cache round-trip and sign-determinism work identically.
2. `test_hashed_ngram_embedding_sign_trick_and_shapes`:
   - Verifies `compute_ngram_hashes_and_signs` and `HashedNgramEmbedding` produce exact shapes, zero Step-0 output when `zero_init=True`, non-zero gradients to `table.weight`, and uncorrelated sign cancellation across distinct n-grams colliding in the same row.
3. `test_polar_express_normuon_twin_rail_and_tail_ema`:
   - Verifies `zeropower_via_polarexpress5` achieves lower singular-value RMSE (`< 0.08`) than 5-step `zeropower_via_newtonschulz5`, `normuon_equalize` preserves Frobenius norm within `< 1e-5`, `StiefelMuon` runs cleanly with `polar_method="polar_express"`, `use_normuon=True`, `use_twin_rail=True`, `decay_mode="cautious"`, and `TailWeightAverager.average_parameters` swaps and restores weights losslessly while preserving Frobenius/spherical norms.
4. `test_mudd_gates_xsa_bos_align_and_sampled_softmax`:
   - Verifies `InputChannelGate`, `apply_gated_xsa`, `MUDDDynamicSkip`, `ForwardSmearGate`, `find_bos_aligned_starts`, `sampled_softcap_linear_ce`, `sampled_povm_ce` (including `log_prior` slicing and coprime fallback), and `FP8LinearWrapper` forward/backward correctness.

### 5.2 Pre-Launch Verification Script (`examples/31_verify_plan021_ppmi_svd_ngram_and_anvil.py`)
1. **Experiment H1 (9-Shard `900M`-Token Throughput Benchmark + 45-Step `QuantumBornGPT` Controlled Ablation)**:
   - Benchmarks `compute_ppmi_svd_embeddings` across `900M` tokens (`9 × 100M` `uint16` token shards) on GPU, recording wall-clock time (`< 15.0s`) and Step-0 collocation coherence metrics.
   - Runs three matched 45-step `QuantumBornGPT` training runs on held-out validation (`T_val=512`):
     - `verify021_45step_qborn_baseline_isotropic` (Isotropic Gaussian init)
     - `verify021_45step_qborn_ppmi_svd_9shard` (PPMI-SVD Co-Occurrence Init with `--ortho-decouple`)
     - `verify021_45step_qborn_ppmi_svd_plus_ngram_anvil` (PPMI-SVD Init + `HashedNgramEmbedding` + Category A `ANVIL`/`NorMuon`/`MUDD`/`TailEMA`)
2. **Experiment H2 (45-Step `ClassicalModdedGPT` Category A+B+C Controlled Comparison)**:
   - Runs matched 45-step `ClassicalModdedGPT` training runs comparing Wave-12 `--leaderboard-exact-classical` vs. full Categories A+B+C (`verify021_45step_classical_catABC_anvil_ngram_ema`).
3. **Experiment H3 (Category C Sampled-Softmax & Systems Verification + SQLite Ingestion)**:
   - Verifies sampled-softmax speedup and exactness, ingests all `verify021_*` runs into `/home/festivity/agy/QuantumDLBlocks/.cache/benchmarks.db` (`run_tier = "micro_verification"`), registers `configs/wave13_fleet.json` in `session_queue` as `QUEUED`, and writes `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_021.json`.
