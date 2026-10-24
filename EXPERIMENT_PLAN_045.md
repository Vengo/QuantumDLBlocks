# Experiment Plan 045 (`Wave 29`): Breaking `< 3.90` Val CE Under `150M` Tokens on FineWeb & Scaling Trajectory to `300M` and `900M` Tokens

## 1. Motivation & Mathematical Root-Cause Analysis

The user posed our next key experimental question:
> *"For a next model training experiment, I'm curious to see if our best techniques can bring CE below 3.9 in under 150 M tokens on FineWeb, and how low a model achieving that target can take loss by the 300M and 900M token marks."*

### Why `Waves 20–28` Stopped Around `4.92–5.36` CE While `Wave 13–16` Reached `4.11` at `100M` and `3.498` at `900M`
When we compare our `Waves 20–28` local runs against our `Waves 11–16` full FineWeb runs, two structural factors explain the gap—and combining them unlocks `< 3.90` Val CE in under `150M` tokens:

1. **Factor 1 — Data Stream Diversity (`8M` Slice vs. Full `100M–900M` Multi-Shard Stream) & Surface N-Gram Bypass (`HashedBigramTrigramEmbedding` + Cross-Layer $V^{(0)}$)**:
   - In `Waves 20–28`, `load_fineweb_tokens(max_train_tokens=8_000_000)` restricted training to the first `8M` tokens of Shard 1, and `ParsevalMuPDyadicGPT` ($10.4\text{M}$–$23.5\text{M}$ params) had to spend its early layers learning surface bigram `(x_{t-1}, x_t)` and trigram `(x_{t-2}, x_{t-1}, x_t)` statistics from scratch.
   - In `Wave 13` (`Plan 021`) and `Wave 16` (`Plan 027`), adding **9-Shard PPMI-SVD Initialization** + **Hashed Bigram/Trigram Surface Embeddings** (`ngram_vocab_size = 262,144`) + **True Cross-Layer Layer-0 Value Residual $V^{(0)}$** + **ANVIL Twin-Rail `StiefelMuon`** brought `QuantumBornGPT` to `4.1178` Val CE at `100M` tokens and `3.9401` at `200M` tokens (`~4.02` at `150M` tokens), finishing at `3.4980` at `900M` tokens—*even though `Wave 16` was still crippled by the legacy concave $k$-boson readout* $k\ln(1 + \tilde{r}_t E_{t,v})$!
2. **Factor 2 — What Happens When We Replace `Wave 16`'s Concave Readout with `Wave 28`'s Flag-Free Parseval $\mu\text{P}$ $\beta_0(d) = 1.48\sqrt{d}$ + Convex-Peak Traceless Born + Dyadic Hierarchy (`Wave 29`)**:
   - In `Waves 24–28`, replacing the concave $k$-boson logarithm with **Flag-Free Parseval $\mu\text{P}$ $\beta_0(d) = 1.48\sqrt{d}$ + Post-Norm Cholesky Metric Absorption + Convex-Peak Traceless Born ($\gamma = 0.25$) + 256-Root Dyadic Conditional Readout** reduced FineWeb-Edu validation CE by **`-0.467` to `-0.665` nats** (`5.6818 -> 5.4372 -> 5.3671` at `7.37M` tokens; `5.1479 -> 5.0162 -> 4.9228` at `29.5M–44.2M` tokens) by eliminating peak-derivative compression $1/(1 + \tilde{r}_t \widehat{O}_{t,y})$ and negative-tail squaring noise.
   - Therefore, unifying our **best backbone & surface-context primitives** (`HashedBigramTrigramEmbedding` + `Cross-Layer V^(0)` + `WaveletCholeskyMultiscaleBranch` + `OvercompleteTwoBeamInterference` + `SubspaceProjectorBornAttention` + `Twin-Rail StiefelMuon`) with our **best `Wave 28` readout** (`ParsevalMuP` $\beta_0(d) = 1.48\sqrt{d}$ + `Convex-Peak Traceless Born` + `256-Root Dyadic Hierarchy` + `FP8` fused chunked kernel) on the genuine **Multi-Shard FineWeb10B stream** (`FineWebMultiShardStream`) is projected to break **`< 3.90` Val CE before `150M` tokens** (reaching **`~3.76–3.84` Val CE at `150M` tokens**), **`~3.54–3.62` Val CE at `300M` tokens**, and **`~3.26–3.34` Val CE at `900M` tokens**!

---

## 2. Testable Hypotheses (`Wave 29` / `EXPERIMENT_PLAN_045.md`)

- **Hypothesis `H1` — Flag-Free Unified `ParsevalMuPNGramSpeedrunGPT` Architecture & Forward/Backward Verification**:
  - Constructing the clean, Rule-6 flag-free [`src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py) combining `ParsevalMuPDyadicGPT` (`Wave 28`) with gated spherical `HashedBigramTrigramEmbedding` and Cross-Layer $V^{(0)}$ value residuals produces finite gradients across all parameter groups, zero `PurityGate`/`kboson_k`/clamps, and `>= 28,000 tok/s` throughput on the local RTX 5060 Ti.
- **Hypothesis `H2` — Controlled Multi-Shard FineWeb Scaling & `< 3.90` Val CE at `< 150M` Tokens**:
  - Training `w29_fineweb_parseval_mup_ngram_v0` on genuine multi-shard FineWeb (`fineweb_train_000001..000009.bin` -> `fineweb_val_000000.bin`, logged live via `StandardizedExperimentLogger` into the QuantumDLBlocks WebUI) achieves strictly lower Validation CE at every token milestone than both `Wave 28` (`ParsevalMuPDyadicGPT` without surface N-gram/$V^{(0)}$) and `Wave 16` (`QuantumBornGPT` with concave $k$-boson readout, `4.1178` at `100M` tok, `~4.02` at `150M` tok), crossing **`< 3.90` Val CE in under `150M` tokens**.
- **Hypothesis `H3` — Asymptotic `150M`, `300M`, and `900M` Token Scaling Law & WSD-Cooldown Projections**:
  - Fitting the 3-parameter Chinchilla + WSD-cooldown scaling law $\text{CE}(N) = E_\infty + A N^{-\alpha}$ across the measured multi-horizon checkpoints (`2.46M`, `7.37M`, `14.75M`, `29.49M`, `50M`, `100M`, `150M` tokens) and `Wave 13–16` 900M reference trajectories quantifies the exact loss achieved at **`150M` tokens (`< 3.90` nats)**, **`300M` tokens (`~3.55–3.64` nats)**, and **`900M` tokens (`~3.26–3.35` nats)**.

---

## 3. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 & Section 6)

- **Existing Scripts Used**:
  - [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py)
  - [`scripts/fineweb_convex_vs_concave_slope_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_convex_vs_concave_slope_tracker.py)
  - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
- **`Build-Before-Launch` Script**:
  - [`scripts/fineweb_150m_300m_900m_scaling_projector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_150m_300m_900m_scaling_projector.py): Evaluates and projects `150M`, `300M`, and `900M` token validation Cross-Entropy across `Wave 11–16` (900M FineWeb), `Wave 20–28` (Wavelet-Cholesky + Parseval $\mu\text{P}$), and `Wave 29` (`ParsevalMuPNGramSpeedrunGPT`), outputting `.cache/fineweb_150m_300m_900m_scaling_projector.json` and registering the runs in the WebUI via `StandardizedExperimentLogger`.
