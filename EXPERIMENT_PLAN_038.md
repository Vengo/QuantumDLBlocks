# Experiment Plan 038: Last-Layer Measurement Operator Against Initial Wavefunctions ($\psi_v^{(0)}$), Traceless Observable Readout ($\operatorname{Tr}(\widehat{O}) = 0$), & Parseval / Incomplete Block-QR Packing Bounds (`Wave 23`)

## 1. Context & Mathematical Motivation

### 1.1 Empirical Diagnosis from `Wave 22` (Plan 037)
In Plan 037 (`Wave 22`), Fix 4 accelerated `Radical2Pow16WaveletCholeskyGPT` by **`2.72x`** (`25,770 tok/s` vs. `9,475 tok/s` at 900 steps; `26,568 tok/s` vs. `11,588 tok/s` at 3,600 steps), and Fix 3 locked the Kraus Stiefel error at `K_err = 0.0133` (vs. `20.70` in `w21`). However, logging the genuine spectral participation ratio $R_{\text{sharp}} = (\operatorname{Tr} G)^2 / \|G\|_F^2$ of `m_sharp` exposed **two critical bottlenecks** that prevented `w22` (`5.0688`) from beating `w21` (`5.0211`):
1. **Unconstrained Detector Cone Collapse (`R_sharp: 249.5 -> 16.5` by Step 900)**:
   In `synthesize_vocab_detectors()`, adding the rank-256 `coarse_transition` table and rank-2048 `pred_wav_2048` table directly to every vocabulary detector $m_v$ (`(50304, 256)`) and updating `up_8192`, `up_65536`, and `wav_*` via unconstrained `AdamW` without vocabulary zero-mean centering caused all `50,257` detectors to collapse into an ~18-dimensional cone (`R_sharp = 16.5`). By Parseval's identity, $\sum_{v=1}^V c_{t,v}^2 = V / R_{\text{sharp}} \approx 50,257 / 16.5 = \mathbf{3,045.9}$, inflating the tail partition function $Z_{\text{tail}}$.
2. **Step-0 Random Jitter & Scale Gain Distortion in `w22`**:
   Adding random Gaussian noise (`iso_jit`) to `wav_65536` and overwriting `scale_gains` to `[0.65, 0.80, 0.90, 1.50]` when `num_fine_banks = 16` artificially inflated Step-0 `R_sharp` while diluting the clean PPMI coarse/mid semantic hierarchy (`w256, r2048, w8192`).

### 1.2 Why Packing $V = 50,304$ Vectors into $\mathbb{R}^d$ Hits the Parseval / Welch Floor (and What Incomplete Block-QR Reveals)
For any set of $V = 50,304$ unit vectors $\{\psi_v\}_{v=1}^V \subset \mathbb{S}^{d-1}$ with Gram matrix $G = \sum_{v=1}^V \psi_v \psi_v^\top \in \mathbb{R}^{d \times d}$ ($\operatorname{Tr}(G) = V$):
- **Standard QR limit**: Exact QR on $M^\top \in \mathbb{R}^{d \times V}$ terminates after $\min(d, V) = d$ columns; at most $d$ vocabulary words can be mutually orthogonal.
- **Incomplete / Windowed Block-QR across $V = 50,304$ vectors**: Even if we partition the $50,304$ vectors into local blocks of size $b \le d$ and apply `torch.linalg.qr` (with damped inter-block Gram-Schmidt projection), making block $m$ internally orthonormal inevitably rotates block $m$ back into the subspace of earlier blocks $1, \dots, m-2$. By Cauchy-Schwarz (`Welch bound`), the global Frobenius norm satisfies $\|G\|_F^2 \ge V^2 / d$, so the average squared overlap per vector is bounded below by:
  $$\frac{1}{V} \sum_{u=1}^V \sum_{v=1}^V c_{u,v}^2 = \frac{V}{R_{\text{sharp}}} \ge \frac{V}{d}, \qquad \sqrt{\mathbb{E}_{u \ne v}[c_{u,v}^2]} \ge \sqrt{\frac{V - d}{d(V - 1)}} \approx \frac{1}{\sqrt{d}}.$$

### 1.3 Two Architectural Solutions for Plan 038 (`Wave 23`)
1. **Last-Layer Measurement Operator Against Initial Wavefunctions $\psi_v^{(0)}$ (`use_last_layer_measurement_operator=True`)**:
   - **Adjoint Equivalence**: Instead of transforming all $V = 50,304$ vocabulary vectors through `readout_chol_bridge`, `coarse_transition`, `pred_wav_2048`, and $R = 16$ Kraus operators (`(16, 256, 256)`) on every step, we apply the measurement operator $\mathcal{M}_{\text{last}}$ and the Kraus bath directly to the **last-layer network output $\psi_t$** ($B \times T = 8,192$ token states — **$6.14\times$ fewer rows than $V = 50,304$**!).
   - **Exact Zero Self-Loop Overlap Directly on $\psi_t$**: Using the input token's initial wavefunction $\psi_{x_t}^{(0)}$ (already computed in `embed_tokens(idx)`), the measurement operator projects $\psi_{x_t}^{(0)}$ out of both the sharp and diffuse output beams in $O(B \cdot T \cdot d)$ time:
     $$\tilde{\psi}_{\text{sharp}, t} = \operatorname{Norm}\!\Big(m_{\text{out}, t} - \langle m_{\text{out}, t} \mid \psi_{x_t}^{(0)} \rangle \, \psi_{x_t}^{(0)}\Big), \qquad \tilde{\psi}_{\text{diff}, t} = \operatorname{Norm}\!\Big(d_{\text{out}, t} - \langle d_{\text{out}, t} \mid \psi_{x_t}^{(0)} \rangle \, \psi_{x_t}^{(0)}\Big)$$
     guaranteeing $\langle \tilde{\psi}_{\text{sharp}, t} \mid \psi_{x_t}^{(0)} \rangle \equiv 0$ and $\langle \tilde{\psi}_{\text{diff}, t} \mid \psi_{x_t}^{(0)} \rangle \equiv 0$.
   - **Tied, Zero-Mean Centered Initial Wavefunctions $\psi_v^{(0)}$**: Both beams contract against the **single** tied initial wavefunction table $\psi_v^{(0)} = \operatorname{Norm}(\psi_v^{\text{raw}} - \bar{\psi}^{\text{raw}})$, keeping $\sum_v \psi_v^{(0)} \approx 0$, routing `up_8192` and `up_65536` through `StiefelMuon` so $R_{\text{sharp}}(\psi^{(0)})$ resists cone collapse, and preserving 100% clean PPMI SVD initialization at Step 0.
2. **Traceless Observable Readout (`use_traceless_observable=True`)**:
   - Replaces the raw inner product $c_{t,v}$ in the sharp $k$-boson beam with the **zero-trace quantum observable** ($\sum_{v=1}^V \widehat{O}_{t,v}^{\text{sharp}} \equiv 0$):
     $$\hat{c}_{t,v} = c_{t,v} - \frac{1}{V} \sum_{u=1}^V c_{t,u}, \qquad \widehat{O}_{t,v}^{\text{sharp}} = \hat{c}_{t,v} + \gamma_{\text{trace}} \left(\operatorname{relu}(\hat{c}_{t,v})^2 - \frac{1}{V} \sum_{u=1}^V \operatorname{relu}(\hat{c}_{t,u})^2\right)$$
   - **Three Simultaneous Benefits**:
     1. Exact zero first-order trace ($\sum_v \hat{c}_{t,v} = 0$) eliminates any residual cone bias $\langle \tilde{\psi}_t \mid \bar{\psi} \rangle$.
     2. Anti-aligned words ($\hat{c}_{t,v} \le 0$) have $\operatorname{relu}(\hat{c}_{t,v}) = 0$ and receive a strictly negative shift $-\gamma_{\text{trace}} \tau_{t,+}$, never a positive quadratic excitation (solving Fix 1).
     3. Unrelated words in the Parseval noise floor ($0 < \hat{c}_{t,v} < \sqrt{\tau_{t,+}}$) have $\operatorname{relu}(\hat{c}_{t,v})^2 - \tau_{t,+} < 0$ (suppressing the tail partition function $Z_{\text{tail}}$), while true target words ($\hat{c}_{t,y_t} \gg \sqrt{\tau_{t,+}}$) get a superlinear positive boost.

---

## 2. Falsifiable Hypotheses (`H1`, `H2`, `H3`)

### Hypothesis 1 (`H1` — Parseval / Incomplete Block-QR Packing Bounds, Exact Traceless Observable Gradients, & Last-Layer Measurement Zero Self-Overlap)
- **Claim**:
  1. Running [`scripts/parseval_tail_overlap_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_tail_overlap_profiler.py) across $V = 50,304$ vectors in dimensions $d \in \{16, 32, 64, 128, 256, 384, 512, 1024, 2048\}$ with incomplete Block-QR orthogonalization and Welch frame-potential descent will confirm that local blocks of size $b \le d$ achieve near-zero intra-block overlap, while global RMS off-diagonal overlap across all $50,304$ vectors remains bounded below by the Welch/Parseval floor $\sqrt{(V - d) / (d(V - 1))}$, and that the **Traceless Observable** reduces the $k$-boson tail partition inflation $\Delta \ln Z_{\text{tail}}$ at $d = 256$ to below the uncentered $d = 1,024$ floor.
  2. `direct_chunked_two_beam_kboson_ce` with `traceless_gamma > 0.0` will match `compute_two_beam_kboson_logits` + `F.cross_entropy` to `< 1e-5` in forward loss and `< 1e-4` in analytical gradients `(dpsi_sharp, dm_sharp, dbeta, dk)`.
  3. `Radical2Pow16WaveletCholeskyGPT` with `use_last_layer_measurement_operator=True` will achieve exact zero self-loop overlap (`max_self_overlap <= 1e-5`) on the measured output state $\tilde{\psi}_t$ while maintaining `< 11.0M` parameters.

### Hypothesis 2 (`H2` — 900-Step / `7.37M`-Token Ablation: Last-Layer Measurement Operator, Traceless Observable, and Combined)
- **Claim**: Over `900` steps (`7.37M` tokens, $B=16, T=512$, `<= 8.0 GB` VRAM) on genuine FineWeb-Edu (`fineweb_train_000001.bin` $\to$ `fineweb_val_000000.bin`):
  1. **`w23_8gb_last_layer_meas_900steps` (`use_last_layer_measurement_operator=True`)**, **`w23_8gb_traceless_obs_900steps` (`use_traceless_observable=True`)**, and **`w23_8gb_last_layer_meas_traceless_900steps` (Combined)** will all outperform Plan 037's `w22` (`5.5613` Val CE) and maintain `sharp_eff_rank >= 65.0` at Step 900 (`> 3.9x` higher than `w22`'s collapsed `16.5`).
  2. **`w23_8gb_last_layer_meas_traceless_900steps` (Combined)** will achieve the lowest 900-step validation CE across all `N = 2^16` architectures (`< 5.4734`, beating `w21`'s `5.4734`, `w22`'s `5.5613`, `40.78M` `QuantumBornGPT`'s `5.6092`, and `53.54M` `ClassicalModdedGPT`'s `5.9152`) at **`>= 25,000 tok/s`**.

### Hypothesis 3 (`H3` — 3,600-Step / `29.49M`-Token Scaling Run Breaking Both `w21` (`5.0211`) and `5.00` Val CE at `> 2.2x` Higher Throughput)
- **Claim**: Over `3,600` steps (`29.49M` tokens, evaluated at steps `0, 300, 900, 1800, 2700, 3600`) on genuine FineWeb-Edu under the `<= 8.0 GB` VRAM cap:
  - **`w23_8gb_last_layer_meas_traceless_3600steps` (`10.64M` params)** will achieve **`< 5.00` Val CE** — beating `w22` (`5.0688`), `w21` (`5.0211`), `w20` (`5.0858`), `38.28M` Parity-Split `QuantumBornGPT` (`5.1157`), and `30.95M` `ClassicalModdedGPT` (`5.5202`), while sustaining `>= 25,000 tok/s` (`> 2.15x` faster than `w21`), `sharp_eff_rank >= 65.0`, and `kraus_stiefel_err <= 0.05`.

---

## 3. Planned Architectural & Code Changes

1. **[`src/qdlblocks/layers/direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py)**:
   - Add `traceless_gamma: float = 0.0` to `_compiled_kboson_chunk_step`, `_DirectChunkedTwoBeamKBosonCEFn`, `direct_chunked_two_beam_kboson_ce`, and `compute_two_beam_kboson_logits` with exact analytical forward and backward support.
2. **[`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py)**:
   - Add `use_last_layer_measurement_operator: bool = False`, `use_traceless_observable: bool = False`, and `traceless_gamma: float = 0.55` to `Radical2Pow16Config`.
   - Add `pyramid.synthesize_initial_wavefunctions()` (zero-mean centered across the active vocabulary so $\sum_v \psi_v^{(0)} = 0$) and `pyramid.apply_last_layer_measurement(psi, psi_diff, psi_in, idx)` (applying `readout_chol_bridge`, `coarse_transition`, `pred_wav_2048`, and the $R=16$ Kraus operators to the $(B, T, d)$ output state and projecting out $\psi_{x_t}^{(0)}$ directly on the output side).
   - Route `up_8192` and `up_65536` to `StiefelMuon` when `use_last_layer_measurement_operator` or `use_traceless_observable` is enabled, and preserve clean noise-free per-bank SVD initialization in `initialize_radical_2pow16_with_ppmi_svd`.
3. **Theory & Empirical Writeup + Promoted CLI Diagnostic Scripts**:
   - [`docs/PARSEVAL_PACKING_AND_DIMENSIONAL_BOUNDS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/PARSEVAL_PACKING_AND_DIMENSIONAL_BOUNDS.md): Comprehensive mathematical + numerical analysis of Fix 1 ($\mathbb{RP}^{d-1}$ identification), Parseval/Welch bounds, incomplete Block-QR orthogonalization of $50,304$ vectors across $d \in \{16, \dots, 2048\}$, dimensional thresholds for `< 3.28` and `< 2.50` nats, Last-Layer Measurement Operator, and Traceless Observable readout.
   - [`scripts/parseval_tail_overlap_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_tail_overlap_profiler.py) (promoted from `scripts/BACKLOG.md`).
   - [`scripts/block_cholesky_kernel_benchmark.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/block_cholesky_kernel_benchmark.py) (promoted from `scripts/BACKLOG.md`).
4. **Verification Script & Tests**:
   - [`examples/48_verify_plan038_last_layer_measurement_and_traceless.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/48_verify_plan038_last_layer_measurement_and_traceless.py)
   - [`tests/test_plan038_last_layer_measurement_and_traceless.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan038_last_layer_measurement_and_traceless.py) & [`tests/test_hypothesis_038_last_layer_measurement_and_traceless.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_038_last_layer_measurement_and_traceless.py).

---

## 4. Verification & Testing Plan

1. **AST & Best-Practices Gate**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/layers/wavelet_cholesky.py src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py scripts/parseval_tail_overlap_profiler.py scripts/block_cholesky_kernel_benchmark.py examples/48_verify_plan038_last_layer_measurement_and_traceless.py
   ```
2. **Unit & Hypothesis Test Suite**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/pytest tests/test_plan037_tail_fixes_and_tooling.py tests/test_hypothesis_037_tail_fixes_and_fast_cholesky.py tests/test_plan038_last_layer_measurement_and_traceless.py tests/test_hypothesis_038_last_layer_measurement_and_traceless.py -v
   ```

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

1. **Existing Scripts Used**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
   - [`scripts/dyadic_bigram_transition_heatmap.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/dyadic_bigram_transition_heatmap.py)
   - [`scripts/wavelet_scale_ablation_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wavelet_scale_ablation_probe.py)
2. **`Build-Before-Launch` Scripts (Promoted from `scripts/BACKLOG.md`)**:
   - [`scripts/parseval_tail_overlap_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_tail_overlap_profiler.py)
   - [`scripts/block_cholesky_kernel_benchmark.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/block_cholesky_kernel_benchmark.py)
3. **`Nice-to-Have` Scripts (Logged in `scripts/BACKLOG.md`)**:
   - `scripts/last_layer_measurement_adjoint_bench.py`: Benchmarks FLOPs, memory traffic, and self-overlap suppression of the $(B \cdot T, d)$ Last-Layer Measurement Operator vs. $(V, d)$ Vocabulary Detector synthesis across batch/sequence shapes.
   - `scripts/traceless_observable_rank_bucket_probe.py`: Profiles logit margins and probability mass across rank buckets (`1`, `2–10`, `11–100`, `101–1000`, `>1000`) as a function of `traceless_gamma`.
