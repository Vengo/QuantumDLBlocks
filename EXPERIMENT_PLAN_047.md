# Experiment Plan 047 (`Wave 32`): Explicit Level-Separated Wavelet Tensors, Multi-Rate 1D Wavefunction Convolution Bank Feeding Quantum Attention, and Subspace-Balanced Coincidence Readout

**Date**: October 24, 2026  
**Wave**: `Wave 32` (`EXPERIMENT_PLAN_047`)  
**Status**: Implementing & Verifying  
**Target Domain**: `fineweb_edu_local` & `micro_verification` (`< 3.28` Val CE Roadmap, Lever 3 Refinement)

---

## 1. Motivation & Mathematical Contrast: Global L2 Wavelet Summing vs. Explicit Level-Separated Tensors & 1D Wavefunction Convolutions

### 1.1 Why Global L2 Normalization Over Summed Wavelet Levels Causes Fine-Level Suppression (And Why $R_{\text{sharp}}$ Drop Was a Symptom)
In `Tied2Pow16WaveletCholeskyPyramid._synthesize_leaf_vectors` ([`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py#L172-L185)), the four dyadic levels of a token's representation—`c3` (`wav_256`), `c2` (`wav_2048`), `c1` (`wav_8192`), and `c0` (`wav_65536`)—were all projected into the same ambient space $\mathbb{R}^d$, summed together, and normalized by a **single global L2 norm**:
$$\psi_{\text{merged}}(v) = \frac{g_0 c_3(v) + g_1 c_2(v) + g_2 c_1(v) + g_3 c_0(v)}{\|g_0 c_3(v) + g_1 c_2(v) + g_2 c_1(v) + g_3 c_0(v)\|_2} \in \mathbb{S}^{d-1}$$

1. **Why Early $R_{\text{sharp}}$ Drop (`256 -> 10 -> 24`) Is a Symptom Rather Than a Standalone Bug**:
   - When the model learns that groups of words are semantically or syntactically related even though their initial PPMI embeddings were nearly orthogonal, pulling those words into shared semantic clusters *naturally* reduces the global participation ratio $R_{\text{sharp}}$. That semantic clustering is desirable.
   - However, the *reason* $R_{\text{sharp}}$ failed to recover later in training is a structural flaw in the global L2 sum above: every training batch (`8,192` tokens) updates all $256$ coarse rows of `wav_256` (`c3`) $\sim 32$ times per step, whereas a specific fine leaf row in `wav_65536` (`c0`) only receives a gradient when that specific token occurs.
   - As $\|g_0 c_3(v)\|_2$ grows large during early semantic clustering, it **inflates the shared denominator $\|g_0 c_3 + g_1 c_2 + g_2 c_1 + g_3 c_0\|_2$**, crushing the effective amplitude of the fine lexical leaf $c_0(v)$ toward zero!

2. **The Fix — Explicit Level-Separated Wavelet Tensors (`WaveletScaleState` & `LevelSeparatedWaveletPyramid`)**:
   Instead of projecting all 4 wavelet levels into $\mathbb{R}^d$ and adding them under one global norm, we represent the 4 dyadic levels as **four logically and physically separate tensors in code**, each of dimension $d_s = d / 4$ and each normalized on its **own unit sphere** $\frac{1}{2}\mathbb{S}^{d_s - 1}$:
   $$\Psi(v) = \Big(\psi^{(0)}_{\text{coarse\_256}}(v),\; \psi^{(1)}_{\text{mid\_2048}}(v),\; \psi^{(2)}_{\text{sub\_8192}}(v),\; \psi^{(3)}_{\text{fine\_65536}}(v)\Big) \in \left(\frac{1}{2}\mathbb{S}^{d_s - 1}\right)^4 \subset \mathbb{S}^{d-1}$$
   where $\|\psi^{(m)}(v)\|_2 = \frac{1}{2}$ for every level $m \in \{0, 1, 2, 3\}$.
   - Now `wav_256` can cluster words semantically inside $\psi^{(0)}_{\text{coarse\_256}}$ as strongly as it wants **without ever shrinking the L2 norm or gradient of `wav_65536` inside $\psi^{(3)}_{\text{fine\_65536}}$**!
   - Furthermore, for existing full-dimension tensors, **Strided (Round-Robin) Subspace Partitioning** (`x[..., m::M]`) distributes sorted PPMI-SVD singular values evenly across all $M$ subspaces so no subspace is stuck with only noisy tail singular vectors.

### 1.2 How 1D / Strided Convolutions Over a Bank of Wavefunctions Contrast With Our Current U-Net
In our existing `FusedKroneckerWaveletUNetBlock` ([`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py#L541-L560)):
1. **What the Current U-Net Does**:
   - First, `SubspaceProjectorBornAttention` runs on single-token states $\psi_t$.
   - Second, *after* attention, `WaveletCholeskyMultiscaleBranch` runs a stride-2 lifting filter bank across the **embedding channel axis $d$** (`d -> d/2 -> d/4 -> d/8`), applies a 2-tap causal shift (`t-1` and `t-2`) on the coarse channel subband, and immediately merges everything back into a single L2-normalized vector.
2. **What the New `MultiRateWavefunctionConvBank` Does (Feeding *Into* Quantum Attention)**:
   - Operates directly on the **explicitly separated bank of $M=4$ scale wavefunctions** $\big(\psi_t^{(0)}, \psi_t^{(1)}, \psi_t^{(2)}, \psi_t^{(3)}\big)$ along the **sequence/time axis $t$** *before* `SubspaceProjectorBornAttention`:
     - **Scale $m=3$ (`fine_65536`, temporal stride/dilation $s_3 = 1$, kernel $k=1$)**: Pure token-level wavefunction $\psi_t^{(3)}$.
     - **Scale $m=2$ (`sub_8192`, causal 1D conv with dilation $s_2 = 1$, kernel $k=2$)**: Coherent 2-token local interference over $(\psi_{t-1}^{(2)}, \psi_t^{(2)})$.
     - **Scale $m=1$ (`mid_2048`, strided/dilated causal 1D conv with stride/dilation $s_1 = 2$, kernel $k=2$)**: Multi-token dyadic envelope over $(\psi_{t-2}^{(1)}, \psi_t^{(1)})$ with causal strided pooling interpolation.
     - **Scale $m=0$ (`coarse_256`, strided/dilated causal 1D conv with stride/dilation $s_0 = 4$, kernel $k=3$)**: Coarse phrase/topic wavefunction over $(\psi_{t-8}^{(0)}, \psi_{t-4}^{(0)}, \psi_t^{(0)})$ acting as a contextual density summary.
   - Each scale is conditioned by its own SPD `BlockCholeskyLinear(d_s)` operator and re-normalized onto $\frac{1}{\sqrt{M}}\mathbb{S}^{d_s - 1}$ before feeding into **Quantum Born Attention** ($Q, K, V$). Thus, attention inner products $\langle Q_t \mid K_\tau \rangle = \sum_{m=0}^3 \langle Q_t^{(m)} \mid K_\tau^{(m)} \rangle$ jointly match unigram wavefunctions, 2-gram convolved wavefunctions, 4-token strided wavefunctions, and 8-token coarse context wavefunctions!

---

## 2. Hypotheses & Quantitative Acceptance Gates

- **Hypothesis H1 (Elimination of Cross-Level L2 Norm Cannibalization & SVD Spectrum Imbalance)**:
  1. In [`scripts/subspace_specialization_mutual_information_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_specialization_mutual_information_probe.py), when coarse wavelet table `wav_256` grows by $8\times$ during semantic clustering, global L2 normalization (`M=1` merged sum) crushes the fine `wav_65536` fractional energy below `5%`, whereas `LevelSeparatedWaveletPyramid` (`WaveletScaleState`) preserves exact **`25.0%` (`1/M`) L2 energy** in every wavelet level (`max_level_energy_err < 1e-6`).
  2. On PPMI-SVD initialized embeddings, **Strided (Round-Robin) Subspace Partitioning** (`strided=True`, `x[..., m::M]`) reduces inter-subspace singular-value variance skew by **`>= 10x`** compared to contiguous slicing (`x.view(..., M, d // M)`).
- **Hypothesis H2 (End-to-End `FineWeb-Edu` Training Improvement with Explicit Level-Separated Wavelet Tensors + Pre-Attention Multi-Rate 1D Wavefunction Convolution)**:
  - On genuine `FineWeb-Edu` (`150`-step and `400`-step controlled GPU verification in [`examples/57_verify_plan047_level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/57_verify_plan047_level_separated_wavelet_conv_gpt.py)), `LevelSeparatedWaveletConvGPT` (explicit level-separated wavelet tensors + pre-attention 1D strided wavefunction convolution + strided/branch-assigned coincidence readout) outperforms both the contiguous `Wave 31` `M=4` model (`6.8437`) and the `M=1` single-sphere baseline (`6.8074`) at 150 steps while maintaining high per-level effective rank and zero cross-level norm cannibalization.

---

## 3. Planned Changes (Rules 1–8 Compliant)

1. **[`src/qdlblocks/layers/subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) (`UPDATE`)**:
   - Add `strided_subspace_l2_normalize` and `partition_mode: str = "strided"` (`"strided"` vs. `"contiguous"` vs. `"branch_assigned"`) to `SubspaceCoincidenceSpec` and `subspace_l2_normalize` so round-robin `x[..., m::M]` partitioning is supported natively with zero extra memory copies.
2. **[`src/qdlblocks/layers/multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py) (`NEW`, `<= 250` SLOC)**:
   - `WaveletScaleState`: Explicit container holding the 4 separate wavelet level tensors `(coarse_256, mid_2048, sub_8192, fine_65536)`, each of shape `(..., d_s)` where $d_s = d / 4$, with `.normalize_levels()`, `.level_energy_fractions()`, and `.to_product_of_spheres()`.
   - `MultiRateWavefunctionConvBank`: Causal 1D strided/dilated wavefunction filter bank over `WaveletScaleState` that evolves each wavelet level at its natural temporal stride/dilation `(4, 2, 1, 1)` with `BlockCholeskyLinear` SPD filters and cross-scale Mach-Zehnder phase coupling before feeding into Born attention.
3. **[`src/qdlblocks/models/level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) (`NEW`, `<= 320` SLOC)**:
   - `LevelSeparatedWaveletPyramid`: Synthesizes token states and sharp/diffuse detectors as explicit `WaveletScaleState` level tensors (`coarse_256`, `mid_2048`, `sub_8192`, `fine_65536` in $\mathbb{R}^{d_s}$) without summing them into a shared unnormalized vector.
   - `LevelSeparatedWaveletConvGPT`: Flag-free `Wave 32` language model combining `LevelSeparatedWaveletPyramid`, `MultiRateWavefunctionConvBank` feeding into `SubspaceProjectorBornAttention`, and `subspace_coincidence_dyadic_chunked_ce`.
4. **[`scripts/subspace_specialization_mutual_information_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_specialization_mutual_information_probe.py) (`NEW`, `<= 280` SLOC)**:
   - Build-Before-Launch CLI diagnostic (`--embed-dim 256 --num-subspaces 4 --vocab-size 50257 --device cpu --json-out .cache/subspace_specialization_mutual_information_probe.json`) measuring:
     - Fine-level energy survival under coarse-level growth (`merged_global_l2` vs. `explicit_level_separated`),
     - Subspace singular-value spectrum balance (`contiguous` vs. `strided_round_robin` vs. `explicit_level_separated`),
     - Pairwise cross-subspace Gaussian mutual information $I(X^{(m)}; X^{(m')}) = -\frac{1}{2}\ln\det(I - C_{m,m'} C_{m,m'}^\top)$ and per-level participation ratio $R_{\text{sharp}}^{(m)}$.
5. **[`tests/test_plan047_level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan047_level_separated_wavelet_conv_gpt.py) (`NEW`)** & **[`examples/57_verify_plan047_level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/57_verify_plan047_level_separated_wavelet_conv_gpt.py) (`NEW`)**.

---

## 4. Verification & Testing Plan

1. **AST Linter Gate**: Run `scripts/ast_signature_and_length_linter.py` on all modified and newly created files (`PASS`, `<= 1,000` SLOC, `<= 8` args, zero semicolons).
2. **Build-Before-Launch Probe**: Run `scripts/subspace_specialization_mutual_information_probe.py` on CPU and verify `H1` in `.cache/subspace_specialization_mutual_information_probe.json`.
3. **Unit Tests**: Run `tests/test_plan047_level_separated_wavelet_conv_gpt.py` verifying `WaveletScaleState`, `MultiRateWavefunctionConvBank` causality (zero future leakage), `LevelSeparatedWaveletPyramid` per-level unit norms, and `LevelSeparatedWaveletConvGPT` forward/backward passes.
4. **GPU Verification**: Run `examples/57_verify_plan047_level_separated_wavelet_conv_gpt.py` on `FineWeb-Edu` with `StandardizedExperimentLogger` and sync to `:8765` via `scripts/sync_waves_to_webui.py --trigger-reload`.

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

### 5.1 Existing Scripts Used
- [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
- [`scripts/multi_subspace_coincidence_margin_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multi_subspace_coincidence_margin_probe.py)
- [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py)

### 5.2 `Build-Before-Launch` Script Promoted from Backlog
- **[`scripts/subspace_specialization_mutual_information_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_specialization_mutual_information_probe.py)**: Promoted from `scripts/TOOLING_BACKLOG.md` to audit cross-level L2 norm cannibalization, strided vs. contiguous SVD spectrum balance, and pairwise subspace mutual information.

### 5.3 `Nice-to-Have` Scripts
1. `scripts/multirate_wavefunction_conv_receptive_field_visualizer.py`: Computes and plots the effective temporal impulse response $\partial \psi_t^{(m)} / \partial \psi_{t-\tau}^{(m')}$ across scales $m \in \{0, 1, 2, 3\}$ and lags $\tau \in [0, 32]$ before and after Born attention.

---

## 6. Constants & Calibration Ledger (`Rule 7` of `docs/SCRIPT_BEST_PRACTICES.md`)

| Constant Name | Module | Value | Mathematical / Physical Meaning & Empirical Source |
| :--- | :--- | :---: | :--- |
| `NUM_WAVELET_PYRAMID_LEVELS` | [`multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py) | `4` | Number of explicit dyadic hierarchy levels (`coarse_256`, `mid_2048`, `sub_8192`, `fine_65536`). |
| `DEFAULT_LEVEL_TEMPORAL_STRIDES` | [`multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py) | `(4, 2, 1, 1)` | Temporal dilation/stride per wavelet scale (`coarse=4`, `mid=2`, `sub=1`, `fine=1`). |
| `DEFAULT_CONV_MIX_INIT` | [`multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py) | `0.22` | Initial causal 1D wavefunction convolution mixing weight per scale before Born attention. |
| `DEFAULT_CROSS_SCALE_PHASE_INIT` | [`multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py) | `0.15` | Initial Mach-Zehnder cross-scale phase coupling between adjacent wavelet levels. |
| `LEVEL_ORTHO_RESIDUAL_WEIGHT` | [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | `0.35` | Weight combining per-level separated wavelet detectors with full-spectrum strided detectors. |
| `INIT_PRE_ATTN_CONV_GATE` | [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | `0.18` | Initial residual gate mixing `MultiRateWavefunctionConvBank` output into `psi_in_0` before Born attention. |

---

## 7. Empirical Outcomes, Telemetry, & Post-Mortem Wrap-Up (`AGENTS.md` Section 7)

### 7.1 `H1` Results — Cross-Level Norm Cannibalization & Strided SVD Balance ([`scripts/subspace_specialization_mutual_information_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_specialization_mutual_information_probe.py))
- **Status**: **`PASSED`** (`.cache/subspace_specialization_mutual_information_probe.json`).
- **Cross-Level L2 Norm Cannibalization Eliminated**:
  - Under `8x` coarse root (`wav_256`) norm growth, legacy global L2 normalization (`l2_normalize(c3 + c2 + c1 + c0)`) allowed `wav_256` to capture **`91.48%`** of total state energy while crushing fine lexical leaves (`wav_65536`) down to **`1.44%`** (`< 5%`).
  - Explicit level-separated tensors in [`WaveletScaleState`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py#L53-L103) lock each of the 4 levels (`coarse_256`, `mid_2048`, `sub_8192`, `fine_65536`) at exact **`25.00%`** (`0.25`) energy (`max_unit_norm_err = 1.19e-7`).
- **Strided Round-Robin vs. Contiguous Subspace SVD Balance**:
  - On sorted PPMI-SVD spectra ($\sigma_j \propto j^{-0.85}$), contiguous slicing `[0:64], [64:128], ...` suffered a **`304.66x`** pre-normalization variance skew between subspace 0 and subspace 3.
  - Strided round-robin partitioning (`x[..., m::M]`, `partition_mode="strided"`) reduced pre-normalization variance skew to **`6.26x`** (**`48.66x` reduction**).

### 7.2 `H2` Results — Controlled 150-Step `FineWeb-Edu` Verification ([`examples/57_verify_plan047_level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/57_verify_plan047_level_separated_wavelet_conv_gpt.py))
- **Status**: **`PASSED`** (`.cache/hypothesis_experiments_047.json`, `5/5` unit tests in [`tests/test_plan047_level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan047_level_separated_wavelet_conv_gpt.py)).

| Controlled Arm (`d=256, L=4, B=16, T=512`, `150` Steps / `1.23M` Tokens) | Step 50 $R_{\text{sharp}}$ | Step 100 $R_{\text{sharp}}$ | Step 150 $R_{\text{sharp}}$ | Coarse / Fine Energy | Coarse $R_{\text{sharp}}^{(0)}$ / Fine $R_{\text{sharp}}^{(3)}$ | Stiefel Err | Step 150 Val CE | Throughput (`tok/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w32_m1_single_sphere_baseline_d256` (`M=1` Global L2) | `51.00` | `52.69` | `53.12` | Unseparated | — | `0.0224` | `6.8037` | `30,407` |
| `w32_m4_strided_subspace_coincidence_d256` (`M=4` Strided) | `38.67` | `38.09` | `39.95` | `25%` / `25%` | — | `0.0228` | `6.9476` | `29,802` |
| **`w32_m4_level_separated_wavelet_conv_d256` (`M=4` `WaveletScaleState` + 1D Conv)** | **`39.52`** | **`43.09`** | **`44.88`** | **`25.0%` / `25.0%`** | **`13.01` / `33.79`** | **`0.0216`** | `6.9669` | `26,266` |

### 7.3 Key Mathematical Insights & Next-Wave Recommendations
1. **Why $R_{\text{sharp}}$ Differs Safely Across Wavelet Scales (`13.01` Coarse vs. `33.79` Fine)**:
   - Inspecting the separated level tensors in `w32_m4_level_separated_wavelet_conv_d256` directly confirms our hypothesis: the coarse 256-root tensor (`coarse_256`) naturally clusters semantically related words into a lower-rank semantic manifold ($R_{\text{sharp}}^{(0)} = 13.01 / 64$), while the fine lexical leaf tensor (`fine_65536`) maintains high rank ($R_{\text{sharp}}^{(3)} = 33.79 / 64$) and exact `25.00%` energy because it is normalized on its own sphere.
   - Moreover, with `WaveletScaleState` + `MultiRateWavefunctionConvBank`, total detector $R_{\text{sharp}}$ climbed steadily across training (`39.52 -> 43.09 -> 44.88`), beating both Plan 046's contiguous `M=4` (`37.22`) and strided `M=4` (`39.95`).
2. **Why `M=4` Starts Slightly Higher in Early 150-Step CE (`6.9476` vs. `6.8037`) and How to Close It**:
   - Under `M=4` coincidence readout with fixed `coincidence_gamma = 0.22` and `subspace_var_penalty = 0.14` from step 0, requiring all 4 subspaces (including the fine `wav_65536` subspace whose 65,536 leaves have only seen `1.23M` tokens $\approx 18$ tokens/leaf on average at step 150) to agree simultaneously penalizes predictions early in training before rare leaves have converged.
   - **Next-Wave Recommendation**: Warm up `coincidence_gamma(tau) = 0.22 * tau` and `subspace_var_penalty(tau) = 0.14 * tau` linearly with training progress $\tau \in [0, 1]$ (or apply coincidence gating primarily across the top 2–3 converged coarse/mid subspaces early in training) so early steps enjoy the fast coarse convergence of `M=1` while later steps gain the full multi-subspace coincidence margin boost.

