# QuantumDLBlocks Reusable Building Blocks & Kernel Catalog (`docs/BUILDING_BLOCKS.md`)

**Authoritative Agent & Researcher Reference for Composable Quantum-Geometric Primitives, Fused GPU Kernels, Stiefel Optimizers, and Diagnostic Probes (Current as of `Wave 37` / `EXPERIMENT_PLAN_052`).**

> **Mandatory Agent Instruction (See [`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md) Sections 5 & 6)**:
> 1. **Read Before Planning or Coding**: Every agent and subagent MUST consult this catalog before drafting a new `EXPERIMENT_PLAN_0XX.md` or writing model/layer/kernel code so that new experiments compose our proven, high-throughput building blocks instead of re-implementing them or reviving superseded legacy patterns.
> 2. **Follow Rule 6 ([`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md))**: Never thread new `use_*` or `is_*` boolean flags into frozen baseline models ([`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py), [`direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py)). Import the reusable building blocks below into a clean, flag-free per-experiment module.
> 3. **Contribute Back**: Whenever an experiment validates a general-purpose kernel, layer, optimizer schedule, or diagnostic script, promote it into `src/qdlblocks/{kernels,layers,optim,benchmarks}/` or `scripts/`, export it in `__init__.py`, and document it in this file and [`README.md`](file:///home/festivity/agy/QuantumDLBlocks/README.md).

---

## 1. Quick-Reference Status Matrix: What to Use vs. What Is Frozen/Legacy

| Subsystem | Canonical Reusable Component (Use This!) | Import Path | Status | Superseded / Frozen Counterpart (Do NOT Extend) |
| :--- | :--- | :--- | :---: | :--- |
| **Fused Readout & CE GPU Kernel** | `compiled_m4_coincidence_chunk_step`, `compiled_parseval_convex_traceless_chunk_step`, `fused_fp8_or_bf16_chunk_gemm`, `quantize_fp8_vocab_transposed` | `from qdlblocks.kernels import ...` ([`fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py)) | **CANONICAL (`Waves 28 & 34`)** | Unfused eager elementwise chunk steps or `float32` backward GEMMs |
| **Fused Attention GPU Kernel** | `flash_born_attention`, `flash_purified_born_attention` (supports both `head_dim = 64` and non-pow2 `head_dim in {48, 80, 96, 112}` via `BLOCK_D = next_power_of_2(HEAD_DIM)`) | `from qdlblocks.kernels import ...` ([`flash_born_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_born_triton.py)) | **CANONICAL (`Wave 34`)** | Materialized $O(B \cdot H \cdot T^2)$ attention matrices at $d \ge 384$ or $T \ge 1024$ |
| **Packed Complex Register Density & Krylov Attention** | `PackedComplexRegisterDensityAttention`, `ComplexRoPE`, `krylov_purified_density_probe`, `pack_hermitian_outer_product_and_ket` | `from qdlblocks.layers import ...` ([`packed_complex_register_density_attn.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/packed_complex_register_density_attn.py)) | **CANONICAL (`Wave 37`)** | Un-refined `Q-Probe` ($\rho_t^{\text{purified}} |q_t^{\text{rot}}\rangle$) or explicit $O(d_h^3)$ matrix squaring `rho @ rho` |
| **Unit-Sphere & Product-of-Spheres Readout** | `parseval_mup_dyadic_chunked_ce`, `subspace_coincidence_dyadic_chunked_ce`, `subspace_l2_normalize` (pair-strided `(..., d//8, 4, 2)`), `strided_subspace_l2_normalize`, `SubspaceCoincidenceSpec` | `from qdlblocks.layers import ...` ([`parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py), [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py)) | **CANONICAL (`Waves 28–34`)** | `direct_chunked_two_beam_kboson_ce` + `ContextualBlochPurityGate` + `kboson_raw_k` + manual clamps (`26.8` / `44.0`) |
| **Level-Separated Wavelets & 1D Conv Bank** | `WaveletScaleState` (pair-strided symplectic layout), `MultiRateWavefunctionConvBank` | `from qdlblocks.layers import WaveletScaleState, MultiRateWavefunctionConvBank` ([`multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py)) | **CANONICAL (`Waves 32–34`)** | Global L2 normalization over summed wavelet levels `l2_normalize(c3+c2+c1+c0)` that crushes fine leaves |
| **SPD Metric Conditioning** | `BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=True)` | `from qdlblocks.layers import BlockCholeskyLinear` ([`wavelet_cholesky_branch.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky_branch.py)) | **CANONICAL (`2.55x` BMM)** | `fast_spd_bmm=False` sequential Python `for b in range(num_blocks)` loop |
| **Multi-Scale Wavelet Branch** | `WaveletCholeskyMultiscaleBranch`, `WaveletCholeskyConfig` | `from qdlblocks.layers import WaveletCholeskyMultiscaleBranch` ([`wavelet_cholesky_branch.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky_branch.py)) | **CANONICAL** | Single-scale Euclidean MLPs |
| **Two-Beam Phase Interferometer** | `OvercompleteTwoBeamInterference` | `from qdlblocks.layers import OvercompleteTwoBeamInterference` ([`overcomplete_two_beam.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_two_beam.py)) | **CANONICAL** | Single-beam additive residuals without Mach-Zehnder phase gating |
| **Dyadic $2^{16}$ Vocab Pyramid** | `Tied2Pow16WaveletCholeskyPyramid`, `build_dyadic_2pow16_spectral_tree` (with $d_{\text{mid}}=d/2, d_{\text{fine}}=d/4$, `num_fine_up_banks=1`) | `from qdlblocks.models import Tied2Pow16WaveletCholeskyPyramid` ([`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py)) | **CANONICAL** | Fixed $(128, 64)$ ranks at $d > 256$ (rank-starved) or `16`-bank ambient synthesis |
| **Flag-Free Language Models** | `GeminiPackedComplexRoPEGPT` (`Wave 37` standalone pure-complex), `LevelSeparatedWaveletConvGPT` (`Waves 32–36`), `ParsevalMuPSubspaceCoincidenceGPT` (`Wave 31`), `ParsevalMuPNGramSpeedrunGPT` (`Waves 29–30`), `ParsevalMuPDyadicGPT` (`Wave 28`) | `from qdlblocks.models import GeminiPackedComplexRoPEGPT, LevelSeparatedWaveletConvGPT, ParsevalMuPSubspaceCoincidenceGPT, ParsevalMuPNGramSpeedrunGPT, ParsevalMuPDyadicGPT` ([`gemini_packed_complex_rope_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/gemini_packed_complex_rope_gpt.py), [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py)) | **CANONICAL (`Waves 28–37`)** | `Radical2Pow16WaveletCholeskyGPT` (frozen `Waves 20–27` baseline) & `QuantumBornGPT` (`Waves 1–19`) |
| **Fast-Loop Vision Ablation Lab** | `WaveletCholeskyVisionClassifier`, `TiedDyadicWaveletClassPyramid`, `load_or_download_imagenet1k_32x32` | `from qdlblocks.models import WaveletCholeskyVisionClassifier` ([`wavelet_cholesky_vision.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_vision.py)) | **CANONICAL (`~11s`/run)** | Slow multi-hour language runs for initial readout/layer screening |
| **Riemannian Stiefel Optimizer** | `StiefelMuon` (`polar_express`, `stiefel_penalty=0.02`, `use_normuon=True`, `use_twin_rail=True`, routes both 3D `kraus_generators` and 2D backbone projections) | `from qdlblocks.optim import StiefelMuon` ([`muon.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/muon.py)) | **CANONICAL (`Wave 34`)** | Hard periodic QR snapping or routing `kraus_generators` / 2D `qkv_proj` to Euclidean `AdamW` |
| **Experiment Logger & WebUI Ingestor** | `StandardizedExperimentLogger`, `ExperimentIntentSpec`, `ingest_all_waves_and_verifications` | `from qdlblocks.benchmarks import StandardizedExperimentLogger, ExperimentIntentSpec` ([`experiment_logger.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/experiment_logger.py)) | **CANONICAL (`Plan 044`)** | Ad-hoc unparsed JSON dumps without `[EXPERIMENT_INTENT]` or `ModelCard v1.0` |

---

## 2. Fused GPU Kernels (`src/qdlblocks/kernels/`)

Writing custom GPU kernels inside a single `examples/XX_*.py` script prevents reuse across waves. All custom Triton and `@torch.compile` (TorchInductor) + Tensor-Core kernels live in [`src/qdlblocks/kernels/`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/__init__.py) and are exported from `qdlblocks.kernels`.

### 2.1 Fused Parseval Convex-Traceless & 50,257-Way $M=4$ Coincidence Chunked CE Kernels ([`fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py))
- **Purpose**:
  - `compiled_parseval_convex_traceless_chunk_step` (`Wave 28`): Fuses the per-chunk $M=1$ vocabulary readout ($V_{\text{chunk}} = 4,096$) into two Tensor-Core GEMMs (`FP8 E4M3` via `torch._scaled_mm` or `BF16`) plus a single TorchInductor `@torch.compile` kernel that computes centered sharp & diffuse observables, rectified-quadratic target peak boost $\gamma \max(\hat{c}_{t,v}, 0)^2$, chunk partition sums, and exact analytical gradients $(g_{\text{sharp}}, g_{\text{diff}})$ in one SRAM pass.
  - `compiled_m4_coincidence_chunk_step` (`Wave 34` / `EXPERIMENT_PLAN_049.md`): Evaluates true 50,257-way $M=4$ Product-of-Spheres coincidence $\widehat{O}_{t,v}^{(4)} = \bar{c} + \gamma \widehat{Q} + \gamma_c \widehat{G} - \lambda_{\text{var}} \widehat{V}_{\text{sub}}$ and exact analytical per-subspace backward pullback `grad_s_m` `(4, C, V_act)` from batched `torch.bmm(psi_m[:, start:end], m_s_m.transpose(1, 2))` with identical `2 * C * d * V_act` GEMM FLOPs to $M=1$.
- **Measured Speedup**: Doubles end-to-end training throughput from `11,116 tok/s` to **`24,658–25,270 tok/s`** at $d=256$ (`2.86 GB` VRAM after `Wave 34` memory optimizations) while avoiding `(N_tok, 50257)` `float32` logit materialization.
- **Usage**:
  ```python
  from qdlblocks.kernels import (
      compiled_m4_coincidence_chunk_step,
      compiled_parseval_convex_traceless_chunk_step,
      fused_fp8_or_bf16_chunk_gemm,
      quantize_fp8_vocab_transposed,
  )
  ```

### 2.2 Fused Triton Flash-Born Attention ([`flash_born_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_born_triton.py))
- **Purpose**: Computes causal Born-rule attention $P(j \mid i) \propto \max(0, \beta_h \sqrt{d_h}\langle K_j \mid Q_i\rangle)^2$ and purified Born attention inside $64 \times 64$ GPU SRAM tiles with atomic-free backward passes (`dQ`, `dK`, `dV`). As of `Wave 34` (`EXPERIMENT_PLAN_049.md`), supports both canonical `head_dim = 64` (`num_heads = max(2, d // 64)`) and non-power-of-2 `head_dim in {48, 80, 96, 112}` via `BLOCK_D = triton.next_power_of_2(HEAD_DIM)` column masking, eliminating the `+2.75 GiB` PyTorch fallback across all widths $d \in \{256, 384, 512, 640, 768, 896, 1024\}$.
- **Usage**:
  ```python
  from qdlblocks.kernels import flash_born_attention, flash_purified_born_attention
  ```

---

## 3. Core Quantum-Geometric Layers & Readouts (`src/qdlblocks/layers/`)

### 3.1 Flag-Free Parseval $\mu\text{P}$ $\sqrt{d}$ Readout (`parseval_mup_chunked_ce.py`)
- **Mathematical Formulation (`Waves 24–28`)**:
  On the unit sphere $\mathbb{S}^{d-1}$, a tight Parseval frame $\{m_v\}_{v=1}^V$ has background inner-product standard deviation $\sigma = 1/\sqrt{d}$ while a true aligned target has $\hat{c}_{t,y} = \mathcal{O}(1)$. Instead of a dynamic per-token `PurityGate` MLP, redundant `kboson_raw_k` mode counter, and manual temperature clamps (`26.8` / `44.0`), [`parseval_mup_dyadic_chunked_ce`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py) uses the deterministic Parseval $\mu\text{P}$ scale $\beta_0(d) = c^*\sqrt{d}$ ($c^* = 1.48$) with the **Convex-Peak Traceless Born Observable** and **256-Root Dyadic Coarse-to-Fine Conditional Readout**:
  $$\widehat{O}_{t,v} = \hat{c}_{t,v} + \gamma\left(\max(\hat{c}_{t,v}, 0)^2 - \tau_{t,+}\right), \qquad \ell_{t,v} = \beta(\tau)\widehat{O}_{t,v} + \tilde{\lambda}_{\text{diff}}(\tau)\hat{f}_{\text{diff}, t, v} + b_v + w_{\text{coarse}}\ln P_{\text{coarse}}(g(v) \mid x_t)$$
- **Post-Normalization Cholesky Temperature Absorption**:
  Via `model.absorb_beta_into_readout_cholesky()`, the scale $\beta_0(d)^{1/2}$ can be absorbed directly into the positive-definite Cholesky factors $L \leftarrow \beta_0(d)^{1/4} L$ of `readout_chol` applied *after* unit-sphere/subspace normalization (exact equivalence `< 7.6e-6`).
- **Usage**:
  ```python
  from qdlblocks.layers import (
      ParsevalChunkedCEReadoutSpec,
      eval_parseval_mup_full_logits,
      parseval_mup_dyadic_chunked_ce,
  )
  ```

### 3.2 Batched SPD `BlockCholeskyLinear` & Sweldens `WaveletCholeskyMultiscaleBranch` (`wavelet_cholesky_branch.py`)
- **`BlockCholeskyLinear(embed_dim, max_block_size=32, fast_spd_bmm=True)`**:
  Applies a block-diagonal Symmetric Positive-Definite (SPD) operator $A = \operatorname{diag}(L_1 L_1^\top, \dots, L_G L_G^\top) \succ 0$ where each $L_g$ is lower-triangular with positive diagonal $\operatorname{softplus}(\text{raw\_diag}_{g,i}) + 10^{-4}$. Always pass `fast_spd_bmm=True` to use the fused batched `torch.bmm` path (`2.55x` faster).
- **`WaveletCholeskyMultiscaleBranch(WaveletCholeskyConfig(...))`**:
  Decomposes feature channels across dyadic octave scales via invertible Sweldens lifting (`split -> predict -> update`), applies scale-specific `BlockCholeskyLinear` operators, and reconstructs via inverse lifting.
- **Usage**:
  ```python
  from qdlblocks.layers import (
      BlockCholeskyLinear,
      RMSNorm,
      WaveletCholeskyConfig,
      WaveletCholeskyMultiscaleBranch,
      apply_soft_powerlaw_spectral_floor,
  )
  ```

### 3.3 Mach-Zehnder Two-Beam Phase Interferometer (`overcomplete_two_beam.py`)
- **`OvercompleteTwoBeamInterference`**:
  Splits features into two coherent beams that interfere through learnable $2 \times 2$ unitary Mach-Zehnder beamsplitters $\begin{pmatrix}\cos\theta & \sin\theta \\ -\sin\theta & \cos\theta\end{pmatrix}$ and cosine relative-phase gates before re-normalizing onto $\mathbb{S}^{d-1}$. On genuine `ILSVRC ImageNet-1K` (`50,000` official validation images), ablating this layer degrades Validation Cross-Entropy by `+0.2025` nats (`8.49% -> 7.56%` Top-1).

### 3.4 Product-of-Spheres $(\frac{1}{\sqrt{M}}\mathbb{S}^{d/M - 1})^M$ & Multi-Subspace Coincidence Born Readout (`subspace_coincidence_born.py` — `Waves 31–34` / `EXPERIMENT_PLAN_046.md` – `EXPERIMENT_PLAN_049.md`)
- **When to Use It**: Whenever additive branches (`wav_256`, `readout_ngram`, or coarse-vs-fine wavelet levels) compete inside a single global L2 normalization and cause the sharp detector participation ratio $R_{\text{sharp}}$ to collapse (e.g. `77.52 -> 24.18` in `Wave 30`), or when single-subspace positive runner-up impostors ($Z_{\text{pos\_runnerup}}$) inflate Cross-Entropy at $V \gg d$.
- **Why It Was Built (`EXPERIMENT_PLAN_046.md` – `EXPERIMENT_PLAN_049.md`)**:
  - `subspace_l2_normalize(x, num_subspaces=M, partition_mode=...)` and `strided_subspace_l2_normalize(x, num_subspaces=M)` project states and detectors onto $\mathcal{M}_{d, M} = (\frac{1}{\sqrt{M}}\mathbb{S}^{d/M - 1})^M \subset \mathbb{S}^{d-1}$ using the canonical **pair-strided symplectic layout** `(*lead, d // (2 * M), M, 2)` (`Wave 34`), preserving both exact unit norm $\|x\|_2 = 1$ and exact even/odd `(a_self, b_orth)` symplectic pair orthogonality while locking equal $1/M$ energy in each subspace.
  - `compute_coarse_cluster_centers` (`Wave 34`) forms root-consistent 256 coarse centroids from non-cancelling Subspace 0 (`COARSE_ROOT_SUBSPACE_BLEND = 0.85`), eliminating the `2.10`-nat detail-cancellation floor (`< 0.02` nats at $\alpha = 1.0$).
  - `subspace_coincidence_dyadic_chunked_ce` and `eval_subspace_coincidence_full_logits` (`Wave 34`) support full 50,257-way $M=4$ fine coincidence via `compiled_m4_coincidence_chunk_step` (`use_fine_m4_chunked_coincidence=True`) with exact train/eval parity.
- **Usage**:
  ```python
  from qdlblocks.layers import (
      SubspaceCoincidenceSpec,
      compute_multi_subspace_coincidence_observable,
      eval_subspace_coincidence_full_logits,
      strided_subspace_l2_normalize,
      subspace_coincidence_dyadic_chunked_ce,
      subspace_l2_normalize,
  )
  ```

### 3.5 Explicit Level-Separated Wavelet Tensors (`WaveletScaleState`) & Pre-Attention Multi-Rate 1D Wavefunction Convolution Bank (`multirate_wavefunction_conv.py` — `Waves 32–34` / `EXPERIMENT_PLAN_047.md` – `EXPERIMENT_PLAN_049.md`)
- **When to Use It**: Whenever synthesizing multi-scale wavelet representations (`coarse_256`, `mid_2048`, `sub_8192`, `fine_65536`) so that each wavelet level is kept as a distinct tensor in code (`WaveletScaleState`) normalized independently onto $\frac{1}{\sqrt{4}}\mathbb{S}^{d/4 - 1}$ (`25.0%` energy per level) and interleaved via pair-strided symplectic pairs `(*lead, d // 8, 4, 2)`, and when evolving token wavefunctions inside every backbone layer via causal 1D strided/dilated wavefunction convolutions (`MultiRateWavefunctionConvBank`, temporal dilations `(4, 2, 1, 1)`) *before* feeding into Quantum Born Attention.
- **Usage**:
  ```python
  from qdlblocks.layers import MultiRateWavefunctionConvBank, WaveletScaleState
  ```

### 3.6 Packed Complex Register Density Attention & Associative V-Space Krylov Probe (`packed_complex_register_density_attn.py` — `Wave 37` / `EXPERIMENT_PLAN_052.md`)
- **When to Use It**: Whenever operating on pure complex Hilbert space registers $|\psi_{b,h,t}\rangle \in \mathbb{C}^{d_h}$ ($d_h = 32$) where context is accumulated as a Hermitian density operator $\rho_t = \sum_{s \le t} w_{ts} |v_s\rangle\langle v_s| \in \mathbb{C}^{32 \times 32}$ and purified via McWeeny purification $\rho_t^{\text{purified}} = \rho_t^2 / \operatorname{Tr}(\rho_t^2)$.
- **Why It Was Built (`EXPERIMENT_PLAN_052.md` — Refinements 1, 3, & 4)**:
  1. **Refinement 1 (Translation-Equivariant `ComplexRoPE` on $Q, K$ Only)**: Rotates $Q$ and $K$ across all 32 complex frequencies while keeping $V$ and the probe in position-invariant content space (`max_rope_shift_err < 3.8e-8`).
  2. **Refinement 3 (Single-Pass `torch.where` 1-GEMM Hermitian + Ket Packing & $O(d_h^2)$ Associative Krylov Evaluation)**: Packs upper-triangular symmetric $\rho_{\text{re}}$, strict-lower-triangular skew-symmetric $\rho_{\text{im}}$, and ket $[v_{\text{re}}, v_{\text{im}}]$ (`width = 1088` at $d_h = 32$) into a single `torch.matmul(w_ts, packed_v)` and evaluates $\rho_t(\rho_t |\bar{v}_t\rangle) / (\|\rho_{\text{re},t}\|_F^2 + \|\rho_{\text{im},t}\|_F^2 + \varepsilon)$ in two $O(d_h^2)$ matrix-vector products (`16x` fewer purification FLOPs than explicit $O(d_h^3)$ matrix squaring, `< 7.5e-8` max error).
  3. **Refinement 4 (`V-Space Krylov Probe` $|\bar{v}_t\rangle$)**: Probing $\rho_t^{\text{purified}}$ with the V-space first-moment anchor $|\bar{v}_t\rangle = \operatorname{normalize}_{\mathbb{C}}(\sum_{s \le t} w_{ts} |v_s\rangle)$ instead of $|q_t\rangle$ eliminates Step-0 $W_Q \perp W_V$ random phase scrambling (`6.45x` higher Step-0 probe norm, `-0.0503` nats / `-28.0` PPL better on `FineWeb-Edu`).
- **Usage**:
  ```python
  from qdlblocks.layers import (
      ComplexLinear,
      ComplexRoPE,
      PackedComplexRegisterDensityAttention,
      krylov_purified_density_probe,
      pack_hermitian_outer_product_and_ket,
      unpack_hermitian_matrix,
  )
  ```

---

## 4. Reference Architectures & Dyadic Pyramids (`src/qdlblocks/models/`)

### 4.1 Flag-Free Parseval $\mu\text{P}$ Dyadic GPT (`parseval_mup_dyadic_gpt.py` — `Wave 28` / `EXPERIMENT_PLAN_043.md`)
- **What It Is**: The clean, flag-free `Wave 28` language model (`zero` `use_*`/`is_*` flags, zero `PurityGate`, zero `kboson_raw_k`, zero clamps).
- **Proportional Dyadic Wavelet Ranks (`Wave 27` & `Wave 28` Discovery)**:
  In `Tied2Pow16WaveletCholeskyPyramid`, vocabulary words ($V = 50,257 \subset N = 2^{16} = 65,536$) are synthesized coarse-to-fine from `wav_256`, `wav_2048`, `wav_8192`, and `wav_65536`. When scaling model width $d > 256$, **always scale wavelet detail ranks proportionally** ($d_{\text{mid}} = d // 2, d_{\text{fine}} = d // 4$, which `ParsevalMuPDyadicConfig` sets automatically) so the vocabulary frame's effective rank $R_{\text{eff}}(M)$ scales with $d$ rather than bottlenecking at $(128, 64)$.
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      ParsevalMuPDyadicConfig,
      ParsevalMuPDyadicGPT,
      build_parseval_mup_optimizers,
      initialize_parseval_mup_with_ppmi_svd,
  )

  cfg = ParsevalMuPDyadicConfig(embed_dim=384, num_heads=6, num_layers=6, readout_gemm_precision="fp8")
  model = ParsevalMuPDyadicGPT(cfg).to(device)
  init_stats = initialize_parseval_mup_with_ppmi_svd(model, ppmi_emb_256, init_tokens, prior_tokens)
  muon_opt, adam_opt = build_parseval_mup_optimizers(model, base_muon_lr=0.026, base_wav_lr=0.0065, base_head_lr=0.0035)
  ```

### 4.2 Fast-Loop Vision Ablation Laboratory (`wavelet_cholesky_vision.py`)
- **What It Is**: `WaveletCholeskyVisionClassifier` + `TiedDyadicWaveletClassPyramid` on cached GPU tensors for `CIFAR-100` ($K=100, N=128$), `1,000-Class Fine-Grained` ($K=1,000, N=1,024$), and genuine `ILSVRC ImageNet-1K` ($K=1,000$ WordNet classes, `1,281,167` train / `50,000` val / `100,000` test across $32 \times 32$ and $64 \times 64$ multi-level 2D Sweldens wavelets).
- **When to Use It**: Whenever you want to test a new quantum building block, interferometer, Kraus bath, or readout equation in **`10–12 seconds` per full training run** before committing to a multi-minute `FineWeb-Edu` language model run.

### 4.3 Flag-Free Parseval $\mu\text{P}$ + Hashed N-Gram + Cross-Layer $V^{(0)}$ Speedrun GPT (`parseval_mup_ngram_speedrun_gpt.py` — `Waves 29–34` / `EXPERIMENT_PLAN_045.md` & `EXPERIMENT_PLAN_049.md`)
- **What It Is**: Unifies `Wave 28`'s flag-free Parseval $\mu\text{P}$ $\beta_0(d) = 1.48\sqrt{d}$ + Convex-Peak Traceless Born + 256-Root Dyadic Hierarchy + FP8 fused readout with `HashedSurfaceNGramProjector` and Cross-Layer $V^{(0)}$ value residuals. As of `Wave 34` (`EXPERIMENT_PLAN_049.md`), `build_parseval_mup_ngram_optimizers` routes all 2D backbone matrices (`qkv_proj`, `out_proj`, `interferometer`) and 3D `kraus_generators` to `StiefelMuon` (`retract_every=2`), excludes `conv_bank.*.diag_raw/tril_raw` from weight decay (`_CHOL_KEYS`), and scales sparse fine wavelet tables (`wav_65536`, `wav_8192`) in `wavelet_fine_sparse` at `FINE_WAVELET_SPARSE_LR = 0.0156 * sqrt(256 / d)`.
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      HashedSurfaceNGramProjector,
      ParsevalMuPNGramSpeedrunConfig,
      ParsevalMuPNGramSpeedrunGPT,
      build_parseval_mup_ngram_optimizers,
      initialize_parseval_mup_ngram_with_ppmi_svd,
  )
  ```

### 4.4 Flag-Free Product-of-Spheres Multi-Subspace Coincidence GPT (`parseval_mup_subspace_coincidence_gpt.py` — `Wave 31` / `EXPERIMENT_PLAN_046.md`)
- **What It Is**: Extends `ParsevalMuPNGramSpeedrunGPT` with Product-of-Spheres $\mathcal{M}_{d, M} = (\frac{1}{\sqrt{M}}\mathbb{S}^{d/M - 1})^M$ detector & pre-readout normalization (`partition_mode="contiguous"` or `"strided"`) and 256-root multi-subspace coincidence gating (`SubspaceCoincidenceSpec`).

### 4.5 Flag-Free Level-Separated Wavelet + Per-Layer Pre-Attention 1D Wavefunction Convolution GPT (`level_separated_wavelet_conv_gpt.py` — `Waves 32–34` / `EXPERIMENT_PLAN_047.md` – `EXPERIMENT_PLAN_049.md`)
- **What It Is**: Separates the 4 dyadic wavelet pyramid levels (`coarse_256`, `mid_2048`, `sub_8192`, `fine_65536`) into distinct tensors in `WaveletScaleState` using symplectic-pair-preserving folding (`_fold_to_level_dim`), evolves them inside **every backbone block (`PreAttnMultiRateWaveletConvBlock`)** via pre-attention `MultiRateWavefunctionConvBank` -> `SubspaceProjectorBornAttention` (with $V^{(0)}$ value residual and canonical `head_dim = 64`) -> post-attention `WaveletCholeskyMultiscaleBranch` + causal coarse shift + `OvercompleteTwoBeamInterference`, and checkpoints `synthesize_detectors()` during training so $d \in [256, 1024]$ fits cleanly in 16 GB VRAM.
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      LevelSeparatedWaveletConvConfig,
      LevelSeparatedWaveletConvGPT,
      PreAttnMultiRateWaveletConvBlock,
      build_level_separated_wavelet_conv_optimizers,
      initialize_level_separated_wavelet_conv_with_ppmi_svd,
      synthesize_level_separated_wavelet_state,
  )
  ```

### 4.6 Standalone Pure-Complex Register Density GPT (`gemini_packed_complex_rope_gpt.py` — `Wave 37` / `EXPERIMENT_PLAN_052.md`)
- **What It Is**: Standalone pure-complex autoregressive language model operating on $H$ quantum registers of dimension $d_h = 32$ complex amplitudes ($\mathbb{C}^{d_{\text{complex}}}$) with `PackedComplexRegisterDensityAttention` (`V-Space Krylov Probe`, Refinements 1, 3, & 4), `ComplexUnitaryInterferenceFFN`, and token-chunked complex Born readout. On `FineWeb-Edu` (`300` steps / `2.46M` tokens at $d_{\text{complex}} = 128, H = 4, d_h = 32 \implies 256$ real DoF/token, $L = 6$), `GeminiPackedComplexRoPEGPT` (`v_krylov`) reached **`6.2951` Val CE (`541.89` PPL)**, beating both the original `Q-Probe` (`6.3454` Val CE, `-0.0503` nats) and `Wave 36` `Option 2` (`6.3157` Val CE, `-0.0206` nats).
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      GeminiPackedComplexRoPEConfig,
      GeminiPackedComplexRoPEGPT,
      build_gemini_complex_optimizers,
      initialize_gemini_complex_gpt_with_ppmi_svd,
  )
  ```

---

## 5. Riemannian Stiefel Optimization & Learning-Rate Schedules (`src/qdlblocks/optim/`)

### 5.1 `StiefelMuon` & Mandatory Routing for `kraus_generators` and 2D Backbone Projections
- **Critical Lessons (`Wave 22` & `Wave 34`)**:
  1. `pyramid.kraus_generators` has shape `(R, d, d)` (`ndim == 3`). Never let a substring filter route `kraus_generators` to Euclidean `AdamW(weight_decay=0.0)`, or its Stiefel error $\|K_r^\top K_r - I\|_F / \sqrt{d}$ will drift to `> 20.0`.
  2. Never omit the 2D `p.ndim == 2 and min(p.shape) >= 32 and max(p.shape) <= 2048` branch when building `StiefelMuon` parameter groups (`Bug 1` in `EXPERIMENT_PLAN_049.md`), or `qkv_proj`, `out_proj`, and `interferometer` matrices fall into `AdamW` at `7.4x` lower LR without Polar Express orthogonalization.
- **Canonical Setup**: Route both 3D `kraus_generators` and 2D orthogonal projection matrices to `StiefelMuon(..., stiefel_penalty=0.02, ortho_method="polar_express", retract_every=2, use_normuon=True, use_twin_rail=True)` and call `model.retract_kraus_generators()` every 2 steps, which keeps `kraus_stiefel_err <= 0.015` smoothly with zero momentum shocks.

### 5.2 Choosing Between NanoGPT `WSD` and Smith 2-Cycle `CLR` (`arXiv:1506.01186`)
- **Fixed-Horizon Runs (`modded_nanogpt_wsd`)**: Best when the total token horizon $T_{\max}$ is fixed in advance and only the final checkpoint at $\tau = 1.00$ matters (`5.4372` at $d=256$, `4.7919` at $d=512$ / `29.49M` tokens). Warning: uncooled intermediate checkpoints at $\tau \in [0.04, 0.60]$ sit at peak LR $\eta_{\max}$ and appear `+0.15` nats worse until the cooldown branch is run (`scripts/isotoken_wsd_cooldown_branch_evaluator.py`).
- **Anytime / Multi-Cycle Runs (`smith_triangular2` with Iso-Integral Scaling)**: Use [`scripts/smith_clr_range_test_vs_wsd_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/smith_clr_range_test_vs_wsd_profiler.py) to find $(\eta_{\text{base}}, \eta_{\max})$ automatically in 40 steps, and use the iso-integral 2-cycle schedule when you need strong calibrated checkpoints at **both** $\tau = 0.50$ (`-0.0452` nats better than uncooled `WSD`) and $\tau = 1.00$.

---

## 6. First-Class Diagnostic & Verification Scripts (`scripts/`)

Before writing any ad-hoc one-liner, check if one of our deterministic CLI tools in `scripts/` already performs the diagnostic:

| Script | Purpose & Example Invocation |
| :--- | :--- |
| [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) | Enforces Rules 1–8 (`<= 1,000` soft / `<= 2,500` hard comment-free SLOC lines, `<= 8` args, zero semicolons, max nested function SLOC). |
| [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py) | Normalizes all `.cache/model_cards/` and `.cache/live_logs/` into `ModelCard v1.0`, syncs `.cache/benchmarks.db` across all 3 WebUI tabs, and reloads `http://127.0.0.1:8765`. |
| [`scripts/subspace_energy_and_rank_drift_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_energy_and_rank_drift_tracker.py) | Audits per-subspace energy fractions `energy_coarse_256..energy_fine_65536`, per-subspace participation ratios `r_sharp_coarse_256..r_sharp_fine_65536`, global `sharp_eff_rank` (`centered_r_sharp`), effective temperature `mean_beta`, and `coincidence_gamma_eff` drift across milestones (`EXPERIMENT_PLAN_049.md`). |
| [`scripts/factorized_256x256_hierarchical_softmax_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/factorized_256x256_hierarchical_softmax_bench.py) | Computes the exact 4-shell (`S_sib,8`, `S_mid,32`, `S_coarse,256`, `S_bg`) dyadic CE floor $\mathcal{L}_{\text{CE}}^{\text{shell}}(d, M, \alpha)$, verifies Subspace-0 root-consistent coarse centroids (`2.10 -> < 0.01` nats), and benchmarks `factorized_256x256_coincidence_ce` (`512` dot products/token, `98.16x` fewer GEMM FLOPs) (`EXPERIMENT_PLAN_049.md`). |
| [`scripts/max_width_vram_and_flash_born_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/max_width_vram_and_flash_born_profiler.py) | Profiles parameter budgets, Flash-Born support (`hd = 64` and `H = 8`), saved detector autograd memory (`MiB`), peak step VRAM (`MiB`), and throughput (`tok/s`) across $d \in \{256, 384, 512, 640, 768, 1024\}$ on the 16 GB GPU (`EXPERIMENT_PLAN_049.md`). |
| [`scripts/multirate_wavefunction_conv_receptive_field_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multirate_wavefunction_conv_receptive_field_profiler.py) | Profiles per-layer pre-attention temporal impulse response across lags `0..16`, strict causality (`max_future_leakage == 0`), and progress-ramped coincidence warmup (`coincidence_gamma(tau) = 0.22 * tau`) (`EXPERIMENT_PLAN_048.md`). |
| [`scripts/subspace_specialization_mutual_information_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_specialization_mutual_information_probe.py) | Audits cross-level L2 norm cannibalization (`1.44%` vs. `25.00%` fine `wav_65536` energy under `8x` coarse growth), contiguous vs. strided SVD variance skew (`304.66x` vs. `6.26x`), and pairwise cross-subspace Gaussian mutual information (`EXPERIMENT_PLAN_047.md`). |
| [`scripts/multi_subspace_coincidence_margin_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multi_subspace_coincidence_margin_probe.py) | Audits Product-of-Spheres $R_{\text{sharp}}$ anti-collapse (`8.30x` at $d=512, M=4$) and single-subspace runner-up impostor CE tax suppression (`-1.3414` nats) (`EXPERIMENT_PLAN_046.md`). |
| [`scripts/sub3p28_at_150m_and_900m_feasibility_analyzer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sub3p28_at_150m_and_900m_feasibility_analyzer.py) | Analyzes the 4 mathematical & engineering levers to reach `< 3.28` Val CE at `900M` and `150M` tokens. |
| [`scripts/fineweb_150m_300m_900m_scaling_projector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_150m_300m_900m_scaling_projector.py) | Fits Chinchilla scaling laws across `Waves 11–17` (`900M` FineWeb) and `Waves 21–30` (`ParsevalMuPNGramSpeedrunGPT`) to evaluate `50M`, `100M`, `150M`, `300M`, `450M`, and `900M` token trajectories. |
| [`scripts/sqlite_benchmark_query_cli.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sqlite_benchmark_query_cli.py) | Queries `.cache/benchmarks.db` by `--domain`, `--wave`, `--horizon-m`, and `--pareto-only` to emit Markdown/JSON leaderboards. |
| [`scripts/webui_static_html_exporter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/webui_static_html_exporter.py) | Exports a self-contained offline HTML snapshot of the 3-tab QuantumDLBlocks WebUI into `.cache/webui_snapshot_export.html`. |
| [`scripts/fineweb_convex_vs_concave_slope_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_convex_vs_concave_slope_tracker.py) | Standardized **Iso-Token (`2.46M`, `7.37M`, `14.75M`, `29.49M`, `44.24M`) & Iso-FLOP (`6 N D`)** comparison table and joint Chinchilla scaling-law fitter across all waves (`w19..w28`). |
| [`scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py) | Verifies empirically on real `FineWeb-Edu` sequences that the $N=2^{16}$ Wavelet-Cholesky pyramid breaks the `3.28` Val CE wall (`< 0.10` nats) and audits proportional vs. fixed wavelet ranks across $d \in \{256, 384, 512\}$. |
| [`scripts/smith_clr_range_test_vs_wsd_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/smith_clr_range_test_vs_wsd_profiler.py) | Runs Leslie Smith's (`arXiv:1506.01186`) exponential LR Range Test on `ParsevalMuPDyadicGPT` and compares `modded_nanogpt_wsd`, `smith_triangular2`, and `smith_1cycle`. |
| [`scripts/parseval_sqrt_d_scale_invariance_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_sqrt_d_scale_invariance_probe.py) | Verifies $\beta_0(d) = c^*\sqrt{d}$ background logit variance invariance across $d \in \{128, 256, 384, 512, 768, 1024\}$ and two-beam algebraic scale unification. |
| [`scripts/unit_sphere_mup_transfer_grid_search.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/unit_sphere_mup_transfer_grid_search.py) | Verifies zero-shot $\mu\text{P}$ transfer of $c^* = 1.48$ from a tiny $d_{\text{base}} = 128$ proxy to $d \in \{256, 384, 512, 768, 1024\}$. |
| [`scripts/cholesky_metric_temperature_absorber.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/cholesky_metric_temperature_absorber.py) | Verifies exact mathematical equivalence (`< 1e-5`) of absorbing $\beta_0(d)^{1/2}$ into post-normalization `BlockCholeskyLinear` factors $L \leftarrow \beta_0(d)^{1/4} L$. |
| [`scripts/imagenet1k_quantum_block_ablation_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/imagenet1k_quantum_block_ablation_profiler.py) | Automated leave-one-out ablation runner for quantum building blocks on genuine `ILSVRC ImageNet-1K` (`50,000` val images). |

---

## 7. How to Contribute Back ("Promote-to-Primitive" Protocol & Rules 7–8)

Whenever you complete an experiment wave or implement a new capability, follow this 5-step checklist to keep `QuantumDLBlocks` modular, self-documenting, and discoverable for the next agent:

1. **Extract Reusable Kernels & Layers Out of `examples/`**:
   - If a custom `@torch.compile` function, Triton kernel, or `torch.autograd.Function` is useful beyond a single verification script, place it in [`src/qdlblocks/kernels/`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/__init__.py) or [`src/qdlblocks/layers/`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/__init__.py) and export it in `__all__`.
2. **Define All Constants at the Top of the File & Sync with `EXPERIMENT_PLAN_0XX.md` (Rule 7)**:
   - Never leave inline magic constants inside functions; define them in `UPPER_SNAKE_CASE` at the top of the file with a comment explaining their meaning and update the `EXPERIMENT_PLAN_0XX.md` Constants Ledger.
3. **Comment Thoroughly with Experiment Provenance (Rule 8 — Comments Are Free of Line Limits!)**:
   - Because [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) excludes comments and docstrings from line limits, write detailed docstrings and inline comments explaining *when* the building block is useful, *why* it was built mathematically, and *which wave/plan (`Wave XX` / `EXPERIMENT_PLAN_0XX.md`)* validated it.
4. **Keep Per-Wave Architectures Clean & Flag-Free (Rule 6)**:
   - Place new self-contained model formulations in `src/qdlblocks/models/` and export them in [`src/qdlblocks/models/__init__.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/__init__.py) without adding boolean toggle flags to earlier wave models.
5. **Update This Catalog (`docs/BUILDING_BLOCKS.md`) and `README.md`**:
   - Add the new component to the Quick-Reference Status Matrix in Section 1 above (marking any superseded component as `FROZEN` or `LEGACY`) so the very next agent session knows immediately what state the building blocks are in.

