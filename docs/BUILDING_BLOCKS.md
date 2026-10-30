# QuantumDLBlocks Reusable Building Blocks & Kernel Catalog (`docs/BUILDING_BLOCKS.md`)

**Authoritative Agent & Researcher Reference for Composable Quantum-Geometric Primitives, Fused GPU Kernels, Stiefel Optimizers, and Diagnostic Probes (Current as of `Wave 47` / `EXPERIMENT_PLAN_062`).**

> **Mandatory Agent Instruction (See [`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md) Sections 5 & 6, [`.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md`](file:///home/festivity/agy/QuantumDLBlocks/.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md), and [`docs/designs/SITEMAP.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/designs/SITEMAP.md))**:
> 1. **Read Before Planning or Coding**: Every agent and subagent MUST consult this catalog before drafting a new `EXPERIMENT_PLAN_0XX.md` or writing model/layer/kernel code so that new experiments compose our proven, high-throughput building blocks instead of re-implementing them or reviving superseded legacy patterns.
> 2. **Follow Rules 6 & 9 ([`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md))**: Never thread new `use_*` or `is_*` boolean flags into frozen baseline models ([`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py), [`direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py)). Import the reusable building blocks below into a clean, flag-free per-experiment module, and ensure all custom `torch.autograd.Function` backward passes compute all input gradients in a single chunk pass.
> 3. **Contribute Back**: Whenever an experiment validates a general-purpose kernel, layer, optimizer schedule, or diagnostic script, promote it into `src/qdlblocks/{kernels,layers,optim,benchmarks}/` or `scripts/`, export it in `__init__.py`, and document it in this file and [`README.md`](file:///home/festivity/agy/QuantumDLBlocks/README.md).

---

## 1. Quick-Reference Status Matrix: What to Use vs. What Is Frozen/Legacy

| Subsystem | Canonical Reusable Component (Use This!) | Import Path | Status | Superseded / Frozen Counterpart (Do NOT Extend) |
| :--- | :--- | :--- | :---: | :--- |
| **Fused Readout & CE GPU Kernel** | `FastCalibratedKroneckerBornChunkedNLLFn`, `evaluate_fast_modded_nanogpt_validation` (`Wave 44` single-pass backward + compiled pre-transposed BF16 `16,384`-chunk evaluator), `CalibratedKroneckerBornChunkedNLLFn`, `eval_calibrated_kronecker_born_probs` (`Waves 42–43`), `KroneckerMultiMeaningBornChunkedNLLFn` (`Wave 41`), `TiedHoloBornChunkedNLLFn` (`Wave 40`), `compiled_m4_coincidence_chunk_step`, `compiled_parseval_convex_traceless_chunk_step` | `from qdlblocks.models import CalibratedKroneckerBornChunkedNLLFn, eval_calibrated_kronecker_born_probs`; `from qdlblocks.kernels import ...` | **CANONICAL (`Waves 28, 34, 40–44`)** | `_ComplexBornChunkedCEFn` (Boltzmann `exp`, Real-part-only overlap, $1/\sqrt{2}$ double-shrinkage, `50,257`-row `meas_proj`) & 2-pass `d_povm` recomputation |
| **Fused Attention GPU Kernel** | `flash_born_attention`, `flash_purified_born_attention` (supports both `head_dim = 64` and non-pow2 `head_dim in {48, 80, 96, 112}` via `BLOCK_D = next_power_of_2(HEAD_DIM)`) | `from qdlblocks.kernels import ...` ([`flash_born_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_born_triton.py)) | **CANONICAL (`Wave 34`)** | Materialized $O(B \cdot H \cdot T^2)$ attention matrices at $d \ge 384$ or $T \ge 1024$ |
| **Fused Complex Norm-RoPE & Associative Krylov Triton Kernels** | `fused_complex_norm_rope_triton` (all $8 \le D \le 2048$), `fused_causal_pboson_born_triton`, `flash_complex_krylov_attention` (`probe_mode="v_krylov_norm"`), `factored_complex_krylov_step` | `from qdlblocks.kernels import ...` ([`flash_complex_krylov_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_complex_krylov_triton.py)) | **CANONICAL (`Waves 38–39`)** | Materializing $(B, H, T, 1088)$ / $(B, H, T, 32, 32)$ Hermitian density tensors or $O(T^3)$ `w_gemm @ g_sq` McWeeny trace GEMMs when `normalize_krylov_ket=True` |
| **All-Layer Riemannian Impure Density GPT & Multi-Layer Grafting** | `RiemannianImpureDensityGPT`, `RiemannianImpureDensityBlock`, `RiemannianImpureDensityConfig`, `AllLayerRiemannianStiefelMuon`, `build_all_layer_riemannian_optimizers`, `riemannian_complex_unitary_muon_step`, `polar_factorize_stiefel_svd`, `fast_rank_r_token_density_pack`, `ImpureQuantumDensityMirrorBlock`, `PureQuantumKetMirrorBlock`, `pack_hermitian_isometric`, `unpack_hermitian_isometric`, `bures_wasserstein_mcweeny_step`, `fit_observable_readouts_ridge` (`Waves 46–47` / `EXPERIMENT_PLAN_061–062`) | `from qdlblocks.models import RiemannianImpureDensityGPT, RiemannianImpureDensityBlock, AllLayerRiemannianStiefelMuon, polar_factorize_stiefel_svd, fast_rank_r_token_density_pack` ([`riemannian_impure_density_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py)); `from qdlblocks.layers import ImpureQuantumDensityMirrorBlock, ...` ([`impure_density_mirror_block.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py)) | **CANONICAL (`Waves 46–47`)** | Rank-1 pure ket attention $\sum_s w_{ts}|v_s\rangle$, unanchored Euclidean input `wte` drift (`+0.64` nats worse), or unweighted Newton-Schulz on SVD factors $(U, V^\top)$ when $\sigma_1 \gg \sigma_d$ |
| **Multi-Head Stinespring Superposition & Packed Complex Krylov Attention** | `MultiHeadStinespringSuperpositionAttention` / `MultiHeadSuperpositionMixedStateAttention` (`Wave 44` $H_g$ intra-factor sub-heads + $O(H_g^2 d_f)$ cross-head Gram purification), `PackedComplexRegisterDensityAttention`, `ComplexRoPE`, `krylov_purified_density_probe` | `from qdlblocks.layers import ...` ([`packed_complex_register_density_attn.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/packed_complex_register_density_attn.py)) | **CANONICAL (`Waves 37–44`)** | Un-refined `Q-Probe` ($\rho_t^{\text{purified}} |q_t^{\text{rot}}\rangle$), explicit $O(d_h^3)$ matrix squaring `rho @ rho`, or 2-pass temporal Krylov over `T` when $H_g \ge 2` |
| **Unit-Sphere & Product-of-Spheres Readout** | `parseval_mup_dyadic_chunked_ce`, `subspace_coincidence_dyadic_chunked_ce`, `subspace_l2_normalize` (pair-strided `(..., d//8, 4, 2)`), `strided_subspace_l2_normalize`, `SubspaceCoincidenceSpec` | `from qdlblocks.layers import ...` ([`parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py), [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py)) | **CANONICAL (`Waves 28–34`)** | `direct_chunked_two_beam_kboson_ce` + `ContextualBlochPurityGate` + `kboson_raw_k` + manual clamps (`26.8` / `44.0`) |
| **Level-Separated Wavelets & 1D Conv Bank** | `WaveletScaleState` (pair-strided symplectic layout), `MultiRateWavefunctionConvBank` | `from qdlblocks.layers import WaveletScaleState, MultiRateWavefunctionConvBank` ([`multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py)) | **CANONICAL (`Waves 32–34`)** | Global L2 normalization over summed wavelet levels `l2_normalize(c3+c2+c1+c0)` that crushes fine leaves |
| **SPD Metric Conditioning** | `BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=True)` | `from qdlblocks.layers import BlockCholeskyLinear` ([`wavelet_cholesky_branch.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky_branch.py)) | **CANONICAL (`2.55x` BMM)** | `fast_spd_bmm=False` sequential Python `for b in range(num_blocks)` loop |
| **Multi-Scale Wavelet Branch** | `WaveletCholeskyMultiscaleBranch`, `WaveletCholeskyConfig` | `from qdlblocks.layers import WaveletCholeskyMultiscaleBranch` ([`wavelet_cholesky_branch.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky_branch.py)) | **CANONICAL** | Single-scale Euclidean MLPs |
| **Two-Beam Phase Interferometer** | `OvercompleteTwoBeamInterference` | `from qdlblocks.layers import OvercompleteTwoBeamInterference` ([`overcomplete_two_beam.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_two_beam.py)) | **CANONICAL** | Single-beam additive residuals without Mach-Zehnder phase gating |
| **Dyadic $2^{16}$ Vocab Pyramid** | `Tied2Pow16WaveletCholeskyPyramid`, `build_dyadic_2pow16_spectral_tree` (with $d_{\text{mid}}=d/2, d_{\text{fine}}=d/4$, `num_fine_up_banks=1`) | `from qdlblocks.models import Tied2Pow16WaveletCholeskyPyramid` ([`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py)) | **CANONICAL** | Fixed $(128, 64)$ ranks at $d > 256$ (rank-starved) or `16`-bank ambient synthesis |
| **Flag-Free Language Models** | `RiemannianImpureDensityGPT` (`Wave 47` all-layer Riemannian Stiefel & $U(16)$ density GPT), `MultiHeadMixedStateKroneckerGPT` (`Wave 44`), `NgramCalibratedKroneckerBornGPT`, `build_or_load_900m_ngram_calibration_cache`, `evaluate_modded_nanogpt_validation`, `MultiShardRareTokenSampler` (`Wave 43` `900M`-token Unigram+Bigram+Trigram Hilbert-space readout calibration), `CalibratedKroneckerBornGPT` (`Wave 42` frozen baseline), `KroneckerMultiMeaning16x4GPT` (`Wave 41` frozen baseline), `TiedHoloBornComplexRoPEGPT` (`Wave 40`), `GeminiPackedComplexRoPEGPT` (`Waves 37–39` frozen baseline), `LevelSeparatedWaveletConvGPT` (`Waves 32–36`), `ParsevalMuPSubspaceCoincidenceGPT` (`Wave 31`), `ParsevalMuPNGramSpeedrunGPT` (`Waves 29–30`), `ParsevalMuPDyadicGPT` (`Wave 28`) | `from qdlblocks.models import RiemannianImpureDensityGPT, NgramCalibratedKroneckerBornGPT, NgramCalibratedKroneckerConfig, build_or_load_900m_ngram_calibration_cache, evaluate_modded_nanogpt_validation` ([`riemannian_impure_density_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py), [`ngram_calibrated_kronecker_born_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/ngram_calibrated_kronecker_born_gpt.py)) | **CANONICAL (`Waves 28–47`)** | `Radical2Pow16WaveletCholeskyGPT` (frozen `Waves 20–27` baseline) & `QuantumBornGPT` (`Waves 1–19`) |
| **Classical Leaderboard Reference Baseline** | `CompiledLeaderboardClassicalGPT`, `CompiledLeaderboardClassicalConfig`, `CompiledLeaderboardOptimizerConfig`, `build_compiled_leaderboard_optimizers`, `evaluate_compiled_leaderboard_validation` (`Wave 45` `45,273 tok/s` / `59.4%` MFU on RTX 5060 Ti via zero-copy `(B, T, H, D)` FlashAttention-2 + in-place Triton `_fused_softcap_ce_inplace_kernel` + Polar Express `StiefelMuon` + interior-layer `v_0` residual) | `from qdlblocks.models import CompiledLeaderboardClassicalGPT, CompiledLeaderboardClassicalConfig, build_compiled_leaderboard_optimizers, evaluate_compiled_leaderboard_validation` ([`compiled_leaderboard_classical_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py)) | **CANONICAL (`Wave 45`)** | `ClassicalModdedGPT` ([`classical_modded_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/classical_modded_gpt.py) — frozen `Wave 12` eager baseline with `(B, T, 50304)` logit materialization, `float32` Rotary upcast, and `12,521 tok/s` / `16.4%` MFU) |
| **Fast-Loop Vision Ablation Lab** | `WaveletCholeskyVisionClassifier`, `TiedDyadicWaveletClassPyramid`, `load_or_download_imagenet1k_32x32` | `from qdlblocks.models import WaveletCholeskyVisionClassifier` ([`wavelet_cholesky_vision.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_vision.py)) | **CANONICAL (`~11s`/run)** | Slow multi-hour language runs for initial readout/layer screening |
| **Riemannian Stiefel Optimizer** | `AllLayerRiemannianStiefelMuon`, `riemannian_complex_unitary_muon_step` (`Wave 47` all-layer 2D Stiefel + 3D complex $U(d_h)$ Lie-algebra skew-Hermitian tangent projection & $1/\sqrt{1+\eta^2}$ geodesic retraction, `1.2e-7` unitary error), `RiemannianStiefelMuon`, `riemannian_stiefel_muon_step` (`Wave 46` exact tangent / horizontal Grassmann projection + Polar Express + $1/\sqrt{1+\eta^2}$ geodesic retraction, `3.3e-7` Stiefel error), `StiefelMuon` (`polar_express`, Joint-Complex Polar Express `zeropower_via_polarexpress5_complex_2d/batched`, `normuon_equalize_complex`, `classify_kronecker_model_parameters` in `Wave 44`) | `from qdlblocks.models import AllLayerRiemannianStiefelMuon, riemannian_complex_unitary_muon_step`; `from qdlblocks.layers import RiemannianStiefelMuon, riemannian_stiefel_muon_step`; `from qdlblocks.optim import StiefelMuon` ([`muon.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/muon.py)) | **CANONICAL (`Waves 34, 38–47`)** | Euclidean `W <- W - lr * PolarExpress(M)` (`1.12` Stiefel drift, `+2.09` nats worse CE), hard periodic `retract_complex_unitary_weights()`, or separate real-half Polar Express |
| **Kronecker Diagnostic Shared Utils & Autograd Gate** | `DiagnosticModelSpec`, `KroneckerProbeConfig`, `load_or_build_calibrated_kronecker_model`, `build_or_load_kronecker_probe_model`, `compute_fineweb_10_frequency_deciles`, `load_fineweb_eval_slice`, `load_kronecker_probe_batch`, `compute_complex_factor_gram_spectrum`, `compute_pairwise_complex_branch_overlap`, `write_standardized_probe_json` ([`kronecker_diagnostic_utils.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/kronecker_diagnostic_utils.py)) & `autograd_efficiency_auditor.py` | `from qdlblocks.benchmarks.kronecker_diagnostic_utils import ...` | **CANONICAL (`Wave 44` / `Tracks 5 & 7`)** | Copy-pasting model construction, checkpoint loading, or Zipf decile partitioning across `scripts/*.py` |
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

### 2.3 Fused Triton Complex Norm-RoPE, Causal $p$-Boson Born, & Scale-Invariant Associative Krylov Kernels ([`flash_complex_krylov_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_complex_krylov_triton.py) — `Waves 38–39` / `EXPERIMENT_PLAN_053–054.md`)
- **Purpose**:
  - `fused_complex_norm_rope_triton`: Single-pass `@triton.jit` forward and backward kernel supporting **arbitrary complex dimensions $8 \le D \le 2048$** (including non-power-of-2 multiples of 32 such as $D \in \{96, 160, 192\}$ and $d_h \in \{16, 32\}$ via `BLOCK_D = triton.next_power_of_2(D)` and `mask_d = offs_d < D`). Computes `float32` SRAM L2 normalization $(z_{\text{re}}^2 + z_{\text{im}}^2)^{-1/2}$, optional full-spectrum complex rotary phase multiplication $\exp(i t \omega_a)$, tangent-space projection backward on $\mathbb{C}\mathbb{S}^{D-1}$, and direct `bfloat16` output under CUDA autocast.
  - `fused_causal_pboson_born_triton`: Single-pass `@triton.jit` causal $p$-boson Born weight forward and backward kernel over `(N_rows, T)` that computes $w_{ts} = \frac{(c_{\text{re},ts}^2 + c_{\text{im},ts}^2)^p \mathbb{I}(s \le t)}{\sum_{j \le t} (c_{\text{re},tj}^2 + c_{\text{im},tj}^2)^p + \varepsilon}$ in `float32` SRAM registers and writes out `w_born` in `bfloat16` without allocating `mag_sq`, `intensity`, or `mask_4d` in HBM.
  - `flash_complex_krylov_attention` & `factored_complex_krylov_step`: Evaluates $\rho_t^{\text{purified}} |\text{probe}_t\rangle = \frac{\rho_t(\rho_t |\text{probe}_t\rangle)}{\operatorname{Tr}(\rho_t^2) + \varepsilon}$ via the **Rank-1 Associative Krylov Identity** (`Wave 38`) and the **Scale-Invariant Krylov Normalization Fast-Path** (`probe_mode="v_krylov_norm"`, `Wave 39` / `EXPERIMENT_PLAN_054.md`), which exploits $\operatorname{normalize}\!\left(\frac{\rho_t^2 |\bar{v}_t\rangle}{\operatorname{Tr}(\rho_t^2)}\right) \equiv \operatorname{normalize}(\rho_t^2 |\bar{v}_t\rangle)$ to eliminate the $O(T^3)$ `w_gemm @ g_sq` McWeeny trace GEMMs (`> 50%` of attention FLOPs). On the RTX 5060 Ti (`B=16, T=512, L=6`), reaches **`48,604 tok/s`** at $d_{\text{complex}}=32$ (`693.6 MiB`), **`43,805 tok/s`** at $d_{\text{complex}}=64$ (`964.9 MiB`), **`33,669 tok/s`** at $d_{\text{complex}}=96$ (`1,261.1 MiB`), and **`28,286 tok/s`** at $d_{\text{complex}}=128$ (`1,665.8 MiB` — `3.56x` faster and `3.21x` less VRAM than `Wave 37`).
- **Usage**:
  ```python
  from qdlblocks.kernels import (
      factored_complex_krylov_step,
      flash_complex_krylov_attention,
      fused_causal_pboson_born_triton,
      fused_complex_norm_rope_triton,
  )
  ```

### 2.4 Fused Triton Mixed-State Kronecker Readout, Flash Causal $p=2$ Born Attention, Stinespring Coupling, & `16x4` Unitary Mixer Kernels ([`fused_kronecker_mixed_state_kernels.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_kronecker_mixed_state_kernels.py) — `Wave 44` / `EXPERIMENT_PLAN_059.md`)
- **Purpose**:
  - `fused_mixed_state_kronecker_chunk_fwd_triton` & `fused_mixed_state_kronecker_chunk_bwd_triton`: Single-launch 2D-tiled `@triton.jit` kernels over `(BLOCK_N = 32, BLOCK_V = 64)` supporting arbitrary polysemy ranks $R \in \{1, 2, 3, 4\}$ (`R_NUM: tl.constexpr`). Replaces TorchInductor's 5-kernel backward split and eliminates all `(R, N, V_c)` `float32` intermediate HBM spills, writing `g_cat` in `bfloat16` and atomically accumulating `d_beta_c`, `d_alpha_c`, and `d_povm_c` in a single chunk pass.
  - `flash_causal_pboson_complex_attn_triton`: SRAM-tiled `(64 x 64)` causal complex $p=2$ Born attention forward and backward (`_flash_causal_pboson_complex_fwd_kernel`, `_flash_causal_pboson_complex_bwd_q_kernel`, `_flash_causal_pboson_complex_bwd_kv_kernel`) that skips all upper-triangular tiles ($s > t$) and never materializes $(B, H, T, T)$ matrices in HBM.
  - `fused_cross_head_gram_stinespring_step` & `fused_kronecker_16x4_unitary_mixer_step`: Compiled single-graph kernels fusing the $O(K^2 d_f)$ cross-head Hermitian Gram trace-1 coupling + Stinespring $(M, K, R)$ unitary contraction + L2 normalization (`~45` small eager kernels $\to 1$) and the 3-stage `16x16` + `4x4` + `RxR` complex Kronecker unitary mixer (`~18` eager kernels $\to 1$), while keeping all reference code paths intact (`model.set_fused_kernels(enabled: bool)`).
- **Measured Multi-Size Speedup (`scripts/multihead_mixed_state_and_eval_speed_profiler.py --multi-size-fused-bench` on RTX 5060 Ti)**:
  - $(B=8, T=256, R=2)$: `569.7 ms -> 372.8 ms` (`3,595.2 -> 5,494.2 tok/s`, **`1.53x` speedup**, `1.42 GiB` VRAM)
  - $(B=16, T=512, R=2)$: `1,331.8 ms -> 631.0 ms` (`6,151.2 -> 12,983.7 tok/s`, **`2.11x` speedup**, `4.93 GiB` VRAM)
  - $(B=32, T=512, R=2)$: `2,377.2 ms -> 1,035.2 ms` (`6,892.1 -> 15,827.2 tok/s`, **`2.30x` speedup**, `9.42 GiB` VRAM)
  - $(B=8, T=1024, R=2)$: `1,330.0 ms -> 622.8 ms` (`6,159.6 -> 13,152.9 tok/s`, **`2.13x` speedup**, `6.05 GiB` VRAM)
  - $(B=8, T=512, R=4)$: `1,281.9 ms -> 702.5 ms` (`3,195.3 -> 5,830.5 tok/s`, **`1.82x` speedup**, `4.47 GiB` VRAM)
- **Usage**:
  ```python
  from qdlblocks.kernels import (
      flash_causal_pboson_complex_attn_triton,
      fused_cross_head_gram_stinespring_step,
      fused_kronecker_16x4_unitary_mixer_step,
      fused_mixed_state_kronecker_chunk_bwd_triton,
      fused_mixed_state_kronecker_chunk_fwd_triton,
  )
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

### 4.6 Standalone Pure-Complex Register Density GPT (`gemini_packed_complex_rope_gpt.py` — `Wave 37` / `EXPERIMENT_PLAN_052.md`, Frozen `Waves 37–39` Baseline)
- **What It Is**: Standalone pure-complex autoregressive language model operating on $H$ quantum registers of dimension $d_h = 32$ complex amplitudes ($\mathbb{C}^{d_{\text{complex}}}$) with `PackedComplexRegisterDensityAttention` (`V-Space Krylov Probe`, Refinements 1, 3, & 4), `ComplexUnitaryInterferenceFFN`, and token-chunked complex Born readout. Kept frozen as the `Waves 37–39` baseline.

### 4.7 Tied Token-Wavefunction & Operator-Ket Holographic Complex Born GPT (`tied_holo_born_complex_gpt.py` — `Wave 40` / `EXPERIMENT_PLAN_055.md`, Frozen `Wave 40` Baseline)
- **What It Is**: Canonical `Wave 40` pure-complex language model that resolves the `Wave 39` long-horizon rank collapse (`R_det: 243 -> 27`) by:
  1. **100% Softmax-Free Polynomial Holographic Complex Born Readout (`eval_tied_holo_born_probs` & `TiedHoloBornChunkedNLLFn`)**: Computes coherent interference $I_{t,v} = |1 + \alpha \langle e_v \mid \Phi_t \rangle_{\mathbb{C}}^{\text{cent}}|^2 + \varepsilon_0$ between a unit reference wave and both the Real and Imaginary parts of the centered Hermitian overlap (with zero $1/\sqrt{2}$ double-shrinkage, so $|\langle e_v \mid e_v \rangle_{\mathbb{C}}|^2 = 1.000000$), raised to polynomial boson power $p = 12$ via `torch.pow(I_tv, boson_p)` in `float32` with **zero `exp`, `softmax`, or `logsumexp` calls**.
  2. **Tied Initial Token Wavefunctions $\langle e_v \rvert$ + Sequence-Side `out_operator`**: Ties the normalized initial token wavefunctions directly into the readout and applies `out_operator` + self-echo Gram-Schmidt projection on the $B \times T$ sequence positions (`8,192` rows vs. `50,257` vocabulary rows — `6.13x` fewer operator GEMM rows/step).
  3. **Smooth Differentiable Manifold & Parseval-Isotropy Penalties (`compute_soft_manifold_and_isotropy_penalties`)**: Enforces unit-sphere norms $\mathcal{L}_{\text{sphere}}$, square complex unitarity $\mathcal{L}_{\text{unitary}}$ ($W W^H \to I$, excluding rectangular FFN layers), and Parseval frame isotropy $\mathcal{L}_{\text{iso}}$ over `valid_vocab_size = 50,257`, preserving $R_{\text{emb}} \ge 240.0$ (`17.6x` higher than `Wave 39`).
  4. **Unbiased Rare-Token Importance Window Sampler (`RareTokenImportanceWindowSampler`)**: Samples candidate FineWeb windows with tail-contrasted Laplace-smoothed rarity and applies exact inverse-probability importance weights (`0.0` analytical bias).
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      RareTokenImportanceWindowSampler,
      TiedHoloBornComplexRoPEGPT,
      TiedHoloBornConfig,
      build_tied_holo_born_optimizers,
      compute_soft_manifold_and_isotropy_penalties,
  )
  ```

### 4.8 Rank-$R$ Multi-Meaning $16 \times 16 \times 16 \times 16 = 65,536$ Kronecker Wavefunction GPT (`kronecker_multimeaning_16x4_gpt.py` — `Wave 41` / `EXPERIMENT_PLAN_056.md`)
- **What It Is**: Canonical `Wave 41` pure-complex language model that resolves the three `Wave 40` post-mortem bottlenecks (`1/128` flat space background collision floor, polysemy collapse in rank-1 kets, and `2M`-token unigram prior truncation) by:
  1. **Rank-$R$ Multi-Meaning $16^4 = 65,536$ Kronecker Wavefunctions (`KroneckerMultiMeaning16x4GPT` & `Kronecker16x4UnitaryMeaningMixer`)**: Embeds every vocabulary word $v \in \{0, \dots, 50,256\}$ as a rank-$R$ entangled superposition $|\Psi_v\rangle = \sum_{r=0}^{R-1} \alpha_{v,r} \bigotimes_{m=0}^{3} |\psi_{v,r}^{(m)}\rangle \in (\mathbb{C}^{16})^{\otimes 4} \cong \mathbb{C}^{65,536}$ (`512x` lower single-branch random background collision $(1/16)^4 = 1/65,536$ vs. $1/128$ in flat $\mathbb{C}^{128}$ while storing only $4 \times 16 = 64$ complex coordinates per meaning branch).
  2. **Single-`torch.bmm` + `@torch.compile` 2-Level Binary Complex Product Tree Readout (`eval_kronecker_multimeaning_born_probs` & `KroneckerMultiMeaningBornChunkedNLLFn`)**: Evaluates both real and imaginary parts of all $4R$ 16D complex factor overlaps per vocabulary chunk in a single batched `torch.bmm(U_batched, W_chunk)` call, combines them via a 2-level binary complex product tree $(16\times 16)\times(16\times 16) = 65,536$ against a phase-balanced reference wave $e^{i\pi/4} = \frac{1+i}{\sqrt{2}}$, and raises the intensity to polynomial boson power $p = 11$ via `torch.pow` with **zero `exp`, `softmax`, or `logsumexp` calls**.
  3. **100M-Token FineWeb Unigram Prior (`load_100m_unigram_prior`)**: Initializes the readout prior from `token_freq_slice_100m.npy` (`49,850` observed tokens + `407` unseen tokens floored at `1.0` count, `15.98x` fewer unseen tokens than `slice_2m` and Step-0 validation NLL `8.04 < 8.20` nats).
  4. **Relaxed Per-Factor Isotropy ($\lambda_{\text{iso}} = 0.05$) & Cross-Meaning Branch Orthogonality (`compute_kronecker_manifold_and_isotropy_penalties`)**: Enforces 4-factor Kronecker branch orthogonality $\mathcal{L}_{\text{branch}} = \frac{1}{V_{\text{valid}}} \sum_v \sum_{r < s} \prod_{m=0}^{3} |\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\psi}_{v,s}^{(m)} \rangle_{\mathbb{C}}|^2$ ($\bar{O}_{\text{branch}} < 10^{-3}$) alongside relaxed per-factor isotropy ($\bar{R}_{\text{factor}} \ge 20/32$).
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      Kronecker16x4UnitaryMeaningMixer,
      KroneckerMultiMeaning16x4GPT,
      KroneckerMultiMeaningBornChunkedNLLFn,
      KroneckerMultiMeaningConfig,
      build_kronecker_multimeaning_optimizers,
      compute_kronecker_manifold_and_isotropy_penalties,
      eval_kronecker_multimeaning_born_probs,
  )
  ```

### 4.9 Calibrated Readout Operator & Tied Kronecker Wavefunction GPT (`calibrated_kronecker_born_gpt.py` — `Wave 42` / `EXPERIMENT_PLAN_057.md`)
- **What It Is**: Canonical `Wave 42` pure-complex language model that eliminates the static frozen `prior_weights` buffer of `Waves 40–41` by calibrating the trainable readout unitaries `out_factor_u_re/im`, tied token factor wavefunctions `embed_re/im`, and trainable POVM operator amplitudes `povm_scale_logits` directly against random-init attention outputs at Step 0:
  1. **Unitary Procrustes Carrier Alignment (`calibrate_readout_to_random_attention_unigram`)**: Samples random-init token sequences through the $L$-layer complex attention + Kronecker mixer backbone to measure the empirical mean pre-readout direction $|\bar{q}_k\rangle \in \mathbb{S}_{\mathbb{C}}^{15}$ and aligns each $16 \times 16$ unitary $\hat{U}_{\text{out},k} \in U(16)$ so $\bar{q}_k \hat{U}_{\text{out},k} = c_{\text{ref}} = \frac{1+i}{4\sqrt{2}}\mathbf{1}_{16}$ (`carrier_alignment_fidelity == 1.0 ± 1e-5`, $\|\hat{U}_{\text{out}} \hat{U}_{\text{out}}^H - I\|_F < 10^{-5}$).
  2. **Geometric Carrier-Angle Wavefunction Tilt ($\rho_v \in [-\rho_{\max}, +\rho_{\max}]$)**: Tilts each token's unit-norm factor ket $|\hat{\psi}_{v,r}^{(m)}\rangle = \rho_v |c_{\text{ref}}\rangle + \sqrt{1 - \rho_v^2}|u_{v,r,\perp}^{(m)}\rangle$ while preserving $\bar{R}_{\text{factor}} \ge 28/32$ and $R_{\text{emb}} \ge 220/256$.
  3. **Closed-Form Algebraic POVM Amplitude Calibration & 5-Input Chunked Autograd (`CalibratedKroneckerBornChunkedNLLFn`)**: Maps `povm_scale_logits` to positive operator weights $w_{\text{povm},v} = \frac{1}{2}(\theta_v + \sqrt{\theta_v^2 + 1}) + \varepsilon_{\text{branch}} > 0$ (100% `exp`/`softplus`-free), solves for $\theta_{\text{povm},v}^{(0)} = a_{\text{eff},v} - \frac{1}{4 a_{\text{eff},v}}$ on random-init attention outputs (`KL < 0.005` nats at Step 0), and propagates exact analytical gradients `d_povm_weights[v0:v1] = delta.sum(dim=0) / povm_chunk.clamp_min(1e-12)` during training.
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      CalibratedKroneckerBornChunkedNLLFn,
      CalibratedKroneckerBornGPT,
      CalibratedKroneckerConfig,
      build_calibrated_kronecker_optimizers,
      compute_calibrated_kronecker_penalties,
      eval_calibrated_kronecker_born_probs,
  )
  ```

### 4.10 `900M`-Token Step-0 Unigram, Bigram, and Trigram Calibrated Kronecker Born GPT (`ngram_calibrated_kronecker_born_gpt.py` — `Wave 43` / `EXPERIMENT_PLAN_058.md`)
- **What It Is**: Canonical `Wave 43` pure-complex language model that extends Step-0 calibration from Unigrams (`Wave 42`) to **`900,000,000`-token Unigram, Bigram, Skip-Bigram, and Hashed-Trigram conditional transitions** in the 15D complex carrier-orthogonal subspace $\mathcal{H}_{\perp} \subset \mathbb{C}^{16}$ per Kronecker factor, while maintaining **zero frozen `prior_weights` buffer** and **zero `exp`/`softmax`/`logsumexp` calls**:
  1. **Isometric Helmert Subspace Embedding (`embed_real30_to_carrier_ortho_c16` & `project_c16_to_carrier_ortho_real30`)**: Maps 30D real spectral coordinates $\mathbf{w} \in \mathbb{R}^{30}$ per Kronecker factor isometrically into $\mathcal{H}_{\perp} = \{\lvert u \rangle \in \mathbb{C}^{16} : \langle c_{\text{ref}} \mid u \rangle = 0\}$, guaranteeing $\langle c_{\text{ref}} \mid \mathcal{E}_{\perp}(\mathbf{w}) \rangle = 0$ (`< 1e-7`) so conditional N-gram amplitudes never perturb the unigram carrier alignment.
  2. **2-Pass `900M`-Token GPU Spectral & Conditional Expectation Cache (`build_or_load_900m_ngram_calibration_cache`)**: Streams all 9 official `modded-nanogpt` FineWeb training shards (`fineweb_train_000001.bin`..`fineweb_train_000009.bin`) to extract Polar-ZCA-whitened spectral vocabulary coordinates (`spectral_vocab_ortho`), $P_{\text{uni}}$-whitened conditional expectation kets for `8,192` direct bigram slots (`bi_slot_ortho_kets`) and `16,384` polynomial XOR-hashed trigram buckets (`tri_bucket_ortho_kets`), and strictly unitary `U(16)` causal bigram and lag-1 skip-bigram operators (`bi_factor_op_re_im`, `tri_factor_op_re_im`) via Complex Unitary Procrustes SVD (`err < 1e-5`).
  3. **Zero-Materialization `10,485,760`-Token Official `modded-nanogpt` Evaluator (`evaluate_modded_nanogpt_validation`) & Multi-Shard Sampler (`MultiShardRareTokenSampler`)**: Evaluates exact polynomial Born NLL, Perplexity, Top-1, and Top-50 accuracy across the full `10.49M`-token `fineweb_val_000000.bin` benchmark slice without allocating `(N, 50304)` logits, and streams importance-sampled windows across all 9 training shards (`900M` tokens).
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      MultiShardRareTokenSampler,
      Ngram900MCalibrationData,
      NgramCalibratedKroneckerBornGPT,
      NgramCalibratedKroneckerConfig,
      build_ngram_calibrated_kronecker_optimizers,
      build_or_load_900m_ngram_calibration_cache,
      compute_ngram_calibrated_kronecker_penalties,
      embed_real30_to_carrier_ortho_c16,
      evaluate_modded_nanogpt_validation,
      project_c16_to_carrier_ortho_real30,
  )
  ```

### 4.11 Compiled Classical `Modded-NanoGPT` Leaderboard Speedrun Baseline (`compiled_leaderboard_classical_gpt.py` — `Wave 45` / `EXPERIMENT_PLAN_060.md`)
- **What It Is**: Canonical `Wave 45` compiled classical reference model matching `KellerJordan/modded-nanogpt` (`L=12, d=768, H=6, d_h=128, d_ff=3072`, `162.21M` params) that eliminates all 5 single-GPU throughput and VRAM bottlenecks of the legacy `Wave 12` [`ClassicalModdedGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/classical_modded_gpt.py) (`12,520.9 tok/s` $\to$ **`45,273.2 tok/s` (`3.62x` speedup)**, **`59.4%` MFU / `34.47 TFLOP/s`**, and `-4.57 GB` Peak VRAM at `B=8, T=1024` on the `16 GB` RTX 5060 Ti):
  1. **Zero-Copy `(B, T, H, D)` FlashAttention-2 Layout & `bfloat16` `NativeLeaderboardRotary`**: Keeps `q, k, v` in physical `(B, T, H, D)` layout through `rms_norm`, `rotary`, and `ve`/`v_0` mixing so `q.transpose(1, 2)` and `attn_out.transpose(1, 2).reshape(B, T, D)` are zero-copy stride views into `F.scaled_dot_product_attention`, and computes rotary embeddings in `bfloat16` without `float32` upcasts (`q.float()`).
  2. **Zero-Extra-VRAM In-Place Triton Softcap Cross-Entropy Kernel (`_fused_softcap_ce_inplace_kernel` & `_CompiledChunkedSoftcapLinearCEFn`)**: Computes `15 * tanh(x / 15)` + online `LogSumExp` + cross-entropy loss + analytical gradient `(p - 1_y) * (1 - tanh^2(x/15)) / N` in-place inside the `(C, 50304)` `bfloat16` chunk buffer (`C = 8,192`) in a single `@triton.jit` pass with `0` bytes of extra HBM allocation.
  3. **Interior-Layer `v_0` Value Residual Fix (`interior_value_mode="v0_residual"`)**: Blends Layer-0 values `v_0` (`v = sa_lambdas[0] * v + sa_lambdas[1] * v_0`) on interior layers `3, 4, 5, 6, 8` where `ve_l is None`, instead of scaling `v` by `0.5` as in legacy `ClassicalModdedGPT`.
  4. **Polar Express `StiefelMuon` (`polar_method="polar_express"`) + Distinct `lm_head` / `wte_and_ve` / `scalars` `AdamW` Groups (`build_compiled_leaderboard_optimizers`)**: Uses optimal 5-iteration minimax Polar Express coefficients (`stiefel_penalty=0.0`), 300-step momentum warmup (`0.85 -> 0.95`), and leaderboard-faithful per-group learning rates (`wte_lr=0.30..0.60`, `head_lr=0.004..0.008`, `muon_lr=0.025..0.05`).
- **Minimal Usage**:
  ```python
  from qdlblocks.models import (
      CompiledLeaderboardClassicalConfig,
      CompiledLeaderboardClassicalGPT,
      CompiledLeaderboardOptimizerConfig,
      build_compiled_leaderboard_optimizers,
      evaluate_compiled_leaderboard_validation,
  )
  ```

---

## 5. Riemannian Stiefel Optimization & Learning-Rate Schedules (`src/qdlblocks/optim/`)

### 5.1 `StiefelMuon` & Mandatory Routing for `kraus_generators` and 2D Backbone Projections
- **Critical Lessons (`Wave 22`, `Wave 34`, & `Waves 40–43`)**:
  1. `pyramid.kraus_generators` has shape `(R, d, d)` (`ndim == 3`). Never let a substring filter route `kraus_generators` to Euclidean `AdamW(weight_decay=0.0)`, or its Stiefel error $\|K_r^\top K_r - I\|_F / \sqrt{d}$ will drift to `> 20.0`.
  2. Never omit the 2D `p.ndim == 2 and min(p.shape) >= 16 and max(p.shape) <= 4096` branch when building `StiefelMuon` parameter groups (`Bug 1` in `EXPERIMENT_PLAN_049.md`), or `qkv_proj`, `out_proj`, and `interferometer` matrices fall into `AdamW` at `7.4x` lower LR without Polar Express orthogonalization.
  3. For complex `ComplexLinear` matrices (`weight_re, weight_im`) and 3D batched `16x16` Kronecker factor unitaries (`factor_u_re, factor_u_im`, `bi_factor_u`, `tri_factor_u`, `smear_u`), set `stiefel_penalty=0.0` in `StiefelMuon` (`build_ngram_calibrated_kronecker_optimizers`) and use `compute_ngram_calibrated_kronecker_penalties` so complex unitarity $W_{\text{re}} W_{\text{re}}^\top + W_{\text{im}} W_{\text{im}}^\top = I$ is enforced on the joint complex operator rather than conflicting real-half Stiefel penalties.

### 5.2 Choosing Between NanoGPT `WSD` and Smith 2-Cycle `CLR` (`arXiv:1506.01186`)
- **Fixed-Horizon Runs (`modded_nanogpt_wsd`)**: Best when the total token horizon $T_{\max}$ is fixed in advance and only the final checkpoint at $\tau = 1.00$ matters (`5.4372` at $d=256$, `4.7919` at $d=512$ / `29.49M` tokens). Warning: uncooled intermediate checkpoints at $\tau \in [0.04, 0.60]$ sit at peak LR $\eta_{\max}$ and appear `+0.15` nats worse until the cooldown branch is run (`scripts/isotoken_wsd_cooldown_branch_evaluator.py`).
- **Anytime / Multi-Cycle Runs (`smith_triangular2` with Iso-Integral Scaling)**: Use [`scripts/smith_clr_range_test_vs_wsd_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/smith_clr_range_test_vs_wsd_profiler.py) to find $(\eta_{\text{base}}, \eta_{\max})$ automatically in 40 steps, and use the iso-integral 2-cycle schedule when you need strong calibrated checkpoints at **both** $\tau = 0.50$ (`-0.0452` nats better than uncooled `WSD`) and $\tau = 1.00$.

---

## 6. First-Class Diagnostic & Verification Scripts (`scripts/`)

Before writing any ad-hoc one-liner, check if one of our deterministic CLI tools in `scripts/` (or [`.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md`](file:///home/festivity/agy/QuantumDLBlocks/.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md)) already performs the diagnostic:

| Script | Purpose & Example Invocation |
| :--- | :--- |
| [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) | Enforces Rules 1–8 (`<= 1,000` soft / `<= 2,500` hard comment-free SLOC lines, `<= 8` args, zero semicolons, max nested function SLOC). |
| [`scripts/verify_plan.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/verify_plan.py) | Single-command plan verification orchestrator (`AST linter` + `pytest` + hypothesis summary check) (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/wave_report.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wave_report.py) | Generates a Markdown + JSON wave leaderboard across `.cache/live_logs/`, `.cache/model_cards/`, and `.cache/benchmarks.db` (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/wave_preflight_hardware_and_throughput_gate.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wave_preflight_hardware_and_throughput_gate.py) | Pre-launch hardware readiness, VRAM budget, FineWeb shard cache, and Kronecker forward+backward throughput gate (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/colab_fleet_watchdog_and_auto_requeue.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet_watchdog_and_auto_requeue.py) | Non-destructive watchdog for remote Colab fleet runs (`--dry-run` default) checking log freshness and stalled runs (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/ngram_helmert_carrier_calibration_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ngram_helmert_carrier_calibration_inspector.py) | Audits Step-0 Helmert carrier-orthogonal calibration $\langle c_{\text{ref}} \mid \mathcal{E}_{\perp}(\mathbf{w}) \rangle = 0$ and 5-stage Step-0 NLL progression (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/trigram_context_smearing_ablation_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/trigram_context_smearing_ablation_probe.py) | Profiles leave-one-out N-gram context channels (`unigram_only`, `no_bigram`, `no_trigram`, `no_smear`, `full_calibrated`) (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/multibranch_polysemy_entropy_and_subspace_angle_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multibranch_polysemy_entropy_and_subspace_angle_probe.py) | Measures polysemy router Shannon entropy $H(\beta_t)$, temporal routing std, and principal subspace angles (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/kronecker_coincidence_gradient_snr_and_rank_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kronecker_coincidence_gradient_snr_and_rank_probe.py) | Audits per-factor Gram effective rank (`>= 31/32`) and coincidence gradient SNR across $\gamma_{\text{coinc}}$ (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/calibrated_kronecker_vocab_spectrum_and_povm_drift_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/calibrated_kronecker_vocab_spectrum_and_povm_drift_probe.py) | Audits `50,257`-token POVM scale drift and frequency-decile spectrum (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/context_adaptive_branch_beta_and_self_echo_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/context_adaptive_branch_beta_and_self_echo_probe.py) | Verifies full complex inner-product self-echo suppression strictly inside $\mathcal{H}_{\perp}$ (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/born_exponent_and_holographic_coincidence_sweep.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/born_exponent_and_holographic_coincidence_sweep.py) | Sweeps polynomial Born exponent $p$ and holographic coincidence $\gamma_{\text{coinc}}$ (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/multilag_window_and_coincidence_schedule_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multilag_window_and_coincidence_schedule_probe.py) | Profiles multi-lag window horizons (`lags 1..3`) and coincidence warmup schedules (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/multihead_gram_krylov_vs_intrahead_ablation_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multihead_gram_krylov_vs_intrahead_ablation_probe.py) | Verifies exact algebraic equivalence (`< 1e-6`) between $O(H_g^2 d_f)$ cross-head Gram purification and $16 \times 16$ density-matrix squaring (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/quantum_superposition_purity_and_gram_spectrum_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/quantum_superposition_purity_and_gram_spectrum_probe.py) | Profiles mixed-state purity $\operatorname{Tr}(\rho^2)$, effective rank, and von Neumann entropy (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/complex_stiefel_orthogonality_and_large_batch_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complex_stiefel_orthogonality_and_large_batch_probe.py) | Audits Joint-Complex Polar Express unitarity, phase equivariance (`< 1e-4`), and batch LR scaling (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/self_echo_repetition_penalty_and_nll_breakdown_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/self_echo_repetition_penalty_and_nll_breakdown_probe.py) | Measures prompt self-echo probability mass $P(y_t = x_t \mid x_{\le t})$ and 10-decile NLL (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/multihead_von_neumann_entropy_and_coherence_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multihead_von_neumann_entropy_and_coherence_inspector.py), [`scripts/polysemy_router_token_sense_clustering_cli.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/polysemy_router_token_sense_clustering_cli.py), [`scripts/polysemy_branch_sense_disambiguation_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/polysemy_branch_sense_disambiguation_inspector.py), [`scripts/kronecker_factor_entanglement_and_mutual_info_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kronecker_factor_entanglement_and_mutual_info_probe.py), [`scripts/povm_scale_drift_and_rarity_decile_analyzer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/povm_scale_drift_and_rarity_decile_analyzer.py), [`scripts/rare_token_gradient_starvation_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/rare_token_gradient_starvation_tracker.py), [`scripts/ngram_quantum_operator_vs_attention_gate_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ngram_quantum_operator_vs_attention_gate_tracker.py), [`scripts/modded_nanogpt_10m_per_decile_ce_breakdown.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/modded_nanogpt_10m_per_decile_ce_breakdown.py), [`scripts/born_povm_boson_power_and_floor_calibrator.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/born_povm_boson_power_and_floor_calibrator.py), [`scripts/random_init_attention_carrier_spectrum_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/random_init_attention_carrier_spectrum_probe.py), [`scripts/grad_accum_effective_batch_scaling_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/grad_accum_effective_batch_scaling_profiler.py), [`scripts/wall_clock_chinchilla_pareto_predictor.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wall_clock_chinchilla_pareto_predictor.py) | Wave 41–44 Backlog Diagnostic Suite powered by [`src/qdlblocks/benchmarks/kronecker_diagnostic_utils.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/kronecker_diagnostic_utils.py) (`EXPERIMENT_PLAN_059.md`). |
| [`scripts/calibrate_900m_ngram_kronecker_readout.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/calibrate_900m_ngram_kronecker_readout.py) | Scans all 9 FineWeb training shards (`900,000,000` tokens), builds or loads `.cache/fineweb_900M_ngram_calibration.pt`, and audits the 5-way Step-0 Unigram/Bigram/Trigram Hilbert-space readout calibration on held-out FineWeb validation tokens (`EXPERIMENT_PLAN_058.md`). |
| [`scripts/kronecker_born_hyperparam_search.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kronecker_born_hyperparam_search.py) | Audits Step-0 random-init attention output calibration (`Unitary Procrustes` carrier alignment + carrier tilt + closed-form algebraic `povm_scale_logits` fit with zero frozen prior buffer) and runs an automated multi-configuration hyperparameter sweep over $(R, p, \eta_{\text{kron}}, \gamma_{\text{coinc}}, \rho_{\max}, \tau_{\text{uni}}, \text{LRs})$ on FineWeb (`EXPERIMENT_PLAN_057.md`). |
| [`scripts/kronecker_16x4_meaning_and_snr_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kronecker_16x4_meaning_and_snr_profiler.py) | Profiles 4-factor $16^4 = 65,536$ Kronecker background collision floor (`510x` SNR vs. flat $\mathbb{C}^{128}$), $K$-sense polysemy coincidence across $R \in \{1, 2, 4\}$, `100M`-token vs. `2M`-token vs. uniform FineWeb unigram priors (`49,850` observed tokens, `15.98x` unseen reduction), and single-`torch.bmm` readout throughput (`EXPERIMENT_PLAN_056.md`). |
| [`scripts/fineweb_token_rarity_and_born_rank_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_token_rarity_and_born_rank_profiler.py) | Profiles FineWeb Zipf token frequency slices (`2M..300M`), audits `RareTokenImportanceWindowSampler` rare-token exposure gains and exact inverse-probability unbiasedness, and benchmarks 100-step Hilbert-space rank preservation (`R_emb: 247.9 vs 14.1`, `17.6x`) (`EXPERIMENT_PLAN_055.md`). |
| [`scripts/complex_krylov_rtx5060ti_vram_sizer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complex_krylov_rtx5060ti_vram_sizer.py) | Profiles `GeminiPackedComplexRoPEGPT` across $d_{\text{complex}} \in \{128, 256, 384, 512, 640, 768\}$ ($H \in \{4, 8, 12, 16, 20, 24\}$, up to `220.04M` params) on the 16 GB RTX 5060 Ti, auditing `float32_activation_leak_count == 0`, peak VRAM (`MiB`), and throughput (`tok/s`) (`EXPERIMENT_PLAN_053.md`). |
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

