# QuantumDLBlocks Reusable Building Blocks & Kernel Catalog (`docs/BUILDING_BLOCKS.md`)

**Authoritative Agent & Researcher Reference for Composable Quantum-Geometric Primitives, Fused GPU Kernels, Stiefel Optimizers, and Diagnostic Probes (Current as of `Wave 29` / `EXPERIMENT_PLAN_045`).**

> **Mandatory Agent Instruction (See [`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md) Sections 5 & 6)**:
> 1. **Read Before Planning or Coding**: Every agent and subagent MUST consult this catalog before drafting a new `EXPERIMENT_PLAN_0XX.md` or writing model/layer/kernel code so that new experiments compose our proven, high-throughput building blocks instead of re-implementing them or reviving superseded legacy patterns.
> 2. **Follow Rule 6 ([`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md))**: Never thread new `use_*` or `is_*` boolean flags into frozen baseline models ([`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py), [`direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py)). Import the reusable building blocks below into a clean, flag-free per-experiment module.
> 3. **Contribute Back**: Whenever an experiment validates a general-purpose kernel, layer, optimizer schedule, or diagnostic script, promote it into `src/qdlblocks/{kernels,layers,optim,benchmarks}/` or `scripts/`, export it in `__init__.py`, and document it in this file and [`README.md`](file:///home/festivity/agy/QuantumDLBlocks/README.md).

---

## 1. Quick-Reference Status Matrix: What to Use vs. What Is Frozen/Legacy

| Subsystem | Canonical Reusable Component (Use This!) | Import Path | Status | Superseded / Frozen Counterpart (Do NOT Extend) |
| :--- | :--- | :--- | :---: | :--- |
| **Fused Readout & CE GPU Kernel** | `compiled_parseval_convex_traceless_chunk_step`, `fused_fp8_or_bf16_chunk_gemm`, `quantize_fp8_vocab_transposed` | `from qdlblocks.kernels import ...` ([`fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py)) | **CANONICAL (`Wave 28`)** | Unfused eager elementwise chunk steps or `float32` backward GEMMs |
| **Fused Attention GPU Kernel** | `flash_born_attention`, `flash_purified_born_attention` | `from qdlblocks.kernels import ...` ([`flash_born_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_born_triton.py)) | **CANONICAL** | Materialized $O(B \cdot H \cdot T^2)$ attention matrices at $d \ge 384$ or $T \ge 1024$ |
| **Unit-Sphere Readout & Loss Layer** | `parseval_mup_dyadic_chunked_ce`, `eval_parseval_mup_full_logits`, `ParsevalChunkedCEReadoutSpec` | `from qdlblocks.layers import ...` ([`parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py)) | **CANONICAL (`Wave 28`)** | `direct_chunked_two_beam_kboson_ce` + `ContextualBlochPurityGate` + `kboson_raw_k` + manual clamps (`26.8` / `44.0`) |
| **SPD Metric Conditioning** | `BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=True)` | `from qdlblocks.layers import BlockCholeskyLinear` ([`wavelet_cholesky_branch.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky_branch.py)) | **CANONICAL (`2.55x` BMM)** | `fast_spd_bmm=False` sequential Python `for b in range(num_blocks)` loop |
| **Multi-Scale Wavelet Branch** | `WaveletCholeskyMultiscaleBranch`, `WaveletCholeskyConfig` | `from qdlblocks.layers import WaveletCholeskyMultiscaleBranch` ([`wavelet_cholesky_branch.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky_branch.py)) | **CANONICAL** | Single-scale Euclidean MLPs |
| **Two-Beam Phase Interferometer** | `OvercompleteTwoBeamInterference` | `from qdlblocks.layers import OvercompleteTwoBeamInterference` ([`overcomplete_two_beam.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_two_beam.py)) | **CANONICAL** | Single-beam additive residuals without Mach-Zehnder phase gating |
| **Dyadic $2^{16}$ Vocab Pyramid** | `Tied2Pow16WaveletCholeskyPyramid`, `build_dyadic_2pow16_spectral_tree` (with $d_{\text{mid}}=d/2, d_{\text{fine}}=d/4$, `num_fine_up_banks=1`) | `from qdlblocks.models import Tied2Pow16WaveletCholeskyPyramid` ([`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py)) | **CANONICAL** | Fixed $(128, 64)$ ranks at $d > 256$ (rank-starved) or `16`-bank ambient synthesis |
| **Flag-Free Language Models** | `ParsevalMuPNGramSpeedrunGPT` (`Wave 29` N-Gram + $V^{(0)}$ Speedrun) & `ParsevalMuPDyadicGPT` (`Wave 28` Base) | `from qdlblocks.models import ParsevalMuPNGramSpeedrunGPT, ParsevalMuPDyadicGPT` ([`parseval_mup_ngram_speedrun_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py)) | **CANONICAL (`Waves 28–29`)** | `Radical2Pow16WaveletCholeskyGPT` (frozen `Waves 20–27` baseline) & `QuantumBornGPT` (`Waves 1–19`) |
| **Fast-Loop Vision Ablation Lab** | `WaveletCholeskyVisionClassifier`, `TiedDyadicWaveletClassPyramid`, `load_or_download_imagenet1k_32x32` | `from qdlblocks.models import WaveletCholeskyVisionClassifier` ([`wavelet_cholesky_vision.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_vision.py)) | **CANONICAL (`~11s`/run)** | Slow multi-hour language runs for initial readout/layer screening |
| **Riemannian Stiefel Optimizer** | `StiefelMuon` (`polar_express`, `stiefel_penalty=0.02`, `use_normuon=True`, `use_twin_rail=True`) | `from qdlblocks.optim import StiefelMuon` ([`muon.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/muon.py)) | **CANONICAL** | Hard periodic QR snapping or routing `kraus_generators` to Euclidean `AdamW` |
| **Experiment Logger & WebUI Ingestor** | `StandardizedExperimentLogger`, `ExperimentIntentSpec`, `ingest_all_waves_and_verifications` | `from qdlblocks.benchmarks import StandardizedExperimentLogger, ExperimentIntentSpec` ([`experiment_logger.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/experiment_logger.py)) | **CANONICAL (`Plan 044`)** | Ad-hoc unparsed JSON dumps without `[EXPERIMENT_INTENT]` or `ModelCard v1.0` |

---

## 2. Fused GPU Kernels (`src/qdlblocks/kernels/`)

Writing custom GPU kernels inside a single `examples/XX_*.py` script prevents reuse across waves. All custom Triton and `@torch.compile` (TorchInductor) + Tensor-Core kernels live in [`src/qdlblocks/kernels/`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/__init__.py) and are exported from `qdlblocks.kernels`.

### 2.1 Fused Parseval Convex-Traceless Chunked CE Kernel ([`fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py))
- **Purpose**: Fuses the entire per-chunk vocabulary readout ($V_{\text{chunk}} = 4,096$) into two Tensor-Core GEMMs (`FP8 E4M3` via `torch._scaled_mm` or `BF16`) plus a single TorchInductor `@torch.compile` kernel (`compiled_parseval_convex_traceless_chunk_step`) that computes centered sharp & diffuse observables, rectified-quadratic target peak boost $\gamma \max(\hat{c}_{t,v}, 0)^2$, chunk partition sums, and exact analytical gradients $(g_{\text{sharp}}, g_{\text{diff}})$ in one SRAM pass.
- **Measured Speedup**: Doubles end-to-end training throughput from `11,116 tok/s` to **`24,658–25,270 tok/s`** at $d=256$ (`5.48 GB` VRAM) and **`18,335 tok/s`** at $d=384$ (`6.89 GB` VRAM) while avoiding `(N_tok, 50257)` `float32` logit materialization.
- **Usage**:
  ```python
  from qdlblocks.kernels import (
      compiled_parseval_convex_traceless_chunk_step,
      fused_fp8_or_bf16_chunk_gemm,
      quantize_fp8_vocab_transposed,
  )
  ```

### 2.2 Fused Triton Flash-Born Attention ([`flash_born_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_born_triton.py))
- **Purpose**: Computes causal Born-rule attention $P(j \mid i) \propto \max(0, \beta_h \sqrt{d_h}\langle K_j \mid Q_i\rangle)^2$ and purified Born attention inside $64 \times 64$ GPU SRAM tiles with atomic-free backward passes (`dQ`, `dK`, `dV`), reducing attention VRAM by up to `94%` and accelerating $T \ge 1024$ or $d \ge 384$ runs by up to `4.39x`.
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
  Via `model.absorb_beta_into_readout_cholesky()`, the scale $\beta_0(d)^{1/2}$ can be absorbed directly into the positive-definite Cholesky factors $L \leftarrow \beta_0(d)^{1/4} L$ of `readout_chol` applied *after* unit-sphere normalization (exact equivalence `< 7.6e-6`).
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

---

## 4. Reference Architectures & Dyadic Pyramids (`src/qdlblocks/models/`)

### 4.1 Flag-Free Parseval $\mu\text{P}$ Dyadic GPT (`parseval_mup_dyadic_gpt.py` — Current Canonical LM)
- **What It Is**: The clean, flag-free `Wave 28` language model (`351` lines, zero `use_*`/`is_*` flags, zero `PurityGate`, zero `kboson_raw_k`, zero clamps).
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

---

## 5. Riemannian Stiefel Optimization & Learning-Rate Schedules (`src/qdlblocks/optim/`)

### 5.1 `StiefelMuon` & Mandatory Routing for `kraus_generators`
- **Critical Lesson (`Wave 22`)**: `pyramid.kraus_generators` has shape `(R, d, d)` (`ndim == 3`). Never let a substring filter route `kraus_generators` to Euclidean `AdamW(weight_decay=0.0)`, or its Stiefel error $\|K_r^\top K_r - I\|_F / \sqrt{d}$ will drift to `> 20.0`.
- **Canonical Setup**: Route `kraus_generators` and 2D orthogonal projection matrices to `StiefelMuon(..., stiefel_penalty=0.02, polar_method="polar_express", use_normuon=True, use_twin_rail=True)` and call `model.retract_kraus_generators()` every 2 steps, which keeps `kraus_stiefel_err <= 0.015` smoothly with zero momentum shocks.

### 5.2 Choosing Between NanoGPT `WSD` and Smith 2-Cycle `CLR` (`arXiv:1506.01186`)
- **Fixed-Horizon Runs (`modded_nanogpt_wsd`)**: Best when the total token horizon $T_{\max}$ is fixed in advance and only the final checkpoint at $\tau = 1.00$ matters (`5.4372` at $d=256$, `4.7919` at $d=512$ / `29.49M` tokens). Warning: uncooled intermediate checkpoints at $\tau \in [0.04, 0.60]$ sit at peak LR $\eta_{\max}$ and appear `+0.15` nats worse until the cooldown branch is run (`scripts/isotoken_wsd_cooldown_branch_evaluator.py`).
- **Anytime / Multi-Cycle Runs (`smith_triangular2` with Iso-Integral Scaling)**: Use [`scripts/smith_clr_range_test_vs_wsd_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/smith_clr_range_test_vs_wsd_profiler.py) to find $(\eta_{\text{base}}, \eta_{\max})$ automatically in 40 steps, and use the iso-integral 2-cycle schedule when you need strong calibrated checkpoints at **both** $\tau = 0.50$ (`-0.0452` nats better than uncooled `WSD`) and $\tau = 1.00$.

---

## 6. First-Class Diagnostic & Verification Scripts (`scripts/`)

Before writing any ad-hoc one-liner, check if one of our deterministic CLI tools in `scripts/` already performs the diagnostic:

| Script | Purpose & Example Invocation |
| :--- | :--- |
| [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) | Enforces Rules 1–6 (`<= 1,000` soft / `<= 2,500` hard lines, `<= 8` args, zero semicolons, max nested function length). |
| [`scripts/fineweb_convex_vs_concave_slope_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_convex_vs_concave_slope_tracker.py) | Standardized **Iso-Token (`2.46M`, `7.37M`, `14.75M`, `29.49M`, `44.24M`) & Iso-FLOP (`6 N D`)** comparison table and joint Chinchilla scaling-law fitter across all waves (`w19..w28`). |
| [`scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py) | Verifies empirically on real `FineWeb-Edu` sequences that the $N=2^{16}$ Wavelet-Cholesky pyramid breaks the `3.28` Val CE wall (`< 0.10` nats) and audits proportional vs. fixed wavelet ranks across $d \in \{256, 384, 512\}$. |
| [`scripts/smith_clr_range_test_vs_wsd_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/smith_clr_range_test_vs_wsd_profiler.py) | Runs Leslie Smith's (`arXiv:1506.01186`) exponential LR Range Test on `ParsevalMuPDyadicGPT` and compares `modded_nanogpt_wsd`, `smith_triangular2`, and `smith_1cycle`. |
| [`scripts/parseval_sqrt_d_scale_invariance_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_sqrt_d_scale_invariance_probe.py) | Verifies $\beta_0(d) = c^*\sqrt{d}$ background logit variance invariance across $d \in \{128, 256, 384, 512, 768, 1024\}$ and two-beam algebraic scale unification. |
| [`scripts/unit_sphere_mup_transfer_grid_search.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/unit_sphere_mup_transfer_grid_search.py) | Verifies zero-shot $\mu\text{P}$ transfer of $c^* = 1.48$ from a tiny $d_{\text{base}} = 128$ proxy to $d \in \{256, 384, 512, 768, 1024\}$. |
| [`scripts/cholesky_metric_temperature_absorber.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/cholesky_metric_temperature_absorber.py) | Verifies exact mathematical equivalence (`< 1e-5`) of absorbing $\beta_0(d)^{1/2}$ into post-normalization `BlockCholeskyLinear` factors $L \leftarrow \beta_0(d)^{1/4} L$. |
| [`scripts/imagenet1k_quantum_block_ablation_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/imagenet1k_quantum_block_ablation_profiler.py) | Automated leave-one-out ablation runner for quantum building blocks on genuine `ILSVRC ImageNet-1K` (`50,000` val images). |

---

## 7. How to Contribute Back ("Promote-to-Primitive" Protocol)

Whenever you complete an experiment wave or implement a new capability, follow this 4-step checklist to keep `QuantumDLBlocks` modular and discoverable for the next agent:

1. **Extract Reusable Kernels & Layers Out of `examples/`**:
   - If a custom `@torch.compile` function, Triton kernel, or `torch.autograd.Function` is useful beyond a single verification script, place it in [`src/qdlblocks/kernels/`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/__init__.py) or [`src/qdlblocks/layers/`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/__init__.py) (`<= 350` lines) and export it in `__all__`.
2. **Keep Per-Wave Architectures Clean & Flag-Free (Rule 6)**:
   - Place new self-contained model formulations in `src/qdlblocks/models/` and export them in [`src/qdlblocks/models/__init__.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/__init__.py) without adding boolean toggle flags to earlier wave models.
3. **Promote Repeatable Diagnostics to `scripts/`**:
   - Any diagnostic or comparison run more than once belongs in `scripts/` (with `--json-out .cache/<name>.json`) and should be logged in [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md).
4. **Update This Catalog (`docs/BUILDING_BLOCKS.md`) and `README.md`**:
   - Add the new component to the Quick-Reference Status Matrix in Section 1 above (marking any superseded component as `FROZEN` or `LEGACY`) so the very next agent session knows immediately what state the building blocks are in.
