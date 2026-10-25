# Experiment Plan 048 (`Wave 33`): Progress-Ramped Multi-Subspace Coincidence Warmup and Per-Layer Pre-Attention 1D Wavefunction Convolution Blocks Replacing the Post-Attention Channel U-Net

**Date**: October 24, 2026  
**Wave**: `Wave 33` (`EXPERIMENT_PLAN_048`)  
**Status**: Implementing & Verifying  
**Target Domain**: `fineweb_edu_local` & `micro_verification` (Section 6 Priority Items 1 & 2 of [`PROJECT_SUMMARY_WAVES_26_TO_32.md`](file:///home/festivity/agy/QuantumDLBlocks/PROJECT_SUMMARY_WAVES_26_TO_32.md#L243-L261))

---

## 1. Motivation & First-Principles Mathematical Formulation

### 1.1 Priority 1 — Why Un-Warmed `M=4` Coincidence Penalized Step-150 CE (`+0.14` Nats) & How Progress-Ramped Coincidence Warmup Resolves It
In `Wave 32` ([`EXPERIMENT_PLAN_047.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_047.md)), separating the four dyadic wavelet scales into [`WaveletScaleState`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py) (`coarse_256`, `mid_2048`, `sub_8192`, `fine_65536`) eliminated cross-level L2 norm cannibalization (`25.00%` exact energy per level) and raised detector effective rank $R_{\text{sharp}}$ to `44.88` (`13.01 / 64` coarse semantic clustering vs. `33.79 / 64` fine lexical diversity). However, at step 150 (`1.23M` tokens), the un-warmed `M=4` runs trailed the `M=1` single-sphere baseline by `+0.14` to `+0.16` nats (`6.9476–6.9669` vs. `6.8037`).

Two mathematical mechanisms caused this early-step friction:

1. **Token-Exposure Asymmetry Between Coarse Roots (`wav_256`) and Fine Leaves (`wav_65536`)**:
   - In each training step of $B \times T = 16 \times 512 = 8,192$ tokens, each of the $256$ coarse roots in `wav_256` receives an average of $8,192 / 256 = 32$ gradient updates per step ($4,800$ updates by step 150), whereas each of the $65,536$ fine leaves in `wav_65536` receives an average of only $8,192 / 65,536 = 0.125$ updates per step ($\sim 18.75$ updates by step 150; $\sim 225$ updates by step 1,800; $\sim 450$ updates by step 3,600).
   - Applying full 4-way geometric coincidence $\gamma_{\text{coinc}} = 0.22$ and inter-subspace variance penalty $\eta_{\text{var}} = 0.14$ at $\tau = 0$ penalizes candidates whenever the under-trained `fine_65536` subspace ($m=3$) has not yet aligned with the rapidly converging `coarse_256` ($m=0$) and `mid_2048` ($m=1$) subspaces.
   - Similarly, forcing hard 4-subspace rescaling (`subspace_l2_normalize`) directly onto the 50,257-way base inner product $\langle \psi_t \mid m_{\text{sharp}, v} \rangle$ at $\tau = 0$ alters the initialized even/odd balance between `a_self` (even indices) and `pred_wav_2048` (`pred_bucket[:, 0::2] *= 0.25`, odd-dominant bigram predecessor) before fine leaves have moved from their initialization.

2. **Coordinate Alignment Between `_fold_to_level_dim` and `to_interleaved_product_of_spheres`**:
   - In `Wave 32`, `_fold_to_level_dim` folded $(..., d) \to (..., d/4)$ via `x.reshape(..., 4, d // 4).mean(dim=-2)` (grouping indices $\{j, j+64, j+128, j+192\}$ into slot $j$), whereas `to_interleaved_product_of_spheres` stacked the 4 levels along `dim=-1` (`[..., d // 4, 4] -> [..., d]`), placing slot $j$ of level $m$ at coordinate $4j + m$.
   - Aligning `_fold_to_level_dim` to average local quartets `x.reshape(..., d // 4, 4).mean(dim=-1)` (and adding the exact inverse `WaveletScaleState.from_interleaved(x)`) ensures that index $j \in \{0, \dots, d/4 - 1\}$ of every level $m \in \{0, 1, 2, 3\}$ corresponds to the same local spectral quartet $\{4j, 4j+1, 4j+2, 4j+3\}$ when interleaved into $\mathbb{R}^d$, and including the 2,048-node predecessor transition term `pred_wav_2048` in `mid_2048` (`idx_m == 1`) preserves full bigram transition fidelity in the separated detector state.

3. **Progress-Ramped Coincidence Warmup Schedule ($\tau \in [0, 1]$)**:
   With normalized training progress $\tau = s / S \in [0, 1]$ and linear warmup ramp $r(\tau) = \operatorname{clamp}\!\left(\frac{\tau - \tau_{\text{start}}}{\tau_{\text{end}} - \tau_{\text{start}}}, 0, 1\right) = \tau$ (for $\tau_{\text{start}} = 0, \tau_{\text{end}} = 1$), we schedule:
   $$\gamma_{\text{coinc}}(\tau) = \gamma_{\text{coinc}, \max} \cdot \tau = 0.22\,\tau, \qquad \eta_{\text{var}}(\tau) = \eta_{\text{var}, \max} \cdot \tau = 0.14\,\tau$$
   $$w_{\text{coarse}}(\tau) = w_{\text{coarse}, \max} \cdot \big(c_{\text{coarse}, 0} + (1 - c_{\text{coarse}, 0})\,\tau\big), \qquad \rho_{\text{bal}}(\tau) = \rho_{\text{bal}, \max} \cdot \tau$$
   - Inside `WaveletScaleState` and the 256-root M-subspace coincidence observable (`compute_multi_subspace_coincidence_observable`), all 4 wavelet subspaces remain strictly normalized on $\left(\frac{1}{2}\mathbb{S}^{d/4 - 1}\right)^4$ (`25.00%` energy each) at all steps $s \ge 0$.
   - In the 50,257-way base readout (`subspace_coincidence_dyadic_chunked_ce`), the state and detector smoothly interpolate on $\mathbb{S}^{d-1}$ with weight $\rho_{\text{bal}}(\tau) = \rho_{\text{bal}, \max} \cdot \tau$ from undistorted unit-sphere normalization at $\tau = 0$ toward subspace-balanced normalization as $\tau \to 1$, while $\gamma_{\text{coinc}}(\tau) = 0.22\,\tau$ and $\eta_{\text{var}}(\tau) = 0.14\,\tau$ ramp in the 4-way geometric coincidence boost as the $65,536$ fine leaves accumulate token updates.

---

### 1.2 Priority 2 — Per-Layer Pre-Attention `MultiRateWavefunctionConvBank` (`PreAttnMultiRateWaveletConvBlock`) Replacing the Post-Attention Channel U-Net
In `Wave 32`, `MultiRateWavefunctionConvBank` was applied only once at the input embedding stem prior to Layer 0, while every backbone layer (`ParsevalWaveletBlock`) still executed:
$$\text{BornAttention}(\psi) \;\longrightarrow\; \text{Post-Attention Channel Sweldens U-Net } (d \to d/2) \;\longrightarrow\; \text{Mach-Zehnder Interferometer}$$

In `Wave 33`, we introduce **`PreAttnMultiRateWaveletConvBlock`**, moving [`MultiRateWavefunctionConvBank`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py) **inside every backbone block $\ell \in \{0, \dots, L-1\}$ prior to [`SubspaceProjectorBornAttention`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/attention.py)** and removing the post-attention channel U-Net (`WaveletCholeskyMultiscaleBranch` + `coarse_causal_chol`) from the block entirely:

1. **Step 1 — Per-Layer Pre-Attention 1D Strided Wavefunction Convolution (`(4, 2, 1, 1)` Temporal Dilations)**:
   Unpack the layer input $\psi^{(\ell)} \in \mathbb{R}^{B \times T \times d}$ into its 4 separated wavelet scale tensors $\Psi^{(\ell)} = \operatorname{from\_interleaved}(\psi^{(\ell)}) = (\psi^{(0)}, \psi^{(1)}, \psi^{(2)}, \psi^{(3)})$, evolve them along the **sequence/time axis $t$** at temporal dilations $(4, 2, 1, 1)$ via `conv_state = self.pre_attn_conv.forward_levels(...)`, and form the pre-attention wavefunction:
   $$\psi_{\text{conv}}^{(\ell)} = \operatorname{to\_interleaved\_product\_of\_spheres}\!\left(\operatorname{MultiRateConv}^{(\ell)}(\Psi^{(\ell)})\right) \in \mathcal{M}_{d, 4} \subset \mathbb{S}^{d-1}$$
   $$\psi_{\text{pre\_attn}}^{(\ell)} = \operatorname{l2\_normalize}\!\left(\psi^{(\ell)} + \tanh(g_{\text{conv}}^{(\ell)}) \odot \psi_{\text{conv}}^{(\ell)}\right) \in \mathbb{S}^{d-1}$$
   Because every layer $\ell \in \{0, \dots, L-1\}$ runs a causal 1D strided convolution at dilations $(4, 2, 1, 1)$ *before* attention, the queries and keys $Q_t^{(\ell)}, K_\tau^{(\ell)}$ entering Born attention at layer $\ell$ aggregate multi-rate temporal wavefunction packets across an expanding causal receptive field of $1 + 8(\ell + 1)$ tokens at the coarse scale without a single acausal leak!
2. **Step 2 — Causal Quantum Born Attention with Cross-Layer $V^{(0)}$ Anchor**:
   $$\psi_{\text{attn}}^{(\ell)} = \operatorname{SubspaceProjectorBornAttention}\!\left(\psi_{\text{pre\_attn}}^{(\ell)},\; \cos, \sin,\; V_{\text{mod}}^{(0)},\; \text{doc\_ids}\right)$$
   $$\psi_a^{(\ell)} = \operatorname{l2\_normalize}\!\left(\psi_{\text{pre\_attn}}^{(\ell)} + \tanh(g_{\text{attn}}^{(\ell)}) \odot \psi_{\text{attn}}^{(\ell)}\right)$$
3. **Step 3 — Mach-Zehnder Two-Beam Phase Interferometer Between Post-Attention State $\psi_a^{(\ell)}$ and Pre-Attention Wavefunction Packet $\psi_{\text{conv}}^{(\ell)}$**:
   $$\psi_{\text{combined}}^{(\ell)} = \operatorname{l2\_normalize}\!\left(0.55\,\operatorname{Interferometer}(\psi_a^{(\ell)}) + 0.45\,\psi_{\text{conv}}^{(\ell)}\right)$$
   $$\psi^{(\ell+1)} = \operatorname{l2\_normalize}\!\left(\psi_a^{(\ell)} + \tanh(g_{\text{fused}}^{(\ell)}) \odot \psi_{\text{combined}}^{(\ell)}\right)$$
   returning $(\psi^{(\ell+1)}, \psi_{\text{conv}}^{(\ell)})$ with **zero post-attention channel U-Net overhead**.

---

## 2. Hypotheses & Quantitative Acceptance Gates

- **Hypothesis H1 (Per-Layer Pre-Attention 1D Wavefunction Receptive Field Expansion & Progress-Ramped Coincidence Warmup Invariants)**:
  1. In [`scripts/multirate_wavefunction_conv_receptive_field_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multirate_wavefunction_conv_receptive_field_profiler.py), stacking $L = 4$ `PreAttnMultiRateWaveletConvBlock` layers expands the pre-attention temporal receptive field entering `SubspaceProjectorBornAttention` to lags $\Delta t \in \{4, 8, 12, 16\}$ (`lag8_to_lag0_ratio >= 0.02` at Layer 3 vs. `0.0` for post-attention channel U-Net at Layer 0 pre-attention) while maintaining strict causality (`max_future_leakage < 1e-6`).
  2. Under `coincidence_gamma(tau) = 0.22 * tau` and `subspace_var_penalty(tau) = 0.14 * tau`, early-step unconverged fine-leaf CE distortion at $\tau = 0.0$ is eliminated (`early_tau0_ce_penalty <= 1e-5` nats vs. `+0.14` nats un-warmed) while full $\tau = 1.0$ coincidence gating suppresses single-subspace runner-up impostors by `>= 1.0` nat.
- **Hypothesis H2 (Closing the Early-Step `M=4` Gap at 150 Steps & Outperforming Over `1,800`-Step / `14.75M`-Token Horizon on `FineWeb-Edu`)**:
  1. In the controlled `150`-step (`1.23M`-token) `FineWeb-Edu` verification (`d=256, L=4, B=16, T=512`), `w33_m4_progress_ramped_pre_attn_conv_d256` (with per-layer pre-attention `MultiRateWavefunctionConvBank` + progress-ramped coincidence warmup) improves Validation CE over the un-warmed `Wave 32` `M=4` baseline (`6.9669`) by **`>= 0.14` nats** (`<= 6.82` Val CE, matching or beating the `M=1` single-sphere baseline `6.8037` while preserving exact `25.00%` per-level energy and `kraus_stiefel_err <= 0.05`).
  2. Over the `1,800`-step (`14.75M`-token) scaling horizon at `d=256, L=6, B=16, T=512` (`w33_level_sep_pre_attn_conv_warmup_d256_1800steps`), `LevelSeparatedWaveletConvGPT` achieves **`< 5.05` Val CE by `7.37M` tokens** (beating `Wave 28` `5.4372` and `Wave 29` `5.0798`) and **`< 4.88` Val CE by `14.75M` tokens** (beating `Wave 25` `5.1980` and `Wave 26` `5.2116` by `> 0.30` nats) while keeping all 4 wavelet levels at exact `25.00%` energy.

---

## 3. Planned Changes (Rules 1–8 Compliant)

1. **[`src/qdlblocks/layers/multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py) (`UPDATE`)**:
   - Add `WaveletScaleState.from_interleaved(x: torch.Tensor) -> WaveletScaleState` so any $(..., d)$ interleaved state can be unpacked into its 4 separated wavelet scale tensors `(coarse_256, mid_2048, sub_8192, fine_65536)` of shape `(..., d // 4)` in $O(1)$ view time before per-layer 1D wavefunction convolution.
2. **[`src/qdlblocks/layers/subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) (`UPDATE`)**:
   - Add `subspace_balance_blend: float = DEFAULT_SUBSPACE_BALANCE_BLEND` (`1.0` default for backward compatibility) to `SubspaceCoincidenceSpec` and `blend_subspace_l2_normalize(...)` so progress-ramped models can smoothly ramp base-readout subspace projection $\rho(\tau) \in [0, 1]$ alongside $\gamma_{\text{coinc}}(\tau) = 0.22\,\tau$ and $\eta_{\text{var}}(\tau) = 0.14\,\tau$ while keeping the 256-root coincidence observable strictly normalized on $\mathcal{M}_{d, M}$.
3. **[`src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py) (`UPDATE`)**:
   - Add progress-ramped coincidence scheduling methods `current_coincidence_ramp()`, `current_coincidence_gamma()`, `current_subspace_var_penalty()`, `current_coarse_coincidence_weight()`, and `current_subspace_balance_blend()` controlled by `coincidence_warmup_start_tau` and `coincidence_warmup_end_tau`.
4. **[`src/qdlblocks/models/level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) (`UPDATE`)**:
   - Implement **`PreAttnMultiRateWaveletConvBlock`**, placing `MultiRateWavefunctionConvBank` (`(4, 2, 1, 1)` temporal dilations) *before* `SubspaceProjectorBornAttention` in every backbone block and removing the post-attention channel U-Net (`WaveletCholeskyMultiscaleBranch`) entirely.
   - Align `_fold_to_level_dim` with `to_interleaved_product_of_spheres`, include `pred_wav_2048` in `mid_2048` (`idx_m == 1`) of `synthesize_level_separated_detectors()`, and enable progress-ramped coincidence warmup (`coincidence_gamma(tau) = 0.22 * tau`, `subspace_var_penalty(tau) = 0.14 * tau`) by default on `LevelSeparatedWaveletConvConfig`.
5. **[`scripts/multirate_wavefunction_conv_receptive_field_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multirate_wavefunction_conv_receptive_field_profiler.py) (`NEW`, Promoted from `scripts/TOOLING_BACKLOG.md`)**:
   - Build-Before-Launch CLI diagnostic auditing per-layer pre-attention temporal receptive field expansion across lags $\Delta t \in [0, 16]$, strict causality, and progress-ramped coincidence warmup trajectories.
6. **[`tests/test_plan048_coincidence_warmup_and_pre_attn_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan048_coincidence_warmup_and_pre_attn_conv.py) (`NEW`)** & **[`examples/58_verify_plan048_coincidence_warmup_and_pre_attn_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/58_verify_plan048_coincidence_warmup_and_pre_attn_conv.py) (`NEW`)**.

---

## 4. Verification & Testing Plan

1. **AST Linter Gate**: Run `scripts/ast_signature_and_length_linter.py` on all modified and newly created files (`PASS`, `<= 1,000` SLOC, `<= 8` args, zero semicolons).
2. **Build-Before-Launch Receptive-Field & Warmup Profiler**: Run `scripts/multirate_wavefunction_conv_receptive_field_profiler.py` on CPU and verify `H1` in `.cache/multirate_wavefunction_conv_receptive_field_profiler.json`.
3. **Unit & Regression Tests**: Run `pytest tests/test_plan047_level_separated_wavelet_conv_gpt.py tests/test_plan048_coincidence_warmup_and_pre_attn_conv.py`.
4. **GPU Verification (`150`-Step Controlled Ablation + `1,800`-Step / `14.75M`-Token Horizon Scaling Run)**: Run `examples/58_verify_plan048_coincidence_warmup_and_pre_attn_conv.py` with `StandardizedExperimentLogger` and sync to `:8765` via `scripts/sync_waves_to_webui.py --trigger-reload`.

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

### 5.1 Existing Scripts Used
- [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
- [`scripts/subspace_specialization_mutual_information_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_specialization_mutual_information_probe.py)
- [`scripts/multi_subspace_coincidence_margin_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multi_subspace_coincidence_margin_probe.py)
- [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py)

### 5.2 `Build-Before-Launch` Script Promoted from Backlog
- **[`scripts/multirate_wavefunction_conv_receptive_field_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multirate_wavefunction_conv_receptive_field_profiler.py)**: Promoted from `scripts/TOOLING_BACKLOG.md` to audit per-layer pre-attention temporal receptive field expansion across lags $\Delta t \in [0, 16]$, zero future leakage, and progress-ramped coincidence warmup trajectories.

### 5.3 `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/subspace_energy_and_rank_drift_tracker.py`: Parses live training logs or checkpoints and plots per-subspace energy fractions $\|\psi^{(m)}\|_2^2$ and per-subspace participation ratios $R_{\text{sharp}}^{(m)}$ across training milestones (`2.46M`, `7.37M`, `14.75M`, `29.49M` tokens).
2. `scripts/factorized_256x256_hierarchical_softmax_bench.py`: Benchmarks exact 2-level $256 \times 256$ hierarchical dyadic softmax (`512` dot products/token) + Rank-4 orthogonal context-modulated readout metric (`Lever 4` of the `< 3.28` at `150M` tokens roadmap).

---

## 6. Constants & Calibration Ledger (`Rule 7` of `docs/SCRIPT_BEST_PRACTICES.md`)

| Constant Name | Module | Value | Mathematical / Physical Meaning & Empirical Source |
| :--- | :--- | :---: | :--- |
| `DEFAULT_SUBSPACE_BALANCE_BLEND` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `1.0` | Default spherical interpolation weight $\rho \in [0, 1]$ between unit-sphere and Product-of-Spheres base readout normalization (`1.0` when un-warmed; ramped as $\rho_{\max} \cdot \tau$ when warmup is enabled). |
| `DEFAULT_COINCIDENCE_WARMUP_START_TAU` | [`parseval_mup_subspace_coincidence_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py) | `0.0` | Normalized progress $\tau_{\text{start}} \in [0, 1)$ at which linear coincidence warmup begins (`coincidence_gamma(tau) = 0.22 * tau`). |
| `DEFAULT_COINCIDENCE_WARMUP_END_TAU` | [`parseval_mup_subspace_coincidence_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py) | `1.0` | Normalized progress $\tau_{\text{end}} \in (0, 1]$ at which coincidence reaches full strength (`0.22` and `0.14`). |
| `DEFAULT_COARSE_WARMUP_FLOOR` | [`parseval_mup_subspace_coincidence_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py) | `0.20` | Initial fraction of `coarse_coincidence_weight` active at $\tau = 0$ (since 256 coarse roots converge 256x faster than fine leaves). |
| `DEFAULT_WARMED_BASE_SUBSPACE_BLEND_MAX` | [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | `0.25` | Terminal Product-of-Spheres base-readout blend $\rho_{\max}$ at $\tau = 1.0$ on `LevelSeparatedWaveletConvGPT`. |
| `LEVEL_ORTHO_RESIDUAL_WEIGHT` | [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | `0.35` | Terminal weight at $\tau = 1.0$ blending level-separated detectors `m_sep` with full-pyramid detectors (ramped as $w_{\text{sep}}(\tau) = 0.05 + 0.30\,\tau$). |
| `LEVEL_ORTHO_RESIDUAL_MIN_WEIGHT` | [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | `0.05` | Step-0 ($\tau = 0$) weight on `m_sep` before fine `wav_65536` leaves accumulate token updates. |
| `PRE_ATTN_BLOCK_INTERFEROMETER_MIX` | [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | `0.55` | Mixing weight on `OvercompleteTwoBeamInterference(psi_a)` vs. `0.45` on pre-attention `psi_conv` inside `PreAttnMultiRateWaveletConvBlock`. |
| `PRE_ATTN_BLOCK_CONV_INIT_SCALE` | [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | `0.24` | Depth-scaled (`0.24 / sqrt(l + 1)`) initial gate mixing per-layer pre-attention `MultiRateWavefunctionConvBank` into `psi_pre_attn`. |

---

## 7. Empirical Results & Post-Mortem Analysis (`Wave 33` Verification)

### 7.1 `H1` Verification (`scripts/multirate_wavefunction_conv_receptive_field_profiler.py`)
- **Pre-Attention Temporal Receptive Field Expansion (`all_blocks_are_pre_attn_conv = True`, `has_legacy_post_attn_channel_unet = False`)**:
  - Zero acausal future leakage (`max_future_leakage = 0.0`, `strict_causality_passed = True`).
  - Cumulative pre-attention impulse sensitivity norms at query position $t = 24$ across lags $\Delta t \in \{0, 1, 2, 4, 8, 12, 16\}$:
    - **Layer 0**: `lag_0 = 0.065857`, `lag_1 = 0.001790`, `lag_2 = 0.001148`, `lag_4 = 0.000921`, `lag_8 = 0.000495`, `lag_12 = 0.000000`, `lag_16 = 0.000000`
    - **Layer 1**: `lag_0 = 0.065722`, `lag_1 = 0.003134`, `lag_2 = 0.001995`, `lag_4 = 0.001628`, `lag_8 = 0.000893`, `lag_12 = 0.000020`, `lag_16 = 0.000006`
    - **Layer 2**: `lag_0 = 0.065566`, `lag_1 = 0.004255`, `lag_2 = 0.002697`, `lag_4 = 0.002226`, `lag_8 = 0.001240`, `lag_12 = 0.000049`, `lag_16 = 0.000015`
    - **Layer 3**: `lag_0 = 0.065399`, `lag_1 = 0.005235`, `lag_2 = 0.003310`, `lag_4 = 0.002756`, `lag_8 = 0.001555` (`lag8/lag0 = 0.0238`), `lag_12 = 0.000084`, `lag_16 = 0.000026`
- **Progress-Ramped Coincidence Warmup (`coincidence_warmup_passed = True`)**:
  - Early-progress CE penalty at $\tau = 0$: `0.0000` nats (`gamma_c(0) = 0.0`, `lambda_var(0) = 0.0`).
  - Terminal $\tau = 1.0$ (`gamma_c(1) = 0.22`, `lambda_var(1) = 0.14`) runner-up suppression margin gain over `M=1`: **`+3.1458` nats** (`ce_m1 = 0.00017` vs. `ce_warmed_coinc = 0.00001`).

### 7.2 `H2` Controlled 150-Step (`1.23M`-Token) FineWeb-Edu Ablation (`d=256, L=4, B=16, T=512`)

| Run Key | Architecture / Coincidence Mode | Params (`M`) | Step 50 CE | Step 150 Val CE (`PPL`) | Top-1 / Top-50 Acc | `R_sharp` (`Centered`) | Level Ranks (`C256 / M2048 / S8192 / F65536`) | `IsoErr` | `tok/s` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w33_m1_single_sphere_baseline_d256` | `M=1` Single-Sphere (`ParsevalMuPSubspaceCoincidenceGPT`) | `22.13M` | `6.7923` | `6.8090` (`906.00`) | `13.69%` / `47.16%` | `53.12` (`46.63`) | — | `0.0226` | `30,183` |
| `w33_m4_unwarmed_post_attn_unet_d256` | `M=4` Unwarmed Static `gamma=0.22` + Post-Attn Channel U-Net | `22.14M` | `7.0944` | `6.9883` (`1083.88`) | `13.15%` / `45.35%` | `42.60` (`38.77`) | `8.56 / 11.39 / 12.50 / 30.46` | `0.0215` | `26,038` |
| **`w33_m4_progress_ramped_pre_attn_conv_d256`** | **`Wave 33`: Progress-Ramped Warmup + Per-Layer `PreAttnMultiRateWaveletConvBlock`** | **`22.12M`** | **`6.8067`** | **`6.8012` (`898.89`)** | **`13.66%` / `47.14%`** | **`53.11` (`48.19`)** | **`9.37 / 39.72 / 20.64 / 20.51`** | **`0.0216`** | **`22,599`** |

- **Key Diagnostic Insight**:
  - Progress-ramped coincidence warmup + quartet-aligned `_fold_to_level_dim` + per-layer `PreAttnMultiRateWaveletConvBlock` improved step-150 validation cross-entropy by **`-0.1871` nats** over unwarmed `M=4` (`6.8012` vs. `6.9883`) and **surpassed the `M=1` single-sphere baseline at step 150 (`6.8012` vs. `6.8090`, `-0.0078` nats)** while expanding centered sharpness rank to **`48.19`** (`+1.56` above `M=1` and `+9.42` above unwarmed `M=4`) and raising mid-level `r_sharp_mid_2048` by **`3.49x`** (`39.72` vs. `11.39`).

### 7.3 `1,800`-Step (`14.75M`-Token) Horizon Scaling Run (`w33_level_sep_pre_attn_conv_warmup_d256_L6_1800steps`)

| Step | Tokens (`M`) | Progress $\tau$ | `gamma_c(tau)` | `lambda_var(tau)` | `blend(tau)` | Val CE (`Val PPL`) | Top-1 / Top-50 Acc | `R_sharp` (`Centered`) | Level Energies (`C/M/S/F`) | Level Ranks (`C256 / M2048 / S8192 / F65536`) | `IsoErr` | `tok/s` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | `0.000` | `0.0000` | `0.0000` | `0.0000` | `7.9464` (`2825.30`) | `9.55%` / `35.65%` | `9.49` (`15.59`) | `0.25 / 0.25 / 0.25 / 0.25` | `9.76 / 22.89 / 22.78 / 14.04` | `2.2e-7` | — |
| `300` | `2.46M` | `0.167` | `0.0183` | `0.0117` | `0.0208` | `6.2827` (`535.22`) | `15.74%` / `51.64%` | `12.15` (`12.68`) | `0.25 / 0.25 / 0.25 / 0.25` | `6.63 / 31.31 / 15.41 / 19.71` | `0.0001` | `24,524` |
| `600` | `4.92M` | `0.333` | `0.0367` | `0.0233` | `0.0417` | `6.1078` (`449.34`) | `16.61%` / `53.39%` | `15.39` (`18.41`) | `0.25 / 0.25 / 0.25 / 0.25` | `5.50 / 30.00 / 13.70 / 19.15` | `0.0202` | `24,242` |
| `900` | `7.37M` | `0.500` | `0.0550` | `0.0350` | `0.0625` | `6.0052` (`405.55`) | `17.43%` / `54.29%` | `11.08` (`12.40`) | `0.25 / 0.25 / 0.25 / 0.25` | `5.30 / 30.39 / 12.28 / 18.46` | `0.0250` | `24,196` |
| `1200` | `9.83M` | `0.667` | `0.0733` | `0.0467` | `0.0833` | `5.8705` (`354.42`) | `18.10%` / `56.44%` | `8.09` (`8.43`) | `0.25 / 0.25 / 0.25 / 0.25` | `4.31 / 29.31 / 12.16 / 18.12` | `0.0198` | `24,182` |
| `1500` | `12.29M` | `0.833` | `0.0917` | `0.0583` | `0.1042` | `5.6834` (`293.95`) | `19.25%` / `58.19%` | `17.40` (`25.65`) | `0.25 / 0.25 / 0.25 / 0.25` | `3.68 / 29.03 / 12.06 / 17.84` | `0.0090` | `24,151` |
| **`1800`** | **`14.75M`** | **`1.000`** | **`0.1100`** | **`0.0700`** | **`0.1250`** | **`5.5664` (`261.48`)** | **`20.34%` / `59.35%`** | **`17.06` (`25.51`)** | **`0.25 / 0.25 / 0.25 / 0.25`** | **`3.42 / 28.91 / 12.61 / 17.68`** | **`0.0050`** | **`24,290`** |

