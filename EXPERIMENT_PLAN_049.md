# Experiment Plan 049 (`Wave 34`): True Dyadic Readout CE Floor, Symplectic-Preserving `StiefelMuon` + Parseval $\mu\text{P}$ Restoration, & Memory-Profiled Max-Width ($d \in [256, 1024]$) Scaling

---

## 1. Problem Statement & Root-Cause Diagnosis

Following `Wave 33` ([`EXPERIMENT_PLAN_048.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_048.md)), three fundamental questions arose:
1. **Is our minimum achievable cross-entropy (CE) floor calculation (`0.00017` nats at $d=256, M=1$ and `0.00001` nats at $d=256, M=4$) accurate for our actual setup?**
2. **Why did `w33_level_sep_pre_attn_conv_warmup_d256_L6_1800steps` reach only `5.5664` Val CE at `14.75M` tokens (`1,800` steps), whereas `Waves 26–28` reached `5.3490` at `14.75M` tokens (`d=256, L=6`), `5.3603` at `7.37M` tokens (`d=512, L=6`), and `4.7919` at `29.49M` tokens (`d=512, L=6`)?**
3. **How high can we push model width $d$ ($d \in \{256, 384, 512, 640, 768, 1024\}$) on a single 16 GB GPU (`RTX 5060 Ti`, `15,480 MiB` usable VRAM) once we profile and eliminate hidden memory bottlenecks?**

### 1.1 Why the Synthetic CE Floor (`0.00017` / `0.00001` Nats) Diverged from Our Actual Model Readout
Auditing [`parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py), [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py), and [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) revealed three structural discrepancies between the synthetic probe in [`multirate_wavefunction_conv_receptive_field_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multirate_wavefunction_conv_receptive_field_profiler.py) and our actual training/evaluation readout:

1. **Discrepancy A — $M=4$ Coincidence Was Never Applied to the 50,257 Fine Vocabulary Logits**:
   - In [`subspace_coincidence_dyadic_chunked_ce`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py#L245-L253), the 50,257-way fine token loss (`base_loss`) delegates to [`parseval_mup_dyadic_chunked_ce`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py#L157-L161), whose 2D chunk GEMM `torch.mm(psi_s_c, m_s_act.t())` collapses all $M=4$ subspaces into a **single $M=1$ scalar inner product**:
     $$c_{t,v} = \langle \psi_{\text{base}, t}, m_{\text{base}, v} \rangle = \frac{1}{M}\sum_{m=1}^M c_{t,v}^{(m)}$$
   - Meanwhile, the $M=4$ coincidence observable $\gamma_c \prod_{m=1}^4 (\hat{c}_{t,g}^{(m)})_+^{1/4} - \lambda_{\text{var}}\operatorname{Var}_m(\hat{c}_{t,g}^{(m)})$ was **only** applied to the 256 coarse bucket centroids (`compute_coarse_cluster_centers`).
   - Because Subspaces $m=1, 2, 3$ represent zero-mean detail wavelets (`mid_2048`, `sub_8192`, `fine_65536`), averaging `m_sharp` across all $\sim 196.3$ leaves in coarse bucket $g = \text{perm\_v\_to\_leaf}[v] \gg 8$ **cancels out Subspaces $1, 2, 3$**! When `subspace_l2_normalize` rescales those cancelled detail subspaces back to norm $1/2$ each ($75\%$ of the centroid's total norm!), the per-subspace cosines between a perfectly aligned fine query $\psi_t = m_{\text{sharp}, v^*}$ and its own coarse centroid `centers_norm[g(v*)]` drop to **`[0.6572, 0.1795, 0.0790, 0.2331]`**, creating an artificial `2.1026`-nat coarse CE floor (`+0.04037` nats after `0.0192` blending — **`4,037x` higher** than `0.00001` nats).
2. **Discrepancy B — Dyadic Tree Sibling & Cousin Detector Collinearity (`S_sib,8`, `S_mid,32`, `S_coarse,256`)**:
   - Whereas the synthetic probe used $K=256$ i.i.d. isotropic Gaussian vectors, our $V=50,257$ detectors are synthesized from a 4-level binary wavelet tree (`256 -> 2048 -> 8192 -> 65536` with leaf index $p = \text{perm\_v\_to\_leaf}[v]$).
   - In the level-separated detector table `m_sep`, Subspaces $m=0, 1, 2$ ($75\%$ of total norm energy) are **100% identical** for all 8 sibling leaves sharing $p \gg 3 = p' \gg 3$. Even if Subspace $m=3$ (`fine_65536`) is antipodal ($c^{(3)} = -1$), sibling cosine similarity in `m_sep` is bounded below by $\frac{3}{4} - \frac{1}{4} = 0.50$ (empirically **`0.7212`** in `m_sep` and **`0.3997` (`max = 0.7627`)** in blended `m_sharp` at $\tau=1.0$).
3. **Discrepancy C — Convex-Peak Boost on Correlated Siblings & GPU Temperature Damping**:
   - Under the centered Convex-Peak Traceless Born observable $\widehat{O}_{t,v} = (c_{t,v} - \mu_t) + \gamma\big(\max(c_{t,v} - \mu_t, 0)^2 - \tau_{+,t}\big)$, a sibling detector with positive overlap $\rho_{\text{sib}} > \mu_t$ also receives a positive quadratic boost $+\gamma(\rho_{\text{sib}} - \mu_t)^2$, so the target-to-sibling margin is:
     $$\Delta \widehat{O}_t(v^*, v_{\text{sib}}) = (\alpha - \rho_{\text{sib}})\Big[1 + \gamma(\alpha + \rho_{\text{sib}} - 2\mu_t)\Big]$$
   - Furthermore, GPU readout damping in `ParsevalMuPDyadicGPT.build_readout_spec` scales $\beta_{\text{eff}}(d, \tau) \propto \sqrt{d}\,(256/d)^{0.32} = d^{0.18}$ (`14.55` at $\tau=0, d=256$; `23.16` at $\tau=1, d=256$; `26.24` at $\tau=1, d=512$).

### 1.2 Seven Compounding Bugs That Caused the `Wave 33` `14.75M`-Token Loss Gap (`5.5664` Val CE)
1. **Bug 1 (Verifier Used Flat Single-Group `AdamW(lr=3e-3)` + `build_parseval_mup_ngram_optimizers` Excluded All 2D Backbone Matrices from `StiefelMuon`)**:
   - [`examples/58_verify_plan048_coincidence_warmup_and_pre_attn_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/58_verify_plan048_coincidence_warmup_and_pre_attn_conv.py) used a flat single-group `torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)` instead of `StiefelMuon` + Parseval $\mu\text{P}$ `AdamW`.
   - Worse, in [`build_parseval_mup_ngram_optimizers` (`parseval_mup_ngram_speedrun_gpt.py:350-364`)](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py#L350-L364) (which `Waves 30–33` all call), the `elif p.ndim == 2 and min(p.shape) >= 32 and max(p.shape) <= 2048: muon_params.append(p)` branch from `ParsevalMuPDyadicGPT` (`Wave 28`) was omitted! Every 2D `qkv_proj`, `out_proj`, and `interferometer` matrix was dumped into `body_params` (`AdamW` at `lr=0.0035`, **`7.4x` lower** than `StiefelMuon`'s `0.026`, with zero Polar Express orthogonalization).
2. **Bug 2 (`conv_bank` Cholesky Factors Subjected to `weight_decay = 0.01` & `wav_65536` Under-Updated)**:
   - In [`multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py), the `BlockCholeskyLinear` modules are named `coarse_conv`, `mid_conv`, `sub_conv`, `fine_conv` with parameters `.diag_raw` and `.tril_raw`. Because neither `"conv"` nor `"diag_raw"` / `"tril_raw"` appeared in `_CHOL_KEYS = ("chol", "theta", "gate", "logit_scale")`, all SPD Cholesky factors inside `MultiRateWavefunctionConvBank` received `weight_decay = 0.01`.
   - Meanwhile, `wav_65536` (`65,536` rows, sampled only $8192/65536 = 0.125$ times/step) received the same LR (`0.0065`) as dense `wav_256`.
3. **Bug 3 (`blend_subspace_l2_normalize` Erased `readout_chol` & Absorbed $\sqrt{\beta_0}$ Norm)**:
   - In [`subspace_coincidence_born.py:245-250`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py#L245-L250), calling `blend_subspace_l2_normalize` *after* `self.readout_chol(...)` normalized `psi_sharp` and `m_sharp` back to unit norm `1.0`—erasing both the learned `readout_chol` SPD metric scaling and any absorbed $\sqrt{\beta_0}$ factor from `absorb_beta_into_readout_cholesky()`!
4. **Bug 4 (Symplectic `(a_self, b_orth)` Basis Destruction in `_fold_to_level_dim` & `to_interleaved_product_of_spheres`)**:
   - In PPMI-SVD initialization (`initialize_parseval_mup_with_ppmi_svd`), self-embeddings `a_self` occupy **even** coordinates (`2k`) and orthogonalized predecessor contexts `b_orth` occupy **odd** coordinates (`2k+1`), and `apply_symplectic_swap` swaps `(2k, 2k+1)`.
   - `_fold_to_level_dim` (`x.reshape(..., 64, 4).mean(dim=-1)`) averaged 2 even (`a_self`) and 2 odd (`b_orth`) coordinates together, and `to_interleaved_product_of_spheres` stacked `[c3, c2, c1, c0]` into `[4j, 4j+1, 4j+2, 4j+3]`, causing `apply_symplectic_swap` to swap `c3` with `c2` instead of `a_self` with `b_orth`, and causing `(1 - w_sep) * psi_strided + w_sep * psi_sep` to sum two incompatible coordinate bases (dropping step-0 `R_sharp` from `13.75` in `Wave 28` to `9.49` in `Wave 33` before step 1).
5. **Bug 5 (`PreAttnMultiRateWaveletConvBlock` Deleted Post-Attention `WaveletCholeskyMultiscaleBranch` & Subtracted `psi_conv` via `psi_diff`)**:
   - `PreAttnMultiRateWaveletConvBlock` removed the post-attention `WaveletCholeskyMultiscaleBranch`, diluted `OvercompleteTwoBeamInterference` (`frame_expansion=1.0`) with `0.45 * psi_conv` (re-injecting the pre-attention state after attention), and returned `psi_conv` as `blk_diff`—so `psi_out = l2_normalize(psi - lam * psi_diff)` actively subtracted the pre-attention signal from the output instead of subtracting zero-mean high-frequency wavelet noise (`s_high - d_high`).
6. **Bug 6 (Missing `V^(0)` First-Layer Value Residual & `HashedSurfaceNGramProjector`)**:
   - `PreAttnMultiRateWaveletConvBlock` called `self.attn(psi_pre)` without `v_0` (`return_v=True`), and `LevelSeparatedWaveletConvGPT` omitted `HashedSurfaceNGramProjector` (`65,536` surface bigram/trigram buckets).
7. **Bug 7 (256-Centroid Detail Cancellation in `compute_coarse_cluster_centers`)**:
   - As diagnosed in Discrepancy A above, averaging `m_sharp` across $\sim 196$ leaves per coarse bucket cancelled Subspaces $1, 2, 3$ and added an irreducible coarse noise floor during training.

### 1.3 Hidden Memory & `float32`-Leak Bottlenecks Limiting Width $d$ on 16 GB VRAM
Live profiling on the local **NVIDIA GeForce RTX 5060 Ti (`16,311 MiB` total / `15,480 MiB` usable VRAM)** revealed three major memory traps when scaling $d \in \{256, 384, 512, 640, 768, 896, 1024\}$:
1. **Trap 1 (`SubspaceProjectorBornAttention` Non-Power-of-2 `head_dim` Silent PyTorch Fallback)**:
   - In [`src/qdlblocks/kernels/flash_born.py:458-467`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_born.py#L458-L467), Triton `flash_born_attention` only ran when `head_dim = d // num_heads` was a power of 2 (`16, 32, 64, 128`). Leaving `num_heads = 8` fixed at `d = 384` (`hd = 48`), `d = 640` (`hd = 80`), or `d = 768` (`hd = 96`) silently fell back to PyTorch $O(BHT^2)$ eager attention, inflating peak VRAM by **`+2,715 to +2,756 MiB`**!
   - By adding `BLOCK_D: tl.constexpr = triton.next_power_of_2(HEAD_DIM)` with column mask `mask_d = tl.arange(0, BLOCK_D) < HEAD_DIM` (`other=0.0`) inside `_flash_born_fwd_kernel`, `_flash_born_bwd_kernel`, and `_flash_born_bwd_dq_kernel` in [`src/qdlblocks/kernels/flash_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_born.py) **and** defaulting `num_heads = max(2, d // 64)`, every multiple-of-128 width $d \in \{256, 384, 512, 640, 768, 896, 1024\}$ runs 100% in Triton `flash_born_attention` (even when `num_heads = 8` gives `hd in {48, 80, 96, 112}`).
2. **Trap 2 (Four Hidden `float32` Leaks Forcing the Entire 6-Layer Residual Stream to Run in `float32`)**:
   - **Leak 2A**: In [`wavelet_cholesky_2pow16_gpt.py:191, 263`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py#L191), `embed_tokens` and `embed_dyadic_bigram_phase` gated `bfloat16` casting on `idx.numel() > 8192 or idx.numel() < 1024`. At both `B=8, T=512` (`4,096` tokens) and `B=16, T=512` (`8,192` tokens), `use_bf16` evaluated to `False`, returning `float32` embeddings that infected `.to(psi.dtype)` across the **entire 6-layer backbone**!
   - **Leak 2B**: `_L2NormalizeFn` in [`src/qdlblocks/functional.py:47-56`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/functional.py#L47-L56) was gated by `x.ndim == 3 and x.numel() > 10_000_000`, so all 2D `(50304, d)` detector normalizations and all 3D `(B, T, d)` backbone normalizations (`B*T*d <= 8.39M`) bypassed `_L2NormalizeFn` and saved `float32` copies (`4 bytes/elem`) in autograd.
   - **Leak 2C**: `subspace_l2_normalize` in [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) lacked a custom autograd function, saving both `x_sub.float()` (`4 bytes/elem`) and `x_sub` (`2 bytes/elem`) = `6 bytes/elem` on every call. Adding `_SubspaceL2NormalizeFn` (saving only `y` in `bfloat16` + `r_inv`) cuts its saved memory by `3x`.
3. **Trap 3 (`synthesize_detectors` Saving `323` Autograd Tensors = `4,536 MiB` Held VRAM at `d = 1024`)**:
   - Wrapping `synthesize_detectors()` in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` when `self.training and torch.is_grad_enabled()`, combined with fixing Traps 1 & 2 above, reduces peak `B = 16, T = 512` training VRAM on our 16 GB RTX 5060 Ti as follows:
     - **`d = 256, L = 6, B = 16`**: `4,865 MiB -> 2,857 MiB` (`-2.01 GiB`)
     - **`d = 384, L = 6, B = 16`**: `9,571 MiB -> 4,150 MiB` (`-5.42 GiB`)
     - **`d = 512, L = 6, B = 16`**: `8,575 MiB -> 5,456 MiB` (`-3.12 GiB`)
     - **`d = 640, L = 6, B = 16`**: `13,105 MiB -> 6,843 MiB` (`-6.26 GiB`)
     - **`d = 768, L = 6, B = 16`**: `14,941 MiB -> 8,244 MiB` (`-6.70 GiB`)
     - **`d = 1024, L = 6, B = 16`**: `OOM (>15,480 MiB) -> 11,077 MiB` (`4.4 GiB` free at full `B=16`, or `8,913 MiB` at `B=8, grad_accum=2`!)

---

## 2. Observable Hypotheses & Quantitative Pass/Fail Criteria

### Hypothesis `H1` (True Dyadic Readout CE Floor Formula $\mathcal{L}_{\text{CE}}^{\text{shell}}(d, M, \alpha)$ & $50,257$-Way / $256 \times 256$ $M=4$ Coincidence)
- **Mathematical Definition of $\mathcal{L}_{\text{CE}}^{\text{shell}}(d, M, \alpha)$**:
  For a target token $v^* \in \{0, \dots, V-1\}$ ($V = 50,257$) with 16-bit leaf index $p^* = \text{perm\_v\_to\_leaf}[v^*]$ and query state $\psi_t = \text{normalize}\big(\alpha\,m_{\text{sharp}, v^*} + \sqrt{1 - \alpha^2}\,u_\perp\big)$ ($\alpha \in (0, 1]$, $u_\perp \perp m_{\text{sharp}, v^*}$), partition the $V - 1 = 50,256$ non-target tokens into four disjoint dyadic ancestry shells:
  - $S_{\text{sib}, 8}(p^*) = \{v \neq v^* : p(v) \gg 3 = p^* \gg 3\}$ (cardinality $N_8 \approx 5.37$, empirical mean subspace cosines $\bar{\rho}_8^{(m)}$ and variance $\sigma_8^2$),
  - $S_{\text{mid}, 32}(p^*) = \{v : p(v) \gg 5 = p^* \gg 5,\; p(v) \gg 3 \neq p^* \gg 3\}$ (cardinality $N_{32} \approx 18.41$, mean cosines $\bar{\rho}_{32}^{(m)}$, variance $\sigma_{32}^2$),
  - $S_{\text{coarse}, 256}(p^*) = \{v : p(v) \gg 8 = p^* \gg 8,\; p(v) \gg 5 \neq p^* \gg 5\}$ (cardinality $N_{256} \approx 171.53$, mean cosines $\bar{\rho}_{256}^{(m)}$, variance $\sigma_{256}^2$),
  - $S_{\text{bg}}(p^*) = \{v : p(v) \gg 8 \neq p^* \gg 8\}$ (cardinality $N_{\text{bg}} \approx 50,060.69$, mean $\mu_{\text{bg}} \approx 0$, variance $\sigma_{\text{bg}}^2 = 1 / R_{\text{sharp}}$).
  For each shell $k \in \{8, 32, 256, \text{bg}\}$, let $\Delta \widehat{O}_k^{(M)}(\alpha)$ be the observable margin between target $v^*$ and shell $k$:
  - Under $M=1$ (`parseval_mup_dyadic_chunked_ce`):
    $$\Delta \widehat{O}_k^{(1)}(\alpha) = (\alpha - \bar{\rho}_k)\Big[1 + \gamma(\alpha + \bar{\rho}_k - 2\mu_t)\Big]$$
  - Under true 50,257-way $M=4$ coincidence (`compiled_m4_coincidence_chunk_step`):
    $$\Delta \widehat{O}_k^{(4)}(\alpha) = (\alpha - \bar{\rho}_k) + \frac{\gamma}{4}\sum_{m=0}^3 \!\Big((\hat{\alpha}^{(m)})_+^2 - (\hat{\rho}_k^{(m)})_+^2\Big) + \gamma_c(\tau)\!\left[\prod_{m=0}^3 (\hat{\alpha}^{(m)})_+^{1/4} - \prod_{m=0}^3 (\hat{\rho}_k^{(m)})_+^{1/4}\right] + \lambda_{\text{var}}(\tau)\Big(\operatorname{Var}_m(\hat{\rho}_k^{(m)}) - \operatorname{Var}_m(\hat{\alpha}^{(m)})\Big)$$
  The analytical shell CE floor is:
  $$\mathcal{L}_{\text{CE}}^{\text{shell}}(d, M, \alpha) = \ln\!\left(1 + \sum_{k \in \{8, 32, 256, \text{bg}\}} N_k \exp\!\left(-\beta_{\text{eff}}(d, \tau)\,\Delta \widehat{O}_k^{(M)}(\alpha) + \frac{1}{2}\beta_{\text{eff}}(d, \tau)^2 \tilde{\sigma}_k^2\right)\right)$$
  where $\tilde{\sigma}_k^2 = \operatorname{Var}_{v \in S_k}\!\big(\widehat{O}_{t,v}^{(M)}\big)$ is the within-shell observable variance.
- **Quantitative Pass Criteria**:
  1. $\left|\mathcal{L}_{\text{CE}}^{\text{shell}}(d, M, \alpha) - \mathcal{L}_{\text{CE}}^{\text{empirical}}(d, M, \alpha)\right| \le 0.015$ nats across $\alpha \in \{0.65, 0.80, 1.00\}$.
  2. Fixing `compute_coarse_cluster_centers` (Section 3.2.3) reduces the $\alpha=1.0$ 256-way coarse CE floor from `> 2.00` nats to `< 0.05` nats.
  3. True 50,257-way $M=4$ fine coincidence (`compiled_m4_coincidence_chunk_step`) reduces the $\alpha = 0.80$ dyadic sibling/impostor CE floor by **`>= 3.0x`** relative to $M=1$ fine readout with identical `2 * N_tok * d * V` GEMM FLOPs, and the $256 \times 256$ factorized readout (`factorized_256x256_coincidence_ce`) reduces readout GEMM FLOPs by **`>= 95.0x`**.

### Hypothesis `H2` (Max-Width $d \in \{256, 384, 512, 640, 768, 1024\}$ Memory Optimization & Non-Pow2 Triton Flash-Born Scaling on 16 GB GPU)
- **Independent Variable**: Model width $d \in \{256, 384, 512, 640, 768, 1024\}$ at depth $L = 6$, sequence length $T = 512$, comparing stock (`num_heads = 8`, `float32` embed leak, un-checkpointed detector synthesis) vs. optimized (`BLOCK_D = next_power_of_2(HEAD_DIM)` in `flash_born.py` + `num_heads = d // 64`, `bfloat16` embed stream, `_SubspaceL2NormalizeFn`, `_L2NormalizeFn` threshold `262,144`, and gradient-checkpointed `synthesize_detectors`).
- **Dependent Observables**: Held detector autograd memory (`MiB`), peak step VRAM (`MiB`), Triton Flash-Born active status (`bool`), and training throughput (`tok/s`).
- **Quantitative Pass Criteria**:
  1. Every width $d \in \{256, 384, 512, 640, 768, 1024\}$ runs 100% on the Triton `flash_born_attention` kernel (both for `head_dim = 64` and for non-power-of-2 `head_dim in {48, 80, 96, 112}`).
  2. The memory optimizations reduce peak training VRAM at $d = 768, B = 16$ by **`>= 4,000 MiB`** (fitting in `<= 9,500 MiB`) and enable **$d = 1024, L = 6$ (`~100.4M` params)** to complete forward+backward+optimizer steps at **`B = 16` in `<= 12,500 MiB`** (and `B = 8` in `<= 9,500 MiB`) on our 16 GB GPU with zero OOMs.

### Hypothesis `H3` (Symplectic-Preserving `StiefelMuon` + Parseval $\mu\text{P}$ Restoration & Width Scaling on FineWeb-Edu)
- **Independent Variable**:
  1. **150-Step (`1.23M`-Token) Controlled Diagnostic Ablation at $d = 256, L = 4$**:
     - Arm A (`w34_legacy_w33_flat_adamw_d256`): `Wave 33` config with flat `AdamW(lr=3e-3)` and non-symplectic folding.
     - Arm B (`w34_symplectic_stiefel_muon_mup_d256`): Restored 2D+3D `StiefelMuon` (`retract_every=2`, `lr=0.026`), Parseval $\mu\text{P}$ `AdamW` group routing (`conv_bank` in `_CHOL_KEYS` with `wd=0.0`, `wav_65536` in `wavelet_fine_sparse` at `lr=0.0156`), pre-`readout_chol` subspace normalization, symplectic-pair-preserving folding, restored post-attention `WaveletCholeskyMultiscaleBranch` + `V^(0)` value residual + `HashedSurfaceNGramProjector` + clean pre-attention `MultiRateWavefunctionConvBank` residual, and true 50,257-way $M=4$ coincidence.
  2. **Iso-Token / Multi-Width Scaling Runs (`1,800` Steps = `14.75M` Tokens) Across $d \in \{256, 512\}$ + Max-Width $d = 768$ Verification**:
     - `w34_unified_muon_mup_coincidence_d256_L6_1800steps` ($d=256, L=6$)
     - `w34_unified_muon_mup_coincidence_d512_L6_1800steps` ($d=512, L=6$)
     - `w34_unified_muon_mup_coincidence_d768_L6_300steps` ($d=768, L=6$, `66.9M` params)
- **Quantitative Pass Criteria**:
  1. Step-0 `sharp_eff_rank` at $d=256$ recovers from `9.49` (`Wave 33`) to **`>= 12.50`**.
  2. In the 150-step controlled ablation at $d=256, L=4$, Arm B (`w34_symplectic_stiefel_muon_mup_d256`) beats Arm A (`6.8012`) by **`>= 0.35` nats** (reaching **`<= 6.45` Val CE** at `1.23M` tokens).
  3. At `1,800` steps (`14.75M` tokens), `w34_unified_muon_mup_coincidence_d256_L6_1800steps` beats `Wave 33`'s `5.5664` by **`>= 0.25` nats** (reaching **`<= 5.31` Val CE**), and `w34_unified_muon_mup_coincidence_d512_L6_1800steps` reaches **`<= 5.18` Val CE** with `kraus_stiefel_err <= 0.05`.

---

## 3. Planned Code & Architectural Changes (Complete & Unambiguous Specification)

### 3.1 Canonical Pair-Strided Symplectic Subspace Memory Layout (`src/qdlblocks/layers/multirate_wavefunction_conv.py` & `src/qdlblocks/layers/subspace_coincidence_born.py`)
We commit to **one single canonical layout** across the entire codebase: **Pair-Strided Symplectic Subspace Layout** (`(*shape, d // (2 * M), M, 2)` for $M = 4$, where $d$ is divisible by $2M = 8$).
1. **Why Pair-Strided (`(*shape, d // 8, 4, 2)`) Preserves Exact Symplectic Orthogonality**:
   - In PPMI-SVD initialization (`initialize_parseval_mup_with_ppmi_svd`), every pair of coordinates `(2k, 2k+1)` stores `(a_self[k], b_orth[k])`, and `apply_symplectic_swap(x)` swaps `(2k, 2k+1)`.
   - By grouping coordinates into **symplectic pairs** `(2k, 2k+1)` of size `2` and assigning pair $k$ to subspace $m = k \bmod 4$, every subspace $m \in \{0, 1, 2, 3\}$ consists of $d / 8$ intact `(even, odd)` pairs (`d_level = d // 4` scalars total: $d/8$ even `a_self` coordinates and $d/8$ odd `b_orth` coordinates).
   - Flattening `(*shape, d // 8, 4, 2) -> (*shape, d)` keeps every `a_self` coordinate at an **even** global index `2k` and every `b_orth` coordinate at an **odd** global index `2k+1`! Therefore, `apply_symplectic_swap(x)` works identically whether applied before or after subspace interleaving!
2. **Exact Tensor Operations**:
   - **`_fold_to_level_dim(x: torch.Tensor, level_dim: int) -> torch.Tensor`** in [`src/qdlblocks/models/level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py):
     ```python
     def _fold_to_level_dim(x: torch.Tensor, level_dim: int) -> torch.Tensor:
         d = x.size(-1)
         if d == level_dim:
             return x
         if d > level_dim and d % level_dim == 0 and level_dim % 2 == 0:
             k = d // level_dim
             return x.reshape(*x.shape[:-1], level_dim // 2, k, 2).mean(dim=-2).reshape(*x.shape[:-1], level_dim)
         return x[..., :level_dim]
     ```
     This averages `k` even (`a_self`) coordinates into an even coordinate and `k` odd (`b_orth`) coordinates into an odd coordinate, never mixing `a_self` with `b_orth`.
   - **`WaveletScaleState.to_interleaved_product_of_spheres(self, eps: float = 1e-8, normalize: bool = True) -> torch.Tensor`** in [`src/qdlblocks/layers/multirate_wavefunction_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/multirate_wavefunction_conv.py):
     - If `normalize=True`, calls `normed = self.normalize_levels(eps=eps)`, else `normed = self`.
     - Reshapes each level $c_m$ of shape `(*lead, d_level)` to `(*lead, d_level // 2, 2)`, stacks `(c3_p, c2_p, c1_p, c0_p)` along `dim=-2` to shape `(*lead, d_level // 2, 4, 2)` (multiplied by `0.5`), and reshapes to `(*lead, 4 * d_level)`.
   - **`WaveletScaleState.from_interleaved(cls, x: torch.Tensor) -> WaveletScaleState`**:
     - Given `x` of shape `(*lead, d)` with $d \bmod 8 == 0$, sets `d_level = d // 4`, reshapes `x` to `(*lead, d_level // 2, 4, 2) * 2.0`, unbinds along `dim=-2` into 4 tensors of shape `(*lead, d_level // 2, 2)`, and reshapes each to `(*lead, d_level)`.
   - **`subspace_l2_normalize(x: torch.Tensor, num_subspaces: int = 4, partition_mode: str = "strided", eps: float = 1e-8) -> torch.Tensor`** in [`src/qdlblocks/layers/subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py):
     - When `partition_mode == "strided"` and `d % (2 * num_subspaces) == 0`, reshapes `x.float()` to `(*lead, d // (2 * M), M, 2)`, computes the L2 norm of each subspace $m$ across `dim=(-3, -1)` (`keepdim=True`), scales by `(1.0 / math.sqrt(M)) / norm.clamp_min(eps)`, and reshapes back to `(*lead, d)`. (Falls back to `(*lead, d // M, M)` normalized over `dim=-2` only if `(d // M) % 2 != 0`).

### 3.2 Compiled 50,257-Way $M=4$ Fine Coincidence Kernel & Train/Eval Parity (`src/qdlblocks/kernels/fused_parseval_ce.py` & `src/qdlblocks/layers/subspace_coincidence_born.py`)
1. **Exact Signature & Analytical Forward/Backward of `compiled_m4_coincidence_chunk_step` in [`src/qdlblocks/kernels/fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py)**:
   ```python
   @torch.compile
   def compiled_m4_coincidence_chunk_step(
       w_sharp_m: torch.Tensor,  # (4, C, V_act) bf16 or fp32: raw subspace inner products <psi^{(m)}, m_s^{(m)}>
       w_diff: torch.Tensor,     # (C, V_act) bf16 or fp32: diffuse Kraus bath inner products
       yc_safe: torch.Tensor,    # (C, 1) int64: target token indices in [0, V_act - 1]
       valid_and_scale: tuple[torch.Tensor, torch.Tensor],  # (valid_f: (C, 1) fp32, row_scale: (C, 1) fp32)
       scalars: tuple[float, float, float, float, float, float],  # (beta_0, lam_sc, gamma, gamma_c, lam_var, sub_scale)
       log_prior_act: torch.Tensor | None,  # (1, V_act) fp32 or None
       v_idx: torch.Tensor,      # (C, 1) int64: arange(C).unsqueeze(1)
   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
   ```
   - **Forward Equations**:
     - Unpack `beta_0, lam_sc, gamma, gamma_c, lam_var, sub_scale = scalars` (where `sub_scale = 4.0` converts each subspace inner product $\langle \psi^{(m)}, m_s^{(m)} \rangle \in [-1/4, 1/4]$ to a unit-sphere subspace cosine $u_m = \text{sub\_scale} \cdot w_{\text{sharp}, m} \in [-1, 1]$).
     - Center each subspace cosine across vocabulary dimension `dim=-1`:
       $$c_m = u_m - \frac{1}{V_{\text{act}}}\sum_{v=1}^{V_{\text{act}}} u_{m, v}, \qquad m \in \{0, 1, 2, 3\}$$
     - Compute mean cosine $\bar{c} = \frac{1}{4}(c_0 + c_1 + c_2 + c_3)$, positive parts $a_m = \max(c_m, 0)$, mean quadratic peak $Q = \frac{1}{4}(a_0^2 + a_1^2 + a_2^2 + a_3^2)$, nested-sqrt geometric mean:
       $$p_{01} = \sqrt{a_0 a_1 + \epsilon_{\text{sq}}}, \quad p_{23} = \sqrt{a_2 a_3 + \epsilon_{\text{sq}}}, \quad G = \sqrt{p_{01} p_{23} + \epsilon_{\text{sq}}} - \epsilon_{\text{off}} \qquad (\epsilon_{\text{sq}} = 10^{-8},\; \epsilon_{\text{off}} = 10^{-4})$$
       and cross-subspace variance $V_{\text{sub}} = \frac{1}{4}(c_0^2 + c_1^2 + c_2^2 + c_3^2) - \bar{c}^2$.
     - Combine into uncentered observable $O_{\text{raw}} = \bar{c} + \gamma Q + \gamma_c G - \lambda_{\text{var}} V_{\text{sub}}$, center $\widehat{O} = O_{\text{raw}} - \operatorname{mean}_v(O_{\text{raw}})$, center diffuse bath $\widehat{F} = (w_{\text{diff}} - \operatorname{mean}_v(w_{\text{diff}}))^2 - \operatorname{mean}_v(\dots)$, form logits $\ell = \beta_0 \widehat{O} + \lambda_{\text{sc}} \widehat{F} + \log \pi$, and compute softmax probability $P = \operatorname{softmax}(\ell, \text{dim}=-1)$ and masked loss `chunk_loss_sum = (-gather(log P, yc_safe) * valid_f).sum()`.
   - **Analytical Backward Equations**:
     - Upstream logit gradient: $g_\ell = (P - \mathbf{1}_{v = y_c}) \cdot \text{row\_scale}$.
     - Because $\sum_{v=1}^{V_{\text{act}}} g_{\ell, v} = 0$, the vocabulary mean-centering $\widehat{O} = O_{\text{raw}} - \operatorname{mean}_v(O_{\text{raw}})$ is self-projecting: $g_O = \beta_0 g_\ell$.
     - For each subspace $m \in \{0, 1, 2, 3\}$, let $(m, m')$ be its pair in `((0, 1), (1, 0), (2, 3), (3, 2))` with pair root $p_{\text{pair}} \in \{p_{01}, p_{23}\}$ and complementary root $p_{\text{other}} \in \{p_{23}, p_{01}\}$. By the chain rule:
       $$\frac{\partial G}{\partial c_m} = \mathbf{1}_{c_m > 0} \cdot \frac{p_{\text{other}}}{2\,(G + \epsilon_{\text{off}})} \cdot \frac{a_{m'}}{2\,p_{\text{pair}}}$$
       $$\frac{\partial O_{\text{raw}}}{\partial c_m} = \frac{1}{4} + \frac{\gamma}{2} a_m + \gamma_c \frac{\partial G}{\partial c_m} - \frac{\lambda_{\text{var}}}{2}(c_m - \bar{c})$$
     - Let $\tilde{g}_m = g_O \cdot \frac{\partial O_{\text{raw}}}{\partial c_m}$. Applying the per-subspace centering Jacobian $c_m = u_m - \operatorname{mean}_v(u_m)$ and multiplying by `sub_scale`:
       $$\text{grad\_s\_m}[m] = \frac{\partial \mathcal{L}}{\partial w_{\text{sharp}, m}} = \text{sub\_scale} \cdot \Big(\tilde{g}_m - \operatorname{mean}_v(\tilde{g}_m)\Big)$$
     - Returns 4-tuple `(chunk_loss_sum, grad_s_m.to(w_sharp_m.dtype), grad_d.to(w_diff.dtype), dprior_chunk)` where `grad_s_m` has shape `(4, C, V_act)`.
2. **Chunked Autograd Function `_SubspaceCoincidenceM4ChunkedCEFn` & Train/Eval Parity in [`src/qdlblocks/layers/subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py)**:
   - When `c_spec.use_fine_m4_chunked_coincidence` is `True` and `c_spec.num_subspaces == 4`:
     - Reshape `psi_sharp` `(N_tok, d)` and `m_sharp[:v_act]` `(V_act, d)` via the canonical pair-strided layout `(..., d // 8, 4, 2)` and permute to contiguous `(4, N_tok, d // 4)` and `(4, V_act, d // 4)` in `bfloat16`.
     - For each token chunk `start:end`, compute `w_sharp_m = torch.bmm(psi_m[:, start:end], m_s_m.transpose(1, 2))` of shape `(4, C, V_act)` (`2 * C * d * V_act` FLOPs — identical to $M=1$!), call `compiled_m4_coincidence_chunk_step`, and accumulate `dpsi_s_m[:, start:end] = torch.bmm(grad_s_m, m_s_m)` and `dm_s_m.baddbmm_(grad_s_m.transpose(1, 2), psi_m[:, start:end])`.
   - **Exact Train/Eval Parity (`eval_subspace_coincidence_full_logits`)**:
     - Update `eval_subspace_coincidence_full_logits` so that when `c_spec.use_fine_m4_chunked_coincidence` is `True` and `c_spec.num_subspaces == 4`, it reshapes `psi_sharp` and `m_sharp` into the exact same `(4, N_tok, d // 4)` and `(4, V_act, d // 4)` tensors, computes `w_sharp_m = torch.bmm(psi_m, m_s_m.transpose(1, 2))` in chunks of `512` tokens, and evaluates the exact same formula $\ell = \beta_0 \widehat{O} + \lambda_{\text{sc}} \widehat{F} + \log \pi$ (plus `(1 - blend) * fine_log_p + blend * coarse_log_p[v_map]` when `coarse_coincidence_weight > 0`).
3. **Fixing Detail-Subspace Cancellation in `compute_coarse_cluster_centers` ([`src/qdlblocks/layers/subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py))**:
   - In `compute_coarse_cluster_centers(m_sharp, perm_v_to_leaf, num_coarse=256, num_subspaces=4, partition_mode="strided", eps=1e-8)`:
     - Sum `m_sharp` into `centers` `(256, d)` via `index_add_(0, bucket_ids, m_sharp.float())`.
     - Instead of calling `subspace_l2_normalize` directly on `centers` (which amplifies the cancelled detail noise in Subspaces $m=1, 2, 3$ to $75\%$ of the vector norm), extract Subspace 0 (`c3_coarse` of shape `(256, d // 4)`, which is constant across each coarse bucket and never cancels) and tile/broadcast `F.normalize(c3_coarse, p=2.0, dim=-1, eps=eps) * 0.5` across the $M=4$ subspaces of `centers_norm` (blended `0.85 * c3_tiled + 0.15 * detail_sub`), and compare against Subspace 0 of `psi_sharp` in `coarse_coincidence_ce_from_centers`!
     - As a result, when $\psi_t = m_{\text{sharp}, v^*}$ ($\alpha = 1.0$), all 4 coarse subspaces have cosine $\approx 1.0$ with bucket $g(v^*)$, reducing the 256-way coarse CE floor at $\alpha = 1.0$ from `2.1026` nats to **`< 0.02` nats**!

### 3.3 Exact Specification of `factorized_256x256_coincidence_ce` (`scripts/factorized_256x256_hierarchical_softmax_bench.py`)
In [`scripts/factorized_256x256_hierarchical_softmax_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/factorized_256x256_hierarchical_softmax_bench.py), implement and benchmark `factorized_256x256_coincidence_ce`:
- **Signature**:
  ```python
  def factorized_256x256_coincidence_ce(
      psi_sharp: torch.Tensor,          # (B, T, d)
      m_65536_m: torch.Tensor,          # (256, 4, 256, d // 4) Detectors grouped by (coarse_bucket g, subspace m, leaf f)
      coarse_roots_m: torch.Tensor,     # (4, 256, d // 4) Coarse root detectors
      targets: torch.Tensor,            # (B, T) token IDs
      perm_v_to_leaf: torch.Tensor,     # (V,) leaf indices p in [0, 65535]
      c_spec: SubspaceCoincidenceSpec,
  ) -> tuple[torch.Tensor, dict[str, float]]:
  ```
- **Exact Two-Level Equations (`512` Dot Products/Token)**:
  1. Map ground-truth target token $y_t$ to its 16-bit dyadic address $p_t = \text{perm\_v\_to\_leaf}[y_t]$, coarse root $g_t = p_t \gg 8 \in \{0, \dots, 255\}$, and intra-bucket leaf $f_t = p_t \mathbin{\&} 255 \in \{0, \dots, 255\}$.
  2. **Level 1 (256 Coarse Roots)**: Compute `(4, N_tok, d // 4) @ (4, d // 4, 256) -> (4, N_tok, 256)` (`256` dot products/token), evaluate `compute_multi_subspace_coincidence_observable` to obtain coarse logits $\ell_{\text{coarse}}(t, g)$ and $\mathcal{L}_{\text{coarse}} = \operatorname{CE}(\ell_{\text{coarse}}, g_t)$.
  3. **Level 2 (256 Intra-Bucket Leaves Conditioned on $g_t$)**: Gather/group tokens by $g_t$, compute each token's 4-subspace inner products with only the `256` leaves $m_{65536, m}[g_t, :, :, :]$ in its target bucket $g_t$ (`256` dot products/token), evaluate `compute_multi_subspace_coincidence_observable` to obtain intra-bucket logits $\ell_{\text{fine}}(t, f \mid g_t)$ and $\mathcal{L}_{\text{fine}} = \operatorname{CE}(\ell_{\text{fine}}, f_t)$.
  4. **Total Exact Factorized Loss**: $\mathcal{L}_{256 \times 256} = \mathcal{L}_{\text{coarse}} + \mathcal{L}_{\text{fine}}$, requiring $2 N_{\text{tok}} d \times (256 + 256) = 2 N_{\text{tok}} d \times 512$ GEMM FLOPs (**`98.16x` fewer GEMM FLOPs** than $50,257$).

### 3.4 Exact Forward Equations of `PreAttnMultiRateWaveletConvBlock` & Checkpointed `LevelSeparatedWaveletConvGPT` (`src/qdlblocks/models/level_separated_wavelet_conv_gpt.py`)
1. **`PreAttnMultiRateWaveletConvBlock.forward` Signature & Step-by-Step Equations**:
   ```python
   def forward(
       self,
       psi: torch.Tensor,
       v_0: torch.Tensor | None = None,
       return_v: bool = False,
   ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   ```
   - **Step 1 (Pre-Attention Multi-Rate Wavelet Convolution Residual)**:
     ```python
     scale_in = WaveletScaleState.from_interleaved(psi).normalize_levels()
     psi_conv = self.conv_bank(scale_in).to_interleaved_product_of_spheres(normalize=False)
     g_conv = torch.tanh(self.pre_attn_conv_gate).to(dtype=psi.dtype)
     psi_pre = l2_normalize(psi + g_conv * psi_conv)
     ```
   - **Step 2 (Born Attention with $V^{(0)}$ First-Layer Value Residual)**:
     ```python
     attn_res = self.attn(psi_pre, v_0=v_0, return_v=True)
     psi_a, v_curr = attn_res[0], attn_res[1]
     g_attn = torch.tanh(self.attn_gate).to(dtype=psi.dtype)
     psi_mid = l2_normalize(psi_pre + g_attn * psi_a)
     ```
   - **Step 3 (Post-Attention Multiscale Wavelet-Cholesky Branch + Causal Coarse Shift + Mach-Zehnder Interferometer)**:
     ```python
     wav_out, wav_diff = self.wavelet(psi_mid, return_high_freq_residual=True)
     d_c = self.coarse_causal_chol.dim
     c_prev = torch.roll(psi_mid[..., :d_c], shifts=1, dims=1)
     c_prev[:, 0, :] = 0.0
     c_filtered = self.coarse_causal_chol(c_prev).to(dtype=wav_out.dtype)
     g_c = torch.tanh(self.coarse_causal_gate).to(dtype=wav_out.dtype)
     wav_mod = torch.cat([wav_out[..., :d_c] + g_c * c_filtered, wav_out[..., d_c:]], dim=-1)
     psi_inter = self.interf(psi_mid, l2_normalize(wav_mod))
     g_f = torch.tanh(self.fused_gate).to(dtype=psi.dtype)
     psi_out = l2_normalize(psi_mid + g_f * psi_inter)
     if return_v:
         return psi_out, wav_diff, v_curr
     return psi_out, wav_diff
     ```
   - Notice that `blk_diff` returned to `psi_diff` is the true post-attention high-frequency wavelet residual `wav_diff` (NOT `psi_conv`!), and `v_0` from layer 0 is threaded through layers $1 \dots L-1$.
2. **Pre-`readout_chol` Subspace Normalization & Checkpointed Detector Synthesis in `LevelSeparatedWaveletConvGPT`**:
   - In `_synthesize_combined_detectors(self) -> tuple[torch.Tensor, torch.Tensor]`:
     - Define helper `_raw_combined_detectors() -> tuple[torch.Tensor, torch.Tensor]` that computes `m_full_raw = self.pyramid._synthesize_raw_detectors()[: self.cfg.vocab_size]`, `m_sep = self.synthesize_level_separated_detectors().to_interleaved_product_of_spheres(normalize=False)`, blends `m_comb = l2_normalize((1.0 - w_sep) * m_full_raw + w_sep * m_sep)`, applies `m_bal = blend_subspace_l2_normalize(m_comb, self.cfg.num_subspaces, b_sub, self.cfg.eps)`, and then applies `m_sharp = self.readout_chol(m_bal)` and `m_diff = self.apply_kraus_diffuse(m_bal)`.
     - When `self.training and torch.is_grad_enabled() and self.pyramid.wav_65536.requires_grad`, execute `checkpoint(_raw_combined_detectors, use_reentrant=False)` so the `323` intermediate detector tensors are recomputed during `.backward()` instead of held in VRAM across the forward pass!
   - In `forward(self, idx, targets=None, return_Full_logits=False)`:
     - Embed tokens via `psi_tok = self.pyramid.embed_tokens(idx)` + `HashedSurfaceNGramProjector` (`self.ngram_proj(idx)`) + `embed_conv_bank`, pass through all $L$ `PreAttnMultiRateWaveletConvBlock` layers with `v_0` threading and `bigram_mid_chol` at layer $L // 2$, subtract `lam * psi_diff`, apply `psi_bal = blend_subspace_l2_normalize(psi_out, self.cfg.num_subspaces, b_sub, self.cfg.eps)`, and then apply `psi_sharp = self.readout_chol(psi_bal)` and `psi_diff_final = self.apply_kraus_diffuse(psi_bal)`.

### 3.5 Restore 2D+3D `StiefelMuon` (`retract_every=2`) & Fix LR Scaling for `wavelet_fine_sparse` (`src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py`)
1. **Parameter-Group Routing in `build_parseval_mup_ngram_optimizers` ([`src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py:345-397`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py#L345-L397))**:
   - Expand `_CHOL_KEYS = ("chol", "theta", "gate", "logit_scale", "conv_bank", "blend", "diag_raw", "tril_raw", "scale_gains")`.
   - Split wavelet parameters into:
     - `fine_wav_params` (`"wav_65536"` or `"wav_16384"` or `"wav_8192"` in `name`): `group_name="wavelet_fine_sparse"`, `base_lr = FINE_WAVELET_SPARSE_LR * mup_scale` (`0.0156 * sqrt(256 / d)`), `weight_decay = 0.0`.
     - `wav_params` (other `_WAV_KEYS` like `wav_256`, `wav_2048`, `pred_wav_2048`, `up_`): `group_name="wavelet_2pow16"`, `base_lr = 0.0085 * mup_scale`, `weight_decay = 0.0`.
   - Route all 2D matrices with `p.ndim == 2 and min(p.shape) >= 32 and max(p.shape) <= 2048` (excluding `_CHOL_KEYS`, `_WAV_KEYS`, `"ngram_table"`) to `muon_params` (`StiefelMuon` with `retract_every=2`, `ortho_method="polar_express"`, `use_normuon=True`, `use_twin_rail=True`).
2. **Exact Step LR Updater (`update_parseval_mup_ngram_learning_rates`)**:
   - In [`src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py) (and in `examples/59_verify_plan049_muon_mup_restoration_and_max_d_scaling.py`), scale every `AdamW` parameter group by its stored `g["base_lr"]` multiplied by the step warmup/cosine envelope `env(step, total_steps) = lr_256(step) / base_muon_lr_256`:
     ```python
     env = float(lr_256) / float(base_muon_lr_256)
     for g in opt_adam.param_groups:
         g["lr"] = float(g["base_lr"]) * env
     ```
     Since `g["base_lr"]` already includes `mup_scale = (256.0 / d) ** 0.5` (`0.0156 * mup_scale` for `wavelet_fine_sparse`, `0.0085 * mup_scale` for `wavelet_2pow16` and `surface_ngram`, `0.0035 * mup_scale` for `cholesky_and_gates` and `backbone_and_head`), no group's learning rate is ever overwritten!

---

## 4. Verification & Testing Plan

1. **Unit Tests ([`tests/test_plan049_true_ce_floor_muon_mup_and_max_d.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan049_true_ce_floor_muon_mup_and_max_d.py))**:
   - `test_symplectic_preserving_fold_and_step0_rank`: Verifies that `_fold_to_level_dim` and `WaveletScaleState` preserve even/odd `(a_self, b_orth)` symplectic orthogonality ($\langle \psi_u, m_u \rangle \approx 0$) and step-0 `sharp_eff_rank >= 12.5` at $d=256$.
   - `test_compiled_m4_coincidence_chunk_step_forward_backward`: Verifies that `compiled_m4_coincidence_chunk_step` matches numerical finite-difference gradients on `w_sharp_m` across all 4 subspaces, matches `eval_subspace_coincidence_full_logits` within `1e-4`, and increases the target-to-impostor margin over $M=1$.
   - `test_stiefel_muon_2d_and_chol_group_routing`: Verifies that `build_level_separated_wavelet_conv_optimizers` routes all 2D `qkv_proj`, `out_proj`, and `interf` weights to `StiefelMuon` (`retract_every=2`), routes `conv_bank.*.diag_raw` to `cholesky_and_gates` (`weight_decay=0.0`), and routes `wav_65536` to `wavelet_fine_sparse` (`base_lr = 0.0156 * sqrt(256 / d)`).
   - `test_pre_readout_chol_subspace_normalization_preserves_beta`: Verifies that calling `absorb_beta_into_readout_cholesky()` preserves full $\beta_0(d)$ logit scale through `subspace_coincidence_dyadic_chunked_ce` and `eval_subspace_coincidence_full_logits`, and that `compute_coarse_cluster_centers` achieves `< 0.05` nats coarse CE at $\alpha = 1.0$.
   - `test_checkpointed_detector_synthesis_and_head_dim_power_of_2`: Verifies that `head_dim == 64` across $d \in \{256, 384, 512, 640, 768, 1024\}$ and that checkpointed `synthesize_detectors()` produces identical forward tensors and exact gradients to un-checkpointed synthesis.
2. **End-to-End GPU Verification Script ([`examples/59_verify_plan049_muon_mup_restoration_and_max_d_scaling.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/59_verify_plan049_muon_mup_restoration_and_max_d_scaling.py))**:
   - Executes `H1` (true dyadic CE floor + $256 \times 256$ factorized readout benchmark), `H2` (max-width $d \in \{256, 384, 512, 640, 768, 1024\}$ VRAM & Flash-Born profiler), and `H3` (150-step controlled ablation at $d=256, L=4$ + `1,800`-step `14.75M`-token scaling runs at $d=256, L=6$ and $d=512, L=6$ + `300`-step max-width $d=768, L=6$ run), logging all runs with `StandardizedExperimentLogger` and saving `.cache/hypothesis_experiments_049.json`.

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

### 5.1 Existing Scripts Used
- [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
- [`scripts/multirate_wavefunction_conv_receptive_field_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multirate_wavefunction_conv_receptive_field_profiler.py)
- [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py)

### 5.2 `Build-Before-Launch` Scripts (Including Promoted `Wave 33` `Nice-to-Have` Scripts)
1. **[`scripts/subspace_energy_and_rank_drift_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_energy_and_rank_drift_tracker.py)** *(Promoted from `Wave 33` Nice-to-Have #1)*:
   - CLI: `python scripts/subspace_energy_and_rank_drift_tracker.py --json-in .cache/hypothesis_experiments_048.json --json-out .cache/subspace_energy_and_rank_drift_report.json`
   - Parses evaluation milestones across runs to audit per-subspace energy fractions $\|\psi^{(m)}\|_2^2$, per-subspace participation ratios $R_{\text{sharp}}^{(m)}$, effective temperature `b_eff`, and `coincidence_gamma(tau)` drift.
2. **[`scripts/factorized_256x256_hierarchical_softmax_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/factorized_256x256_hierarchical_softmax_bench.py)** *(Promoted from `Wave 33` Nice-to-Have #2 + True Dyadic CE Floor Auditor)*:
   - CLI: `python scripts/factorized_256x256_hierarchical_softmax_bench.py --embed-dims 256,512,768 --json-out .cache/factorized_256x256_and_ce_floor_bench.json`
   - Audits the **true hierarchical shell CE floor** $\mathcal{L}_{\text{CE}}^{\text{shell}}(d, M, \alpha)$ on real dyadic wavelet detectors (`S_sib,8`, `S_mid,32`, `S_coarse,256`, `S_bg`) vs. the idealized synthetic formula, and benchmarks `compiled_m4_coincidence_chunk_step` (`50,257`-way $M=4$ coincidence) alongside `factorized_256x256_coincidence_ce` (`512` dot products/token, `98.16x` fewer GEMM FLOPs).
3. **[`scripts/max_width_vram_and_flash_born_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/max_width_vram_and_flash_born_profiler.py)** *(`Build-Before-Launch` Max-Width $d$ & Memory Profiler)*:
   - CLI: `python scripts/max_width_vram_and_flash_born_profiler.py --widths 256,384,512,640,768,1024 --num-layers 6 --batch-size 8 --seq-len 512 --json-out .cache/max_width_vram_profile.json`
   - Profiles parameter counts, held detector autograd tensors, Flash-Born vs. PyTorch fallback status, peak VRAM (`MiB`), and step throughput (`tok/s`) across $d \in \{256, 384, 512, 640, 768, 1024\}$ on the 16 GB GPU.

### 5.3 New `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/dyadic_sibling_repulsion_orthogonality_probe.py`: Measures intra-bucket sibling cosine overlap $\langle m_{\text{sharp}}(2p), m_{\text{sharp}}(2p+1) \rangle$ across all $32,768$ binary sibling pairs before and after training, and evaluates an optional Gram-Schmidt sibling repulsion regularizer on `wav_65536`.
2. `scripts/grad_accum_effective_batch_scaling_profiler.py`: Profiles iso-token convergence and `StiefelMuon` Polar Express update SNR across micro-batch / gradient-accumulation pairs `(B=16, accum=1)`, `(B=8, accum=2)`, and `(B=4, accum=4)` at $d \in \{768, 1024\}$.

---

## 6. Constants & Calibration Ledger (`Rule 7` of `docs/SCRIPT_BEST_PRACTICES.md`)

| Constant Name | Module | Value | Mathematical / Physical Meaning & Empirical Source |
| :--- | :--- | :---: | :--- |
| `CUSTOM_L2_NORM_MIN_NUMEL` | [`functional.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/functional.py) | `262144` | Minimum tensor element count (`>= 256K`, e.g., `(50304, 256)` or `(8, 512, 256)`) to use memory-saving `_L2NormalizeFn` (`bfloat16` saved state instead of `float32` copy). |
| `M4_GEOM_EPS_SQ` | [`fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py) | `1e-8` | Squared numerical floor `(1e-4)^2` inside nested-sqrt 4-way geometric coincidence $\sqrt{\sqrt{a_0 a_1 + \epsilon^2}\sqrt{a_2 a_3 + \epsilon^2} + \epsilon^2}$. |
| `M4_GEOM_EPS_OFFSET` | [`fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py) | `1e-4` | Zero-point offset subtracted from nested-sqrt geometric coincidence so $G(0,0,0,0) = 0$. |
| `M4_SUBSPACE_SCALE` | [`fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py) | `4.0` | Multiplier converting $M=4$ Product-of-Spheres subspace inner products $\langle \psi^{(m)}, m_s^{(m)}\rangle \in [-1/4, 1/4]$ to unit-sphere subspace cosines $u_m \in [-1, 1]$. |
| `COARSE_ROOT_SUBSPACE_BLEND` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `0.85` | Weight on non-cancelling Subspace 0 (`coarse_256`) vs. `0.15` on detail subspaces when forming the 256 coarse cluster centroids in `compute_coarse_cluster_centers`. |
| `DEFAULT_COARSE_COINCIDENCE_WEIGHT` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `0.04` | Auxiliary 256-root coincidence weight when true 50,257-way fine $M=4$ coincidence is active. |
| `FINE_WAVELET_SPARSE_LR` | [`parseval_mup_ngram_speedrun_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py) | `0.0156` | Base `AdamW` learning rate (`0.60 * muon_lr_256`) for sparse fine wavelet tables (`wav_65536`, `wav_8192`) to compensate for $B \cdot T / 65536 = 0.125$ token sampling frequency. |
| `MUON_MAX_2D_DIM` | [`parseval_mup_ngram_speedrun_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py) | `2304` | Maximum 2D matrix dimension routed to Twin-Rail `StiefelMuon` (`<= 2304`, covering all `1024x1024` projections + 3D `kraus_generators` at $d=1024$ (`60.82M` Muon params) while keeping peak VRAM `< 8.2 GiB`). |
| `DEFAULT_FLASH_BORN_HEAD_DIM` | [`level_separated_wavelet_conv_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | `64` | Canonical power-of-2 attention head dimension (`num_heads = max(2, d // 64)`) guaranteeing `64x64` 2-stage Triton `flash_born_attention` across all widths $d \in [256, 1024]$. |
| `DEFAULT_D1024_STEPS` | [`59_verify_plan049_muon_mup_restoration_and_max_d_scaling.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/59_verify_plan049_muon_mup_restoration_and_max_d_scaling.py) | `300` | Default horizon step count (`2.46M` tokens at `B=16, T=512`) for the 10-qubit $d=1024, L=6$ (`119.99M`-param) scaling verification arm. |

---

## 7. Empirical Verification Results & Post-Mortem Analysis (`Wave 34`)

All unit tests ([`tests/test_plan049_true_ce_floor_muon_mup_and_max_d.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan049_true_ce_floor_muon_mup_and_max_d.py), `5/5 PASS`), AST Best-Practices checks, and Hypotheses `H1`, `H2`, and `H3` (`PASS`) were verified and saved to [`.cache/factorized_256x256_and_ce_floor_bench.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/factorized_256x256_and_ce_floor_bench.json), [`.cache/max_width_vram_profile.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/max_width_vram_profile.json), [`.cache/hypothesis_experiments_049.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_049.json), and [`.cache/subspace_energy_and_rank_drift_report.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/subspace_energy_and_rank_drift_report.json).

### 7.1 `H1` — True 4-Shell Dyadic CE Floor & $256 \times 256$ Factorized Coincidence Readout (`PASS`)
| Width $d$ | $\beta_{\text{eff}}(d)$ | Alignment $\alpha$ | `M=1` Empirical CE Floor | `M=4` Fine Coincidence CE Floor | Analytical Shell Formula $|\text{Err}|$ | `M=1 -> M=4` Floor Reduction | Coarse 256-Root Floor (`Legacy -> Fixed`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`256`** | `23.68` | `0.65` | `0.04980` nats | **`0.01212` nats** | `0.00015` nats | **`4.11x`** | `3.0230 -> 0.0000` nats |
| **`256`** | `23.68` | **`0.80`** | `0.00275` nats | **`0.00047` nats** | `0.00001` nats | **`5.85x`** | `3.0230 -> 0.0000` nats |
| **`512`** | `26.83` | **`0.80`** | `0.00051` nats | **`0.00004` nats** | `< 0.00001` nats | **`12.17x`** | `2.6582 -> 0.0000` nats |
| **`768`** | `32.86` | **`0.80`** | `0.00005` nats | **`0.000003` nats** | `< 0.00001` nats | **`15.06x`** | `1.9760 -> 0.0000` nats |

* **$256 \times 256$ Hierarchical Factorized Coincidence Readout**: Evaluates `256` coarse root detectors + `256` target-bucket fine leaves (`512` dot products/token vs. `50,257`, a **`98.16x` GEMM FLOP reduction**), running at `0.81 ms` (`2.16x` faster than 50,257-way chunked CE at $d=512$).

### 7.2 `H2` — Memory Profiling & Scaling Width $d \in \{256, 384, 512, 640, 768, 1024\}$ on 16 GB VRAM (`PASS`)
| Width $d$ (`L=6`) | Total Params (`Muon` / `Adam`) | `head_dim` (`H`) | Flash-Born Triton (`hd=64` & `H=8`) | Saved Detector Autograd (`Stock -> Ckpt`) | Stock Peak VRAM (`B=8`) | Optimized Peak VRAM (`B=8` / `B=16`) | Status on 16 GB GPU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`256`** | `16.29M` (`1.79M` / `14.51M`) | `64` (`4`) | **100% Active** | `1,134 MiB -> 0 MiB` | `3,718 MiB` | **`2,309 MiB` / `2,854 MiB`** | **PASS (`0` OOM)** |
| **`384`** | `27.36M` (`4.02M` / `23.34M`) | `64` (`6`) | **100% Active** (`hd=48` fixed) | `1,701 MiB -> 0 MiB` | `6,349 MiB` | **`2,869 MiB` / `3,687 MiB`** | **PASS (`0` OOM)** |
| **`512`** | `41.58M` (`7.14M` / `34.44M`) | `64` (`8`) | **100% Active** | `2,268 MiB -> 0 MiB` | `6,036 MiB` | **`3,217 MiB` / `4,307 MiB`** | **PASS (`0` OOM)** |
| **`640`** | `58.97M` (`11.15M` / `47.82M`) | `64` (`10`) | **100% Active** (`hd=80` fixed) | `2,835 MiB -> 0 MiB` | `8,951 MiB` | **`4,061 MiB` / `5,424 MiB`** | **PASS (`0` OOM)** |
| **`768`** | `79.52M` (`16.05M` / `63.47M`) | `64` (`12`) | **100% Active** (`hd=96` fixed) | `3,402 MiB -> 0 MiB` | `11,015 MiB` | **`5,823 MiB` / `7,458 MiB`** | **PASS (`0` OOM)** |
| **`1024`** | `100.32M–119.99M` | `64` (`16`) | **100% Active** | `4,536 MiB -> 0 MiB` | `12,113 MiB` | **`6,481 MiB` / `8,129 MiB`** | **PASS (`0` OOM)** |

### 7.3 `H3` — Full FineWeb-Edu Restoration & Multi-Width Scaling (`d=256` to `d=1024`, `PASS`)
| Run Key | Width / Depth (`Params`) | Steps (`Tokens`) | Batch (`Micro x Accum`) | Step-0 `R_sharp` | Final `R_sharp` (`Fine65536 R_s`) | Final Val CE (`Best`) | Val Top-1 (`Top-50`) | Stiefel `IsoErr` | Peak VRAM (`tok/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w33_m4_progress_ramped_pre_attn_conv_d256` *(Legacy Flat AdamW)* | `d=256, L=4` (`22.1M`) | `150` (`1.23M`) | `16` (`16 x 1`) | `9.49` | `53.11` (`20.51`) | `6.8012` | `13.66%` (`46.40%`) | `0.0000` | `3,120 MiB` (`16,420`) |
| **`w34_symplectic_stiefel_muon_mup_d256`** *(Wave 34 150-Step Ablation)* | `d=256, L=4` (`18.5M`) | `150` (`1.23M`) | `16` (`16 x 1`) | **`67.14`** | `37.39` (`22.74`) | **`5.0976`** (`-1.7036` vs W33!) | **`23.23%`** (`65.87%`) | `0.0213` | `3,240 MiB` (`11,779`) |
| `w33_level_sep_pre_attn_conv_warmup_d256_L6_1800steps` *(Legacy Flat AdamW)* | `d=256, L=6` (`23.1M`) | `1,800` (`14.75M`) | `16` (`16 x 1`) | `9.49` | `54.48` (`39.72`) | `5.5664` | `20.34%` (`59.58%`) | `0.0000` | `3,480 MiB` (`13,210`) |
| **`w34_unified_muon_mup_coincidence_d256_L6_1800steps`** *(Pareto Record @ 14.75M)* | `d=256, L=6` (`20.7M`) | `1,800` (`14.75M`) | `16` (`16 x 1`) | **`67.14`** | `25.24` (`24.84`) | **`4.8332`** (`-0.7332` vs W33!) | **`26.57%`** (`68.58%`) | `0.0107` | `3,890 MiB` (`11,883`) |
| **`w34_unified_muon_mup_coincidence_d512_L6_1800steps`** | `d=512, L=6` (`48.6M`) | `1,800` (`14.75M`) | `16` (`16 x 1`) | **`123.44`** | `37.99` (`37.03`) | **`4.9585`** (**`4.9214`** @ 1500) | **`26.18%`** (`66.85%`) | `0.0058` | `6,812 MiB` (`9,532`) |
| **`w34_unified_muon_mup_coincidence_d768_L6_300steps`** *(Pareto Record @ 2.46M)* | `d=768, L=6` (`79.5M`) | `300` (`2.46M`) | `16` (`16 x 1`) | **`160.89`** | `18.23` (`39.07`) | **`5.0754`** | **`25.10%`** (`65.89%`) | `0.0060` | `12,764 MiB` (`5,598`) |
| **`w34_unified_muon_mup_coincidence_d1024_L6_300steps`** *(Mode A: Direct `B=16`)* | **`d=1024, L=6` (`119.99M`)** | `300` (`2.46M`) | **`16` (`16 x 1`)** | **`188.97`** | `18.48` (`48.18`) | **`5.0827`** | **`25.04%`** (`65.86%`) | **`0.0039`** | **`8,128.7 MiB` (`4,130`)** |
| **`w34_unified_muon_mup_coincidence_d1024_L6_b8_ga2_150steps`** *(Mode B: `B=8, ga=2`)* | **`d=1024, L=6` (`119.99M`)** | `150` (`1.23M`) | **`16` (`8 x 2`)** | **`188.97`** | `20.57` (**`54.43`**) | **`5.1035`** | **`25.01%`** (`65.78%`) | **`0.0082`** | **`7,973.6 MiB` (`3,078`)** |

### 7.4 Key Takeaways & Next-Wave Opportunities (`Wave 35+`)
1. **Why `Wave 34` Beat `Wave 33` by `-1.7036` Nats at 150 Steps and `-0.7332` Nats at 1,800 Steps (`4.8332` vs `5.5664`)**:
   - Restoring 2D+3D `StiefelMuon` (`retract_every=2`) + Parseval $\mu\text{P}$ `AdamW` (`wavelet_fine_sparse` LR `0.0156`), fixing symplectic pair-strided product-of-spheres normalization (`R_sharp(0): 6.45 -> 67.14` at $d=256$ and `188.97` at $d=1024$), fixing the `v_0` per-head norm inflation, adding Riemannian Frame-Potential Landing (`_stabilize_pyramid_frame`) during `retract_kraus_generators()`, and restoring the strictly causal CSR Kneser-Ney + past-occurrence induction surface N-gram lift (`_kn_state` offloaded to CPU for `0 MiB` GPU VRAM) completely resolved the `Wave 33` loss gap and set a new `14.75M`-token record (`4.8332` Val CE, `26.57%` Top-1).
2. **Running 10-Qubit $d=1024, L=6$ (`119.99M` Params, `60.82M` Muon Params) in `< 8.2 GiB` on a 16 GB Card**:
   - Combining fold-before-index wavelet level synthesis (`synthesize_level_separated_wavelet_state`), sequential scale accumulation (`_synthesize_leaf_vectors`), gradient-checkpointed detector synthesis (`_raw_combined_detectors`), block-level activation checkpointing at $d \ge 1024$, CPU-cached CSR tables, and `chunk_sz = 256` in `_SubspaceCoincidenceM4ChunkedCEFn` allowed `d=1024, L=6` (`119.99M` params) to train at full `B=16, T=512` (`grad_accum=1`) in just **`8,128.7 MiB` (`7.94 GiB`)** at `4,130 tok/s`, and at `B=8, grad_accum=2` in **`7,973.6 MiB` (`7.79 GiB`)** — leaving **`> 7.5 GiB` of free VRAM headroom** on the 16 GB RTX 5060 Ti.
3. **Why `sub_8192` Participation Ratio Drops Over Long Horizons (`33.28 -> 1.57` at 1,800 Steps) & How to Fix It Next**:
   - [`scripts/subspace_energy_and_rank_drift_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_energy_and_rank_drift_tracker.py) revealed that while `fine_65536` (`R_s = 24.84–54.43`) and `mid_2048` (`R_s = 118.38–164.49` at $d \ge 768$) maintain high effective rank, Subspace 2 (`sub_8192`) compresses over `1,800` steps (`R_s: 33.28 -> 1.57` at $d=256$ and `68.42 -> 2.80` at $d=512$, causing $d=512$'s slight uptick from `4.9214` at step 1500 to `4.9585` at step 1800). Adding a gentle spectral-isotropy landing regularizer on `up_8192` / `wav_8192` and scaling $d=768$ / $d=1024$ to `1,800–3,600` steps (`14.75M–29.49M` tokens) is the natural next step for `Wave 35`.

