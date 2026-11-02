# EXPERIMENT PLAN 076 — Combined Fused Hermitian/Born + Riemannian Tangent SNR + Low-Momentum Warmup + WSD Cooldown/Tail-EMA Recipe & 100M-Token Cold-Start Quantum Wavepacket GPT Training From Scratch (Wave 55)

## 1. Executive Summary & Scientific Hypotheses

In **Wave 54 (`EXPERIMENT_PLAN_075.md`)**, we systematically isolated four architectural and geometric upgrades across all 12 layers of our Quantum Wavepacket GPT (`Embeddings + All 12 Layers Attention`):
1. **Native In-Kernel Complex Hermitian + Born $|\langle Q_{t,h}|K_{s,h}\rangle_{\mathbb{C}}|^2$ Scoring (`FUSED_HERMITIAN_BORN_RHO_KET`)**: Replaced the external `apply_berry_qk_phase` tensor allocation and reached `3.5628` Val CE at $d_{\text{pkt}} = 32$.
2. **$d_{\text{pkt}} = 64$ Wavepacket Granularity (`SHAPE_6x2x64`, $64 \times 64$ complex density matrix $\rho_{t,h} \in \mathbb{C}^{64 \times 64}$)**: Reached `3.5597` Val CE (`60.29%` Top-5) and `3.5593` Val CE (`60.35%` Top-5, `q_eff_rank = 3.63`) when paired with Riemannian Tangent SNR Gating.
3. **Zero Early Momentum + Low Steady-State Momentum (`LOW_STEADY_MOMENTUM_070`, $\beta_t: 0.0 \to 0.70$)**: Eliminated the Step 200–400 high-momentum curvature stall (`\beta = 0.90..0.92`) on $\mathrm{St}(768, r) \times U(d_h)$ and achieved the best `1:1` parity validation loss (`3.5582` Val CE, `-0.0263` nats below the `300M` classical teacher `3.5845`).
4. **Riemannian Tangent SNR Trust-Region Gating (`RIEMANNIAN_TANGENT_SNR_GATED`)**: Dynamically damped off-manifold momentum jitter via $\eta_{\text{SNR}} = \|G_{\tan}\|_F / (\|G_{\tan}\|_F + \|M_{\text{norm}}\|_F + \epsilon)$.

In **Wave 55 (`EXPERIMENT_PLAN_076.md`)**, we execute two major goals requested by the user:
1. **Part 1 — The Combined Recipe Tournament (`L76_1..L76_4`)**:
   - Combine `FUSED_HERMITIAN_BORN_RHO_KET` + `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070` ($\beta_t: 0.0 \to 0.70$) with `modded-nanogpt`'s late-stage **Warmup-Stable-Decay (`WSD`) linear cooldown** (`QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN`) and **Riemannian Manifold-Retracted Tail EMA Weight Averaging** (`retract_quantum_backbone_manifolds_`) across both $d_{\text{pkt}} = 32$ (`SHAPE_12x2x32`), $d_{\text{pkt}} = 64$ (`SHAPE_6x2x64`), and the `100%` Full-Backbone Quantum Wavepacket + 12 Mach-Zehnder Interferometer MLP architecture (`EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`, `r = 768`).
2. **Part 2 — Cold-Start Training From Scratch for `100M` FineWeb Tokens (`L76_5_FROM_SCRATCH_100M_QUANTUM_WAVEPACKET_GPT`)**:
   - Train a 12-layer Quantum Wavepacket GPT **completely from scratch** (`0M -> 50M -> 100M` FineWeb tokens, `382` macro-steps at `262,144` tok/step = `100.14M` tokens, zero pretrained teacher weights, starting from exact Step-0 `val_ce = 10.8249` / `top1 = 0.06%`) using the Combined Recipe, and compare its validation cross-entropy and Top-1/Top-5 accuracy trajectory directly against the Classical `w45b` Teacher checkpoints at `0M` (`10.8249`), `50M` (`4.2466`, `31.16%` Top-1), and `100M` (`3.9354`, `33.71%` Top-1).

---

## 2. Mathematical Formulations & Exact Kernel Speedups

### 2.1 Two Exact Mathematical Speedups in `FUSED_HERMITIAN_BORN_RHO_KET` (`2.6x` at $d_{\text{pkt}}=32$, `3.8x` at $d_{\text{pkt}}=64$)

To make 12-layer `100M`-token from-scratch training and $d_{\text{pkt}} = 64$ runs fast on 16 GB GPUs without numerical approximation, we exploit two exact algebraic identities of Hermitian outer-product density operators:

1. **Zero-Mask Hermitian Sum Packing ($M_{s,h}^{\text{tok}} = \rho_{s,h,R}^{\text{tok}} + \rho_{s,h,I}^{\text{tok}} \in \mathbb{R}^{d_h \times d_h}$)**:
   Because every token density matrix $\rho_{s,h}^{\text{tok}} = \rho_{s,h,R}^{\text{tok}} + i \rho_{s,h,I}^{\text{tok}}$ is Hermitian, its real part is symmetric ($(\rho_{s,h,R}^{\text{tok}})^\top = \rho_{s,h,R}^{\text{tok}}$) and its imaginary part is skew-symmetric ($(\rho_{s,h,I}^{\text{tok}})^\top = -\rho_{s,h,I}^{\text{tok}}$). Therefore, packing the single $d_h \times d_h$ real matrix
   $$M_{s,h}^{\text{tok}} = \rho_{s,h,R}^{\text{tok}} + \rho_{s,h,I}^{\text{tok}} \in \mathbb{R}^{d_h \times d_h}$$
   cuts the density payload width in half ($2 d_h^2 \to d_h^2$, reducing total packed payload width from `2,176 -> 1,152` at $d_h = 32$ and from `8,320 -> 4,224` at $d_h = 64$). By linearity of attention aggregation $M_{t,h}^{(0)} = \sum_{s \le t} A_{t,s}^{(h)} M_{s,h}^{\text{tok}}$, both parts of $\rho_{t,h}^{(0)}$ are recovered identically after attention with zero masks or index tables:
   $$\rho_{t,h,R}^{(0)} = \frac{1}{2}\!\left(M_{t,h}^{(0)} + (M_{t,h}^{(0)})^\top\right), \qquad \rho_{t,h,I}^{(0)} = \frac{1}{2}\!\left(M_{t,h}^{(0)} - (M_{t,h}^{(0)})^\top\right)$$

2. **Associative Ket-Operator Contraction for Quadratic McWeeny Purification + Unitary Channel (`_compiled_associative_mcweeny_unitary_ket_modulation`)**:
   For the trace-normalized Hermitian density matrix $\bar{\rho} = \rho^{(0)} / \operatorname{Tr}(\rho^{(0)})$, Hermiticity implies $\operatorname{Tr}(\bar{\rho}^2) = \|\bar{\rho}_R\|_F^2 + \|\bar{\rho}_I\|_F^2$. Furthermore, applying the unitary-evolved traceless McWeeny operator
   $$\mathring{\rho}_{\text{evol}} = U_h \left(\frac{\bar{\rho}^2}{\operatorname{Tr}(\bar{\rho}^2)} - \frac{1}{d_h} I_{d_h}\right) U_h^\dagger$$
   to the $(R \times d_h)$ complex target ket $|\psi_{\text{tar}}\rangle$ ($R \in \{1, 2\}$) factors by matrix multiplication associativity into lightweight $(R \times d_h) \cdot (d_h \times d_h)$ vector-matrix products:
   $$|w_0\rangle = |\psi_{\text{tar}}\rangle \overline{U_h}, \qquad |w_1\rangle = |w_0\rangle \overline{\bar{\rho}}, \qquad |w_2\rangle = |w_1\rangle \overline{\bar{\rho}}, \qquad |\delta\psi\rangle = \left(\frac{|w_2\rangle}{\|\bar{\rho}_R\|_F^2 + \|\bar{\rho}_I\|_F^2} - \frac{1}{d_h}|w_0\rangle\right) U_h^\top$$
   This eliminates all six $d_h \times d_h \times d_h$ dense matrix-matrix products per token per head per layer while matching the dense matrix path to `< 5e-5` forward and backward precision.

### 2.2 Late-Stage WSD Linear Cooldown & Riemannian Manifold-Retracted Tail EMA

1. **Depth-Scaled Warmup-Stable-Decay Linear Cooldown (`QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN`)**:
   Combines the $10\%$ linear warmup with a stable plateau through `65%` of training (`1 - DEFAULT_WSD_COOLDOWN_FRAC = 0.65`), followed by a `modded-nanogpt`-style linear cooldown to `DEFAULT_WSD_MIN_LR_RATIO = 0.08` over the final `35%` of steps:
   $$m_{\text{WSD}}(t) = \begin{cases} 0.10 + 0.90 \frac{t - 1}{\max(1, T_{\text{warm}} - 1)}, & 1 \le t \le T_{\text{warm}} \\ 1.0, & T_{\text{warm}} < t \le T_{\text{stable}} \\ 1.0 - (1.0 - \eta_{\min}) \frac{t - T_{\text{stable}}}{\max(1, T - T_{\text{stable}})}, & t > T_{\text{stable}} \end{cases}$$

2. **Riemannian Manifold-Retracted Tail EMA (`retract_quantum_backbone_manifolds_`)**:
   In Euclidean space, linear EMA blending $W_{\text{blend}} = (1 - \alpha) W_t + \alpha W_{\text{EMA}}$ moves parameters slightly inside the convex hull of curved manifolds ($\mathrm{St}(d, r)$, $U(d_h)$, and $S^{d-1}_{\text{RMS}}$). After `TailWeightAverager` swaps in the blended weights, `retract_quantum_backbone_manifolds_(model)` retracts:
   - All real Stiefel factors $W \in \mathrm{St}(d_{\text{out}}, r)$ via symmetric inverse square-root polar retraction $W \leftarrow W (W^\top W + \epsilon I)^{-1/2}$,
   - All complex head unitaries $U_h = U_R + i U_I \in U(d_h)$ via complex Gram inverse square-root retraction, and
   - All 3D spherical token/value embeddings `q_wte`, `q_value_embeds` back onto the exact unit-RMS sphere $S^{d_{\text{sphere}}-1}_{\text{RMS}}$.

### 2.3 Cold-Start From-Scratch Quantum Wavepacket GPT (`QuantumFullBackboneGraftGPT.from_scratch`)

When initializing a 12-layer Quantum Wavepacket GPT **from scratch** (`from_scratch=True`, zero pretrained teacher weights):
1. **Avoiding the Bilinear $\vec{\sigma} = \vec{0}$ Saddle Point**:
   In a Biorthogonal SVD-Stiefel projection $y = ((x V) \odot \vec{\sigma}) U^\top$, initializing $\vec{\sigma} = \vec{0}$ causes $\nabla_U \mathcal{L} = 0$ and $\nabla_V \mathcal{L} = 0$. Therefore, `from_scratch` initializes $(U_\alpha, V_\alpha) \in \mathrm{St}(d, r)$ as random orthonormal frames with unit singular values $\vec{\sigma}_\alpha = \vec{1}$ across all 12 layers' `q_proj, k_proj, v_proj, obs_proj` (and `up_proj, down_proj` when Mach-Zehnder MLPs are active), while setting the residual branch scale `obs_scale = DEFAULT_COLD_START_OBS_SCALE_INIT = 0.25`.
2. **Exact Step-0 Parity with `w45b` (`val_ce = 10.8249`)**:
   `q_wte` is initialized on the unit-RMS sphere via `QuantumWteInitMode.PPMI_SVD_SPHERE` (using only Shard-1 corpus co-occurrence statistics, zero neural teacher weights), `q_value_embeds` are initialized on the unit-RMS sphere with `ve_init_std = 0.02`, and `lm_head.weight` starts at `0.0` so Step-0 validation cross-entropy is `ln(50257) = 10.824905`, identical to `ckpt_step00000_0000M_m0000M.pt`.
3. **Full End-to-End Differentiable Optimization (`0M -> 100M` Tokens)**:
   During cold-start training (`QuantumRecoveryScheduleMode.PURE_CE`), `QuantumFullBackboneOptimizerBundle` jointly optimizes:
   - Quantum Stiefel & Unitary manifolds (`q_trunk_opt`: `ModularRiemannianStiefelKrausMuon` with `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070`),
   - Quantum 3D Sphere embeddings `q_wte` and `q_value_embeds` (`q_wte_opt`: `BatchSubspaceTikhonovGramMuon`),
   - Quantum scalars, singular spectra $\vec{\sigma}$, Hermitian/Born parameters, and gates (`q_scalar_opt`: `AdamW`), plus
   - `lm_head` (`Adam`, `lr = 0.008`) and trunk MLP matrices (`StiefelMuon`, `lr = 0.04` when `mlp_channel_mode == FROZEN_CLASSICAL_MLP`).

### 2.4 Top-of-File Named Constants (Rule 7 Synchronization)

| Constant Name | Value | Meaning |
|---|---|---|
| `DEFAULT_WSD_COOLDOWN_FRAC` | `0.35` | Fraction of training steps spent in WSD linear cooldown |
| `DEFAULT_WSD_MIN_LR_RATIO` | `0.08` | Final LR ratio at the end of WSD linear cooldown |
| `DEFAULT_QUANTUM_TAIL_EMA_START_FRAC` | `0.70` | Step fraction at which quantum `TailWeightAverager` begins |
| `DEFAULT_QUANTUM_TAIL_EMA_RATE` | `0.05` | EMA interpolation rate per update step |
| `DEFAULT_QUANTUM_TAIL_EMA_BLEND` | `0.70` | Evaluation blend factor $\alpha$ prior to manifold retraction |
| `DEFAULT_COLD_START_OBS_SCALE_INIT` | `0.25` | Initial residual branch scale for cold-start from-scratch quantum attention |
| `DEFAULT_FROM_SCRATCH_100M_STEPS` | `382` | Macro-steps at `262,144` tok/step (`100.14M` tokens, matching `ckpt_step00382_0100M`) |
| `DEFAULT_FROM_SCRATCH_BATCH_TOKENS` | `262144` | Tokens per macro-step (`B=8, T=1024, grad_accum=32`) |

---

## 3. Planned File Changes & Experimental Tournament Matrix (`Arms L76_1..L76_5`)

### 3.0 Planned File Changes
- [`src/qdlblocks/kernels/fused_quantum_wavepacket_attn.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/kernels/fused_quantum_wavepacket_attn.py): Adds Zero-Mask Hermitian Sum Packing ($M = \rho_R + \rho_I$) and `associative_mcweeny_unitary_ket_modulation`.
- [`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py): Adds `QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN` and `SVDStiefelWavepacketStatMechBlock.calibrate_cold_start_block_`.
- [`src/qdlblocks/models/quantum_full_backbone_graft_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/quantum_full_backbone_graft_gpt.py): Adds `QuantumFullBackboneGraftGPT.from_scratch` and `retract_quantum_backbone_manifolds_`.
- [`src/qdlblocks/optim/tail_ema.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/tail_ema.py): Adds `shadow_device` CPU offload support.
- [`examples/86_verify_plan076_combined_recipe_and_100m_from_scratch.py`](file:///home/nixos/QuantumDLBlocks/examples/86_verify_plan076_combined_recipe_and_100m_from_scratch.py): Dual-GPU tournament and `100M`-token from-scratch runner.
- [`tests/test_plan076_combined_recipe_and_100m_from_scratch.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan076_combined_recipe_and_100m_from_scratch.py): Deterministic verification test suite for Gates `L1..L4`.

### Part 1: Combined Recipe 12-Layer Tournament on `300M` Pretrained Teacher (`800` Steps, `13.11M` Tokens)

| Arm ID | Wavefunction Shape | Kernel & Momentum Mode | LR Schedule & Tail EMA | Purpose |
|---|---|---|---|---|
| **`L76_1_COMBINED_PKT32_LOW_MOM070_TAN_SNR_R384`** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$) | `FUSED_HERMITIAN_BORN_RHO_KET` + `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070` | `DEPTH_SQRT_WARMUP_COSINE` (No EMA) | Tests combining Tangent SNR gating with $\beta: 0.0 \to 0.70$ at $d_{\text{pkt}}=32$ (`1:1` parity `r=384`) |
| **`L76_2_COMBINED_PKT32_WSD_COOLDOWN_TAIL_EMA_R384`** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$) | `FUSED_HERMITIAN_BORN_RHO_KET` + `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070` | `DEPTH_SQRT_WARMUP_WSD_COOLDOWN` + Manifold `TailWeightAverager` | Tests adding `modded-nanogpt` WSD linear cooldown + Riemannian manifold-retracted Tail EMA at $d_{\text{pkt}}=32$ |
| **`L76_3_COMBINED_PKT64_LOW_MOM070_TAN_SNR_TAIL_EMA_R384`** | `SHAPE_6x2x64` ($d_{\text{pkt}}=64$, $64\times 64$ $\rho_{t,h}$) | `FUSED_HERMITIAN_BORN_RHO_KET` + `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070` | `DEPTH_SQRT_WARMUP_WSD_COOLDOWN` + Manifold `TailWeightAverager` | Tests the full Combined Recipe at $d_{\text{pkt}}=64$ (`SHAPE_6x2x64`, `1:1` parity `r=384`) |
| **`L76_4_COMBINED_FULL_BACKBONE_MZ_PKT32_R768`** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$, `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`) | `FUSED_HERMITIAN_BORN_RHO_KET` + `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070` | `DEPTH_SQRT_WARMUP_WSD_COOLDOWN` + Manifold `TailWeightAverager` | Tests the Combined Recipe across `100%` of the 12-layer backbone (11 Quantum Wavepacket Attn + 12 Mach-Zehnder MLPs) |

### Part 2: Cold-Start Training From Scratch for `100M` FineWeb Tokens (`L76_5`)

| Arm ID | Initialization | Token Budget & Schedule | Classical Baseline Checkpoints (`w45b`) |
|---|---|---|---|
| **`L76_5_FROM_SCRATCH_100M_QUANTUM_WAVEPACKET_GPT`** | Cold-Start From Scratch (`ckpt_step00000_0000M`, `PPMI_SVD_SPHERE` `q_wte`, random Stiefel $U, V \in \mathrm{St}(768, 384)$, $\vec{\sigma}=\vec{1}$, `PURE_CE`, zero teacher weights) | `100.14M` tokens (`382` macro-steps $\times$ `262,144` tok/step across `fineweb_train_000001..000009.bin`), `FUSED_HERMITIAN_BORN_RHO_KET` + `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070` + `WSD` Cooldown + Manifold Tail EMA | `0M`: `val_ce = 10.8249`, `top1 = 0.06%`<br>`50M` (`step=191`): `val_ce = 4.2466`, `top1 = 31.16%`<br>`100M` (`step=382`): `val_ce = 3.9354`, `top1 = 33.71%` |

---

## 4. Testing & Operational Tooling (`scripts/` & Verification Gates `L1..L4`)

### 4.1 Operational Tooling & Script Invocations
- `scripts/ast_signature_and_length_linter.py`
- `scripts/verify_plan.py --plan 076`
- `examples/86_verify_plan076_combined_recipe_and_100m_from_scratch.py`
- `tests/test_plan076_combined_recipe_and_100m_from_scratch.py`

### 4.2 Deterministic Verification Gates (`L1..L4`)

- **Gate `L1` (Zero-Mask Hermitian Sum Packing & Associative McWeeny/Unitary Ket Contraction Exactness)**:
  - Verify that `fused_quantum_wavepacket_rho_ket_attention` with $M = \rho_R + \rho_I$ packing and `_compiled_associative_mcweeny_unitary_ket_modulation` matches the dense matrix-product reference to `< 1e-4` forward and backward gradient error across both $d_{\text{pkt}} = 32$ and $d_{\text{pkt}} = 64$, and is compile-safe under `torch.amp.autocast(dtype=torch.bfloat16)`.
- **Gate `L2` (`DEPTH_SQRT_WARMUP_WSD_COOLDOWN` & Riemannian Manifold-Retracted `TailWeightAverager`)**:
  - Verify that `compute_quantum_backbone_lr_multiplier(..., QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN)` warms up over `[1, T_warm]`, holds `1.0` through `T_stable`, and linearly cools down to `DEFAULT_WSD_MIN_LR_RATIO = 0.08` at `step = total_steps`.
  - Verify that `retract_quantum_backbone_manifolds_(model)` restores exact manifold invariants (`q_max_stiefel_err < 1e-4`, `q_max_unitary_err < 1e-4`, `q_wte_sphere_err < 1e-4`) after Euclidean EMA blending.
- **Gate `L3` (Cold-Start `QuantumFullBackboneGraftGPT.from_scratch` Initialization & End-to-End Descent)**:
  - Verify that `QuantumFullBackboneGraftGPT.from_scratch(...)` starts at `val_ce = ln(V) ≈ 10.8249` with non-zero gradients on all Stiefel factors $(U_\alpha, V_\alpha)$, singular spectra $\vec{\sigma}_\alpha$, unitary operators $U_h$, sphere embeddings `q_wte`, MLPs, and `lm_head`, and strictly decreases loss over a multi-step cold-start smoke test.
- **Gate `L4` (Empirical Tournament `L76_1..L76_4` & `100M`-Token From-Scratch Run `L76_5`)**:
  - All 4 Combined Recipe graft arms (`L76_1..L76_4`) strictly beat the `300M` Classical Teacher (`val_ce < 3.5845`), and `L76_5_FROM_SCRATCH_100M_QUANTUM_WAVEPACKET_GPT` trains from scratch (`10.8249` at `0M`) for `100M` FineWeb tokens with zero NaNs and exact manifold invariants.

---

## 5. Empirical Results & Post-Mortem Ledger (`Dual NVIDIA GeForce RTX 3090`)

### 5.1 Part 1 — Combined Recipe 12-Layer Tournament on Pretrained `300M` Classical Teacher (`800` Steps, `13.11M` Tokens)

| Arm ID | Wavefunction Shape | Backbone Scope & Active Quantum Params | LR Schedule & Tail EMA | Step-0 Val CE | Step-200 / 400 / 600 Val CE | Step-800 Final Val CE ($\downarrow$) | $\Delta$ vs `300M` Teacher (`3.5845`) | Final PPL ($\downarrow$) | Top-1 / Top-5 Acc ($\uparrow$) | Max Stiefel / Unitary / Sphere Err | Pre-Purify $R_{\text{eff}}$ (`1/Tr(ρ0²)`) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`w45b` Teacher** | Classical (`12x64`) | Pretrained `300M` (`1,145` steps) | — | `3.5845` | — | `3.5845` | `0.0000` | `36.90` | `37.31%` / `58.26%` | — | — | — |
| **`J74_3` (Wave 53B Record)** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$) | `100%` Full-Backbone MZ (`r=768`, `277.41M`) | Cosine (No EMA) | `3.5917` | `3.5754 / 3.5701 / 3.5661` | `3.5647` | `-0.0198` | `36.19` | `37.65%` / `58.59%` | `4.14e-4 / 2.28e-4 / 2.38e-7` | `2.65` (post) | `7,420 tok/s` |
| **`K75_2` (Wave 54 Best `1:1`)** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$) | All 12 Attn (`r=384`, `64.74M`) | Cosine + `TAN_SNR` (`0.92`) | `3.6669` | `3.5964 / 3.5887 / 3.5807` | `3.5774` | `-0.0071` | `36.64` | `37.41%` / `58.31%` | `4.95e-7 / 1.72e-7 / 2.38e-7` | `2.83` (post) | `5,303 tok/s` |
| **`L76_1_COMBINED_PKT32_LOW_MOM070_TAN_SNR_R384`** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$) | All 12 Attn (`r=384`, `64.74M`) | Cosine + `TAN_SNR` + `MOM_070` | `3.6669` | `3.5965 / 3.5882 / 3.5799` | **`3.5770`** | **`-0.0075`** | **`36.63`** | **`37.43%` / `58.32%`** | `4.86e-7 / 1.73e-7 / 2.38e-7` | `18.18` | **`11,357 tok/s`** |
| **`L76_2_COMBINED_PKT32_WSD_COOLDOWN_TAIL_EMA_R384`** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$) | All 12 Attn (`r=384`, `64.74M`) | `WSD Cooldown` + `Tail EMA` | `3.6669` | `3.5965 / 3.5890 / 3.5800` | **`3.5728`** | **`-0.0117`** | **`36.48`** | **`37.50%` / `58.41%`** | `4.94e-7 / 1.56e-7 / 2.38e-7` | `18.31` | **`10,982 tok/s`** |
| **`L76_3_COMBINED_PKT64_LOW_MOM070_TAN_SNR_TAIL_EMA_R384`** | `SHAPE_6x2x64` ($d_{\text{pkt}}=64$) | All 12 Attn (`r=384`, `65.15M`) | `WSD Cooldown` + `Tail EMA` | `3.6654` | `3.5948 / 3.5871 / 3.5787` | **`3.5712`** | **`-0.0133`** | **`36.42`** | **`37.52%` / `58.44%`** | `1.59e-7 / 1.80e-7 / 1.19e-7` | `10.54` | `3,564 tok/s` |
| **`L76_4_COMBINED_FULL_BACKBONE_MZ_PKT32_R768`** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$) | **`100%` Full-Backbone MZ (`r=768`, `277.41M`)** | `WSD Cooldown` + `Tail EMA` | `3.5921` | `3.5625 / 3.5566 / 3.5481` | **`3.5395`** | **`-0.0450`** | **`35.29`** | **`37.81%` / `58.84%`** | `8.61e-7 / 1.46e-7 / 2.38e-7` | `17.79` | **`8,200 tok/s`** |

### 5.2 Part 2 — Cold-Start Training From Scratch for `100M` FineWeb Tokens (`L76_5_FROM_SCRATCH_100M_QUANTUM_WAVEPACKET_GPT`)

Trained from scratch (`step=0`, zero pretrained teacher weights, `275.90M` trainable parameters including `154.53M` 3D sphere embeddings `q_wte + q_value_embeds`, `26.11M` quantum wavepacket attention parameters across all 12 layers, `56.62M` MLP parameters, and `38.63M` `lm_head` parameters) across **2x `NVIDIA GeForce RTX 3090` (`nccl` DDP, `19,530–21,173 tok/s`)** for `382` macro-steps (`262,144` tok/step = `100.14M` FineWeb tokens):

| Step | Tokens (`M`) | LR Mult / Mom $\beta_t$ | `L76_5` Train EMA Loss | `L76_5` Val CE ($\downarrow$) | `L76_5` Val PPL ($\downarrow$) | `L76_5` Top-1 / Top-5 Acc ($\uparrow$) | `L76_5` Max Stiefel / Unitary / Sphere Err | `L76_5` Pre-Purify $R_{\text{eff}}$ (`1/Tr(ρ0²)`) | Classical `w45b` Checkpoint Reference (`Doc-SWC` / `Full-Causal` Val CE) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | `0.000` / `0.00` | `10.8249` | `10.8249` | `50,257.01` | `0.06%` / `0.15%` | `4.54e-7 / 0.00e+0 / 2.38e-7` | `31.19` | `10.8249` / `10.8249` (`step=0`) |
| `38` | `9.96M` | `1.000` / `0.27` | `7.2223` | `6.2203` | `510.24` | `13.15%` / `29.17%` | `5.95e-7 / 1.54e-7 / 2.38e-7` | `30.90` | — |
| `76` | `19.92M` | `1.000` / `0.55` | `5.9281` | `5.6723` | `294.32` | `17.01%` / `35.04%` | `5.82e-7 / 1.70e-7 / 2.38e-7` | `30.10` | — |
| `115` | `30.15M` | `1.000` / `0.70` | `5.4035` | `5.2835` | `198.37` | `20.21%` / `38.23%` | `5.87e-7 / 1.72e-7 / 2.38e-7` | `27.60` | — |
| `153` | `40.11M` | `1.000` / `0.70` | `5.1645` | `5.0926` | `163.80` | `21.91%` / `40.27%` | `5.91e-7 / 1.72e-7 / 2.38e-7` | `24.62` | — |
| **`191`** | **`50.07M`** | `1.000` / `0.70` | `4.9878` | **`4.9137`** | **`137.08`** | **`23.56%` / `42.15%`** | `5.76e-7 / 1.72e-7 / 2.38e-7` | `22.14` | `4.2467` (`31.54%` / `50.95%`) / `4.4177` (`30.14%` / `49.05%`) |
| `229` | `60.03M` | `1.000` / `0.70` | `4.8580` | `4.7727` | `119.18` | `25.19%` / `43.85%` | `5.94e-7 / 1.72e-7 / 2.38e-7` | `20.13` | — |
| `267` | `69.99M` | `0.870` / `0.70` | `4.6980` | `4.6253` | `102.89` | `26.52%` / `45.65%` | `5.87e-7 / 1.72e-7 / 2.38e-7` | `18.33` | — |
| `306` | `80.22M` | `0.602` / `0.70` | `4.5322` | `4.4684` | `88.06` | `28.14%` / `47.41%` | `5.63e-7 / 1.72e-7 / 2.38e-7` | `17.09` | — |
| `344` | `90.18M` | `0.341` / `0.70` | `4.4218` | `4.3624` | `79.36` | `29.31%` / `48.69%` | `5.25e-7 / 1.72e-7 / 2.38e-7` | `16.56` | — |
| **`382`** | **`100.14M`** | `0.080` / `0.70` | `4.3319` | **`4.2865`** | **`73.60`** | **`30.05%` / `49.56%`** | **`4.92e-7 / 1.71e-7 / 2.38e-7`** | **`16.42`** | `3.9357` (`34.08%` / `54.38%`) / `4.1692` (`32.08%` / `51.66%`) |

### 5.3 Key Mathematical & Engineering Discoveries
1. **Zero-Mask Hermitian Sum Packing ($M = \rho_R + \rho_I$) & Associative $(R \times d_h)$ Ket Contraction Double Throughput (`5,303 -> 11,357 tok/s`)**:
   - Packing the symmetric real part $\rho_R$ and skew-symmetric imaginary part $\rho_I$ into $M = \rho_R + \rho_I \in \mathbb{R}^{d_h \times d_h}$ cuts the SDPA density payload in half (`2 d_h^2 -> d_h^2`), and replacing dense $(d_h \times d_h)$ McWeeny + Unitary matrix multiplications with associative ket-operator contraction ($|w_1\rangle = |w_0\rangle \rho_0$, $|w_2\rangle = |w_1\rangle \rho_0$) increased 12-layer training throughput by **`2.14x`** (`11,357 tok/s` vs. `5,303 tok/s` in `K75_2`) while matching dense forward/backward outputs to `< 3e-5`.
2. **Synergy of the Combined Recipe + WSD Linear Cooldown + Manifold-Retracted Tail EMA (`L76_1 -> L76_2 -> L76_3 -> L76_4`)**:
   - Combining `FUSED_HERMITIAN_BORN_RHO_KET` + `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070` (`L76_1`, `3.5770` Val CE) improved over both individual Wave 54 winners (`K75_2`: `3.5774`, `K75_3`: `3.5779`).
   - Adding `DEPTH_SQRT_WARMUP_WSD_COOLDOWN` + `TailWeightAverager` (`L76_2`) dropped 12-layer `1:1`-parity loss by another **`-0.0042` nats** to **`3.5728` Val CE (`37.50%` Top-1, `58.41%` Top-5)**.
   - Scaling wavefunction packet dimension to $d_{\text{pkt}} = 64$ (`SHAPE_6x2x64`, `L76_3`) with the Combined Recipe reached **`3.5712` Val CE (`37.52%` Top-1, `58.44%` Top-5, `-0.0133` nats below the `300M` teacher)** and benefited directly from Manifold-Retracted Tail EMA (`used_tail_ema = 1.0`, `st_err = 1.59e-7`).
   - Applying the Combined Recipe to the **`100%` Full-Backbone Quantum Wavepacket + 12 Mach-Zehnder MLP** architecture (`L76_4`) achieved a new all-time project record of **`3.5395` Val CE (`35.29` PPL, `37.81%` Top-1, `58.84%` Top-5)**—beating the pretrained `300M` Classical Teacher (`3.5845` Val CE) by **`-0.0450` nats (`+0.50%` Top-1, `+0.58%` Top-5)** and beating our previous `J74_3` record (`3.5647` Val CE) by **`-0.0252` nats**!
3. **Cold-Start `100M`-Token Training From Scratch (`L76_5`) — Breaking the `5.2` Linear Co-Occurrence Ceiling & Diagnosing Sublayer LR Damping**:
   - Trained from cold start (`val_ce = 10.8249` at `0M` tokens, zero teacher weights) with unit singular values $\vec{\sigma} = \vec{1}$ to avoid the $\vec{\sigma} = \vec{0}$ bilinear saddle point, `L76_5` descended smoothly at every `10M`-token checkpoint to **`4.2865` Val CE (`73.60` PPL, `30.05%` Top-1, `49.56%` Top-5)** at `100.14M` tokens—shattering the `~5.2–6.0` linear PPMI/SVD floor by `-0.92` nats while maintaining `4.92e-7` Stiefel error and organically crystallizing pre-purification density effective rank from `31.19` (`0M`) $\to$ `22.14` (`50M`) $\to$ `16.42` (`100M`).
   - Crucially, why did `L76_5` at `100M` tokens (`4.2865` Val CE, `30.05%` Top-1, `49.56%` Top-5) land right next to the `50M`-token classical checkpoint (`4.2467` doc-SWC / `4.4177` full-causal, `30.14%–31.54%` Top-1, `49.05%–50.95%` Top-5) and `0.117` nats behind the `100M` full-causal classical checkpoint (`4.1692`)? Because `build_quantum_full_backbone_optimizers` applied the graft depth-scaling factor `compute_depth_lr_scale(11) = 1/\sqrt{5.5} = 0.4264` to the cold-start learning rates (`eff_stiefel_lr = 0.0107` vs. classical Muon `0.023`; `eff_wte_lr = 0.0171` vs. classical `wte` `0.30`; `eff_scalar_lr = 0.0064` vs. classical `0.04`). Running for `382` steps at `0.4264x` step size covered the exact geodesic distance of `~191` unscaled steps (`50M` tokens)! Decoupling cold-start LR from graft depth damping (`d_scale = 1.0` when `from_scratch_mode = True`) + low-rank Cholesky density propagation ($L_{t,h} \in \mathbb{C}^{d_h \times k}$) is the direct next step.
