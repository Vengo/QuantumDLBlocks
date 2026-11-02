# EXPERIMENT PLAN 077 — Unscaled Cold-Start Riemannian Learning Rates, Inversion-Free Low-Rank Cholesky/Gram Density Propagation, and 300M-Token From-Scratch Quantum Wavepacket GPT Training (Wave 56)

## 1. Executive Summary & Scientific Hypotheses

In **Wave 55 (`EXPERIMENT_PLAN_076.md`)**, we achieved two milestones:
1. **New All-Time Project Record on the `300M` Teacher (`L76_4_COMBINED_FULL_BACKBONE_MZ_PKT32_R768`)**: Reaching **`3.5395` Val CE** (`35.29` PPL, `37.81%` Top-1, `58.84%` Top-5, `8.61e-7` Stiefel error), beating the `300M` Classical Teacher (`3.5845` Val CE) by **`-0.0450` nats (`+0.50%` Top-1, `+0.58%` Top-5)** across `100%` of the 12-layer backbone (`q_wte`, `q_value_embeds`, 11 Quantum Wavepacket Attention layers, and 12 Mach-Zehnder Interferometer MLPs).
2. **First `100M`-Token Cold-Start From-Scratch Run (`L76_5_FROM_SCRATCH_100M_QUANTUM_WAVEPACKET_GPT`)**: Training from `step=0` (`val_ce = 10.8249`, zero teacher weights) for `100.14M` FineWeb tokens (`382` steps) descended smoothly to **`4.2865` Val CE** (`30.05%` Top-1, `49.56%` Top-5, `4.92e-7` Stiefel error) and crystallized pre-purification density effective rank from `31.19` (`0M`) $\to$ `22.14` (`50M`) $\to$ `16.42` (`100M`).

More importantly, our diagnostic inspection of `L76_5` uncovered **two exact root causes** why the `100M`-token cold-start run landed at the `50M`-token classical teacher mark (`4.2865` vs. `4.2467`), along with a major $O(k_{\text{chol}} d_h)$ kernel scaling breakthrough:
1. **Root Cause 1 — Teacher-Graft Sublayer Depth Damping (`0.4264x`) Was Inadvertently Active on Cold Start**:
   In [`build_quantum_full_backbone_optimizers`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/quantum_full_backbone_graft_gpt.py), `d_scale = compute_depth_lr_scale(n_sub) = 1.0 / sqrt(11 / 2) = 0.4264` was applied even when `model.from_scratch_mode == True`. That damping factor was designed strictly to prevent graft shock when inserting random/newly-factored quantum layers into a pre-converged `300M` teacher. On cold start (`from_scratch_mode == True`), it throttled effective learning rates down to `eff_stiefel_lr = 0.00639` (vs. classical Muon `0.023–0.040`), `eff_wte_lr = 0.01066` (vs. classical `wte` `0.08–0.30`), and `eff_scalar_lr = 0.00512`. Consequently, `382` steps at `0.4264x` step size covered the exact geodesic distance of `~163` unscaled steps (`~43M` tokens).
2. **Root Cause 2 — Dynamic Sliding-Window Expansion (`ws_blocks: 3 -> 6` at `150M` Tokens / `step = 574`)**:
   Inspecting the saved `w45b` classical teacher checkpoints (`0M, 50M, 100M, 150M, 200M, 250M, 300M`) revealed that `w45b` trains with `ws_blocks = 3` (`long_window = 384`, `short_window = 128`) from `0M` to `150M` tokens (`steps 0..573`), and expands in-place to `ws_blocks = 6` (`long_window = 768`, `short_window = 384` with YaRN RoPE scaling) at `step = 574` (`150M -> 300M` tokens).
3. **Breakthrough 3 — Inversion-Free Low-Rank Cholesky / Gram Density Propagation (`FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY`)**:
   Instead of propagating a full $d_h \times d_h$ outer-product matrix $M_{s,h}^{\text{tok}} \in \mathbb{R}^{d_h^2}$ (`1,024` floats/head at $d_h=32$; `4,096` floats/head at $d_h=64$) through causal SDPA, each token emits a rank-$k_{\text{chol}}$ ($k_{\text{chol}} = 4$) complex Cholesky/ket factor $F_{s,h} \in \mathbb{C}^{k_{\text{chol}} \times d_h}$ ($2 k_{\text{chol}} d_h = 256$ floats at $d_h=32$, $512$ floats at $d_h=64$ — a **$4\times$ to $8\times$ density bandwidth reduction**). Causal attention linearly aggregates $L_{t,h} = \sum_{s \le t} A_{t,s}^{(h)} F_{s,h} \in \mathbb{C}^{k_{\text{chol}} \times d_h}$, guaranteeing $\rho_{t,h}^{(0)} = \hat{L}_{t,h}^\dagger \hat{L}_{t,h} \succeq 0$ with $\operatorname{Tr}(\rho_{t,h}^{(0)}) = 1$ by construction, and performs Quadratic McWeeny purification via the $4 \times 4$ Hermitian Gram matrix $G_{t,h} = \hat{L}_{t,h} \hat{L}_{t,h}^\dagger \in \mathbb{C}^{4 \times 4}$ in $O(k_{\text{chol}}^2 d_h)$ FLOPs with **zero Cayley or triangular matrix inversions**.

In **Wave 56 (`EXPERIMENT_PLAN_077.md`)**, we implement all three advances, run a fast head-to-head kernel tournament (`M77_1`, `M77_2`), and execute a full **`300M`-token cold-start from-scratch run (`M77_3_FROM_SCRATCH_300M_QUANTUM_WAVEPACKET_GPT`, `1,145` macro-steps at `262,144` tok/step = `300.15M` FineWeb tokens)** across `2x NVIDIA GeForce RTX 3090` (`nccl` DDP).

---

## 2. Mathematical Formulations

### 2.1 Inversion-Free Low-Rank Cholesky / Gram Density Propagation (`FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY`)

For each token $s$ and head $h$, let $F_{s,h} = F_{s,h,R} + i F_{s,h,I} \in \mathbb{C}^{k_{\text{chol}} \times d_h}$ ($k_{\text{chol}} = 4$) be the unit-Frobenius complex Cholesky/ket factor formed from the four orthogonal physical wavepacket branches at token $s$ (normalized packet ket $|r_{s,h}\rangle$, mixed value ket $|v_{s,h}\rangle$, rotary key ket $|k_{s,h}\rangle$, and rotary query ket $|q_{s,h}\rangle$):
$$\|F_{s,h}\|_F^2 = \sum_{m=1}^{k_{\text{chol}}} \left(\|F_{s,h,R,m}\|_2^2 + \|F_{s,h,I,m}\|_2^2\right) = 1$$
1. **Single-Pass Causal Aggregation**:
   Using `_FusedHermitianBornRhoKetAttnFn`, causal Hermitian+Born attention aggregates both $F_{s,h}$ and $|v_{\text{mixed}, s, h}\rangle$ in a single pass:
   $$L_{t,h} = \sum_{s \le t} A_{t,s}^{(h)} F_{s,h} \in \mathbb{C}^{k_{\text{chol}} \times d_h}, \qquad |\bar{v}_{t,h}\rangle = \sum_{s \le t} A_{t,s}^{(h)} |v_{\text{mixed}, s, h}\rangle \in \mathbb{C}^{R \times d_h}$$
   Normalized to unit Frobenius norm $\hat{L}_{t,h} = L_{t,h} / \sqrt{\|L_{t,h}\|_F^2 + \epsilon}$, the induced context density operator is the Cholesky outer product
   $$\bar{\rho}_{t,h} = \hat{L}_{t,h}^\dagger \hat{L}_{t,h} \in \mathbb{C}^{d_h \times d_h}, \qquad \bar{\rho}_{t,h} \succeq 0, \qquad \operatorname{Tr}(\bar{\rho}_{t,h}) = \|\hat{L}_{t,h}\|_F^2 = 1$$
2. **$4 \times 4$ Gram Matrix McWeeny Purification (`associative_low_rank_cholesky_gram_modulation`)**:
   Instead of forming the $d_h \times d_h$ matrix $\bar{\rho}_{t,h}$, we form the $k_{\text{chol}} \times k_{\text{chol}}$ ($4 \times 4$) Hermitian Gram matrix:
   $$G_{t,h} = \hat{L}_{t,h} \hat{L}_{t,h}^\dagger = G_{t,h,R} + i G_{t,h,I} \in \mathbb{C}^{k_{\text{chol}} \times k_{\text{chol}}}$$
   where $G_{R} = \hat{L}_R \hat{L}_R^\top + \hat{L}_I \hat{L}_I^\top$ and $G_{I} = \hat{L}_I \hat{L}_R^\top - \hat{L}_R \hat{L}_I^\top$. By cyclic trace invariance:
   $$\operatorname{Tr}(\bar{\rho}_{t,h}^2) = \operatorname{Tr}(\hat{L}^\dagger \hat{L} \hat{L}^\dagger \hat{L}) = \operatorname{Tr}(G_{t,h}^2) = \|G_{t,h,R}\|_F^2 + \|G_{t,h,I}\|_F^2$$
   And for any rotated target ket $|w_0\rangle = |\psi_{\text{tar}}\rangle \overline{U_h} \in \mathbb{C}^{R \times d_h}$, applying $\overline{\bar{\rho}_{t,h}^2} = (\hat{L}^\dagger G_{t,h} \hat{L})^*$ factors into three tiny contractions:
   $$|c_1\rangle = |w_0\rangle \hat{L}_{t,h}^\dagger \in \mathbb{C}^{R \times k_{\text{chol}}}, \qquad |c_2\rangle = |c_1\rangle \overline{G_{t,h}} \in \mathbb{C}^{R \times k_{\text{chol}}}, \qquad |w_2\rangle = |c_2\rangle \overline{\hat{L}_{t,h}} \in \mathbb{C}^{R \times d_h}$$
   followed by $|\delta\psi\rangle = \left(\frac{|w_2\rangle}{\|G_{t,h}\|_F^2} - \frac{1}{d_h}|w_0\rangle\right) U_h^\top$. This matches dense $d_h \times d_h$ McWeeny purification on $\bar{\rho}_{t,h} = \hat{L}_{t,h}^\dagger \hat{L}_{t,h}$ to `< 1e-6` numerical precision while reducing density SDPA payload width by **$4\times$ at $d_h=32$** (`1,024 -> 256`) and **$8\times$ at $d_h=64$** (`4,096 -> 512`).

### 2.2 Unscaled Cold-Start Riemannian Learning Rates & `150M`-Token Window Expansion

1. **Decoupling Cold-Start Learning Rates from Graft Depth Damping**:
   In [`build_quantum_full_backbone_optimizers`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/quantum_full_backbone_graft_gpt.py):
   $$d_{\text{scale}} = \begin{cases} 1.0, & \text{if } \texttt{model.from\_scratch\_mode} = \text{True} \\ 1 / \sqrt{\max(1.0, N_{\text{sub}} / 2.0)}, & \text{otherwise (teacher-graft recovery)} \end{cases}$$
   With `DEFAULT_FROM_SCRATCH_WTE_LR = 0.060`, `DEFAULT_FROM_SCRATCH_STIEFEL_LR = 0.025`, `DEFAULT_FROM_SCRATCH_UNITARY_LR = 0.025`, `DEFAULT_FROM_SCRATCH_SCALAR_LR = 0.020`, and `DEFAULT_FROM_SCRATCH_HEAD_LR = 0.008`, cold-start training steps at full Riemannian geodesic speed from Step 1 while `RIEMANNIAN_TANGENT_SNR_GATED` + `LOW_STEADY_MOMENTUM_070` prevent manifold overshoot.
2. **Exact `w45b` Sliding-Window Curriculum (`ws_blocks: 3 -> 6` at Step `574` / `150M` Tokens)**:
   `QuantumFullBackboneGraftGPT.set_sliding_window_blocks(ws_blocks, apply_yarn=True)` updates both `self.classical.set_sliding_window_blocks(ws_blocks, apply_yarn=apply_yarn)` and `self.q_rope` so that during `300M`-token cold-start training (`1,145` steps), `ws_blocks` transitions from `3` (`0M..150M` tokens, `steps 1..573`) to `6` (`150M..300M` tokens, `steps 574..1145`), matching the saved `w45b` teacher checkpoints identically.

### 2.3 Top-of-File Named Constants (Rule 7 Synchronization)

| Constant Name | Value | Meaning |
|---|---|---|
| `DEFAULT_LOW_RANK_CHOLESKY_K` | `4` | Rank $k_{\text{chol}}$ of complex Cholesky factor $L_{t,h} \in \mathbb{C}^{k_{\text{chol}} \times d_h}$ |
| `DEFAULT_FROM_SCRATCH_WTE_LR` | `0.060` | Unscaled cold-start 3D sphere embedding learning rate |
| `DEFAULT_FROM_SCRATCH_STIEFEL_LR` | `0.025` | Unscaled cold-start Stiefel manifold learning rate |
| `DEFAULT_FROM_SCRATCH_UNITARY_LR` | `0.025` | Unscaled cold-start Unitary $U(d_h)$ learning rate |
| `DEFAULT_FROM_SCRATCH_SCALAR_LR` | `0.020` | Unscaled cold-start singular-value & gate AdamW learning rate |
| `DEFAULT_FROM_SCRATCH_300M_STEPS` | `1145` | Macro-steps at `262,144` tok/step (`300.15M` tokens, matching `ckpt_step01145_0300M`) |
| `DEFAULT_WS_EXPAND_STEP_150M` | `574` | Macro-step (`150.2M` tokens) where `ws_blocks` expands from `3` to `6` with YaRN |

---

## 3. Planned File Changes & Experimental Tournament Matrix (`Arms M77_1..M77_3`)

### 3.0 Planned File Changes
- [`src/qdlblocks/kernels/fused_quantum_wavepacket_attn.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/kernels/fused_quantum_wavepacket_attn.py): Adds `QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY`, `DEFAULT_LOW_RANK_CHOLESKY_K = 4`, `fused_low_rank_cholesky_ket_attention`, and `associative_low_rank_cholesky_gram_modulation`.
- [`src/qdlblocks/models/quantum_full_backbone_graft_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/quantum_full_backbone_graft_gpt.py): Decouples `d_scale = 1.0` when `from_scratch_mode = True` in `build_quantum_full_backbone_optimizers`, updates cold-start LR constants, and adds `QuantumFullBackboneGraftGPT.set_sliding_window_blocks`.
- [`examples/87_verify_plan077_unscaled_cold_start_cholesky_and_300m_from_scratch.py`](file:///home/nixos/QuantumDLBlocks/examples/87_verify_plan077_unscaled_cold_start_cholesky_and_300m_from_scratch.py): Runs the Low-Rank Cholesky kernel tournament (`M77_1`, `M77_2`) and the 2-GPU `nccl` DDP **`300M`-token cold-start from-scratch run (`M77_3_FROM_SCRATCH_300M_QUANTUM_WAVEPACKET_GPT`)** with incremental JSON checkpointing every `25M` tokens.
- [`tests/test_plan077_unscaled_cold_start_cholesky_and_300m_from_scratch.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan077_unscaled_cold_start_cholesky_and_300m_from_scratch.py): Deterministic unit test suite for Gates `M1..M4`.

### Part 1: Inversion-Free Low-Rank Cholesky / Gram Kernel Tournament on `300M` Teacher (`800` Steps)

| Arm ID | Wavefunction Shape | Attention Kernel Mode | SDPA Density Payload / Head | Purpose |
|---|---|---|---|---|
| **`M77_1_LOW_RANK_CHOLESKY_PKT32_R384`** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$) | `FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY` ($k_{\text{chol}}=4$) | `256` floats (`4x` smaller than `1,024`) | Tests $4\times 4$ Gram McWeeny purification + low-rank Cholesky density propagation at $d_{\text{pkt}}=32$ |
| **`M77_2_LOW_RANK_CHOLESKY_PKT64_R384`** | `SHAPE_6x2x64` ($d_{\text{pkt}}=64$) | `FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY` ($k_{\text{chol}}=4$) | `512` floats (`8x` smaller than `4,096`) | Tests $8\times$ SDPA bandwidth reduction + $4\times 4$ Gram McWeeny purification at $d_{\text{pkt}}=64$ |

### Part 2: Full `300M`-Token Cold-Start From-Scratch Run (`M77_3`)

| Arm ID | Initialization | Token Budget & Schedule | Classical Teacher Trajectory Checkpoints (`w45b`) |
|---|---|---|---|
| **`M77_3_FROM_SCRATCH_300M_QUANTUM_WAVEPACKET_GPT`** | Cold-start (`step=0`, `val_ce = 10.8249`, zero teacher weights, `PPMI_SVD_SPHERE` `q_wte`, `d_scale = 1.0` unscaled cold-start LRs) | **`300.15M` tokens** (`1,145` steps $\times$ `262,144` tok/step across `2x RTX 3090` `nccl` DDP, `ws_blocks: 3 -> 6` at `150M` tokens, evaluated every `25M` tokens) | `0M` (`10.8249`), `50M` (`4.2467`), `100M` (`3.9357`), `150M` (`3.7969`), `200M` (`3.6685`), `250M` (`3.6210`), `300M` (`3.5844`) |

---

## 4. Testing, Verification Gates (`Gates M1..M4`), and Operational Tooling

### 4.1 Operational Tooling
- `scripts/verify_plan.py --plan 077`: Audits plan structure, quantitative thresholds, and AST compliance (`<= 8` args, zero semicolon packing).
- `tests/test_plan077_unscaled_cold_start_cholesky_and_300m_from_scratch.py`: Deterministic unit tests for Gates `M1..M4`.
- `examples/87_verify_plan077_unscaled_cold_start_cholesky_and_300m_from_scratch.py`: GPU tournament and 2-GPU `nccl` DDP `300M`-token cold-start runner.

### 4.2 Verification Gates (`Gates M1..M4`)

1. **`Gate M1` — Exact Algebraic Parity of $4 \times 4$ Gram McWeeny Contraction (`associative_low_rank_cholesky_gram_modulation`)**:
   Verify that for any complex Cholesky factor $\hat{L}_{t,h} \in \mathbb{C}^{k_{\text{chol}} \times d_h}$ ($d_h \in \{32, 64\}, k_{\text{chol}} = 4$), `associative_low_rank_cholesky_gram_modulation` matches `associative_mcweeny_unitary_ket_modulation` evaluated on the materialized outer-product density matrix $\bar{\rho}_{t,h} = \hat{L}_{t,h}^\dagger \hat{L}_{t,h}$ to `< 1e-5` max absolute difference in both output wavefunctions and purity $\operatorname{Tr}(\bar{\rho}^2) = \|G\|_F^2$, with finite gradients on all inputs.
2. **`Gate M2` — Unscaled Cold-Start Learning Rate Verification (`d_scale = 1.0` when `from_scratch_mode = True`)**:
   Verify that `build_quantum_full_backbone_optimizers(model)` applies `d_scale = 1.0` (`eff_stiefel_lr = 0.025`, `eff_wte_lr = 0.060`, `eff_scalar_lr = 0.020`) when `model.from_scratch_mode == True`, while preserving `d_scale = compute_depth_lr_scale(n_sub) < 1.0` when `model.from_scratch_mode == False`.
3. **`Gate M3` — `FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY` End-to-End 12-Layer Forward/Backward & Manifold Integrity**:
   Verify that `QuantumFullBackboneGraftGPT` configured with `QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY` executes forward, backward, and optimizer steps cleanly at both $d_{\text{pkt}} = 32$ and $d_{\text{pkt}} = 64$ with `< 1e-4` Stiefel/Unitary/Sphere error.
4. **`Gate M4` — Cold-Start `set_sliding_window_blocks` Curriculum & Step-0 Parity (`10.8249`)**:
   Verify that `QuantumFullBackboneGraftGPT.from_scratch` starts at exact uniform cross-entropy $\ln(50257) \approx 10.8249$, supports in-place `set_sliding_window_blocks(6, apply_yarn=True)` window expansion (`long_window: 384 -> 768`, `short_window: 128 -> 384`), and strictly reduces loss over synthetic training steps.
