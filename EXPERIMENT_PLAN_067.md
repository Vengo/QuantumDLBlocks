# EXPERIMENT PLAN 067 (Wave 52C): Outer-Product Hilbert Granularity ($d_{\text{pkt}} \in \{8, 16, 32\}$), Multi-Token Causal Density-Matrix State Preparation, and Two-Speed Persistent Subspace Recovery

**Date**: 2026-11-01
**Status**: Revised for Final Independent Context-Free Review (`egm-implementation-readiness`)
**Prerequisite Plans**:
- [`EXPERIMENT_PLAN_065.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_065.md) (`Wave 52A`: 12-Arm Tournament across Wavefunction Shapes, State I/O, Interior Attention Ops, and Replacement Scope — **Completed**)
- [`EXPERIMENT_PLAN_066.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_066.md) (`Wave 52B`: Procrustes Spherical `wte` Alignment, Frozen `VE_0` Coupling, and Two-Stage Interface/Feature Warmup — **In Progress on GPUs**)
**Target Implementation Files**:
- [`src/qdlblocks/layers/wavepacket_statmech_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/wavepacket_statmech_block.py) (hosts generalized $d_{\text{pkt}} \in \{8, 16, 32\}$ outer-product/chunking helpers and `form_causal_ensemble_rho_chunks` so `quantum_frontend_graft.py` stays well below the `1,000` executable SLOC limit)
- [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py)
- [`tests/test_plan067_quantum_hilbert_granularity_and_persistent_recovery.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan067_quantum_hilbert_granularity_and_persistent_recovery.py)
- [`examples/77_verify_plan067_hilbert_granularity_and_sub300m_recovery.py`](file:///home/nixos/QuantumDLBlocks/examples/77_verify_plan067_hilbert_granularity_and_sub300m_recovery.py)

---

## 1. Empirical Discoveries from Plan 065 & Plan 066 Motivating Plan 067

Our completed 12-arm **Plan 065 (`Wave 52A`)** tournament (`500` steps = `4.10M` tokens on the frozen `300M`-token Classical Teacher, `val_ce = 3.5845`, `top1 = 37.31%`) and live **Plan 066 (`Wave 52B`)** runs established four concrete empirical facts:

1. **Larger Wavefunction Packets (`(V, 2, 512)`, `T3`: `3.9218`) and Compact Wavepackets (`(V, 2, 256)`, `T2`: `3.9390` at `0.23M` L0 params) Outperform the `(V, 24, 2, 16)` Baseline (`T1`: `3.9457`)**:
   - However, in Plan 065, both `SHAPE_1x2x256` and `SHAPE_1x2x512` factored their internal attention outer products into packets of dimension $d_{\text{pkt}} = 16$ ($16 \times 16$ complex density matrices), while `SHAPE_256x2x2` (`T4`: `3.9311`) used $d_{\text{pkt}} = 2$ ($2 \times 2$ Bloch density matrices).
   - **Open Question (Axis H)**: At fixed total embedding dimension $D_{\text{emb}} \in \{768, 1024\}$, how does the **interior outer-product Hilbert dimension $d_{\text{pkt}} \in \{8, 16, 32\}$** ($8 \times 8$ vs. $16 \times 16$ vs. $32 \times 32$ complex outer products $|\tilde{v}_{s,h,r}\rangle\langle\tilde{v}_{s,h,r}| \in \mathbb{C}^{d_{\text{pkt}} \times d_{\text{pkt}}}$) affect validation cross-entropy, quantum effective rank, parameter count, and throughput?
2. **Density-Matrix Input/Output (`DM_IN_DM_OUT`, `T5`: `3.9232` Val CE, `54.26%` Top-5, `q_rank = 5.98`) Significantly Beats Pure Wavefunction I/O (`WF_IN_WF_OUT`, `T1`: `3.9457`, `q_rank = 1.55`)**:
   - Entering attention with a rank-2 mixed density matrix $\rho_{b,t}^{\text{in}} = \frac{1}{1+|\alpha_0|}(\hat{\psi}_{b,t}\hat{\psi}_{b,t}^\dagger + |\alpha_0|\hat{\psi}_{0,b,t}\hat{\psi}_{0,b,t}^\dagger)$ improved validation loss by `-0.0225` nats and boosted quantum effective rank by `3.86x` (`5.98` vs. `1.55`).
   - **Open Question (Axis I)**: In the classical teacher, `smear_gate` mixes token $t$ with token $t-1$. In `WF_IN_WF_OUT`, adding $\psi_{b,t-1}$ to $\psi_{b,t}$ is a coherent amplitude superposition that remains rank-1 ($\operatorname{Tr}(\rho^2) = 1$). What if we instead form a **Rank-3 Causal Multi-Token Ensemble Density Matrix** at the input to Layer 0:
     $$\rho_{b,t}^{\text{in}} = \frac{\hat{\psi}_{b,t}\hat{\psi}_{b,t}^\dagger + |\alpha_0|\,\hat{\psi}_{0,b,t}\hat{\psi}_{0,b,t}^\dagger + |\alpha_{\text{lag}}|\,\hat{\psi}_{b,t-1}\hat{\psi}_{b,t-1}^\dagger}{1 + |\alpha_0| + \mathbf{1}_{t > 0}|\alpha_{\text{lag}}|}$$
     so that temporal bigrams $(t-1, t)$ enter Attention as a true statistical ensemble of density matrices?
3. **Why Plan 066 `R4_TEACHER_SPHERE_VE0_2STAGE` Hit `val_ce = 3.8475` (`34.46%` Top-1, `55.22%` Top-5) in Just `100` Steps (`0.82M` Tokens) Before Drifting When Alignment Turned Off**:
   - In Plan 066, `TEACHER_PROCRUSTES_SPHERE` reduced Step-0 (`0`-token) validation loss from `8.2096` (`4.64%` Top-1) to **`5.1122` (`21.95%` Top-1, `40.40%` Top-5)** (`4.8975` / `24.15%` Top-1 for `R5`), and by Step 100 (`0.82M` tokens) `R4` reached **`val_ce = 3.8475` (`34.46%` Top-1, `55.22%` Top-5)** — already halfway between the `100M`-token (`3.9354`) and `150M`-token (`3.7495`) classical teacher checkpoints in `< 1M` tokens!
   - However, when `T_align = 150` abruptly dropped the scale-invariant `_normalized_cosine_mse` alignment weight $\lambda_{\text{align}}(t)$ to `0.0` while `VocabManifoldMuon` was still running at a high from-scratch learning rate (`wte_lr = 0.04`), stochastic mini-batch 1-hot CE gradients perturbed `q_wte` away from the teacher's coordinate frame (`val_ce = 3.9051` at Step 200) before recovering to `3.8009` (`35.05%` Top-1) at Step 500 when the cosine LR cooled down.
   - **Open Question (Axis J)**: If we combine (a) a **Two-Speed Riemannian Learning Rate** (`wte_lr = 0.006` for the already-aligned `q_wte` sphere + `stiefel_lr = 0.035` for the randomly initialized Layer-0 Stiefel/Unitary operators) with (b) **Persistent Floored Subspace Co-Distillation (`PERSISTENT_COSINE_FEAT_AND_CE`)** that keeps `_normalized_cosine_mse` active across the full run with a cosine decay from $\lambda_{\text{align},0} = 1.5$ to a positive floor $\lambda_{\text{align},\min} = 0.25\,\lambda_{\text{align},0} = 0.375$ (`DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO: Final[float] = 0.25`), does validation loss descend monotonically below `3.70` and approach `3.5845` within a tiny fraction of `300M` tokens?

---

## 2. Mathematical Formulation of New Axes (Axes H, I, and J)

### 2.1 Axis H — Outer-Product Hilbert Dimension $d_{\text{pkt}} \in \{8, 16, 32\}$ (`QuantumWavefunctionShape` Extension)

We extend `QuantumWavefunctionShape(str, Enum)` and `resolve_wavefunction_shape_spec(wf_shape)` with three new outer-product granularities:
1. `SHAPE_48x2x8 = "48x2x8"` (`D_emb = 768`, **many smaller $8 \times 8$ complex wavefunctions**):
   - `wte_slices = 96`, `wte_sphere_dim = 8`, `d_emb = 768`, `pkt_count = 96` (`48` complex modes), `pkt_dim = 8`, `q_heads = 6` (`v_rank = R = 8` complex modes/head).
2. `SHAPE_12x2x32 = "12x2x32"` (`D_emb = 768`, **fewer, larger $32 \times 32$ complex wavefunctions**):
   - `wte_slices = 24`, `wte_sphere_dim = 32`, `d_emb = 768`, `pkt_count = 24` (`12` complex modes), `pkt_dim = 32`, `q_heads = 6` (`v_rank = R = 2` complex modes/head).
3. `SHAPE_1x2x512_PKT32 = "1x2x512_pkt32"` (`D_emb = 1024`, **overcomplete `(V, 2, 512)` with $32 \times 32$ complex outer products**):
   - `wte_slices = 2`, `wte_sphere_dim = 512`, `d_emb = 1024`, `pkt_count = 32` (`16` complex modes), `pkt_dim = 32`, `q_heads = 8` (`v_rank = R = 2` complex modes/head).

#### Exact Chunk-Indexing Formulas for Arbitrary $(R, d_h) \in \{(8, 8),\; (4, 16),\; (2, 32)\}$
Across all three $(R, d_h)$ pairs (`v_rank` $= R \in \{8, 4, 2\}$, `d_h` $= d_{\text{pkt}} \in \{8, 16, 32\}$), the product $2 \cdot R \cdot d_h = 128$ (`sdpa_dim = 128`) is invariant. To avoid conflating the number of $\rho$ chunks (`k_rho`) with the number of modes per head (`v_rank`), both [`form_wavepacket_outer_product_and_v`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/wavepacket_statmech_block.py) (along with [`unpack_attended_rho_and_v`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/wavepacket_statmech_block.py)) and [`ConfigurableWavepacketStatMechBlock._run_attention_and_evolution`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py) use the exact chunk-indexing formulas:
$$\texttt{k\_rho} = \max(1,\; \texttt{d\_h} // \texttt{v\_rank}) \in \{1, 4, 16\}, \qquad \texttt{row\_chunk} = \texttt{v\_rank} \in \{8, 4, 2\}, \qquad \texttt{sdpa\_dim} = 2 \cdot \texttt{row\_chunk} \cdot \texttt{d\_h} = 128$$
Specifically:
- `rho_tok = torch.stack([rho_re, rho_im], dim=-3)` has shape `(B, T, H, 2, d_h, d_h)` and is reshaped into:
  `sdpa_rho = rho_tok.permute(0, 1, 2, 4, 3, 5).reshape(B, T, H, k_rho, row_chunk, 2, d_h).reshape(B, T, H, k_rho, sdpa_dim)`
  where `k_rho * row_chunk == d_h` (`1 * 8 = 8`, `4 * 4 = 16`, `16 * 2 = 32`) and `row_chunk * 2 * d_h == 128` (`sdpa_dim`) identically!
- `sdpa_ket = v_norm_kets.reshape(B, T, H, 1, sdpa_dim)` has `1` chunk of size `2 * v_rank * d_h = 128`.
- When `state_io_mode == QuantumStateIOMode.DM_IN_DM_OUT`: `chunks = k_rho` (`sdpa_v = sdpa_rho.reshape(B, T, H * k_rho, sdpa_dim)`).
- For all other `state_io_mode` values (including `WF_IN_WF_OUT`, `WF_IN_DENSITY_OBS_OUT`, and `CAUSAL_ENSEMBLE_DM_IO`): `chunks = k_rho + 1` (`sdpa_v = torch.cat([sdpa_rho, sdpa_ket], dim=3).reshape(B, T, H * (k_rho + 1), sdpa_dim)`), and after `flash_attn_func`, `out_chunks[:, :, :, :k_rho, :]` is unpacked via `.reshape(B, T, H, d_h, 2, d_h).permute(0, 1, 2, 4, 3, 5)` into `(rho_0_re, rho_0_im)` of shape `(B, T, H, d_h, d_h)`, while `out_chunks[:, :, :, k_rho, :]` is unpacked via `.reshape(B, T, H, 2, v_rank, d_h)` into `(v_bar_re, v_bar_im)`.
- To keep [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py) strictly `< 1,000` executable SLOC, the chunk-packing and Rank-3 causal ensemble density-matrix helper `build_configurable_rho_and_ket_sdpa_chunks` is implemented in [`src/qdlblocks/layers/wavepacket_statmech_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/wavepacket_statmech_block.py) and called by `ConfigurableWavepacketStatMechBlock`.

---

### 2.2 Axis I — Rank-3 Causal Multi-Token Ensemble Density-Matrix Input (`CAUSAL_ENSEMBLE_DM_IO`)

1. **Enum & Constants**:
   - Extend `QuantumStateIOMode(str, Enum)` with `CAUSAL_ENSEMBLE_DM_IO = "causal_ensemble_dm_io"`.
   - Define top-of-module constants in `wavepacket_statmech_block.py` and re-export in `quantum_frontend_graft.py`:
     - `DEFAULT_DM_RES_MIX_INIT: Final[float] = 0.25`
     - `DEFAULT_DM_LAG_MIX_INIT: Final[float] = 0.20`
     - `DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO: Final[float] = 0.25`
2. **Learnable Scalar Parameters & Clamping**:
   - On `ConfigurableWavepacketStatMechBlock`, declare:
     - `self.dm_res_mix = nn.Parameter(torch.tensor(DEFAULT_DM_RES_MIX_INIT, dtype=torch.float32))`
     - `self.dm_lag_mix = nn.Parameter(torch.tensor(DEFAULT_DM_LAG_MIX_INIT, dtype=torch.float32))`
     with `self.dm_res_mix.requires_grad_(self.state_io_mode in (QuantumStateIOMode.DM_IN_DM_OUT, QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO))` and `self.dm_lag_mix.requires_grad_(self.state_io_mode == QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO)`.
   - Register both in `ConfigurableWavepacketStatMechBlock.scalar_parameters()` (which automatically filters by `p.requires_grad`).
   - In `FrozenTrunkQuantumFrontEndGPT.restrict_quantum_scalars_()`, clamp `dm_res_mix` and `dm_lag_mix` to `[0.01, 1.00]`.
3. **Exact Tensor Construction & Readout Path for `CAUSAL_ENSEMBLE_DM_IO`**:
   - Given normalized $V$-projected kets $|\tilde{v}_{b,t,h,r}\rangle = \texttt{self.\_normalize\_mode\_kets}(v_{\text{wp}})$ and normalized pre-projection input kets $|\tilde{v}_{0,b,t,h,r}\rangle = \texttt{self.\_normalize\_mode\_kets}(h_{\text{norm}})$ (each of shape `(B, T, H, 2, R, d_h)`):
     - Construct the causal lag-1 ket tensor $|\tilde{v}^{\text{lag}}_{b,t,h,r}\rangle$ of shape `(B, T, H, 2, R, d_h)` by zero-padding 1 step at $t = 0$ along `dim=1`:
       `v_lag_kets = torch.cat([torch.zeros_like(v_norm_kets[:, :1]), v_norm_kets[:, :-1]], dim=1)` when `T > 1` else `torch.zeros_like(v_norm_kets)`.
     - Compute the three trace-1 head outer products:
       $$\rho_{b,t,h}^{(v)} = \frac{1}{R}\sum_{r=1}^R |\tilde{v}_{b,t,h,r}\rangle\langle\tilde{v}_{b,t,h,r}|, \quad \rho_{b,t,h}^{(0)} = \frac{1}{R}\sum_{r=1}^R |\tilde{v}_{0,b,t,h,r}\rangle\langle\tilde{v}_{0,b,t,h,r}|, \quad \rho_{b,t,h}^{(\text{lag})} = \frac{1}{R}\sum_{r=1}^R |\tilde{v}^{\text{lag}}_{b,t,h,r}\rangle\langle\tilde{v}^{\text{lag}}_{b,t,h,r}|$$
     - Combine with position-dependent normalization $Z_{b,t} = 1 + |\alpha_0| + \mathbf{1}_{t > 0}|\alpha_{\text{lag}}|$ (where $\alpha_0 = \texttt{self.dm\_res\_mix}$, $\alpha_{\text{lag}} = \texttt{self.dm\_lag\_mix}$):
       $$\rho_{b,t,h}^{\text{tok}} = \frac{1}{Z_{b,t}}\left(\rho_{b,t,h}^{(v)} + |\alpha_0|\,\rho_{b,t,h}^{(0)} + \mathbf{1}_{t > 0}|\alpha_{\text{lag}}|\,\rho_{b,t,h}^{(\text{lag})}\right)$$
       so that $\operatorname{Tr}(\rho_{b,t,h}^{\text{tok}}) = 1$ identically at both $t = 0$ and $t > 0$!
     - **Readout Path**: `CAUSAL_ENSEMBLE_DM_IO` packs `chunks = k_rho + 1` (`torch.cat([sdpa_rho, sdpa_ket], dim=3)`) through `flash_attn_func` and modulates the $V$-space wavefunctions (`ATTENDED_RHO_ON_LOCAL_V` or `ATTENDED_RHO_ON_ATTENDED_V`) via the purified and channel-evolved $\rho_{\text{evol}, t,h}$ — combining the high quantum effective rank of multi-state density-matrix preparation with direct $V$-space wavefunction readout.

---

### 2.3 Axis J — Persistent Floored Teacher-Subspace Co-Distillation & Two-Speed Riemannian LR

1. **Extending `QuantumRecoveryScheduleMode(str, Enum)`**:
   - Add `PERSISTENT_COSINE_FEAT_AND_CE = "persistent_cosine_feat_and_ce"`.
   - In [`FrozenTrunkQuantumFrontEndGPT.forward`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py):
     - Change the alignment guard condition to:
       `is_persistent = (sched == QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE)`
       `if self.training and targets is not None and t_align > 0 and (0 <= self.training_step < t_align or is_persistent) and sched != QuantumRecoveryScheduleMode.PURE_CE:`
     - Compute $\lambda_{\text{align}}(t)$:
       - If `is_persistent`:
         `prog = min(1.0, float(max(0, self.training_step)) / float(t_align))`
         `floor_w = DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO * float(self.graft_cfg.interface_align_weight)`
         `lam_t = floor_w + 0.5 * (float(self.graft_cfg.interface_align_weight) - floor_w) * (1.0 + math.cos(math.pi * prog))`
       - Otherwise (`INTERFACE_WARMUP_TO_PURE_CE` or `INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE`):
         `lam_t = 0.5 * float(self.graft_cfg.interface_align_weight) * (1.0 + math.cos(math.pi * float(self.training_step) / float(t_align)))`
     - Include final-feature cosine MSE (`DEFAULT_FINAL_FEAT_ALIGN_SCALE * _normalized_cosine_mse(x_final, x_final_c)`) whenever `sched in (QuantumRecoveryScheduleMode.INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE, QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE)`.
2. **Two-Speed Riemannian Learning Rate for Procrustes-Initialized `q_wte`**:
   - Add `--wte-lr` (default `0.006` in [`examples/77_verify_plan067_hilbert_granularity_and_sub300m_recovery.py`](file:///home/nixos/QuantumDLBlocks/examples/77_verify_plan067_hilbert_granularity_and_sub300m_recovery.py)) and `--stiefel-lr` (default `0.035`) so `VocabManifoldMuon` fine-tunes the Procrustes-aligned `q_wte` gently while `RiemannianStiefelOptimizer` adapts Layer-0 Stiefel/Unitary matrices rapidly.

---

## 3. Experimental Design (`examples/77_verify_plan067_hilbert_granularity_and_sub300m_recovery.py`)

### 3.1 Phase 1 — Screening Axes H, I, and J (`500` Steps = `4.10M` Tokens on the `300M` Teacher)

| Arm ID | Shape & Hilbert Dim $d_{\text{pkt}}$ (Axis H) | State I/O (Axis I) | Recovery Curriculum & LR (Axis J) | Active L0 Params | Primary Hypothesis Tested |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`G1_PKT8_48x2x8_DM`** | `SHAPE_48x2x8` ($d_{\text{pkt}}=8$, $8 \times 8$ $\rho$) | `CAUSAL_ENSEMBLE_DM_IO` | `PERSISTENT_COSINE_FEAT_AND_CE` (`wte_lr=0.006`) | `0.19M` | Do 48 smaller $8 \times 8$ complex wavefunctions (`2` FlashAttention chunks/head) match $16 \times 16$ at higher speed? |
| **`G2_PKT16_24x2x16_DM`** | `SHAPE_24x2x16` ($d_{\text{pkt}}=16$, $16 \times 16$ $\rho$) | `CAUSAL_ENSEMBLE_DM_IO` | `PERSISTENT_COSINE_FEAT_AND_CE` (`wte_lr=0.006`) | `0.20M` | Baseline $16 \times 16$ Hilbert dimension with Rank-3 Causal Ensemble DM + Persistent Co-Distillation + Two-Speed LR. |
| **`G3_PKT32_12x2x32_DM`** | `SHAPE_12x2x32` ($d_{\text{pkt}}=32$, $32 \times 32$ $\rho$) | `CAUSAL_ENSEMBLE_DM_IO` | `PERSISTENT_COSINE_FEAT_AND_CE` (`wte_lr=0.006`) | `0.23M` | Do fewer, larger $32 \times 32$ complex outer products at $D_{\text{emb}}=768$ increase expressive capacity? |
| **`G4_PKT32_1x2x512_DM`** | `SHAPE_1x2x512_PKT32` ($d_{\text{pkt}}=32$, `1024`D) | `CAUSAL_ENSEMBLE_DM_IO` | `PERSISTENT_COSINE_FEAT_AND_CE` (`wte_lr=0.006`) | `0.38M` | Combines `(V, 2, 512)` overcomplete wavefunctions with $32 \times 32$ complex outer products and Rank-3 Causal DM. |

### 3.2 Phase 2 — Sub-`300M`-Token Full Recovery Demonstrations (`cuda:0` and `cuda:1`)
1. **`P1_SUB50M_RECOVERY_VS_50M_TEACHER` (`cuda:0`, vs. `50M` Teacher `4.2466` / `31.16%` Top-1)**:
   - Demonstrates exact reproduction and surpassing of the `50M`-token Classical Teacher in `< 2M` tokens (`> 25x` token compression) and tracks how far below `4.2466` the grafted model reaches by `16.4M` tokens (`2,000` steps).
2. **`P2_SUB300M_RECOVERY_VS_300M_TEACHER` (`cuda:1`, vs. `300M` Teacher `3.5845` / `37.31%` Top-1)**:
   - Runs the winning Plan 067 configuration with `PERSISTENT_COSINE_FEAT_AND_CE` and `wte_lr = 0.006` to measure the full sub-`300M` recovery trajectory toward `3.5845`.

---

## 4. Verification Gates (`tests/test_plan067_quantum_hilbert_granularity_and_persistent_recovery.py`)

1. **Gate `H1` — Exact Zero-Padding Outer-Product Aggregation Across $d_{\text{pkt}} \in \{8, 16, 32\}$**:
   - Verifies that `SHAPE_48x2x8` ($d_{\text{pkt}}=8$), `SHAPE_24x2x16` ($d_{\text{pkt}}=16$), `SHAPE_12x2x32` ($d_{\text{pkt}}=32$), and `SHAPE_1x2x512_PKT32` ($d_{\text{pkt}}=32$) all satisfy exact unit trace (`|Tr(rho) - 1| < 1e-4`), Hermiticity (`< 1e-5`), PSD eigenvalues (`min_eig >= -1e-5`), and strict causal masking (`< 1e-5`).
2. **Gate `H2` — Rank-3 Causal Ensemble Density Matrix (`CAUSAL_ENSEMBLE_DM_IO`) Physicality & Gradient Flow**:
   - Verifies that `CAUSAL_ENSEMBLE_DM_IO` produces a higher Step-0 quantum effective rank (`q_l0_eff_rank > 1.5`) than `WF_IN_WF_OUT` and that both `dm_lag_mix` and `dm_res_mix` receive finite non-zero gradients and are registered in `scalar_parameters()`.
3. **Gate `H3` — `PERSISTENT_COSINE_FEAT_AND_CE` Positive Floor Invariant**:
   - Verifies that for `t >= interface_warmup_steps`, $\lambda_{\text{align}}(t) = 0.25\,\lambda_{\text{align},0} > 0$ remains strictly active (`loss_total > loss_pure_ce`) and bounded.
4. **Gate `H4` — Monotonic Multi-Step Loss Descent with Two-Speed Riemannian LR**:
   - Verifies that over 10 optimization steps with `wte_lr = 0.006` and `stiefel_lr = 0.035`, total loss and pure token CE both decrease strictly while preserving `q_max_stiefel_err < 1e-5` and `q_wte_sphere_err < 1e-5`.
