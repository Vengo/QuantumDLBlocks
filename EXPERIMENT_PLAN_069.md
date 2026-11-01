# EXPERIMENT PLAN 069: Interior Quantum Purification, CPTP Kraus Channels, Outer-Product Coupling Modes, and Multi-Granularity Hilbert Shapes (`(V, 2, 256)`, `(V, 2, 512)`, `(V, 256, 2, 2)`) under Biorthogonal SVD-Stiefel Calibration (`Wave 52E`)

**Date:** 2026-11-01  
**Status:** Drafted — Pending Independent Context-Free Review (`egm-implementation-readiness`)  
**Codebase Target:** `/home/nixos/QuantumDLBlocks`  
**Preceding Plans:**
- [`EXPERIMENT_PLAN_065.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_065.md) (`Wave 52A`: 12-Arm Quantum Front-End Tournament under uncalibrated Kronecker-factored Stiefel projections)
- [`EXPERIMENT_PLAN_066.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_066.md) (`Wave 52B`: Teacher-Procrustes Sphere Alignment & Frozen `value_embeds[0]` Coupling)
- [`EXPERIMENT_PLAN_067.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_067.md) (`Wave 52C`: Outer-Product Hilbert Granularity $d_{\text{pkt}} \in \{8, 16, 32\}$ & Rank-3 Causal Ensemble DM I/O)
- [`EXPERIMENT_PLAN_068.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_068.md) (`Wave 52D`: Biorthogonal SVD-Stiefel Operator Geometry & Fubini-Study Projective Self-State Orthogonalization)

---

## 1. Executive Summary & Architectural Motivation

In `Plans 065–067`, we evaluated quantum wavefunction shapes (`(V, 2, 256)`, `(V, 2, 512)`, `(V, 256, 2, 2)`, `(V, 48, 2, 8)`, `(V, 24, 2, 16)`, `(V, 12, 2, 32)`), state I/O modes (`WF_IN_WF_OUT`, `CAUSAL_ENSEMBLE_DM_IO`, `WF_IN_DENSITY_OBS_OUT`), and interior attention couplings (`ATTENDED_RHO_ON_ATTENDED_V`, `ATTENDED_RHO_ON_LOCAL_V`, `KV_CROSS_OUTER_ON_Q`) under **rank-1 Kronecker-factored Stiefel projections** ($W_{\text{fact}} = M_{\text{mode}} \otimes_b W_{\text{pkt}}$), which captured only `14.4%–15.6%` of the teacher's Frobenius operator energy.

In `Plan 068` (`Wave 52D`), replacing those rank-1 Kronecker subblocks with **Biorthogonal SVD-Stiefel Wavepacket Projections** ($W_\alpha(x) = ((x V_\alpha)\odot \vec{\sigma}_\alpha)U_\alpha^\top$) and **Fubini-Study Projective Self-State Orthogonalization** ($(I - \tanh(\alpha_h)|\hat{\psi}_{v,h}\rangle\langle\hat{\psi}_{v,h}|)$) removed the projection bottleneck (`100%` Frobenius energy capture at $r = 768$) and improved Step-0 validation cross-entropy by nearly `2.0` nats (`5.65 -> 3.67`).

Now that the projection bottleneck is removed, we can rigorously answer the core open physics and architectural questions posed by the user **on a calibrated, apples-to-apples footing**:
1. **Interior Quantum Stat-Mech Purification & Open-System Channel Physics (`Axis N`)**:
   - Inside Quantum Layer 0, does open-system **Rank-$K$ CPTP Kraus Evolution** ($\mathcal{E}(\hat{\rho}_{\text{pure}}) = \sum_{k=1}^K E_k \hat{\rho}_{\text{pure}} E_k^\dagger$ with $\sum_{k=1}^K E_k^\dagger E_k = I_{d_h}$, $K=4$) outperform closed-system **Unitary $U(d_h)$ Evolution** ($U \hat{\rho}_{\text{pure}} U^\dagger$)?
   - How do the three non-linear quantum density-matrix purifiers in [`PurificationMode`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_statmech_channels.py#L63-L69) — **Quadratic McWeeny Purification** (`QUADRATIC_MCWEENY`, $\rho_0^2 / \operatorname{Tr}(\rho_0^2)$), **Cubic Fermi-Dirac Idempotent Purification** (`CUBIC_FERMI_DIRAC`, $(3\rho_0^2 - 2\rho_0^3)/\operatorname{Tr}(3\rho_0^2 - 2\rho_0^3)$), and **Learnable-Temperature Convex McWeeny Purification** (`LEARNABLE_TEMP_MCWEENY`, per-head learnable convex combination over $\{\rho_0, \rho_{\text{quad}}, \rho_{\text{FD}}\}$ parameterized by `purify_logits`) — compare when the boundary projections are SVD-Stiefel calibrated?
2. **Interior Outer-Product Coupling & State I/O under SVD-Stiefel Calibration (`Axis O`)**:
   - Once Fubini-Study self-state orthogonalization removes the trivial $j=t$ self-echo parallel to $|\hat{\psi}_{v,t}\rangle$, how do the interior outer-product attention mechanisms compare:
     - **Pure Wavefunction Outer-Product on Attended Ket (`WF_IN_WF_OUT` + `ATTENDED_RHO_ON_ATTENDED_V`)**
     - **Rank-3 Causal Ensemble Density Matrix on Attended Ket (`CAUSAL_ENSEMBLE_DM_IO` + `ATTENDED_RHO_ON_ATTENDED_V`)**
     - **Rank-3 Causal Ensemble Density Matrix on Local Ket (`CAUSAL_ENSEMBLE_DM_IO` + `ATTENDED_RHO_ON_LOCAL_V`)**
     - **Born-Rule Diagonal Observable Readout Modulation (`WF_IN_DENSITY_OBS_OUT` + `ATTENDED_RHO_ON_ATTENDED_V`)**
     - **Key-Value Cross Outer-Product Acting on Query Ket (`WF_IN_WF_OUT` + `KV_CROSS_OUTER_ON_Q`)**
3. **Full Wavefunction Shape Spectrum under SVD-Stiefel Calibration (`Axis P`)**:
   - How do the user's candidate embedding shapes compare when every shape is equipped with Biorthogonal SVD-Stiefel projections and Fubini-Study Self-State Orthogonalization:
     - **`(V, 2, 256)` (`SHAPE_1x2x256`, $D_{\text{emb}}=512$, compressed 256-complex wavefunction)**
     - **`(V, 2, 512)` (`SHAPE_1x2x512`, $D_{\text{emb}}=1024$, expanded 512-complex wavefunction with $d_{\text{pkt}}=16$ and $d_{\text{pkt}}=32$)**
     - **`(V, 256, 2, 2)` (`SHAPE_256x2x2`, $D_{\text{emb}}=1024$, `256` two-level Bloch spinor qubits with analytical $2\times 2$ Bloch McWeeny purification and $U(2)^{\otimes 256}$ evolution)**
     - **`(V, 48, 2, 8)` (`SHAPE_48x2x8`, $D_{\text{emb}}=768$, $d_{\text{pkt}}=8$)**
     - **`(V, 24, 2, 16)` (`SHAPE_24x2x16`, $D_{\text{emb}}=768$, $d_{\text{pkt}}=16$)**
     - **`(V, 12, 2, 32)` (`SHAPE_12x2x32`, $D_{\text{emb}}=768$, $d_{\text{pkt}}=32$)**

---

## 2. Mathematical Specification of New & Extended Components

### 2.1 Supporting `256`-Qubit Spinor `(V, 256, 2, 2)` Inside `SVDStiefelWavepacketStatMechBlock` (`Axis P`)
In [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py), [`QubitSpinorBlochStatMechBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py#L307-L435) models $M_{\text{qubit}} = 256$ two-level quantum spinors ($d_{\text{pkt}} = 4$ real $= \mathbb{C}^2$, grouped into $H = 8$ heads of $32$ qubits per head, `head_dim = 128`, $D_{\text{emb}} = 1024$).
Previously, `QubitSpinorBlochStatMechBlock` only supported Kronecker-factored projections (`w_q_pkt, m_q_mode`) and was bypassed by `SVDStiefelWavepacketStatMechBlock`.

To place `(V, 256, 2, 2)` on an exact apples-to-apples footing with `(V, 24, 2, 16)`, `(V, 2, 256)`, and `(V, 2, 512)`:
1. We add attributes `self.d_h = self.d_pkt` (`= 4`), `self.v_rank = self.qubits_per_head // 2` (`= 16`, so `2 * self.v_rank * self.d_h = 2 * 16 * 4 = 128 = sdpa_dim`), `self.state_io_mode = QuantumStateIOMode.WF_IN_WF_OUT`, and `self.channel_mode = QuantumChannelMode.UNITARY_U16` to [`QubitSpinorBlochStatMechBlock.__init__`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py#L310-L334).
2. When `SVDStiefelWavepacketStatMechBlock` initializes a wrapped `core_block`:
   - `SVDStiefelWavepacketStatMechBlock.__init__` guards `getattr(self.core_block, "state_io_mode", None)` and `getattr(self.core_block, "channel_mode", QuantumChannelMode.UNITARY_U16)`.
   - When `channel_mode != QuantumChannelMode.CPTP_KRAUS_RANK_K`, `SVDStiefelWavepacketStatMechBlock.__init__` inspects `u_count, u_dim = self.core_block.head_unitary_re.size(0), self.core_block.head_unitary_re.size(-1)` (`u_count = 256, u_dim = 2` for `QubitSpinorBlochStatMechBlock`; `u_count = H, u_dim = d_h` for `ConfigurableWavepacketStatMechBlock`), initializing `head_unitary_re` to $I_{u_{\text{dim}}}$ expanded to `(u_count, u_dim, u_dim)` and `head_unitary_im` to zero so Step-0 starts at the identity unitary channel.
3. We extract `_run_attention_and_evolution(self, h_norm, q_rot, q_h, k_h, v_wp) -> torch.Tensor` on `QubitSpinorBlochStatMechBlock`:
   - Given `q_h, k_h` of shape `(B, T, 8, 128)` and `v_wp` of shape `(B, T, 256, 4)`:
   - Normalizes each qubit spinor $|\hat{\psi}_{q}\rangle = (\alpha_r + i\alpha_i, \beta_r + i\beta_i)^\top \in \mathbb{C}^2$ (`vn` of shape `(B, T, 256, 4)`).
   - Computes the 3D Bloch vector $\vec{b} = (2(\alpha_r\beta_r + \alpha_i\beta_i),\, 2(\alpha_r\beta_i - \alpha_i\beta_r),\, |\alpha|^2 - |\beta|^2) \in \mathbb{R}^3$ and homogeneous 4-vector $(\vec{b}, 1) \in \mathbb{R}^4$.
   - Runs a single fused causal SDPA call across 2 chunks (`bloch_4` and `ket_4`) to obtain the ensemble-averaged Bloch vector $\vec{r} = \vec{b}_{\text{att}} / b_{\text{att}, 4}$ and attended spinor $|\bar{\psi}_v\rangle$.
   - Applies analytical $2\times 2$ Bloch McWeeny purification $\vec{r}_{\text{pure}} = \frac{2\vec{r}}{1 + \|\vec{r}\|^2}$, reconstructs the purified $2\times 2$ Hermitian density matrix $\hat{\rho}_{\text{pure}} = \frac{1}{2}(I_2 + r_x \sigma_x + r_y \sigma_y + r_z \sigma_z)$, evolves via $U_q \hat{\rho}_{\text{pure}} U_q^\dagger$ with $U_q \in U(2)$, and acts on $U_q |\bar{\psi}_v\rangle$:
     $$|\psi_{\text{out}, q}\rangle = \operatorname{RMSNorm}_4\!\left(\left(I_2 + \frac{1}{2} U_q \hat{\rho}_{\text{pure}, q} U_q^\dagger\right) U_q |\bar{\psi}_{v, q}\rangle\right)$$
4. In `FrozenTrunkQuantumFrontEndGPT.__init__`, when `self.graft_cfg.stiefel_proj_mode != QuantumStiefelProjMode.KRONEKER_FACTORED_STIEFEL`, **both** `QubitSpinorBlochStatMechBlock` (`SHAPE_256x2x2`) and `ConfigurableWavepacketStatMechBlock` (all other shapes) are wrapped inside `SVDStiefelWavepacketStatMechBlock` and calibrated via `calibrate_from_teacher_block0_`.

### 2.2 Open-System Rank-$K$ CPTP Kraus Channel Initialization under SVD-Stiefel Calibration (`Axis N`)
In `ConfigurableWavepacketStatMechBlock.__init__` and `SVDStiefelWavepacketStatMechBlock.__init__`:
- When `self.core_block.channel_mode == QuantumChannelMode.UNITARY_U16`, `head_unitary_re` (`(u_count, u_dim, u_dim)`) is initialized to $I_{u_{\text{dim}}}$ and `head_unitary_im` to $0$.
- When `self.core_block.channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K`, `ConfigurableWavepacketStatMechBlock.__init__` constructs `head_unitary_re, head_unitary_im = initialize_complex_kraus_stiefel_pair(self.num_heads, self.d_h, kraus_rank=DEFAULT_KRAUS_CHANNEL_RANK)` of shape `(H, K * d_h, d_h)` ($K = 4$) satisfying $\sum_{k=1}^K E_{h,k}^\dagger E_{h,k} = I_{d_h}$ (placing dominant weight in Kraus slot $k=0$ and orthogonal perturbations in slots $k=1..K-1$ so all $K=4$ Kraus operators receive non-zero gradients at Step 0), and `SVDStiefelWavepacketStatMechBlock.__init__` preserves this initialized Kraus Stiefel pair without overwriting it.

---

## 3. Experimental Design (`Wave 52E` — 12-Arm Calibrated Physics & Granularity Matrix)

All Phase-1 arms (`W1..W10`) run for **`500` steps (`4.10M` tokens = `1.37%` of `300M`)** on the frozen **`300M` Classical Teacher** (`val_ce = 3.5845`, `top1_acc = 37.31%`) with `batch_size = 8`, `seq_len = 1024`, `grad_accum = 1` (`8,192` tokens/step), `eval_every = 100` (`20` validation batches = `163,840` tokens), `stiefel_proj_mode = BIORTHOGONAL_SVD_STIEFEL_R768`, `self_ortho_mode = FUBINI_STUDY_SELF_ORTHO`, `wte_init_mode = TEACHER_PROCRUSTES_SPHERE`, `ve_coupling_mode = FROZEN_TEACHER_VE0_COUPLED`, `recovery_schedule_mode = PERSISTENT_COSINE_FEAT_AND_CE`, `interface_warmup_steps = 300`, `interface_align_weight = 0.5`, `wte_lr = 0.004`, and `stiefel_lr = 0.012`.

| Arm ID | Axis Tested | `wf_shape` | `state_io_mode` | `attn_interior_mode` | `purification_mode` | `channel_mode` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`W1_MCWEENY_U16_CAUSAL_ATT_V`** | Baseline (`N1/O2/P5`) | `SHAPE_24x2x16` | `CAUSAL_ENSEMBLE_DM_IO` | `ATTENDED_RHO_ON_ATTENDED_V` | `QUADRATIC_MCWEENY` | `UNITARY_U16` |
| **`W2_CUBIC_FD_U16_CAUSAL_ATT_V`** | Axis N (`Cubic FD`) | `SHAPE_24x2x16` | `CAUSAL_ENSEMBLE_DM_IO` | `ATTENDED_RHO_ON_ATTENDED_V` | `CUBIC_FERMI_DIRAC` | `UNITARY_U16` |
| **`W3_LEARN_TEMP_U16_CAUSAL_ATT_V`** | Axis N (`Learnable Temp`) | `SHAPE_24x2x16` | `CAUSAL_ENSEMBLE_DM_IO` | `ATTENDED_RHO_ON_ATTENDED_V` | `LEARNABLE_TEMP_MCWEENY` | `UNITARY_U16` |
| **`W4_MCWEENY_KRAUS_CAUSAL_ATT_V`** | Axis N (`Kraus K=4`) | `SHAPE_24x2x16` | `CAUSAL_ENSEMBLE_DM_IO` | `ATTENDED_RHO_ON_ATTENDED_V` | `QUADRATIC_MCWEENY` | `CPTP_KRAUS_RANK_K` |
| **`W5_PURE_WF_IO_ATT_V`** | Axis O (`WF_IN_WF_OUT`) | `SHAPE_24x2x16` | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `QUADRATIC_MCWEENY` | `UNITARY_U16` |
| **`W6_HYBRID_BORN_OBS_ATT_V`** | Axis O (`BORN_OBS_OUT`) | `SHAPE_24x2x16` | `WF_IN_DENSITY_OBS_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `QUADRATIC_MCWEENY` | `UNITARY_U16` |
| **`W7_KV_CROSS_OUTER_ON_Q`** | Axis O (`KV_CROSS`) | `SHAPE_24x2x16` | `WF_IN_WF_OUT` | `KV_CROSS_OUTER_ON_Q` | `QUADRATIC_MCWEENY` | `UNITARY_U16` |
| **`W8_SHAPE_1X2X256_SVD`** | Axis P (`(V, 2, 256)`) | `SHAPE_1x2x256` | `CAUSAL_ENSEMBLE_DM_IO` | `ATTENDED_RHO_ON_ATTENDED_V` | `QUADRATIC_MCWEENY` | `UNITARY_U16` |
| **`W9_SHAPE_48X2X8_SVD`** | Axis P (`(V, 48, 2, 8)`) | `SHAPE_48x2x8` | `CAUSAL_ENSEMBLE_DM_IO` | `ATTENDED_RHO_ON_ATTENDED_V` | `QUADRATIC_MCWEENY` | `UNITARY_U16` |
| **`W10_SHAPE_256X2X2_QUBIT_SVD`** | Axis P (`(V, 256, 2, 2)`) | `SHAPE_256x2x2` | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `QUADRATIC_MCWEENY` | `UNITARY_U16` |

### Phase 2 — Extended Sub-`300M` Token Champion Run (`1,200` Steps = `9.83M` Tokens = `3.28%` of `300M`)
- **`W11_CHAMPION_300M_RECOVERY_1200STEPS`**: Runs the winning interior physics + wavefunction shape combination on the `300M` Teacher (`3.5845`) for `1,200` steps (`9.83M` tokens) to test whether it matches or beats `3.5845` within `3.28%` of the `300M`-token budget.
- **`W12_CHAMPION_50M_EXCEED_800STEPS`**: Runs the winning configuration on the `50M` Teacher (`4.2467`) for `800` steps (`6.55M` tokens) to quantify how far below `4.2467` the quantum front-end reaches.

---

## 4. Files to Create or Modify

### 4.1 Modify [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py) and [`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py)
1. In [`QubitSpinorBlochStatMechBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py#L307-L435):
   - Set `self.d_h = self.d_pkt` (`= 4`), `self.v_rank = self.qubits_per_head // 2` (`= 16`), `self.state_io_mode = QuantumStateIOMode.WF_IN_WF_OUT`, and `self.channel_mode = QuantumChannelMode.UNITARY_U16` in `__init__`.
   - Extract `_run_attention_and_evolution(self, h_norm: torch.Tensor, q_rot: torch.Tensor, q_h: torch.Tensor, k_h: torch.Tensor, v_wp: torch.Tensor) -> torch.Tensor` so both `QubitSpinorBlochStatMechBlock.forward` and `SVDStiefelWavepacketStatMechBlock.forward` call it directly.
2. In [`SVDStiefelWavepacketStatMechBlock.__init__`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py#L259-L312):
   - Support both `ConfigurableWavepacketStatMechBlock` and `QubitSpinorBlochStatMechBlock` as `core_block`:
     - Guard `getattr(self.core_block, "state_io_mode", None)` and `getattr(self.core_block, "channel_mode", QuantumChannelMode.UNITARY_U16)`.
     - When `channel_mode != QuantumChannelMode.CPTP_KRAUS_RANK_K`, read `u_count, u_dim = self.core_block.head_unitary_re.size(0), self.core_block.head_unitary_re.size(-1)` and initialize `head_unitary_re` to $I_{u_{\text{dim}}}$ expanded to `(u_count, u_dim, u_dim)` and `head_unitary_im` to zero.
     - In `restrict_scalars_()`, guard `dm_res_mix` and `dm_lag_mix` clamping with `if hasattr(self.core_block, "dm_res_mix"):`.
3. In [`FrozenTrunkQuantumFrontEndGPT.__init__`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py#L771-L798):
   - When `is_svd_stiefel` is `True`, wrap `blk0` (whether `QubitSpinorBlochStatMechBlock` or `ConfigurableWavepacketStatMechBlock`) inside `SVDStiefelWavepacketStatMechBlock` and calibrate via `calibrate_from_teacher_block0_`.
4. Keep both files strictly `<= 1,000` SLOC and all functions `<= 8` parameters (`docs/SCRIPT_BEST_PRACTICES.md`).

### 4.2 Create [`examples/79_verify_plan069_interior_physics_and_shapes_svd_graft.py`](file:///home/nixos/QuantumDLBlocks/examples/79_verify_plan069_interior_physics_and_shapes_svd_graft.py)
- CLI runner supporting `--arm` / `--arms` (`W1..W12`), `--teacher-ckpt`, `--use-50m-teacher`, `--steps`, `--eval-every`, `--device`, and `--out-json`.

### 4.3 Create [`tests/test_plan069_interior_physics_and_shapes_svd_graft.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan069_interior_physics_and_shapes_svd_graft.py)
- Implements verification gates `H1..H4`.

---

## 5. Verification Gates (`H1..H4`)

1. **Gate H1 — Calibrated `256`-Qubit Spinor `(V, 256, 2, 2)` & Compressed `(V, 2, 256)` SVD-Stiefel Forward/Backward/Step Verification**:
   - Instantiate `FrozenTrunkQuantumFrontEndGPT` with `stiefel_proj_mode = BIORTHOGONAL_SVD_STIEFEL_R768` and `self_ortho_mode = FUBINI_STUDY_SELF_ORTHO` for `SHAPE_256x2x2` (`256` qubits, $D_{\text{emb}}=1024$, $r_{\text{eff}}=768$) and `SHAPE_1x2x256` ($D_{\text{emb}}=512$, $r_{\text{eff}}=512$).
   - Verify 2 forward/backward/step iterations preserve exact frozen trunk invariance (`max_abs_trunk_diff == 0.0`), `0 < q_l0_purity <= 1.0`, `q_l0_eff_rank >= 1.0`, `q_max_stiefel_err < 1e-4`, and `q_wte_sphere_err < 1e-4`.
2. **Gate H2 — Purification Modes (`QUADRATIC_MCWEENY`, `CUBIC_FERMI_DIRAC`, `LEARNABLE_TEMP_MCWEENY`) & Open-System `CPTP_KRAUS_RANK_K` Physicality under SVD-Stiefel**:
   - Instantiate `FrozenTrunkQuantumFrontEndGPT` across all 3 `PurificationMode` values (`QUADRATIC_MCWEENY`, `CUBIC_FERMI_DIRAC`, `LEARNABLE_TEMP_MCWEENY`) and both `QuantumChannelMode` values (`UNITARY_U16`, `CPTP_KRAUS_RANK_K`) with `BIORTHOGONAL_SVD_STIEFEL_R768`.
   - Execute a forward/backward/step pass and verify finite loss, finite non-zero gradients on `head_unitary_re` and `head_unitary_im` (and on `purify_logits` when `LEARNABLE_TEMP_MCWEENY` is active), and complex unitary/Kraus manifold orthogonality error $\|E^\dagger E - I\|_F / \sqrt{d_h} < 10^{-4}$.
3. **Gate H3 — Interior Coupling Modes (`ATTENDED_RHO_ON_ATTENDED_V`, `ATTENDED_RHO_ON_LOCAL_V`, `KV_CROSS_OUTER_ON_Q`, `WF_IN_DENSITY_OBS_OUT`) Step-0 & Gradient Gate**:
   - Verify all 4 interior/state-I/O configurations under `BIORTHOGONAL_SVD_STIEFEL_R768` produce finite outputs, finite gradients on all active quantum parameters (including `diag_obs_gamma` when `WF_IN_DENSITY_OBS_OUT` is active), and zero dead trainable parameters.
4. **Gate H4 — Best-Practices & SLOC Gate (`docs/SCRIPT_BEST_PRACTICES.md`)**:
   - AST verification that [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py), [`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py), [`examples/79_verify_plan069_interior_physics_and_shapes_svd_graft.py`](file:///home/nixos/QuantumDLBlocks/examples/79_verify_plan069_interior_physics_and_shapes_svd_graft.py), and [`tests/test_plan069_interior_physics_and_shapes_svd_graft.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan069_interior_physics_and_shapes_svd_graft.py) each have `<= 1,000` SLOC, `<= 8` arguments per function, and zero semicolon statement packing.
