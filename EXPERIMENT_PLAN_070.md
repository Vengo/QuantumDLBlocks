# EXPERIMENT PLAN 070 — Wave 52F: Sub-Parity & 1:1 Parameter-Matched SVD-Stiefel Ranks ($r \in \{64, 128, 192, 256, 384\}$) on $(V, 12, 2, 32)$, Interior Density-Operator Commutator & Local-Value Couplings, and Multi-Layer (`Emb + L0 + L1`) Calibrated Quantum Front-End Grafting

**Date**: 2026-11-01
**Author**: Antigravity (Autonomous Overnight Session — Wave 52F)
**Predecessor**: [`EXPERIMENT_PLAN_069.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_069.md) (Wave 52E — 10-Arm Calibrated Interior Physics & Wavefunction Shapes Tournament)
**Status**: Approved (`0 Blockers, 0 Majors, 0 Nits`) — Ready for Implementation & Verification

---

## 1. Executive Summary & Motivation

In **Wave 52D (`EXPERIMENT_PLAN_068.md`)**, we proved that replacing the initial token embeddings (`wte`, `smear_gate`) and Layer-0 attention (`blocks[0].attn`) of the frozen pretrained `CompiledLeaderboardClassicalGPT` teacher checkpoints (`50M` and `300M` tokens) with a calibrated **Biorthogonal SVD-Stiefel Quantum Wavepacket StatMech** front-end strictly **exceeds both the `50M` and `300M` Classical Teachers in `< 6.6M` tokens (`2.18%` of `300M`)**:
- **`E3_REPRODUCE_300M_TEACHER_12X2X32`** (`(V, 12, 2, 32)` with $d_{\text{pkt}}=32$, `r=768`, `800` steps = `6.55M` tokens): **`val_ce = 3.5811`** (beating the `300M` Classical Teacher's `3.5845` by **`-0.0034` nats**), **`top1_acc = 37.44%`** (`+0.13%`), **`top5_acc = 58.34%`** (`+0.08%`).
- **`E1_EXCEED_50M_TEACHER_24X2X16`** (`(V, 24, 2, 16)` with $d_{\text{pkt}}=16$, `r=768`, `800` steps = `6.55M` tokens): **`val_ce = 4.2246`** (beating the `50M` Classical Teacher's `4.2467` by **`-0.0221` nats**), **`top1_acc = 31.68%`** (`+0.14%`), **`top5_acc = 51.12%`** (`+0.17%`).

While **Wave 52E (`EXPERIMENT_PLAN_069.md`)** is currently running on `cuda:0` and `cuda:1` to benchmark purification laws, channel dynamics, state I/O modes, and Hilbert shapes (`(V, 2, 256)`, `(V, 48, 2, 8)`, `(V, 256, 2, 2)`) at `r=768`, three fundamental architectural questions from the user's research brief remain unanswered:

1. **Axis N — Sub-Parity & Exact 1:1 Parameter-Matched SVD-Stiefel Rank Scaling on $(V, 12, 2, 32)$ ($d_{\text{pkt}}=32$)**:
   - In `Plan 068`, our rank sweep (`D1..D4`, $r \in \{128, 256, 512, 768\}$) was conducted on $(V, 24, 2, 16)$ ($d_{\text{pkt}}=16$), whereas $(V, 12, 2, 32)$ ($d_{\text{pkt}}=32$) was only tested at $r = 768$ (`4.72M` Layer-0 quantum operator parameters = $2.0\times$ the Classical Teacher's `2.36M` Layer-0 Attention parameters, since $4 \times 768^2 = 2,359,296$).
   - Each Biorthogonal SVD-Stiefel projection $W_\alpha(x) = ((x V_\alpha) \odot \vec{\sigma}_\alpha) U_\alpha^\top$ ($\alpha \in \{q, k, v, \text{obs}\}$ on $D_{\text{in}} = D_{\text{out}} = 768$) has $2 \times 768 \times r + r = 1,537 r$ parameters ($6,148 r$ across all 4 projections).
   - Therefore:
     - $r = 64$: **`0.39M`** Layer-0 operator parameters (**$6.0\times$ fewer** than Classical Layer-0 Attention!)
     - $r = 128$: **`0.79M`** Layer-0 operator parameters (**$3.0\times$ fewer** than Classical Layer-0 Attention!)
     - $r = 192$: **`1.18M`** Layer-0 operator parameters (**$2.0\times$ fewer** than Classical Layer-0 Attention!)
     - $r = 256$: **`1.58M`** Layer-0 operator parameters (**$1.5\times$ fewer** than Classical Layer-0 Attention!)
     - $r = 384$: **`2.36M`** Layer-0 operator parameters (**exact $1.0\times$ 1:1 parameter parity** with Classical Layer-0 Attention: $4 \times 2 \times 768 \times 384 = 2,359,296 = 4 \times 768^2$!)
   - **Scientific Question**: On the winning $(V, 12, 2, 32)$ ($d_{\text{pkt}}=32$) geometry, can our quantum front-end **strictly exceed the `300M` Pretrained Classical Teacher (`val_ce < 3.5845`) at exact 1:1 parameter parity ($r=384$, `2.36M` params) and even at sub-parity ($r \in \{64, 128, 192, 256\}$, $1.5\times$ to $6.0\times$ fewer Layer-0 parameters than Classical Attention)**?

2. **Axis O — Interior Outer-Product Value Coupling & Liouville-von Neumann Commutator Dynamics (`QuantumAttnInteriorMode`)**:
   - The user specifically asked: *"I think I want the attention mechanism to form outer products of input wave functions, and then use those outer products to combine the input's projection into V space... maybe I'm making wrong decisions about interior operations of the attention block. There are lots of small decisions there, and I'd like them listed and tested."*
   - Once causal attention forms the purified, channel-evolved context density operator $\tilde{\rho}_{\text{evol}, t} \in \mathbb{C}^{H \times d_h \times d_h}$ and the normalized value wavepacket, how should outer products combine projections in $V$ space? At exact 1:1 parameter parity ($r=384$, $(V, 12, 2, 32)$), we compare four distinct physical laws:
     1. `ATTENDED_RHO_ON_ATTENDED_V`: Context density operator $\tilde{\rho}_{\text{evol}, t}$ modulates causally attended value wavepacket $\bar{V}_{u, t}$.
     2. `ATTENDED_RHO_ON_LOCAL_V`: Context density operator $\tilde{\rho}_{\text{evol}, t}$ acts on the current token's local value wavepacket $V_{\text{local}, u, t}$ added to $\bar{V}_{u, t}$.
     3. `SYMMETRIZED_LOCAL_CONTEXT_RHO_V`: Coherent convex combination $\tilde{\rho}_{\text{sym}, t} = \cos^2(\theta_h)\tilde{\rho}_{\text{evol}, t} + \sin^2(\theta_h)\rho_{\text{local}, t}$ between the token's own local outer-product $\rho_{\text{local}, t}$ and the context outer-product $\tilde{\rho}_{\text{evol}, t}$ acting on $\bar{V}_{u, t}$.
     4. `LIOUVILLE_COMMUTATOR_RHO_V`: First-order quantum Liouville-von Neumann commutator evolution $\tilde{\rho}_{\text{liouville}, t} = \tilde{\rho}_{\text{evol}, t} + \tanh(\beta_h)\left(-i [\rho_{\text{local}, t}, \tilde{\rho}_{\text{evol}, t}]\right)$ acting on $\bar{V}_{u, t}$, capturing non-commutative phase interference between the token's local density operator $\rho_{\text{local}, t}$ and the causal context bath $\tilde{\rho}_{\text{evol}, t}$.

3. **Axis P — Multi-Layer Calibrated Quantum Front-End Graft (`Emb + Layer 0 + Layer 1` vs. `Emb + Layer 0`)**:
   - In `Plan 065` (`T11_SCOPE_EMB_L0_L1`), replacing both Layer 0 and Layer 1 degraded validation loss (`3.8891` CE) solely because `T11` used uncalibrated `0.20M`-parameter Kronecker blocks without teacher singular-value calibration.
   - What happens when we replace **both Layer-0 Attention (`blocks[0].attn`) AND Layer-1 Attention (`blocks[1].attn`)** (along with `wte` and `smear_gate`, keeping frozen `MLP0`, `MLP1`, and `blocks[2..11]`) using **two calibrated `SVDStiefelWavepacketStatMechBlock` modules** (`quantum_block0` calibrated from `teacher.blocks[0]` and `quantum_block1` calibrated from `teacher.blocks[1]`, with `value_embeds[1]` coupled into `quantum_block1` via `blocks[1].ve_gate`)?

---

## 2. Mathematical & Algorithmic Specifications

### 2.1 Axis N: Sub-Parity & 1:1 Parameter-Matched SVD-Stiefel Ranks ($r \in \{64, 128, 192, 256, 384\}$)

We extend [`QuantumStiefelProjMode`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py#L44-L52) and [`resolve_svd_stiefel_rank`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py#L140-L154) with three additional rank modes:
- `QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R64 = "biorthogonal_svd_stiefel_r64"` ($r = 64$, `0.39M` L0 operator params = $6.0\times$ smaller than Classical Attn0)
- `QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R192 = "biorthogonal_svd_stiefel_r192"` ($r = 192$, `1.18M` L0 operator params = $2.0\times$ smaller than Classical Attn0)
- `QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384 = "biorthogonal_svd_stiefel_r384"` ($r = 384$, `2.36M` L0 operator params = **exact $1.0\times$ 1:1 parameter parity** with Classical Attn0)

For any rank $r \in \{64, 128, 192, 256, 384, 512, 768\}$, each projection $\alpha \in \{q, k, v, \text{obs}\}$ in [`BiorthogonalSVDStiefelProjection`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py#L157-L236) computes:
$$W_\alpha(x) = \left((x V_\alpha) \odot \vec{\sigma}_\alpha\right) U_\alpha^\top, \qquad U_\alpha \in \mathrm{St}(D_{\text{out}}, r_{\text{eff}}), \quad V_\alpha \in \mathrm{St}(D_{\text{in}}, r_{\text{eff}}), \quad \vec{\sigma}_\alpha \in \mathbb{R}^{r_{\text{eff}}}$$
where $r_{\text{eff}} = \min(r, D_{\text{in}}, D_{\text{out}})$, initialized from the top-$r_{\text{eff}}$ truncated SVD of the corresponding classical teacher matrix ($W_Q, W_K, W_V, W_O$).

---

### 2.2 Axis O: Interior Outer-Product Value Coupling & Liouville-von Neumann Commutator Dynamics

We extend [`QuantumAttnInteriorMode`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py#L94-L100) with two new physical outer-product coupling laws:
- `QuantumAttnInteriorMode.SYMMETRIZED_LOCAL_CONTEXT_RHO_V = "symmetrized_local_context_rho_v"`
- `QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V = "liouville_commutator_rho_v"`

Let $V_{\text{local}, t} \in \mathbb{R}^{B \times T \times M_{\text{pkt}} \times d_h}$ be the current token's value wavepacket (`v_wp`).
1. **Normalized Local Ket in Evolved Unitary Basis**:
   Using `self._normalize_mode_kets(v_wp)`, we obtain $(v_{\text{loc}, R}, v_{\text{loc}, I}) \in \mathbb{R}^{B \times H \times T \times R \times d_h}$ satisfying $\sum_{r=1}^R \|v_{\text{loc}, r}\|_2^2 = 1$. Rotating by the head unitary $U_0^\dagger = (U_{0, R}^\top - i U_{0, I}^\top)$:
   $$L_R = v_{\text{loc}, R} U_{0, R}^\top - v_{\text{loc}, I} U_{0, I}^\top, \qquad L_I = v_{\text{loc}, R} U_{0, I}^\top + v_{\text{loc}, I} U_{0, R}^\top \quad \in \mathbb{R}^{B \times H \times T \times R \times d_h}$$
2. **Unit-Trace Local Self-Density Matrix $\rho_{\text{local}} = A_R + i A_I \in \mathbb{C}^{B \times H \times T \times d_h \times d_h}$**:
   $$A_R = L_R^\top L_R + L_I^\top L_I, \qquad A_I = L_I^\top L_R - L_R^\top L_I$$
   Notice $A_R^\top = A_R$, $\operatorname{Tr}(A_R) = 1$ (PSD), and $A_I^\top = -A_I$, so $A = A_R + i A_I$ is a strictly Hermitian, unit-trace PSD density matrix matching the unit-trace evolved context density operator $B = \tilde{\rho}_{\text{evol}} = B_R + i B_I \in \mathbb{C}^{B \times H \times T \times d_h \times d_h}$ ($\operatorname{Tr}(B_R) = 1$).
3. **Mode 3 (`SYMMETRIZED_LOCAL_CONTEXT_RHO_V`)**:
   With learnable per-head logit $\theta \in \mathbb{R}^H$ (`self.local_rho_mix_logit`, `requires_grad=(self.attn_interior_mode == QuantumAttnInteriorMode.SYMMETRIZED_LOCAL_CONTEXT_RHO_V)`, initialized to $\theta_0 = -1.10$ so $\sigma(\theta_0) \approx 0.25$, clamped to $[-4.0, 4.0]$):
   $$\alpha_h = \sigma(\theta_h) \in (0, 1), \qquad \tilde{\rho}_{\text{sym}, R} = (1 - \alpha_h) B_R + \alpha_h A_R, \qquad \tilde{\rho}_{\text{sym}, I} = (1 - \alpha_h) B_I + \alpha_h A_I$$
   Because $\operatorname{Tr}(A_R) = \operatorname{Tr}(B_R) = 1$, $\tilde{\rho}_{\text{sym}}$ is an exact trace-1 convex combination of two PSD density operators. Acting on the rotated attended ket $(V_{u, R}, V_{u, I})$:
   $$\text{wp}_{u, R} = V_{u, R} + \frac{1}{2}\left(V_{u, R} \tilde{\rho}_{\text{sym}, R} + V_{u, I} \tilde{\rho}_{\text{sym}, I}\right), \qquad \text{wp}_{u, I} = V_{u, I} + \frac{1}{2}\left(V_{u, I} \tilde{\rho}_{\text{sym}, R} - V_{u, R} \tilde{\rho}_{\text{sym}, I}\right)$$
4. **Mode 4 (`LIOUVILLE_COMMUTATOR_RHO_V`)**:
   Since both $A = \rho_{\text{local}}$ and $B = \tilde{\rho}_{\text{evol}}$ are Hermitian ($A^\dagger = A, B^\dagger = B$), their product $P = A B = P_R + i P_I$ with
   $$P_R = A_R B_R - A_I B_I, \qquad P_I = A_R B_I + A_I B_R$$
   satisfies $B A = (A B)^\dagger = P^\dagger = P_R^\top - i P_I^\top$. Furthermore, because $\operatorname{Tr}(\text{sym}\cdot\text{skew}) \equiv 0$, $\operatorname{Tr}(P_I) \equiv 0$. Therefore, the Hermitian traceless Liouville-von Neumann commutator $C = -i [A, B] = -i (P - P^\dagger) = C_R + i C_I$ is given in exact closed form by a **single** complex matrix multiplication:
   $$C_R = P_I + P_I^\top \quad (\text{symmetric, }\operatorname{Tr}(C_R) \equiv 0), \qquad C_I = P_R^\top - P_R \quad (\text{skew-symmetric})$$
   With learnable per-head coupling $\beta \in \mathbb{R}^H$ (`self.liouville_beta`, `requires_grad=(self.attn_interior_mode == QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V)`, initialized to $0.25$, clamped to $[-2.0, 2.0]$):
   $$\tilde{\rho}_{\text{liouville}, R} = B_R + \tanh(\beta_h)\, C_R, \qquad \tilde{\rho}_{\text{liouville}, I} = B_I + \tanh(\beta_h)\, C_I$$
   which preserves exact unit trace $\operatorname{Tr}(\tilde{\rho}_{\text{liouville}, R}) = 1$ and acts on $(V_{u, R}, V_{u, I})$ via $\text{wp}_u = V_u + \frac{1}{2} V_u \tilde{\rho}_{\text{liouville}}^\top$.
5. **Parameter Registration & Clamping**:
   Both `self.local_rho_mix_logit` and `self.liouville_beta` on `ConfigurableWavepacketStatMechBlock` are returned by `ConfigurableWavepacketStatMechBlock.scalar_parameters()` when `requires_grad` is `True`, and are clamped in both `SVDStiefelWavepacketStatMechBlock.restrict_scalars_()` and `FrozenTrunkQuantumFrontEndGPT.restrict_quantum_scalars_()` (`local_rho_mix_logit.clamp_(-4.0, 4.0)`, `liouville_beta.clamp_(-2.0, 2.0)`).

We implement the helper function `apply_interior_rho_v_coupling(...)` in [`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py) and call it from `ConfigurableWavepacketStatMechBlock._run_attention_and_evolution`.

---

### 2.3 Axis P: Multi-Layer Calibrated SVD-Stiefel Quantum Front-End (`Emb + Layer 0 + Layer 1`)

We add `QuantumReplacementScope.EMB_PLUS_L0_L1 = "emb_plus_l0_l1"` to [`QuantumReplacementScope`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py#L106-L112) and generalize [`SVDStiefelWavepacketStatMechBlock.forward`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py#L386-L439) to accept an optional `x_0_768: torch.Tensor | None = None`:
- When `x_0_768 is None` (Layer 0): `x_in_768 = (l0 + l1) * x_768`.
- When `x_0_768 is not None` (Layer $l \ge 1$): `x_in_768 = l0 * x_768 + l1 * x_0_768`, matching `CompiledLeaderboardBlock.forward` (`x = lam_x * x + lam_x0 * x_0`)!

When `graft_cfg.replacement_scope == QuantumReplacementScope.EMB_PLUS_L0_L1` and `is_svd_stiefel` is `True`:
1. `FrozenTrunkQuantumFrontEndGPT.__init__` instantiates `self.quantum_block1: SVDStiefelWavepacketStatMechBlock | None` alongside `self.quantum_block0`, calibrated from `self.classical.blocks[1]` via `svd_blk1.calibrate_from_teacher_block0_(self.classical.blocks[1], self.w_in_bridge)` (keeping `blocks[0].fc1/fc2` and `blocks[1].fc1/fc2` active as the frozen MLPs of Layers 0 and 1).
2. In `encode_quantum_layer0(idx)`, the frozen `MLP0` (`blk0.fc2(F.relu(blk0.fc1(h_mlp)).square())`) is executed whenever `self.graft_cfg.replacement_scope in (QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0, QuantumReplacementScope.EMB_PLUS_L0_L1)` so Layer 0 always emits the complete post-MLP0 hidden state `x_1` under `EMB_PLUS_L0_L1`.
3. On `FrozenTrunkQuantumFrontEndGPT`, all four parameter-management methods — `quantum_real_stiefel_parameters()`, `quantum_complex_unitary_pairs()`, `quantum_scalar_parameters()`, and `restrict_quantum_scalars_()` — iterate over `(self.quantum_block0, self.quantum_block1)` whenever `blk is not None`, ensuring Layer-1 quantum Stiefel, Unitary, and Scalar parameters are stepped and clamped alongside Layer 0.
4. During `forward_features(idx)`:
   - Layer 0 runs via `self.encode_quantum_layer0(idx)` returning `(x_0, x_1, v_0)`, and `encoder_states.append(x_1)`.
   - Layer 1 (`self.quantum_block1` when `EMB_PLUS_L0_L1` is active) is executed via helper `encode_svd_stiefel_layer1(...)` in `quantum_svd_stiefel_graft.py`, which computes:
     - Gated value embedding `ve1_wp_gated` from `self.classical.value_embeds[1](idx)` and `self.classical.blocks[1].ve_gate` (or `v_0` mapped to quantum packets if no `value_embeds[1]` is present).
     - `x_mid_1, _, _ = self.quantum_block1(x_1, cos_4d, sin_4d, q_scale, ve_wp_gated=ve1_wp_gated, x_0_768=x_0)`.
     - Frozen `MLP1`: `h_mlp1 = F.rms_norm(x_mid_1, (self.embed_dim,), eps=self.cfg.eps)`, `x_2 = x_mid_1 + blk1.fc2(F.relu(blk1.fc1(h_mlp1)).square())`.
     - Appends `x_2` to `encoder_states` and runs frozen classical `blocks[2..11]`.
5. `compute_teacher_layer1_target(idx, x_0_c, x_1_c, v_0_c) -> torch.Tensor` computes the detached teacher Layer-1 post-MLP target `x_2_c` from `self.classical.blocks[1]` so Phase-1 interface warmup aligns both `x_1` and `x_2` when `self.quantum_block1 is not None`.

---

## 3. Experimental Design (`11` Arms Across Phase 1 & Phase 2)

### Phase 1: 10-Arm Sub-Parity Rank Ladder, Interior Commutator Physics & Multi-Layer Scope Tournament (`500` Steps = `4.10M` Tokens = `1.37%` of `300M` Budget on `300M` Teacher)

All Phase-1 arms attach to the frozen `300M` Classical Teacher (`ckpt_step01145_0300M_m0300M.pt`, reference `val_ce = 3.5845`, `top1_acc = 37.31%`, `top5_acc = 58.26%`), use `wte_init_mode = TEACHER_PROCRUSTES_SPHERE`, `self_ortho_mode = FUBINI_STUDY_SELF_ORTHO`, `state_io_mode = CAUSAL_ENSEMBLE_DM_IO`, `purification_mode = QUADRATIC_MCWEENY`, `channel_mode = UNITARY_U16`, and train for `500` steps (`100` Phase-1 cosine + MSE interface warmup steps + `400` pure hard-CE steps, batch `8 x 1024 = 8,192` tokens/step):

| Arm ID | Axis Tested | Shape | SVD-Stiefel Rank $r$ | L0 Quantum Params (vs. `2.36M` Teacher Attn0) | Interior Mode | Replacement Scope |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`F1_R64_SUBPARITY_6X_12X2X32`** | Axis N ($6\times$ Smaller) | `(V, 12, 2, 32)` | $r = 64$ | **`0.39M`** ($0.17\times$) | `ATTENDED_RHO_ON_ATTENDED_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F2_R128_SUBPARITY_3X_12X2X32`** | Axis N ($3\times$ Smaller) | `(V, 12, 2, 32)` | $r = 128$ | **`0.79M`** ($0.33\times$) | `ATTENDED_RHO_ON_ATTENDED_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F3_R192_SUBPARITY_2X_12X2X32`** | Axis N ($2\times$ Smaller) | `(V, 12, 2, 32)` | $r = 192$ | **`1.18M`** ($0.50\times$) | `ATTENDED_RHO_ON_ATTENDED_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F4_R256_SUBPARITY_1P5X_12X2X32`** | Axis N ($1.5\times$ Smaller) | `(V, 12, 2, 32)` | $r = 256$ | **`1.58M`** ($0.67\times$) | `ATTENDED_RHO_ON_ATTENDED_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F5_R384_PARITY_1X_12X2X32`** | Axis N (**Exact $1\times$ Parity**) | `(V, 12, 2, 32)` | $r = 384$ | **`2.36M`** ($1.00\times$) | `ATTENDED_RHO_ON_ATTENDED_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F6_R384_PARITY_1X_24X2X16`** | Axis N ($1\times$ Parity, $d_{\text{pkt}}=16$) | `(V, 24, 2, 16)` | $r = 384$ | **`2.36M`** ($1.00\times$) | `ATTENDED_RHO_ON_ATTENDED_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F7_R384_LOCAL_V_12X2X32`** | Axis O (Local $V$ Coupling) | `(V, 12, 2, 32)` | $r = 384$ | **`2.36M`** ($1.00\times$) | `ATTENDED_RHO_ON_LOCAL_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F8_R384_SYM_LOCAL_CONTEXT_12X2X32`** | Axis O (Symmetrized $\rho_{\text{loc}} + \rho_{\text{ctx}}$) | `(V, 12, 2, 32)` | $r = 384$ | **`2.36M`** ($1.00\times$) | `SYMMETRIZED_LOCAL_CONTEXT_RHO_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F9_R384_LIOUVILLE_COMM_12X2X32`** | Axis O (Liouville $-i[\rho_{\text{loc}}, \rho_{\text{ctx}}]$) | `(V, 12, 2, 32)` | $r = 384$ | **`2.36M`** ($1.00\times$) | `LIOUVILLE_COMMUTATOR_RHO_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` |
| **`F10_MULTILAYER_L0_L1_R384_12X2X32`** | Axis P (`Emb + L0 + L1` Calibrated) | `(V, 12, 2, 32)` | $r = 384$ | **`2.36M`/layer** ($1.00\times$) | `ATTENDED_RHO_ON_ATTENDED_V` | `EMB_PLUS_L0_L1` |

### Phase 2: Extended Sub-Parity & Multi-Layer Confirmation Runs (`800` Steps = `6.55M` Tokens = `2.18%` of `300M`)
- **`F11_SUBPARITY_R192_EXCEED_300M_800STEPS`**: Trains the $2\times$-smaller sub-parity `r=192` (`1.18M` L0 params vs. `2.36M` Teacher Attn0) configuration with the winning interior physics from `F5..F9` for `800` steps (`6.55M` tokens) on the `300M` Teacher to verify that even at **half the Layer-0 parameter count of Classical Attention**, the quantum wavepacket front-end strictly outperforms the `300M` Classical Teacher (`val_ce < 3.5845`).

---

## 4. Files to Create / Modify & Code Quality Gates (`docs/SCRIPT_BEST_PRACTICES.md`)

1. **[`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py)** (currently `455` lines / `< 400` SLOC; will remain `< 600` SLOC, well under the `<= 1,000` SLOC limit):
   - Add `BIORTHOGONAL_SVD_STIEFEL_R64`, `BIORTHOGONAL_SVD_STIEFEL_R192`, `BIORTHOGONAL_SVD_STIEFEL_R384` to `QuantumStiefelProjMode` and `resolve_svd_stiefel_rank`.
   - Implement `apply_interior_rho_v_coupling(...)` (`<= 8` parameters) supporting `ATTENDED_RHO_ON_ATTENDED_V`, `ATTENDED_RHO_ON_LOCAL_V`, `SYMMETRIZED_LOCAL_CONTEXT_RHO_V`, and `LIOUVILLE_COMMUTATOR_RHO_V` (exact closed-form Hermitian commutator $C_R = P_I + P_I^\top$, $C_I = P_R^\top - P_R$).
   - Add optional `x_0_768: torch.Tensor | None = None` to `SVDStiefelWavepacketStatMechBlock.forward` (`<= 8` parameters) and implement helper `encode_svd_stiefel_layer1(...)` so `quantum_frontend_graft.py` stays well under `1,000` SLOC.
2. **[`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py)** (currently `993` SLOC; we will move `initialize_arbitrary_shape_ppmi_sphere_embeddings_` / bridge helpers or delegate multi-layer SVD-Stiefel Layer-1 encoding to `quantum_svd_stiefel_graft.py` so `quantum_frontend_graft.py` remains `<= 995` SLOC):
   - Add `SYMMETRIZED_LOCAL_CONTEXT_RHO_V` and `LIOUVILLE_COMMUTATOR_RHO_V` to `QuantumAttnInteriorMode`.
   - Add `local_rho_mix_logit` and `liouville_beta` per-head scalar parameters to `ConfigurableWavepacketStatMechBlock`.
   - Wire `self.quantum_block1` in `FrozenTrunkQuantumFrontEndGPT` when `replacement_scope == QuantumReplacementScope.EMB_PLUS_L0_L1`.
3. **[`tests/test_plan070_subparity_commutator_and_multilayer_graft.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan070_subparity_commutator_and_multilayer_graft.py)** (`< 300` SLOC):
   - **Gate H1**: Verifies exact parameter counts and Stiefel column-orthonormality $\|U^\top U - I\|_F < 10^{-5}$, $\|V^\top V - I\|_F < 10^{-5}$ for `R64`, `R192`, and `R384` (verifying `R384` has $4 \times 2 \times 768 \times 384 = 2,359,296$ Stiefel matrix elements = exact 1:1 parity with $4 \times 768^2$).
   - **Gate H2**: Verifies that in `LIOUVILLE_COMMUTATOR_RHO_V`, the commutator $C = -i [\rho_{\text{local}}, \tilde{\rho}_{\text{evol}}]$ is strictly Hermitian ($\|C_R - C_R^\top\|_F < 10^{-5}$, $\|C_I + C_I^\top\|_F < 10^{-5}$), traceless ($|\operatorname{Tr}(C_R)| < 10^{-5}$), and propagates finite gradients to `liouville_beta`.
   - **Gate H3**: Verifies that `EMB_PLUS_L0_L1` with calibrated `SVDStiefelWavepacketStatMechBlock` initializes both `quantum_block0` and `quantum_block1`, keeps all classical trunk parameters frozen (`requires_grad == False`), and updates both Layer-0 and Layer-1 Stiefel/Unitary/Scalar parameters cleanly under a single optimizer step.
   - **Gate H4**: AST best-practices compliance audit (`<= 1,000` SLOC, `<= 8` args per function, zero semicolon statement packing) across all modified and new files.
4. **[`examples/80_verify_plan070_subparity_commutator_and_multilayer_graft.py`](file:///home/nixos/QuantumDLBlocks/examples/80_verify_plan070_subparity_commutator_and_multilayer_graft.py)** (`< 350` SLOC):
   - CLI benchmark runner executing `F1..F10` (`500` steps) and `F11` (`800` steps) in parallel across `cuda:0` and `cuda:1`, saving telemetry to `.cache/w52f_plan070_*.json`.
