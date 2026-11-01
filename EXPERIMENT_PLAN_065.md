# EXPERIMENT PLAN 065 (Wave 52A): Surgical Quantum Front-End Graft onto a Frozen Pretrained Classical GPT — Wavefunction Shape Taxonomy, State Representation I/O, and Interior Attention Operator Ablations

**Status**: `REVISED — AWAITING FINAL REVIEWER SIGN-OFF`  
**Wave**: `Wave 52A`  
**Date**: `2026-11-01`  
**Target Hardware**: `Dual NVIDIA GeForce RTX 3090 (2 x 24 GB VRAM, cuda:0 & cuda:1)`  
**Parent Baseline**: Pretrained `CompiledLeaderboardClassicalGPT` Checkpoints ([`w45b_classical_teacher_ckpt50m_l12_d768_b262k`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k)) & Wave 51B Zero-Packing Wavepacket Block ([`src/qdlblocks/layers/wavepacket_statmech_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/wavepacket_statmech_block.py))

---

## 1. Executive Summary & Core Scientific Question

### 1.1 The Core Question
If we take a **working, pretrained 12-layer classical Transformer** (`CompiledLeaderboardClassicalGPT`, $L=12, d_{\text{model}}=768, H=6, d_{\text{head}}=128, d_{\text{ff}}=3072, V=50304$), **freeze the entire downstream classical architecture** (`blocks[1..11]`, `value_embeds`, `skip_weights`, and `lm_head`), and **surgically replace the initial token embeddings (`wte`, `smear_gate`) and Layer 0 (`blocks[0]`)** with a quantum-inspired architecture:
1. **Can the quantum front-end learn to drive the frozen classical trunk and reproduce or exceed the original classical model's validation cross-entropy loss within a `< 300M`-token budget?**
2. **Which quantum front-end design choices** across wavefunction shape, state representation (complex wavefunction I/O vs. density-matrix I/O), interior attention operations, and Layer-0 replacement scope maximize validation cross-entropy recovery per token?

### 1.2 Pretrained Classical Teacher Checkpoints Available on Disk
We use the verified `Wave 45B` `CompiledLeaderboardClassicalGPT` checkpoints stored in [`/home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k):
- **Fast Screening Teacher (`m0050M`, `50.1M` tokens)**: `ckpt_step00191_0050M_m0050M.pt` — `val_ce = 4.2466`, `top1_acc = 31.16%`, `current_ws_blocks = 3`.
- **Full-Budget Target Teacher (`m0300M`, `300.2M` tokens)**: `ckpt_step01145_0300M_m0300M.pt` — **`val_ce = 3.5842`**, **`top1_acc = 36.88%`**, `current_ws_blocks = 6`.

### 1.3 Two-Plan Execution Strategy (`EXPERIMENT_PLAN_065.md` $\to$ `EXPERIMENT_PLAN_066.md`)
- **`EXPERIMENT_PLAN_065.md` (This Plan — `Wave 52A` Design-Space Tournament & Ablation Matrix)**:
  - Builds the exact **Frozen-Trunk Quantum Front-End Graft Harness** ([`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py)) that preserves 100% of the frozen classical trunk's forward physics (`FlexAttention` document + sliding-window `BlockMask`, `YaRN` rotary scaling, outer-layer `value_embeds` on layers `1, 2, 9, 10, 11`, and U-Net `skip_weights`).
  - Enumerates and systematically tests **4 design axes (`12` tournament arms `T0..T11`)** on the frozen `300M`-token classical teacher (`ckpt_step01145_0300M_m0300M.pt`) under a controlled screening budget across `cuda:0` and `cuda:1` to identify the Pareto-optimal wavefunction shape, state I/O representation, interior attention operator, and alignment schedule.
- **`EXPERIMENT_PLAN_066.md` (`Wave 52B` — Sub-`300M` Token Full Recovery & Teacher-Exceeding Scaling Run)**:
  - Drafted and reviewed with an independent reviewer while `EXPERIMENT_PLAN_065.md` runs on the GPUs.
  - Takes the winning configuration(s) from Plan 065 and executes long-horizon FineWeb training (`50M` $\to$ `150M` $\to$ `<300M` tokens) on the frozen classical teacher to test full loss restoration (`<= 3.5842` on `m0300M` and `<= 4.2466` on `m0050M`).

---

## 2. Exact Classical Interface Contract at the Layer-0 Boundary

In [`CompiledLeaderboardClassicalGPT._run_backbone_eager`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L785-L838), the boundary between the front-end (`wte`, `smear_gate`, `blocks[0]`) and the downstream trunk (`blocks[1..11]`, `skip_weights`, `lm_head`) consists of three tensors per microbatch `idx` $\in \{0, \dots, V-1\}^{B \times T}$:
1. **`x_0` $\in \mathbb{R}^{B \times T \times 768}$**: The RMS-normalized initial token embedding state (`F.rms_norm(raw_emb_768, (768,), eps=1e-6)`). Every downstream block $l \in \{1, \dots, 11\}$ mixes `x_0` into its input via `x = block_lambdas[0] * x + block_lambdas[1] * x_0`.
2. **`x_1` $\in \mathbb{R}^{B \times T \times 768}$**: The output hidden state of Layer 0 (`blocks[0]`). It serves both as the sequential input to `blocks[1]` and as the first encoder skip state `encoder_states[0] = x_1`, which is added into the final decoder block `blocks[11]` via `sigmoid(skip_weights[5]) * x_1`.
3. **`v_0` $\in \mathbb{R}^{B \times T \times 6 \times 128}$**: The Layer-0 value projection tensor passed through `blocks[1..11]`.

### 2.1 Critical Trunk Fidelity Requirement (Fixing Prior `_run_classical_tail` Omissions)
In the Wave 50/51 `HybridQuantumFrontEndClassicalGPT._run_classical_tail`, `blocks[1..11]` were called with `ve_l=None, block_mask=None`. For a **frozen pretrained `w45b` checkpoint**, `blocks[1, 2, 9, 10, 11]` were trained with non-zero `value_embeds[0..2]` and `FlexAttention` sliding-window `BlockMask` (`ws_blocks = 6` at `300M` tokens).
Therefore, `FrozenTrunkQuantumFrontEndGPT._run_frozen_classical_tail` in [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py) MUST:
1. Call `self.classical.set_sliding_window_blocks(ckpt["current_ws_blocks"], apply_yarn=True)` upon loading a teacher checkpoint.
2. Construct `bm_long, bm_short = self.classical._build_flex_block_masks(idx)` on each forward pass and pass `layer_bm = bm_long if block.is_long_window_layer else bm_short` to every block $l \in \{1, \dots, 11\}$.
3. Compute `ve_banks = [ve_emb(idx) for ve_emb in self.classical.value_embeds]` (under `torch.no_grad()`) and pass `ve_l = ve_banks[bank_idx]` to every block $l \in \{1, \dots, 11\}$ where `bank_idx = self.classical._ve_bank_index(l)` is not `None` ($l \in \{1, 2, 9, 10, 11\}$).

### 2.2 Avoiding Dead-Parameter Gradient Traps on `v_0` and `v_lambdas`
In `w45b` teacher checkpoints, `cfg.interior_value_mode == "unit_v"`, which means `blocks[1..11]` do not consume `v_0` during `PURE_CE` (`interface_warmup_steps == 0`). Furthermore, at Layer 0 (`layer_idx == 0`), incoming `v_0` is always `None`. To guarantee that **100% of active trainable parameters receive finite, non-zero gradients** in Gate `H2`:
1. At Layer 0, `v_lambdas` is **not instantiated as a trainable parameter** (`requires_grad=False`).
2. `v_0` $\in \mathbb{R}^{B \times T \times 6 \times 128}$ is computed by applying the **shared** input Stiefel bridge `project_to_768(v_wp.reshape(B, T, D_emb), self.w_in_bridge)` (which already receives gradients through `x_0`) and viewing as `(B, T, 6, 128)`. Zero trainable parameters exclusively feed `v_0`.

---

## 3. Taxonomy of Small Decisions to List & Test (`Axes A, B, C, D`)

### Axis A: Wavefunction Embedding Shape, Physical 3D `q_wte` Storage, & Native Layer-0 Execution (`QuantumWavefunctionShape`)

To ensure compatibility with `BatchSubspaceTikhonovGramMuon.step_active` in [`vocab_manifold_muon.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py#L326-L332) (which requires `p.ndim == 3`), `q_wte` is always registered as a **3D `nn.Parameter` of physical shape `(V, M_slices, d_sphere)`** where every row slice lies on the RMS hypersphere $\sqrt{d_{\text{sphere}}}\,\mathbb{S}^{d_{\text{sphere}} - 1}$. Moreover, **Quantum Layer 0 executes natively at total real width $D_{\text{emb}} = M_{\text{slices}} \times d_{\text{sphere}}$** and bridges to $d_{\text{model}} = 768$ at the Layer-0 boundary:

| Shape Enum (`QuantumWavefunctionShape`) | Logical Shape | Physical 3D `q_wte` Shape `(V, M_slices, d_sphere)` | Total Real Width $D_{\text{emb}}$ | Layer-0 Internal Packet Shape `(B, T, M_pkt, d_h)` | Layer-0 Attention ($H_q \times R \times d_h$) | Boundary Bridge to $d_{\text{model}} = 768$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SHAPE_24x2x16` (Baseline)** | `(V, 24, 2, 16)` | `(V, 48, 16)` ($d_{\text{sphere}}=16$) | `768` | `(B, T, 48, 16)` | $H_q = 6, R = 4, d_h = 16$ (`30` SDPA heads of `128`D) | Factored `(48, 16, 16)` + `(16, 48, 48)` Stiefel (`w_in_bridge = None`) |
| **`SHAPE_1x2x256` (Fewer, Larger — Compressed)** | `(V, 2, 256)` | `(V, 2, 256)` ($d_{\text{sphere}}=256$) | `512` | `(B, T, 32, 16)` | $H_q = 4, R = 4, d_h = 16$ (`20` SDPA heads of `128`D) | Rectangular Stiefel matrices $W_{\text{in,bridge}}, W_{\text{out,bridge}} \in \mathrm{St}_{\text{col}}(768, 512)$ (`x_768 = z_512 @ W.T`) |
| **`SHAPE_1x2x512` (Fewer, Larger — Overcomplete)** | `(V, 2, 512)` | `(V, 2, 512)` ($d_{\text{sphere}}=512$) | `1024` | `(B, T, 64, 16)` | $H_q = 8, R = 4, d_h = 16$ (`40` SDPA heads of `128`D) | Rectangular Stiefel matrices $W_{\text{in,bridge}}, W_{\text{out,bridge}} \in \mathrm{St}_{\text{col}}(1024, 768)$ (`x_768 = z_1024 @ W`) |
| **`SHAPE_256x2x2` (Many Small Qubit Spinors)** | `(V, 256, 2, 2)` | `(V, 256, 4)` ($d_{\text{sphere}}=4$, Bloch 3-sphere $2\,\mathbb{S}^3$) | `1024` | `(B, T, 256, 4)` | $H_q = 8$ heads $\times G = 32$ qubits ($d_h = 2$, `16` SDPA heads of `128`D) | Rectangular Stiefel matrices $W_{\text{in,bridge}}, W_{\text{out,bridge}} \in \mathrm{St}_{\text{col}}(1024, 768)$ (`x_768 = z_1024 @ W`) |

#### 3.1.1 Deterministic PPMI-SVD Initialization for Arbitrary `(V, M_slices, d_sphere)` (`initialize_arbitrary_shape_ppmi_sphere_embeddings_`)
Given the cached `(V, 768)` Shifted-PPMI + Randomized SVD matrix $P_{768} \in \mathbb{R}^{V \times 768}$ (`_DEFAULT_PPMI_CACHE_PATH`):
1. Construct $P_{D_{\text{emb}}} \in \mathbb{R}^{V \times D_{\text{emb}}}$ deterministically:
   - If $D_{\text{emb}} == 768$: $P_{D_{\text{emb}}} = P_{768}$.
   - If $D_{\text{emb}} == 512 < 768$: take the top `512` spectral columns $P_{512} = P_{768}[:, :512]$.
   - If $D_{\text{emb}} == 1024 > 768$: concatenate $P_{768}$ with the sign-flipped top `256` spectral columns: $P_{1024} = [P_{768},\, -P_{768}[:, :256]]$.
2. Reshape $P_{D_{\text{emb}}}$ to `(V, d_sphere, M_slices)` so singular values are interleaved across all $d_{\text{sphere}}$ coordinates of every slice, right-multiply by a deterministic Haar-orthogonal mode-balancing matrix $Q_M \in O(M_{\text{slices}})$ (`DEFAULT_PPMI_ORTHO_SEED = 2026064`), transpose to `(V, M_slices, d_sphere)`, and project each row onto $\sqrt{d_{\text{sphere}}}\,\mathbb{S}^{d_{\text{sphere}} - 1}$ via `project_rows_to_rms_sphere_`.

#### 3.1.2 Exact Layer-0 Handoff Equations (`x_0`, `x_1`, `v_0`) for All 4 Shapes
Let `raw_wp` $\in \mathbb{R}^{B \times T \times M_{\text{pkt}} \times d_{\text{pkt}}}$ be the embedded token wavepackets after multi-lag causal wavepacket dispersion (`apply_causal_wavepacket_dispersion`) and per-packet `F.rms_norm(..., (d_pkt,))`, where $(M_{\text{pkt}}, d_{\text{pkt}}) \in \{(48, 16), (32, 16), (64, 16), (256, 4)\}$ and $D_{\text{emb}} = M_{\text{pkt}} d_{\text{pkt}}$:
1. Apply the initial factored Stiefel rotation `x_0_wp = apply_factored_wavepacket_stiefel(tok_wp, self.w_in_pkt, self.w_in_mode) * self.in_scale` $\in \mathbb{R}^{B \times T \times M_{\text{pkt}} \times d_{\text{pkt}}}$.
2. Define the bridge projection $\Pi_{D_{\text{emb}} \to 768}(z, W)$:
   - If $D_{\text{emb}} == 768$ (`W is None`): $\Pi(z, \text{None}) = z$.
   - If $D_{\text{emb}} == 512$ ($W \in \mathrm{St}_{\text{col}}(768, 512)$): $\Pi(z, W) = z W^\top \in \mathbb{R}^{B \times T \times 768}$.
   - If $D_{\text{emb}} == 1024$ ($W \in \mathrm{St}_{\text{col}}(1024, 768)$): $\Pi(z, W) = z W \in \mathbb{R}^{B \times T \times 768}$.
3. Compute `x_0` $\in \mathbb{R}^{B \times T \times 768}$:
   $$x_0 = \operatorname{RMSNorm}_{768}\!\Big(\Pi_{D_{\text{emb}} \to 768}\big(\operatorname{reshape}(x_{0,\text{wp}}, (B, T, D_{\text{emb}})),\, W_{\text{in,bridge}}\big)\Big)$$
4. Run Quantum Layer 0 natively on `x_0_wp` $\in \mathbb{R}^{B \times T \times M_{\text{pkt}} \times d_{\text{pkt}}}$ to obtain post-attention state `x_mid_wp` (and, if `FULL_BLOCK0_QUANTUM`, post-Mach-Zehnder state `x_1_wp`) and Layer-0 value wavepackets `v_wp`:
   - Compute `v_0` $\in \mathbb{R}^{B \times T \times 6 \times 128}$ using the shared `W_in_bridge`:
     $$v_0 = \operatorname{reshape}\!\Big(\Pi_{D_{\text{emb}} \to 768}\big(\operatorname{reshape}(v_{\text{wp}}, (B, T, D_{\text{emb}})),\, W_{\text{in,bridge}}\big),\, (B, T, 6, 128)\Big)$$
   - If `replacement_scope == FULL_BLOCK0_QUANTUM`:
     $$x_1 = \Pi_{D_{\text{emb}} \to 768}\big(\operatorname{reshape}(x_{1,\text{wp}}, (B, T, D_{\text{emb}})),\, W_{\text{out,bridge}}\big)$$
   - If `replacement_scope == ATTN0_QUANTUM_KEEP_FROZEN_MLP0`:
     $$x_{\text{mid},768} = x_0 + \Pi_{D_{\text{emb}} \to 768}\big(\operatorname{reshape}(\text{obs\_scale} \cdot d_{\text{obs}}, (B, T, D_{\text{emb}})),\, W_{\text{out,bridge}}\big)$$
     $$x_1 = x_{\text{mid},768} + \text{fc2}_{\text{frozen}}\!\Big(\operatorname{ReLU}\!\big(\text{fc1}_{\text{frozen}}(\operatorname{RMSNorm}_{768}(x_{\text{mid},768}))\big)^2\Big)$$

#### 3.1.3 Complete Specification of `QubitSpinorBlochStatMechBlock` (`SHAPE_256x2x2`)
For `SHAPE_256x2x2`, the layer state is `x` $\in \mathbb{R}^{B \times T \times 256 \times 4}$, representing $M_c = 256$ 2-level complex spinors $|\psi_m\rangle = (\alpha_m, \beta_m)^\top \in \mathbb{C}^2$ with real coordinates $(\alpha_{\text{re}}, \alpha_{\text{im}}, \beta_{\text{re}}, \beta_{\text{im}}) \in \mathbb{R}^4$:
1. **Factored Stiefel Projections (`(256, 4, 4)` + `(4, 256, 256)`)**:
   - `w_q_pkt, w_k_pkt, w_v_pkt, w_obs_pkt, w_beam_a_pkt, w_beam_b_pkt` $\in \mathrm{St}_{\text{col}}(4, 4)^{256}$ (shape `(256, 4, 4)`), `w_chan_pkt` $\in \mathrm{St}_{\text{col}}(4, 4)^{512}$ (shape `(512, 4, 4)`).
   - `m_q_mode, m_k_mode, m_v_mode, m_obs_mode` $\in \mathrm{St}_{\text{col}}(256, 256)^4$ (shape `(4, 256, 256)`), `m_beam_a_mode, m_beam_b_mode` $\in \mathbb{R}^{4 \times 256 \times 512}$ (tall transpose in $\mathrm{St}_{\text{col}}(512, 256)^4$), `m_chan_mode` $\in \mathrm{St}_{\text{col}}(512, 256)^4$ (shape `(4, 512, 256)`).
2. **2-Chunk Bloch + Spinor FlashAttention (`D = 128`)**:
   - Partition the $256$ qubits into $H_q = 8$ heads of $G = 32$ qubits each (`32 * 4 = 128` floats per head).
   - Apply complex rotary phase $e^{i t \omega_{g,c}}$ ($g \in \{0,\dots,31\}, c \in \{0,1\}$) to `q_wp, k_wp` viewed as `(B, T, 8, 32, 2, 2)`, RMS-normalize per head to `(B, T, 8, 128)`, and expand across `2` chunks to `(B, 16, T, 128)`.
   - For each token $s$ and qubit $m \in \{0, \dots, 255\}$, normalize $v_{s,m} = (\alpha_{\text{re}}, \alpha_{\text{im}}, \beta_{\text{re}}, \beta_{\text{im}})$ to unit complex norm $\|\tilde{v}_{s,m}\|_2^2 = |\tilde{\alpha}_{s,m}|^2 + |\tilde{\beta}_{s,m}|^2 = 1$, and form the 4D Bloch vector $b_{s,m} = (b_x, b_y, b_z, b_0) \in \mathbb{R}^4$:
     $$b_x = 2(\tilde{\alpha}_{\text{re}}\tilde{\beta}_{\text{re}} + \tilde{\alpha}_{\text{im}}\tilde{\beta}_{\text{im}}), \quad b_y = 2(\tilde{\alpha}_{\text{re}}\tilde{\beta}_{\text{im}} - \tilde{\alpha}_{\text{im}}\tilde{\beta}_{\text{re}}), \quad b_z = |\tilde{\alpha}|^2 - |\tilde{\beta}|^2, \quad b_0 = |\tilde{\alpha}|^2 + |\tilde{\beta}|^2 = 1$$
   - Stack Chunk 0 (`b_{s,m}` viewed as `(B, T, 8, 1, 128)`) and Chunk 1 (`\tilde{v}_{s,m}` viewed as `(B, T, 8, 1, 128)`) into `sdpa_v` of shape `(B, 16, T, 128)` and run causal `F.scaled_dot_product_attention` (`scale = 0.22`).
3. **Analytical $2 \times 2$ Bloch McWeeny Purification & $U(2)$ Evolution**:
   - Unpack attended Chunk 0 into $\vec{r}_{t,m} = (r_x, r_y, r_z) \in \mathbb{R}^3$ (`\|\vec{r}_{t,m}\|_2 \le 1$) and attended Chunk 1 into spinor $|\bar{v}_{t,m}\rangle \in \mathbb{C}^2$ (normalized to $|\tilde{\bar{v}}_{t,m}\rangle$).
   - Analytical McWeeny purification on $\mathbb{C}^{2 \times 2}$:
     $$\mathcal{P}_{t,m} = \frac{1 + \|\vec{r}_{t,m}\|_2^2}{2} \in [0.5, 1.0], \qquad \text{eff\_rank}_{t,m} = \frac{1}{\mathcal{P}_{t,m}} \in [1.0, 2.0], \qquad \vec{r}_{\text{pure}, t,m} = \frac{2\,\vec{r}_{t,m}}{1 + \|\vec{r}_{t,m}\|_2^2}$$
   - Reconstruct the $2 \times 2$ Hermitian PSD density matrix $\rho_{\text{pure}, t,m} = \frac{1}{2}\begin{pmatrix} 1 + r_{\text{pure},z} & r_{\text{pure},x} - i r_{\text{pure},y} \\ r_{\text{pure},x} + i r_{\text{pure},y} & 1 - r_{\text{pure},z} \end{pmatrix} \in \mathrm{Herm}_+(2)$, conjugate by per-qubit unitary $U_m \in U(2)$ (`head_unitary_re, head_unitary_im` of shape `(256, 2, 2)`): $\rho_{\text{evol}, t,m} = U_m \rho_{\text{pure}, t,m} U_m^\dagger$, and apply $\rho_{\text{evol}, t,m}$ to the $U_m$-rotated spinor $|\tilde{\bar{v}}_{t,m}\rangle$:
     $$|\psi^{\text{out}}_{t,m}\rangle = \big(I_2 + \tfrac{1}{2}\rho_{\text{evol}, t,m}\big) U_m |\tilde{\bar{v}}_{t,m}\rangle \in \mathbb{C}^2, \qquad y_{\text{wp}} = \operatorname{RMSNorm}_4\!\big(\operatorname{view}(|\psi^{\text{out}}_{t,m}\rangle, (B, T, 256, 4))\big)$$

---

### Axis B: State Representation Contract (`QuantumStateIOMode`)

1. **`WF_IN_WF_OUT` (Complex Wavefunction In $\to$ Complex Wavefunction Out)**:
   - Input is complex wavefunction $\Psi_{\text{in}}$; density matrices $\rho_{t,h}$ are formed internally as $d_h \times d_h$ operators that multiply the $V$-space wavefunctions $|v'_{t,h,r}\rangle = (I_{d_h} + \frac{1}{2}\rho_{\text{evol}, t,h}) U_h^{(0)} |\tilde{\bar{v}}_{t,h,r}\rangle$; output is `F.rms_norm(..., (d_h,))` of shape `(B, T, M_pkt, d_h)`.
2. **`DM_IN_DM_OUT` (Density Matrix In $\to$ Density Matrix Out / Liouville-Kraus Transport)**:
   - Token wavepackets at position $s$ immediately form the initial token density matrix $\rho^{(0)}_{s,h} = \sum_{r=1}^R |\tilde{v}_{s,h,r}\rangle\langle\tilde{v}_{s,h,r}| \in \mathrm{Herm}_+(d_h)$. Only the $R=4$ row-chunks (`128`D each) of $\rho^{(0)}_{s,h}$ are passed through SDPA (`4 * H_q` SDPA heads of `128`D; no linear ket chunk!).
   - Causal attention produces the mixed ensemble $\bar{\rho}_{t,h} = \sum_{s \le t} A_{t,s}^{(h)} \rho^{(0)}_{s,h} \in \mathrm{Herm}_+(d_h)$, which is purified to $\rho_{\text{pure}, t,h}$ and evolved to $\rho_{\text{evol}, t,h} \in \mathrm{Herm}_+(d_h)$.
   - Readout back to `(B, T, M_pkt, d_h)` applies $(I_{d_h} + \frac{1}{2}\rho_{\text{evol}, t,h})$ to the first $R=4$ orthonormal columns $|u_{h,r}\rangle = U_h^{(0)}[:, :, r] \in \mathbb{C}^{d_h}$ ($r \in \{0, \dots, R-1\}$) of the head unitary $U_h^{(0)}$:
     $$|\psi^{\text{out}}_{t,h,r}\rangle = \big(I_{d_h} + \tfrac{1}{2}\rho_{\text{evol}, t,h}\big) |u_{h,r}\rangle \in \mathbb{C}^{d_h}, \qquad y_{\text{wp}} = \operatorname{RMSNorm}_{d_h}\!\big(\operatorname{stack}(\text{Re}, \text{Im})\big)$$
3. **`WF_IN_DENSITY_OBS_OUT` (Wavefunction In $\to$ Wavefunction + Born Diagonal Observable Out)**:
   - Computes $y_{\text{wp}} \in \mathbb{R}^{B \times T \times H_q \times 2R \times d_h}$ as in `WF_IN_WF_OUT`, extracts the evolved Born diagonal populations $d_{t,h,p} = (\rho_{\text{evol},\text{re}, t,h})_{p,p} \in \mathbb{R}_+$ ($\sum_{p=1}^{d_h} d_{t,h,p} = 1$), and modulates each Hilbert coordinate $p \in \{1, \dots, d_h\}$ via a learnable per-head scalar gain $\gamma_{\text{diag}} \in \mathbb{R}^{H_q}$ (initialized to `0.25`):
     $$y^{\text{obs}}_{t,h,m,p} = \operatorname{RMSNorm}_{d_h}\!\Big(y_{t,h,m,p} \cdot \big(1 + \tanh(\gamma_{\text{diag}, h}) \cdot (d_h \cdot d_{t,h,p} - 1)\big)\Big)$$

---

### Axis C: Interior Attention Operations (`QuantumAttnInteriorMode`)

1. **`ATTENDED_RHO_ON_ATTENDED_V` (`v7` Standard — `5` Chunks of `128`D per Head)**:
   - Causally attend $\rho_{t,h}^{(0)} = \sum_{s \le t} A_{t,s}^{(h)} \sum_{r=1}^R |\tilde{v}_{s,h,r}\rangle\langle\tilde{v}_{s,h,r}|$ (`4` chunks) and $|\bar{v}_{t,h,r}\rangle = \sum_{s \le t} A_{t,s}^{(h)} |\tilde{v}_{s,h,r}\rangle$ (`1` chunk); purify/evolve $\rho_{t,h}^{(0)} \mapsto \rho_{\text{evol}, t,h}$; output $(I_{d_h} + \frac{1}{2}\rho_{\text{evol}, t,h}) U_h^{(0)} |\tilde{\bar{v}}_{t,h,r}\rangle$.
2. **`ATTENDED_RHO_ON_LOCAL_V` (Attended Input Outer-Product Acting on Local $V$-Projection — `5` Chunks of `128`D per Head)**:
   - Form the outer product from the normalized **input wavefunctions** $\rho_{s,h}^{\text{in}} = \sum_{r=1}^R |\tilde{\psi}_{s,h,r}^{\text{in}}\rangle\langle\tilde{\psi}_{s,h,r}^{\text{in}}|$ (`4` chunks) alongside $|\tilde{v}_{s,h,r}\rangle$ (`1` chunk), causally attend $\rho_{t,h}^{(0)} = \sum_{s \le t} A_{t,s}^{(h)} \rho_{s,h}^{\text{in}}$ and $|\bar{v}_{t,h,r}\rangle$, purify/evolve $\rho_{t,h}^{(0)} \mapsto \rho_{\text{evol}, t,h}$, and use $\rho_{\text{evol}, t,h}$ to combine the current token's local $V$-projection $|\tilde{v}_{t,h,r}\rangle$ with $|\tilde{\bar{v}}_{t,h,r}\rangle$:
     $$|\psi^{\text{out}}_{t,h,r}\rangle = U_h^{(0)} |\tilde{\bar{v}}_{t,h,r}\rangle + \tfrac{1}{2}\rho_{\text{evol}, t,h} U_h^{(0)} |\tilde{v}_{t,h,r}\rangle$$
3. **`KV_CROSS_OUTER_ON_Q` (Cross-Outer Operator $\sum_{s \le t} A_{t,s} |v_s\rangle\langle\psi_s^{\text{in}}|$ Acting on Query — `4` Chunks of `128`D per Head)**:
   - At each token $s$, form the rank-$R$ cross-outer product $\Omega_{s,h} = \sum_{r=1}^R |\tilde{v}_{s,h,r}\rangle\langle\tilde{\psi}_{s,h,r}^{\text{in}}| \in \mathbb{C}^{d_h \times d_h}$ (`4` row-chunks of `128`D).
   - Causally attend $\bar{\Omega}_{t,h} = \sum_{s \le t} A_{t,s}^{(h)} \Omega_{s,h} \in \mathbb{C}^{d_h \times d_h}$ (`4` chunks of `128`D).
   - Form the strictly Hermitian, unit-trace, positive semi-definite Gram density operator:
     $$\rho_{t,h}^{(0)} = \frac{\bar{\Omega}_{t,h}\,\bar{\Omega}_{t,h}^\dagger}{\operatorname{Tr}\big(\bar{\Omega}_{t,h}\,\bar{\Omega}_{t,h}^\dagger\big) + \epsilon} \in \mathrm{Herm}_+(d_h), \qquad \operatorname{Tr}(\rho_{t,h}^{(0)}) = 1, \quad \rho_{t,h}^{(0)} \succeq 0$$
   - Purify and evolve $\rho_{t,h}^{(0)} \mapsto \rho_{\text{evol}, t,h}$, apply $\bar{\Omega}_{t,h}$ to the current token's normalized query ket $|\hat{v}_{t,h,r}\rangle = \operatorname{normalize}(\bar{\Omega}_{t,h} |\tilde{q}_{t,h,r}\rangle)$, and output $(I_{d_h} + \frac{1}{2}\rho_{\text{evol}, t,h}) U_h^{(0)} |\hat{v}_{t,h,r}\rangle$.
4. **`PRE_ATTN_LOCAL_OUTER_ON_V` (Pre-SDPA Local Outer-Product Modulation — `1` Chunk of `128`D per Head)**:
   - At each token $s$, form the input outer product $\rho_{s,h}^{(0)} = \sum_{r=1}^R |\tilde{\psi}_{s,h,r}^{\text{in}}\rangle\langle\tilde{\psi}_{s,h,r}^{\text{in}}| \in \mathrm{Herm}_+(d_h)$, purify and evolve locally to $\rho_{\text{evol}, s,h}$, transform token $s$'s projection into $V$ space via $|v'_{s,h,r}\rangle = \operatorname{normalize}\big((I_{d_h} + \frac{1}{2}\rho_{\text{evol}, s,h}) U_h^{(0)} |\tilde{v}_{s,h,r}\rangle\big)$, and aggregate $|v'_{s,h,r}\rangle$ causally via a **single-chunk (`D = 128`)** FlashAttention-2 call (`H_q` heads of `128`D).

---

### Axis D: Layer-0 Replacement Scope & Teacher Interface Alignment (`QuantumReplacementScope`)

1. **Replacement Scope (`QuantumReplacementScope`)**:
   - **`FULL_BLOCK0_QUANTUM`**: Replace `wte` + `smear_gate` + the entire `blocks[0]`.
   - **`ATTN0_QUANTUM_KEEP_FROZEN_MLP0`**: Replace `wte` + `smear_gate` + `blocks[0].attn`, while routing the post-attention residual `x_mid_768` through the **frozen pretrained classical `blocks[0].fc1` + `blocks[0].fc2` ($\text{ReLU}^2$ MLP)**. In this mode, the quantum Mach-Zehnder parameters (`w_beam_a_pkt, w_beam_b_pkt, w_chan_pkt, m_beam_a_mode, m_beam_b_mode, m_chan_mode, chan_scale`) have `requires_grad_(False)` and are omitted from `real_stiefel_parameters()` and `scalar_parameters()`.
2. **Teacher Interface Distillation Warmup (`interface_warmup_steps`)**:
   - For step $0 \le t < T_{\text{align}}$ (`T_align = interface_warmup_steps`), define the cosine decay schedule and elementwise MSE alignment loss against the detached classical teacher's Layer-0 states $(x_0^{\text{C}}, x_1^{\text{C}}, v_0^{\text{C}})$:
     $$\lambda_{\text{align}}(t) = \frac{1}{2}\,\lambda_0\left(1 + \cos\!\left(\frac{\pi t}{T_{\text{align}}}\right)\right), \qquad \lambda_0 = \text{DEFAULT\_GRAFT\_INTERFACE\_ALIGN\_WEIGHT} = 1.0$$
     $$\mathcal{L}_{\text{total}}(t) = \mathcal{L}_{\text{CE}} + \lambda_{\text{align}}(t)\,\Big(\operatorname{MSE}(x_0^{\text{Q}}, x_0^{\text{C}}) + \operatorname{MSE}(x_1^{\text{Q}}, x_1^{\text{C}}) + 0.25\,\operatorname{MSE}(v_0^{\text{Q}}, v_0^{\text{C}})\Big)$$
     where $\operatorname{MSE}(a, b) = \operatorname{mean}((a.\text{float}() - b.\text{float}())^2)$, and $\lambda_{\text{align}}(t) = 0$ for $t \ge T_{\text{align}}$.

---

## 4. Controlled Screening Tournament Matrix (`12` Arms `T0..T11` on Frozen `300M` Classical Teacher)

Every arm loads [`ckpt_step01145_0300M_m0300M.pt`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/ckpt_step01145_0300M_m0300M.pt) (`300.2M` tokens, reference `val_ce = 3.5842`, `top1 = 36.88%`) and trains only the quantum front-end on FineWeb Shard 1 (`B=8, T=1024`), evaluating on the `65,536`-token FineWeb Shard-0 validation slice:

| Arm ID | Shape (`Axis A`) | State I/O (`Axis B`) | Interior Attn Op (`Axis C`) | Scope & Warmup (`Axis D`) | Question Answered |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`T0_TEACHER_REF`** | Classical `(V, 768)` | Classical | Classical `blocks[0]` | Frozen Teacher (`0` steps) | Verifies exact `3.5842` Val CE (`36.88%` Top-1) pass-through. |
| **`T1_WP_24x2x16_BASE`** | `SHAPE_24x2x16` (`768`D) | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Baseline `v7` pure-WF `(V, 24, 2, 16)` replacing all of `wte + blocks[0]`. |
| **`T2_WP_1x2x256`** | `SHAPE_1x2x256` (`512`D) | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Tests fewer, larger complex wavefunctions `(V, 2, 256)` (`512 -> 768` Stiefel lift). |
| **`T3_WP_1x2x512`** | `SHAPE_1x2x512` (`1024`D) | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Tests overcomplete large complex wavefunction `(V, 2, 512)` (`1024 -> 768` Stiefel proj). |
| **`T4_QUBIT_256x2x2`** | `SHAPE_256x2x2` (`1024`D) | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` ($2\times 2$ Bloch) | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Tests many small 2-level qubit spinors `(V, 256, 2, 2)` with analytical $2\times 2$ Bloch purification. |
| **`T5_DM_IO_24x2x16`** | `SHAPE_24x2x16` (`768`D) | `DM_IN_DM_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Tests whether Density-Matrix I/O (`DM_IN_DM_OUT`) is more efficiently learnable than `WF_IN_WF_OUT`. |
| **`T6_HYBRID_OBS_OUT`** | `SHAPE_24x2x16` (`768`D) | `WF_IN_DENSITY_OBS_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Tests Born diagonal population modulation (`WF_IN_DENSITY_OBS_OUT`). |
| **`T7_INT_LOCAL_V`** | `SHAPE_24x2x16` (`768`D) | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_LOCAL_V` | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Tests input outer product $\rho_t = \sum_{s\le t} A_{t,s}|\psi_s^{\text{in}}\rangle\langle\psi_s^{\text{in}}|$ modulating local $|v_t\rangle$. |
| **`T8_INT_KV_CROSS`** | `SHAPE_24x2x16` (`768`D) | `WF_IN_WF_OUT` | `KV_CROSS_OUTER_ON_Q` | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Tests cross-outer operator $\sum_{s\le t} A_{t,s}|v_s\rangle\langle\psi_s^{\text{in}}|$ acting on $|q_t\rangle$. |
| **`T9_INT_PRE_ATTN`** | `SHAPE_24x2x16` (`768`D) | `WF_IN_WF_OUT` | `PRE_ATTN_LOCAL_OUTER_ON_V` | `FULL_BLOCK0_QUANTUM` (`PURE_CE`) | Tests pre-attention local outer-product $V$-modulation (`1`-chunk SDPA). |
| **`T10_SCOPE_KEEP_MLP0`** | `SHAPE_24x2x16` (`768`D) | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` (`PURE_CE`) | Tests replacing `wte + blocks[0].attn` while keeping frozen classical `blocks[0].mlp`. |
| **`T11_ALIGN_WARMUP`** | `SHAPE_24x2x16` (`768`D) | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_ATTENDED_V` | `FULL_BLOCK0_QUANTUM` (`WARMUP_INTERFACE_PLUS_CE`) | Tests Layer-0 teacher interface warmup (`x_0, x_1, v_0`) + pure CE. |

---

## 5. Top-of-File Named Constants Ledger (Rule 7 of `docs/SCRIPT_BEST_PRACTICES.md`)

| Constant Name | Value | Module | Description |
| :--- | :---: | :--- | :--- |
| `DEFAULT_TEACHER_300M_CKPT_PATH` | `.cache/checkpoints/w45b_.../ckpt_step01145_0300M_m0300M.pt` | `quantum_frontend_graft.py` | Default `300.2M`-token pretrained classical teacher checkpoint (`val_ce = 3.5842`). |
| `DEFAULT_TEACHER_50M_CKPT_PATH` | `.cache/checkpoints/w45b_.../ckpt_step00191_0050M_m0050M.pt` | `quantum_frontend_graft.py` | Default `50.1M`-token pretrained classical teacher checkpoint (`val_ce = 4.2466`). |
| `DEFAULT_WF_LARGE_256_REAL_DIM` | `512` | `quantum_frontend_graft.py` | Total real embedding width for `(V, 2, 256)` single complex wavefunction (`2 * 256`). |
| `DEFAULT_WF_LARGE_512_REAL_DIM` | `1024` | `quantum_frontend_graft.py` | Total real embedding width for `(V, 2, 512)` overcomplete complex wavefunction (`2 * 512`). |
| `DEFAULT_QUBIT_SPINOR_COUNT` | `256` | `quantum_frontend_graft.py` | Number of 2-level complex spinors in `(V, 256, 2, 2)` (`(V, 256, 4)` physical 3D storage). |
| `DEFAULT_QUBIT_SPHERE_DIM` | `4` | `quantum_frontend_graft.py` | Real sphere dimension per qubit spinor ($2\,\mathbb{S}^3 \subset \mathbb{R}^4$). |
| `DEFAULT_QUBIT_HEADS` | `8` | `quantum_frontend_graft.py` | Number of attention heads for the 256-qubit block (`32` qubits $\times 4 = 128$D per head). |
| `DEFAULT_GRAFT_INTERFACE_ALIGN_WEIGHT` | `1.0` | `quantum_frontend_graft.py` | Initial weight $\lambda_0$ for Layer-0 `(x_0, x_1, v_0)` interface warmup. |
| `DEFAULT_GRAFT_V0_ALIGN_SCALE` | `0.25` | `quantum_frontend_graft.py` | Relative weight on `v_0` MSE inside the Layer-0 interface alignment loss. |
| `DEFAULT_GRAFT_DIAG_OBS_INIT` | `0.25` | `quantum_frontend_graft.py` | Initial scalar value for $\gamma_{\text{diag}}$ in `WF_IN_DENSITY_OBS_OUT`. |
| `DEFAULT_GRAFT_STIEFEL_LR` | `0.04` | `quantum_frontend_graft.py` | Learning rate for `ModularRiemannianStiefelKrausMuon` (`RIEMANNIAN_TANGENT_TRANSPORTED`). |
| `DEFAULT_GRAFT_UNITARY_LR` | `0.02` | `quantum_frontend_graft.py` | Unitary/Kraus geodesic step size for `ModularRiemannianStiefelKrausMuon`. |
| `DEFAULT_GRAFT_WTE_LR` | `0.12` | `quantum_frontend_graft.py` | Active-row hypersphere learning rate for `BatchSubspaceTikhonovGramMuon` on 3D `q_wte`. |
| `DEFAULT_GRAFT_SCALAR_LR` | `0.04` | `quantum_frontend_graft.py` | AdamW learning rate for 1D quantum front-end scalar parameters. |

---

## 6. Planned File Changes & Verification Gates (`H1`–`H6`)

### 6.1 Planned Files (`<= 1,000` SLOC Soft Limit, `<= 8` Args per Function)
1. **`[CREATE]` [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py)**:
   - Implements `QuantumWavefunctionShape`, `QuantumStateIOMode`, `QuantumAttnInteriorMode`, `QuantumReplacementScope`, `QuantumFrontEndGraftConfig`, `initialize_arbitrary_shape_ppmi_sphere_embeddings_`, `QubitSpinorBlochStatMechBlock`, `ConfigurableWavepacketStatMechBlock`, `FrozenTrunkQuantumFrontEndGPT`, and `build_frozen_trunk_quantum_frontend_optimizers` (using `StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED` so all 2D rectangular and 3D square Stiefel tensors maintain strict $W^\top W = I_d$).
2. **`[CREATE]` [`tests/test_plan065_quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan065_quantum_frontend_graft.py)**:
   - Deterministic unit tests verifying Gates `H1`–`H6`.
3. **`[CREATE]` [`examples/75_verify_plan065_frozen_trunk_quantum_frontend_tournament.py`](file:///home/nixos/QuantumDLBlocks/examples/75_verify_plan065_frozen_trunk_quantum_frontend_tournament.py)**:
   - CLI tournament and scaling runner supporting `--arm {T0..T11}`, `--teacher-ckpt`, `--steps`, `--eval-every`, `--interface-warmup-steps`, `--device`, and `--out-json`.

### 6.2 Verification Gates (`H1`–`H6`)
- **Gate `H1` (Exact Pretrained Classical Teacher Pass-Through)**: Loading `ckpt_step01145_0300M_m0300M.pt` into `FrozenTrunkQuantumFrontEndGPT` in teacher reference mode (`T0_TEACHER_REF`) reproduces `val_ce = 3.584221` within `< 1e-4` on the `65,536`-token FineWeb Shard-0 slice.
- **Gate `H2` (Strict Trunk Freeze & 100% Active Front-End Gradients)**: Across backward passes in all quantum graft modes (both `FULL_BLOCK0_QUANTUM` and `ATTN0_QUANTUM_KEEP_FROZEN_MLP0`), `all(p.grad is None for p in frozen_trunk_params)` holds strictly, while 100% of active quantum front-end parameters returned by the model receive finite, non-zero gradients (`p.grad is not None and p.grad.abs().max() > 0`).
- **Gate `H3` (Wavefunction Shape Taxonomy `(V,2,256)`, `(V,2,512)`, `(V,256,2,2)`, `(V,24,2,16)` Causal & Strict Stiefel/Sphere Invariance)**: All 4 shape modes preserve strict causal autoregressiveness (`max |logits[:, :t] - logits_mut[:, :t]| < 1e-5`), unit-RMS sphere norms on 3D `q_wte` (`max |rms - 1| < 1e-5`), and strict Stiefel orthogonality $\|W^\top W - I_d\|_F / \sqrt{d} < 10^{-5}$ (with taller dimension on `dim=-2` and shorter dimension $d$ on `dim=-1`) across 3 optimizer steps.
- **Gate `H4` (State I/O & Interior Attention Operator Physicality)**: All 3 `QuantumStateIOMode`s (`WF_IN_WF_OUT`, `DM_IN_DM_OUT`, `WF_IN_DENSITY_OBS_OUT`) and all 4 `QuantumAttnInteriorMode`s (`ATTENDED_RHO_ON_ATTENDED_V`, `ATTENDED_RHO_ON_LOCAL_V`, `KV_CROSS_OUTER_ON_Q`, `PRE_ATTN_LOCAL_OUTER_ON_V`) produce unit-trace (`< 1e-5`) and PSD (`min_eig > -1e-6`) internal density matrices $\rho^{(0)}_{t,h}$.
- **Gate `H5` (Replacement Scope & Teacher Interface Warmup Descent)**: Both `FULL_BLOCK0_QUANTUM` and `ATTN0_QUANTUM_KEEP_FROZEN_MLP0` reduce loss over training steps, and `WARMUP_INTERFACE_PLUS_CE` reduces `(x_0, x_1, v_0)` interface MSE against the classical teacher by `> 50%` within 10 steps.
- **Gate `H6` (AST & SLOC Linter Compliance)**: All created/modified files pass `scripts/ast_signature_and_length_linter.py` (`<= 1,000` SLOC soft limit, `<= 8` args per function, `0` semicolons).
