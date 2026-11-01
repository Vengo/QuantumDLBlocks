# EXPERIMENT PLAN 066: Teacher-Subspace Spherical Warm-Start, Quantum Value-Bank Coupling & Sub-300M-Token Recovery on a Frozen Pretrained Classical GPT (`Wave 52B`)

**Date:** 2026-11-01  
**Status:** APPROVED BY INDEPENDENT REVIEWER (`035f390f-cf8b-4cd9-b7fb-acf6643e4129`) — Ready for Implementation  
**Prerequisite:** [`EXPERIMENT_PLAN_065.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_065.md) (`Wave 52A`: Surgical Quantum Front-End Graft Tournament Across Wavefunction Shapes, State I/O, Interior Attention Operators, and Replacement Scopes)  
**Primary Code Targets:**
- [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py)
- [`tests/test_plan066_quantum_frontend_recovery.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan066_quantum_frontend_recovery.py)
- [`examples/76_verify_plan066_quantum_frontend_sub300m_recovery.py`](file:///home/nixos/QuantumDLBlocks/examples/76_verify_plan066_quantum_frontend_sub300m_recovery.py)

---

## 1. Executive Summary & Architectural Motivation

While [`EXPERIMENT_PLAN_065.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_065.md) (`Wave 52A`) executes its 12-arm (`T0..T11`) screening tournament across the four structural axes (Wavefunction Shape `Axis A`, State I/O `Axis B`, Interior Attention Operations `Axis C`, and Layer-0 Replacement Scope `Axis D`), a fundamental mathematical question governs whether a grafted quantum front-end can **reproduce or exceed** a pretrained classical Transformer's validation cross-entropy well within a `< 300M`-token budget:

> **Why does grafting a randomly or PPMI-SVD-initialized Quantum Front-End (`q_wte` + Quantum Layer 0) onto a frozen pretrained classical trunk (`blocks[1..11]`, `value_embeds`, `skip_weights`, `lm_head`) start at a high initial loss (`val_ce ~ 9.5–10.5`), and what exact mathematical mechanisms allow it to recover and surpass the classical front-end in a fraction of the token budget?**

We identify three concrete structural mechanisms that directly govern the token efficiency and asymptotic ceiling of a surgical quantum front-end graft:

1. **Axis E — Token Embedding Subspace Alignment (`QuantumWteInitMode`)**:
   - In Plan 065 (`PPMI_SVD_SPHERE`), `q_wte` is initialized from `ppmi_svd_50304_768_shard1.pt`. Because the singular vectors of the PPMI matrix define an arbitrary orthogonal coordinate frame relative to the frozen teacher's trained embedding matrix $W_{\text{teacher}} = \text{classical.wte.weight} \in \mathbb{R}^{V \times 768}$, the initial quantum state $x_0$ has near-zero cosine alignment with the coordinate basis expected by frozen `blocks[1..11]` and `lm_head`.
   - We introduce **`TEACHER_PROCRUSTES_SPHERE`**: a zero-token, closed-form spherical initialization that maps the frozen teacher's RMS-normalized token embedding table $E_{\text{teacher}} = \operatorname{RMSNorm}(W_{\text{teacher}}) \in \mathbb{R}^{V \times 768}$ directly onto the 3D quantum wavepacket manifold $(\sqrt{d_{\text{sphere}}}\,\mathbb{S}^{d_{\text{sphere}}-1})^{M_{\text{slices}}}$ (and initializes the Stiefel boundary bridges $W_{\text{in,bridge}}, W_{\text{out,bridge}} \in \mathrm{St}_{\text{col}}(\max(D_{\text{emb}}, 768), \min(D_{\text{emb}}, 768))$ via economy SVD / isometric embedding when $D_{\text{emb}} \ne 768$).
2. **Axis F — Layer-0 Value-Embedding Bank Coupling (`QuantumValueEmbedCouplingMode`)**:
   - In `CompiledLeaderboardClassicalGPT`, Layer 0 (`blocks[0]`) does not form its value vectors $V^{(0)}$ solely from `x_0`; it blends `v = s_0 * v + s_1 * ve_g * value_embeds[0](idx)`. In Plan 065, Quantum Layer 0 ignored `classical.value_embeds[0]`, depriving the quantum outer-product density operator $\rho_{s,h}^{\text{tok}} = \sum_{r=1}^R |\tilde{v}_{s,h,r}\rangle\langle\tilde{v}_{s,h,r}|$ of the frozen teacher's lexical value bank.
   - We introduce **`FROZEN_TEACHER_VE0_COUPLED`**: projects `classical.value_embeds[0](idx)` into Quantum Layer 0's wavepacket space `(B, T, M_pkt, d_pkt)` via the shared input Stiefel bridge and couples it into `v_wp` via a learnable 2-scalar gate `ve0_lambdas` prior to forming the $16 \times 16$ (or $2 \times 2$ Bloch) outer-product density matrices.
3. **Axis G — Multi-Stage Recovery Curriculum & Sub-300M Scaling (`QuantumRecoveryScheduleMode`)**:
   - Compares pure end-to-end cross-entropy (`PURE_CE`) against a two-stage **`INTERFACE_WARMUP_TO_PURE_CE`** schedule and **`INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE`** (which aligns both the Layer-0 boundary states `(x_0, x_1, v_0)` and the final pre-readout feature tensor `x_final` against the frozen classical teacher during the warmup window, then releases the quantum front-end to pure CE so it can exceed the teacher).

---

## 2. Mathematical & Code Specifications for Axes E, F, and G

### 2.1 Axis E — Token Embedding Subspace Alignment (`QuantumWteInitMode`)

We define enum `QuantumWteInitMode(str, Enum)` in [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py):
- `PPMI_SVD_SPHERE = "ppmi_svd_sphere"` (Plan 065 default: initializes `q_wte` from `.cache/ppmi_svd_50304_768_shard1.pt`).
- `TEACHER_PROCRUSTES_SPHERE = "teacher_procrustes_sphere"`:
  Given the frozen classical teacher's embedding weight $W_{\text{teacher}} = \text{classical.wte.weight}.\text{detach}().\text{float}() \in \mathbb{R}^{V \times 768}$, define the RMS-normalized teacher table:
  $$E_{\text{teacher}} = \operatorname{RMSNorm}(W_{\text{teacher}}, (768,), \epsilon=10^{-6}) \in \mathbb{R}^{V \times 768}$$
  Then, depending on $D_{\text{emb}} = M_{\text{slices}} \cdot d_{\text{sphere}}$:
  1. **Case $D_{\text{emb}} = 768$ (`SHAPE_24x2x16`, `q_wte` shape `(V, 48, 16)`)**:
     - Reshape $E_{\text{teacher}}$ to `(V, 48, 16)` and project every 16D slice onto $\sqrt{16}\,\mathbb{S}^{15}$ via `project_rows_to_rms_sphere_(q_wte)`.
     - Set `w_in_pkt = I_{16}`, `w_in_mode = I_{48}`, and `wp_causal_re = zeros`, `wp_causal_im = zeros` at initialization so that at Step 0 (`t = 0`), `x_0` is immediately aligned with $E_{\text{teacher}}$ (`cos_sim(x_0, x_0_teacher) > 0.96` at Step 0) while `wp_causal_re/im` remain `requires_grad=True` to learn causal wavepacket dispersion during training.
  2. **Case $D_{\text{emb}} = 512 < 768$ (`SHAPE_1x2x256`, `q_wte` shape `(V, 2, 256)`)**:
     - Compute the economy SVD of $E_{\text{teacher}} \in \mathbb{R}^{V \times 768}$ on CPU in `float32`:
       $$U, \Sigma, V_{\text{svd}}^\top = \operatorname{svd\_lowrank\_or\_full}(E_{\text{teacher}})$$
       Extract the top-$512$ right singular vectors $W_{\star} = V_{\text{svd}}[:, :512] \in \mathrm{St}_{\text{col}}(768, 512)$ (`shape (768, 512)`, $W_{\star}^\top W_{\star} = I_{512}$).
     - Initialize `w_in_bridge.copy_(W_star)` and `w_out_bridge.copy_(W_star)`.
     - Project $E_{\text{teacher}}$ into $\mathbb{R}^{V \times 512}$ via $Z_{512} = E_{\text{teacher}} W_{\star} \in \mathbb{R}^{V \times 512}$, reshape to `(V, 2, 256)`, and apply `project_rows_to_rms_sphere_(q_wte)`.
  3. **Case $D_{\text{emb}} = 1024 > 768$ (`SHAPE_1x2x512` `(V, 2, 512)` or `SHAPE_256x2x2` `(V, 256, 4)`)**:
     - Initialize `w_in_bridge` and `w_out_bridge` of shape `(1024, 768)` as identical column-orthonormal Stiefel matrices $W_{\star} \in \mathrm{St}_{\text{col}}(1024, 768)$ ($W_{\star}^\top W_{\star} = I_{768}$).
     - Lift $E_{\text{teacher}} \in \mathbb{R}^{V \times 768}$ isometrically into $\mathbb{R}^{V \times 1024}$ via:
       $$Z_{1024} = E_{\text{teacher}} W_{\star}^\top \in \mathbb{R}^{V \times 1024}$$
       Notice that $Z_{1024} W_{\star} = E_{\text{teacher}} (W_{\star}^\top W_{\star}) = E_{\text{teacher}}$ identically!
     - Reshape $Z_{1024}$ to `(V, M_slices, d_sphere)` (`(V, 2, 512)` or `(V, 256, 4)`) and apply `project_rows_to_rms_sphere_(q_wte)`.
  4. **Zero Initial Causal Dispersion Across All Shapes**: For all $D_{\text{emb}} \in \{512, 768, 1024\}$ under `TEACHER_PROCRUSTES_SPHERE`, initialize `wp_causal_re.zero_()` and `wp_causal_im.zero_()` (`requires_grad=True`) so multi-lag wavepacket dispersion does not perturb Step-0 subspace alignment.

---

### 2.2 Axis F — Layer-0 Value-Embedding Bank Coupling (`QuantumValueEmbedCouplingMode`)

We define enum `QuantumValueEmbedCouplingMode(str, Enum)`:
- `NONE = "none"` (Plan 065 default).
- `FROZEN_TEACHER_VE0_COUPLED = "frozen_teacher_ve0_coupled"`:
  - When `classical.value_embeds is not None and len(classical.value_embeds) > 0`:
    1. In `FrozenTrunkQuantumFrontEndGPT.__init__`, register a trainable 2-element scalar gate `self.ve0_lambdas = nn.Parameter(torch.tensor([1.0, 0.25], dtype=torch.float32))` (included in `quantum_scalar_parameters()` and clamped to `[0.10, 2.00]` for index 0 and `[-1.00, 1.00]` for index 1 in `restrict_quantum_scalars_()`).
    2. In `encode_quantum_layer0(self, idx)`:
       With `torch.no_grad()`, fetch the frozen teacher's Layer-0 value embedding $V_{\text{bank},0} = \text{classical.value_embeds}[0](\text{idx}).\text{to}(\text{work\_dtype}) \in \mathbb{R}^{B \times T \times 768}$.
       Map $V_{\text{bank},0}$ into the quantum Layer-0 wavepacket space `(B, T, pkt_count, pkt_dim)`:
       - If $D_{\text{emb}} = 768$ (`w_in_bridge is None`): reshape $V_{\text{bank},0}$ to `(B, T, pkt_count, pkt_dim)`.
       - If $D_{\text{emb}} = 512$ (`w_in_bridge` shape `(768, 512)`): compute $(V_{\text{bank},0} W_{\text{in,bridge}}).\text{view}(B, T, \text{pkt\_count}, \text{pkt\_dim})$.
       - If $D_{\text{emb}} = 1024$ (`w_in_bridge` shape `(1024, 768)`): compute $(V_{\text{bank},0} W_{\text{in,bridge}}^\top).\text{view}(B, T, \text{pkt\_count}, \text{pkt\_dim})$.
       Normalize per packet: $\tilde{V}_{\text{bank},0} = \operatorname{RMSNorm}(V_{\text{bank},0}^{\text{wp}}, (\text{pkt\_dim},), \epsilon=10^{-6})$.
    3. Pass `ve_wp = self.ve0_lambdas[1].to(work_dtype) * \tilde{V}_{\text{bank},0}` and `v_scale = self.ve0_lambdas[0].to(work_dtype)` into `quantum_block0(x_0_wp, x_0=x_0_wp, ve_wp=ve_wp, v_scale=v_scale)`, where inside `quantum_block0.forward`:
       $$v_{\text{wp}} \leftarrow v_{\text{scale}} \cdot v_{\text{wp}} + \text{ve\_wp}$$
       immediately before forming the outer-product density operator $\rho_{s,h}^{\text{tok}}$ and value kets!
    4. When `ve_embed_coupling_mode == QuantumValueEmbedCouplingMode.NONE`, `self.ve0_lambdas` is `None` (zero dead parameters).

---

### 2.3 Axis G — Multi-Stage Recovery Curriculum (`QuantumRecoveryScheduleMode`)

We define enum `QuantumRecoveryScheduleMode(str, Enum)`:
- `PURE_CE = "pure_ce"`: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}$ for all steps $t \ge 0$.
- `INTERFACE_WARMUP_TO_PURE_CE = "interface_warmup_to_pure_ce"`:
  For $0 \le t < T_{\text{align}}$ (`interface_warmup_steps > 0`):
  $$\lambda_{\text{align}}(t) = \frac{\lambda_{\text{align},0}}{2}\left(1 + \cos\left(\frac{\pi t}{T_{\text{align}}}\right)\right)$$
  $$\mathcal{L}_{\text{total}}(t) = \mathcal{L}_{\text{CE}} + \lambda_{\text{align}}(t)\left(\frac{1}{B T D}\left(\|x_0 - x_0^{\text{teacher}}\|_F^2 + \|x_1 - x_1^{\text{teacher}}\|_F^2 + 0.25\,\|v_0 - v_0^{\text{teacher}}\|_F^2\right)\right)$$
  For $t \ge T_{\text{align}}$, $\lambda_{\text{align}}(t) = 0$ and $\mathcal{L}_{\text{total}}(t) = \mathcal{L}_{\text{CE}}$.
- `INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE = "interface_and_final_feat_cosine_to_ce"`:
  During the warmup window $0 \le t < T_{\text{align}}$, in addition to Layer-0 interface alignment $\mathcal{L}_{\text{interface}}(x_0, x_1, v_0)$, computes the detached final RMSNormed feature tensor of the frozen classical teacher $x_{\text{final}}^{\text{teacher}} = \text{classical.forward\_features}(\text{idx}).\text{detach}() \in \mathbb{R}^{B \times T \times 768}$ and adds a final-feature cosine/MSE alignment term:
  $$\mathcal{L}_{\text{final\_feat}} = \operatorname{MSE}\!\left(x_{\text{final}}^{\text{float}},\, x_{\text{final}}^{\text{teacher, float}}\right)$$
  $$\mathcal{L}_{\text{total}}(t) = \mathcal{L}_{\text{CE}} + \lambda_{\text{align}}(t)\left(\mathcal{L}_{\text{interface}} + 0.5\,\mathcal{L}_{\text{final\_feat}}\right)$$
  Why `INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE` is memory-safe and fast: unlike full vocabulary KL distillation (which requires materializing a `(B, T, 50304)` teacher logit tensor), matching $x_{\text{final}} \in \mathbb{R}^{B \times T \times 768}$ before the frozen shared `lm_head` aligns the exact sufficient statistic of the teacher's output distribution in $768$ dimensions while preserving the fused chunked softcap CE kernel (`_CompiledChunkedSoftcapLinearCEFn`)! After step $T_{\text{align}}$, $\lambda_{\text{align}}(t) = 0$ so the quantum front-end optimizes pure token CE without teacher-ceiling constraints.

---

## 3. Experimental Design & Sub-300M-Token Scaling Matrix

### 3.1 Phase 1 — Screening Axes E, F, and G (`500` Steps = `4.10M` Tokens on the `300M` Teacher)

All Phase-1 arms load [`ckpt_step01145_0300M_m0300M.pt`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/ckpt_step01145_0300M_m0300M.pt) (`val_ce = 3.5842`, `top1_acc = 36.88%`) and run for `500` steps (`B=8, T=1024, grad_accum=1`, `4.10M` tokens, evaluated every `100` steps on `65,536` validation tokens):

| Arm ID | Wavefunction Shape | `wte_init_mode` (Axis E) | `ve_coupling` (Axis F) | `recovery_schedule` (Axis G) | Primary Hypothesis Tested |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`R1_TEACHER_SPHERE_24x2x16`** | `SHAPE_24x2x16` (`48x16`) | `TEACHER_PROCRUSTES_SPHERE` | `NONE` | `PURE_CE` | Does initializing `q_wte` from spherical-projected $E_{\text{teacher}}$ eliminate the Step-0 coordinate mismatch vs. `T1` (`PPMI_SVD_SPHERE`)? |
| **`R2_TEACHER_SPHERE_VE0_24x2x16`** | `SHAPE_24x2x16` (`48x16`) | `TEACHER_PROCRUSTES_SPHERE` | `FROZEN_TEACHER_VE0_COUPLED` | `PURE_CE` | Does coupling `value_embeds[0]` into Quantum Layer 0's outer-product $V$-space accelerate recovery? |
| **`R3_TEACHER_SPHERE_VE0_KEEP_MLP0`** | `SHAPE_24x2x16` (`48x16`) | `TEACHER_PROCRUSTES_SPHERE` | `FROZEN_TEACHER_VE0_COUPLED` | `PURE_CE` (`ATTN0_QUANTUM_KEEP_FROZEN_MLP0`) | Does keeping frozen `blocks[0].mlp` + `TEACHER_PROCRUSTES_SPHERE` + `VE0` coupling yield the fastest drop in `val_ce`? |
| **`R4_TEACHER_SPHERE_VE0_2STAGE`** | `SHAPE_24x2x16` (`48x16`) | `TEACHER_PROCRUSTES_SPHERE` | `FROZEN_TEACHER_VE0_COUPLED` | `INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE` (`T_align=150`) | Does two-stage `(x_0, x_1, v_0, x_final)` teacher alignment followed by pure CE outperform pure CE? |
| **`R5_BEST_P065_SHAPE_UPGRADED`** | Best shape from Plan 065 (`1x2x256`, `1x2x512`, or `256x2x2`) | `TEACHER_PROCRUSTES_SPHERE` | `FROZEN_TEACHER_VE0_COUPLED` | `INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE` (`T_align=150`) | Combines the winning non-baseline shape/interior operator from Plan 065 with Axes E, F, and G. |

### 3.2 Phase 2 — Long-Horizon Sub-`300M`-Token Recovery Runs (`cuda:0` and `cuda:1`)

Using the top-performing configuration from Phase 1 + Plan 065:
1. **Run `S1_RECOVERY_VS_50M_TEACHER` (`cuda:0`, `3,000` steps = `24.58M` tokens, `< 50%` of teacher's `50M` budget)**:
   - Grafts the winning quantum front-end onto the **`50M`-token Classical Teacher** ([`ckpt_step00191_0050M_m0050M.pt`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/ckpt_step00191_0050M_m0050M.pt), reference `val_ce = 4.2466`, `top1_acc = 31.16%`).
   - Logs validation cross-entropy and Top-1/Top-5 accuracy every `250` steps (`2.05M` tokens) up to `24.58M` tokens to measure the exact token count at which the quantum front-end reproduces (`<= 4.2466`) and exceeds (`< 4.2466`) the `50M`-token classical teacher.
2. **Run `S2_RECOVERY_VS_300M_TEACHER` (`cuda:1`, `6,000` steps at `grad_accum=2` = `98.30M` tokens, `< 33%` of teacher's `300M` budget)**:
   - Grafts the winning quantum front-end onto the **`300M`-token Classical Teacher** ([`ckpt_step01145_0300M_m0300M.pt`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/ckpt_step01145_0300M_m0300M.pt), reference `val_ce = 3.5842`, `top1_acc = 36.88%`).
   - Logs the full recovery curve at `4.1M`, `8.2M`, `16.4M`, `32.8M`, `49.2M`, `65.5M`, `81.9M`, and `98.3M` tokens (`< 100M` tokens, well under the `300M`-token ceiling).

---

## 4. Verification Gates (`tests/test_plan066_quantum_frontend_recovery.py`)

Before launching Phase 1 or Phase 2 GPU runs, [`tests/test_plan066_quantum_frontend_recovery.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan066_quantum_frontend_recovery.py) must pass all 4 unit verification gates:

1. **Gate `H1` — `TEACHER_PROCRUSTES_SPHERE` Exact Manifold & High Step-0 Cosine Alignment**:
   - Verifies that across all 4 shapes (`SHAPE_24x2x16`, `SHAPE_1x2x256`, `SHAPE_1x2x512`, `SHAPE_256x2x2`), `QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE` initializes 3D `q_wte` strictly on $(\sqrt{d_{\text{sphere}}}\,\mathbb{S}^{d_{\text{sphere}}-1})^{M_{\text{slices}}}$ (`q_wte_sphere_err < 1e-5`) and Stiefel bridges on $\mathrm{St}_{\text{col}}$ (`q_max_stiefel_err < 1e-5`).
   - Verifies that for `SHAPE_24x2x16` (`D_emb = 768`) and `SHAPE_1x2x512` / `SHAPE_256x2x2` (`D_emb = 1024`), the Step-0 cosine similarity between quantum `x_0` and teacher `x_0_c` (when `smear_lambda == 0`) exceeds `0.90` (compared to `< 0.05` for `PPMI_SVD_SPHERE`).
2. **Gate `H2` — `FROZEN_TEACHER_VE0_COUPLED` Gradient Flow & Zero-Dead-Parameter Invariant**:
   - Verifies that when `ve_coupling_mode == QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED`, `model.ve0_lambdas` receives finite non-zero gradients (`grad.abs().max() > 0`) while `classical.value_embeds[0].weight.grad is None`.
   - Verifies that when `ve_coupling_mode == QuantumValueEmbedCouplingMode.NONE`, `model.ve0_lambdas is None`.
3. **Gate `H3` — `INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE` Schedule Transition**:
   - Verifies that for steps $t < T_{\text{align}}$, `forward` includes both $\mathcal{L}_{\text{interface}}$ and $0.5\,\mathcal{L}_{\text{final\_feat}}$ scaled by $\lambda_{\text{align}}(t)$, and that for $t \ge T_{\text{align}}$, the loss equals the exact pure CE loss (`< 1e-6` diff).
4. **Gate `H4` — Rapid Multi-Step Loss Descent with `TEACHER_PROCRUSTES_SPHERE` + `FROZEN_TEACHER_VE0_COUPLED`**:
   - Verifies that over 10 optimization steps on a deterministic batch, the combined Plan 066 graft achieves strictly lower initial loss at Step 0 and strictly lower final loss at Step 10 than the unaligned `PPMI_SVD_SPHERE` baseline while maintaining `q_max_stiefel_err < 1e-5` and `q_wte_sphere_err < 1e-5`.

---

## 5. Compliance with `docs/SCRIPT_BEST_PRACTICES.md`

- All new enums (`QuantumWteInitMode`, `QuantumValueEmbedCouplingMode`, `QuantumRecoveryScheduleMode`) inherit from `(str, Enum)` (Rule 3).
- All file sizes remain strictly `<= 1,000` SLOC soft limit and all functions take `<= 8` arguments (`ast_signature_and_length_linter.py` enforced).
- All constants (`DEFAULT_VE0_LAMBDA_INIT`, `DEFAULT_FINAL_FEAT_ALIGN_SCALE`) are declared at the top of the module in `UPPER_SNAKE_CASE` (Rule 7).
