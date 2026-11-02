# EXPERIMENT PLAN 074: Document-Boundary Sliding-Window Quantum Wavepacket Attention (`Doc-SWC`) & Depth-Scaled $1/\sqrt{N_{\text{sub}}}$ Geodesic Trust-Region Optimization Across All 12 Layers (`Wave 53B`)

**Date:** 2026-11-01  
**Wave:** `Wave 53B` (`EXPERIMENT_PLAN_074.md`)  
**Status:** Proposed for Independent Implementation-Readiness Review  
**Predecessor Waves:** `Wave 53A` (`EXPERIMENT_PLAN_073.md`, progressive and all-12-layer quantum wavepacket replacement `I73_1..I73_8`) and `Waves 52A–52H` (`Plans 065–072`).

---

## 1. Executive Summary & Root-Cause Findings from `Wave 53A` (`EXPERIMENT_PLAN_073.md`)

In `Wave 53A` (`EXPERIMENT_PLAN_073.md`), we tested replacing progressive depths (`L0..L2`, `L0..L5`) and the entire 12-layer backbone (`L0..L11` attention and `L0..L11` full blocks with Mach-Zehnder MLP interferometers) of the `300M`-token Classical Teacher (`val_ce = 3.5845`, `top1 = 37.31%`, `top5 = 58.26%`). While all runs maintained strict Riemannian manifold health (`q_max_stiefel_err < 1e-6`, `q_wte_sphere_err < 1e-6`, `q_rank = 2.75..2.96`) and converged to `val_ce = 3.6101` (`L0..L2`), `3.6359` (`L0..L5`), `3.6375` (`All 12 Attn R768`), and `3.6816` (`All 12 Attn R384`), two exact mechanisms prevented the multi-layer and full-backbone quantum models from beating the `3.5845` Classical Teacher:

### 1.1. Root Cause 1: Missing Document-Boundary + Long/Short Sliding-Window Causal Masking (`Doc-SWC`)
- In the pretrained `300M` Classical Teacher checkpoint (`ckpt_step01145_0300M_m0300M.pt`), `current_ws_blocks = 6` (`DEFAULT_FLEX_BLOCK_SIZE = 128`), which enforces:
  1. **Document-Boundary Isolation (`docs[b, q] == docs[b, kv]`)**: Using `docs = (idx == 50256).cumsum(dim=1)`, tokens in a new document never attend across `<BOS>` (`50256`) to prior documents packed into the same `1024`-token context window.
  2. **Long vs. Short Sliding-Window Horizons**:
     - Long-window layers $\ell \in \{0, 4\}$ use `long_window_buf = 768` tokens, and final Layer `11` uses `final_window_buf = 768` tokens.
     - Short-window interior layers $\ell \in \{1, 2, 3, 5, 6, 8, 9, 10\}$ use `short_window_buf = 384` tokens.
- In `Wave 53A` (`Plan 073`), `_run_entropy_preserving_coherent_ket_attention` called `F.scaled_dot_product_attention(..., is_causal=True)` without the document-boundary or sliding-window mask, causing all 11 quantum attention layers (especially the 8 short-window layers) to attend across the full `1024` tokens and leak across `<BOS>` document boundaries. Furthermore, setting `use_flex_doc_swc = False` to avoid uncompiled `flex_attention` also removed `Doc-SWC` from the remaining classical layers in `I73_1` and `I73_2`.
- **Direct Diagnostic Verification on `cuda:0`**:
  Passing the exact `(B, 1, T, T)` boolean `Doc-SWC` mask (`(q >= kv) & (docs[:, q] == docs[:, kv]) & ((q - kv) < window_len)`) into `F.scaled_dot_product_attention` alongside depth-calibrated $\beta_{\rho,\text{init}} = 0.05 / \sqrt{N_{\text{attn}}}$ (`0.015` for 11 layers) immediately drops Step-0 validation CE of the **All-12-Layer Quantum Wavepacket Attention model (`EMB_PLUS_ALL_12_ATTN`, `R768`)** from `3.6545` down to **`3.5920` (`top1 = 37.18%`, `top5 = 58.14%`)**—within **`0.0075` nats** of the `300M` Classical Teacher (`3.5845`) at Step 0 across all 12 layers!

### 1.2. Root Cause 2: Simultaneous Multi-Layer Stiefel Geodesic Step-Size Overshoot
- In `Wave 53A`, all arms used a constant learning rate (`lr_mult = 1.0`, `stiefel_lr = 0.008`) for the first `70%` of training (`steps 1..560`) and only decayed `lr_mult` during the final `30%` (`steps 560..800`).
- Because `ModularRiemannianStiefelKrausMuon` normalizes the tangent velocity $U_{\text{tan}}^{(\ell)}$ of every Stiefel matrix independently to unit RMS, stepping $N_{\text{sub}} = 11$ attention layers (`88` Stiefel matrices) or $N_{\text{sub}} = 23$ full-block sublayers (`136` Stiefel matrices) simultaneously with `stiefel_lr = 0.008` at Step 1 induces a cumulative trajectory perturbation $\propto \sqrt{N_{\text{sub}}}\,\eta_{\text{stiefel}}$.
- As a result, in `I73_3` (`All 12 Attn R768`), `val_ce` rose during the high-LR plateau (`3.6545` at Step 0 $\to$ `3.6971` at Step 200 $\to$ `3.7238` at Step 600), and then plunged by **`-0.0863` nats (`3.7238` $\to$ `3.6375`)** during Steps `600..800` as soon as `lr_mult` decayed toward `0.15` (`eff_stiefel_lr = 0.0012`)! The exact same late plunge occurred in `I73_4` (`-0.1069` nats in Steps `600..800`) and `I73_5` (`-0.1176` nats in Steps `600..800`).

---

## 2. Exact Mathematical & Architectural Solutions (`Wave 53B`)

### 2.1. Axis U — Fused Document-Boundary + Sliding-Window Causal Masking (`QuantumDocSWCMode`)
1. **Strategy Enum (`QuantumDocSWCMode`)**:
   - `FULL_CAUSAL_1024 = "full_causal_1024"` (Unbounded `is_causal=True` baseline from `Plan 073`)
   - `TEACHER_DOC_SWC_MASK = "teacher_doc_swc_mask"` (Exact Document-Boundary + Long/Short/Final Sliding-Window Causal mask)
2. **Exact Mask Construction (`build_doc_swc_sdpa_masks`)**:
   Given token indices `idx` of shape `(B, T)` and window lengths `(long_win, short_win, final_win)` read directly from the teacher buffers `(long_window_buf, short_window_buf, final_window_buf)`:
   $$\text{docs}_{b, t} = \sum_{s=0}^{t} \mathbb{I}[\text{idx}_{b, s} = 50256], \qquad \Delta_{q, k} = q - k$$
   $$\mathcal{M}_{b, 1, q, k}^{(W)} = (\Delta_{q, k} \ge 0) \;\wedge\; (\text{docs}_{b, q} = \text{docs}_{b, k}) \;\wedge\; (\Delta_{q, k} < W)$$
   producing three boolean tensors `(mask_long, mask_short, mask_final)` of shape `(B, 1, T, T)` (`8 MB` total at `B=8, T=1024`), which broadcast across any number of SDPA heads `(B, H_sdpa, T, d)`.
3. **Layerwise Dispatch Across Both Quantum and Unreplaced Classical Layers (`<= 8` Args)**:
   - Each `SVDStiefelWavepacketStatMechBlock` stores `self.layer_idx: int`, `self.is_long_window_layer: bool`, and `self.ket_amplitude_mode: str = "entropy_preserving_coherent_ket"` (calibrated from `teacher_blk` and `graft_cfg.ket_amplitude_mode.value`).
   - In `QuantumFullBackboneGraftGPT.encode_all_layers`, layer $\ell$ selects:
     $$\mathcal{M}^{(\ell)} = \begin{cases} \text{mask\_final}, & \ell = L - 1 \\ \text{mask\_long}, & \ell < L - 1 \text{ and is\_long\_window\_layer}(\ell) \\ \text{mask\_short}, & \text{otherwise} \end{cases}$$
   - `execute_quantum_wavepacket_attn_layer(svd_blk, x_768, x_0_768, cos_4d, sin_4d, q_scale, ve_or_v0_wp=None, attn_mask=None)` (**8 parameters**, reading `svd_blk.ket_amplitude_mode`) passes `attn_mask` to `_run_entropy_preserving_coherent_ket_attention(svd_blk, v_wp, v_mixed, q_rot, k_rot, attn_mask=attn_mask)` (**6 parameters**), which calls `F.scaled_dot_product_attention(sdpa_q, sdpa_k, sdpa_v, attn_mask=attn_mask, is_causal=(attn_mask is None), scale=svd_blk.sdpa_scale)`.
   - For unreplaced classical blocks (in partial-depth arms `L0..L2` and `L0..L5`) and in teacher trajectory extraction (`_extract_teacher_doc_swc_trajectories`), `QuantumFullBackboneGraftGPT._run_classical_block_sdpa(self, t_blk, x, x_0, cos_4d, sin_4d, v_0, ve_l, attn_mask=None)` (**8 non-`self` parameters**, reading `q_scale = self.classical.rope.yarn_q_scale`) executes the classical block attention with `F.scaled_dot_product_attention(q_bhtd, k_bhtd, v_bhtd, attn_mask=attn_mask, is_causal=(attn_mask is None), scale=t_blk.attn_scale)` without invoking uncompiled `flex_attention`.

### 2.2. Axis V — Depth-Scaled $1/\sqrt{N_{\text{sub}}}$ Geodesic Trust Region & Linear-Warmup Cosine Schedule (`QuantumDepthLRMode`)
1. **Strategy Enum (`QuantumDepthLRMode`)**:
   - `UNIFORM_FLAT_THEN_COSINE = "uniform_flat_then_cosine"` (Legacy `Plan 073` schedule: flat `1.0` for `70%`, cosine to `0.15`)
   - `DEPTH_SQRT_WARMUP_COSINE = "depth_sqrt_warmup_cosine"` (Depth-scaled base LR + `10%` linear warmup + `90%` cosine decay to `0.10`)
2. **Exact Depth Scaling Law & Closed-Form Warmup-Cosine Schedule (`quantum_svd_stiefel_graft.py`)**:
   - To keep both modules well below the `1,000` SLOC limit, `build_doc_swc_sdpa_masks`, `compute_depth_lr_scale`, and `compute_quantum_backbone_lr_multiplier` are implemented in [`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py) and re-exported in [`src/qdlblocks/models/quantum_full_backbone_graft_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/quantum_full_backbone_graft_gpt.py).
   - Let $N_{\text{sub}} = |\mathcal{L}_{\text{attn}}| + |\mathcal{L}_{\text{mlp}}|$ be the number of replaced quantum sublayers, and define the depth trust-region factor:
     $$\gamma_{\text{depth}}(N_{\text{sub}}) = \frac{1}{\sqrt{\max(1.0,\; N_{\text{sub}} / 2.0)}}$$
     - For $N_{\text{sub}} = 3$ (`L0..L2` Attn): $\gamma_{\text{depth}}(3) = \sqrt{2/3} \approx 0.8165$.
     - For $N_{\text{sub}} = 6$ (`L0..L5` Attn): $\gamma_{\text{depth}}(6) = \sqrt{2/6} \approx 0.5774$.
     - For $N_{\text{sub}} = 11$ (`All 12` Attn): $\gamma_{\text{depth}}(11) = \sqrt{2/11} \approx 0.4264$.
     - For $N_{\text{sub}} = 23$ (`All 12` Full Blocks Attn + MZ-MLP): $\gamma_{\text{depth}}(23) = \sqrt{2/23} \approx 0.2949$.
   - When `depth_lr_mode == QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE`, `build_quantum_full_backbone_optimizers` scales `stiefel_lr`, `unitary_lr`, `wte_lr`, and `scalar_lr` by $\gamma_{\text{depth}}(N_{\text{sub}})$ (from reference 2-layer base rates `stiefel_lr=0.006, unitary_lr=0.010, wte_lr=0.003, scalar_lr=0.005`), and `compute_quantum_backbone_lr_multiplier(step: int, total_steps: int, depth_lr_mode: QuantumDepthLRMode) -> float` computes:
     - With $T_{\text{warm}} = \max(1, \lfloor 0.10 \cdot T_{\text{total}} \rfloor)$, for $1 \le \text{step} \le T_{\text{warm}}$:
       $$\text{lr\_mult}(\text{step}) = \begin{cases} 1.0, & T_{\text{warm}} = 1 \\ 0.10 + 0.90 \cdot \frac{\text{step} - 1}{T_{\text{warm}} - 1}, & T_{\text{warm}} > 1 \end{cases}$$
     - For $T_{\text{warm}} < \text{step} \le T_{\text{total}}$:
       $$\text{lr\_mult}(\text{step}) = 0.10 + 0.90 \cdot \frac{1}{2}\left(1 + \cos\!\left(\pi \frac{\text{step} - T_{\text{warm}}}{\max(1, T_{\text{total}} - T_{\text{warm}})}\right)\right)$$
3. **Depth-Calibrated Step-0 Quantum Couplings**:
   - `DEFAULT_BASE_RHO_COUPLING_INIT = 0.05` so $\beta_{\rho,\text{init}} = 0.05 / \sqrt{N_{\text{attn}}}$ (`0.015` for 11 layers, clamped to `[0.002, 1.00]`).
   - `DEFAULT_MZ_THETA_INIT = 0.015` and `DEFAULT_MZ_GAMMA_INIT = 0.20` so `SVDStiefelMachZehnderInterferometer` starts with `> 0.9999` Step-0 cosine similarity across all 12 layers while preserving strictly non-zero quadrature gradients $\nabla_{\theta_{\text{mz}}}\mathcal{L} \neq 0, \nabla_{\gamma_{\text{mz}}}\mathcal{L} \neq 0$ from Step 1.

---

## 3. Eight-Arm Experimental Matrix (`J74_1 .. J74_8`)

All 8 arms use `(V, 12, 2, 32)`, `WF_IN_WF_OUT`, `ENTROPY_PRESERVING_COHERENT_KET`, `TEACHER_DOC_SWC_MASK`, `DEPTH_SQRT_WARMUP_COSINE`, `BERRY_PHASE_COMPLEX_QK`, `FUBINI_STUDY_SELF_ORTHO`, and `ATTENDED_RHO_ON_LOCAL_V` on the `300M` Classical Teacher (`val_ce = 3.5845`, `top1 = 37.31%`, `top5 = 58.26%`), trained for **`800` steps (`6.55M` FineWeb tokens = `2.18%` of `300M`, evaluating every `200` steps)**:

| Arm ID | Depth Scope (`Replaced Layers`) | Attention Rank (`Q-Attn Params` vs. `25.95M` Classical) | MLP Mode (`Q-MLP Params` vs. `56.62M` Classical) | Scientific Question Tested |
| :--- | :--- | :---: | :---: | :--- |
| **`J74_1_DEPTH3_L0_TO_L2_DOC_SWC_R384`** | `EMB_PLUS_L0_TO_L2_ATTN` (`wte + L0..L2`, `25%` depth) | **`R384` (`7.12M` = `1.00x` Parity)** | `FROZEN_CLASSICAL_MLP` | Does `wte + L0..L2` (`3` layers) at `1:1` parity beat `3.5845` with `Doc-SWC` + depth-scaled LR? |
| **`J74_2_DEPTH6_ENCODER_L0_TO_L5_DOC_SWC_R384`** | `EMB_PLUS_L0_TO_L5_ATTN` (`wte + L0..L5`, `50%` encoder half) | **`R384` (`14.24M` = `1.00x` Parity)** | `FROZEN_CLASSICAL_MLP` | Does replacing the **entire 6-layer U-Net encoder (`L0..L5`)** at `1:1` parity beat `3.5845`? |
| **`J74_3_ALL12_ATTN_DOC_SWC_R768`** | **`EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11`)** | `R768` (`52.08M` across 11 attn layers) | `FROZEN_CLASSICAL_MLP` | Does replacing `wte` + **all attention layers (`L0..L11`)** at full rank beat the `300M` Teacher (`3.5845`)? |
| **`J74_4_ALL12_ATTN_DOC_SWC_R384_PARITY`** | **`EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11`)** | **`R384` (`26.11M` vs `25.95M` = `1.00x` Parity!)** | `FROZEN_CLASSICAL_MLP` | Does replacing `wte` + **all 12 layers' attention at exact `1:1` parameter parity** beat `3.5845`? |
| **`J74_5_ALL12_ATTN_DOC_SWC_R192_2X`** | **`EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11`)** | **`R192` (`13.13M` vs `25.95M` = `0.50x` `2x` Smaller!)** | `FROZEN_CLASSICAL_MLP` | Can `2x`-compressed quantum attention across all 12 layers (`13.13M` vs `25.95M`) match/beat `3.5845`? |
| **`J74_6_ALL12_ATTN_DOC_SWC_R128_3X`** | **`EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11`)** | **`R128` (`8.80M` vs `25.95M` = `0.34x` `3x` Smaller!)** | `FROZEN_CLASSICAL_MLP` | How close does `3x`-compressed quantum attention across all 12 layers (`8.80M` vs `25.95M`) get to `3.5845`? |
| **`J74_7_ALL12_FULL_BLOCKS_MZ_DOC_SWC_R768`** | **`EMB_PLUS_ALL_12_FULL_BLOCKS_MZ` (`100%` Backbone `wte + ve + L0..L11 Attn & MZ-MLP`)** | `R768` (`52.08M` Q-Attn) | **`SVD_STIEFEL_MACH_ZEHNDER` (`R768`, `70.80M` Q-MLP)** | **100% Full-Backbone Quantum Replacement**: does a 100% quantum 12-layer network beat `3.5845`? |
| **`J74_8_ALL12_FULL_BLOCKS_MZ_DOC_SWC_R384_COMPRESSED`** | **`EMB_PLUS_ALL_12_FULL_BLOCKS_MZ` (`100%` Backbone `wte + ve + L0..L11 Attn & MZ-MLP`)** | **`R384` (`26.11M` = `1.00x` Q-Attn)** | **`SVD_STIEFEL_MACH_ZEHNDER` (`R384`, `35.40M` = `0.625x` Q-MLP!)** | **Compressed 100% Full-Backbone Quantum Replacement** (`61.51M` vs `82.58M` = `25.5%` smaller trunk across all 12 layers)! |

### 3.1. Dual-GPU Execution Plan
- **GPU 0 (`cuda:0`)**: Runs `J74_1..J74_4` -> `.cache/w53b_plan074_part1_cuda0.json`.
- **GPU 1 (`cuda:1`)**: Runs `J74_5..J74_8` -> `.cache/w53b_plan074_part2_cuda1.json`.

---

## 4. Verification Gates (`tests/test_plan074_doc_swc_and_depth_scaled_full_backbone.py`)

- **Gate I1 (Document-Boundary + Long/Short/Final Sliding-Window Mask Exactness)**:
  - Verifies that `build_doc_swc_sdpa_masks(idx, long_win=8, short_win=4, final_win=8)` blocks attention across `<BOS>` (`50256`) document boundaries (`mask[b, 0, q, k] == False` whenever `docs[b, q] != docs[b, k]`), enforces causality (`q >= k`), and enforces window horizons `q - k < W`.
- **Gate I2 (Depth-Scaled $1/\sqrt{N_{\text{sub}}/2}$ LR Scaling & Linear-Warmup Cosine Schedule)**:
  - Verifies that `compute_depth_lr_scale(n_sub)` equals $1/\sqrt{\max(1, N_{\text{sub}}/2)}$, and `compute_quantum_backbone_lr_multiplier` warms up linearly from `0.10` to `1.00` over the first `10%` of steps and decays smoothly via cosine to `0.10` at `step == steps`.
- **Gate I3 (End-to-End 12-Layer Forward, Backward, Riemannian Step with `TEACHER_DOC_SWC_MASK`)**:
  - Verifies 2 steps of `QuantumFullBackboneGraftGPT` on CPU with `TEACHER_DOC_SWC_MASK` and `DEPTH_SQRT_WARMUP_COSINE` for both `EMB_PLUS_ALL_12_ATTN` and `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`, checking finite loss, non-zero gradients across all 12 layers, and `< 1e-4` Stiefel/sphere manifold error.
- **Gate I4 (`docs/SCRIPT_BEST_PRACTICES.md` AST Compliance)**:
  - Verifies `<= 1,000` SLOC, `<= 8` arguments per function, strategy enums, top-of-file constants, and zero semicolon statement packing.
