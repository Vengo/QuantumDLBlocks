# EXPERIMENT PLAN 072 — Wave 52H: Sub-`300M` Teacher Superiority Across the Compression Frontier (`1x` Parity $r=384$, `2x` $r=192$, `3x` $r=128$, `6x` $r=64$), Calibrated Local-$V$ & Liouville Commutator Synergy, and Extended Multi-Layer (`Emb + L0 + L1`) Recovery

**Date**: 2026-11-01
**Author**: Antigravity (Autonomous Overnight Session — Wave 52H)
**Predecessor**: [`EXPERIMENT_PLAN_071.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_071.md) (Wave 52G — $(V, 12, 2, 32)$ Synergistic Interior Physics & U(1) Berry-Phase Complex Hermitian QK Coupling)
**Status**: Approved — Reviewed by Independent Context-Free Reviewer (`egm-implementation-readiness`, `0 Blockers, 0 Majors, 0 Nits`)

---

## 1. Executive Summary & Motivation

Across **Waves 52A–52G (`Plans 065–071`, `63` completed GPU runs)**, we discovered:
1. **Wave 52F (`Plan 070`) Breakthrough — `2x` Compressed Quantum Layer 0 Beats the `300M` Classical Teacher**:
   - `F11_SUBPARITY_R192_EXCEED_300M_800STEPS` ($r = 192$, **`1.19M` Layer-0 parameters = $0.50\times$ ($2\times$ smaller than) Classical Layer-0 Attention `2.36M`**) reached **`val_ce = 3.5836` (`-0.0009` nats below `3.5845`), `top1 = 37.44%` (`+0.13%`), `top5 = 58.33%` (`+0.07%`)** in `800` steps (`6.55M` tokens = `2.18%` of `300M`).
   - Meanwhile, at `500` steps (`4.10M` tokens), exact 1:1 parameter parity ($r=384$, `2.37M` params) with **`ATTENDED_RHO_ON_LOCAL_V` (`F7`: `val_ce = 3.5846`, `top1 = 37.40%`)** and **Liouville-von Neumann Commutator `LIOUVILLE_COMMUTATOR_RHO_V` (`F9`: `val_ce = 3.5848`, `top1 = 37.39%`)** outperformed `ATTENDED_RHO_ON_ATTENDED_V` (`F5`: `3.5852`).
2. **Wave 52G (`Plan 071`) Breakthrough — All Five $r=768$ `(V, 12, 2, 32)` Interior Physics Arms Beat the `300M` Teacher in `500` Steps (`4.10M` Tokens)**:
   - `G71_1_PKT32_PURE_WF_IO` (`WF_IN_WF_OUT`): **`val_ce = 3.5825` (`-0.0020` nats), `top1 = 37.36%`, `top5 = 58.29%`**.
   - `G71_4_PKT32_BERRY_PHASE_QK` (`BERRY_PHASE_COMPLEX_QK`): **`val_ce = 3.5830` (`-0.0015` nats), `top1 = 37.37%`, `top5 = 58.38%` (`+0.12%`)**.
   - `G71_2_PKT32_CUBIC_FD` (`CUBIC_FERMI_DIRAC`): **`val_ce = 3.5831` (`-0.0014` nats), `top1 = 37.36%`, `top5 = 58.34%`**.
   - `G71_5_PKT32_SYNERGY_FULL`: **`val_ce = 3.5834` (`-0.0011` nats), `top1 = 37.36%`, `top5 = 58.33%`**.
   - `G71_3_PKT32_KRAUS_RANK4` (`CPTP_KRAUS_RANK_K`, $K=4$): **`val_ce = 3.5838` (`-0.0007` nats), `top1 = 37.37%`, `top5 = 58.36%`**.

**Wave 52H (`EXPERIMENT_PLAN_072.md`)** closes three remaining scientific questions:
1. **Axis T — Exact `800`-Step (`6.55M`-Token) Compression Pareto Frontier (`r = 384` `1x` Parity, `r = 128` `3x` Compression, `r = 64` `6x` Compression)**:
   - In `Plan 071`, `G71_8` used `warmup_steps = 450` over `1,200` steps, which caused a transient post-warmup dip at Step 600–800 compared to the proven `warmup_steps = 300` over `800` steps (`E2`, `E3`).
   - By running `800` steps (`6.55M` tokens = `2.18%` of `300M`) with the proven `warmup_steps = 300` schedule on `(V, 12, 2, 32)` across:
     - **`H72_1_R384_PARITY_LOCAL_V_800STEPS`** ($r=384$, `1.00x` parity `2.37M` params + `ATTENDED_RHO_ON_LOCAL_V`, `F7` winner),
     - **`H72_2_R384_PARITY_LIOUVILLE_COMM_800STEPS`** ($r=384$, `1.00x` parity `2.37M` params + `LIOUVILLE_COMMUTATOR_RHO_V`, `F9` winner),
     - **`H72_3_R128_SUBPARITY_3X_800STEPS`** ($r=128$, `0.34x` = `3x` smaller `0.80M` params + `WF_IN_WF_OUT`),
     - **`H72_4_R64_SUBPARITY_6X_800STEPS`** ($r=64$, `0.17x` = `6x` smaller `0.41M` params + `WF_IN_WF_OUT`),
     we map out the exact `6.55M`-token Pareto frontier from `6x` compression ($r=64$) to `1x` parity ($r=384$) and `2x` rank ($r=768$).
2. **Axis U — Full-Rank ($r=768$) Local-$V$ & Liouville Commutator + Berry-Phase QK Synergy (`500` Steps = `4.10M` Tokens)**:
   - Testing `ATTENDED_RHO_ON_LOCAL_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`H72_5_R768_LOCAL_V_BERRY_WF`) and `LIOUVILLE_COMMUTATOR_RHO_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` (`H72_6_R768_LIOUVILLE_BERRY_WF`) at $r=768$ on `(V, 12, 2, 32)` to see if they surpass `D6` (`3.5823`) and `G71_1` (`3.5825`) at `500` steps!
3. **Axis V — Multi-Layer (`Emb + L0 + L1`) Calibrated Recovery at $r=768$ (`800` Steps = `6.55M` Tokens)**:
   - Testing `EMB_PLUS_L0_L1` with $r=768$ (`H72_7_MULTILAYER_L0_L1_R768_800STEPS`) over `800` steps (`6.55M` tokens) on `(V, 12, 2, 32)` to determine how close a **2-layer quantum replacement (`wte + blocks[0].attn + blocks[1].attn`)** gets to the `300M` Classical Teacher (`3.5845`).

---

## 2. Experimental Matrix (`7` Arms `H72_1..H72_7`)

All arms use `wf_shape = SHAPE_12x2x32` ($d_{\text{pkt}}=32$), `wte_init_mode = TEACHER_PROCRUSTES_SPHERE`, `ve_coupling_mode = FROZEN_TEACHER_VE0_COUPLED`, `self_ortho_mode = FUBINI_STUDY_SELF_ORTHO`, `recovery_schedule_mode = PERSISTENT_COSINE_FEAT_AND_CE` (`interface_warmup_steps = 300`, `interface_align_weight = 0.5`, `wte_lr = 0.004`, `stiefel_lr = 0.012`), evaluated on the `300M` Classical Teacher (`3.5845` CE, `37.31%` Top-1, `58.26%` Top-5):

| Arm ID | SVD-Stiefel Rank (`L0 Params`) | Scope | State I/O Mode | Interior Coupling | QK Phase Mode | Steps (`Tokens`) | Scientific Question |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`H72_1_R384_PARITY_LOCAL_V_800STEPS`** | `R384` (`2.37M`, `1.00x`) | `L0_ATTN` | `WF_IN_WF_OUT` | **`ATTENDED_RHO_ON_LOCAL_V`** | `BERRY_PHASE_COMPLEX_QK` | `800` (`6.55M`) | Does `1.00x` parameter parity with `LOCAL_V` (`F7` winner) strictly beat `3.5845` CE at `6.55M` tokens? |
| **`H72_2_R384_PARITY_LIOUVILLE_800STEPS`** | `R384` (`2.37M`, `1.00x`) | `L0_ATTN` | `WF_IN_WF_OUT` | **`LIOUVILLE_COMMUTATOR_RHO_V`** | `BERRY_PHASE_COMPLEX_QK` | `800` (`6.55M`) | Does `1.00x` parameter parity with Liouville commutator (`F9` winner) strictly beat `3.5845` CE at `6.55M` tokens? |
| **`H72_3_R128_SUBPARITY_3X_800STEPS`** | **`R128` (`0.80M`, `0.34x`)** | `L0_ATTN` | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_LOCAL_V` | `BERRY_PHASE_COMPLEX_QK` | `800` (`6.55M`) | Does **`3x` compressed** Quantum Layer 0 (`0.80M` vs. `2.36M`) beat `3.5845` CE at `6.55M` tokens? |
| **`H72_4_R64_SUBPARITY_6X_800STEPS`** | **`R64` (`0.41M`, `0.17x`)** | `L0_ATTN` | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_LOCAL_V` | `BERRY_PHASE_COMPLEX_QK` | `800` (`6.55M`) | How close does **`6x` compressed** Quantum Layer 0 (`0.41M` vs. `2.36M`) get at `6.55M` tokens? |
| **`H72_5_R768_LOCAL_V_BERRY_WF_500S`** | `R768` (`4.73M`, `2.00x`) | `L0_ATTN` | `WF_IN_WF_OUT` | **`ATTENDED_RHO_ON_LOCAL_V`** | `BERRY_PHASE_COMPLEX_QK` | `500` (`4.10M`) | Does `LOCAL_V` + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` set a new 500-step record (`< 3.5823`)? |
| **`H72_6_R768_LIOUVILLE_BERRY_WF_500S`** | `R768` (`4.73M`, `2.00x`) | `L0_ATTN` | `WF_IN_WF_OUT` | **`LIOUVILLE_COMMUTATOR_RHO_V`** | `BERRY_PHASE_COMPLEX_QK` | `500` (`4.10M`) | Does Liouville commutator + `WF_IN_WF_OUT` + `BERRY_PHASE_COMPLEX_QK` beat `3.5825` at `500` steps? |
| **`H72_7_MULTILAYER_L0_L1_R768_800S`** | `R768` (`4.73M` x 2) | **`EMB_PLUS_L0_L1`** | `WF_IN_WF_OUT` | `ATTENDED_RHO_ON_LOCAL_V` | `BERRY_PHASE_COMPLEX_QK` | `800` (`6.55M`) | How much of the 2-layer (`Emb + L0 + L1`) gap closes at $r=768$ over `800` steps (`6.55M` tokens)? |

---

## 3. Files to Create & Verification Gates (`H1..H3`)

No modifications to `src/qdlblocks/` are required (all primitives are already implemented and verified in `quantum_svd_stiefel_graft.py` and `quantum_frontend_graft.py`).
1. **[`tests/test_plan072_compression_frontier_and_local_v_synergy.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan072_compression_frontier_and_local_v_synergy.py)** (`< 200` SLOC):
   - **Gate H1**: Verifies all 7 `Plan 072` arm configurations (`H72_1..H72_7`) resolve via `resolve_plan072_arm_config` with exact expected ranks, scopes, interior modes, and parameter counts.
   - **Gate H2**: Verifies forward + backward + Riemannian step on CPU for `H72_1`, `H72_6`, and `H72_7` (`EMB_PLUS_L0_L1` with `ATTENDED_RHO_ON_LOCAL_V` + `BERRY_PHASE_COMPLEX_QK`).
   - **Gate H3**: AST best-practices audit (`<= 1,000` SLOC, `<= 8` args per function) across all Wave 52H files.
2. **[`examples/82_verify_plan072_compression_frontier_and_local_v_synergy.py`](file:///home/nixos/QuantumDLBlocks/examples/82_verify_plan072_compression_frontier_and_local_v_synergy.py)** (`< 200` SLOC):
   - CLI benchmark runner executing `H72_1..H72_7` on `cuda:0` and `cuda:1`.
