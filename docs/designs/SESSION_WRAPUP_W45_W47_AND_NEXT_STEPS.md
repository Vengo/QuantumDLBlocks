# Session Wrap-Up (`Waves 45–47` / `EXPERIMENT_PLAN_060–062`) & Next Steps

**Date**: October 29, 2026
**Scope**:
1. **Wave 45 & 45b ([`EXPERIMENT_PLAN_060.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_060.md))**: Compiled Leaderboard Classical GPT (`CompiledLeaderboardClassicalGPT`) + Full `50M`-Token Interval Teacher Checkpoint Sweep (`w45b_classical_teacher_ckpt50m_l12_d768_b262k`).
2. **Wave 46 ([`EXPERIMENT_PLAN_061.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_061.md))**: Impure Density-Matrix Layer Mirroring, Bures-Wasserstein McWeeny Natural Gradient Purification, Isometric $\sqrt{2}$ Hermitian Packing, & Exact `RiemannianStiefelMuon` (`ImpureQuantumDensityMirrorBlock`).
3. **Wave 47 ([`EXPERIMENT_PLAN_062.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_062.md))**: Full-Stack All-Layer Riemannian Impure Density GPT (`RiemannianImpureDensityGPT`) + Multi-Layer (`3-Layer` & `6-Layer`) Surgical Quantum Distillation on Dual RTX 3090s.

---

## 1. Executive Summary of Completed Waves (`Waves 45–47`)

### 1.1 Wave 45 & 45b: Classical Leaderboard Alignment & `50M`-Token Teacher Checkpoint Sweep ([`EXPERIMENT_PLAN_060.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_060.md))
- **Module**: [`src/qdlblocks/models/compiled_leaderboard_classical_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py) (`CompiledLeaderboardClassicalGPT`)
- **What Was Fixed vs. Legacy `ClassicalModdedGPT`**:
  1. Zero-copy `(B, T, H, D)` FlashAttention-2 with progressive window schedule $W_s(s) \in \{3, 6, 8\} \times 128$ (evaluated at $W_{\text{val}} = 8$).
  2. In-place Triton softcapped cross-entropy kernel (`_fused_softcap_ce_inplace_kernel`, with automatic NixOS/Windows-safe fallback to `_softcap_ce_chunk_elementwise_fallback` when `/sbin/ldconfig` is absent).
  3. 5-step Polar Express (`polar_express`) + NorMuon variance equalization + decoupled weight decay + $1.5\times$ `v_proj` LR boost + layerwise $v_0$ value residual ($\lambda_v = 0.50$ on interior layers $l \in [1, L-2]$).
- **Measured Results**:
  - Throughput jumped **`3.62x`** (`12,521 -> 45,273 tok/s`, `59.4%` MFU) and VRAM dropped **`2.31x`** (`12.22 -> 5.29 GiB`).
  - **`w45b_compiled_leaderboard_l12_d768_b262k_sub3p28`**: Reached **`3.3479` Official `10.49M`-Token Val CE** at `550.50M` tokens (`3.3106` 65K screening CE).
  - **`w45b_classical_teacher_ckpt50m_l12_d768_b262k`**: Saved 50M-token interval teacher checkpoints (`ckpt_step00000_0000M.pt` through `ckpt_step01908_0500M.pt` and `550M`) in `.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/`.

### 1.2 Wave 46: Impure Density-Matrix Layer Mirroring & `RiemannianStiefelMuon` ([`EXPERIMENT_PLAN_061.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_061.md))
- **Module**: [`src/qdlblocks/layers/impure_density_mirror_block.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py) (`ImpureQuantumDensityMirrorBlock`, `PureQuantumKetMirrorBlock`, `RiemannianStiefelMuon`)
- **Key Findings (`examples/71_verify_plan061_quantum_layer_mirroring.py` on `150M`-Token Teacher Layer 11)**:
  - **H1 (`PASS`) — Impure Density vs. Rank-1 Pure Ket**:
    - Rank-1 pure ket attention $\sum_s w_{ts}|v_s\rangle$ suffers **`16.0x` midpoint hallucination** at $p=2$.
    - Rank-4 impure density $\rho_t = \sum_s w_{ts}\sum_{r=1}^4 \pi_{t,r}|v_{s,r}\rangle\langle v_{s,r}|$ with Isometric $\sqrt{2}$ off-diagonal Hermitian packing and Bures-Wasserstein McWeeny purification reduced normalized update MSE by **`20.0%`** (`0.0972 -> 0.0777`, cosine similarity **`99.04%`**).
  - **H2 (`PASS`) — Exact `RiemannianStiefelMuon` vs. Euclidean Muon**:
    - Tangent projection $\Pi_{T_W}(G) = G - W\operatorname{sym}(W^\top G)$ + Polar Express + $1/\sqrt{1+\eta^2}$ geodesic retraction maintained Stiefel orthogonality error at **`3.70e-7`** vs. `1.124` for Euclidean Muon (**`3.04e6x` reduction**).
  - **H3 (`PASS`) — Surgical Layer-11 Quantum Graft**:
    - Replacing Classical Layer 11 with an unaligned quantum block degraded Val CE from `3.7972` to `4.2943` (`+0.4972` nats damage); Golub-Pereyra variable projection + `RiemannianStiefelMuon` recovered Val CE to **`3.8825`** (**`82.8%` of the ablation gap recovered**).

### 1.3 Wave 47: All-Layer `RiemannianImpureDensityGPT` & Multi-Layer Surgical Grafts ([`EXPERIMENT_PLAN_062.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_062.md))
- **Module**: [`src/qdlblocks/models/riemannian_impure_density_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py) (`RiemannianImpureDensityGPT`, `RiemannianImpureDensityBlock`, `AllLayerRiemannianStiefelMuon`, `riemannian_complex_unitary_muon_step`, `polar_factorize_stiefel_svd`, `fast_rank_r_token_density_pack`)
- **Key Findings (`examples/72_verify_plan062_riemannian_impure_density_gpt.py` on `400.03M`-Token Teacher across Dual RTX 3090s)**:
  - **H1 (`PASS`) — All-Layer `AllLayerRiemannianStiefelMuon` ($L=6, d=768$)**:
    - Beat Euclidean Muon by **`+2.0878` nats** (`7.0792` vs. `9.1669` Val CE) while maintaining **`3.33e-7` Stiefel error** across all 2D real projections and **`1.23e-7` unitary error** across all 3D per-head $U(16)$ complex mixers, and beat Rank-1 Pure Ket (`PURE_KET_R1`) by **`+0.1026` nats** (`+0.2062` nats at Step 0).
  - **H2 (`PASS`) — Stationary Input Wavefunction Sphere Anchor (`ANCHORED_SPHERE`) + $V_0$ Value Ket Highway**:
    - Beat unanchored Euclidean input embeddings without $V_0$ by **`+0.6367` nats** (`7.0792` vs. `7.7159` Val CE), confirming the Gemini Notebook hypothesis that unconstrained Euclidean input drift causes a moving-target basis shift that harms validation generalization.
  - **H3 (`PASS`) — Multi-Layer Surgical Quantum Grafts (`3-Layer` & `6-Layer` on `400.03M`-Token Teacher)**:
    - **3-Layer Graft (`Layers 9, 10, 11`)**: Unaligned `5.7336` Val CE $\to$ Aligned Polar-Stiefel **`4.6655` Val CE** (`4.5889` peak, **`54.6%–58.5%` gap recovered**), post-finetune Stiefel error **`2.59e-7`**. Per-layer update cosine similarity: Layer 9 `99.44%`, Layer 10 **`99.82%`** (`0.0101` MSE), Layer 11 **`99.63%`** (`0.0253` MSE).
    - **6-Layer Graft (`Layers 6, 7, 8, 9, 10, 11` — Half of the 12-Layer Model!)**: Unaligned `7.0443` Val CE $\to$ Aligned Polar-Stiefel **`5.5106` Val CE** (**`46.9%` gap recovered**), post-finetune Stiefel error **`2.58e-7`**.

---

## 2. Recommended Next Steps for the Dual RTX 3090 Session

1. **Full 12-Layer Layerwise Progressive Quantum Distillation Across Checkpoints (`50M -> 550M`)**:
   - Using `RiemannianImpureDensityBlock` and `polar_factorize_stiefel_svd`, fit all 12 layers in parallel across the two RTX 3090s (e.g., Layers `0..5` on GPU 0 and Layers `6..11` on GPU 1), then run joint end-to-end `AllLayerRiemannianStiefelMuon` training with singular-value-weighted Stiefel preconditioning $\widetilde{G}_{U} = G_{U} \operatorname{diag}(\sigma / \sigma_{\max})$ so that right-hand Stiefel factors `w_readout_vt` and `w_mlp_vt` preserve the Golub-Pereyra singular-value spectrum during long-horizon fine-tuning.
2. **Long-Horizon Standalone `RiemannianImpureDensityGPT` Speedrun on Dual RTX 3090s (`DDP` or `grad_accum` at `262,144` Tokens/Step)**:
   - Train a standalone 12-layer `RiemannianImpureDensityGPT` (or hybrid `6 Classical + 6 Quantum Impure Density` model) initialized from the aligned Stiefel-SVD teacher factors using `AllLayerRiemannianStiefelMuon`.
