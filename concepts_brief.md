# Multi-Lens Brainstorm Synthesis (`concepts_brief.md`) — Wave 48 (`EXPERIMENT_PLAN_063.md`)

**Target Feature Concept**: Standalone, Teacher-Free Complex Quantum Born GPT on Official `FineWeb` (`10,485,760`-token Shard 0 benchmark in [`fineweb_val_000000.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin), targeting `< 3.28` Val CE under a `<= 330M` token horizon on `2x NVIDIA GeForce RTX 3090 24GB` GPUs).

**Synthesized Swarm Lenses**:
1. `egm-design` (*Skeptical Architect Brainstormer*)
2. `egm-codebase-details` (*Codebase Integration Brainstormer*)
3. `egm-zoom-out` (*Strategic & Systems Alignment Brainstormer*)
4. `egm-comprehension` (*UX & Mathematical Ergonomics Brainstormer*)
5. `egm-implementation-readiness` (*Implementation & Verification Brainstormer*)
6. `egm-structural` (*Structural & Modularity Brainstormer*)

---

## 1. Executive Synthesis: The 3 Structural Bottlenecks Blocking Standalone `< 3.28` Val CE

Across Waves 45–47 ([`EXPERIMENT_PLAN_060.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_060.md)–[`EXPERIMENT_PLAN_062.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_062.md)), Rank-$R$ ($R=4$) impure density matrices with Isometric $\sqrt{2}$ Hermitian packing and Bures-Wasserstein McWeeny purification mirrored trained classical transformer layers to `99.83%` cosine similarity (`0.0093` relative MSE on Layer 10). However, auditing [`riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py) and [`examples/72_verify_plan062_riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/72_verify_plan062_riemannian_impure_density_gpt.py) reveals three structural blockers preventing `RiemannianImpureDensityGPT` from training **standalone from scratch (teacher-free)** to `< 3.28` Val CE on FineWeb:

1. **Blocker 1 — Quadratic $U(1)$ Phase Erasure & Missing First-Moment Uhlmann Gauge Anchor in `RiemannianImpureDensityBlock`**:
   - In [`RiemannianImpureDensityBlock.compute_packed_density_features`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L438-L471), token value states are mapped exclusively into quadratic outer products $\rho_{s,h}^{\text{tok}} = \sum_{r=1}^R p_{s,h,r} |\hat{v}_{s,h}^{(r)}\rangle\langle\hat{v}_{s,h}^{(r)}| \in \text{Herm}(\mathbb{C}^{16})$ via [`fast_rank_r_token_density_pack`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L227-L255) (`width = 256` per head).
   - Because $|v\rangle\langle v|$ is invariant under $|v\rangle \mapsto e^{i\theta}|v\rangle$, **all first-moment linear complex phase information is erased before causal attention mixing**.
   - Furthermore, [`extract_rank2_krylov_eigenpair`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L162-L213) currently seeds its Lanczos-Krylov iteration from a purely real diagonal vector `v0_re = sqrt(diag(rho_re)), v0_im = 0` ([lines 168–171](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L168-L171)), which suffers arbitrary $U(1)$ eigenvector sign/phase flips across tokens and steps and is not called inside `RiemannianImpureDensityBlock`.

2. **Blocker 2 — Hidden Classical Teacher & Periodic Golub-Pereyra Ridge-Regression Crutch**:
   - In [`examples/72_verify_plan062_riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/72_verify_plan062_riemannian_impure_density_gpt.py#L218-L328), `_initialize_all_layer_quantum_gpt_from_teacher` copied `wte` and `lm_head` from a `400M`-token [`CompiledLeaderboardClassicalGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py) checkpoint, froze `lm_head`, and called [`fit_stiefel_observables_polar_ridge`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L495-L555) every 20 steps on 8 cached teacher trajectory batches while excluding `w_obs_u, w_obs_vt, w_chan_u, w_chan_vt` from `AllLayerRiemannianStiefelMuon` ([lines 266–276](file:///home/nixos/QuantumDLBlocks/examples/72_verify_plan062_riemannian_impure_density_gpt.py#L266-L276)).
   - Why did Wave 47 exclude readout matrices from `AllLayerRiemannianStiefelMuon`? Because [`RiemannianImpureDensityBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L301-L320) over-parameterized each readout as a 3-factor SVD product $M = U \operatorname{diag}(\sigma) V^\top$ (`w_obs_u`, `obs_sigma`, `w_obs_vt`). When $\sigma_d \approx 10^{-4}$, unweighted Newton-Schulz on $U$ and $V^\top$ amplifies tiny tail-singular-value gradients by $1/\sigma_d \approx 10^4\times$, scrambling the dominant singular modes.
   - In a **standalone, teacher-free model**, there is no Golub-Pereyra ridge target to factorize: each readout should be a **single direct semi-orthogonal Stiefel matrix** $W_{\text{obs}} \in \text{St}(D_{\text{obs}}, d)$ (and $W_{\text{chan}} \in \text{St}(d_{\text{chan}}, d)$) scaled by a learnable post-readout vector/scalar gain (initialized near zero at Step 0 for residual highway stability) and updated end-to-end by **Tikhonov-damped `RiemannianStiefelMuon`**.

3. **Blocker 3 — Boundary Gauge & Active-Token Optimization Mismatch at $V = 50,304$**:
   - In [`RiemannianImpureDensityGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L716-L720), `InputBoundaryMode.ANCHORED_SPHERE` froze `wte.weight.requires_grad_(False)` completely. While freezing random Gaussian embeddings worked when `wte` was copied from a 400M-token teacher, freezing a **random Step-0 `wte`** in a standalone scratch run traps the model in a random token geometry (`> 7.0` Val CE).
   - Conversely, unconstrained Euclidean `AdamW` on `wte` and `lm_head` causes radial norm drift across Zipfian frequency deciles. We need **active-token Batch-Subspace $(Q, N_{\text{act}}, A)$ Tikhonov Gram Muon** and **Riemannian Sphere Tangent Momentum (with SNR trust-region gating)** compared head-to-head against tuned `AdamW` on the $V = 50,304$ vocabulary registers, paired with both the **Rank-2 Krylov Born Readout** and the **Chunked Softcap Cross-Entropy Readout Baseline**.

---

## 2. Pillar-by-Pillar Mathematical & Architectural Design

### Pillar 1: Fused `288`-Width Isometric $\sqrt{2}$ Hermitian + First-Moment Ket Attention, Bures-Wasserstein McWeeny Purification, & Uhlmann Phase-Locked Rank-2 Krylov Extraction

#### 1.1 Fused Single-GEMM Packing (`width = d_h^2 + 2 d_h = 256 + 32 = 288` per head)
- **Lens Synthesis (`egm-design` + `egm-codebase-details`)**:
  - In [`AgyRewrittenGeminiBornComplexPackedWithRoPE.py`](file:///home/nixos/QuantumDLBlocks/AgyRewrittenGeminiBornComplexPackedWithRoPE.py#L88-L95) (`pack_hermitian_outer_product_and_ket`), the Hermitian outer product and first-moment ket were concatenated into a single GEMM, but the off-diagonal triangle entries lacked the $\sqrt{2}$ Frobenius-isometric scale factor (`SQRT2_ISOMETRIC_OFFDIAG_SCALE = 1.4142135623730951`), breaking $\langle \tilde{P}(A), \tilde{P}(B)\rangle = \operatorname{Tr}(AB)$ by $2\times$ on off-diagonals.
  - Conversely, [`fast_rank_r_token_density_pack`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L227-L255) in `riemannian_impure_density_gpt.py` includes the exact $\sqrt{2}$ isometric mask (`re_pack_mask`, `im_pack_mask`), and contracts $R = 4$ value branches in $O(d_h^2 R)$, but omits the $2 d_h = 32$ first-moment ket.
  - **Unified Fused Solution (`fast_rank_r_hermitian_and_ket_pack`)**:
    For each token $s$ and head $h \in \{1, \dots, 6\}$ with $d_h = 16$ and $R = 4$ value branches $V_{s,h} = (v_{\text{re},s,h}, v_{\text{im},s,h}) \in \mathbb{R}^{R \times 16} \times \mathbb{R}^{R \times 16}$:
    1. **Isometric Rank-$R$ Density Matrix ($d_h^2 = 256$ dims)**:
       Normalize total branch energy $E_{s,h} = \sum_{r=1}^R \|v_{s,h}^{(r)}\|_2^2 + \varepsilon$, set $\tilde{V}_{s,h} = V_{s,h} / \sqrt{E_{s,h}}$, contract $\rho_{s,h}^{\text{tok}} = \tilde{V}_{s,h}^\dagger \tilde{V}_{s,h} \in \text{Herm}(\mathbb{C}^{16})$ via $(16 \times 4) \times (4 \times 16)$ matmuls, and apply the $\sqrt{2}$ isometric masks to obtain $\tilde{P}(\rho_{s,h}^{\text{tok}}) \in \mathbb{R}^{256}$.
    2. **First-Moment Coherent Anchor Ket ($2 d_h = 32$ dims)**:
       Extract the primary coherent ket $|v_{s,h}^{\text{coh}}\rangle \in \mathbb{C}^{16}$ (either branch $r=0$ or energy-weighted coherent sum $\operatorname{normalize}_{\mathbb{C}}(\sum_{r=1}^R w_r v_{s,h}^{(r)})$) as $[\hat{v}_{\text{re},s,h}^{\text{coh}}, \hat{v}_{\text{im},s,h}^{\text{coh}}] \in \mathbb{R}^{32}$.
    3. **Single Tensor-Core Attention GEMM (`width = 288 = 9 * 32`)**:
       Concatenate $\Pi_{s,h} = [\tilde{P}(\rho_{s,h}^{\text{tok}}), \hat{v}_{\text{re},s,h}^{\text{coh}}, \hat{v}_{\text{im},s,h}^{\text{coh}}] \in \mathbb{R}^{288}$ ($288 = 9 \times 32$ is an exact multiple of 32 and 16 for Tensor-Core MMA alignment).
       Execute **one** batched GEMM per layer:
       $$M_{t,h} = \sum_{s \le t} w_{ts,h} \Pi_{s,h} = \big[\underbrace{\tilde{P}(\rho_{t,h}^{(0)})}_{256}, \underbrace{\bar{v}_{\text{re},t,h}^{\text{raw}}}_{16}, \underbrace{\bar{v}_{\text{im},t,h}^{\text{raw}}}_{16}\big] \in \mathbb{R}^{288}$$
       where $|\bar{v}_{t,h}\rangle = \operatorname{normalize}_{\mathbb{C}}(\bar{v}_{\text{re},t,h}^{\text{raw}} + i \bar{v}_{\text{im},t,h}^{\text{raw}}) \in \mathbb{C}\mathbb{S}^{15}$ is the contextual first-moment V-space anchor ket.

#### 1.2 Bures-Wasserstein McWeeny Purification & Uhlmann Phase-Locked Rank-2 Lanczos-Krylov Eigenpair Extraction
- **Lens Synthesis (`egm-comprehension` + `egm-design`)**:
  - Unpack $\tilde{P}^{-1}(\tilde{P}(\rho_{t,h}^{(0)})) \to (\rho_{\text{re},t,h}^{(0)}, \rho_{\text{im},t,h}^{(0)})$ and apply 1 step of [`bures_wasserstein_mcweeny_step`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L130-L160):
    $$\rho_{t,h}^{(1)} = \frac{(\rho_{t,h}^{(0)})^2}{\operatorname{Tr}((\rho_{t,h}^{(0)})^2) + \varepsilon}$$
  - **Fixing the Gauge Ambiguity in `extract_rank2_krylov_eigenpair` via Uhlmann Phase-Locking**:
    Instead of initializing Lanczos-Krylov power iteration with `v0_re = sqrt(diag(rho_re)), v0_im = 0` (which discards complex phase), seed `extract_rank2_krylov_eigenpair` directly with the contextual first-moment anchor ket $|\phi_{t,h}^{(0)}\rangle = |\bar{v}_{t,h}\rangle$:
    1. **Top Eigenmode $(\lambda_1, |\phi_1\rangle)$**:
       Apply 1–2 matvec steps $|w_1\rangle = \rho_{t,h}^{(1)} |\bar{v}_{t,h}\rangle$, normalize $|\tilde{\phi}_1\rangle = |w_1\rangle / (\|w_1\|_2 + \varepsilon)$, and compute Rayleigh quotient $\lambda_1 = \langle \tilde{\phi}_1 \mid \rho_{t,h}^{(1)} \mid \tilde{\phi}_1\rangle \in [0, 1]$.
       **Uhlmann Parallel-Transport Phase Lock**: Lock the global $U(1)$ phase of $|\phi_1\rangle$ to the first-moment anchor $|\bar{v}_{t,h}\rangle$ so $\langle \bar{v}_{t,h} \mid \phi_1 \rangle_{\mathbb{C}} \in \mathbb{R}_{\ge 0}$:
       $$|\phi_1\rangle = \frac{\langle \tilde{\phi}_1 \mid \bar{v}_{t,h}\rangle_{\mathbb{C}}}{|\langle \tilde{\phi}_1 \mid \bar{v}_{t,h}\rangle_{\mathbb{C}}| + \varepsilon} |\tilde{\phi}_1\rangle$$
    2. **Orthogonal Second Krylov Eigenmode $(\lambda_2, |\phi_2\rangle)$**:
       Compute the Lanczos residual $|u_2\rangle = \rho_{t,h}^{(1)} |\phi_1\rangle - \lambda_1 |\phi_1\rangle$, enforce exact complex Gram-Schmidt orthogonality $|u_2^\perp\rangle = |u_2\rangle - \langle \phi_1 \mid u_2\rangle_{\mathbb{C}} |\phi_1\rangle$, normalize $|\phi_2\rangle = |u_2^\perp\rangle / (\|u_2^\perp\|_2 + \varepsilon)$, and evaluate $\lambda_2 = \langle \phi_2 \mid \rho_{t,h}^{(1)} \mid \phi_2\rangle \in [0, 1]$.
  - **Why This Eliminates Phase Erasure**:
    By evolving $\rho_{t,h}^{(1)}$ through the per-head unitary $U_h \in U(16)$ and combining the isometric packed density observables $\tilde{P}(U_h \rho_{t,h}^{(1)} U_h^\dagger) \in \mathbb{R}^{256}$ with the phase-locked Rank-2 Krylov eigenmodes $(\sqrt{\lambda_1} U_h |\phi_1\rangle, \sqrt{\lambda_2} U_h |\phi_2\rangle)$, the block preserves both **second-moment polysemous subspace geometry** ($\operatorname{EffRank} > 1$) and **first-moment coherent interferometric phase**.

---

### Pillar 2: Removing the Wave 47 Teacher/Golub-Pereyra Crutch & Tikhonov-Damped `RiemannianStiefelMuon`

#### 2.1 Why Wave 47 Needed Periodic Golub-Pereyra Ridge Projection—and How to Eliminate It
- **Lens Synthesis (`egm-design` + `egm-implementation-readiness`)**:
  - In Wave 47 ([`riemannian_impure_density_gpt.py` lines 301–320](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L301-L320)), `w_obs` was split into `w_obs_u` $(1536 \times 768)$, `obs_sigma` $(768,)$, and `w_obs_vt` $(768 \times 768)$ specifically so `fit_stiefel_observables_polar_ridge` could store the exact SVD of a ridge regression matrix $M = U \Sigma V^\top$.
  - When training from scratch without a teacher:
    1. **Replace the 3-Factor SVD Readout with a Direct Semi-Orthogonal Stiefel Operator + Learnable Residual Gain**:
       Parameterize the attention observable readout as a single tall Stiefel matrix $W_{\text{obs}} \in \text{St}(D_{\text{obs}}, d)$ ($W_{\text{obs}}^\top W_{\text{obs}} = I_d$) multiplied by a learnable per-channel or scalar residual scale $\alpha_{\text{obs}} \in \mathbb{R}^d$ (initialized to $\sigma_0 / \sqrt{2 L}$), and similarly for the Mach-Zehnder channel down-projection $W_{\text{chan}} \in \text{St}(d_{\text{chan}}, d)$ with scale $\alpha_{\text{chan}} \in \mathbb{R}^d$.
       - **VRAM & FLOP Reduction**: Eliminates two $768 \times 768$ square orthogonal matrices (`w_obs_vt`, `w_chan_vt`) per layer—saving **$24$ Newton-Schulz $768 \times 768$ matrix orthogonalizations per step** and **$24 \times 768^2$ GEMM FLOPs per forward/backward pass**.
    2. **Tikhonov-Damped Newton-Schulz / Polar Express in `RiemannianStiefelMuon`**:
       In [`_newton_schulz5_tall_real`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L216-L231) and [`_complex_newton_schulz5`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L114-L138), standard Frobenius-normalized Newton-Schulz drives every non-zero singular value of the tangent momentum $M_{\text{tan}}$ toward $1.0$, even when a singular direction has low gradient SNR.
       Adding **Tikhonov Gram damping** $\lambda_{\text{tik}} > 0$ (e.g. $\lambda_{\text{tik}} = 10^{-3}$) to the Gram matrix inside the first Newton-Schulz / Polar Express iteration:
       $$G_{\text{tik}} = X^\dagger X + \lambda_{\text{tik}} I_d, \qquad X \leftarrow \frac{1}{2} X (3 I_d - G_{\text{tik}})$$
       smoothly damps ill-conditioned noise directions while keeping high-SNR tangent directions on the unit-step geodesic $W_{\text{next}} = (W - \eta U_{\text{tan}}) / \sqrt{1 + \eta^2}$ (`mean_stiefel_err < 1e-6`, `mean_unitary_err < 1e-6`).
    3. **100% Backprop Parameter Coverage**:
       Every layer's `w_q`, `w_k`, `w_v`, `w_obs`, `w_beam_a`, `w_beam_b`, `w_chan` (2D real Stiefel) and `(head_unitary_re, head_unitary_im)` (3D complex $U(16)$) are registered in `AllLayerRiemannianStiefelMuon` from Step 0—zero calls to `fit_stiefel_observables_polar_ridge`, zero teacher checkpoints, and zero cached trajectory batches.

---

### Pillar 3: Boundary Readouts (`Rank-2 Krylov Born` vs. `Softcap Linear CE`) & Vocabulary Optimization (`Batch-Subspace (Q, N_act, A) Tikhonov Gram Muon` vs. `Riemannian Sphere Tangent Momentum` vs. `AdamW`)

#### 3.1 Two Clean Readout Modes at the $V = 50,304$ Vocabulary Boundary
- **Lens Synthesis (`egm-zoom-out` + `egm-comprehension`)**:
  To isolate the contribution of the quantum density backbone from the vocabulary boundary measurement while targeting `< 3.28` Val CE on `fineweb_val_000000.bin`, the standalone model supports a strategy enum `VocabularyReadoutMode`:
  1. **`VocabularyReadoutMode.RANK2_KRYLOV_BORN` (Quantum Rank-2 Krylov Born Measurement)**:
     - Projects the final normalized hidden state (or aggregated multi-head Rank-2 Krylov eigenpair $((\lambda_{1,t}, |\Phi_{1,t}\rangle), (\lambda_{2,t}, |\Phi_{2,t}\rangle))$ with $\lambda_{1,t} + \lambda_{2,t} = 1$) against complex or real unit-norm vocabulary measurement kets $|e_v\rangle$ ($v \in \{0, \dots, 50,303\}$):
       $$p_t(v) \propto \lambda_{1,t} |\langle e_v \mid \Phi_{1,t}\rangle|^{2p} + \lambda_{2,t} |\langle e_v \mid \Phi_{2,t}\rangle|^{2p} \quad (\text{or calibrated log-intensity / softcap Born logits})$$
     - Evaluated in $O(1)$-extra-VRAM chunks (`chunk_tokens = 8192`) satisfying **Rule 9** (single-pass chunk backward with zero duplicate power/GEMM calls).
  2. **`VocabularyReadoutMode.SOFTCAP_LINEAR_CE` (Numerically Stable Softcap NLL Baseline)**:
     - Uses the zero-extra-VRAM in-place Triton kernel [`_CompiledChunkedSoftcapLinearCEFn`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L329-L433) ($15 \tanh(z / 15)$) for an apples-to-apples leaderboard readout baseline against [`CompiledLeaderboardClassicalGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py) (`3.3909` Val CE at 150M / `< 3.28` target at $\le 330\text{M}$ tokens).

#### 3.2 Active-Token Vocabulary Optimizers at $V = 50,304$ (`VocabOptimizerMode`)
- **Lens Synthesis (`egm-design` + `egm-implementation-readiness`)**:
  Why is standard `Muon` or `RiemannianStiefelMuon` infeasible on the full $V \times d = 50,304 \times 768$ vocabulary table? Because a dense $50,304 \times 768$ Newton-Schulz step costs $5 \times 2 \times 50,304 \times 768^2 \approx 296\text{ GFLOPs}$ per step and mixes gradients of `~45,000` un-sampled tokens (whose current-step gradient is zero) with the $N_{\text{act}} \le B \times T \approx 4,096\text{–}8,192$ active tokens in the batch.
  We design and compare three optimizer strategies for `wte` (and `lm_head`):
  1. **Active-Token Batch-Subspace $(Q, N_{\text{act}}, A)$ Tikhonov Gram Muon (`VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON`)**:
     - Let $\mathcal{I}_{\text{act}} = \operatorname{unique}(\text{idx} \cup \text{targets})$ be the $N_{\text{act}} \ll V$ active token rows in the current microbatch/step (or for `lm_head`, factor the chunked gradient $G_{\text{lm}} = P^\top X_{\text{act}} \in \mathbb{R}^{V \times d}$ through the $d \times d$ batch feature subspace $Q \in \text{St}(d, d)$).
     - On the active token rows $G_{\text{act}} \in \mathbb{R}^{N_{\text{act}} \times d}$ (with sparse momentum updated strictly on active rows or decayed lazily), form the small $d \times d$ ($768 \times 768$) **Tikhonov-damped feature Gram matrix**:
       $$A = G_{\text{act}}^\top G_{\text{act}} + \lambda_{\text{tik}} \|G_{\text{act}}\|_F^2 I_d \in \mathbb{R}^{d \times d}$$
     - Compute the inverse square root $A^{-1/2}$ in $O(d^3)$ time on the $768 \times 768$ matrix via 5-step Newton-Schulz / Polar Express (`_newton_schulz_inv_sqrt`), and precondition the active rows $U_{\text{act}} = G_{\text{act}} A^{-1/2}$ before retracting each active vocabulary row onto the unit RMS sphere $\sqrt{d}\,\mathbb{S}^{d-1}$.
     - **Complexity**: Reduces vocabulary orthogonalization cost from $O(V d^2)$ ($50,304$ rows) to $O(N_{\text{act}} d^2 + d^3)$ ($\le 4,096$ active rows) with zero inactive-token momentum pollution.
  2. **Riemannian Sphere Tangent Momentum with SNR Trust-Region Gating (`VocabOptimizerMode.SPHERE_TANGENT_SNR_MUON`)**:
     - Each vocabulary row $w_v \in \sqrt{d}\,\mathbb{S}^{d-1}$ lives on the oblique sphere manifold.
     - For each active token $v \in \mathcal{I}_{\text{act}}$:
       1. Project gradient $g_v$ onto the sphere tangent space at $w_v$:
          $$g_{v,\perp} = g_v - \frac{\langle w_v, g_v\rangle}{\|w_v\|_2^2} w_v$$
       2. Update tangent momentum $m_v \leftarrow \beta m_v + (1 - \beta) g_{v,\perp}$ and second moment $v_v \leftarrow \beta_2 v_v + (1 - \beta_2) \|g_{v,\perp}\|_2^2$ (parallel-transporting $m_v$ back to $T_{w_v}\mathbb{S}^{d-1}$ via $m_v \leftarrow m_v - \frac{\langle w_v, m_v\rangle}{\|w_v\|_2^2} w_v$).
       3. **SNR Trust-Region Gate**: Compute per-token signal-to-noise ratio $\text{SNR}_v = \frac{\|m_v\|_2}{\sqrt{v_v} + \varepsilon} \in [0, 1]$ and gate the angular geodesic step size:
          $$\theta_v = \eta_{\text{sphere}} \cdot \frac{\text{SNR}_v}{\text{SNR}_v + \tau_{\text{snr}}}$$
          so rare/noisy tokens with low gradient SNR take conservative steps while high-SNR tokens traverse the great-circle geodesic:
          $$w_v^{\text{next}} = \cos(\theta_v) w_v - \sin(\theta_v) \sqrt{d} \frac{m_v}{\|m_v\|_2 + \varepsilon}$$
  3. **Euclidean `AdamW` Baseline (`VocabOptimizerMode.ADAMW_BASELINE`)**:
     - Standard decoupled `AdamW` (`wte_lr = 0.30`, `head_lr = 0.008`, `betas = (0.8, 0.95)`) matching `CompiledLeaderboardClassicalGPT` for controlled ablation.

---

## 3. Multi-Lens Review & Risk Triage (`6` Specialized Lenses)

### Lens 1: `egm-design` (*Skeptical Architect Brainstormer*)
- **Risk 1 — Throughput & Memory Overhead of `extract_rank2_krylov_eigenpair` + $16 \times 16$ Unpacking Across 12 Layers**:
  - In [`RiemannianImpureDensityBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L446-L449), unpacking `(B, H, T, 256)` to `float32` `(B, H, T, 16, 16)` and running McWeeny squaring + Rank-2 Krylov extraction in eager PyTorch creates ~35 small elementwise/matmul kernel launches per layer (`420+` launches across 12 layers).
  - **Locked Resolution**: Fuse the $16 \times 16$ unpacking + McWeeny purification + Uhlmann Rank-2 Krylov extraction into a `@torch.compile` helper (`_compiled_mcweeny_uhlmann_krylov_step`) using `bfloat16` batched matmuls with `float32` trace/norm reductions.
- **Risk 2 — Architectural Framing (`SmearGate` + `U-Net` Skip Highway vs. Clean Sequential Stack)**:
  - [`CompiledLeaderboardClassicalGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py) uses lag-1 bigram token smearing (`SmearGate`) and encoder-to-decoder U-Net skip connections (`layers 0..5 -> layers 11..6`).
  - **Locked Resolution**: Omit `SmearGate` and `U-Net` skip connections in this wave to evaluate and interpret the standalone quantum density + Rank-2 Krylov architecture cleanly on its own merits. Scaling model capacity or introducing `SmearGate`/`U-Net` skips are reserved for follow-up waves if needed.

### Lens 2: `egm-codebase-details` (*Codebase Integration Brainstormer*)
- **Exact File & Symbol Reuse Map**:
  - **Do NOT modify frozen Wave 47 [`src/qdlblocks/models/riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py) with boolean flags (Rule 6)**:
    Keep `riemannian_impure_density_gpt.py` intact so `tests/test_plan062_riemannian_impure_density_gpt.py` continues to pass 100%.
  - **Promote Reusable Primitives into [`src/qdlblocks/layers/impure_density_mirror_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py)** (currently `866` total lines, `~550` SLOC, well under the `1,000` SLOC soft limit):
    - `fast_rank_r_hermitian_and_ket_pack(v_re, v_im, re_pack_mask, im_pack_mask)`: Fuses $R=4$ isometric $\sqrt{2}$ density packing (`256`) + coherent first-moment ket (`32`) into width `288`.
    - `extract_rank2_krylov_eigenpair_uhlmann(rho_re, rho_im, v_bar_re, v_bar_im)`: Uhlmann phase-locked top-2 Lanczos-Krylov eigenpair extractor seeded by the contextual first-moment ket $|\bar{v}_{t,h}\rangle$.
    - `riemannian_stiefel_muon_step` & `riemannian_complex_unitary_muon_step`: Add `tikhonov_lambda: float = 1e-3` damping to the Gram matrix in Newton-Schulz.
  - **Create New Flag-Free Standalone Model Module `src/qdlblocks/models/standalone_quantum_born_gpt.py` (Rule 6)**:
    - Houses `StandaloneQuantumBornConfig` (`<= 8` fields, Rule 2), `VocabularyReadoutMode`, `VocabOptimizerMode`, `StandaloneQuantumBornBlock`, `StandaloneQuantumBornGPT`, `BatchSubspaceTikhonovGramMuon`, `RiemannianSphereTangentSNROptimizer`, and `build_standalone_quantum_born_optimizers`.
  - **Official Validation Evaluator**:
    - Reuse [`evaluate_compiled_leaderboard_validation`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L1034-L1142) (or a readout-polymorphic wrapper) on `/home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin` (`DEFAULT_MODDED_VAL_TOKENS = 10_485_760` for final/milestone evals; `DEFAULT_FAST_VAL_TOKENS = 65_536` for rapid screening).

### Lens 3: `egm-zoom-out` (*Strategic & Systems Alignment Brainstormer*)
- **Hardware & Dual-RTX 3090 Execution Strategy (`2x 24 GB` VRAM)**:
  - On `2x NVIDIA GeForce RTX 3090` (`24 GB` each, `35.6 TFLOPS` BF16 Tensor Core per GPU):
    - **Phase 1 (Fast Hypothesis & Ablation Verification — H1, H2, H3)**:
      Run ablation arms across `cuda:0` and `cuda:1` over a `10M–25M`-token screening horizon (`65,536`-token FineWeb Shard 0 slice) to verify H1 (288-width Uhlmann phase-locked Krylov + McWeeny vs. quadratic phase-erased 256-width & pure ket), H2 (Teacher-free direct Stiefel readout + Tikhonov-damped `RiemannianStiefelMuon` vs. Euclidean Muon / undamped SVD readout), and H3 (`BatchSubspaceTikhonovGramMuon` & `RiemannianSphereTangentSNROptimizer` vs. `AdamW` across Rank-2 Krylov Born and Softcap CE readouts).
    - **Phase 2 (Full `<= 330M`-Token Standalone Speedrun on `2x RTX 3090` Targeting `< 3.28` Val CE)**:
      Run multi-GPU data-parallel training across `fineweb_train_000001.bin`..`fineweb_train_000004.bin` (`330M` tokens = `1,260` steps at `262,144` tokens/step = `2 GPUs x 8 grad_accum x B=16 x T=1024`), evaluating on the full `10,485,760`-token `fineweb_val_000000.bin` Shard 0 benchmark.

### Lens 4: `egm-comprehension` (*UX & Mathematical Ergonomics Brainstormer*)
- **Mathematical Clarity of the 3 Core Equations**:
  1. **Why Input Attention Width `288` Is the Minimal Sufficient Statistic**:
     Every head $h \in \{1, \dots, 6\}$ with $d_h = 16$ requires:
     - $d_h^2 = 256$ real numbers for the $\sqrt{2}$-isometric Hermitian density matrix $\tilde{P}(\rho_{t,h}^{(0)}) \in \mathbb{R}^{256}$ (capturing second-order quantum correlations and polysemous rank), plus
     - $2 d_h = 32$ real numbers for the complex first-moment anchor ket $|\bar{v}_{t,h}\rangle \in \mathbb{C}^{16}$ (capturing linear interferometric phase and Uhlmann parallel-transport orientation).
     - Total per-head attention value width = $256 + 32 = 288$.
  2. **Locked Option B (`320` Dims/Head) for Internal Readout & Vocabulary Born Readout**:
     - Inside each block, after $U_h \in U(16)$ evolution, the observable feature vector concatenates the packed purified density matrix $\tilde{P}(\rho_{t,h}^{\text{out}}) \in \mathbb{R}^{256}$ and **both** Uhlmann phase-locked Rank-2 Krylov eigenmodes $[\sqrt{\lambda_{1,t,h}}\phi_{1,\text{re}}, \sqrt{\lambda_{1,t,h}}\phi_{1,\text{im}}, \sqrt{\lambda_{2,t,h}}\phi_{2,\text{re}}, \sqrt{\lambda_{2,t,h}}\phi_{2,\text{im}}] \in \mathbb{R}^{64}$ (**Option B: `320` dims/head**, $6 \times 320 = 1,920$ total), projected to $d = 768$ via $W_{\text{obs}} \in \text{St}(1920, 768)$.
     - At the vocabulary boundary (`RANK2_KRYLOV_BORN`), the final hidden state $h_t \in \mathbb{R}^{768}$ is split into a Rank-2 complex ket pair $(|\Psi_{1,t}\rangle, |\Psi_{2,t}\rangle)$ weighted by $(\lambda_{1,t}, \lambda_{2,t})$ and measured against the vocabulary wavefunctions $|e_v\rangle$ via chunked log-sum-exp / softcapped Born intensity so gradients remain bounded and well-conditioned at Step 0.

### Lens 5: `egm-implementation-readiness` (*Implementation & Verification Brainstormer*)
- **Concrete Verification Gates (`H1`, `H2`, `H3`, `H4`)**:
  - **Unit Test Gate (`tests/test_plan063_standalone_quantum_born_gpt.py`)**:
    1. `test_fused_288_hermitian_and_ket_pack_isometry_and_phase`: Verifies `< 1e-6` agreement between single-GEMM `288`-width attention mixing and separate density + ket mixing, and proves that rotating a token value ket by $e^{i\theta}$ changes the Uhlmann phase-locked Krylov eigenmode $|\phi_1\rangle$ by $e^{i\theta}$ (non-zero phase sensitivity, eliminating quadratic phase erasure) while leaving $|\phi_1\rangle\langle\phi_1|$ gauge-consistent.
    2. `test_tikhonov_riemannian_stiefel_and_unitary_muon_stability`: Verifies that 100 steps of Tikhonov-damped `RiemannianStiefelMuon` on direct Stiefel readouts $W_{\text{obs}} \in \text{St}(1920, 768)$ and complex $U(16)$ unitaries maintain Stiefel/Unitary error `< 1e-5` without any Golub-Pereyra SVD re-projection.
    3. `test_batch_subspace_gram_muon_and_sphere_tangent_snr_optimizer`: Verifies that `BatchSubspaceTikhonovGramMuon` updates only active token rows $\mathcal{I}_{\text{act}}$ (`0.0` drift on inactive rows, unit-RMS sphere norm $\|w_v\|_2 = \sqrt{d}$ to `< 1e-5`) and `RiemannianSphereTangentSNROptimizer` enforces tangent orthogonality $\langle w_v, m_v\rangle = 0$ and bounded SNR trust-region steps.
    4. `test_standalone_quantum_born_gpt_teacher_free_convergence`: Verifies that `StandaloneQuantumBornGPT` trains end-to-end from random initialization (zero teacher checkpoint) with both `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE` readouts and satisfies Rule 9 single-pass chunked autograd efficiency.
  - **Linter & Autograd Gate**:
    - Must pass `scripts/ast_signature_and_length_linter.py` (`<= 1,000` SLOC soft limit, `<= 8` args per function, zero semicolons) and `src/qdlblocks/benchmarks/autograd_efficiency_auditor.py` (Rule 9).

### Lens 6: `egm-structural` (*Structural & Modularity Brainstormer*)
- **Clean Module Partitioning (Enforcing Rules 1, 2, 3, 6, 7, 8, 9)**:
  1. [`src/qdlblocks/layers/impure_density_mirror_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py):
     - Add `fast_rank_r_hermitian_and_ket_pack` and `extract_rank2_krylov_eigenpair_uhlmann`, plus optional `tikhonov_lambda` in `riemannian_stiefel_muon_step`.
  2. `src/qdlblocks/optim/vocab_manifold_muon.py` (New focused optimizer module, `~220` SLOC):
     - Implements `BatchSubspaceTikhonovGramMuon` (`(Q, N_act, A)` active-token Tikhonov Gram preconditioned Muon with sphere retraction) and `RiemannianSphereTangentSNROptimizer` (oblique sphere tangent momentum + parallel transport + SNR trust-region geodesic step).
  3. `src/qdlblocks/models/standalone_quantum_born_gpt.py` (New flag-free model module, `~550` SLOC):
     - Implements `VocabularyReadoutMode`, `VocabOptimizerMode`, `StandaloneQuantumBornConfig` (`<= 8` fields), `StandaloneQuantumBornBlock` (288-width fused attention + `@torch.compile` McWeeny & Uhlmann Rank-2 Krylov + 320-dim/head direct Stiefel readouts + Mach-Zehnder two-beam interference), `Rank2KrylovBornChunkedCEFn` (Rule 9 single-pass chunked autograd), `StandaloneQuantumBornGPT` (pure sequential quantum stack without `SmearGate` or `U-Net` skips), and `build_standalone_quantum_born_optimizers`.
  4. `examples/73_verify_plan063_standalone_quantum_born_gpt.py` (Verification & Dual-3090 FineWeb Speedrun Runner, `~650` SLOC):
     - Runs H1, H2, H3 ablations and the standalone `<= 330M`-token FineWeb training run on `2x RTX 3090`, logging via `StandardizedExperimentLogger` and evaluating on the `10,485,760`-token `fineweb_val_000000.bin` benchmark.
  5. `tests/test_plan063_standalone_quantum_born_gpt.py` (`~140` SLOC):
     - Deterministic unit tests for all new primitives.

---

## 4. Proposed Top-of-File Named Constants Ledger (Rule 7)

| Constant Name | Value | Mathematical / Physical Rationale |
| :--- | :---: | :--- |
| `DEFAULT_FUSED_HEAD_PACK_DIM` | `288` | Fused per-head attention GEMM width $d_h^2 + 2 d_h = 256 + 32 = 288$ ($9 \times 32$, Tensor-Core aligned) |
| `DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM` | `320` | Option B per-head observable readout width combining $\tilde{P}(\rho_{\text{out}}) \in \mathbb{R}^{256}$ + both Rank-2 Krylov eigenmodes $\in \mathbb{R}^{64}$ ($6 \times 320 = 1,920$ total) |
| `DEFAULT_TIKHONOV_GRAM_LAMBDA` | `1e-3` | Relative Tikhonov damping $\lambda_{\text{tik}}$ in Newton-Schulz and batch-subspace Gram preconditioning |
| `DEFAULT_UHLMANN_PHASE_EPS` | `1e-7` | Numerical floor for Uhlmann parallel-transport phase alignment $\langle \tilde{\phi}_1 \mid \bar{v}\rangle / |\langle \tilde{\phi}_1 \mid \bar{v}\rangle|$ |
| `DEFAULT_SPHERE_SNR_TRUST_TAU` | `0.50` | SNR trust-region half-saturation threshold $\tau_{\text{snr}}$ for `RiemannianSphereTangentSNROptimizer` |
| `DEFAULT_SPHERE_GEODESIC_LR` | `0.08` | Base angular step size $\eta_{\text{sphere}}$ (radians) on $\sqrt{d}\,\mathbb{S}^{d-1}$ for active vocabulary rows |
| `DEFAULT_BATCH_GRAM_MUON_LR` | `0.06` | Base step size for active-token Batch-Subspace $(Q, N_{\text{act}}, A)$ Tikhonov Gram Muon |
| `DEFAULT_DIRECT_READOUT_INIT_SIGMA` | `0.06` | Step-0 residual gain initialization for direct Stiefel readouts $W_{\text{obs}}, W_{\text{chan}}$ |

---

## 5. Recommended Operational Tooling & Script Gate (`AGENTS.md` Section 3)

1. **Existing Scripts to Use**:
   - `scripts/ast_signature_and_length_linter.py` (Enforce Rule 1 `<= 1,000` SLOC and Rule 2 `<= 8` args).
   - `scripts/verify_plan.py --plan-num 63` (Automated AST + pytest + hypothesis JSON verification).
   - `scripts/sync_waves_to_webui.py --trigger-reload` (Sync Wave 48 model cards and SQLite benchmark rows to WebUI).
2. **`Build-Before-Launch` Scripts**:
   - `examples/73_verify_plan063_standalone_quantum_born_gpt.py`: Supports both `--mode verify` (fast H1/H2/H3 verification on Dual RTX 3090s) and `--mode speedrun-330m` (standalone multi-GPU FineWeb training up to `330M` tokens evaluated on all `10,485,760` tokens of `fineweb_val_000000.bin`).
3. **`Nice-to-Have` Scripts (Surfaced for User Review)**:
   - `scripts/uhlmann_krylov_phase_and_vocab_snr_probe.py`: A standalone diagnostic CLI that measures (a) Uhlmann phase-locking stability under complex value ket rotations, (b) top-2 Krylov eigenvalue gap $\lambda_1 - \lambda_2$ across layers, and (c) per-decile vocabulary gradient SNR and sphere norm drift across `BatchSubspaceTikhonovGramMuon`, `RiemannianSphereTangentSNROptimizer`, and `AdamW`.
