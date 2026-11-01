# Wave 48 (`EXPERIMENT_PLAN_063`): Standalone Phase-Locked Krylov & Impure Density Quantum Born GPT

*   **Author:** QuantumDLBlocks Engineering & Research
*   **Status:** Approved
*   **Date:** 2026-10-30

--------------------------------------------------------------------------------

## Overview

Wave 48 ([`EXPERIMENT_PLAN_063.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_063.md)) specifies a standalone, end-to-end trainable Complex Quantum Born GPT (`StandaloneQuantumBornGPT`) evaluated on the `Modded-NanoGPT` `FineWeb` validation benchmark (`10,485,760` tokens from Shard 0 in [`fineweb_val_000000.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin)).

Prior waves ([`EXPERIMENT_PLAN_060.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_060.md)–[`EXPERIMENT_PLAN_062.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_062.md)) demonstrated that Rank-$R$ ($R=4$) impure density matrices with Isometric $\sqrt{2}$ Hermitian packing and Bures-Wasserstein McWeeny purification can mirror individual layers of a pre-trained classical transformer to `99.83%` cosine similarity. However, the Wave 47 model ([`RiemannianImpureDensityGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py)) could not train standalone from scratch because it relied on a pre-trained classical teacher checkpoint for its embeddings and periodically overwrote its internal Stiefel readout layers via Golub-Pereyra ridge regression. Furthermore, passing only quadratic outer products $\rho = |v\rangle\langle v|$ through causal attention erased all first-moment linear sign and complex phase information ($|v\rangle \mapsto e^{i\theta}|v\rangle$).

This design eliminates all classical teacher dependencies and ridge-regression refits, restores linear phase coherence via fused `288`-width Hermitian + first-moment ket attention and Uhlmann phase-locked Rank-2 Lanczos-Krylov extraction (`320`-dim/head internal readout), and benchmarks geometric vocabulary optimization on the `10,485,760`-token FineWeb validation set on dual `NVIDIA GeForce RTX 3090` (`24 GB`) GPUs.

## The Problem

Three structural defects prevent the existing [`RiemannianImpureDensityGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py) architecture from training standalone from Step 0 to competitive Cross-Entropy (`< 3.28` Val CE target within `<= 330M` tokens) on FineWeb:

1.  **Quadratic $U(1)$ Phase Erasure in Density-Matrix Attention**:
    *   In [`RiemannianImpureDensityBlock.compute_packed_density_features`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L438-L471), token value states are mapped exclusively into quadratic Hermitian outer products $\rho_{s,h}^{\text{tok}} = \sum_{r=1}^R p_{s,h,r} |\hat{v}_{s,h}^{(r)}\rangle\langle\hat{v}_{s,h}^{(r)}| \in \text{Herm}(\mathbb{C}^{16})$ (`256` real dimensions per head).
    *   Because $|v\rangle\langle v|$ is invariant under $|v\rangle \mapsto -|v\rangle$ and $|v\rangle \mapsto e^{i\theta}|v\rangle$, every token's linear sign and complex interferometric phase are destroyed before causal attention mixing.
    *   In addition, [`extract_rank2_krylov_eigenpair`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L162-L213) seeds its Lanczos-Krylov iteration from a real diagonal vector (`v0_re = sqrt(diag(rho_re)), v0_im = 0`), causing arbitrary eigenvector sign flips across tokens and steps.

2.  **Hidden Classical Teacher Dependency & Unconditioned 3-Factor SVD Readouts**:
    *   In [`examples/72_verify_plan062_riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/72_verify_plan062_riemannian_impure_density_gpt.py#L218-L328), training copied `wte` and `lm_head` from a `400M`-token [`CompiledLeaderboardClassicalGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py) checkpoint and invoked [`fit_stiefel_observables_polar_ridge`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L495-L555) every 20 steps against cached teacher activations while freezing `w_obs_u`, `w_obs_vt`, `w_chan_u`, and `w_chan_vt` during gradient steps.
    *   These readout weights were excluded from [`AllLayerRiemannianStiefelMuon`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L585-L693) because parameterizing a readout as $U \operatorname{diag}(\sigma) V^\top$ causes unweighted Newton-Schulz orthogonalization to amplify tail singular directions ($\sigma_{768} \approx 10^{-4}$) by $10^4\times$.
    *   Without a classical teacher, the model needs direct semi-orthogonal Stiefel readouts ($W_{\text{obs}} \in \text{St}(1920, 768)$, $W_{\text{chan}} \in \text{St}(1536, 768)$) with learnable residual gains updated at every step by Tikhonov-damped `RiemannianStiefelMuon`.

3.  **Vocabulary Boundary Gauge Drift and Rare-Token Gradient Noise at $V = 50,304$**:
    *   Freezing random Step-0 token embeddings (`InputBoundaryMode.ANCHORED_SPHERE`) prevents the model from learning token geometry (`> 7.0` Val CE), whereas dense $50,304 \times 768$ Newton-Schulz orthogonalization triggers three noise traps identified in [`Top-1_Accuracy_vs_Cross-Entropy_Divergence.pdf`](file:///home/nixos/.gemini/antigravity-cli/brain/f62ee9d6-5742-4eef-9d7b-167f825ed294/.user_uploaded/media_1793332634476_f0f8bf6a.pdf) (Pages 1–5): **Trap 1** (partition-function denominator bleed across all $50,304$ rows), **Trap 2** (Rank-1 single-occurrence rare-token gradient noise amplification under Newton-Schulz), and **Trap 3** (momentum ghost drift on the `~45,000` un-sampled tokens in each batch).
    *   Keeping `wte` (`nn.Embedding(50304, 768)`) and `lm_head` (`nn.Linear(768, 50304, bias=False)`) untied allows active-token manifold optimizers—`BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer`—to step `wte.weight` strictly on active input indices $\mathcal{I}_{\text{in}} = \operatorname{unique}(\text{idx})$ and `lm_head.weight` on active union indices $\mathcal{I}_{\text{act}} = \operatorname{unique}(\text{idx} \cup \text{targets})$ (or via `AdamW`), compared head-to-head against Euclidean `AdamW`.

## Goals

*   **G1 (Zero Teacher or Ridge-Regression Dependency)**: Train `StandaloneQuantumBornGPT` 100% end-to-end via backpropagation from random initialization on FineWeb shards without loading classical teacher checkpoints or running periodic Golub-Pereyra ridge regressions.
*   **G2 (Eliminate Quadratic Phase Erasure via Fused `288`-Width Packing & `320`-Width Option B Readout)**: Pack the $\sqrt{2}$-isometric Rank-$R$ Hermitian density matrix (`256` dims) and the first-moment coherent value ket (`32` dims) into a single `288`-width Tensor-Core attention GEMM per head, purify via Bures-Wasserstein McWeeny squaring, extract the Uhlmann phase-locked Rank-2 Lanczos-Krylov eigenpair $((\lambda_1, |\phi_1\rangle), (\lambda_2, |\phi_2\rangle))$ inside a `@torch.compile` helper (`bfloat16` matmuls, `float32` trace/norm reductions), and expose both eigenmodes alongside the unit-scale purified density (`256 + 64 = 320` dims/head, `1,920` total) to each layer's direct Stiefel readout $W_{\text{obs}} \in \text{St}(1920, 768)$.
*   **G3 (100% Backbone Coverage with Tikhonov-Damped `RiemannianStiefelMuon`)**: Update all 2D real Stiefel matrices (`w_q`, `w_k`, `w_v`, `w_obs`, `w_beam_a`, `w_beam_b`, `w_chan`) and 3D complex head unitaries (`head_unitary_re`, `head_unitary_im` $\in U(16)$) via geodesic retraction with Tikhonov Gram damping ($\lambda_{\text{tik}} = 10^{-3}$), maintaining manifold orthogonality error `< 1e-5`.
*   **G4 (Active-Token Vocabulary Optimization & Readout Comparison)**: Implement and benchmark `BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer` against `AdamW` across both the quantum `RANK2_KRYLOV_BORN` readout and the `SOFTCAP_LINEAR_CE` readout baseline.
*   **G5 (FineWeb Shard 0 Validation Benchmark)**: Execute controlled H1–H3 ablations and a standalone multi-GPU FineWeb training run (`<= 330M` token horizon on `2x RTX 3090`), evaluating Cross-Entropy on all `10,485,760` validation tokens of [`fineweb_val_000000.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin) toward the `< 3.28` Val CE target.

## Non-Goals

*   **NG1 (No Classical Teacher Distillation or Layer-Replacement Surgery)**: This wave does not load checkpoints from [`w45b_classical_teacher_ckpt50m_l12_d768_b262k`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/checkpoints_manifest.json) or fit ridge regressions against teacher activations.
*   **NG2 (No `SmearGate` or `U-Net` Skip Scaffolding in Wave 48)**: To evaluate the standalone quantum density and Rank-2 Krylov architecture without confounding shortcuts, lag-1 bigram `SmearGate` and encoder-to-decoder `U-Net` skip connections are omitted in this wave.
*   **NG3 (No Breaking Changes to Frozen Wave 47 Modules)**: Existing files [`src/qdlblocks/models/riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py) and [`tests/test_plan062_riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan062_riemannian_impure_density_gpt.py) remain unmodified so prior wave verification tests continue to pass.

## Open Questions

*   *None; all architectural and review findings (`C-B1`–`C-B3`, `C-M1`–`C-M5`, `C-N1`–`C-N2`) have been resolved and locked.*

## Critical User Journeys (CUJs)

### CUJ 1: Fast Multi-GPU Hypothesis Verification (`H1`, `H2`, `H3`) on FineWeb

*   **Given**: Dual `NVIDIA GeForce RTX 3090` (`24 GB`) GPUs and cached FineWeb binary shards in `/home/nixos/QuantumDLBlocks/.cache/fineweb_100M/`.
*   **When**: The researcher runs [`examples/73_verify_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py) `--mode verify`.
*   **Then**: The runner executes controlled screening ablations testing:
    1.  **H1**: Fused `288`-width attention + Uhlmann phase-locked Rank-2 Krylov (`320`-dim/head Option B readout, `QuantumAttentionMode.UHLMANN_KRYLOV_320`) vs. phase-erased `256`-width quadratic density attention (`QuantumAttentionMode.PHASE_ERASED_DENSITY_256`).
    2.  **H2**: Direct Stiefel readout + Tikhonov-damped `RiemannianStiefelMuon` vs. undamped Stiefel/Euclidean updates.
    3.  **H3**: Active-token `BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer` vs. `AdamW` across `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE` readouts.
    All telemetry and hypothesis verdicts are persisted via [`StandardizedExperimentLogger`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/benchmarks/experiment_logger.py).

### CUJ 2: Standalone FineWeb Speedrun Evaluation (`<= 330M` Tokens on `10,485,760` Validation Tokens)

*   **Given**: The best-performing standalone quantum configuration from CUJ 1 and `2x RTX 3090` GPUs.
*   **When**: The researcher runs [`examples/73_verify_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py) `--mode speedrun-330m`.
*   **Then**: `StandaloneQuantumBornGPT` trains from random Step-0 initialization across FineWeb training shards (`<= 330M` tokens) with zero teacher dependency and evaluates validation Cross-Entropy and Top-1 accuracy across the full `10,485,760`-token [`fineweb_val_000000.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin) shard.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### 1. System Data Flow & Block Architecture

`StandaloneQuantumBornGPT` is a sequential $L$-layer ($L = 12$, $d = 768$, $H = 6$ heads, complex head dimension $d_h = 16$, Rank-$R = 4$ value branches, channel hidden dimension $d_{\text{chan}} = 1536$) quantum transformer without `SmearGate` or `U-Net` skip connections. It retains the per-layer residual highway scalars `block_lambdas` $\in \mathbb{R}^2$ ($x_{\text{base}}^{(l)} = \lambda_{\text{blk},0}^{(l)} x^{(l)} + \lambda_{\text{blk},1}^{(l)} x_0$) and `v_lambdas` $\in \mathbb{R}^2$ ($V_{\text{mixed}}^{(l)} = \lambda_{v,0}^{(l)} (h_{\text{norm}}^{(l)} W_V^{(l)}) + \lambda_{v,1}^{(l)} V_0$) from [`RiemannianImpureDensityBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L406-L417).

```text
Token Indices idx (B, T)
   │
   ▼
[Untied Vocabulary Embedding wte: V=50304 x d=768] ──► RMS-Normalized x_0 (B, T, 768)
   │
   ▼  (For each Layer l = 0 .. L-1)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ StandaloneQuantumBornBlock (Layer l)                                                   │
│                                                                                        │
│ 1. Stiefel Q, K, V Projections:                                                        │
│    • q_re, q_im (B, H=6, T, d_h=16) via w_q ∈ St(768, 192) + Complex RoPE + QK-Norm    │
│    • k_re, k_im (B, H=6, T, d_h=16) via w_k ∈ St(768, 192) + Complex RoPE + QK-Norm    │
│    • v_re, v_im (B, H=6, T, R=4, d_h=16) via w_v ∈ St(768, 768) (+ V_0 v_lambdas mix)  │
│                                                                                        │
│ 2. Fused 288-Width Isometric Hermitian + First-Moment Ket Packing:                     │
│    • Rank-R=4 trace-1 density ρ_tok = Ṽ† Ṽ ∈ Herm(C^16) packed with √2 off-diagonal   │
│      isometry into P̃(ρ_tok) ∈ R^256                                                    │
│    • First-moment coherent value ket |v_coh⟩ ∈ C^16 packed into R^32                   │
│    • Concatenated per-head value feature Π_s,h = [P̃(ρ_tok), v_coh_re, v_coh_im] ∈ R^288│
│                                                                                        │
│ 3. Single Tensor-Core Causal p-Boson Born Attention GEMM (width = 288 = 9 * 32):       │
│    • Causal p-Boson Born Weights W_ts,h = |⟨q_t | k_s⟩_C|^{2p} / ∑_{s'≤t} |⟨q_t|k_s'⟩|^{2p}│
│    • Contextual mixed state M_t,h = ∑_{s≤t} W_ts,h Π_s,h = [P̃(ρ^(0)), v̄_re, v̄_im]     │
│                                                                                        │
│ 4. @torch.compile Fused McWeeny + Unitary U(16) + Uhlmann Rank-2 Krylov Extraction:    │
│    • Unpack P̃^-1(P̃(ρ^(0))) -> (ρ_re^(0), ρ_im^(0)) ∈ (B, H, T, 16, 16)                 │
│    • 1-Step Bures-Wasserstein McWeeny Purification (bfloat16 matmul, float32 trace):   │
│      ρ^(1) = (ρ^(0))^2 / (Tr((ρ^(0))^2) + ε)                                           │
│    • Per-head Complex Unitary Evolution U_h ∈ U(16):                                   │
│      ρ_out = U_h ρ^(1) U_h†,   |v̄_out⟩ = U_h |v̄⟩ / (‖v̄‖_2 + ε)                         │
│    • Uhlmann Phase-Locked Rank-2 Lanczos-Krylov Eigenpair (seeded by |v̄_out⟩):         │
│      - Top mode: |w_1⟩ = ρ_out |v̄_out⟩, |φ̃_1⟩ = |w_1⟩ / ‖w_1‖_2, λ_1 = ⟨φ̃_1|ρ_out|φ̃_1⟩│
│      - Uhlmann phase lock: |φ_1⟩ = (⟨φ̃_1 | v̄_out⟩ / |⟨φ̃_1 | v̄_out⟩|) |φ̃_1⟩             │
│      - Second orthogonal mode: |u_2⟩ = ρ_out |φ_1⟩ - λ_1 |φ_1⟩,                        │
│        |u_2^⊥⟩ = |u_2⟩ - ⟨φ_1 | u_2⟩ |φ_1⟩, |φ_2⟩ = |u_2^⊥⟩ / ‖u_2^⊥‖_2,               │
│        λ_2 = ⟨φ_2 | ρ_out | φ_2⟩                                                       │
│                                                                                        │
│ 5. Option B (320-Dim/Head) Direct Stiefel Observable Readout (Unit Isometric Scale):   │
│    • Per-head observable O_t,h = [P̃(ρ_out) (256), √λ_1 φ_1 (32), √λ_2 φ_2 (32)] ∈ R^320│
│    • Flatten 6 heads -> O_t ∈ R^1920                                                   │
│    • Direct Stiefel projection: Δx_attn = α_obs ⊙ (O_t @ W_obs), W_obs ∈ St(1920, 768) │
│                                                                                        │
│ 6. Mach-Zehnder Two-Beam Interferometer Channel Mixing (d_chan = 1536):                │
│    • Two-beam wide Stiefel up-projections w_beam_a, w_beam_b of shape (768, 1536)      │
│    • Polarization identity: h_int = ReLU(u_a + u_b)^2 - 0.25 * ReLU(u_a - u_b)^2       │
│    • Direct Stiefel down-projection: Δx_chan = α_chan ⊙ (h_int @ W_chan),              │
│      W_chan ∈ St(1536, 768)                                                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
[Final Untied Vocabulary Readout (Rule-9 Single-Pass Chunked Autograd, chunk_tokens = 8192)]
   ├─► Mode 1: VocabularyReadoutMode.RANK2_KRYLOV_BORN (Rank2KrylovBornChunkedCEFn)
   │     Partitions h_t = [h_1, h_2] ∈ R^384 ⊕ R^384 and lm_head E = [E_1, E_2] weighted by
   │     head-averaged final-layer Krylov (λ̂_1, λ̂_2):
   │     I_{t,v} = λ̂_{1,t} ⟨e_{1,v}, h_{1,t}⟩^2 + λ̂_{2,t} ⟨e_{2,v}, h_{2,t}⟩^2 + ε
   │     z_{t,v} = 15 * tanh((τ_born * log(I_{t,v})) / 15)
   └─► Mode 2: VocabularyReadoutMode.SOFTCAP_LINEAR_CE (_CompiledChunkedSoftcapLinearCEFn)
         Evaluates 15 * tanh((h_t @ E^T) / 15) Cross-Entropy via in-place Triton kernel
```

### 2. Core Mathematical Primitives

#### 2.1 Fused `288`-Width Isometric $\sqrt{2}$ Hermitian + First-Moment Ket Packing (`fast_rank_r_hermitian_and_ket_pack`)
Given complex value branches $v_{\text{re}}, v_{\text{im}} \in \mathbb{R}^{B \times H \times T \times R \times d_h}$ ($H=6, R=4, d_h=16$):
1.  **Branch Energy Normalization**:
    $$E_{b,h,t} = \sum_{r=1}^R \sum_{i=1}^{d_h} \left(v_{\text{re},b,h,t,r,i}^2 + v_{\text{im},b,h,t,r,i}^2\right) + \varepsilon, \qquad \tilde{v}_{b,h,t,r} = \frac{v_{b,h,t,r}}{\sqrt{E_{b,h,t}}}$$
2.  **Isometric $\sqrt{2}$ Hermitian Outer Product (`256` dims)**:
    Contract the $R=4$ branches via $(d_h \times R) \times (R \times d_h)$ batched matrix multiplication:
    $$\rho_{\text{re}}^{\text{tok}} = \tilde{v}_{\text{re}}^\top \tilde{v}_{\text{re}} + \tilde{v}_{\text{im}}^\top \tilde{v}_{\text{im}}, \qquad \rho_{\text{im}}^{\text{tok}} = \tilde{v}_{\text{im}}^\top \tilde{v}_{\text{re}} - \tilde{v}_{\text{re}}^\top \tilde{v}_{\text{im}}$$
    Apply precomputed upper/lower-triangular masks (`re_pack_mask`, `im_pack_mask` with off-diagonal scale $\sqrt{2}$) so $\tilde{P}(\rho^{\text{tok}}) \in \mathbb{R}^{256}$ satisfies exact Frobenius isometry $\langle \tilde{P}(A), \tilde{P}(B)\rangle_{\mathbb{R}^{256}} = \operatorname{Tr}(A B)$.
3.  **First-Moment Coherent Value Ket (`32` dims)**:
    Compute the coherent branch superposition $v^{\text{coh}} = \sum_{r=1}^R v_{b,h,t,r} \in \mathbb{C}^{16}$, normalize to unit complex norm $\hat{v}^{\text{coh}} = v^{\text{coh}} / (\|v^{\text{coh}}\|_2 + \varepsilon)$, and pack $[\hat{v}_{\text{re}}^{\text{coh}}, \hat{v}_{\text{im}}^{\text{coh}}] \in \mathbb{R}^{32}$.
4.  **Fused Output (`288` dims)**:
    Return $\Pi = [\tilde{P}(\rho^{\text{tok}}), \hat{v}_{\text{re}}^{\text{coh}}, \hat{v}_{\text{im}}^{\text{coh}}] \in \mathbb{R}^{B \times H \times T \times 288}$.

#### 2.2 Causal $p$-Boson Born Attention & `@torch.compile` Fused McWeeny + Unitary Evolution + Uhlmann Phase-Locked Rank-2 Krylov Extraction (`_compiled_mcweeny_uhlmann_krylov_step`)
1.  **Causal $p$-Boson Born Attention Weights ($p=2$)**:
    Matching [`RiemannianImpureDensityBlock.compute_packed_density_features`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py#L429-L436), compute complex inner products $\langle q_{t,h} \mid k_{s,h}\rangle_{\mathbb{C}} = (q_{\text{re},t,h}^\top k_{\text{re},s,h} + q_{\text{im},t,h}^\top k_{\text{im},s,h}) + i(q_{\text{re},t,h}^\top k_{\text{im},s,h} - q_{\text{im},t,h}^\top k_{\text{re},s,h})$ and form the causal $p$-boson ($p=2$) Born probability weights:
    $$W_{ts,h} = \frac{|\langle q_{t,h} \mid k_{s,h}\rangle_{\mathbb{C}}|^{2p} \mathbf{1}_{s \le t}}{\sum_{s' \le t} |\langle q_{t,h} \mid k_{s',h}\rangle_{\mathbb{C}}|^{2p} + \varepsilon}$$
    Execute a single batched Tensor-Core GEMM $M = W \Pi \in \mathbb{R}^{B \times H \times T \times 288}$, and slice $M$ into $\tilde{P}(\rho^{(0)}) \in \mathbb{R}^{B \times H \times T \times 256}$ and $\bar{v}_{\text{raw}} = (\bar{v}_{\text{re}}, \bar{v}_{\text{im}}) \in \mathbb{R}^{B \times H \times T \times 32}$.
2.  **Unpack & 1-Step McWeeny Purification (`bfloat16` GEMM, `float32` trace)**:
    Unpack $\tilde{P}(\rho^{(0)}) \to (\rho_{\text{re}}^{(0)}, \rho_{\text{im}}^{(0)}) \in \mathbb{R}^{B \times H \times T \times 16 \times 16}$ using [`unpack_hermitian_isometric`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L106-L117). Compute the complex square in `bfloat16`:
    $$S_{\text{re}} = (\rho_{\text{re}}^{(0)})^2 - (\rho_{\text{im}}^{(0)})^2, \qquad S_{\text{im}} = \rho_{\text{re}}^{(0)} \rho_{\text{im}}^{(0)} + \rho_{\text{im}}^{(0)} \rho_{\text{re}}^{(0)}$$
    Accumulate the trace in `float32`: $\tau = \sum_{i=1}^{16} S_{\text{re},ii} + \varepsilon$, and normalize $\rho^{(1)} = S / \tau$.
3.  **Per-Head Complex Unitary Evolution $U_h \in U(16)$**:
    Conjugate $\rho_{\text{out}} = U_h \rho^{(1)} U_h^\dagger$ and evolve the first-moment anchor ket $|\bar{v}_{\text{out}}\rangle = \operatorname{normalize}_{\mathbb{C}}(U_h |\bar{v}_{\text{raw}}\rangle)$.
4.  **Uhlmann Phase-Locked Rank-2 Lanczos-Krylov Eigenpair (`extract_rank2_krylov_eigenpair_uhlmann`)**:
    *   **Step 1 (Primary Krylov Vector)**: Apply $\rho_{\text{out}}$ to the anchor ket $|w_1\rangle = \rho_{\text{out}} |\bar{v}_{\text{out}}\rangle$ and normalize $|\tilde{\phi}_1\rangle = |w_1\rangle / (\|w_1\|_2 + \varepsilon)$.
    *   **Step 2 (Uhlmann Parallel-Transport Phase Lock)**: Compute the complex inner product $c = \langle \tilde{\phi}_1 \mid \bar{v}_{\text{out}}\rangle_{\mathbb{C}} = c_{\text{re}} + i c_{\text{im}}$ and rotate $|\tilde{\phi}_1\rangle$ by $c / (|c| + \varepsilon_{\text{uhl}})$ so $\langle \bar{v}_{\text{out}} \mid \phi_1\rangle_{\mathbb{C}} \in \mathbb{R}_{\ge 0}$:
        $$|\phi_1\rangle = \frac{\langle \tilde{\phi}_1 \mid \bar{v}_{\text{out}}\rangle_{\mathbb{C}}}{|\langle \tilde{\phi}_1 \mid \bar{v}_{\text{out}}\rangle_{\mathbb{C}}| + \varepsilon_{\text{uhl}}} |\tilde{\phi}_1\rangle$$
        This eliminates quadratic sign/phase ambiguity: if all input value kets rotate by $e^{i\theta}$, $|\bar{v}_{\text{out}}\rangle$ rotates by $e^{i\theta}$ and $|\phi_1\rangle$ rotates covariantly by $e^{i\theta}$.
    *   **Step 3 (Second Orthogonal Krylov Eigenmode)**: Compute Rayleigh quotient $\lambda_1 = \langle \phi_1 \mid \rho_{\text{out}} \mid \phi_1\rangle \in [0, 1]$, form the Lanczos residual $|u_2\rangle = \rho_{\text{out}} |\phi_1\rangle - \lambda_1 |\phi_1\rangle$, project out $|\phi_1\rangle$ via complex Gram-Schmidt $|u_2^\perp\rangle = |u_2\rangle - \langle \phi_1 \mid u_2\rangle_{\mathbb{C}} |\phi_1\rangle$, normalize $|\phi_2\rangle = |u_2^\perp\rangle / (\|u_2^\perp\|_2 + \varepsilon)$, and evaluate $\lambda_2 = \langle \phi_2 \mid \rho_{\text{out}} \mid \phi_2\rangle \in [0, 1]$.
5.  **Option B (`320`-Dim/Head) Observable Feature Packing at Unit Isometric Scale**:
    Concatenate the unit-isometric-scale packed density $\tilde{P}(\rho_{\text{out}}) \in \mathbb{R}^{256}$ ($\|\tilde{P}(\rho_{\text{out}})\|_2 = \sqrt{\operatorname{Tr}(\rho_{\text{out}}^2)} \le 1$, omitting Wave 47's `* 16.0` multiplier) with both eigenvalue-weighted Krylov eigenmodes $[\sqrt{\lambda_1}\phi_{1,\text{re}}, \sqrt{\lambda_1}\phi_{1,\text{im}}, \sqrt{\lambda_2}\phi_{2,\text{re}}, \sqrt{\lambda_2}\phi_{2,\text{im}}] \in \mathbb{R}^{64}$ to produce $O_{b,h,t} \in \mathbb{R}^{320}$ per head ($6 \times 320 = 1,920$ across heads), plus head-averaged Rayleigh quotients $(\bar{\lambda}_{1,b,t}, \bar{\lambda}_{2,b,t}) \in \mathbb{R}^{B \times T \times 2}$.

#### 2.3 Direct Stiefel Readout & Tikhonov-Damped `RiemannianStiefelMuon`
1.  **Direct Semi-Orthogonal Readout & Mach-Zehnder Parameterization**:
    Each block stores `w_obs` $\in \text{St}(1920, 768)$ ($W_{\text{obs}}^\top W_{\text{obs}} = I_{768}$) and a learnable vector gain `obs_scale` $\in \mathbb{R}^{768}$ initialized to `DEFAULT_DIRECT_READOUT_INIT_SIGMA = 0.06`. The Mach-Zehnder channel mixer uses wide Stiefel up-projections `w_beam_a`, `w_beam_b` of shape `(768, 1536)` ($W W^\top = I_{768}$), polarization activation $h_{\text{int}} = \operatorname{ReLU}(u_a + u_b)^2 - \frac{1}{4}\operatorname{ReLU}(u_a - u_b)^2$, and tall Stiefel down-projection `w_chan` $\in \text{St}(1536, 768)$ ($W_{\text{chan}}^\top W_{\text{chan}} = I_{768}$) with learnable vector gain `chan_scale` $\in \mathbb{R}^{768}$.
2.  **Tikhonov-Damped Tangent Newton-Schulz (`tikhonov_lambda = 1e-3`)**:
    Matching [`riemannian_stiefel_muon_step`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L266-L290), for any real Stiefel matrix $W \in \text{St}(m, n)$ ($m \ge n$; wide matrices with $m < n$ are transposed before and after):
    *   Project gradient $G$ and Nesterov momentum $G_{\text{nest}}$ onto the tangent space ($G_{\text{tan}} = G_{\text{nest}} - W(W^\top G_{\text{nest}})$ horizontal Grassmann projection for rectangular $m > n$, and $G_{\text{tan}} = G_{\text{nest}} - W \operatorname{sym}(W^\top G_{\text{nest}})$ for square $m = n$).
    *   Normalize $X_0 = G_{\text{tan}} / (\|G_{\text{tan}}\|_F + \varepsilon)$. In the first Newton-Schulz iteration of `_newton_schulz5_tall_real`, add Tikhonov Gram damping $\lambda_{\text{tik}} I_n$ to suppress tail-singular-value noise amplification:
        $$A_0 = X_0^\top X_0 + \lambda_{\text{tik}} I_n, \qquad X_1 = \frac{1}{2} X_0 (3 I_n - A_0)$$
        followed by 4 standard cubic Newton-Schulz steps to produce $U_{\text{tan}}$.
    *   Retract along the Stiefel geodesic $W_{\text{raw}} = (W - \eta U_{\text{tan}}) / \sqrt{1 + \eta^2}$ and apply the 1-step Schulz manifold polish $W_{\text{next}} = \frac{1}{2} W_{\text{raw}} (3 I_n - W_{\text{raw}}^\top W_{\text{raw}})$.
    The same Tikhonov Gram damping is applied to 3D complex head unitaries $U_h \in U(16)$ in `riemannian_complex_unitary_muon_step`.

#### 2.4 Active-Token Vocabulary Manifold Optimizers ([`src/qdlblocks/optim/vocab_manifold_muon.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py))
`wte` (`nn.Embedding(50304, 768)`) and `lm_head` (`nn.Linear(768, 50304, bias=False)`) are untied. `wte.weight` receives sparse non-zero gradients strictly on $\mathcal{I}_{\text{in}} = \operatorname{unique}(\text{idx})$ (`step_active`), while `lm_head.weight` receives dense partition-function repulsion across all $50,304$ rows (`step_full`):
1.  **`BatchSubspaceTikhonovGramMuon` (`VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON`)**:
    *   For `wte.weight` (`step_active`), slices active gradients $G_{\text{act}} = G[\mathcal{I}]$ and updates momentum $M_{\text{act}} = M[\mathcal{I}]$ strictly on $\mathcal{I}$ (preventing Trap 1 denominator bleed and Trap 3 momentum ghost drift on inactive rows). For `lm_head.weight` (`step_full`), updates all rows so non-target/padding rows receive repulsive partition-function gradients.
    *   Projects each momentum row onto the tangent space of the unit-RMS sphere $\sqrt{d}\,\mathbb{S}^{d-1}$ at $W$: $M_{\text{tan}} = M - \frac{\langle M, W\rangle}{d} W$, and computes per-row SNR trust-region gate $\gamma_v = \text{SNR}_v / (\text{SNR}_v + \tau_{\text{snr}})$.
    *   Forms the $d \times d$ ($768 \times 768$) trace-balanced Tikhonov Gram matrix ($\alpha_{\text{gram}} = \text{DEFAULT\_GRAM\_WHITENING\_MIX} = 0.25$):
        $$A = (1 - \alpha_{\text{gram}} + \lambda_{\text{tik}}) I_d + \alpha_{\text{gram}} \frac{M_{\text{tan}}^\top M_{\text{tan}}}{\operatorname{Tr}(M_{\text{tan}}^\top M_{\text{tan}}) / d + \varepsilon} \in \mathbb{R}^{d \times d}$$
        and computes $A^{-1/2}$ via 6-step Frobenius-normalized coupled Newton-Schulz (`_newton_schulz_coupled_inv_sqrt`) to precondition $U = M_{\text{tan}} A^{-1/2}$ without suppressing the rank-1 unigram carrier direction $\bar{h}$.
    *   Re-projects $U_{\text{tan}} = \Pi_{T_W}(U)$ and retracts each updated row onto $\sqrt{d}\,\mathbb{S}^{d-1}$ along its great circle with angular step $\theta_v = \eta_{\text{gram}} \gamma_v$.
2.  **`RiemannianSphereTangentSNROptimizer` (`VocabOptimizerMode.SPHERE_TANGENT_SNR_MUON`)**:
    *   Projects $g_v$ onto $T_{w_v}\mathbb{S}^{d-1}$ ($g_{v,\perp} = g_v - \frac{\langle w_v, g_v\rangle}{d} w_v$), updates tangent momentum $m_v \leftarrow \beta_1 m_v + (1 - \beta_1) g_{v,\perp}$ ($\beta_1 = 0.9$) and second moment $v_v \leftarrow \beta_2 v_v + (1 - \beta_2) \|g_{v,\perp}\|_2^2$ ($\beta_2 = 0.99$), parallel-transports $m_v$ to $T_{w_v}\mathbb{S}^{d-1}$, computes $\text{SNR}_v = \|m_v\|_2 / (\sqrt{v_v} + \varepsilon)$, gates the angular step $\theta_v = \eta_{\text{sphere}} \frac{\text{SNR}_v}{\text{SNR}_v + \tau_{\text{snr}}}$, and takes the great-circle step $w_v^{\text{next}} = \cos(\theta_v) w_v - \sin(\theta_v) \sqrt{d} \frac{m_v}{\|m_v\|_2 + \varepsilon}$.
3.  **`AdamW` Baseline (`VocabOptimizerMode.ADAMW_BASELINE`)**:
    *   Standard decoupled `AdamW` on `wte` and `lm_head` for controlled ablation.

#### 2.5 Rule-9 Single-Pass Chunked Vocabulary Readouts
1.  **`VocabularyReadoutMode.RANK2_KRYLOV_BORN` (`Rank2KrylovBornChunkedCEFn`)**:
    *   Partitions the final RMS-normalized hidden state $h_t = [h_{1,t}, h_{2,t}] \in \mathbb{R}^{384} \oplus \mathbb{R}^{384}$ and vocabulary matrix $E = [E_1, E_2] \in \mathbb{R}^{V \times 384} \oplus \mathbb{R}^{V \times 384}$ with zero extra parameter matrices, weighted by the normalized head-averaged Krylov Rayleigh quotients $(\hat{\lambda}_{1,t}, \hat{\lambda}_{2,t})$ forwarded from the final quantum block ($\hat{\lambda}_{m,t} = \bar{\lambda}_{m,t} / (\bar{\lambda}_{1,t} + \bar{\lambda}_{2,t} + \varepsilon)$).
    *   In chunks of `chunk_tokens = 8192`, computes normalized branch overlaps $\tilde{A}_{m,t,v} = \langle e_{m,v}, h_{m,t}\rangle / \sqrt{d/2}$, coherent homodyne Born intensity $I_{t,v} = \hat{\lambda}_{1,t} (1 + \frac{1}{2}\tilde{A}_{1,t,v})^2 + \hat{\lambda}_{2,t} (1 + \frac{1}{2}\tilde{A}_{2,t,v})^2 + \varepsilon_{\text{born}}$ ($\varepsilon_{\text{born}} = \text{DEFAULT\_BORN\_LOG\_INTENSITY\_EPS} = 0.25$), and softcapped log-Born logits:
        $$z_{t,v} = 15 \tanh\!\left(\frac{\tau_{\text{born}} \log(I_{t,v})}{15}\right), \qquad \tau_{\text{born}} = \text{DEFAULT\_KRYLOV\_BORN\_LOG\_TEMP} = 2.5$$
    *   Accumulates Cross-Entropy loss and analytical gradients $\mathrm{d}h_t, \mathrm{d}E, \mathrm{d}\hat{\lambda}_t$ in a **single pass** per chunk (`forward()` computes $\mathrm{d}h_t, \mathrm{d}E, \mathrm{d}\hat{\lambda}_t$ in-place and `backward()` scales by `grad_output`, executing zero GEMMs or power ops in `backward()` per [`AGENTS.md`](file:///home/nixos/QuantumDLBlocks/AGENTS.md) Rule 9).
2.  **`VocabularyReadoutMode.SOFTCAP_LINEAR_CE`**:
    *   Reuses [`_CompiledChunkedSoftcapLinearCEFn`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L329-L433) ($15 \tanh(z / 15)$) for direct comparison with the classical leaderboard baseline.

--------------------------------------------------------------------------------

## Alternatives Considered

### Alternative A: Option A (`288` Dims/Head) Internal Readout (Single Krylov Mode + Packed Density)

*   **Mechanism**: Concatenate the packed purified density matrix $\tilde{P}(\rho_{\text{out}}) \in \mathbb{R}^{256}$ with only the top Uhlmann phase-locked Krylov eigenket $\sqrt{\lambda_1}|\phi_1\rangle \in \mathbb{R}^{32}$ inside each layer (`288` dims/head, $W_{\text{obs}} \in \text{St}(1728, 768)$), reserving the second orthogonal Krylov eigenmode $|\phi_2\rangle$ strictly for the final vocabulary readout.
*   **Pros**: Saves `32` observable channels per head (`1,728` vs. `1,920` total width, `~10%` smaller $W_{\text{obs}}$ GEMM) and keeps the input and output attention widths symmetric at `288`.
*   **Cons**: Hides the orthogonal polysemy branch $\sqrt{\lambda_2}|\phi_2\rangle$ from intermediate layers' linear residual projections, forcing intermediate layers to reconstruct secondary eigenmodes purely from the quadratic entries of $\tilde{P}(\rho_{\text{out}})$ without Uhlmann phase information.
*   **Reason for Rejection**: Rejected in favor of **Option B (`320` dims/head)** so every layer's direct Stiefel readout has explicit linear access to both phase-locked Rank-2 Krylov eigenmodes $[\sqrt{\lambda_1}|\phi_1\rangle, \sqrt{\lambda_2}|\phi_2\rangle] \in \mathbb{R}^{64}$ alongside $\tilde{P}(\rho_{\text{out}}) \in \mathbb{R}^{256}$.

### Alternative B: 3-Factor SVD Readout Parameterization ($U \operatorname{diag}(\sigma) V^\top$) with Periodic Golub-Pereyra Ridge Refits

*   **Mechanism**: Retain Wave 47's factorization of each readout into `w_obs_u` $\in \text{St}(D_{\text{obs}}, 768)$, `obs_sigma` $\in \mathbb{R}^{768}$, and `w_obs_vt` $\in \text{St}(768, 768)$, periodically solving ridge regressions against cached targets.
*   **Pros**: Matches the exact SVD structure used when distilling from a pre-trained classical teacher in Wave 47.
*   **Cons**: Requires a pre-trained teacher checkpoint for ridge targets; when updated by gradient descent, small tail singular values ($\sigma_{768} \approx 10^{-4}$) amplify Newton-Schulz gradient noise on $U$ and $V^\top$ by $1/\sigma_d \approx 10^4\times$, and add $24$ redundant $768 \times 768$ GEMMs and orthogonalizations per step.
*   **Reason for Rejection**: Violates the standalone teacher-free requirement (`G1`) and destabilizes end-to-end gradient training. Replaced by a single direct Stiefel matrix $W_{\text{obs}} \in \text{St}(1920, 768)$ + learnable vector gain updated by Tikhonov-damped `RiemannianStiefelMuon`.

### Alternative C: Dense Full-Vocabulary ($50,304 \times 768$) Newton-Schulz Orthogonalization

*   **Mechanism**: Apply standard `RiemannianStiefelMuon` or dense Newton-Schulz across the entire $50,304 \times 768$ token embedding matrix `wte.weight` at every step.
*   **Pros**: Reuses the backbone Stiefel optimizer without active-token indexing.
*   **Cons**: Costs `~296 GFLOPs` per step for a single $50,304 \times 768$ 5-step Newton-Schulz pass, contaminates `~45,000` un-sampled token rows with momentum ghost updates or denominator bleed (Traps 1 & 3), and amplifies Rank-1 single-occurrence token gradients into high-norm noise (Trap 2).
*   **Reason for Rejection**: Computationally wasteful and statistically noisy under Zipfian token frequencies. Replaced by active-token `BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer` restricted to active indices.

### Alternative D: Including `SmearGate` and `U-Net` Encoder-Decoder Skip Connections in Wave 48

*   **Mechanism**: Include the `Modded-NanoGPT` lag-1 bigram `SmearGate` and encoder-to-decoder U-Net skip connections (`layers 0..5 -> 11..6`) inside `StandaloneQuantumBornGPT` immediately.
*   **Pros**: Provides `0.15–0.25` nats of inductive bias from classical leaderboard engineering.
*   **Cons**: Confounds the empirical attribution of whether validation Cross-Entropy gains come from the phase-locked Rank-2 Krylov quantum attention mechanism or from classical U-Net/bigram shortcuts.
*   **Reason for Rejection**: Deferred to future waves so Wave 48 isolates and characterizes the pure sequential quantum density + Rank-2 Krylov architecture first.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Performance & Latency**:
    *   Unpacking $16 \times 16$ Hermitian matrices, running 1-step McWeeny purification, and extracting the Uhlmann phase-locked Rank-2 Krylov eigenpair must execute inside a `@torch.compile` helper (`_compiled_mcweeny_uhlmann_krylov_step`) using `bfloat16` batched matrix multiplications and `float32` trace/norm reductions to avoid eager kernel launch bottlenecks across 12 layers.
    *   Vocabulary readout (`RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE`) must execute in chunks of `chunk_tokens = 8192` with single-pass custom autograd backward loops (**Rule 9**), keeping peak activation VRAM within `24 GB` per RTX 3090 GPU at `B = 8..16, T = 1024`.
*   **Code Quality & Repository Invariants ([`AGENTS.md`](file:///home/nixos/QuantumDLBlocks/AGENTS.md))**:
    *   All modified and newly created Python files must pass [`scripts/ast_signature_and_length_linter.py`](file:///home/nixos/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) (`<= 1,000` SLOC soft limit and zero semicolons per Rule 1, `<= 8` parameters per function/dataclass per Rule 2, strategy/enum dispatch per Rule 3, top-of-file `UPPER_SNAKE_CASE` constants per Rule 7).
    *   Frozen Wave 47 modules ([`src/qdlblocks/models/riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py)) must remain untouched (Rule 6).

--------------------------------------------------------------------------------

### Additional Resources

*   Attached Research Reference: [*"Top-1 Accuracy vs Cross-Entropy Divergence"*](file:///home/nixos/.gemini/antigravity-cli/brain/f62ee9d6-5742-4eef-9d7b-167f825ed294/.user_uploaded/media_1793332634476_f0f8bf6a.pdf) (`Top-1_Accuracy_vs_Cross-Entropy_Divergence.pdf`, Pages 1–37)
*   Prior Wave Plans: [`EXPERIMENT_PLAN_060.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_060.md), [`EXPERIMENT_PLAN_061.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_061.md), [`EXPERIMENT_PLAN_062.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_062.md)
*   Existing Complex Packed Prototype: [`AgyRewrittenGeminiBornComplexPackedWithRoPE.py`](file:///home/nixos/QuantumDLBlocks/AgyRewrittenGeminiBornComplexPackedWithRoPE.py)
*   Existing Classical Leaderboard Baseline: [`src/qdlblocks/models/compiled_leaderboard_classical_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py)

### Future Improvements

*   **Scaling Capacity & Hybrid Scaffolding**: Once the baseline sequential `StandaloneQuantumBornGPT` scaling curve is established on FineWeb Shard 0, evaluate wider/deeper quantum stacks ($L = 16\text{–}24$ or $d = 1024$) and optionally re-introduce `SmearGate` and `U-Net` skip connections in a subsequent wave to test additive gains toward `< 3.28` Val CE.
*   **Fused Triton Kernel for $16 \times 16$ McWeeny + Krylov**: If `@torch.compile` Inductor fusion leaves residual shared-memory traffic on the $16 \times 16$ head density matrices, a custom warp-level register Triton kernel can hold each $16 \times 16$ Hermitian matrix entirely in SRAM registers.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Target Files to Create/Modify

#### 1. [`/home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_063.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_063.md) (Create)

*   **Type / Scope**: Wave 48 Experiment Plan & Hypothesis Ledger
*   **Rationale**: Synchronizes the formal Wave 48 experiment hypotheses (`H1`, `H2`, `H3`, `H4`), top-of-file named constants (Rule 7), planned changes, testing gates, and operational tooling with [`scripts/verify_plan.py`](file:///home/nixos/QuantumDLBlocks/scripts/verify_plan.py) (`REQUIRED_PLAN_HEADINGS = ("Hypotheses", "Planned", "Testing", "Operational Tooling")`) and [`StandardizedExperimentLogger`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/benchmarks/experiment_logger.py).
*   **Proposed Changes**:
    *   Include the 5 mandatory sections enforced by `scripts/verify_plan.py` and `AGENTS.md`:
        1. `## 1. Executive Summary & Architectural Motivation`
        2. `## 2. Quantitative Hypotheses (H1, H2, H3, H4)`
        3. `## 3. Top-of-File Named Constants Ledger (Rule 7)` (`DEFAULT_FUSED_HEAD_PACK_DIM = 288`, `DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM = 320`, `DEFAULT_TIKHONOV_GRAM_LAMBDA = 1e-3`, `DEFAULT_UHLMANN_PHASE_EPS = 1e-7`, `DEFAULT_SPHERE_SNR_TRUST_TAU = 0.50`, `DEFAULT_SPHERE_GEODESIC_LR = 0.08`, `DEFAULT_BATCH_GRAM_MUON_LR = 0.06`, `DEFAULT_DIRECT_READOUT_INIT_SIGMA = 0.06`, `DEFAULT_BORN_READOUT_TEMP = 4.0`)
        4. `## 4. Planned Changes`
        5. `## 5. Operational Tooling, Script Opportunities, & Testing` (`Existing Scripts Used`, `Build-Before-Launch Scripts`, `Nice-to-Have Scripts`).

#### 2. [`/home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py) & [`/home/nixos/QuantumDLBlocks/src/qdlblocks/layers/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/__init__.py) (Modify)

*   **Type / Scope**: Core Quantum Density, Krylov, & Stiefel/Unitary Optimizer Primitives (`<= 1,000` SLOC, `<= 8` args per function)
*   **Rationale**: Promotes reusable fused packing, Uhlmann phase-locked Rank-2 Krylov extraction, Tikhonov-damped Stiefel Newton-Schulz, and complex $U(16)$ unitary Muon into the shared layer library without mutating frozen [`riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py).
*   **Proposed Changes**:
    *   Add top-of-file constants `DEFAULT_FUSED_HEAD_PACK_DIM = 288`, `DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM = 320`, `DEFAULT_TIKHONOV_GRAM_LAMBDA = 1e-3`, and `DEFAULT_UHLMANN_PHASE_EPS = 1e-7`.
    *   Add `fast_rank_r_hermitian_and_ket_pack(v_re, v_im, re_pack_mask, im_pack_mask, eps=TRACE_NORM_EPS) -> Tensor` (`5` args): Contracts Rank-$R$ complex value branches into an isometric $\sqrt{2}$-scaled Hermitian vector (`256` dims) and concatenates the unit-norm first-moment coherent value ket (`32` dims) into a single `288`-wide tensor `(B, H, T, 288)`.
    *   Add `extract_rank2_krylov_eigenpair_uhlmann(rho_re, rho_im, v_bar_re, v_bar_im, eps=DEFAULT_UHLMANN_PHASE_EPS) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]` (`5` args): Seeds 2-step Lanczos-Krylov extraction from the contextual first-moment anchor ket $|\bar{v}\rangle$, applies the Uhlmann parallel-transport phase lock $|\phi_1\rangle = (\langle \tilde{\phi}_1 \mid \bar{v}\rangle / |\langle \tilde{\phi}_1 \mid \bar{v}\rangle|) |\tilde{\phi}_1\rangle$, and extracts the orthogonal second eigenmode $|\phi_2\rangle$ and Rayleigh quotients $(\lambda_1, \lambda_2)$.
    *   Update `_newton_schulz5_tall_real(mat, steps=DEFAULT_NS5_STEPS, tikhonov_lambda=0.0)` (`3` args) and `riemannian_stiefel_muon_step(w, grad, momentum_buf, lr=0.02, beta=DEFAULT_STIEFEL_MOMENTUM_BETA, ns_steps=DEFAULT_NS5_STEPS, tikhonov_lambda=0.0)` (`7` args) to support Tikhonov Gram damping $A_0 = X_0^\top X_0 + \lambda_{\text{tik}} I_n$ in the first Newton-Schulz iteration.
    *   Promote `_complex_newton_schulz5(g_re, g_im, steps=DEFAULT_NS5_STEPS, eps=TRACE_NORM_EPS, tikhonov_lambda=0.0)` (`5` args) and `riemannian_complex_unitary_muon_step(u_re, u_im, grad_re, grad_im, mom_re, mom_im, lr=0.02, tikhonov_lambda=0.0)` (`8` args, batched over 3D `(H, 16, 16)`) into `impure_density_mirror_block.py` and re-export in `src/qdlblocks/layers/__init__.py`.

#### 3. [`/home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py) & [`/home/nixos/QuantumDLBlocks/src/qdlblocks/optim/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/__init__.py) (Create / Modify)

*   **Type / Scope**: Active-Token Vocabulary Manifold Optimizers (`~220` SLOC)
*   **Rationale**: Isolates the two active-token vocabulary manifold optimizers (`BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer`) so they update strictly active token rows without polluting inactive vocabulary rows or exceeding SLOC limits.
*   **Proposed Changes**:
    *   Implement `_newton_schulz_coupled_inv_sqrt(gram_a, steps=5) -> Tensor`: Computes $A^{-1/2}$ on the $d \times d$ ($768 \times 768$) Tikhonov-damped active-token feature Gram matrix $A = \tilde{M}_{\text{tan}}^\top \tilde{M}_{\text{tan}} + \lambda_{\text{tik}} I_d$ via $Y_0 = A, Z_0 = I_d$, $T_k = \frac{1}{2}(3 I_d - Z_k Y_k), Y_{k+1} = Y_k T_k, Z_{k+1} = T_k Z_k$.
    *   Implement `BatchSubspaceTikhonovGramMuon(torch.optim.Optimizer)`:
        *   `step_active(active_idx: Tensor)`: Slices $G[\mathcal{I}]$, updates momentum strictly on $\mathcal{I}$, projects onto the tangent space of $\sqrt{d}\,\mathbb{S}^{d-1}$, preconditions via $U_{\text{act}} = \tilde{M}_{\text{tan}} A^{-1/2}$, re-projects onto $T_W \mathbb{S}^{d-1}$, and retracts updated rows onto $\sqrt{d}\,\mathbb{S}^{d-1}$.
    *   Implement `RiemannianSphereTangentSNROptimizer(torch.optim.Optimizer)`:
        *   `step_active(active_idx: Tensor)`: Projects active row gradients onto $T_{w_v}\mathbb{S}^{d-1}$, updates tangent momentum $m_v$ and second moment $v_v$, parallel-transports $m_v$ back to $T_{w_v}\mathbb{S}^{d-1}$, gates the angular geodesic step size $\theta_v = \eta_{\text{sphere}} \frac{\text{SNR}_v}{\text{SNR}_v + \tau_{\text{snr}}}$, and applies the great-circle retraction $w_v^{\text{next}} = \cos(\theta_v) w_v - \sin(\theta_v) \sqrt{d} \frac{m_v}{\|m_v\|_2 + \varepsilon}$.
    *   Export both optimizers in `src/qdlblocks/optim/__init__.py`.

#### 4. [`/home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py) & [`/home/nixos/QuantumDLBlocks/src/qdlblocks/models/__init__.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/__init__.py) (Create / Modify)

*   **Type / Scope**: Flag-Free Standalone Quantum Born GPT Model & Readouts (`~580` SLOC)
*   **Rationale**: Enforces Rule 6 (no boolean experiment flags added to frozen `riemannian_impure_density_gpt.py`) by providing a flag-free, standalone, teacher-free model with Option B (`320`-dim/head) direct Stiefel readouts, `@torch.compile` fused McWeeny + Uhlmann Krylov step, and Rule-9 single-pass chunked readouts.
*   **Proposed Changes**:
    *   Define `QuantumAttentionMode(str, Enum)` (`UHLMANN_KRYLOV_320`, `PHASE_ERASED_DENSITY_256`), `VocabularyReadoutMode(str, Enum)` (`RANK2_KRYLOV_BORN`, `SOFTCAP_LINEAR_CE`), and `VocabOptimizerMode(str, Enum)` (`BATCH_SUBSPACE_GRAM_MUON`, `SPHERE_TANGENT_SNR_MUON`, `ADAMW_BASELINE`).
    *   Define `StandaloneQuantumBornConfig` (`8` fields per Rule 2: `vocab_size=50304`, `n_layer=12`, `n_head=6`, `n_embd=768`, `v_rank=4`, `attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320`, `readout_mode=VocabularyReadoutMode.RANK2_KRYLOV_BORN`, `vocab_optim_mode=VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON`; complex head dimension is derived as `n_embd // (2 * 4 * n_head) = 16` or `DEFAULT_COMPLEX_HEAD_DIM = 16`).
    *   Implement `_compiled_mcweeny_uhlmann_krylov_step(packed_mixed, u_re, u_im, re_pack_mask, im_pack_mask) -> tuple[Tensor, Tensor]` (`5` args): Decorated with `@torch.compile` (falling back to eager execution when `torch.compile` is disabled or running on CPU), executes $16 \times 16$ unpacking via `unpack_hermitian_isometric`, `bfloat16` McWeeny squaring with `float32` trace normalization, $U(16)$ unitary conjugation, Uhlmann phase-locked Rank-2 Krylov extraction, and Option B `320`-dim/head unit-scale observable concatenation plus head-averaged Rayleigh quotients $(\bar{\lambda}_1, \bar{\lambda}_2)$.
    *   Implement `StandaloneQuantumBornBlock(nn.Module)`: Sequential quantum block with `w_q`, `w_k` $\in \text{St}(768, 192)$, `w_v` $\in \text{St}(768, 768)$, `block_lambdas`, `v_lambdas`, direct Stiefel readout `w_obs` $\in \text{St}(1920, 768)$ (or $\text{St}(1536, 768)$ in `PHASE_ERASED_DENSITY_256` mode) + `obs_scale`, complex unitary `(head_unitary_re, head_unitary_im)` $\in U(16)$, and Mach-Zehnder two-beam interferometer `w_beam_a`, `w_beam_b` of shape `(768, 1536)`, `w_chan` $\in \text{St}(1536, 768)$ + `chan_scale`.
    *   Implement `Rank2KrylovBornChunkedCEFn(torch.autograd.Function)` (`forward(ctx, h_norm, krylov_weights, lm_weight, targets, chunk_tokens=8192, temp=4.0)` — `6` args): Rule-9 single-pass chunked Rank-2 Krylov Born vocabulary loss and analytical in-place gradient accumulation.
    *   Implement `StandaloneQuantumBornGPT(nn.Module)` and `StandaloneRiemannianOptimizerBundle`: Coordinates Tikhonov-damped `RiemannianStiefelMuon` on 100% of 2D/3D backbone weights alongside the selected `VocabOptimizerMode` on untied `wte` ($\mathcal{I}_{\text{in}}$) and `lm_head` ($\mathcal{I}_{\text{act}}$), and export in `src/qdlblocks/models/__init__.py`.

#### 5. [`/home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py) (Create)

*   **Type / Scope**: Wave 48 Verification & Dual-RTX 3090 FineWeb Speedrun Runner (`~650` SLOC)
*   **Rationale**: Provides the automated verification harness for `H1`, `H2`, `H3`, and `H4` and the multi-GPU (`2x RTX 3090`) standalone FineWeb training loop evaluated on the `10,485,760`-token `fineweb_val_000000.bin` Shard 0 validation set.
*   **Proposed Changes**:
    *   Implement `verify_h1_uhlmann_krylov_phase_and_ablation()`: Verifies phase covariance of `UHLMANN_KRYLOV_320` and compares validation CE against `PHASE_ERASED_DENSITY_256` on FineWeb.
    *   Implement `verify_h2_direct_stiefel_tikhonov_muon()`: Compares direct Stiefel readouts updated with Tikhonov-damped `RiemannianStiefelMuon` ($\lambda_{\text{tik}} = 10^{-3}$) against undamped updates and verifies Stiefel/Unitary manifold error `< 1e-5`.
    *   Implement `verify_h3_active_vocab_manifold_optimizers()`: Benchmarks `BatchSubspaceTikhonovGramMuon`, `RiemannianSphereTangentSNROptimizer`, and `AdamW` across `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE` readouts on FineWeb Shard 0.
    *   Implement `run_standalone_fineweb_speedrun()`: Executes multi-GPU (`2x RTX 3090`) standalone training across FineWeb shards up to the target token horizon and evaluates full `10,485,760`-token Shard 0 validation CE.

#### 6. [`/home/nixos/QuantumDLBlocks/tests/test_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan063_standalone_quantum_born_gpt.py) (Create)

*   **Type / Scope**: Deterministic Pytest Suite (`~150` SLOC)
*   **Rationale**: Validates mathematical invariants, phase covariance, active-token isolation, manifold orthogonality, and Rule-9 autograd efficiency in `< 15 seconds`.
*   **Proposed Changes**:
    *   `test_fused_288_hermitian_and_ket_pack_isometry_and_phase`: Asserts `< 1e-6` agreement between single-GEMM `288`-width mixing and split density+ket mixing, and verifies that global phase rotation $|v\rangle \mapsto e^{i\theta}|v\rangle$ rotates $|\phi_1\rangle$ covariantly (eliminating quadratic phase erasure).
    *   `test_tikhonov_riemannian_stiefel_and_unitary_muon_stability`: Asserts `< 1e-5` Stiefel and Unitary manifold error over multi-step gradient updates on direct `w_obs` $\in \text{St}(1920, 768)$ and $U(16)$ parameters without SVD ridge refits.
    *   `test_batch_subspace_gram_muon_and_sphere_tangent_snr_active_isolation`: Asserts zero parameter or momentum drift on inactive token rows ($v \notin \mathcal{I}$) and unit-RMS sphere preservation $\|w_v\|_2 = \sqrt{d}$ on active rows.
    *   `test_standalone_quantum_born_gpt_end_to_end_and_rule9_autograd`: Verifies teacher-free loss decrease from Step 0 for both `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE` readouts and asserts Rule-9 single-pass backward operation counts (`bwd_pow == 0`, `bwd_mm == 0` inside `Rank2KrylovBornChunkedCEFn.backward`).

### Verification Plan

*   **Static Analysis & Linting**:
    *   `/home/nixos/scratch/.venv/bin/python scripts/ast_signature_and_length_linter.py src/qdlblocks/layers/impure_density_mirror_block.py src/qdlblocks/optim/vocab_manifold_muon.py src/qdlblocks/models/standalone_quantum_born_gpt.py examples/73_verify_plan063_standalone_quantum_born_gpt.py tests/test_plan063_standalone_quantum_born_gpt.py`
*   **Unit Tests (Wave 48 + Regression Check on Wave 47)**:
    *   `/home/nixos/scratch/.venv/bin/pytest tests/test_plan063_standalone_quantum_born_gpt.py tests/test_plan062_riemannian_impure_density_gpt.py -q`
*   **Automated Plan Gate**:
    *   `/home/nixos/scratch/.venv/bin/python scripts/verify_plan.py --plan-num 63`
*   **E2E FineWeb Verification & Multi-GPU Speedrun**:
    *   `/home/nixos/scratch/.venv/bin/python examples/73_verify_plan063_standalone_quantum_born_gpt.py --mode verify`
    *   `/home/nixos/scratch/.venv/bin/python examples/73_verify_plan063_standalone_quantum_born_gpt.py --mode speedrun-330m`

--------------------------------------------------------------------------------

<!-- Provenance: Indicates initial structural drafting via EGM. -->

*Drafted with EGM*
