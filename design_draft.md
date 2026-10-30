# Wave 48 (`EXPERIMENT_PLAN_063`): Standalone Phase-Locked Krylov & Impure Density Quantum Born GPT

*   **Author:** QuantumDLBlocks Engineering & Research
*   **Status:** Draft
*   **Date:** 2026-10-30

--------------------------------------------------------------------------------

## Overview

Wave 48 (`EXPERIMENT_PLAN_063`) specifies a standalone, end-to-end trainable Complex Quantum Born GPT (`StandaloneQuantumBornGPT`) evaluated on the `Modded-NanoGPT` `FineWeb` validation benchmark (`10,485,760` tokens from Shard 0 in [`fineweb_val_000000.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin)).

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
    *   These readout weights were excluded from `AllLayerRiemannianStiefelMuon` because parameterizing a readout as $U \operatorname{diag}(\sigma) V^\top$ causes unweighted Newton-Schulz orthogonalization to amplify tail singular directions ($\sigma_{768} \approx 10^{-4}$) by $10^4\times$.
    *   Without a classical teacher, the model needs direct semi-orthogonal Stiefel readouts ($W_{\text{obs}} \in \text{St}(1920, 768)$, $W_{\text{chan}} \in \text{St}(d_{\text{chan}}, 768)$) with learnable residual gains updated at every step by Tikhonov-damped `RiemannianStiefelMuon`.

3.  **Vocabulary Boundary Gauge Drift and Rare-Token Gradient Noise at $V = 50,304$**:
    *   Freezing random Step-0 token embeddings (`InputBoundaryMode.ANCHORED_SPHERE`) prevents the model from learning token geometry (`> 7.0` Val CE), whereas dense $50,304 \times 768$ Newton-Schulz orthogonalization mixes `~45,000` inactive tokens with the $N_{\text{act}} \le 8,192$ active tokens in each batch and amplifies Rank-1 rare-token gradient noise.
    *   Active-token manifold optimizers—specifically Batch-Subspace $(Q, N_{\text{act}}, A)$ Tikhonov Gram Muon and Riemannian Sphere Tangent Momentum with SNR trust-region gating—must be restricted to active token indices $\mathcal{I}_{\text{act}} = \operatorname{unique}(\text{idx} \cup \text{targets})$ and evaluated alongside Euclidean `AdamW`.

## Goals

*   **G1 (Zero Teacher or Ridge-Regression Dependency)**: Train `StandaloneQuantumBornGPT` 100% end-to-end via backpropagation from random initialization on FineWeb shards without loading classical teacher checkpoints or running periodic Golub-Pereyra ridge regressions.
*   **G2 (Eliminate Quadratic Phase Erasure via Fused `288`-Width Packing & `320`-Width Option B Readout)**: Pack the $\sqrt{2}$-isometric Rank-$R$ Hermitian density matrix (`256` dims) and the first-moment coherent value ket (`32` dims) into a single `288`-width Tensor-Core attention GEMM per head, purify via Bures-Wasserstein McWeeny squaring, extract the Uhlmann phase-locked Rank-2 Lanczos-Krylov eigenpair $((\lambda_1, |\phi_1\rangle), (\lambda_2, |\phi_2\rangle))$ inside a `@torch.compile` helper (`bfloat16` matmuls, `float32` trace/norm reductions), and expose both eigenmodes alongside the purified density (`256 + 64 = 320` dims/head, `1,920` total) to each layer's direct Stiefel readout $W_{\text{obs}} \in \text{St}(1920, 768)$.
*   **G3 (100% Backbone Coverage with Tikhonov-Damped `RiemannianStiefelMuon`)**: Update all 2D real Stiefel matrices (`w_q`, `w_k`, `w_v`, `w_obs`, `w_beam_a`, `w_beam_b`, `w_chan`) and 3D complex head unitaries (`head_unitary_re`, `head_unitary_im` $\in U(16)$) via geodesic retraction with Tikhonov Gram damping ($\lambda_{\text{tik}} = 10^{-3}$), maintaining manifold orthogonality error `< 1e-5`.
*   **G4 (Active-Token Vocabulary Optimization & Readout Comparison)**: Implement and benchmark `BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer` (updating strictly active token rows $\mathcal{I}_{\text{act}}$) against `AdamW` across both the quantum `RANK2_KRYLOV_BORN` readout and the `SOFTCAP_LINEAR_CE` readout baseline.
*   **G5 (FineWeb Shard 0 Validation Benchmark)**: Execute controlled H1–H3 ablations and a standalone multi-GPU FineWeb training run (`<= 330M` token horizon on `2x RTX 3090`), evaluating Cross-Entropy on all `10,485,760` validation tokens of [`fineweb_val_000000.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin) toward the `< 3.28` Val CE target.

## Non-Goals

*   **NG1 (No Classical Teacher Distillation or Layer-Replacement Surgery)**: This wave does not load checkpoints from [`w45b_classical_teacher_ckpt50m_l12_d768_b262k`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/checkpoints_manifest.json) or fit ridge regressions against teacher activations.
*   **NG2 (No `SmearGate` or `U-Net` Skip Scaffolding in Wave 48)**: To keep the standalone quantum density and Rank-2 Krylov architecture clean and interpretable while characterizing its native scaling dynamics, lag-1 bigram `SmearGate` and encoder-to-decoder `U-Net` skip connections are omitted in this wave.
*   **NG3 (No Breaking Changes to Frozen Wave 47 Modules)**: Existing files [`src/qdlblocks/models/riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py) and [`tests/test_plan062_riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan062_riemannian_impure_density_gpt.py) remain unmodified so prior wave verification tests continue to pass.

## Open Questions

*   *None remaining from brainstorming; all architectural choices (Option B `320`-dim/head readout, sequential stack without `SmearGate`/`U-Net`, `@torch.compile` fusion with `bfloat16` matmuls and `float32` trace/norm reductions) have been locked.*

## Critical User Journeys (CUJs)

### CUJ 1: Fast Multi-GPU Hypothesis Verification (`H1`, `H2`, `H3`) on FineWeb

*   **Given**: Dual `NVIDIA GeForce RTX 3090` (`24 GB`) GPUs and cached FineWeb binary shards in `/home/nixos/QuantumDLBlocks/.cache/fineweb_100M/`.
*   **When**: The researcher runs `examples/73_verify_plan063_standalone_quantum_born_gpt.py --mode verify`.
*   **Then**: The runner executes controlled screening ablations testing:
    1.  **H1**: Fused `288`-width attention + Uhlmann phase-locked Rank-2 Krylov (`320`-dim/head Option B readout) vs. phase-erased `256`-width quadratic density attention.
    2.  **H2**: Direct Stiefel readout + Tikhonov-damped `RiemannianStiefelMuon` vs. undamped Stiefel/Euclidean updates.
    3.  **H3**: Active-token `BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer` vs. `AdamW` across `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE` readouts.
    All telemetry and hypothesis verdicts are persisted via `StandardizedExperimentLogger`.

### CUJ 2: Standalone FineWeb Speedrun Evaluation (`<= 330M` Tokens on `10,485,760` Validation Tokens)

*   **Given**: The best-performing standalone quantum configuration from CUJ 1 and `2x RTX 3090` GPUs.
*   **When**: The researcher runs `examples/73_verify_plan063_standalone_quantum_born_gpt.py --mode speedrun-330m`.
*   **Then**: `StandaloneQuantumBornGPT` trains from random Step-0 initialization across FineWeb training shards (`<= 330M` tokens) with zero teacher dependency and evaluates validation Cross-Entropy and Top-1 accuracy across the full `10,485,760`-token [`fineweb_val_000000.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin) shard.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### 1. System Data Flow & Block Architecture

`StandaloneQuantumBornGPT` is a sequential $L$-layer ($L = 12$, $d = 768$, $H = 6$ heads, complex head dimension $d_h = 16$, Rank-$R = 4$ value branches) quantum transformer without `SmearGate` or `U-Net` skip connections. It preserves a clean residual highway with a learnable $V_0$ first-layer value residual mix on interior layers.

```text
Token Indices idx (B, T)
   │
   ▼
[Vocabulary Embedding wte: V=50304 x d=768] ──► Sphere / RMS-Normalized x_0 (B, T, 768)
   │
   ▼  (For each Layer l = 0 .. L-1)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ StandaloneQuantumBornBlock (Layer l)                                                   │
│                                                                                        │
│ 1. Stiefel Q, K, V Projections:                                                        │
│    • q_re, q_im (B, H=6, T, d_h=16) via w_q ∈ St(768, 192) + Complex RoPE + QK-Norm    │
│    • k_re, k_im (B, H=6, T, d_h=16) via w_k ∈ St(768, 192) + Complex RoPE + QK-Norm    │
│    • v_re, v_im (B, H=6, T, R=4, d_h=16) via w_v ∈ St(768, 768) (+ V_0 highway mix)    │
│                                                                                        │
│ 2. Fused 288-Width Isometric Hermitian + First-Moment Ket Packing:                     │
│    • Rank-R=4 trace-1 density ρ_tok = Ṽ† Ṽ ∈ Herm(C^16) packed with √2 off-diagonal   │
│      isometry into P̃(ρ_tok) ∈ R^256                                                    │
│    • First-moment coherent value ket |v_coh⟩ ∈ C^16 packed into R^32                   │
│    • Concatenated per-head value feature Π_s,h = [P̃(ρ_tok), v_coh_re, v_coh_im] ∈ R^288│
│                                                                                        │
│ 3. Single Tensor-Core Causal Attention GEMM (width = 288 = 9 * 32):                    │
│    • Causal Born / Softmax-Aligned Weights W_ts,h = Softmax_s(Scale * Re⟨q_t | k_s⟩)   │
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
│ 5. Option B (320-Dim/Head) Direct Stiefel Observable Readout:                          │
│    • Per-head observable O_t,h = [P̃(ρ_out) (256), √λ_1 φ_1 (32), √λ_2 φ_2 (32)] ∈ R^320│
│    • Flatten 6 heads -> O_t ∈ R^1920                                                   │
│    • Direct Stiefel projection: Δx_attn = α_obs ⊙ (O_t @ W_obs), W_obs ∈ St(1920, 768) │
│                                                                                        │
│ 6. Mach-Zehnder Two-Beam Interferometer Channel Mixing:                                │
│    • Two-beam Stiefel up-projections w_beam_a, w_beam_b ∈ St(3072, 768)                │
│    • Interferometric gating φ_a(u_a) ⊙ u_b                                             │
│    • Direct Stiefel down-projection: Δx_chan = α_chan ⊙ (h_int @ W_chan),              │
│      W_chan ∈ St(3072, 768)                                                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
[Final Vocabulary Readout (Chunked Rule-9 Single-Pass Autograd, chunk_tokens = 8192)]
   ├─► Mode 1: VocabularyReadoutMode.RANK2_KRYLOV_BORN
   │     Splits final state / Krylov pair into (λ_1, |Ψ_1⟩), (λ_2, |Ψ_2⟩) and evaluates
   │     calibrated log-Born probability p_t(v) ∝ λ_1 |⟨e_v|Ψ_1⟩|^2 + λ_2 |⟨e_v|Ψ_2⟩|^2
   └─► Mode 2: VocabularyReadoutMode.SOFTCAP_LINEAR_CE
         Evaluates 15 * tanh((h_t @ W_lm^T) / 15) Cross-Entropy via in-place Triton kernel
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

#### 2.2 `@torch.compile` Fused McWeeny Purification + Unitary Evolution + Uhlmann Phase-Locked Rank-2 Krylov Extraction (`_compiled_mcweeny_uhlmann_krylov_step`)
After causal attention mixing $M = W_{\text{attn}} \Pi \in \mathbb{R}^{B \times H \times T \times 288}$, slice $M$ into $\tilde{P}(\rho^{(0)}) \in \mathbb{R}^{B \times H \times T \times 256}$ and $\bar{v}_{\text{raw}} = (\bar{v}_{\text{re}}, \bar{v}_{\text{im}}) \in \mathbb{R}^{B \times H \times T \times 32}$:
1.  **Unpack & 1-Step McWeeny Purification (`bfloat16` GEMM, `float32` trace)**:
    Unpack $\tilde{P}(\rho^{(0)}) \to (\rho_{\text{re}}^{(0)}, \rho_{\text{im}}^{(0)}) \in \mathbb{R}^{B \times H \times T \times 16 \times 16}$. Compute the complex square in `bfloat16`:
    $$S_{\text{re}} = (\rho_{\text{re}}^{(0)})^2 - (\rho_{\text{im}}^{(0)})^2, \qquad S_{\text{im}} = \rho_{\text{re}}^{(0)} \rho_{\text{im}}^{(0)} + \rho_{\text{im}}^{(0)} \rho_{\text{re}}^{(0)}$$
    Accumulate the trace in `float32`: $\tau = \sum_{i=1}^{16} S_{\text{re},ii} + \varepsilon$, and normalize $\rho^{(1)} = S / \tau$.
2.  **Per-Head Complex Unitary Evolution $U_h \in U(16)$**:
    Conjugate $\rho_{\text{out}} = U_h \rho^{(1)} U_h^\dagger$ and evolve the first-moment anchor ket $|\bar{v}_{\text{out}}\rangle = \operatorname{normalize}_{\mathbb{C}}(U_h |\bar{v}_{\text{raw}}\rangle)$.
3.  **Uhlmann Phase-Locked Rank-2 Lanczos-Krylov Eigenpair (`extract_rank2_krylov_eigenpair_uhlmann`)**:
    *   **Step 1 (Primary Krylov Vector)**: Apply $\rho_{\text{out}}$ to the anchor ket $|w_1\rangle = \rho_{\text{out}} |\bar{v}_{\text{out}}\rangle$ and normalize $|\tilde{\phi}_1\rangle = |w_1\rangle / (\|w_1\|_2 + \varepsilon)$.
    *   **Step 2 (Uhlmann Parallel-Transport Phase Lock)**: Compute the complex inner product $c = \langle \tilde{\phi}_1 \mid \bar{v}_{\text{out}}\rangle_{\mathbb{C}} = c_{\text{re}} + i c_{\text{im}}$ and rotate $|\tilde{\phi}_1\rangle$ by $c / (|c| + \varepsilon_{\text{uhl}})$ so $\langle \bar{v}_{\text{out}} \mid \phi_1\rangle_{\mathbb{C}} \in \mathbb{R}_{\ge 0}$:
        $$|\phi_1\rangle = \frac{\langle \tilde{\phi}_1 \mid \bar{v}_{\text{out}}\rangle_{\mathbb{C}}}{|\langle \tilde{\phi}_1 \mid \bar{v}_{\text{out}}\rangle_{\mathbb{C}}| + \varepsilon_{\text{uhl}}} |\tilde{\phi}_1\rangle$$
        This eliminates quadratic sign/phase ambiguity: if all input value kets rotate by $e^{i\theta}$, $|\bar{v}_{\text{out}}\rangle$ rotates by $e^{i\theta}$ and $|\phi_1\rangle$ rotates covariantly by $e^{i\theta}$.
    *   **Step 3 (Second Orthogonal Krylov Eigenmode)**: Compute Rayleigh quotient $\lambda_1 = \langle \phi_1 \mid \rho_{\text{out}} \mid \phi_1\rangle \in [0, 1]$, form the Lanczos residual $|u_2\rangle = \rho_{\text{out}} |\phi_1\rangle - \lambda_1 |\phi_1\rangle$, project out $|\phi_1\rangle$ via complex Gram-Schmidt $|u_2^\perp\rangle = |u_2\rangle - \langle \phi_1 \mid u_2\rangle_{\mathbb{C}} |\phi_1\rangle$, normalize $|\phi_2\rangle = |u_2^\perp\rangle / (\|u_2^\perp\|_2 + \varepsilon)$, and evaluate $\lambda_2 = \langle \phi_2 \mid \rho_{\text{out}} \mid \phi_2\rangle \in [0, 1]$.
4.  **Option B (`320`-Dim/Head) Observable Feature Packing**:
    Concatenate the isometric packed density $\tilde{P}(\rho_{\text{out}}) \in \mathbb{R}^{256}$ with both eigenvalue-weighted Krylov eigenmodes $[\sqrt{\lambda_1}\phi_{1,\text{re}}, \sqrt{\lambda_1}\phi_{1,\text{im}}, \sqrt{\lambda_2}\phi_{2,\text{re}}, \sqrt{\lambda_2}\phi_{2,\text{im}}] \in \mathbb{R}^{64}$ to produce $O_{b,h,t} \in \mathbb{R}^{320}$ per head ($6 \times 320 = 1,920$ across heads).

#### 2.3 Direct Stiefel Readout & Tikhonov-Damped `RiemannianStiefelMuon`
1.  **Direct Semi-Orthogonal Readout Parameterization**:
    Each block stores `w_obs` $\in \text{St}(1920, 768)$ ($W_{\text{obs}}^\top W_{\text{obs}} = I_{768}$) and a learnable vector gain `obs_scale` $\in \mathbb{R}^{768}$ initialized to `DEFAULT_DIRECT_READOUT_INIT_SIGMA = 0.06`, plus `w_chan` $\in \text{St}(3072, 768)$ and `chan_scale` $\in \mathbb{R}^{768}$.
2.  **Tikhonov-Damped Tangent Newton-Schulz (`tikhonov_lambda = 1e-3`)**:
    For any tall real Stiefel parameter $W \in \text{St}(m, n)$ ($m \ge n$) with Euclidean gradient $G$:
    *   Project $G$ onto the Riemannian tangent space: $G_{\text{tan}} = G - W \operatorname{sym}(W^\top G)$.
    *   Update Nesterov tangent momentum $M_{\text{tan}}$ and project back to $T_W \text{St}(m, n)$.
    *   Normalize $X_0 = M_{\text{tan}} / (\|M_{\text{tan}}\|_F + \varepsilon)$. In the first Newton-Schulz iteration, add Tikhonov Gram damping $\lambda_{\text{tik}} I_n$ to suppress tail-singular-value noise amplification:
        $$A_0 = X_0^\top X_0 + \lambda_{\text{tik}} I_n, \qquad X_1 = \frac{1}{2} X_0 (3 I_n - A_0)$$
        followed by 4 standard cubic Newton-Schulz steps and final tangent re-projection $U_{\text{tan}} = \Pi_{T_W}(X_5) \frac{\sqrt{n}}{\|\Pi_{T_W}(X_5)\|_F + \varepsilon}$.
    *   Retract along the unit-norm Stiefel geodesic:
        $$W_{\text{next}} = \frac{W - \eta U_{\text{tan}}}{\sqrt{1 + \eta^2}}$$
    The same Tikhonov Gram damping is applied to 3D complex head unitaries $U_h \in U(16)$ in `riemannian_complex_unitary_muon_step`.

#### 2.4 Active-Token Vocabulary Manifold Optimizers (`src/qdlblocks/optim/vocab_manifold_muon.py`)
Given active batch token indices $\mathcal{I}_{\text{act}} = \operatorname{unique}(\text{idx} \cup \text{targets})$ ($N_{\text{act}} = |\mathcal{I}_{\text{act}}| \le B \times T$):
1.  **`BatchSubspaceTikhonovGramMuon` (`VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON`)**:
    *   Slices active gradients $G_{\text{act}} = G[\mathcal{I}_{\text{act}}] \in \mathbb{R}^{N_{\text{act}} \times d}$ and updates momentum $M_{\text{act}} = M[\mathcal{I}_{\text{act}}]$ strictly on $\mathcal{I}_{\text{act}}$ (preventing Trap 1 & Trap 3 ghost updates on the `~45,000` inactive tokens).
    *   Projects each active momentum row onto the tangent space of the unit-RMS sphere $\sqrt{d}\,\mathbb{S}^{d-1}$ at $W[\mathcal{I}_{\text{act}}]$.
    *   Forms the small $d \times d$ ($768 \times 768$) Tikhonov-damped feature Gram matrix:
        $$A = \tilde{M}_{\text{act}}^\top \tilde{M}_{\text{act}} + \lambda_{\text{tik}} I_d \in \mathbb{R}^{d \times d}, \qquad \tilde{M}_{\text{act}} = \frac{M_{\text{act}}}{\|M_{\text{act}}\|_F + \varepsilon}$$
        and computes the preconditioned active update $U_{\text{act}} = \tilde{M}_{\text{act}} A^{-1/2}$ via 5-step coupled Newton-Schulz on the $768 \times 768$ matrix $A$.
    *   Applies the adaptive median SNR trust-region gate per active token row and retracts updated rows onto $\sqrt{d}\,\mathbb{S}^{d-1}$.
2.  **`RiemannianSphereTangentSNROptimizer` (`VocabOptimizerMode.SPHERE_TANGENT_SNR_MUON`)**:
    *   On active rows $v \in \mathcal{I}_{\text{act}}$, projects $g_v$ onto $T_{w_v}\mathbb{S}^{d-1}$, updates tangent momentum $m_v$ and second moment $v_v$, parallel-transports $m_v$ to $T_{w_v}\mathbb{S}^{d-1}$, computes $\text{SNR}_v = \|m_v\|_2 / (\sqrt{v_v} + \varepsilon)$, gates the angular step $\theta_v = \eta_{\text{sphere}} \frac{\text{SNR}_v}{\text{SNR}_v + \tau_{\text{snr}}}$, and takes the exact great-circle step $w_v^{\text{next}} = \cos(\theta_v) w_v - \sin(\theta_v) \sqrt{d} \frac{m_v}{\|m_v\|_2 + \varepsilon}$.
3.  **`AdamW` Baseline (`VocabOptimizerMode.ADAMW_BASELINE`)**:
    *   Standard decoupled `AdamW` on `wte` and `lm_head` for controlled ablation.

#### 2.5 Rule-9 Single-Pass Chunked Vocabulary Readouts
1.  **`VocabularyReadoutMode.RANK2_KRYLOV_BORN` (`Rank2KrylovBornChunkedCEFn`)**:
    *   Maps the final normalized hidden state $h_t \in \mathbb{R}^{768}$ into a Rank-2 complex measurement state pair $(|\Psi_{1,t}\rangle, |\Psi_{2,t}\rangle) \in \mathbb{C}^{384} \times \mathbb{C}^{384}$ (or two real measurement branches in $\mathbb{R}^{768}$) weighted by normalized Krylov eigenvalues $(\hat{\lambda}_{1,t}, \hat{\lambda}_{2,t})$ ($\hat{\lambda}_{1,t} + \hat{\lambda}_{2,t} = 1$).
    *   In chunks of `chunk_tokens = 8192`, computes the Born overlap intensities against the vocabulary wavefunctions $E \in \mathbb{R}^{V \times 768}$, applies temperature-scaled log-intensity / softcap calibration, accumulates Cross-Entropy loss and in-place chunk gradients in a **single pass** per chunk (zero duplicate matmuls in `backward()`, satisfying Rule 9).
2.  **`VocabularyReadoutMode.SOFTCAP_LINEAR_CE`**:
    *   Reuses [`compiled_chunked_softcap_linear_ce`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L436-L453) ($15 \tanh(z / 15)$) for direct comparison with the classical leaderboard baseline.

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
*   **Reason for Rejection**: Computationally wasteful and statistically noisy under Zipfian token frequencies. Replaced by active-token `BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer` restricted to $\mathcal{I}_{\text{act}} = \operatorname{unique}(\text{idx} \cup \text{targets})$.

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
*   **Code Quality & Repository Invariants (`AGENTS.md`)**:
    *   All modified and newly created Python files must pass `scripts/ast_signature_and_length_linter.py` (`<= 1,000` SLOC soft limit per Rule 1, `<= 8` parameters per function/dataclass per Rule 2, zero semicolons per Rule 3, top-of-file `UPPER_SNAKE_CASE` constants per Rule 7).
    *   Frozen Wave 47 modules ([`src/qdlblocks/models/riemannian_impure_density_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/riemannian_impure_density_gpt.py)) must remain untouched (Rule 6).

--------------------------------------------------------------------------------

### Additional Resources

*   Attached Research Reference: *"Top-1 Accuracy vs Cross-Entropy Divergence"* (`Top-1_Accuracy_vs_Cross-Entropy_Divergence.pdf`, Pages 1–37)
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
*   **Rationale**: Synchronizes the formal Wave 48 experiment hypotheses (`H1`, `H2`, `H3`, `H4`), top-of-file named constants (Rule 7), and verification commands with [`scripts/verify_plan.py`](file:///home/nixos/QuantumDLBlocks/scripts/verify_plan.py) and `StandardizedExperimentLogger`.
*   **Proposed Changes**:
    *   Define the 4 falsifiable hypotheses (`H1`: Fused `288`-width + Uhlmann Rank-2 Krylov `320`-dim/head Option B vs. phase-erased `256`-width attention; `H2`: Direct Stiefel readout + Tikhonov-damped `RiemannianStiefelMuon` vs. undamped Stiefel/Euclidean updates; `H3`: Active-token `BatchSubspaceTikhonovGramMuon` & `RiemannianSphereTangentSNROptimizer` vs. `AdamW` across `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE`; `H4`: Standalone FineWeb Shard 0 benchmark trajectory up to `<= 330M` tokens).
    *   Document the synchronized top-of-file `UPPER_SNAKE_CASE` constant table (`DEFAULT_FUSED_HEAD_PACK_DIM = 288`, `DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM = 320`, `DEFAULT_TIKHONOV_GRAM_LAMBDA = 1e-3`, `DEFAULT_UHLMANN_PHASE_EPS = 1e-7`, `DEFAULT_SPHERE_SNR_TRUST_TAU = 0.50`, `DEFAULT_SPHERE_GEODESIC_LR = 0.08`, `DEFAULT_BATCH_GRAM_MUON_LR = 0.06`, `DEFAULT_DIRECT_READOUT_INIT_SIGMA = 0.06`).

#### 2. [`/home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py) (Modify)

*   **Type / Scope**: Core Quantum Density & Krylov Layer Primitives (`<= 1,000` SLOC, `<= 8` args per function)
*   **Rationale**: Promotes reusable fused packing, Uhlmann phase-locked Rank-2 Krylov extraction, and Tikhonov-damped Stiefel Newton-Schulz into the shared layer library without breaking backward compatibility for Waves 45–47.
*   **Proposed Changes**:
    *   Add top-of-file constants `DEFAULT_FUSED_HEAD_PACK_DIM = 288`, `DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM = 320`, `DEFAULT_TIKHONOV_GRAM_LAMBDA = 1e-3`, and `DEFAULT_UHLMANN_PHASE_EPS = 1e-7`.
    *   Add `fast_rank_r_hermitian_and_ket_pack(v_re, v_im, re_pack_mask, im_pack_mask, eps=1e-6) -> Tensor`: Contracts Rank-$R$ complex value branches into an isometric $\sqrt{2}$-scaled Hermitian vector (`256` dims) and concatenates the unit-norm first-moment coherent value ket (`32` dims) into a single `288`-wide tensor `(B, H, T, 288)`.
    *   Add `extract_rank2_krylov_eigenpair_uhlmann(rho_re, rho_im, v_bar_re, v_bar_im, eps=1e-7) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]`: Seeds 2-step Lanczos-Krylov extraction from the contextual first-moment anchor ket $|\bar{v}\rangle$, applies the Uhlmann parallel-transport phase lock $|\phi_1\rangle = (\langle \tilde{\phi}_1 \mid \bar{v}\rangle / |\langle \tilde{\phi}_1 \mid \bar{v}\rangle|) |\tilde{\phi}_1\rangle$, and extracts the orthogonal second eigenmode $|\phi_2\rangle$ and Rayleigh quotients $(\lambda_1, \lambda_2)$.
    *   Update `_newton_schulz5_tall_real(g, steps=5, eps=1e-7, tikhonov_lambda=0.0)` and `riemannian_stiefel_muon_step` (keeping total parameters `<= 8` per Rule 2) to support Tikhonov Gram damping $A_0 = X_0^\top X_0 + \lambda_{\text{tik}} I_n$ in the first Newton-Schulz iteration.

#### 3. [`/home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py) (Create)

*   **Type / Scope**: Active-Token Vocabulary Manifold Optimizers (`~220` SLOC)
*   **Rationale**: Isolates the two active-token vocabulary manifold optimizers (`BatchSubspaceTikhonovGramMuon` and `RiemannianSphereTangentSNROptimizer`) so they update strictly active token rows $\mathcal{I}_{\text{act}} = \operatorname{unique}(\text{idx} \cup \text{targets})$ without polluting inactive vocabulary rows or exceeding SLOC limits.
*   **Proposed Changes**:
    *   Implement `_newton_schulz_coupled_inv_sqrt(gram_a, steps=5, eps=1e-6) -> Tensor`: Computes $A^{-1/2}$ on the $d \times d$ ($768 \times 768$) Tikhonov-damped active-token feature Gram matrix $A = \tilde{M}_{\text{act}}^\top \tilde{M}_{\text{act}} + \lambda_{\text{tik}} I_d$.
    *   Implement `BatchSubspaceTikhonovGramMuon(torch.optim.Optimizer)`:
        *   `step_active(active_idx: Tensor)`: Slices $G[\mathcal{I}_{\text{act}}]$, updates momentum strictly on $\mathcal{I}_{\text{act}}$, projects onto the tangent space of $\sqrt{d}\,\mathbb{S}^{d-1}$, preconditions via $U_{\text{act}} = \tilde{M}_{\text{act}} A^{-1/2}$, applies the adaptive median SNR trust-region gate, and retracts updated rows onto $\sqrt{d}\,\mathbb{S}^{d-1}$.
    *   Implement `RiemannianSphereTangentSNROptimizer(torch.optim.Optimizer)`:
        *   `step_active(active_idx: Tensor)`: Projects active row gradients onto $T_{w_v}\mathbb{S}^{d-1}$, updates tangent momentum $m_v$ and second moment $v_v$, parallel-transports $m_v$ back to $T_{w_v}\mathbb{S}^{d-1}$, gates the angular geodesic step size $\theta_v = \eta_{\text{sphere}} \frac{\text{SNR}_v}{\text{SNR}_v + \tau_{\text{snr}}}$, and applies the great-circle retraction $w_v^{\text{next}} = \cos(\theta_v) w_v - \sin(\theta_v) \sqrt{d} \frac{m_v}{\|m_v\|_2 + \varepsilon}$.

#### 4. [`/home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py) (Create)

*   **Type / Scope**: Flag-Free Standalone Quantum Born GPT Model & Readouts (`~580` SLOC)
*   **Rationale**: Enforces Rule 6 (no boolean experiment flags added to frozen `riemannian_impure_density_gpt.py`) by providing a clean, standalone, teacher-free model with Option B (`320`-dim/head) direct Stiefel readouts, `@torch.compile` fused McWeeny + Uhlmann Krylov step, and Rule-9 single-pass chunked readouts.
*   **Proposed Changes**:
    *   Define `VocabularyReadoutMode(str, Enum)` (`RANK2_KRYLOV_BORN`, `SOFTCAP_LINEAR_CE`) and `VocabOptimizerMode(str, Enum)` (`BATCH_SUBSPACE_GRAM_MUON`, `SPHERE_TANGENT_SNR_MUON`, `ADAMW_BASELINE`).
    *   Define `StandaloneQuantumBornConfig` (`<= 8` fields per Rule 2: `vocab_size=50304`, `n_layer=12`, `n_head=6`, `n_embd=768`, `head_dim_complex=16`, `v_rank=4`, `readout_mode=VocabularyReadoutMode.RANK2_KRYLOV_BORN`, `vocab_optim_mode=VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON`).
    *   Implement `_compiled_mcweeny_uhlmann_krylov_step(packed_mixed, re_unpack_idx, re_unpack_scale, im_unpack_idx, im_unpack_scale, u_re, u_im, masks) -> Tensor`: Decorated with `@torch.compile` (with graceful eager fallback when compilation is disabled in unit tests), executes $16 \times 16$ unpacking, `bfloat16` McWeeny squaring with `float32` trace normalization, $U(16)$ unitary conjugation, Uhlmann phase-locked Rank-2 Krylov extraction, and Option B `320`-dim/head observable concatenation.
    *   Implement `StandaloneQuantumBornBlock(nn.Module)`: Sequential quantum block with `w_q`, `w_k`, `w_v`, direct Stiefel readout `w_obs` $\in \text{St}(1920, 768)$ + `obs_scale`, complex unitary `(head_unitary_re, head_unitary_im)` $\in U(16)$, and Mach-Zehnder two-beam interferometer `w_beam_a`, `w_beam_b`, `w_chan` $\in \text{St}(3072, 768)$ + `chan_scale`.
    *   Implement `Rank2KrylovBornChunkedCEFn(torch.autograd.Function)`: Rule-9 single-pass chunked (`chunk_tokens = 8192`) Rank-2 Krylov Born vocabulary loss and backward pass.
    *   Implement `StandaloneQuantumBornGPT(nn.Module)` and `StandaloneRiemannianOptimizerBundle`: Coordinates Tikhonov-damped `AllLayerRiemannianStiefelMuon` on 100% of 2D/3D backbone weights alongside the selected `VocabOptimizerMode` on active vocabulary indices $\mathcal{I}_{\text{act}}$.

#### 5. [`/home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py) (Create)

*   **Type / Scope**: Wave 48 Verification & Dual-RTX 3090 FineWeb Speedrun Runner (`~650` SLOC)
*   **Rationale**: Provides the automated verification harness for `H1`, `H2`, `H3`, and `H4` and the multi-GPU (`2x RTX 3090`) standalone FineWeb training loop evaluated on the `10,485,760`-token `fineweb_val_000000.bin` Shard 0 validation set.
*   **Proposed Changes**:
    *   Implement `verify_h1_uhlmann_krylov_phase_and_ablation()`: Verifies phase covariance of `288`-width + Uhlmann Rank-2 Krylov (`320`-dim/head Option B) and compares validation CE against phase-erased `256`-width quadratic density attention on FineWeb.
    *   Implement `verify_h2_direct_stiefel_tikhonov_muon()`: Compares direct Stiefel readouts updated with Tikhonov-damped `RiemannianStiefelMuon` ($\lambda_{\text{tik}} = 10^{-3}$) against undamped updates and verifies Stiefel/Unitary manifold error `< 1e-5`.
    *   Implement `verify_h3_active_vocab_manifold_optimizers()`: Benchmarks `BatchSubspaceTikhonovGramMuon`, `RiemannianSphereTangentSNROptimizer`, and `AdamW` across `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE` readouts on FineWeb Shard 0.
    *   Implement `run_standalone_fineweb_speedrun()`: Executes multi-GPU (`2x RTX 3090`) standalone training across FineWeb shards up to the target token horizon and evaluates full `10,485,760`-token Shard 0 validation CE.

#### 6. [`/home/nixos/QuantumDLBlocks/tests/test_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan063_standalone_quantum_born_gpt.py) (Create)

*   **Type / Scope**: Deterministic Pytest Suite (`~150` SLOC)
*   **Rationale**: Validates mathematical invariants, phase covariance, active-token isolation, manifold orthogonality, and Rule-9 autograd efficiency in `< 15 seconds`.
*   **Proposed Changes**:
    *   `test_fused_288_hermitian_and_ket_pack_isometry_and_phase`: Asserts `< 1e-6` agreement between single-GEMM `288`-width mixing and split density+ket mixing, and verifies that global phase rotation $|v\rangle \mapsto e^{i\theta}|v\rangle$ rotates $|\phi_1\rangle$ covariantly (eliminating quadratic phase erasure).
    *   `test_tikhonov_riemannian_stiefel_and_unitary_muon_stability`: Asserts `< 1e-5` Stiefel and Unitary manifold error over multi-step gradient updates on direct `w_obs` $\in \text{St}(1920, 768)$ and $U(16)$ parameters without SVD ridge refits.
    *   `test_batch_subspace_gram_muon_and_sphere_tangent_snr_active_isolation`: Asserts zero parameter or momentum drift on inactive token rows ($v \notin \mathcal{I}_{\text{act}}$) and unit-RMS sphere preservation $\|w_v\|_2 = \sqrt{d}$ on active rows.
    *   `test_standalone_quantum_born_gpt_end_to_end_and_rule9_autograd`: Verifies teacher-free loss decrease from Step 0 for both `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE` readouts and passes `autograd_efficiency_auditor`.

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
