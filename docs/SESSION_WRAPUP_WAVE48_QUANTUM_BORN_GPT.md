# Session Wrap-Up: End-to-End Quantum Born-Rule Attention & Riemannian Manifold Optimization on FineWeb (`QuantumDLBlocks` — Wave 48)

## 1. Executive Summary
In this session, we designed, implemented, debugged, and benchmarked **[`StandaloneQuantumBornGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L708)** ([`EXPERIMENT_PLAN_063.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_063.md) / Wave 48 in the `QuantumDLBlocks` repository)—a 12-layer, `768`-dimensional (`~150M`-parameter) autoregressive language model whose attention mechanism is grounded in **quantum density-matrix transport, algebraic $p$-boson Born-rule coincidence probabilities, Uhlmann parallel-transport phase, and Rank-2 Lanczos-Krylov eigenmode extraction**, and whose weight matrices are optimized directly on **Riemannian manifolds** (Stiefel manifolds $\mathrm{St}(m, n)$, unitary groups $U(16)$, and hyperspheres $S^{d-1}(\sqrt{d})$).

Starting from a codebase where prior quantum language model prototypes (`Wave 47`) relied on a hidden pre-trained classical transformer teacher and suffered from rank-1 quantum state collapse, we:
1. Built a **100% standalone, teacher-free architecture** ([`StandaloneQuantumBornGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L708)) and a verified **900M-token multi-shard FineWeb streaming pipeline** ([`MultiShardTokenStream`](file:///home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py#L105)).
2. Diagnosed why early standalone runs converged slower than classical transformers by running **surgical Layer 11 swap ablations** on a pre-trained `400M`-token classical Modded-NanoGPT checkpoint ([`CompiledLeaderboardClassicalGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L670)).
3. Uncovered and fixed **five architectural/mathematical bottlenecks** in the quantum attention block—(1) Value-RoPE axis scrambling, (2) radial purity-collapse incentives in unnormalized density/Krylov readouts, (3) a $4\times$ coherent wavepacket rank bottleneck from summing across branches, (4) a Hilbert-space frame noise floor on $\mathbb{C}^{16}$ vs. $\mathbb{C}^{64}$ with $p=3$ Born coincidence $|\langle q_t \mid k_s \rangle|^6$, and (5) scrambled value-embedding injection & missing partial key offset—plus a runner-script bug that silently set `grad_accum = 4` on runs with $\ge 1,000$ steps.
4. Showed on the `400M`-token Layer 11 swap benchmark (`1.64M` to `16.38M` unseen FineWeb tokens) that **pure algebraic polynomial Born attention ($|\langle q_t \mid k_s \rangle|^6$ on $\mathbb{C}^{64}$, with zero `exp()` or `softmax()`) outperforms both Cold-Start Classical Scaled Dot-Product Attention (`SDPA`) Softmax (`3.7517` vs. `3.7660` Val CE at `2,000` steps) and Warm-Gate Classical `SDPA` Softmax (`3.7609` vs. `3.7656` Val CE at `200` steps)** while maintaining a rich mixed quantum state (`mean_purity = 0.153–0.335`, `mean_eff_rank = 3.06–8.53`).
5. Uncovered and fixed **two standalone readout/vocabulary boundary bugs** (`logit_scale` zero-crossing sign flip at Step 8 and dense softmax gradient overwriting of inactive vocabulary rows in `head_opt`), while optimizing complex inner products into 2 `bfloat16` GEMMs with `@torch.compile` to cut 12-layer VRAM from `26.13 GiB` (OOM) to **`15.67 GB`** and boost single-GPU throughput by **`+58%` (`8,271` $\to$ `13,060 tok/s`)**.
6. Demonstrated a **`2.5x` token-efficiency gain** in standalone end-to-end FineWeb training (`5.7103` Val CE and `18.87%` Top-1 accuracy at `8.19M` tokens in `v3`, matching the original Wave 48 run's `20.48M`-token Val CE of `5.6991`), and reached **`5.2494` Val CE (`22.02%` Top-1 accuracy)** at `20.48M` tokens in `v2` (`-0.45` nats ahead of original Wave 48).

---

## 2. Problems We Have Been Solving and Discussing

### 2.1. Moving from Teacher-Assisted Shortcuts to True Standalone End-to-End Training
Prior experiments in the repository (`Wave 47` / `QuantumKrylovBornGPT`) were not training end-to-end from token cross-entropy gradients alone. Due to an earlier coding-agent miscommunication, those scripts loaded a pre-trained classical transformer teacher ([`CompiledLeaderboardClassicalGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L670)) and solved a closed-form ridge regression problem to align quantum layer outputs with teacher activations. Our first goal was to eliminate all teacher dependencies and train a standalone quantum-inspired language model end-to-end from random initialization.

### 2.2. Preventing Quadratic Phase Erasure & Formulating End-to-End Quantum Born Attention
In standard classical attention, causal weights are computed via exponential softmax over real inner products. In a quantum formulation based on the **multi-boson Born rule**, transition weights between complex query and key states $|q_t\rangle, |k_s\rangle$ are pure algebraic powers of their Hermitian inner product magnitude:
$$S_{t,s} = \mathbf{1}_{s \le t}\,\left|\langle q_t \mid k_s \rangle\right|^{2p}, \qquad w_{t,s} = \frac{S_{t,s}}{\sum_{u \le t} S_{t,u} + \epsilon}$$
with **zero `exp()` or `softmax()` calls**. However, as analyzed in our theoretical reference (`media_1793332634476_f0f8bf6a.pdf`), a naive Born-rule layer that only computes $|\langle q_t \mid k_s \rangle|^{2p}$ and takes a weighted average of real value vectors erases relative complex phases (**quadratic phase erasure**).

To preserve quantum phase interference, **[`StandaloneQuantumBornGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L708)** (`12` layers, `d_model = 768`, $H = 6$ heads, head complex dimension $d_h = 16$, wavepacket branch rank $R = 4$) executes the following complete forward pass on input token indices $\text{idx} \in \{0, \dots, 50303\}^{B \times T}$:

1. **Token Embedding, RMS Normalization, and Stiefel Input Projection**:
   - Token embeddings $E = \mathrm{wte}(\text{idx}) \in \mathbb{R}^{B \times T \times 768}$ are cast to `bfloat16`, RMS-normalized per token, and projected through an input Stiefel matrix $W_{\text{in}} \in \mathrm{St}(768, 768)$ scaled by learnable scalar $\alpha_{\text{in}} \in \mathbb{R}$ (`in_scale`, initialized to `1.0`) to form both the initial hidden state $X^{(0)}$ and the residual-highway anchor $X_{\text{anchor}} = X^{(0)}$:
     $$E_{\text{norm}} = \mathrm{RMSNorm}(E), \qquad X^{(0)} = \alpha_{\text{in}}\,E_{\text{norm}}\,W_{\text{in}} \in \mathbb{R}^{B \times T \times 768}$$

2. **Per-Block Residual Highway ($\lambda_{\text{block}}$), Value Highway ($\lambda_v$), and Complex Wavepacket Projections**:
   - Entering Block $\ell \in \{0, \dots, 11\}$ ([`StandaloneQuantumBornBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L285)), the current hidden state $X^{(\ell)}$ is blended with the initial anchor $X^{(0)}$ via learnable 2-vector $\lambda_{\text{block}}^{(\ell)} = [\lambda_{\text{block},0}^{(\ell)}, \lambda_{\text{block},1}^{(\ell)}] \in \mathbb{R}^2$ (initialized to $[1.0, 0.0]$):
     $$X_{\text{cur}}^{(\ell)} = \lambda_{\text{block},0}^{(\ell)}\,X^{(\ell)} + \lambda_{\text{block},1}^{(\ell)}\,X^{(0)}, \qquad H_{\text{norm}}^{(\ell)} = \mathrm{RMSNorm}\!\big(X_{\text{cur}}^{(\ell)}\big)$$
   - Stiefel matrices $W_q^{(\ell)}, W_k^{(\ell)}, W_v^{(\ell)} \in \mathrm{St}(768, 768)$ project $H_{\text{norm}}^{(\ell)}$ into $H = 6$ heads. Each head has complex dimension $d_h = 16$ and $R = 4$ branches, giving $R \times d_h = 64$ complex dimensions ($128$ real dimensions) per head ($6 \times 128 = 768$).
   - **Value Highway**: In Layer $\ell = 0$, the value projection $V_{\text{proj}}^{(0)} = H_{\text{norm}}^{(0)} W_v^{(0)}$ defines the global value-highway tensor $V^{(0)} = V_{\text{proj}}^{(0)}$. In Layers $\ell \ge 1$, $V_{\text{proj}}^{(\ell)}$ is blended with $V^{(0)}$ via learnable 2-vector $\lambda_v^{(\ell)} = [\lambda_{v,0}^{(\ell)}, \lambda_{v,1}^{(\ell)}] \in \mathbb{R}^2$ (initialized to $[0.85, 0.15]$):
     $$V_{\text{mixed}}^{(\ell)} = \lambda_{v,0}^{(\ell)}\,V_{\text{proj}}^{(\ell)} + \lambda_{v,1}^{(\ell)}\,V^{(0)}$$
     For each batch $b$, head $h$, and token $s$, $V_{\text{mixed}}^{(\ell)}$ is reshaped into $R=4$ complex row vectors $\psi_{b,h,s,r}^\top \in \mathbb{C}^{1 \times 16}$ ($r \in \{1,2,3,4\}$) stacked as a matrix $\Psi_{b,h,s} \in \mathbb{C}^{4 \times 16}$, with **no** RoPE applied to values.
   - **Queries, Keys, and Partial Key Offset**: Queries $q_{b,h,t} \in \mathbb{C}^{64}$ and keys $k_{b,h,s} \in \mathbb{C}^{64}$ are L2-normalized per head ($\|q_{b,h,t}\|_2 = \|k_{b,h,s}\|_2 = 1$) and rotated by complex Rotary Position Embeddings (RoPE) $e^{i t \omega_j}$. On long-window layers (`layer_idx in {3, 7, 11}`), [`apply_partial_key_offset`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/isometric.py#L430) shifts the second quarter of key channels (`channels [16:32]` of the 64 complex channels) forward by one token position ($k_{b,h,s,16:32} \leftarrow k_{b,h,s-1,16:32}$, with $s=0$ zero-padded), allowing a single attention layer to match `[prev_token, current_token]` bigrams (induction heads).

3. **Two-Matmul Complex Inner Products & Causal $p=3$ Born Coincidence Weights**:
   - Writing $q_t = q_{\text{re},t} + i q_{\text{im},t}$ and $k_s = k_{\text{re},s} + i k_{\text{im},s} \in \mathbb{C}^{64}$ as real row vectors, we form concatenated vectors $q_{\text{cat},t} = [q_{\text{re},t}, q_{\text{im},t}] \in \mathbb{R}^{128}$, $k_{\text{cat},s} = [k_{\text{re},s}, k_{\text{im},s}] \in \mathbb{R}^{128}$, and $k_{\text{skew},s} = [-k_{\text{im},s}, k_{\text{re},s}] \in \mathbb{R}^{128}$. This computes both real and imaginary parts of the complex inner product in **two `bfloat16` GEMMs** instead of four:
     $$\text{ip}_{\text{re},t,s} = q_{\text{cat},t} k_{\text{cat},s}^\top = q_{\text{re},t}\cdot k_{\text{re},s} + q_{\text{im},t}\cdot k_{\text{im},s} = \mathrm{Re}\langle q_t \mid k_s \rangle$$
     $$\text{ip}_{\text{im},t,s} = q_{\text{cat},t} k_{\text{skew},s}^\top = q_{\text{im},t}\cdot k_{\text{re},s} - q_{\text{re},t}\cdot k_{\text{im},s} = -\mathrm{Im}\langle q_t \mid k_s \rangle$$
   - Because the Born modulus squared $|\langle q_t \mid k_s \rangle|^2 = \text{ip}_{\text{re},t,s}^2 + \text{ip}_{\text{im},t,s}^2$ depends only on the square of $\text{ip}_{\text{im},t,s}$, the sign of $\text{ip}_{\text{im},t,s}$ is invariant. Using $p=3$ (3-boson coincidence, [`_causal_born_p3_weights_impl`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L155)), causal weights are:
     $$S_{t,s} = \mathbf{1}_{s \le t}\left(\text{ip}_{\text{re},t,s}^2 + \text{ip}_{\text{im},t,s}^2\right)^3, \qquad w_{t,s} = \frac{S_{t,s}}{\sum_{u \le t} S_{t,u} + \epsilon}$$

4. **Fused Token Density Matrix, First-Moment Anchor Ket, and 4-Branch Wavepacket Transport**:
   - Each token $s$ normalizes its $4 \times 16$ value wavepacket matrix $\Psi_s \in \mathbb{C}^{4 \times 16}$ (whose $r$-th row is the conjugate bra $\langle \psi_{s,r}|$ or transposed ket $\psi_{s,r}^\top \in \mathbb{C}^{1 \times 16}$) by total wavepacket energy ($\tilde{\Psi}_s = \Psi_s / \|\Psi_s\|_F$) so that $\tilde{\Psi}_s^\dagger \tilde{\Psi}_s$ forms a **trace-1, rank-$R$ Hermitian positive semi-definite (PSD) density matrix** ([`fast_rank_r_hermitian_and_ket_pack`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L135)):
     $$\rho_s^{\text{tok}} = \tilde{\Psi}_s^\dagger \tilde{\Psi}_s = \frac{\sum_{r=1}^4 |\psi_{s,r}\rangle\langle\psi_{s,r}|}{\sum_{r=1}^4 \|\psi_{s,r}\|_2^2} \in \mathbb{C}^{16 \times 16}, \qquad \mathrm{Tr}(\rho_s^{\text{tok}}) = 1$$
   - Using an isometric packing $\tilde{P} : \mathrm{Herm}(16) \to \mathbb{R}^{256}$ (which stores the 16 real diagonal entries and $\sqrt{2}$ times the 120 upper-triangular real and 120 lower-triangular imaginary off-diagonal entries so that $\|\tilde{P}(\rho)\|_2^2 = \|\rho\|_F^2 = \mathrm{Tr}(\rho^2)$), each token packs into a `416`-dim real vector per head:
     - $\tilde{P}(\rho_s^{\text{tok}}) \in \mathbb{R}^{256}$ (`dims [0:256]`),
     - the unit-norm coherent first-moment ket $|\hat{v}_s^{\text{coh}}\rangle = \frac{\sum_{r=1}^4 |\psi_{s,r}\rangle}{\|\sum_{r=1}^4 |\psi_{s,r}\rangle\|_2} \in \mathbb{C}^{16} \cong \mathbb{R}^{32}$ (`dims [256:288]`), and
     - the 4 flattened complex wavepacket branches $\{|\psi_{s,r}\rangle\}_{r=1}^4 \in \mathbb{C}^{4 \times 16} \cong \mathbb{R}^{128}$ (`dims [288:416]`).
   - A single causal attention GEMM (`(B, H, T, T) @ (B, H, T, 416)`) aggregates all three across $s \le t$:
     $$\rho_t^{(0)} = \sum_{s \le t} w_{t,s}\,\rho_s^{\text{tok}}, \qquad |\bar{v}_t\rangle = \sum_{s \le t} w_{t,s}\,|\hat{v}_s^{\text{coh}}\rangle, \qquad |\bar{\psi}_{t,r}\rangle = \sum_{s \le t} w_{t,s}\,|\psi_{s,r}\rangle \quad (r=1..4)$$

5. **Bures-Wasserstein McWeeny Purification, $U(16)$ Evolution, and Uhlmann Phase-Locked Krylov Readout (`320` dims/head)**:
   - $\rho_t^{(0)}$ undergoes 1-step inversion-free **Bures-Wasserstein McWeeny purification** ([`bures_wasserstein_mcweeny_step`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L171)) followed by per-head learned unitary conjugation $U_h \in U(16)$ ([`_mcweeny_uhlmann_krylov_step_impl`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L165)):
     $$\rho_t^{(1)} = \frac{\left(\rho_t^{(0)}\right)^2}{\mathrm{Tr}\!\left(\left(\rho_t^{(0)}\right)^2\right)}, \qquad \rho_{t,\text{out}} = U_h\,\rho_t^{(1)}\,U_h^\dagger$$
   - Consistent with $\rho_{t,\text{out}} = U_h \rho_t^{(1)} U_h^\dagger$, the column anchor ket $|\bar{v}_t\rangle$ and the 4 column wavepacket kets $\{|\bar{\psi}_{t,r}\rangle\}_{r=1}^4$ are evolved by the same unitary $U_h$:
     $$|\bar{v}_{t,U}\rangle = U_h\,|\bar{v}_t\rangle \in \mathbb{C}^{16}, \qquad |\bar{\psi}_{t,r,U}\rangle = U_h\,|\bar{\psi}_{t,r}\rangle \in \mathbb{C}^{16} \quad (r=1..4)$$
   - **Uhlmann Phase-Locked Rank-2 Lanczos-Krylov Extraction** ([`extract_rank2_krylov_eigenpair_uhlmann`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L271)):
     1. Normalize the evolved anchor ket: $|\phi_1^{(0)}\rangle = \frac{|\bar{v}_{t,U}\rangle}{\||\bar{v}_{t,U}\rangle\|_2 + \epsilon}$.
     2. Apply one Hermitian matrix-vector step: $|\tilde{\phi}_1\rangle = \frac{\rho_{t,\text{out}}|\phi_1^{(0)}\rangle}{\|\rho_{t,\text{out}}|\phi_1^{(0)}\rangle\|_2 + \epsilon}$.
     3. Compute overlap $c = \langle \phi_1^{(0)} \mid \tilde{\phi}_1 \rangle = \sum_{j=1}^{16} \overline{\phi_{1,j}^{(0)}}\,\tilde{\phi}_{1,j}$ and rotate $|\tilde{\phi}_1\rangle$ by the **Uhlmann parallel-transport phase factor** $e^{-i\arg(c)} = \frac{\overline{c}}{|c|}$ so that $|\phi_1\rangle$ is gauge-locked to the anchor ket with strictly real, non-negative inner product:
        $$|\phi_1\rangle = \frac{\overline{\langle \phi_1^{(0)} \mid \tilde{\phi}_1 \rangle}}{\big|\langle \phi_1^{(0)} \mid \tilde{\phi}_1 \rangle\big|}\,|\tilde{\phi}_1\rangle \implies \langle \phi_1^{(0)} \mid \phi_1 \rangle = \big|\langle \phi_1^{(0)} \mid \tilde{\phi}_1 \rangle\big| \in \mathbb{R}_{\ge 0}$$
        with first Rayleigh quotient $\lambda_1 = \mathrm{clamp}\!\big(\langle \phi_1 \mid \rho_{t,\text{out}} \mid \phi_1 \rangle, 0, 1\big)$.
     4. Form the Lanczos residual $|u_2\rangle = \rho_{t,\text{out}}|\phi_1\rangle - \lambda_1|\phi_1\rangle$, apply a second modified Gram-Schmidt projection $|u_2^\perp\rangle = |u_2\rangle - \langle \phi_1 \mid u_2 \rangle|\phi_1\rangle$ for numerical orthogonality, normalize $|\phi_2\rangle = \frac{|u_2^\perp\rangle}{\||u_2^\perp\rangle\|_2 + \epsilon}$ (which inherits phase from $|\phi_1\rangle$), and compute $\lambda_2 = \mathrm{clamp}\!\big(\langle \phi_2 \mid \rho_{t,\text{out}} \mid \phi_2 \rangle, 0, 1\big)$.
   - **Per-Token RMS-Normalized `320`-Dim Readout**:
     We apply per-token `F.rms_norm` separately to:
     - $\text{packed\_rho}_{\text{out}} = \mathrm{RMSNorm}\!\big(\tilde{P}(\rho_{t,\text{out}})\big) \in \mathbb{R}^{256}$,
     - $\psi_{U,\text{flat}} = \mathrm{RMSNorm}\!\big(\mathrm{Concat}_{r=1}^4[\mathrm{Re}|\bar{\psi}_{t,r,U}\rangle, \mathrm{Im}|\bar{\psi}_{t,r,U}\rangle]\big) \in \mathbb{R}^{128}$, and
     - $\text{krylov}_{64} = \mathrm{RMSNorm}\!\big([\sqrt{\lambda_1}\mathrm{Re}|\phi_1\rangle, \sqrt{\lambda_1}\mathrm{Im}|\phi_1\rangle, \sqrt{\lambda_2}\mathrm{Re}|\phi_2\rangle, \sqrt{\lambda_2}\mathrm{Im}|\phi_2\rangle]\big) \in \mathbb{R}^{64}$.
     To inject the 128-dim linear wavepacket $\psi_{U,\text{flat}}$ without expanding the per-head readout width from `320` to `448` (preserving the exact `St(1920, 768)` parameter count and VRAM footprint), we additively fuse $\psi_{U,\text{flat}}$ into the first 128 coordinates of $\text{packed\_rho}_{\text{out}}$, keep coordinates `128:256` unchanged, and concatenate $\text{krylov}_{64}$:
     $$\text{obs}_{320} = \mathrm{Concat}\!\left[\frac{1}{\sqrt{2}}\big(\text{packed\_rho}_{\text{out}, 0:128} + \psi_{U,\text{flat}}\big),\;\; \text{packed\_rho}_{\text{out}, 128:256},\;\; \text{krylov}_{64}\right] \in \mathbb{R}^{320}$$
     Concatenating across all $H=6$ heads (`320 * 6 = 1,920` dims) and projecting via $W_{\text{obs}}^{(\ell)} \in \mathrm{St}(1920, 768)$ with scalar gain $\alpha_{\text{obs}}^{(\ell)}$ (initialized to `0.25`) gives:
     $$X_{\text{mid}}^{(\ell)} = X_{\text{cur}}^{(\ell)} + \alpha_{\text{obs}}^{(\ell)}\,\mathrm{Concat}_{h=1}^6(\text{obs}_{320, h})\,W_{\text{obs}}^{(\ell)}$$

6. **Two-Beam Mach-Zehnder Quantum Interference Channel Sublayer**:
   - Two Stiefel beam-splitter projections $W_{\text{beam},A}^{(\ell)}, W_{\text{beam},B}^{(\ell)} \in \mathrm{St}(3072, 768)$ act on $M = \mathrm{RMSNorm}(X_{\text{mid}}^{(\ell)})$ to compute constructive minus destructive intensity interference before projecting back via $W_{\text{chan}}^{(\ell)} \in \mathrm{St}(3072, 768)$ with scalar gain $\alpha_{\text{chan}}^{(\ell)}$ (initialized to `0.25`):
     $$I_{\text{MZ}} = \mathrm{ReLU}(M W_{\text{beam},A}^{(\ell)\top} + M W_{\text{beam},B}^{(\ell)\top})^2 - \frac{1}{4}\mathrm{ReLU}(M W_{\text{beam},A}^{(\ell)\top} - M W_{\text{beam},B}^{(\ell)\top})^2, \qquad X^{(\ell+1)} = X_{\text{mid}}^{(\ell)} + \alpha_{\text{chan}}^{(\ell)}\,I_{\text{MZ}}\,W_{\text{chan}}^{(\ell)}$$

7. **Final RMSNorm & Selectable Vocabulary Readout Modes (`SOFTCAP_LINEAR_CE` vs. `RANK2_KRYLOV_BORN`)**:
   - After Layer 11 outputs $X^{(12)}$, we compute $H_{\text{final}} = \mathrm{RMSNorm}(X^{(12)}) \in \mathbb{R}^{B \times T \times 768}$ and map to vocabulary logits $z_{t,v}$ ($v \in \{0, \dots, 50303\}$) using one of two selectable readout modes:
     - **Mode A — `SOFTCAP_LINEAR_CE` (Used in all `Wave 48 v1`, `v2`, and `v3` FineWeb speedrun benchmarks in Section 5)**:
       $$z_{t,v} = C_{\text{cap}}\tanh\!\left(\frac{s_{\text{eff}}\,\langle h_{\text{final},t}, w_{\text{lm},v}\rangle}{C_{\text{cap}}}\right), \qquad C_{\text{cap}} = 15.0$$
       where $w_{\text{lm},v} \in S^{767}(\sqrt{768})$ is row $v$ of `lm_head.weight` and $s_{\text{eff}} = \mathrm{clamp}(\text{logit\_scale}, s_{\min}, 8s_{\min})$ with $s_{\min} = 0.75/\sqrt{768}$.
     - **Mode B — `RANK2_KRYLOV_BORN` (Quantum Born Log-Intensity Readout, [`Rank2KrylovBornChunkedCEFn`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L480), verified in Gates `H1`–`H3`)**:
       Splits $h_{\text{final},t} = [h_{t,1}, h_{t,2}] \in \mathbb{R}^{2 \times 384}$ and $w_{\text{lm},v} = [w_{v,1}, w_{v,2}] \in \mathbb{R}^{2 \times 384}$, weights the two sub-spaces by head-averaged Krylov eigenvalue ratios $\bar{\kappa}_{t,m} = \frac{\bar{\lambda}_{t,m}}{\bar{\lambda}_{t,1} + \bar{\lambda}_{t,2}}$ ($m \in \{1,2\}$), and computes softcapped log-Born intensity:
       $$I_{t,v} = \sum_{m=1}^2 \bar{\kappa}_{t,m}\left(1 + \frac{\langle h_{t,m}, w_{v,m}\rangle}{2\sqrt{384}}\right)^2, \qquad z_{t,v} = C_{\text{cap}}\tanh\!\left(\frac{\tau_{\text{born}}\log(I_{t,v} + \epsilon_{\text{born}})}{C_{\text{cap}}}\right)$$
       where $\tau_{\text{born}} = \exp(\text{born\_log\_temp})$ (`DEFAULT_KRYLOV_BORN_LOG_TEMP = 6.0`, $\epsilon_{\text{born}} = 0.01$).

### 2.3. Full-Network Riemannian Manifold Optimizers & Update Equations
Every matrix parameter in [`StandaloneQuantumBornGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L708) is updated on its Riemannian manifold:

1. **Real Stiefel Manifold Optimizer ([`AllLayerTikhonovRiemannianStiefelMuon`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py#L650), $\eta_{\text{stiefel}} = 0.025$, $\beta = 0.90$, $\lambda_{\text{tik}} = 10^{-3}$, $K=5$ NS steps)**:
   - Operates on the tall orientation $W \in \mathrm{St}(D, d)$ ($D \ge d$, $W^\top W = I_d$; wide matrices are transposed before and after the step):
     1. *Nesterov Momentum*: $M \leftarrow \beta M + G$, $G_{\text{nest}} = G + \beta M$.
     2. *Tangent / Horizontal Projection*: $M_{\text{tan}} = G_{\text{nest}} - \frac{1}{2}W(W^\top G_{\text{nest}} + G_{\text{nest}}^\top W)$ if $D=d$, or $M_{\text{tan}} = G_{\text{nest}} - W(W^\top G_{\text{nest}})$ if $D > d$.
     3. *Tikhonov-Damped Newton-Schulz Polar Factor*: Normalize $X_0 = M_{\text{tan}} / \|M_{\text{tan}}\|_F$, run $K=5$ iterations of $X_{k+1} = \frac{1}{2}X_k(3I_d - (X_k^\top X_k + \delta_{k,0}\lambda_{\text{tik}}I_d))$, and re-project $U_{\text{tan}}$ onto the tangent space at $W$.
     4. *Geodesic Retraction + 1-Step Schulz Polish*: $W_{\text{raw}} = \frac{W - \eta_{\text{stiefel}} U_{\text{tan}}}{\sqrt{1 + \eta_{\text{stiefel}}^2}}$, followed by $W_{\text{next}} = \frac{1}{2}W_{\text{raw}}(3I_d - W_{\text{raw}}^\top W_{\text{raw}})$.

2. **Complex Unitary Group Optimizer ([`riemannian_complex_unitary_muon_step`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L370) on $U_h \in U(16)$, $\eta_{\text{unitary}} = 0.020$, $\beta = 0.90$, $\lambda_{\text{tik}} = 10^{-3}$)**:
   - Projects complex Nesterov momentum $G_{\text{nest}} \in \mathbb{C}^{16 \times 16}$ onto the skew-Hermitian tangent space via $\Omega = \frac{1}{2}(U_h^\dagger G_{\text{nest}} - G_{\text{nest}}^\dagger U_h)$ and $T = U_h \Omega$, applies 6-step complex Newton-Schulz orthogonalization $Z = \mathrm{ComplexNS}(T)$, re-projects $T_{\text{unit}} = \frac{1}{2}U_h(U_h^\dagger Z - Z^\dagger U_h)$, and retracts via $U_{\text{raw}} = \frac{U_h - \eta_{\text{unitary}} T_{\text{unit}}}{\sqrt{1 + \eta_{\text{unitary}}^2}}$ followed by a complex Schulz polish $\frac{1}{2}U_{\text{raw}}(3I_{16} - U_{\text{raw}}^\dagger U_{\text{raw}})$.

3. **Vocabulary Hypersphere Optimizer ([`BatchSubspaceTikhonovGramMuon`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py#L105) on $S^{767}(\sqrt{768})$, $\eta_{\text{vocab}} = 0.06$, $\beta_1 = 0.90$, $\beta_2 = 0.99$, $\tau_{\text{snr}} = 0.50$, $\alpha_{\text{mix}} = 0.25$)**:
   - **Algorithmic Difference Between `step_active(idx_act)` and `step_full()`**:
     - `step_active(idx_act)` selects **only the subset of active vocabulary rows** $W_{\text{act}} \in \mathbb{R}^{N_{\text{act}} \times 768}$ ($N_{\text{act}} \approx 1,800 \ll 50,304$), leaving the momentum buffers, second-moment buffers, and weights of all other $50,304 - N_{\text{act}}$ inactive rows completely untouched.
     - `step_full()` runs the update over **all $N = 50,304$ rows** simultaneously.
   - Given the selected rows $W_S \in \mathbb{R}^{|S| \times 768}$ ($S = \text{idx\_act}$ or $S = \{0..50303\}$) with Euclidean gradient $G_S$:
     1. *Sphere Tangent Projection & EMA Moments*: Project $G_{\text{tan},v} = G_{S,v} - \frac{\langle G_{S,v}, W_{S,v}\rangle}{\|W_{S,v}\|_2^2} W_{S,v}$, update first moment $M_{S,v} \leftarrow \beta_1 M_{S,v} + (1-\beta_1)G_{\text{tan},v}$ and second moment $v_{S,v} \leftarrow \beta_2 v_{S,v} + (1-\beta_2)\|G_{\text{tan},v}\|_2^2$, and re-project $M_{\text{tan},v}$ onto $T_{W_{S,v}}S^{767}(\sqrt{768})$.
     2. *Coupled Newton-Schulz $768 \times 768$ Feature-Gram Whitening*: Form the $768 \times 768$ feature Gram matrix $\Gamma = M_{\text{tan}}^\top M_{\text{tan}} / \big(\frac{1}{768}\mathrm{Tr}(M_{\text{tan}}^\top M_{\text{tan}})\big)$, damp via $A = (1 - \alpha_{\text{mix}} + \lambda_{\text{tik}})I_{768} + \alpha_{\text{mix}}\Gamma$, compute $A^{-1/2}$ via 6-step coupled Newton-Schulz ($Y_0 = A/\|A\|_F, Z_0 = I_{768}, T_k = \frac{1}{2}(3I - Z_k Y_k), Y_{k+1} = Y_k T_k, Z_{k+1} = T_k Z_k$), and whiten $U_{\text{tan}} = \mathrm{Proj}_{\text{tan}}(M_{\text{tan}} A^{-1/2})$.
     3. *SNR Trust-Region Great-Circle Geodesic Retraction*: With per-row signal-to-noise ratio $\mathrm{SNR}_v = \|M_{\text{tan},v}\|_2 / (\sqrt{v_{S,v}} + \epsilon)$ and relative magnitude $r_v = \min(2.5, \|U_{\text{tan},v}\|_2 / \mathrm{RMS}(\|U_{\text{tan}}\|_2))$, each row rotates along a great circle on $S^{767}(\sqrt{768})$ by angle $\theta_v = \eta_{\text{vocab}}\,r_v\,\frac{\mathrm{SNR}_v}{\mathrm{SNR}_v + \tau_{\text{snr}}}$:
        $$W_{S,v}^{\text{next}} = \cos(\theta_v)\,W_{S,v} - \sin(\theta_v)\,\sqrt{768}\,\frac{U_{\text{tan},v}}{\|U_{\text{tan},v}\|_2 + \epsilon}$$
        followed by exact radial normalization onto $\sqrt{768}\,S^{767}$ and tangent parallel transport of $M_{S,v}$.

4. **1D/Scalar Parameters (`AdamW`, `lr = 0.005`, $\beta = (0.9, 0.95)$, `weight_decay = 0.0`)**:
   - Updates $\alpha_{\text{in}}$, $\alpha_{\text{obs}}^{(\ell)}$, $\alpha_{\text{chan}}^{(\ell)}$, $\lambda_{\text{block}}^{(\ell)}$, $\lambda_v^{(\ell)}$, and `logit_scale` / `born_log_temp`.

### 2.4. Data Pipeline & Validation Evaluation Protocol
- **Training Stream (`900M` Tokens)**: [`MultiShardTokenStream`](file:///home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py#L105) streams contiguous `(B=8, T=1024)` batches (`8,192` tokens/step) sequentially without repetition across all 9 local FineWeb training shards (`fineweb_train_000001.bin` through `000009.bin`, `100M` GPT-2 tokens each = `900M` tokens).
- **Validation Protocol**: The official validation shard `fineweb_val_000000.bin` contains `10,485,760` tokens in total. To enable fast, deterministic evaluation every `100`–`250` steps during training and Layer 11 ablations, **all validation cross-entropy (`Val CE`), perplexity (`Val PPL`), and Top-1 accuracy numbers reported in Sections 4 and 5 are evaluated on the exact same deterministic `65,536`-token prefix (`8` contiguous batches of `B=8, T=1024`) of `fineweb_val_000000.bin`**.

---

## 3. Outcomes We Are Looking to Achieve & Verification Metrics

1. **North-Star Benchmark Target (`< 3.28` Val CE on FineWeb)**:
   - Our overarching reference benchmark is the **Modded-NanoGPT FineWeb speedrun** (`< 3.28` Validation Cross-Entropy). On our dual-RTX-3090 workstation, our milestones are: (a) proving standalone teacher-free end-to-end training, (b) matching or beating classical Scaled Dot-Product Attention (`SDPA`) when swapped into Layer 11 of a `400M`-token checkpoint, and (c) reaching `< 3.28` Val CE within a token-matched `< 330M`-token budget.
2. **Exact Physical & Geometric Telemetry Definitions (`Plan 063` Gates `H1`–`H3`)**:
   - **Per-Token Purity (`mean_purity`) & Effective Rank (`mean_eff_rank`)**:
     For each batch $b$, head $h$, and token $t$, post-McWeeny density purity is $\mathcal{P}_{b,h,t} = \mathrm{Tr}(\rho_{b,h,t}^{(1)\,2}) \in [\frac{1}{16}, 1]$, and per-token inverse-purity effective rank is $\mathcal{R}_{b,h,t} = 1 / \mathcal{P}_{b,h,t} \in [1, 16]$. We report the batch/head/token means:
     $$\text{mean\_purity} = \mathbb{E}_{b,h,t}\!\left[\mathcal{P}_{b,h,t}\right], \qquad \text{mean\_eff\_rank} = \mathbb{E}_{b,h,t}\!\left[\frac{1}{\mathcal{P}_{b,h,t}}\right]$$
     *(Because $f(x) = 1/x$ is strictly convex on $x > 0$, **Jensen's Inequality** implies $\text{mean\_eff\_rank} = \mathbb{E}[1/\mathcal{P}] > 1/\mathbb{E}[\mathcal{P}] = 1/\text{mean\_purity}$ whenever purity varies across tokens and heads; e.g., $\text{mean\_purity} = 0.236$ gives $\text{mean\_eff\_rank} = 6.91 > 1/0.236 = 4.24$, whereas a collapsed rank-1 state has $\mathcal{P}_{b,h,t} \equiv 1.0$ and $\text{mean\_eff\_rank} \equiv 1.0$.)*
   - **Uhlmann Phase-Lock Error (`uhlmann_im_err < 1e-6`, Gate `H1`)**:
     Measures $\max_{b,h,t} \big|\mathrm{Im}\langle \phi_{1,b,h,t}^{(0)} \mid \phi_{1,b,h,t} \rangle\big|$, verifying that Uhlmann parallel-transport gauge fixing locks the top Krylov eigenket $|\phi_1\rangle$ to have a strictly real, non-negative inner product with the normalized anchor ket $|\phi_1^{(0)}\rangle = |\bar{v}_{t,U}\rangle / \||\bar{v}_{t,U}\rangle\|_2$ (`1.86e-08` in our verification gate).
   - **Stiefel & Unitary Manifold Errors (`max_stiefel_err < 1e-5`, `mean_unitary_err < 1e-5`, Gate `H3`)**:
     For any tall Stiefel matrix $W \in \mathbb{R}^{m \times n}$ ($m \ge n$), $\text{stiefel\_err}(W) = \frac{1}{n}\|W^\top W - I_n\|_F$ (and $\frac{1}{m}\|W W^\top - I_m\|_F$ if $m < n$); for $U_h \in \mathbb{C}^{16 \times 16}$, $\text{unitary\_err}(U_h) = \frac{1}{16}\|U_h^\dagger U_h - I_{16}\|_F$.
   - **Vocabulary Hypersphere Error (`max_wte_sphere_rms_err < 1e-5`, Gate `H2`)**:
     Measures $\max_{v} \big|\|w_v\|_2 / \sqrt{768} - 1\big|$ (`2.38e-07` throughout training).

---

## 4. Reference Classical Baseline (`CompiledLeaderboardClassicalGPT`)
Our local classical reference model ([`CompiledLeaderboardClassicalGPT`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L670), `12` layers, `d_model = 768`, `6` heads, `head_dim = 128`) implements the Modded-NanoGPT speedrun architecture, including several macro-architectural accelerators that we intentionally omitted from `StandaloneQuantumBornGPT` during initial isolation:
- **`SmearGate`** (adjacent-token embedding mixing at $t-1 \to t$),
- **U-Net Encoder-Decoder Skip Connections** (connecting Layers `0..5` to Layers `11..6`),
- **Three Token Value Embedding Banks (`value_embeds`)** injected directly into attention values via learned gates (`ve_gate`),
- **Attention Output Gates (`attn_gate`)** and **Document-Boundary Sliding-Window Attention** (`Layers 3, 7, 11` use long windows; other layers use short windows with **YaRN**—Yet another RoPE extensioN—frequency scaling), trained at a large macro-batch size of `262,144` tokens/step.

Its FineWeb validation trajectory (from [`.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/checkpoints_manifest.json`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/checkpoints_manifest.json)) is:
- **`0M` tokens (Step 0)**: `10.8249` Val CE (`0.06%` Top-1)
- **`50M` tokens (Step 191)**: `4.2466` Val CE (`31.16%` Top-1)
- **`100M` tokens (Step 382)**: `3.9354` Val CE (`33.71%` Top-1)
- **`200M` tokens (Step 763)**: `3.6850` Val CE (`36.06%` Top-1)
- **`300M` tokens (Step 1145)**: `3.5842` Val CE (`36.88%` Top-1)
- **`400M` tokens (Step 1526)**: `3.4665` Val CE with sliding-window masks (`37.99%` Top-1), or **`3.5100` Val CE** under standard full-causal evaluation (`use_flex_doc_swc = False`) on the `65,536`-token validation slice.

---

## 5. Steps We Have Taken & Quantitative Results

### Step 1: First Standalone FineWeb Run (`Wave 48 v1`) & Root-Cause Diagnosis
Running `StandaloneQuantumBornGPT` (`Wave 48 v1`, `SOFTCAP_LINEAR_CE` readout) end-to-end from scratch on 1 GPU (`B=8, T=1024 = 8,192` tokens/step, `8,271 tok/s`) lowered Val CE from `10.8258` (Step 0) $\to$ `6.5786` (`4.10M` tokens) $\to$ `6.2248` (`8.19M` tokens) $\to$ `5.6991` (`20.48M` tokens, `17.56%` Top-1) $\to$ `4.9189` (`120.4M` tokens), where we paused the run to diagnose why convergence trailed the classical baseline.

Inspecting internal activations, density spectra, and gradients revealed **five architectural/mathematical bottlenecks** in the quantum block (corresponding 1-to-1 with `Fix 1` through `Fix 5` in Step 2 below), plus a **runner-script `grad_accum = 4` bug**:

1. **Bottleneck 1 (`Fix 1`) — Value-RoPE Axis Scrambling & `1/16` Unnormalized `obs_320` RMS Attenuation**:
   - In `Wave 48 v1`, rotary embeddings were applied to `v_re, v_im` after flattening `(B, H, T, R=4, d_h=16)` into `(B, H, 4*T, 16)`, applying position $4t + r$ rotary phase to token $t$'s $r$-th value branch and scrambling positional alignment with queries/keys.
   - Simultaneously, because a trace-1 density matrix with purity $\sim 0.10$ spread across 256 coordinates has per-coordinate RMS $\sqrt{0.10 / 256} \approx 0.02 \ll 1.0$, the raw `obs_concat` feature vector had $\sim 1/16$ the RMS amplitude expected by $W_{\text{obs}} \in \mathrm{St}(1920, 768)$. In `Fix 1`, we removed Value-RoPE and applied a single joint `F.rms_norm(obs_concat, (1920,))` before $W_{\text{obs}}$.
2. **Bottleneck 2 (`Fix 2`) — Radial Purity-Collapse Incentive When Density & Krylov Blocks Are Not Normalized Separately**:
   - Even with joint `F.rms_norm(obs_concat, (1920,))` in `Fix 1`, within each head the isometric density block $\tilde{P}(\rho_{t,\text{out}}) \in \mathbb{R}^{256}$ has Euclidean L2 norm $\|\tilde{P}(\rho_{t,\text{out}})\|_2 = \sqrt{\mathrm{Tr}(\rho_{t,\text{out}}^2)} = \sqrt{\text{purity}}$, and `krylov_raw` has L2 norm $\sqrt{\lambda_1 + \lambda_2}$.
   - Without separate per-head, per-token `F.rms_norm` on `packed_rho_256` (`256` dims) and `krylov_64` (`64` dims), any downstream gradient trying to increase a head's feature weight entering $W_{\text{obs}}$ directly rewarded $W_v$ for maximizing $\mathrm{Tr}(\rho_{t,\text{out}}^2) \to 1.0$!
   - By strict convexity of $\mathrm{Tr}(\rho^2)$, $\rho_t^{(0)} = \sum_{s \le t} w_{t,s} \rho_s^{\text{tok}}$ achieves maximum purity (`= 1.0`) when all tokens $s$ collapse toward the **same constant rank-1 pure state** (within 40 steps, token-to-token density cosine jumped from `0.215` to `0.707` and post-McWeeny purity collapsed from `0.100` to **`0.989` (`eff_rank = 1.01`)**). Applying per-token `F.rms_norm` separately to `packed_rho_256` (`256` dims) and `krylov_64` (`64` dims) in `Fix 2` made their norms invariant to $\mathrm{Tr}(\rho_{t,\text{out}}^2)$ and evolved the Uhlmann anchor ket by $U_h$, completely eliminating purity collapse (`purity = 0.236, eff_rank = 6.91`).
3. **Bottleneck 3 (`Fix 3`) — $4\times$ Coherent Wavepacket Rank Bottleneck (`768 -> 192` Dims)**:
   - Because $\rho_s^{\text{tok}} = \tilde{\Psi}_s^\dagger \tilde{\Psi}_s \in \mathbb{C}^{16 \times 16}$ is invariant under any $U(4)$ rotation across the $R=4$ branches, and `fast_rank_r_hermitian_and_ket_pack` summed the $R=4$ branches (`v_re.sum(dim=-2)`) into a single `32`-real-dim ket $|\hat{v}_s^{\text{coh}}\rangle$ per head (`6 * 32 = 192` dims across 6 heads), the linear coherent pathway had a $4\times$ rank bottleneck. `Fix 3` transports all $R=4$ branches (`128` real dims/head), evolves them by $U_h$, and fuses `psi_u_flat` directly into `obs_320`.
4. **Bottleneck 4 (`Fix 4`) — Hilbert-Space Frame Noise Floor on $\mathbb{C}^{16}$ vs. $\mathbb{C}^{64}$**:
   - In `Wave 48 v1`, queries and keys only projected to $1$ branch of $\mathbb{C}^{16}$ per head (`192` real dims total) with $p=2$ Born weights ($|\langle q_t \mid k_s \rangle|^4$).
   - By Haar measure integration on the complex unit sphere in $\mathbb{C}^d$, two independent random unit kets have expected $2p$-th overlap:
     $$\mathbb{E}\!\left[|\langle q_t \mid k_s \rangle|^{2p}\right] = \frac{p!}{d(d+1)\cdots(d+p-1)}$$
   - For $d = 16, p = 2$, $\mathbb{E}[|\langle q_t \mid k_s \rangle|^4] = \frac{2}{16 \times 17} = \frac{1}{136} \approx 0.00735$. Summing over $512$ causal positions creates a random background floor of $512 / 136 \approx \mathbf{3.76}$, drowning out a single aligned target token ($\le 1.0$)!
   - In `Fix 4`, expanding $Q, K$ to the full 4-branch wavepacket dimension $d = R \times d_h = \mathbf{64}$ complex dims/head (`768` real dims total, matching $W_v$) with $p = 3$ Born coincidence ($|\langle q_t \mid k_s \rangle|^6$) reduces the random pair expectation by **$336\times$**:
     $$\mathbb{E}\!\left[|\langle q_t \mid k_s \rangle|^6\right] = \frac{3!}{64 \times 65 \times 66} = \frac{6}{274,560} = \frac{1}{45,760} \approx 2.18 \times 10^{-5}$$
     Even summed over $T = 1024$ causal tokens, the random background floor is only **$1024 / 45,760 \approx 0.0224$**!
5. **Bottleneck 5 (`Fix 5`) — Direct Value-Embedding (`ve_11`) Coordinate Alignment & Partial Key Offset**:
   - In the classical checkpoint's Layer 11, the pre-trained token value embedding $\text{ve}_{11}(\text{idx}) \in \mathbb{R}^{B \times T \times 768}$ (`value_embeds[2]`) is already aligned to Layer 11's head space and added directly via $V_{\text{mixed}} = s_0 V_{\text{proj}} + s_1 \text{ve}_{11}$. In `Fix 1–4`, our swap harness multiplied $\text{ve}_{11}$ by a random Stiefel matrix $W_{\text{ve}} \in \mathrm{St}(768, 768)$, scrambling the pre-trained token value embeddings. `Fix 5` injects $\text{ve}_{11}$ directly (`s_0 * (H_norm @ W_v) + s_1 * ve_11` with learnable $s_0, s_1$) and enables Layer 11's `apply_partial_key_offset`.
6. **Runner-Script Bug — Silent `4x` Optimizer Step Reduction (`grad_accum = 4`)**:
   - In [`73_verify_plan063_standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/examples/73_verify_plan063_standalone_quantum_born_gpt.py), a conditional set `grad_accum = 4` whenever `steps >= 1000`, cutting parameter updates per token by $4\times$ on long standalone runs. We fixed `grad_accum = 1` unconditionally.

### Step 2: Surgical Layer 11 Swap Ablations on the `400M`-Token Classical Checkpoint
We loaded the `400M`-token `CompiledLeaderboardClassicalGPT` checkpoint (`3.5100` full-causal Val CE on the `65,536`-token validation slice), froze Layers `0..10`, embeddings, `lm_head`, and Layer 11's MLP (`fc1, fc2`), and zeroed out **only Layer 11's attention sublayer** (`Zero-Attention Baseline = 3.8407` Val CE, a `+0.3307`-nat penalty). We then trained replacement Layer 11 attention modules on unseen tokens from Shard 5 (`fineweb_train_000005.bin`, `B=8, T=1024`):

- **Classical Reference Baselines (`Ref A` vs. `Ref B`)**:
  - **`Ref A: Cold-Start Classical Layer 11 (SDPA Softmax)`**: Resets all of Layer 11's attention parameters—$W_q, W_k, W_v, W_o$ (`qkv_proj`, `out_proj`), value mixing scalars `sa_lambdas` ($[0.5, 0.5]$), value-embedding gate `ve_gate` ($0$), attention output gate `attn_gate` ($0$), and cross-head alpha `xsa_alpha` ($0$)—to their Step-0 initialization while keeping the pre-trained token table `value_embeds[2]` (`ve_11`) and Layer 11's MLP (`fc1, fc2`) frozen, training $W_q, W_k, W_v, W_o$ with Stiefel-Muon on the exact same batches and cosine learning-rate schedule (`0.04 -> 0.006`) as Quantum Layer 11.
  - **`Ref B: Warm-Gate Classical Layer 11 (SDPA Softmax)`**: Resets only $W_q, W_k, W_v, W_o$ while **retaining the 400M-token checkpoint's pre-trained `sa_lambdas` ($[2.31, 5.75]$), `ve_gate`, `attn_gate`, and `xsa_alpha`**.

| Layer 11 Attention Configuration (`Layers 0..10` & `B11 MLP` Frozen) | 200-Step Val CE (`1.64M` tok) | 2,000-Step Val CE (`16.38M` tok) | Gain vs. Zero-Attn (`3.8407`) | Post-McWeeny `mean_purity` | `mean_eff_rank` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full 12L Classical Checkpoint Ceiling (`400M` tok, full causal)** | `3.5100` | `3.5100` | `+0.3307` nats (`100%`) | — | — |
| **0. Layer 11 Zero-Attention Baseline (`attn_delta = 0`, MLP intact)** | `3.8407` | `3.8407` | `+0.0000` nats (`0.0%`) | — | — |
| **1. Original `Wave 48 v1` `StandaloneQuantumBornBlock`** | `3.8393` | — | `+0.0014` nats | `0.982` *(collapsed)* | `1.02` |
| **2. `+ Fix 1`: Remove Value-RoPE & Add Joint `F.rms_norm(obs_concat)`** | `3.8059` | — | `+0.0348` nats | `0.989` *(collapsed)* | `1.01` |
| **3. `+ Fix 2`: Separate Per-Token `F.rms_norm` on `packed_rho_256` & `krylov_64`** | `3.8026` | — | `+0.0381` nats | **`0.236`** *(healthy)* | **`6.91`** |
| **4. `+ Fix 3`: Preserve All $R=4$ Coherent Wavepacket Branches (`psi_u_flat`)** | `3.7950` | — | `+0.0457` nats | **`0.231`** | **`6.11`** |
| **5. `+ Fix 4`: Full-Rank $Q, K \in \mathbb{C}^{64}$ + $p=3$ Born $|\langle q_t \mid k_s \rangle|^6$** | **`3.7754`** | **`3.7517`** | **`+0.0890` nats (`26.9%`)** | **`0.335`** | **`3.06`** |
| **6. `+ Fix 5`: Direct `ve_11` Injection (`s0*V + s1*ve_11`) + Key Offset** | **`3.7609`** | — | **`+0.0798` nats** *(in 200 steps)* | **`0.153`** | **`8.53`** |
| **Ref A: Cold-Start Classical Layer 11 (`SDPA` Softmax, same init/schedule)** | `3.7915` | `3.7660` | `+0.0747` nats (`22.6%`) | — | — |
| **Ref B: Warm-Gate Classical Layer 11 (`SDPA` Softmax, pre-trained gates)** | `3.7656` | — | `+0.0751` nats | — | — |

### Step 3: Porting Fixes to `StandaloneQuantumBornGPT`, Memory/Speed Optimization, and Boundary Fixes (`v2` & `v3`)
We landed all five fixes into [`src/qdlblocks/models/standalone_quantum_born_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/standalone_quantum_born_gpt.py):
- **VRAM & Throughput Optimization**: Computing $\langle q_t \mid k_s \rangle$ on $\mathbb{C}^{64}$ via the 2-GEMM `q_cat, k_cat, k_skew` formulation and compiling `_causal_born_p3_weights_impl` and `_mcweeny_uhlmann_krylov_step_impl` with `@torch.compile` reduced 12-layer `(B=8, T=1024)` peak memory from `26.13 GiB` (OOM) to **`15.67 GB` (`-10.46 GB`)** and boosted single-GPU throughput from `8,271 tok/s` to **`12,329–13,100 tok/s` (`+49%` to `+58%`)**.
- **Standalone Run `Wave 48 v2` (`2,500` steps = `20.48M` tokens, `SOFTCAP_LINEAR_CE` readout, [`.cache/w48_v2_fixed_standalone_softcap_20m.json`](file:///home/nixos/QuantumDLBlocks/.cache/w48_v2_fixed_standalone_softcap_20m.json))**:
  - Reached **`5.2494` Val CE (`22.02%` Top-1, `mean_purity = 0.437`, `mean_eff_rank = 3.45`)** at `20.48M` tokens—beating `Wave 48 v1` (`5.6991` Val CE, `17.56%` Top-1) by **`-0.4497` nats** and **`+4.46%` Top-1**.
- **Uncovering Two Standalone Boundary Bugs (`v2` $\to$ `v3`)**:
  1. **`logit_scale` Zero-Crossing & Sign Flip at Step 8**: In `SOFTCAP_LINEAR_CE` mode, rows of $W_{\text{lm}}$ have L2 norm $\sqrt{768}$ and `logit_scale` is initialized at $1/\sqrt{768} \approx 0.03608$. At Step 0, unit-variance random logits yielded `Step 0 Val CE = 11.2913` (worse than uniform $\ln(50304) = 10.8258$), creating a `+26.85` gradient on `logit_scale` that caused `AdamW(lr=0.005)` to push `logit_scale` **through zero at Step 8 into negative values (`-0.0118`)**! For the first `~850` steps of `v2`, the network multiplied its logits by a negative scalar before `logit_scale` crossed back above zero around Step 900. Clamping `eff_scale = self.logit_scale.clamp(min=0.75/sqrt(d), max=8.0*min_scale)` completely prevented the sign flip.
  2. **Dense Softmax Gradient Overwriting of Inactive Vocabulary Rows in `head_opt` & Forward `F.rms_norm(wte)`**:
     - Because softmax probabilities $p_{t,v} > 0$ are non-zero for all $50,304$ vocabulary rows every batch, `lm_head.weight` receives dense gradients on all $50,304$ rows, and calling `head_opt.step_full(lr_scale=1.0)` rotated all `~48,500` absent vocabulary rows by `~0.06` radians per step. Splitting `head_opt` into `0.75 * step_active(idx U targets) + 0.25 * step_full()` focused $75\%$ of the geodesic update on active batch/target tokens while retaining $25\%$ global negative-sampling drift.
     - Additionally, even though `wte.weight` rows are retracted onto $S^{767}(\sqrt{768})$ (`float32` RMS $= 1.0$) after each step, casting `self.wte(idx)` to `bfloat16` introduces slight quantization norm drift, and without `F.rms_norm` in the forward graph, radial gradients parallel to $e_v$ flow into `wte.weight.grad`. Applying `F.rms_norm(self.wte(idx).to(bfloat16), (768,))` in `forward_hidden` both restores exact `bfloat16` unit RMS and multiplies incoming backward gradients by the Jacobian of $\mathrm{RMSNorm}$ at unit RMS, $J_{\mathrm{RMSNorm}}(e_v) = I_{768} - \frac{1}{768}e_v e_v^\top$, automatically projecting embedding gradients onto the sphere tangent space $T_{e_v}S^{767}(\sqrt{768})$ during backpropagation.
- **Standalone Run `Wave 48 v3` (`1,000` steps = `8.19M` tokens, `SOFTCAP_LINEAR_CE` readout, [`.cache/w48_v3_clamped_standalone_softcap_8m.json`](file:///home/nixos/QuantumDLBlocks/.cache/w48_v3_clamped_standalone_softcap_8m.json))**:
  - Reached **`5.7103` Val CE (`18.87%` Top-1, `5.1046` Train CE, `mean_purity = 0.455`, `mean_eff_rank = 3.28`)** at `8.19M` tokens—**`-0.8112` nats lower** than `v2` at `8.19M` tokens (`6.5215`) and matching `Wave 48 v1`'s `20.48M`-token Val CE (`5.6991`) in **`2.5x` fewer tokens**!

| Standalone 12L FineWeb Run (`B=8, T=1024`, `SOFTCAP_LINEAR_CE`) | Throughput (`tok/s`) | Val CE @ `4.10M` | Val CE @ `8.19M` (`Top-1`) | Val CE @ `20.48M` (`Top-1`) | `mean_purity` / `mean_eff_rank` | `max_stiefel_err` / `sphere_err` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Wave 48 v1` (Original Standalone)** | `8,271` | `6.5786` | `6.2248` (`15.32%`) | `5.6991` (`17.56%`) | `1.000` / `1.00` *(Collapsed)* | `6.4e-07` / `2.4e-07` |
| **`Wave 48 v2` (+ Fixes 1–4 & `grad_accum=1`)** | **`13,060`** | `6.8059`* | `6.5215`* (`13.87%`) | **`5.2494` (`22.02%`)** | **`0.437` / `3.45`** | `6.4e-07` / `2.4e-07` |
| **`Wave 48 v3` (+ `logit_scale` Clamp & Active `head_opt`)** | **`12,329`** | **`6.1688`** | **`5.7103` (`18.87%`)** | *(Next to run)* | **`0.455` / `3.28`** | `6.4e-07` / `2.4e-07` |

*\*Footnote: In `Wave 48 v2`, `self.logit_scale` was unclamped and crossed zero into negative values (`-0.0118`) at Step 8 before recovering positive sign around Step 900 (`7.37M` tokens), temporarily depressing the `4.10M` and `8.19M` scores before `v2` surged to `5.2494` Val CE at `20.48M` tokens. `Wave 48 v3` clamps `logit_scale >= 0.75/sqrt(768)` from Step 0.*

---

## 6. Recommended Next Steps

1. **Run a `20.48M`–`100M` Token Standalone Trajectory of `Wave 48 v3`**:
   - Since `v3` (`5.7103` Val CE at `8.19M` tokens) is `0.8112` nats ahead of `v2` at `8.19M` tokens (`6.5215`), and `v2` dropped `-1.27` nats between `8.19M` and `20.48M` tokens (`5.2494`), running `v3` out to `20.48M`–`100M` tokens should break below `5.0` Val CE by `20M` tokens and establish the clean standalone scaling curve.
2. **Multi-Layer Classical-to-Quantum Swap (`Layers 3, 7, 11`) & Extended Direct-`ve_11` Curve**:
   - Run Configuration 6 (`Fix 5`: Direct `ve_11` Quantum + partial key offset, which reached `3.7609` in 200 steps) for `2,000` steps with a learnable output scale matching the classical residual stream (`RMS = 446.6`), and test swapping all three global long-window layers (`Layers 3, 7, 11`) simultaneously to evaluate multi-layer quantum phase composition.
3. **FlashAttention-Style Tiled Triton Kernel for $p=3$ Complex Born Coincidence**:
   - Currently, `@torch.compile`d `_causal_born_p3_weights_impl` materializes the `(B, H, T, T)` causal Born weight matrix in `bfloat16`. Writing a tiled Triton kernel that computes $|\langle q_t \mid k_s \rangle|^6$ and accumulates the `416`-dim packed density/ket/wavepacket vectors online in SRAM without materializing the $T \times T$ matrix in HBM will further increase throughput beyond `13,100 tok/s` and support longer contexts ($T = 2048\text{–}4096$).
4. **Integrate Modded-NanoGPT Macro-Architectural Accelerators into `StandaloneQuantumBornGPT`**:
   - To close the remaining gap between standalone training (`8,192` tokens/step without macro-heuristics) and the `CompiledLeaderboardClassicalGPT` speedrun baseline (`4.25` Val CE at `50M` tokens, `3.47` at `400M` tokens), integrate **U-Net encoder-decoder skip connections**, **token Value Embedding banks (`value_embeds`)**, **`SmearGate`**, and multi-GPU Distributed Data Parallel (DDP) across both RTX 3090 GPUs.
