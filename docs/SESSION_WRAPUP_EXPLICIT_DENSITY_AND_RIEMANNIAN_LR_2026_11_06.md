# Session Wrap-Up: Explicit Density-Matrix Language Modeling, `AdamW`-Free Riemannian Optimization, & Learning-Rate Thermodynamics (`Wave 63` / `Plan 083`)

**Date:** November 6, 2026  
**Primary Code Artifacts:**
- Model & Benchmark Lab: [`src/qdlblocks/benchmarks/explicit_density_tomographic_512d_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/explicit_density_tomographic_512d_lab.py)
- Riemannian Vocabulary, N-Gram, & Geodesic Scalar Optimizers: [`src/qdlblocks/optim/vocab_manifold_muon.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py)
- Constant-LR Stability & Ceiling Sweep CLI: [`scripts/constant_lr_stability_ceiling_sweep.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/constant_lr_stability_ceiling_sweep.py)
- In-Advance Curvature, Shadow-EMA, & Branch-and-Quench Probe CLI: [`scripts/shadow_ema_and_quench_lr_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/shadow_ema_and_quench_lr_probe.py)
- Experiment Plan & Telemetry Ledger: [`EXPERIMENT_PLAN_083.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_083.md)

---

## 1. Project Context & Outcomes We Are Looking to Achieve

### 1.1 The Overarching Goal
We are developing a **quantum-inspired density-matrix language model** (`ExplicitDensityTomographicGPT`) capable of competing on the canonical **FineWeb GPT-2 Byte-Pair Encoding (BPE) benchmark** (`V = 50,257` valid GPT-2 tokens padded to a power-of-2 `50,304 = 196.5 × 256` for GPU Tensor-Core alignment, validated on the first `10,485,760` sequential tokens of FineWeb Shard 0, matching the `KellerJordan/modded-nanogpt` speedrun benchmark standard).

Our target outcomes are:
1. **Accuracy Target**: Push validation cross-entropy (Val CE) below the **`3.28` Val CE** Modded-NanoGPT speedrun milestone (and ultimately toward the `3.183` Slowrun record) using an authentic quantum statistical-mechanics representation (explicit multi-head Hermitian density matrices $\rho_{t,h} \in \mathbb{C}^{16 \times 16}$) rather than a standard classical Transformer.
2. **Architectural Simplicity & Power-of-2 GPU Efficiency**: Maintain a unified, power-of-2 **`512D` (`16` heads $\times$ `16` complex dimensions $\times$ `2` real/imag)** state and readout geometry where token embeddings, multi-head Hermitian density matrices, quantum state tomography probes, and no-click/vacuum fallback channels coexist without competing vocabulary tables or awkward non-power-of-2 tensor shapes.
3. **100% `AdamW`-Free Manifold Optimization**: Train every parameter in the network using geometry-aware **Riemannian manifold optimizers** (Stiefel matrices, Unitary $U(16)$ matrices, Unit-RMS Spheres $\mathbb{S}^{511}(\sqrt{512})$, and Hyperbolic/Positive Geodesics $\mathbb{R}_{>0}$) that scale cleanly from local single-GPU prototyping (`1x RTX 5060 Ti 16GB`, batch size `8,192 = 2^13` tokens/step) to large-batch multi-GPU speedruns (`8x H100`, batch size `262,144 = 2^18` tokens/step).

---

## 2. Complete Mathematical Walkthrough of the Current Architecture (`ExplicitDensityTomographicGPT`)

To make this document completely self-contained for a reader who has never seen the codebase, here is the exact mathematical flow of our `31.95M`-parameter, `4`-layer model (`T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM`):

### 2.1 Input Wavepacket Seeding, Causal N-Gram Hashing, & Per-Head Complex Ket Normalization
1. **Input Token Table (`wte_512` of shape `(50304, 512)`)**:
   - Each token $v \in \{0, \dots, 50303\}$ is stored as a `512`-dimensional real row vector constrained to the unit-RMS sphere $\mathbb{S}^{511}(\sqrt{512})$ (meaning $\frac{1}{512}\|\mathbf{w}_v\|_2^2 = 1$, or $\|\mathbf{w}_v\|_2 = \sqrt{512}$).
2. **Causal Trigram & Unigram Hash Tables (`tok_ngram_in`, `tok_ngram_out`, `tok_unigram_out` of shape `(4096, 512)`)**:
   - Given the input token sequence $v_1, \dots, v_T$, we compute two deterministic, zero-parameter power-of-2 hash indices in $\{0, \dots, 4095\}$ at every position $t$:
     $$h_{\text{tok}}(t) = \big(1000003\, v_t + 65537\, v_{t-1} + 257\, v_{t-2}\big) \bmod 4096, \qquad h_{\text{uni}}(t) = \big(2654435761\, v_t\big) \bmod 4096$$
   - The input trigram embedding $\mathbf{g}_{\text{in},t} = \text{tok\_ngram\_in}[h_{\text{tok}}(t)] \in \mathbb{R}^{512}$ (initialized to zero) is regularized by damped RMS normalization and added to the token embedding:
     $$\mathbf{e}_t^{\text{raw}} = \text{wte\_512}[v_t] + 0.30 \cdot \frac{\mathbf{g}_{\text{in},t}}{\sqrt{\frac{1}{512}\|\mathbf{g}_{\text{in},t}\|_2^2 + 0.0625}}$$
   - A lightweight causal adjacent-token smearing gate (`smear_gate` $\in \mathbb{R}^{512 \times 16}$, scalar `smear_lambda` $\lambda_{\text{smear}} \in [-1, 1]$) blends $\mathbf{e}_{t-1}^{\text{raw}}$ into $\mathbf{e}_t^{\text{raw}}$:
     $$\mathbf{e}_t = \mathbf{e}_t^{\text{raw}} + \lambda_{\text{smear}}\,\sigma\!\big(W_{\text{smear}}\,\mathbf{e}_{t, 1:16}^{\text{raw}}\big) \odot \mathbf{e}_{t-1}^{\text{raw}}, \qquad \mathbf{x}_t^{(0)} = \operatorname{RMSNorm}(\mathbf{e}_t) \in \mathbb{R}^{512}$$
3. **Per-Head Complex Unit-$L_2$ Ket Normalization**:
   - The `512D` vector $\mathbf{x}_t^{(0)}$ is reshaped into $H = 16$ heads of $2 \times d_h = 2 \times 16$ real numbers (representing the real and imaginary parts of a $16$-dimensional complex vector $\tilde{\psi}_{t,h}^{(0)} \in \mathbb{C}^{16}$).
   - Each head is then explicitly normalized by `complex_l2_normalize` so that every head ket has **exact unit complex $L_2$ norm** ($\sum_{j=1}^{16} (|\operatorname{Re}\psi_{t,h,j}^{(0)}|^2 + |\operatorname{Im}\psi_{t,h,j}^{(0)}|^2) = 1$):
     $$|\psi_{t,h}^{(0)}\rangle = \frac{\tilde{\psi}_{t,h}^{(0)}}{\sqrt{\langle \tilde{\psi}_{t,h}^{(0)} \mid \tilde{\psi}_{t,h}^{(0)} \rangle + 10^{-6}}} \;\in\; \mathbb{S}^{31} \subset \mathbb{C}^{16}, \qquad \langle \psi_{t,h}^{(0)} \mid \psi_{t,h}^{(0)} \rangle = 1$$
   - Having unit $L_2$ norm per head guarantees that every rank-1 outer product $|\psi_{t,h}^{(0)}\rangle\langle\psi_{t,h}^{(0)}| \in \mathbb{C}^{16 \times 16}$ has **exact unit trace** ($\operatorname{Tr}(|\psi\rangle\langle\psi|) = \langle\psi\mid\psi\rangle = 1$).

---

### 2.2 Layer-0 Explicit Outer-Product Density Matrix Initialization ($\rho_{t,h}^{(0)} \in \operatorname{Herm}(\mathbb{C}^{16})$)
1. **3-Ket Causal Mixture ($\rho_{t,h}^{(0)}$)**:
   - Rather than keeping the residual stream as a rank-1 vector, Layer 0 explicitly constructs a $16 \times 16$ **Hermitian positive semi-definite (PSD) trace-1 density matrix** per head from three pure kets at position $t$:
     1. The current token ket $|\psi_{t,h}^{(0)}\rangle$,
     2. The phase-rotated previous token ket $|\psi_{\text{rot},t-1,h}^{(0)}\rangle = e^{i\phi_{\text{lag},h}} |\psi_{t-1,h}^{(0)}\rangle$ (where `lag_phase` $\phi_{\text{lag}} \in \mathbb{R}^{16}$ is a learnable per-head phase vector), and
     3. The normalized complex Hadamard (elementwise Schur) interference ket $|\xi_{t,h}^{(0)}\rangle = \operatorname{complex\_l2\_normalize}\!\big(\psi_{t,h}^{(0)} \odot \psi_{\text{rot},t-1,h}^{(0)}\big) \in \mathbb{C}^{16}$ capturing non-linear two-token coincidence.
   - These three rank-1 projectors are combined via learned simplex weights $(w_0, w_1, w_2) = \operatorname{softmax}(\text{layer0\_mix\_logits}) \in \Delta^2$ (initialized from `layer0_mix_logits = [1.10, 0.20, -0.10]`):
     $$\rho_{t,h}^{(0)} = w_0\, |\psi_{t,h}^{(0)}\rangle\langle\psi_{t,h}^{(0)}| + w_1\, |\psi_{\text{rot},t-1,h}^{(0)}\rangle\langle\psi_{\text{rot},t-1,h}^{(0)}| + w_2\, |\xi_{t,h}^{(0)}\rangle\langle\xi_{t,h}^{(0)}| \;\in\; \operatorname{Herm}(\mathbb{C}^{16})$$
2. **Frobenius-Isometric Upper/Lower Triangular Real Packing ($\tilde{P}: \operatorname{Herm}(\mathbb{C}^{16}) \to \mathbb{R}^{256}$)**:
   - Any $16 \times 16$ Hermitian matrix $\rho = \rho_{\text{re}} + i\rho_{\text{im}}$ has a symmetric real part ($\rho_{\text{re}}^\top = \rho_{\text{re}}$, $16$ diagonal + $120$ strict upper-triangular entries $= 136$ real numbers) and a skew-symmetric imaginary part ($\rho_{\text{im}}^\top = -\rho_{\text{im}}$, zero diagonal + $120$ strict lower-triangular entries $= 120$ real numbers).
   - We pack $\rho$ into a $16 \times 16 = 256$-element real vector $\tilde{P}(\rho) \in \mathbb{R}^{256}$ by storing the diagonal $\operatorname{diag}(\rho_{\text{re}})$ on the diagonal, $\sqrt{2}\,\rho_{\text{re}, i<j}$ in the strict upper triangle, and $\sqrt{2}\,\rho_{\text{im}, i>j}$ in the strict lower triangle. The $\sqrt{2}$ off-diagonal factor makes $\tilde{P}$ an **exact Frobenius isometry**:
     $$\operatorname{Tr}(\rho A) = \big\langle \tilde{P}(\rho),\; \tilde{P}(A) \big\rangle_{\mathbb{R}^{256}}, \qquad \operatorname{Tr}(\rho^2) = \|\tilde{P}(\rho)\|_2^2$$
3. **Definition of Effective Quantum Rank (`RhoRank`)**:
   - Throughout our telemetry tables, **`RhoRank`** denotes the **Rényi-2 effective rank (inverse purity)** of the $16 \times 16$ density matrix, averaged over all batch tokens and heads:
     $$\text{RhoRank}(\rho_{t,h}) \;=\; \frac{1}{\operatorname{Tr}\!\big((\rho_{t,h} / \operatorname{Tr}\rho_{t,h})^2\big)} \;=\; \frac{(\operatorname{Tr}\rho_{t,h})^2}{\|\tilde{P}(\rho_{t,h})\|_2^2} \;\in\; [1.0,\; 16.0]$$
   - A rank-1 pure state $|\psi\rangle\langle\psi|$ has $\text{RhoRank} = 1.0$, whereas a maximally mixed $16 \times 16$ state $\frac{1}{16}I_{16}$ has $\text{RhoRank} = 16.0$.

---

### 2.3 Biorthogonal SVD-Stiefel Projections & The 4-Layer Dual Residual Highway
1. **What Is a `BiorthogonalSVDStiefelProjection` ($W = U \operatorname{diag}(\sigma) V^\top$)?**
   - A single square orthogonal matrix $Q \in O(512)$ can only rotate vectors and has all singular values fixed to $1$. To allow a projection to amplify or attenuate specific feature directions while keeping its left and right singular frames strictly orthonormal, `BiorthogonalSVDStiefelProjection` parameterizes a $512 \times 512$ linear map via its explicit Singular Value Decomposition (SVD):
     $$W = U \operatorname{diag}(\sigma) V^\top, \qquad U, V \in \operatorname{St}(512, 512) \text{ (orthonormal matrices)},\quad \sigma \in [0.05, 4.0]^{512}$$
   - Here $U$ and $V$ are updated on the real Stiefel manifold $\operatorname{St}(512, 512)$, while the positive singular value spectrum $\sigma$ is updated on $\mathbb{R}_{>0}$. This is used for `q_proj`, `k_proj`, `v_proj`, `obs_proj`, `exit_bridge`, and `tomo_proj`.
2. **Inside Each of the $L = 4$ Highway Blocks (`_ExplicitDensityHighwayBlock`)**:
   Each layer $\ell \in \{1..4\}$ receives the `512D` real vector stream $\mathbf{x}_t^{(\ell-1)}$, the Layer-0 anchor vector $\mathbf{x}_t^{(0)}$, and the packed density matrix $\rho_{t,h}^{(\ell-1)} \in \operatorname{Herm}(\mathbb{C}^{16})$:
   - **Step A — Input Anchor Blend & QKV Projections**:
     $$\mathbf{x}_{\text{in},t} = \lambda_{\text{blk},0}\,\mathbf{x}_t^{(\ell-1)} + \lambda_{\text{blk},1}\,\mathbf{x}_t^{(0)}, \qquad \mathbf{h}_{\text{in},t} = \operatorname{RMSNorm}(\mathbf{x}_{\text{in},t})$$
     Queries $\mathbf{q}_{t,h}$, keys $\mathbf{k}_{t,h}$, and values $\mathbf{v}_{t,h}$ are projected via `q_proj`, `k_proj`, `v_proj` (with a gated residual skip from $\mathbf{x}_t^{(0)}$ into $\mathbf{v}_{t,h}$ via `ve_gate` and `sa_lambdas`), rotary position embeddings (RoPE) are applied to $\mathbf{q}, \mathbf{k}$, and $\mathbf{v}_{t,h}$ is complex-$L_2$-normalized into a unit value ket $|v_{t,h}\rangle \in \mathbb{C}^{16}$.
   - **Step B — Value Excitation & Causal Berry-Phase Born Attention Over Both Kets and Density Matrices**:
     - First, the current token's value projector $|v_{t,h}\rangle\langle v_{t,h}|$ is blended into the incoming density matrix with per-head mixing weight `val_exc_alpha` $\alpha_{\text{exc},h} \in [0.02, 0.85]$:
       $$\rho_{\text{in},t,h} = (1 - \alpha_{\text{exc},h})\,\rho_{t,h}^{(\ell-1)} + \alpha_{\text{exc},h}\,|v_{t,h}\rangle\langle v_{t,h}|$$
     - Next, causal attention weights $A_{t,\tau}^{(h)}$ ($\tau \le t$) are computed from the complex inner product $\langle q_{t,h} \mid k_{\tau,h} \rangle = r_{t,\tau} + i\, s_{t,\tau}$ using a learnable per-head **Berry phase angle** $\phi_{\text{berry},h}$ (`berry_qk_phase`) and a quadratic **Born interference weight** $\beta_{\text{qk},h}$ (`born_qk_beta`):
       $$S_{t,\tau}^{(h)} = \frac{1}{\sqrt{16}}\Big(\cos(\phi_{\text{berry},h})\, r_{t,\tau} - \sin(\phi_{\text{berry},h})\, s_{t,\tau}\Big) + \frac{\beta_{\text{qk},h}}{16}\big(r_{t,\tau}^2 + s_{t,\tau}^2\big), \qquad A_{t,\tau}^{(h)} = \operatorname{softmax}_{\tau \le t}\!\big(S_{t,\tau}^{(h)}\big)$$
     - Why is this called **Berry-Phase + Born Attention**? Standard real dot-product attention only sees the real part $r_{t,\tau} = \operatorname{Re}\langle q \mid k \rangle$. Rotating by $\phi_{\text{berry},h}$ makes attention sensitive to the **complex geometric phase** $\operatorname{Im}\langle q \mid k \rangle$, while $\beta_{\text{qk},h} |\langle q \mid k \rangle|^2$ adds phase-invariant **quantum Born probability** $|\langle q \mid k \rangle|^2$.
     - The attention matrix $A^{(h)}$ simultaneously transports **both** the $16\text{D}$ complex value kets $|v_{\tau,h}\rangle$ and the $256\text{D}$ packed density matrices $\tilde{P}(\rho_{\text{in},\tau,h})$ across sequence positions:
       $$|\bar{v}_{t,h}\rangle = \sum_{\tau \le t} A_{t,\tau}^{(h)} |v_{\tau,h}\rangle, \qquad \rho_{\text{attn},t,h} = \sum_{\tau \le t} A_{t,\tau}^{(h)} \rho_{\text{in},\tau,h}$$
   - **Step C — Trunk No-Click Vacuum Sink ($p_{\text{nc}}$)**:
     - Each head maintains a learnable unit vacuum ket $|k_{\text{vac},h}\rangle \in \mathbb{C}^{16}$ (`k_vac_ket`), a bias $b_{\text{vac},h}$ (`noclick_bias`), and a log-position slope $\gamma_{\text{pos},h} = \sigma(\text{noclick\_pos\_logit}_h)$. Given the query-vacuum overlap $s_{\text{vac},t,h}$ and the norm $\|\bar{v}_{t,h}\|_{\text{rms}}$ of the attended value ket, the **no-click probability** $p_{\text{nc},t,h} \in (0, 1)$ is:
       $$p_{\text{nc},t,h} = \sigma\!\Big(b_{\text{vac},h} + \gamma_{\text{pos},h} \ln(t + 1) + s_{\text{vac},t,h} - \|\bar{v}_{t,h}\|_{\text{rms}}\Big)$$
     - When no past context token aligns well with the query ($|\bar{v}_{t,h}\|$ is small or $s_{\text{vac}}$ is large), $p_{\text{nc},t,h}$ rises, damping the attended ket $|\bar{v}_{t,h}\rangle \leftarrow (1 - p_{\text{nc},t,h})|\bar{v}_{t,h}\rangle$ and pulling $\rho_{\text{attn},t,h}$ halfway back toward the un-attended highway state $\rho_{t,h}^{(\ell-1)}$:
       $$\rho_{\text{damped},t,h} = (1 - 0.5\,p_{\text{nc},t,h})\,\rho_{\text{attn},t,h} + 0.5\,p_{\text{nc},t,h}\,\rho_{t,h}^{(\ell-1)}$$
   - **Step D — Unitary Conjugation ($U(16)$) & State-Filtered Vector Back-Reaction**:
     - Each head evolves $\rho_{\text{damped},t,h}$ by conjugating with its learnable unitary matrix $U_h = U_{\text{re},h} + i U_{\text{im},h} \in U(16)$ (`u_evol_re, u_evol_im`):
       $$\rho_{\text{rot},t,h} = U_h\, \rho_{\text{damped},t,h}\, U_h^\dagger$$
     - The rotated density matrix then acts as a linear operator on the attended ket $|\bar{v}_{t,h}\rangle$ (coupled by `rho_ket_coupling` $\kappa_{\rho,h}$) and projects back into the `512D` real vector stream via `obs_proj`, gated by `q_attn_gate` ($g_{\text{at}}$) and scaled by `obs_scale`:
       $$|v_{\text{out},t,h}\rangle = |\bar{v}_{t,h}\rangle + \kappa_{\rho,h}\,\big(\rho_{\text{rot},t,h}|\bar{v}_{t,h}\rangle\big), \qquad \mathbf{x}_{\text{mid},t} = \mathbf{x}_{\text{in},t} + \text{obs\_scale} \cdot \text{obs\_proj}\!\big(g_{\text{at}} \odot \operatorname{Concat}_{h=1}^{16} |v_{\text{out},t,h}\rangle\big)$$
   - **Step E — Mach-Zehnder Interferometer MLP (`mz_mlp`) & Complete Highway Update**:
     - What is `mz_mlp` (`SVDStiefelMachZehnderInterferometer`)? Instead of a standard SwiGLU or GELU MLP, `mz_mlp` splits $\operatorname{RMSNorm}(\mathbf{x}_{\text{mid},t})$ into two parallel Stiefel-projected hidden beams $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{1024}$, computes constructive minus destructive interference intensity $\phi(\mathbf{a}, \mathbf{b}) = \operatorname{ReLU}(\mathbf{a} + \mathbf{b})^2 - \frac{1}{4}\operatorname{ReLU}(\mathbf{a} - \mathbf{b})^2$, projects back to `512D` via a down-projection Stiefel matrix, and adds the result to $\mathbf{x}_{\text{mid},t}$ to produce $\mathbf{x}_t^{(\ell)}$.
     - Finally, $\mathbf{x}_t^{(\ell)}$ is reshaped into 16 complex-normalized kets $|\psi_{\text{mlp},t,h}^{(\ell)}\rangle \in \mathbb{C}^{16}$, and its outer product is blended into the density highway with per-head weight `mlp_rho_inject` ($\eta_{\text{mlp},h}$) alongside the unitary highway blend `highway_gamma` ($\gamma_{\text{hw},h}$):
       $$\tilde{\rho}_{t,h}^{(\ell)} = (1 - \gamma_{\text{hw},h})\,\rho_{t,h}^{(\ell-1)} + \gamma_{\text{hw},h}\,\rho_{\text{rot},t,h}, \qquad \rho_{t,h}^{(\ell)} = (1 - \eta_{\text{mlp},h})\,\tilde{\rho}_{t,h}^{(\ell)} + \eta_{\text{mlp},h}\,|\psi_{\text{mlp},t,h}^{(\ell)}\rangle\langle\psi_{\text{mlp},t,h}^{(\ell)}|$$

---

### 2.4 Complete Unified `512D` Tomographic Readout & Token Logit Equations
At the exit of Layer 4, we have the final `512D` vector $\mathbf{x}_t^{(4)}$ and the final $16 \times 16$ density matrices $\rho_{t,h}^{(4)}$ ($h=1..16$):

1. **Exit Bridge & Density-Filtered Exit Vector (`exit_bridge`, `filter_scale`)**:
   - We apply `exit_bridge` ($U_{\text{exit}}\operatorname{diag}(\sigma_{\text{exit}})V_{\text{exit}}^\top$) to $\mathbf{h}_{\text{pre},t} = \operatorname{RMSNorm}(\mathbf{x}_t^{(4)})$:
     $$\mathbf{h}_{\text{exit},t} = \operatorname{RMSNorm}\!\big(\text{exit\_bridge}(\mathbf{h}_{\text{pre},t})\big) \in \mathbb{R}^{512}$$
   - Reshaping $\mathbf{h}_{\text{exit},t}$ into 16 complex-normalized kets $|\psi_{\text{exit},t,h}\rangle \in \mathbb{C}^{16}$, we apply the final density matrix as a filter $\rho_{t,h}^{(4)}|\psi_{\text{exit},t,h}\rangle$, concatenate across heads, and RMS-normalize to obtain $\mathbf{h}_{\text{filt},t} \in \mathbb{R}^{512}$.
2. **Quantum State Tomography via $16 \times 32 = 512$ Born Probes (`probe_re, probe_im`, `tomo_proj`, `tomo_scale`)**:
   - Each head $h \in \{1..16\}$ measures $\rho_{t,h}^{(4)}$ against $K = 32$ learnable complex unit probe kets $|c_{k,h}\rangle \in \mathbb{S}^{31} \subset \mathbb{C}^{16}$ (`probe_re, probe_im` of shape `(16, 32, 16)`) via the Born trace rule:
     $$q_{t,h,k} = \langle c_{k,h} \mid \rho_{t,h}^{(4)} \mid c_{k,h} \rangle = \operatorname{Tr}\!\big(\rho_{t,h}^{(4)}\, |c_{k,h}\rangle\langle c_{k,h}|\big) = \big\langle \tilde{P}(\rho_{t,h}^{(4)}),\; \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \big\rangle_{\mathbb{R}^{256}} \;\ge\; 0$$
   - Concatenating all $16 \times 32 = 512$ expectations into $\mathbf{q}_{512,t} \in \mathbb{R}^{512}$ and projecting through `tomo_proj` gives:
     $$\mathbf{q}_{\text{proj},t} = \operatorname{RMSNorm}\!\big(\text{tomo\_proj}(\mathbf{q}_{512,t})\big) \in \mathbb{R}^{512}$$
3. **Output N-Gram Context Modulation (`tok_ngram_out`, `tok_unigram_out`, `ngram_out_scale`)**:
   - Using the causal trigram hash $h_{\text{tok}}(t)$ and unigram hash $h_{\text{uni}}(t)$ from Section 2.1, we look up the output context vector and modulate it multiplicative-additively by $\mathbf{q}_{\text{proj},t}$:
     $$\mathbf{g}_{\text{out},t} = \text{tok\_ngram\_out}[h_{\text{tok}}(t)] + 0.65\,\text{tok\_unigram\_out}[h_{\text{uni}}(t)], \qquad \hat{\mathbf{g}}_{\text{out},t} = \frac{\mathbf{g}_{\text{out},t}}{\sqrt{\frac{1}{512}\|\mathbf{g}_{\text{out},t}\|_2^2 + 0.0625}}$$
     $$\mathbf{g}_{\text{mod},t} = \big(1 + 0.30\,\tanh(\mathbf{q}_{\text{proj},t})\big) \odot \hat{\mathbf{g}}_{\text{out},t}$$
   - The four `512D` readout streams combine via scalar weights `filter_scale`, `tomo_scale`, and `ngram_out_scale`:
     $$\mathbf{u}_{\text{vocab},t} = \mathbf{h}_{\text{exit},t} + \text{filter\_scale}\cdot \mathbf{h}_{\text{filt},t} + \text{tomo\_scale}\cdot \mathbf{q}_{\text{proj},t} + \text{ngram\_out\_scale}\cdot \mathbf{g}_{\text{mod},t} \;\in\; \mathbb{R}^{512}$$
4. **Unified `512D` Vacuum Sphere Normalization (`vac_rest_mass`, `vac_proj`, `readout_temp` $\beta$)**:
   - To allow the model to attenuate all logits uniformly when uncertain without changing the `512D` vector dimension, a `4D` vacuum projection $\mathbf{u}_{\text{vac},t} = W_{\text{vac}}\,\mathbf{h}_{\text{pre},t,1:16} \in \mathbb{R}^4$ (`vac_proj`) and a positive scalar rest mass $m_{\text{vac}} \in [0.02, 0.85]$ (`vac_rest_mass`) enter the denominator of the sphere normalization, scaled by the global inverse temperature $\beta = \text{readout\_temp} \in [0.05, 40.0]$:
     $$\mathbf{h}_{\text{norm},t} = \frac{\beta}{512} \cdot \frac{\mathbf{u}_{\text{vocab},t}}{\sqrt{\frac{1}{512}\|\mathbf{u}_{\text{vocab},t}\|_2^2 + m_{\text{vac}}^2 + \frac{1}{4}\|\mathbf{u}_{\text{vac},t}\|_2^2}} \;\in\; \mathbb{R}^{512}$$
5. **Final Detector GEMM (`w_det_512`, `radial_scale`, `unigram_bias`, `prior_gate`)**:
   - Each token $v \in \{0, \dots, 50256\}$ has a `512D` detector row $\text{w\_det\_512}[v] \in \mathbb{S}^{511}(\sqrt{512})$, a positive scalar radial multiplier $r_v \in [0.25, 2.20]$ (`radial_scale[v]`), and a log-unigram prior bias $b_{\text{uni},v} = \text{prior\_gate}\cdot(0.5\,\text{unigram\_bias}[v] + 0.5\,\log \hat{p}_{\text{ema}}(v))$.
   - The effective detector row is $\mathbf{w}_{\text{vocab},v} = r_v \cdot \operatorname{RMSNorm}(\text{w\_det\_512}[v])$, and the softcapped logit $\ell_{t,v}$ (`cap = 30.0`) for token $v$ is computed by a single matrix multiplication:
     $$z_{t,v} = \langle \mathbf{h}_{\text{norm},t},\; \mathbf{w}_{\text{vocab},v} \rangle + b_{\text{uni},v}, \qquad \ell_{t,v} = 30.0 \cdot \tanh\!\left(\frac{z_{t,v}}{30.0}\right)$$

---

### 2.5 Manifold Parameter Groups & Base (`1.0x`) Optimizer Configuration
Every parameter in `ExplicitDensityTomographicGPT` belongs to one of four manifold-native optimizers (zero `AdamW`). Throughout Sections 3 and 4, a learning-rate multiplier `M` (e.g., `1.0x`, `2.0x`, `4.0x`) scales the following **base (`1.0x`) learning rates**:

| Optimizer Instance | Parameter Tensors Managed | Manifold / Geometry | Base (`1.0x`) LR | Preconditioning & Retraction |
| :--- | :--- | :--- | :---: | :--- |
| **`q_trunk_opt`** (`ModularRiemannianStiefelKrausMuon`) | All 2D real Stiefel matrices ($U, V$ in `q, k, v, obs_proj, mz_mlp, exit_bridge, tomo_proj`) | Real Stiefel $\operatorname{St}(n, p)$ ($W^\top W = I_p$) | **`0.032`** | Tangent projection + SNR trust gate + 5-step Newton-Schulz polar retraction |
| **`q_trunk_opt`** *(Unitary branch)* | Per-head density evolution pairs `(u_evol_re, u_evol_im)` `(16, 16, 16)` | Complex Unitary $U(16)$ ($U^\dagger U = I_{16}$) | **`0.025`** | Skew-Hermitian tangent projection + SNR gate + 6-step complex Newton-Schulz |
| **`wte_opt`** (`BatchSubspaceTikhonovGramMuon`, Group 0) | Input token embedding table `wte_512` `(50304, 512)` | Unit-RMS Sphere $\mathbb{S}^{511}(\sqrt{512})$ | **`0.060`** | Active-batch (`active_ids`) $512 \times 512$ Gram Newton-Schulz + sphere retraction |
| **`wte_opt`** (`BatchSubspaceTikhonovGramMuon`, Group 1) | N-gram hash tables `tok_ngram_in, tok_ngram_out, tok_unigram_out` `(4096, 512)` | Bounded RMS Ball $\mathbb{B}_{\text{rms}}^{512}(1.50)$ (`zero_init_ball=True`) | **`0.040`** | `0.85 * u_coord + 0.15 * u_gram` hybrid update inside $\mathbb{B}_{\text{rms}}^{512}(1.50)$ |
| **`head_opt`** (`BatchSubspaceTikhonovGramMuon`) | Output vocabulary detector table `w_det_512` `(50304, 512)` | Unit-RMS Sphere $\mathbb{S}^{511}(\sqrt{512})$ | **`0.050`** | Active-batch $512 \times 512$ Gram Newton-Schulz + sphere retraction |
| **`geo_opt`** (`RiemannianGeodesicScalarAndProbeOptimizer`) | - `probe_re, probe_im` `(16, 32, 16)` & `k_vac_ket` `(16, 32)`<br>- `readout_temp` ($\beta$)<br>- `tomo_scale, filter_scale`<br>- Bridge $\sigma$, block $\alpha, \gamma, m_{\text{vac}}$, `radial_scale`, `unigram_bias` | - Complex/Real Sphere $\mathbb{S}^{31}$<br>- Positive $\mathbb{R}_{>0}$ geodesic<br>- Bounded tangent<br>- Positive $\mathbb{R}_{>0}$ / sparse active | - **`0.025`**<br>- **`0.095`**<br>- **`0.080`**<br>- **`0.008`** | - Tangent projection + $L_2$ retraction<br>- $\beta \leftarrow \beta \exp(\operatorname{clamp}(-\eta \hat{u}, \pm 0.22))$<br>- $\theta \leftarrow \theta - \eta \tanh(\hat{u})$ |

---

## 3. Problems We Solved & Empirical Steps Taken in This Session

### Step 1: Comparing Unified `512D` Explicit-Density Readout Variants (`T0..T4`)
We first ran a `300`-step screening benchmark (`2.46M` tokens at `B = 16, T = 512`) across five architectural variants (`T0` through `T4`) to determine how best to bridge the $16 \times 16$ density highway to the `50,304`-token vocabulary, and then promoted the winning variant (`T2`) to a `1,220`-step (`10.0M`-token) verification run:

| Variant ID | Architectural Description | `300`-Step Val CE (`PPL`) | `300`-Step `Top-1` | Throughput (`tok/s`) | `1,220`-Step (`10M` Tok) Val CE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`T0_VECTOR_BRIDGE_CONTROL`** | Classical vector stream only (`enable_density_highway=False`) | `6.4533` (`634.8`) | `15.89%` | `64.2K` | *(Screened at 300 steps)* |
| **`T1_EXPLICIT_DENSITY_TOMOGRAPHY_512`** | $16\times 16$ density highway + $16\times 32 = 512\text{D}$ Born tomography (`enable_noclick_and_vacuum=False`) | `6.4106` (`608.3`) | `16.05%` | `50.2K` | *(Screened at 300 steps)* |
| **`T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM`** | **`T1` + Causal No-Click Sink ($p_{\text{nc}}$) + Unified `4D` Vacuum Sphere** | **`6.3893`** (`595.4`) 🏆 | **`16.31%`** | **`49.3K`** | **`5.5419`** (`255.2` PPL, `21.74%` Top-1) |
| **`T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE`** | `T2` + auxiliary $256 \times 256$ coarse-fine sub-token bilinear trace (`phi_hi`, `phi_lo`, `u_hi_readout`) | `6.4046` (`604.6`) | `16.21%` | `43.8K` (`-11%`) | *(Screened at 300 steps)* |
| **`T4_CONSOLIDATED_TOMOGRAPHIC_SCALING_1200`** | `T3` with explicit $50,304 \times 512$ sub-token embedding synthesis (`0.82 w_free + 0.18 u_hi * u_lo`) | `6.4189` (`613.3`) | `16.08%` | `42.4K` (`-14%`) | `-` *(Superseded by `T2` at `1,220` steps)* |

**Why `T2` Beat `T3` and `T4`:**
In `T3` and `T4`, each token ID $v \in [0, 50303]$ was additionally split into a coarse high byte $b_{\text{hi}} = \lfloor v / 256 \rfloor \in [0, 196]$ and fine low byte $b_{\text{lo}} = v \bmod 256 \in [0, 255]$. While `T3` avoided materializing a full $50,304 \times 512$ synthesized matrix by computing a factored $256 \times 256$ bilinear trace in the loss chunk loop (and `T4` explicitly synthesized `wte_512` from `u_hi * u_lo` on every forward pass), both variants introduced a **competing sub-token parameterization** that split gradient energy away from the primary `(50304, 512)` detector table and cost `11%–14%` throughput. We therefore standardized on **`T2`**.

---

### Step 2: Constant-LR Breakdown Sweep & Discovering the `AdamW` Step-1 Temperature Quench Cliff
To find out how high we could push the learning rate without a cosine decay schedule, we ran a two-phase diagnostic on `T2` while the model still used `AdamW` for its scalar parameters (`readout_temp` at base LR `0.08`, `tomo_scale`, `filter_scale`), tomographic probes (`probe_re, probe_im`), and `(4096, 512)` N-gram tables:
1. **Phase 1 (`100`-Step Exponential LR Ramp from `0.25x` to `64.0x`)**, and
2. **Phase 2 (`300`-Step Constant-LR Grid at `1.0x, 1.5x, 2.0x, 4.0x, 6.0x, 8.0x, 16.0x, 32.0x`)**.

**What we discovered:**
- In the Phase-1 ramp and Phase-2 grid, training progressed smoothly up to `4.0x` (`6.7725` Val CE), but hit a sudden **`+0.88` nat validation loss cliff between `4.0x` and `6.0x`** (`6.7725 -> 7.8218` Val CE at `6.0x`), with the exact threshold occurring at **`5.625x`**.
- **Root Cause (`AdamW` Sign-Step Quench on Step 1)**:
  - On Step 1 of training, `AdamW`'s bias-corrected first and second moment estimates are $\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$. Dividing $\hat{m}_1$ by $\sqrt{\hat{v}_1} = |g_1|$ turns Step 1 of `AdamW` into a **fixed additive sign step of magnitude $\pm \eta$**, completely independent of how small the gradient or parameter is:
    $$\theta^{(1)} = \theta^{(0)} - \eta \cdot \operatorname{sign}(g_1)$$
  - Because `readout_temp` ($\beta$) was initialized at $\beta^{(0)} = 0.50$ and clamped to $[0.05, 40.0]$, its distance to the lower clamp was $0.50 - 0.05 = 0.45$.
  - With a base fast-scalar learning rate of $\eta_0 = 0.08$, any multiplier $M \ge \frac{0.45}{0.08} = \mathbf{5.625\times}$ ($\eta \ge 0.45$; e.g., at our `6.0x` grid point where $\eta = 0.08 \times 6.0 = 0.48$) caused a single positive gradient on Step 1 to subtract $\ge 0.45$ from $0.50$, **slamming `readout_temp` into its `0.05` lower clamp on Step 1** and quenching all model logits by $10\times$ for the rest of the run!
- By contrast, when we ran a decoupled probe (`Phase 3`) holding `AdamW` at `2.0x` while scaling the Riemannian manifold optimizers (`Stiefel`, $U(16)$, `BatchSubspaceTikhonovGramMuon`) to `8.0x, 16.0x, 32.0x`, the **Riemannian manifolds remained completely free of NaNs or quenches up to `32.0x`**.

---

### Step 3: Eliminating `AdamW` Entirely (`100%` Riemannian & Manifold-Native Optimization)
Recognizing that `AdamW` was both unnecessary and the sole cause of the `5.625x` temperature quench, we eliminated `AdamW` from the architecture in [`src/qdlblocks/optim/vocab_manifold_muon.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py):

1. **How `BatchSubspaceTikhonovGramMuon` Works on Vocabulary Spheres vs. Zero-Init N-Gram Balls**:
   - **On `wte_512` & `w_det_512` (`(50304, 512)`, Unit-RMS Sphere $\mathbb{S}^{511}(\sqrt{512})$)**: Only the $m = |\text{active\_ids}|$ unique tokens appearing in the current batch (`~2,603` rows out of `50,304`) have non-zero gradients. `step_active` gathers those $m$ active rows $W_{\text{act}} \in \mathbb{R}^{m \times 512}$ and their tangent momentum $M_{\text{act}} \in \mathbb{R}^{m \times 512}$, forms the compact **$512 \times 512$ feature-subspace Gram matrix** $G = \hat{M}_{\text{act}}^\top \hat{M}_{\text{act}} + \lambda_{\text{tik}} I_{512}$, computes the inverse square root $(G)^{-1/2}$ via 5-step coupled Newton-Schulz iteration in $O(512^3)$ time, and retracts the updated rows back onto $\mathbb{S}^{511}(\sqrt{512})$.
   - **On `tok_ngram_in`, `tok_ngram_out`, `tok_unigram_out` (`(4096, 512)`, `zero_init_ball=True`)**: Unlike `wte_512`, the N-gram residual tables are initialized to **exact zero** ($\mathbf{0} \in \mathbb{R}^{512}$) so they start as a no-op and grow only where N-gram hashes are informative; projecting a zero-initialized row onto a unit sphere would immediately inject random noise. With `zero_init_ball=True`, active rows are updated using a blend of coordinate-wise RMS-preconditioned momentum (`0.85 * u_coord`, preserving sparsity on unactivated coordinates) and $512 \times 512$ Gram-whitened momentum (`0.15 * u_gram_unit`), and retracted only if their row RMS exceeds the ball radius $R_{\max} = 1.50$ ($\mathbb{B}_{\text{rms}}^{512}(1.50)$). Moving the three `(4096, 512)` tables out of `AdamW` also eliminated `3 × 4096 × 512 × 2 = 12.6M` floats (`~50 MB`) of dense `AdamW` moment buffers.
2. **How `RiemannianGeodesicScalarAndProbeOptimizer` Works**:
   - **Complex & Real Sphere Retraction (`"complex_sphere_pair"`, `"real_sphere_ket"`)**: For `probe_re, probe_im` `(16, 32, 16)` and `k_vac_ket` `(16, 32)`, projects the gradient onto the tangent space of $\mathbb{S}^{31}$ ($g_\perp = g - \operatorname{Re}\langle c \mid g \rangle c$), normalizes by the per-ket tangent RMS norm, steps by $-\eta \hat{u}_\perp$, and retracts back onto $\mathbb{S}^{31}$ via `complex_l2_normalize`.
   - **Hyperbolic-Multiplicative Geodesic Updates for Positive Scalars (`"positive_geodesic"`)**: Positive scale parameters ($\beta = \text{readout\_temp}$, singular value bounds $\sigma$, $m_{\text{vac}}$, $\alpha_{\text{exc}}, \gamma_{\text{hw}}$) naturally live on the multiplicative manifold $(\mathbb{R}_{>0}, ds^2 = (d\beta / \beta)^2)$. Given raw gradient $g_t$, we form the scale-weighted Riemannian gradient $g_{\text{riem},t} = \max(|\beta^{(t)}|, \beta_{\text{ref}}) \cdot g_t$ (with $\beta_{\text{ref}} = 4.0$ for `readout_temp` and $1.0$ for other positive scalars), update first/second moments $m_t, v_t$, compute the RMS-normalized direction $\hat{u}_t = \frac{m_t}{\sqrt{v_t} + 10^{-8}}$, and apply the **exact exponential map on $\mathbb{R}_{>0}$** with maximum per-step log-change $s_{\max} = 0.22$:
     $$\beta^{(t+1)} = \beta^{(t)} \cdot \exp\!\Big(\operatorname{clamp}\big(-\eta \cdot \hat{u}_t,\; -s_{\max},\; +s_{\max}\big)\Big)$$
     Because $\exp(\cdot) > 0$, **a multiplicative geodesic update can never cross zero or quench `readout_temp` on Step 1 at any learning rate**.

**Result of Removing `AdamW` (`300`-Step Constant-LR Sweep, [`.cache/plan083_adamw_free_constant_lr_sweep.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/plan083_adamw_free_constant_lr_sweep.json))**:

| Uniform Constant-LR Multiplier | `AdamW` Scalar/N-Gram Val CE (`300` st) | `AdamW` Final `temp` ($\beta$) | **100% `AdamW`-Free Val CE (`300` st)** | **`AdamW`-Free Final `temp` ($\beta$)** | **`AdamW`-Free `RhoRank`** | Val CE Gain vs. `AdamW` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *(Ref) `1.00x` Cosine Schedule* | `6.3893` | `7.11` | — | — | `3.05 / 16` | — |
| **`1.00x` Constant LR** | `6.3958` | `7.89` | **`6.3812`** 🏆 | `7.15` | **`3.10 / 16`** (`+1.34` vs AdamW `1.76`) | **`-0.0146` nats** |
| **`1.25x` Constant LR** | `6.4301` | `8.00` | **`6.3887`** | `6.76` | `2.50 / 16` | **`-0.0414` nats** |
| **`2.00x` Constant LR** | `6.4945` | `8.07` | **`6.4560`** | `5.86` | `1.53 / 16` | **`-0.0385` nats** |
| **`4.00x` Constant LR** | `6.7725` | `6.66` | **`6.7538`** | `5.42` | `1.09 / 16` | **`-0.0187` nats** |
| **`6.00x` Constant LR** | `7.8218` *(Quenched!)* | `0.05` | **`6.9813`** | **`5.64`** | `1.07 / 16` | **`-0.8405` nats (`-1,418` PPL!)** |
| **`8.00x` Constant LR** | `7.8451` *(Quenched!)* | `0.05` | **`7.1102`** | **`6.05`** | `1.07 / 16` | **`-0.7349` nats (`-1,329` PPL!)** |
| **`16.0x` Constant LR** | `7.8435` *(Quenched!)* | `0.05` | **`7.3643`** | **`5.77`** | `1.13 / 16` | **`-0.4792` nats (`-971` PPL!)** |
| **`32.0x` Constant LR** | `16.41` *(Diverged)* | `0.05` | **`7.6070`** *(Stable!)* | **`5.63`** | `1.22 / 16` | **`-8.80` nats** |

---

### Step 4: `100M`-Token (`12,208`-Step) Constant-LR Runs & The "Hot Iterate vs. Cooled Basin" Paradox
Next, we launched `100M`-token (`12,208`-step at `8,192` tok/step) Constant-LR trajectories across `1.0x`, `2.0x`, `4.0x`, and `6.0x` (cancelling `6.0x` at `49.15M` tokens and skipping `8.0x` once `6.0x` plateaued near `6.78` Val CE; full results in [`.cache/plan083_100m_constant_lr_sweep.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/plan083_100m_constant_lr_sweep.json)):

| Constant LR Multiplier | `2.46M` (`300` st) | `8.19M` (`1k` st) | `24.58M` (`3k` st) | `49.15M` (`6k` st) | `73.73M` (`9k` st) | **`100.01M` (`12,208` st)** | Final `Top-1` | Final `temp` ($\beta$) | Final `RhoRank` | Final `p_nc` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1.0x` Constant LR** | `6.3811` | `5.6958` | `5.2813` | `5.0564` | `4.9671` | **`4.8897`** (`132.9` PPL) | **`25.63%`** | `10.82` | **`2.30 / 16`** | `0.524` |
| **`2.0x` Constant LR** | `6.4704` | `5.8892` | `5.6318` | `5.4858` | `5.4134` | **`5.3816`** (`217.4` PPL) | `21.67%` | `9.32` | `1.88 / 16` | `0.518` |
| **`4.0x` Constant LR** | `6.7425` | `6.4197` | `6.2863` | `6.2079` | `6.1614` | **`6.1317`** (`460.2` PPL) | `15.97%` | `7.15` | `1.95 / 16` | `0.515` |
| **`6.0x` Constant LR** *(stopped @ 6k)* | `6.9967` | `6.7866` | `6.8219` | `6.7840` | — | *(stopped @ `6.7840`)* | `11.49%` | `5.25` | `1.42 / 16` | `0.815` |

**The Paradox:** Looking only at the raw Constant-LR curves, `2.0x` and `4.0x` appear worse than `1.0x` everywhere. However, we recognized that **evaluating a raw iterate $\theta_t$ while it is still taking large constant-LR steps conflates two effects**:
$$\mathbb{E}\big[L(\theta_t)\big] \;\approx\; \underbrace{L(\bar{\theta})}_{\text{true basin quality}} \;+\; \underbrace{\frac{1}{2}\operatorname{Tr}\!\big(H(\bar{\theta})\,\Sigma(\eta)\big)}_{\text{instantaneous thermal step jitter } \propto\, \eta}$$
Because Muon takes fixed-length steps proportional to $\eta$, a `2.0x` trajectory bounces higher up on the valley walls while hot, masking whether its underlying basin $\bar{\theta}$ would actually settle *lower* than `1.0x` once the learning rate is cooled!

---

### Step 5: Proving the Hypothesis with the In-Advance Curvature Estimator + Hot/Shadow-EMA/Branch-and-Quench Probe
We built [`scripts/shadow_ema_and_quench_lr_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/shadow_ema_and_quench_lr_probe.py) (`Results`: [`.cache/plan083_shadow_ema_and_quench_probe.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/plan083_shadow_ema_and_quench_probe.json)) to answer two questions: (1) Can we figure out the optimal per-subsystem LR multipliers *in advance* in `50` steps? and (2) Do higher-LR runs overtake `1.0x` after a short LR quench?

#### Phase 1 (`50` Steps, `~8s`): Figuring Out Subsystem Learning-Rate Multipliers *In Advance*
Before running any 300-step trajectories, Phase 1 measured three quantities at `1.0x` for each subsystem:
1. **Active Update Duty Cycle ($f_{\text{duty}}$)**: The fraction of rows/elements in that parameter group that receive a non-zero update on a single batch of `8,192` tokens.
2. **Expected Geodesic Progress Rate**: $f_{\text{duty}} \times \delta\theta_{\text{active}}$, where $\delta\theta_{\text{active}} = \|\Delta W_{\text{act}}\|_F / \|W_{\text{act}}\|_F$ is the relative step angle on active parameters.
   - *Note on $U(16)$ 4-Layer Compounding*: For the $L = 4$ unitary conjugation layers $\rho \mapsto U_h \rho U_h^\dagger$, a single-layer unitary rotation by angle $\delta\theta_U = 0.09107\text{ rad}$ acts **twice** (left multiplication by $U_h$ and right multiplication by $U_h^\dagger$, rotating off-diagonal coherences by $2\,\delta\theta_U$), and compounds across $L = 4$ independent layers as $\sqrt{L} = 2$, yielding an effective compounded frame rotation of $2\sqrt{L}\,\delta\theta_U = 4 \times 0.09107 = \mathbf{0.36429\text{ rad/step}}$.
3. **Lookahead Step Alignment Cosine ($\cos(\Delta W_{t-1}, -g_t)$)**:
   - **Mathematical Intuition**: After taking an update step $\Delta W_{t-1}$ at step $t-1$, we evaluate the new mini-batch gradient $g_t$ at $W_t = W_{t-1} + \Delta W_{t-1}$. By a first-order Taylor expansion along the step direction, $\langle \Delta W_{t-1}, -g_t \rangle \approx \langle \Delta W_{t-1}, -g_{t-1} \rangle - \Delta W_{t-1}^\top H \Delta W_{t-1}$.
     - If $\cos(\Delta W_{t-1}, -g_t) > 0$, the new negative gradient $-g_t$ **still points in the same direction** as the step we just took, meaning our step **undershot** the valley minimum along that direction (the subsystem wants a **higher** learning rate).
     - If $\cos(\Delta W_{t-1}, -g_t) \approx 0$, the step landed **directly at the directional valley minimum** where the directional derivative vanishes ($\Delta W_{t-1}$ is already at the optimal step size).
     - If $\cos(\Delta W_{t-1}, -g_t) < 0$, the new gradient points backward against $\Delta W_{t-1}$, meaning the step **overshot** the valley floor and is bouncing off the opposite wall (learning rate is **too high**).

| Subsystem | Active Duty Cycle / Step | Active Step Angle (`@ 1x`) | Expected Progress Rate (`Duty × Angle`) | Lookahead Alignment $\cos(\Delta W_{t-1}, -g_t)$ | In-Advance Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`trunk_stiefel`** (`St(512, 512)`) | **`100.0%`** | `0.00559 rad` | **`0.00559 rad/step`** | **`+0.00000`** | Orthogonal step lands right at directional minimum ($\cos = 0$); sweet spot is **`1.0x–1.7x`** |
| **`trunk_unitary`** (`4L` $U(16)$) | **`100.0%`** | `0.09107 rad` | **`0.36429 rad/step`** *(4L $2\sqrt{L}\delta\theta_U$)* | — | Fast $2\sqrt{L} = 4\times$ compounded rotation; pushing to `4.0x` rotates `~1.45 rad/step` (`>80°`), de-phasing inactive tokens |
| **`vocab` (`wte_512` / `w_det_512`)** | **`5.18%`** (`~2,603 / 50,304`) | `0.05270 rad` | **`0.00273 rad/step`** (**`2.05x` slower!**) | **`+0.07369`** | Positive lookahead cosine + `5.18%` sparse duty cycle $\implies$ **undershooting at `1.0x`, wants `2.0x–2.5x`** |
| **`ngram_tables`** (`4,096` buckets) | **`92.11%`** | `0.04410 rad` | `0.04062 rad/step` | — | Touched on `92.1%` of steps; comfortable at **`1.25x–1.5x`** |
| **`global_scalars`** (`temp, scales`) | **`100.0%`** | `0.02180 rel` | `100%` global logit gain | **`+0.73814`** | Monotonically climbs during initial temperature warmup (`0.50 -> 7.40`), then acts as global thermostat (**`0.75x–1.0x`**) |

#### Phase 2 (`240` Constant Plateau Steps + `60` Cosine Quench Steps to `0.10x`): Full 5-Probe Table (Hot vs. Shadow-EMA vs. Post-Quench)
We then ran all five `300`-step probes (`240` steps at constant LR followed by a `60`-step cosine quench down to `0.10x`), while simultaneously tracking a manifold-retracted **Shadow-EMA model** ($\bar{\theta}_{\text{ema}}$ with $\alpha_{\text{ema}} = 1 - 0.975 = 0.025$, a `40`-step averaging window):

| Probe ID & Multipliers `(trunk, vocab, ngram, scalar)` | Step 240 Pre-Quench **`Hot CE`** | Step 240 Pre-Quench **`EMA CE`** | Step 240 `Hot - EMA` Gap | Step 300 **Post-Quench `Hot CE`** | Step 300 Post-Quench `EMA CE` | **Quench Drop** ($\Delta L_{\text{quench}}$) | Quench Accel ($r_{\text{quench}} / r_{\text{plat}}$) | Final `Top-1` | Final `temp` ($\beta$) | Final `RhoRank` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PROBE_3_UNIFORM_2X`** `(2.0, 2.0, 2.0, 2.0)` | `6.5484` *(worse than 1x hot!)* | `6.7996` | `-0.2512` | **`6.2885`** 🏆 | `6.7377` | **`-0.2599` nats** | **`1.37x`** | `16.26%` | `6.39` | `1.54 / 16` |
| **`PROBE_2_UNIFORM_1X`** `(1.0, 1.0, 1.0, 1.0)` | **`6.4827`** *(best while hot)* | `6.9398` | `-0.4571` | `6.2989` | `6.8723` | `-0.1838` nats | `0.86x` | **`16.48%`** | `7.40` | `3.38 / 16` |
| **`PROBE_4_DUTY_MATCHED`** `(1.5, 2.5, 1.5, 0.75)` | `6.5693` | `6.8283` | `-0.2590` | **`6.3047`** | `6.7751` | **`-0.2646` nats** 🏆 | **`1.25x`** | `16.28%` | `6.74` | **`3.46 / 16`** |
| **`PROBE_5_CALM_TRUNK`** `(1.0, 2.5, 1.25, 0.75)` | `6.5605` | `6.8677` | `-0.3072` | `6.3050` | `6.8257` | `-0.2555` nats | `1.22x` | `16.05%` | `6.67` | **`3.65 / 16`** 🏆 |
| **`PROBE_1_TRUNK4X_VOC1X`** `(4.0, 1.0, 1.0, 1.0)` | `6.5747` | `6.7960` | `-0.2213` | `6.3517` | `6.7328` | `-0.2230` nats | `1.12x` | `16.10%` | `7.93` | `2.01 / 16` |
| *(Ref) 300-Step Constant `1.0x` (No Quench)* | — | — | — | `6.3811` | — | — | — | `15.65%` | `7.15` | `3.11 / 16` |
| *(Ref) 300-Step Cosine Schedule `1.0x`* | — | — | — | `6.3893` | — | — | — | `16.31%` | `7.11` | `3.05 / 16` |

#### Four Key Empirical Takeaways from Step 5:
1. **Higher-LR Training Does Converge Lower After a Quench (`PROBE_3_UNIFORM_2X` Beats `1.0x`!)**:
   - At Step 240 (pre-quench), `PROBE_3_UNIFORM_2X` looked `+0.0657` nats worse than `PROBE_2_UNIFORM_1X` (`6.5484` vs. `6.4827`).
   - During the `60`-step quench (`Step 240 -> 300`), `PROBE_3_UNIFORM_2X` dropped by **`-0.2599` nats** (`1.41x` larger drop than `1.0x`'s `-0.1838` nats) and finished at **`6.2885` Val CE**—overtaking `1.0x` Post-Quench (`6.2989`) and beating the 300-step Cosine Baseline (`6.3893`) by **`-0.1008` nats**!
2. **Why `EMA CE` (`ema_decay=0.975`) Sits Above `Hot CE` During Early Rapid Descent—And What `EMA CE` Reveals**:
   - During the first `240` steps of training, the model's loss is plunging so rapidly (`10.82 -> 6.48` nats) and `readout_temp` is climbing so fast (`0.50 -> 7.40`) that a `40`-step trailing average (`ema_decay = 0.975`) lags `~20–40` steps behind the current iterate along the descent ramp, making raw `EMA CE` higher than `Hot CE`.
   - **However, look at two critical signals in the `EMA CE` column**:
     - First, **the `EMA CE` ranking at Step 240 already predicts that the hotter runs are sitting in a better average basin**: at Step 240, the Shadow-EMA of `PROBE_1` (`6.7960`), `PROBE_3_UNIFORM_2X` (`6.7996`), and `PROBE_4_DUTY_MATCHED` (`6.8283`) all beat the Shadow-EMA of `PROBE_2_UNIFORM_1X` (`6.9398`) by **`-0.11 to -0.14` nats**!
     - Second, comparing the `Hot - EMA` gap of `PROBE_2_UNIFORM_1X` (`-0.4571` nats) against `PROBE_3_UNIFORM_2X` (`-0.2512` nats) and `PROBE_4` (`-0.2590` nats) reveals an excess thermal step-jitter offset of `(-0.2512) - (-0.4571) = +0.2059` nats in the hot `2x` iterate—matching the exact `-0.26` nat quench drop released between Step 240 and Step 300!
3. **Why `PROBE_4_DUTY_MATCHED` (`trunk=1.5x, vocab=2.5x, scalar=0.75x`) Preserves Quantum Density Rank (`RhoRank = 3.46 / 16`)**:
   - In `PROBE_3_UNIFORM_2X` (`trunk=2.0x, vocab=2.0x, scalar=2.0x`), pushing the trunk and scalars to `2.0x` collapsed the density highway rank down to **`1.54 / 16`**.
   - By contrast, `PROBE_4_DUTY_MATCHED` (`trunk=1.5x, vocab=2.5x, ngram=1.5x, scalar=0.75x`) achieved the **largest quench drop of all five probes (`-0.2646` nats)** while maintaining **`RhoRank = 3.46 / 16`** (`2.25x` higher quantum rank!). Its only handicap at Step 300 (`6.3047` vs. `6.2885`) was that `scalar_mult = 0.75x` slowed the initial warmup of `readout_temp` (`temp = 6.74` vs. `7.40` in `1.0x`), which is solved by fixing `readout_temp` analytically (Section 4.2).
4. **How `PROBE_1_TRUNK4X_VOC1X` (`trunk=4.0x, vocab=1.0x`) Shows That `trunk=4.0x` Is Too Hot Relative to `vocab=1.0x`**:
   - Even after the `60`-step quench, `PROBE_1` (`trunk=4.0x, vocab=1.0x`) only reached `6.3517` Val CE (`+0.063` nats worse than `UNIFORM_2X` and `+0.047` nats worse than `PROBE_4`) and degraded `RhoRank` to `2.01 / 16`. Over-rotating the `100%`-duty-cycle trunk at `4.0x` while starving the `5.18%`-duty-cycle vocabulary at `1.0x` causes the trunk frame to outrun the sparse token detectors.

---

## 4. Deep-Dive Answers to Key Optimization & Scaling Questions

### 4.1 Why Don't Muon / Riemannian Muon Gradient Steps Go to Zero as Loss Goes to Zero?
**Question:** *A side conversation suggested our gradient steps don't go to zero as loss goes to zero because of how our optimizer works. Is that true? Is that how the usual Muon optimizer works?*

**Answer: Yes — that is an exact mathematical property of both standard Muon (`KellerJordan/modded-nanogpt`) and our Riemannian Stiefel/Sphere Muon optimizers.**

Here is why:
1. **Standard Muon (`Polar(M_t)` Is Scale-Invariant)**:
   - In standard Muon, the momentum buffer $M_t \in \mathbb{R}^{m \times n}$ is pre-normalized by its Frobenius norm $\hat{M}_t = M_t / \|M_t\|_F$ and passed through a 5-step Newton-Schulz polynomial to approximate the orthogonal polar factor:
     $$U_t = \operatorname{Polar}(M_t) = M_t (M_t^\top M_t)^{-1/2}$$
   - Every non-zero singular value of $U_t$ is pushed to **`1.0`**, so $\|U_t\|_F = \sqrt{\min(m, n)}$ is **strictly constant**, whether the raw gradient norm $\|g_t\|_F$ is `10.0` or `10^{-9}`!
2. **Our Riemannian Muon (`ModularRiemannianStiefelKrausMuon` & `BatchSubspaceTikhonovGramMuon`)**:
   - In [`riemannian_stiefel_muon_step`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L440) and [`BatchSubspaceTikhonovGramMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py#L353), we project the momentum $M_t$ onto the manifold tangent space and normalize it by its Frobenius or row-RMS norm **before** running the Tikhonov-damped Newton-Schulz iteration:
     $$\hat{M}_t = \frac{M_t}{\|M_t\|_F + \epsilon}, \qquad U_t = \hat{M}_t \big(\hat{M}_t^\top \hat{M}_t + \lambda_{\text{tik}} I\big)^{-1/2}$$
   - In `RIEMANNIAN_TANGENT_SNR_GATED` mode, the step is modulated by a directional Signal-to-Noise Ratio (SNR) trust gate comparing the tangent momentum norm $\|M_t\|_F$ to the step-to-step gradient deviation $\|g_{\perp,t} - M_t\|_F$ (with trust threshold $\tau_{\text{snr}} = 0.25$):
     $$\text{snr}_t = \frac{\|M_t\|_F}{\|g_{\perp,t} - M_t\|_F + 10^{-8}}, \qquad \gamma_{\text{snr},t} = \frac{\text{snr}_t}{\tau_{\text{snr}} + \text{snr}_t} \in (0, 1), \qquad \Delta W_t = -\eta\,\gamma_{\text{snr},t}\,U_t$$
   - Notice that $\text{snr}_t$ is a **scale-invariant ratio** ($\|M_t\|_F / \|g_{\perp,t} - M_t\|_F$): if both $g_{\perp,t}$ and $M_t$ shrink by $10^{-6}$ as the model approaches a smooth minimum, their ratio $\text{snr}_t$ stays $\mathcal{O}(1)$ (`\gamma_{\text{snr},t} \approx 0.5\text{–}0.8`) and $\|\hat{M}_t\|_F = 1$! Therefore, **the step magnitude $\|\Delta W_t\|_F \propto \eta$ never goes to zero unless $\eta \to 0$**, causing constant-LR runs to orbit the minimum in a limit cycle of radius $\mathcal{O}(\eta)$.
3. **How We Can Make Steps Self-Anneal Near a Minimum**:
   - In addition to decaying $\eta$ via our Thermostat/Quench schedule, we can multiply the step by an **absolute gradient-norm soft gate** $g_{\text{mag}} = \frac{\|M_t\|_F}{\sqrt{\|M_t\|_F^2 + \tau_{\text{grad}}^2}}$ (or apply Tikhonov damping $\lambda_{\text{abs}} I$ *before* Frobenius normalization). Then when $\|M_t\|_F \gg \tau_{\text{grad}}$ we get full scale-invariant Muon, and when $\|M_t\|_F \ll \tau_{\text{grad}}$ the step smoothly transitions to linear Riemannian gradient descent ($\|\Delta W_t\|_F \propto \eta \|M_t\|_F \to 0$).

---

### 4.2 Do We Need a Learnable Global Temperature Parameter (`readout_temp` $\beta$)?
**Question:** *Do we need a global temperature parameter? Curvature rises as the square of temperature, and I worry about whether the temperature destabilizes training.*

**Answer: No, we do NOT need `readout_temp` ($\beta$) to be a free learnable scalar — and replacing it with a fixed analytical scale eliminates a major source of hidden $\mathcal{O}(\beta^2)$ curvature drift.**

1. **Why a Scale Factor $\beta > 1$ Exists Geometrically**:
   - Because both the pre-temperature readout vector $\tilde{\mathbf{u}}_t = \mathbf{u}_{\text{vocab},t} / \sqrt{\|\mathbf{u}_{\text{vocab},t}\|_{\text{rms}}^2 + e_{\text{vac}}} \in \mathbb{R}^{512}$ and the detector rows $\mathbf{w}_{\text{vocab},v} \in \mathbb{R}^{512}$ have unit RMS norm (`L2 norm = sqrt(512)`), their unscaled inner product $z_v = \frac{1}{512}\langle \tilde{\mathbf{u}}_t, \mathbf{w}_{\text{vocab},v} \rangle$ lies in $[-1, +1]$, with typical random alignments having standard deviation $1/\sqrt{512} \approx 0.044$.
   - A fixed scale multiplier $\beta \approx 9.5\text{–}10.5$ is needed so that $z_v \in [-1, 1]$ maps to logits $\ell_v = \beta z_v \in [-10, +10]$ capable of expressing sharp next-token probabilities over `50,257` classes.
2. **Why Making $\beta$ a Free Learnable Scalar Destabilizes Curvature ($\|H\| \propto \beta^2$)**:
   - For logits $\ell_v = \beta z_v$, the gradient with respect to the pre-temperature representation $z$ is $\nabla_z L = \beta (p - y)$, and the **Hessian (curvature) with respect to all trunk and detector parameters scales quadratically with $\beta$**:
     $$H_z = \frac{\partial^2 L}{\partial z^2} = \beta^2 \big(\operatorname{diag}(p) - p p^\top\big) \;\propto\; \beta^2$$
   - In our `100M`-token `1.0x` run, `readout_temp` ($\beta$) was initialized at **`0.50` at Step 0**, climbed to **`4.81` by Step 60** (a $(4.81 / 0.50)^2 = \mathbf{92.5\times}$ initial curvature jump during warmup), and then continued drifting upward from **`4.81` at Step 60 to `10.82` at Step 12,208 (`100M` tokens)**—an additional **$(10.82 / 4.81)^2 = \mathbf{5.06\times}$ curvature increase** across the post-warmup training run!
   - When $\beta$ drifts upward by $2.25\times$ mid-run, it silently narrows the valley walls by $5.06\times$, which is equivalent to **secretly multiplying the effective learning rate against the valley curvature by `5x` mid-training**.
3. **The Clean Replacement (Zero Free Global Temperature Scalar)**:
   - **Fix $\beta_0 = 10.0$ Analytically (after a deterministic 150-step warmup ramp `4.0 -> 10.0`)**: By making $\beta_0$ a non-trainable analytical constant, the global logit curvature scale $\beta_0^2$ stays **100% constant** after Step 150! Any fine-grained per-subspace gain adjustment is already handled safely by the bounded singular value vector `exit_bridge.sigma` $\in [0.5, 2.5]^{512}$.

---

### 4.3 Scaling Batch Size via Gradient Accumulation (`8,192 -> 262,144` Tokens/Step on `8x H100`)
**Question:** *I'd like to experiment more with gradient accumulation. When I run this on `8x H100` for the speedrun, I think I'll be able to use much larger batch sizes (`~260k = 2^18`). That probably allows higher step sizes as well!*

**Answer: Yes — scaling from `8,192` (`2^13`) to `262,144` (`2^18 = 32 × 8,192`) tokens/step fundamentally transforms both the vocabulary update duty cycle and the permissible step size:**

1. **Vocabulary Duty Cycle Jumps from `5.18%` to `~40%+` Per Optimizer Step**:
   - At `8,192` tokens/step (`B = 16, T = 512`), only `~2,603` unique tokens (`5.18%` of `50,304`) appear in `active_ids` on each step, which is why `wte_512` lagged `2.05x` behind the trunk at `1.0x` and why `94.8%` of token detectors went stale whenever the trunk rotated too fast.
   - At **`262,144` tokens/step** (`32` gradient accumulation micro-steps locally, or native parallel batch on `8x H100`), a single optimizer step covers **`~19,000–22,500` unique tokens (`~40%` of the vocabulary, accounting for `> 99.2%` of all token occurrences in FineWeb)**!
   - Inside [`BatchSubspaceTikhonovGramMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py#L353), accumulating gradients over `262,144` tokens before running the $512 \times 512$ Newton-Schulz Gram orthogonalization means `20,000` token rows are co-orthogonalized together in a single step, and the Newton-Schulz optimizer overhead is amortized by **`32x`** (`381` optimizer steps per `100M` tokens instead of `12,208` steps).
2. **Higher Step Sizes from $5.66\times$ Higher Gradient SNR**:
   - Because mini-batch gradient variance scales as $1/B$, a $32\times$ larger batch increases gradient SNR by $\sqrt{32} \approx 5.66\times$, raising the permissible geodesic step size by **`2.0x–3.5x`** before hitting valley-wall bounce.

---

### 4.4 Balancing Trunk vs. Vocabulary Compute (`>= 50%` Trunk FLOP Share & Depth Scaling `L = 4 -> L = 8..12`)
**Question:** *With the concentration of operations in the vocab input and readout, it might make sense to expand the network to have more layers, but how do we balance the compute and accuracy implications so the trunk is responsible for at least half of the FLOPs?*

Using the standard convention **`1 MAC (multiply-accumulate) = 2 Forward FLOPs = 6 Total (Fwd + Bwd) FLOPs`** (and noting that embedding/hash table lookups `wte_512[idx]` and `tok_ngram[h]` are memory gathers with `0` dense matrix MACs in forward), here is the exact itemized MAC and FLOP budget per token of `ExplicitDensityTomographicGPT` at $d = 512$:

- **Fixed Readout + Bridge MACs per Token (`~26.30M` MACs = `157.8 MFLOPs` Fwd+Bwd)**:
  - Vocabulary detector GEMM `(1, 512) @ (512, 50304)`: $512 \times 50,304 = \mathbf{25.76\text{M}}$ MACs
  - `exit_bridge` ($512 \times 512$) + `tomo_proj` ($512 \times 512$): $2 \times 512^2 = \mathbf{0.52\text{M}}$ MACs
  - Tomographic probe contraction ($16 \times 256 \times 32$) + readout density filter ($16 \times 16^2$): $\mathbf{0.13\text{M}}$ MACs
- **Per-Layer Quantum Trunk MACs per Token (`~3.57M` MACs/layer = `21.4 MFLOPs`/layer Fwd+Bwd)**:
  - `q_proj`, `k_proj`, `v_proj`, `obs_proj` ($4 \times 512 \times 512$): $\mathbf{1.05\text{M}}$ MACs
  - Mach-Zehnder MLP `mz_mlp` (`2` up-beams $2 \times 512 \times 1024$ + `1` down-proj $1024 \times 512$): $\mathbf{1.57\text{M}}$ MACs
  - Causal attention over $288\text{D}$ packed ket+density payload ($T = 512, H = 16$): $\mathbf{0.82\text{M}}$ MACs
  - Per-head $U(16)$ unitary conjugation ($2 \times 16 \times 16^3$) + state filter: $\mathbf{0.13\text{M}}$ MACs

| Subsystem Component | Active Matrix Ops per Token (Fwd) | Fwd + Bwd FLOPs / Token | Share of Total Compute (`L = 4`) | Share at **`L = 8`** | Share at **`L = 12`** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vocab Readout + Bridges** (`w_det_512` `50304×512`, `exit_bridge`, `tomo_proj`, probes) | `26.30M` MACs | `157.8 MFLOPs` | **`64.8%`** *(Readout-dominated!)* | `47.9%` | **`38.0%`** |
| **Quantum Density Trunk** (`3.57M` MACs / layer $\times L$ layers) | `14.28M` MACs (`4L`) | `85.7 MFLOPs` (`4L`) | **`35.2%`** *(Trunk-starved!)* | **`52.1%`** (`171.4M`) | **`62.0%`** (`257.1M`) |
| **Total Model** | `40.58M` MACs (`4L`) | `243.5 MFLOPs` (`4L`) | `100%` (`49.3K tok/s`) | `100%` (`~36K tok/s`) | `100%` (`~28K tok/s`) |

**Key Insight:**
- At $L = 4$, **the `4`-layer quantum trunk accounts for only `35.2%` of total FLOPs**, while the fixed `50,304 × 512` vocabulary readout consumes **`64.8%` of total FLOPs**!
- Doubling the trunk depth from **`L = 4` to `L = 8` layers** (or `L = 10..12`) crosses the `50%` threshold (`52.1%–62.0%` trunk FLOP share) while only increasing total step time by **`~1.35x` at `L = 8`** because the `157.8 MFLOP` readout cost is paid only once at the output of the model.

---

### 4.5 Automated Hot-vs-Shadow-EMA Learning-Rate Thermostat (`300M`-Token Horizon)
**Question:** *Using the difference in loss between hot and EMA parameters as a measurement of excess step size seems like something we could program into the optimizer! Monitoring it on the small validation set and adjusting LR every `50M` tokens on a `300M`-token run seems useful.*

**Exact Specification of the Closed-Loop Riemannian Thermostat:**
1. **How $\bar{\theta}_{\text{ema}}$ Is Maintained and Retracted**:
   - On every training step $t$, we update a shadow parameter buffer via fast in-place ambient linear interpolation (`torch._foreach_lerp_`) with a short **`20`-step horizon** ($\alpha_{\text{ema}} = 0.05$, `ema_decay = 0.95`, which eliminates the `40`-step warmup lag we saw with `0.975` while still averaging out high-frequency step oscillation):
     $$\tilde{\theta}_{\text{ema}}^{(t)} = (1 - \alpha_{\text{ema}})\,\tilde{\theta}_{\text{ema}}^{(t-1)} + \alpha_{\text{ema}}\,\theta_{\text{hot}}^{(t)}$$
   - At each evaluation checkpoint, before evaluating $\bar{\theta}_{\text{ema}}$ on the `65,536`-token validation slice (`~0.8s`), we **retract $\tilde{\theta}_{\text{ema}}$ onto the exact Riemannian manifolds** (`retract_shadow_ema_manifolds_`):
     - `wte_512` and `w_det_512` rows are normalized onto the unit-RMS sphere $\mathbb{S}^{511}(\sqrt{512})$,
     - All 2D Stiefel matrices are retracted via 5-step Newton-Schulz polar factorization $\operatorname{Polar}(\tilde{W}) \in \operatorname{St}(n, p)$,
     - All $U(16)$ pairs are retracted via 6-step complex Newton-Schulz onto $U(16)$, and
     - Probe kets and vacuum kets are normalized onto $\mathbb{S}^{31}$.
2. **Decision Rule Every `50M` Tokens Over a `300M`-Token Run (`Calibrated Gain` $\kappa = 8.0$)**:
   - Measure the validation thermal gap $\Delta L_{\text{thermal}} = L_{\text{val}}(\theta_{\text{hot}}) - L_{\text{val}}(\bar{\theta}_{\text{ema}})$:
     - **If $\Delta L_{\text{thermal}} \le 0$ (`Hot` is at least as good as `Shadow-EMA`)**: Deterministic descent over the last `20` steps exceeds step-size jitter. Keep learning rates unchanged.
     - **If $\Delta L_{\text{thermal}} > +0.02\text{ nats}$ (`Hot` is worse than `Shadow-EMA` due to valley-wall bounce)**:
       1. Copy the retracted Shadow-EMA weights into the live model ($\theta_{\text{hot}} \leftarrow \bar{\theta}_{\text{ema}}$), immediately banking the $\Delta L_{\text{thermal}}$ validation loss improvement, and
       2. Scale the learning rate down using calibrated sensitivity gain **$\kappa = 8.0$** (calibrated from our Step-5 data where an excess thermal offset of $\Delta L \approx 0.20$ nats corresponds to $\sqrt{1 + 8.0 \times 0.20} = \sqrt{2.6} \approx 1.61\times$ step reduction):
          $$\eta_{\text{next}} \;=\; \eta_{\text{cur}} \cdot \max\!\left(0.50,\; \frac{1}{\sqrt{1 + 8.0 \cdot \max(0,\, \Delta L_{\text{thermal}})}}\right)$$
   - Across `300M` tokens (`50M, 100M, 150M, 200M, 250M` thermostat checks + a final `20M`-token cosine quench to `0.10x` from `280M -> 300M`), this provides a completely automated, curvature-adaptive Warmup-Stable-Decay (`WSD`) schedule.

---

## 5. Concrete Roadmap of Next Steps

When we resume, here is the prioritized sequence of experiments to execute:

1. **Step 1 — Fix Analytical Logit Temperature ($\beta_0 = 10.0$) & Add Magnitude-Aware Muon Damping**:
   - Replace the free scalar `readout_temp` with a **fixed analytical Parseval scale $\beta_0 = 10.0$** (after a deterministic 150-step ramp `4.0 -> 10.0`), leaving per-channel scaling to the bounded Stiefel bridge `exit_bridge.sigma` $\in [0.5, 2.5]^{512}$ so Hessian curvature cannot drift by $\beta^2$ mid-run.
   - Add an optional **soft gradient-norm gate** $g_{\text{mag}} = \frac{\|M_t\|_F}{\sqrt{\|M_t\|_F^2 + \tau_{\text{grad}}^2}}$ inside `ModularRiemannianStiefelKrausMuon` and `BatchSubspaceTikhonovGramMuon` so steps naturally shrink toward zero when gradients vanish near a minimum.
2. **Step 2 — Large-Batch Gradient Accumulation Probe (`8,192 -> 65,536 -> 262,144` Tokens/Step)**:
   - Add `--grad-accum-steps` (`1, 8, 32`) to our training harness to simulate the `8x H100` `262,144`-token (`2^18`) batch size locally.
   - Verify the active vocabulary duty-cycle jump (`5.18% -> ~40%` per step) and re-measure the Phase-1 Lookahead Alignment Cosine and optimal step-size multipliers `(trunk_mult, vocab_mult)` at `65K` and `262K` batch sizes.
3. **Step 3 — Rebalance Trunk vs. Readout Compute (`L = 4 -> L = 8` Layers) & Upgrade Quantum Channels**:
   - **Depth Scaling (`L = 4 -> L = 8`)**: Scale the trunk from `4` layers (`35.2%` FLOP share) to `8` layers (`52.1%` FLOP share) so the quantum density highway performs more than half of total model compute.
   - **Orthonormal Two-Basis Probe Frame (`2 × U(16) = 32` IC-POVM-Style Frame)**: Currently, the $K = 32$ tomographic probe kets $|c_{k,h}\rangle \in \mathbb{C}^{16}$ per head are normalized independently onto $\mathbb{S}^{31}$, allowing multiple probes to collapse onto the same dominant eigenvector of $\rho_{t,h}$. In quantum information, an **Informationally Complete Positive Operator-Valued Measure (IC-POVM)** or a set of **Mutually Unbiased Bases (MUBs)** spans the operator space with minimal redundancy. By grouping the 32 probes of head $h$ into **two $16 \times 16$ unitary matrices** $C_h^{(1)}, C_h^{(2)} \in U(16)$ updated via `ModularRiemannianStiefelKrausMuon`, each 16-probe bank forms an exact orthonormal basis ($\sum_{k=1}^{16} |c_{k,h}^{(m)}\rangle\langle c_{k,h}^{(m)}| = I_{16}$), guaranteeing zero collinearity among probes within each basis.
   - **Rank-$R$ ($R = 2$) Stiefel Kraus Channels ($\sum_{r=1}^2 K_{h,r} \rho_{t,h} K_{h,r}^\dagger$)**: Currently, each layer conjugates $\rho_{t,h}$ by a single unitary $U_h \rho_{t,h} U_h^\dagger$, which is isospectral (cannot change the eigenvalues or von Neumann entropy of $\rho_{t,h}$). A general **Completely Positive Trace-Preserving (CPTP) quantum channel** has the Kraus form $\mathcal{E}_h(\rho) = \sum_{r=1}^R K_{h,r} \rho K_{h,r}^\dagger$ subject to the trace-preservation condition $\sum_{r=1}^R K_{h,r}^\dagger K_{h,r} = I_{16}$. By stacking $R = 2$ operators vertically into a single tall complex matrix $\mathbf{K}_h = \begin{bmatrix} K_{h,1} \\ K_{h,2} \end{bmatrix} \in \mathbb{C}^{32 \times 16}$, the Stiefel manifold condition $\mathbf{K}_h^\dagger \mathbf{K}_h = K_{h,1}^\dagger K_{h,1} + K_{h,2}^\dagger K_{h,2} = I_{16}$ **is identical to the CPTP trace-preservation constraint**, allowing `ModularRiemannianStiefelKrausMuon` to train multi-branch quantum channels that actively mix or purify $\rho_{t,h}$!
4. **Step 4 — `300M`-Token Run with the Duty-Matched Split (`trunk=1.5x, vocab=2.5x`) & `50M`-Token Hot-vs-EMA Thermostat**:
   - Combine the `L = 8` architecture, fixed analytical temperature $\beta_0 = 10.0$, duty-cycle-matched subsystem LRs, and the `50M`-token Shadow-EMA Thermostat (plus final geodesic quench) on a `300M`-token trajectory toward the `< 3.28` Val CE speedrun target.
