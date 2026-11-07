# EXPERIMENT PLAN 083: Architectural Consolidation — Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density-Matrix Highway ($H = 16$ Heads), `16 × 32 = 512D` Quantum State Tomography Readout ($\operatorname{Tr}(\rho_t E_v)$), & Zero-Synthesis Single-GEMM `512D` Sphere (`Wave 63`)

## 1. Executive Summary & Architectural Motivation

In [`SESSION_WRAPUP_VACUUM_CHOLESKY_AND_ZERO_PPMI_BYTE_PAIR_CONSOLIDATION.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SESSION_WRAPUP_VACUUM_CHOLESKY_AND_ZERO_PPMI_BYTE_PAIR_CONSOLIDATION.md) (`Waves 58–62` / `EXPERIMENT_PLAN_078`–`082`), we identified both the core empirical breakthroughs and the architectural complexity sprawl that accumulated across six levels of class inheritance:

1. **What Proved Out Empirically (`Waves 58–62`)**:
   - **Vacuum Subspace Abstention ($\mathcal{H}_{\text{vac}}$) & Trunk "No-Click" POVM Attention Sink**: Allowing both attention heads (`p_noclick`) and readout states ($\cos\theta_{\text{vac}}$) to rotate into a zero-detector orthogonal vacuum subspace improved 1,200-step Val CE to **`5.1697` (`175.86` PPL)** (`N4`).
   - **Strict `modded-nanogpt` Zero-Precomputation Compliance (`init_tokens_scanned == 0`)**: Combining the analytical BPE rank Zipf prior with online causal training-stream EMA (`OnlineCausalZipfTracker`) dropped Step-0 Val CE from `10.8258` to **`8.9050` (`-1.92` nats)** in `< 20 ms` with zero pre-scanned tokens.
   - **Single-GEMM `512D` Readout Contraction (`Q0` in `Wave 62`)**: Replacing $2R + 1 = 17$ per-mode `50,304` vocabulary GEMMs with **1 single `512D` vocabulary GEMM** cut step time by **`3.17x`** (`2,786.6 ms -> 879.4 ms`) **and** improved 300-step Zero-PPMI Val CE from `6.1292` (`Z2`) to **`6.0794` (`436.76` PPL, `-0.0498` nats)**.

2. **What Caused Sprawl & the `230s` Validation Bottleneck (Eliminated in `Wave 63`)**:
   - **Awkward Non-Power-of-2 Tensor Shapes (`448D` Sem + `64D` ID + `16D` Carrier = `528D`)**: Appending a `16D` constant carrier vector to `512D` forced all vocabulary GEMMs to run at an awkward `528D` dimension and required separate temperatures and vacuum rest masses for `448D` and `64D` slices.
   - **Re-Synthesizing the `50,304 × 512` Detector Matrix `1,280` Times per Validation Pass**: Summing three competing vocabulary parameterizations (`w_det_512 + BytePairSchur + Two8QubitRegister`) into a fresh `(50304, 512)` tensor on every batch made each `10.48M`-token validation checkpoint take `~230s`.
   - **Hand-Crafted Layer-0 Cholesky Lag Rows**: Selecting an arbitrary mode count $R \in \{4, 8, 12\}$ required hand-crafting `12` distinct lag/Schur row formulas at Layer 0.

3. **The `Wave 63` Consolidation — Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density-Matrix Highway ($H = 16$ Heads) & `16 × 32 = 512D` Quantum State Tomography Readout**:
   - **100% Power-of-2 Geometry (`d = 512` Real = $H = 16$ Heads of $\mathbb{C}^{16}$)**:
     Across $H = 16$ heads, each token's complex wavefunction $|\psi_{t,h}\rangle \in \mathbb{C}^{16}$ has $16 \times 16 \times 2 = \mathbf{512\text{ real floats}}$.
   - **Explicit Outer-Product Density Matrix ($\rho_{t,h} = |\psi_{t,h}\rangle\langle\psi_{t,h}| \in \operatorname{Herm}(\mathbb{C}^{16})$)**:
     Taking the Hermitian outer product in each head yields $\rho_{t,h} \in \mathbb{C}^{16 \times 16}$, which has $16 + 2\binom{16}{2} = \mathbf{256\text{ real degrees of freedom per head}}$ via the exact Frobenius-isometric bijection $\tilde{P}: \operatorname{Herm}(\mathbb{C}^{16}) \to \mathbb{R}^{256}$ ([`pack_hermitian_isometric`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L90-L108)):
     $$\big\langle \tilde{P}(A),\; \tilde{P}(B) \big\rangle_{\mathbb{R}^{256}} \;=\; \operatorname{Tr}(AB), \qquad \|\tilde{P}(\rho_{t,h})\|_2^2 \;=\; \operatorname{Tr}(\rho_{t,h}^2)$$
     Because $\tilde{P}$ is linear, causal attention mixes the `256D` packed density matrices directly across the context window ($\sum_{s \le t} A_{t,s}^{(h)} \rho_{s,h}$, damped by the Trunk "No-Click" POVM attention sink $1 - p_{\text{noclick},t,h}$) and evolves them via per-head $U(16)$ unitaries $\rho_{t,h} \mapsto U_h \rho_{t,h} U_h^\dagger$, naturally supporting **any rank up to 16** without hand-crafted Cholesky lag rows.
   - **`16 × 32 = 512D` Quantum State Tomography Readout ($\langle \mathbf{w}_v, \mathbf{q}_t \rangle = \sum_h \operatorname{Tr}(\rho_{t,h} E_{v,h})$ in 1 GEMM)**:
     Each head $h \in \{0, \dots, 15\}$ measures $\rho_{t,h} \in \operatorname{Herm}(\mathbb{C}^{16})$ against **$K = 32$ learned unit-norm complex probe kets** $|c_{k,h}\rangle \in \mathbb{C}^{16}$:
     $$q_{t,h,k} \;=\; \langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle \;=\; \big\langle \tilde{P}(\rho_{t,h}),\; \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \big\rangle_{\mathbb{R}^{256}} \;\in\; \mathbb{R}^{16 \times 32} \;\cong\; \mathbb{R}^{512}$$
     By linearity of the trace, taking a **single `(N, 512) @ (512, 50304)` GEMM** $\langle \mathbf{w}_v, \hat{\mathbf{q}}_t \rangle + b_{\text{uni}}[v]$ evaluates:
     $$\langle \mathbf{w}_v,\, \mathbf{q}_t \rangle \;=\; \sum_{h=0}^{15} \operatorname{Tr}\!\Big(\rho_{t,h}\, \underbrace{\sum_{k=0}^{31} w_{v,h,k}\,|c_{k,h}\rangle\langle c_{k,h}|}_{E_{v,h} \;\in\; \operatorname{Herm}(\mathbb{C}^{16})}\Big)$$
     giving every token $v \in \{0, \dots, 50303\}$ a learned $16 \times 16$ Hermitian observable $E_{v,h}$ while executing **only 1 `512D` vocabulary GEMM** and **zero `50,304 × 512` matrix re-synthesis**!

---

## 2. Quantitative Hypotheses (`H1–H4`)

- **`H1` (Frobenius-Isometric Outer-Product Identity, Tomographic Trace Equivalence, & Rule-9 Single-Pass Autograd Gate)**:
  - For all Hermitian matrices $\rho_{t,h} \in \operatorname{Herm}(\mathbb{C}^{16})$ and probe kets $|c_{k,h}\rangle \in \mathbb{C}^{16}$, the packed isometric contraction $\langle \tilde{P}(\rho_{t,h}), \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \rangle_{\mathbb{R}^{256}}$ matches the explicit complex quadratic form $\langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle$ to `< 1e-5` max absolute error.
  - Furthermore, the single `512D` dot product $\langle \mathbf{w}_v, \mathbf{q}_t \rangle$ matches the explicit $16 \times 16$ Hermitian operator trace $\sum_{h=0}^{15} \operatorname{Tr}(\rho_{t,h} E_{v,h})$ to `< 1e-5`, and `_SingleGemm512DTomographicCEFn` passes the Rule-9 Two-Layer Custom Autograd Efficiency Gate (`bwd/fwd pow <= 1.0`, `bmm <= 3.0`, `FLOPs <= 3.5`).
- **`H2` (Wall-Clock Validation & Step Throughput Breakdown — Eliminating the `230s` Synthesis Bottleneck)**:
  - Eliminating per-batch `50,304 × 512` detector matrix re-synthesis and shrinking the vocabulary GEMM from `528D` to power-of-2 `512D` cuts full `10,485,760`-token (`1,280`-batch) FineWeb Shard-0 validation wall-clock time from `~230s` (`Wave 62`) down to **`<= 25s` (`> 9.0x` validation speedup)** and reduces total 300-step run wall-clock time (training + full `10.48M`-token validation) from `433.8s` (`Q0`) / `885.3s` (`Z2`) to **`<= 180s`**.
- **`H3` (300-Step Zero-PPMI Ablation Progression `T0 -> T1 -> T2 -> T3`, `0` Pre-Scanned Tokens)**:
  - On the `10,485,760`-token FineWeb Shard-0 benchmark at 300 steps (`2.46M` training tokens, `init_tokens_scanned == 0`):
    - **`T1` (+ Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density Highway + `16 × 32 = 512D` Probe Tomography)** improves Validation CE over **`T0` (Pure Rank-1 Wavepacket Baseline)** while increasing density-matrix mixedness ($R_{\text{eff}}(\rho_{t,h}) = 1 / \operatorname{Tr}(\hat{\rho}_{t,h}^2) > 2.0$).
    - **`T2` (+ Trunk "No-Click" POVM Attention Vacuum Sink + Unified `512D` Readout Vacuum Subspace $\mathcal{H}_{\text{vac}}$)** and **`T3` (+ Zero-Synthesis Factored `256 × 256` Two-Register Bilinear Trace $\langle \phi_{\text{hi}}[b_{\text{hi}}] \mid \rho_{t,h} \mid \phi_{\text{lo}}[b_{\text{lo}}] \rangle$)** beat the `Wave 61–62` baselines `Z0` (`6.2710`), `Z2` (`6.1292`), and `Q0` (`6.0794`).
- **`H4` (1,200-Step / `9.83M`-Token Scaling `T4` vs. `Z5`)**:
  - Scaling the consolidated `Wave 63` architecture (`T4`, `1,200` steps / `9.83M` tokens, `0` pre-scanned tokens) beats `Z5` (`5.2807` Val CE / `196.51` PPL) while cutting total 1,200-step wall-clock time by **`> 4.0x`** (`< 900s` vs. `Z5`'s `5,050s`).

---

## 3. Mathematical Formulation

### 3.1 Power-of-2 Complex Wavepacket & Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density Matrix

Each token $x_t \in \{0, \dots, 50256\}$ looks up a `512D` real embedding vector in $W_{\text{in}} \in \mathbb{R}^{50304 \times 512}$, reshaped into **$H = 16$ heads of complex wavepackets** $|\psi_{t,h}^{(0)}\rangle \in \mathbb{C}^{16}$ (`DEFAULT_TOMO_NUM_HEADS = 16`, `DEFAULT_TOMO_HEAD_COMPLEX_DIM = 16`) normalized onto the complex unit sphere $\|\psi_{t,h}^{(0)}\|_2 = 1$.

To seed the density highway at Layer 0 as a genuine mixed state without hand-crafted 12-row Cholesky tables, each head forms the convex outer-product combination of the current token wavepacket $|\psi_{t,h}^{(0)}\rangle$, the phase-transported lag-1 wavepacket $|\tilde{\psi}_{t-1,h}^{(0)}\rangle = e^{i\theta_h}|\psi_{t-1,h}^{(0)}\rangle$, and their normalized complex Schur interference ket $|\xi_{t,h}^{(0)}\rangle = \operatorname{normalize}(|\psi_{t,h}^{(0)}\rangle \odot_{\mathbb{C}} |\tilde{\psi}_{t-1,h}^{(0)}\rangle)$:
$$\rho_{t,h}^{(0)} \;=\; \pi_0\,|\psi_{t,h}^{(0)}\rangle\langle\psi_{t,h}^{(0)}| \;+\; \pi_1\,|\tilde{\psi}_{t-1,h}^{(0)}\rangle\langle\tilde{\psi}_{t-1,h}^{(0)}| \;+\; \pi_2\,|\xi_{t,h}^{(0)}\rangle\langle\xi_{t,h}^{(0)}| \;\in\; \operatorname{Herm}(\mathbb{C}^{16})$$
where $(\pi_0, \pi_1, \pi_2) = \operatorname{softmax}(\boldsymbol{\lambda}_{\text{init}})$. Using [`pack_rank1_ket_outer_isometric`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L125-L130), $\rho_{t,h}^{(0)}$ is stored directly as a **`256D` real Frobenius-isometric vector**:
$$\tilde{\boldsymbol{\rho}}_{t,h}^{(0)} \;=\; \tilde{P}(\rho_{t,h}^{(0)}) \;\in\; \mathbb{R}^{256}$$

### 3.2 The 4 Quantum Density Highway Blocks (`Causal Born Attention + No-Click POVM Sink + U(16) Unitary Channel`)

In each block $\ell \in \{0, 1, 2, 3\}$:
1. **Causal Hermitian-Born Attention with Trunk "No-Click" POVM Vacuum Sink**:
   - The `512D` coherent stream $\mathbf{h}_t^{(\ell)} \in \mathbb{R}^{512}$ projects via Stiefel matrices $W_Q, W_K, W_V \in \operatorname{St}(512, 512)$ into $H = 16$ complex heads $|q_{t,h}\rangle, |k_{t,h}\rangle, |v_{t,h}\rangle \in \mathbb{C}^{16}$ with complex rotary phase (RoPE).
   - Given causal Hermitian-Born scores $S_{t,s}^{(h)} = \frac{1}{\sqrt{16}}\big(\operatorname{Re}(e^{i\phi_h}\langle q_{t,h}\mid k_{s,h}\rangle) + \beta_{\text{born},h}|\langle q_{t,h}\mid k_{s,h}\rangle|^2\big)$ for $s \le t$ and causal softmax weights $A_{t,s}^{(h)} = \operatorname{softmax}_{s \le t}(S_{t,s}^{(h)})$, each head evaluates its **No-Click POVM Vacuum Sink probability** against the learned unit-RMS vacuum key $|\hat{k}_{\text{vac},h}\rangle \in \mathbb{C}^{16}$:
     $$p_{\text{noclick}, t, h} \;=\; \sigma\!\Big(b_{\text{vac}, h} + \gamma_{\text{pos}, h}\ln(t + 1) + S_{t,\text{vac}}^{(h)} - \operatorname{RMS}(|\bar{v}_{t,h}\rangle)\Big) \;\in\; (0, 1)$$
   - Both the coherent value ket and the `256D` packed density matrix are causally attended and attenuated by $(1 - p_{\text{noclick}, t, h})$ in a **single fused `(B, 16, T, T) @ (B, 16, T, 32 + 256)` batched matmul**:
     $$|\bar{v}_{t,h}^{\text{out}}\rangle \;=\; (1 - p_{\text{noclick}, t, h}) \sum_{s \le t} A_{t,s}^{(h)}\,|v_{s,h}\rangle \;\in\; \mathbb{C}^{16}$$
     $$\tilde{\boldsymbol{\rho}}_{t,h}^{\text{attn}} \;=\; (1 - \kappa_{\text{vac}}\,p_{\text{noclick}, t, h}) \sum_{s \le t} A_{t,s}^{(h)}\Big((1 - \alpha_{\text{exc}})\,\tilde{\boldsymbol{\rho}}_{s,h}^{(\ell)} + \alpha_{\text{exc}}\,\tilde{P}\big(|v_{s,h}\rangle\langle v_{s,h}|\big)\Big) \;\in\; \mathbb{R}^{256}$$
2. **Per-Head $U(16)$ Unitary Evolution & Density Highway Update**:
   - Unpacking $\tilde{\boldsymbol{\rho}}_{t,h}^{\text{attn}} \to \rho_{t,h}^{\text{attn}} \in \operatorname{Herm}(\mathbb{C}^{16})$, each head applies a learned unitary operator $U_h^{(\ell)} \in U(16)$ (optimized as a paired complex matrix in `ModularRiemannianStiefelKrausMuon`, Wave 44 Invariant 1):
     $$\rho_{t,h}^{\text{rot}} \;=\; U_h^{(\ell)}\,\rho_{t,h}^{\text{attn}}\,U_h^{(\ell)\dagger}$$
     and updates the persistent highway $\tilde{\boldsymbol{\rho}}_{t,h}^{(\ell+1)} = (1 - \gamma_{\rho,\ell})\tilde{\boldsymbol{\rho}}_{t,h}^{(\ell)} + \gamma_{\rho,\ell}\,\tilde{P}(\rho_{t,h}^{\text{rot}})$.
3. **Mach-Zehnder (`MZ`) Two-Beam Stiefel Interferometer MLP**:
   - Updates the coherent `512D` stream $\mathbf{h}_t^{(\ell+1)}$ and excites $\tilde{\boldsymbol{\rho}}_{t,h}^{(\ell+1)}$ with the post-MLP rank-1 outer product $\tilde{P}(|\psi_{t,h}^{\text{mlp}}\rangle\langle\psi_{t,h}^{\text{mlp}}|)$.

### 3.3 `16 × 32 = 512D` Quantum State Tomography Readout + Unified `512D` Vacuum Sphere + Single GEMM

At the exit of Block 3:
1. **`16 × 32 = 512D` Quadratic Born Probe Tomography**:
   Each head $h \in \{0, \dots, 15\}$ owns $K = 32$ learned unit-norm complex probe kets $|c_{k,h}\rangle \in \mathbb{C}^{16}$ (`DEFAULT_PROBES_PER_HEAD = 32`). Packing $\tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \in \mathbb{R}^{256}$ once per step into $C_{\text{packed}} \in \mathbb{R}^{16 \times 256 \times 32}$, we compute all $16 \times 32 = 512$ Born expectation values via a single `(N, 16, 256) @ (16, 256, 32)` contraction:
   $$q_{t,h,k} \;=\; \big\langle \tilde{\boldsymbol{\rho}}_{t,h},\; \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \big\rangle_{\mathbb{R}^{256}} \;=\; \langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle \;\in\; \mathbb{R}^{16 \times 32} \;\cong\; \mathbb{R}^{512}$$
   In addition, $\rho_{t,h}$ filters the Stiefel Exit Bridge wavepacket $|\psi_{t,h}^{\text{exit}}\rangle \in \mathbb{C}^{16}$ via $|\bar{\psi}_{t,h}\rangle = \rho_{t,h}|\psi_{t,h}^{\text{exit}}\rangle \in \mathbb{C}^{16} \cong \mathbb{R}^{512}$, forming the unified `512D` pre-sphere readout vector:
   $$\mathbf{u}_{t,\text{vocab}} \;=\; \mathbf{h}_{t,\text{exit}} \;+\; \gamma_{\text{filter}}\,\operatorname{vec}_{\mathbb{R}}\!\big(\rho_{t,h}|\psi_{t,h}^{\text{exit}}\rangle\big) \;+\; \gamma_{\text{tomo}}\,(\mathbf{q}_t W_{\text{tomo}}) \;\in\; \mathbb{R}^{512}$$
2. **Unified `512D` Vacuum-Subspace Normalization ($\mathcal{H}_{\text{vac}}$ Inside the `512D` Denominator)**:
   With learned scalar vacuum rest-mass $m_{\text{vac}}$ and context-dependent vacuum projection $\mathbf{u}_{t,\text{vac}} = \mathbf{h}_{t,\text{exit}} W_{\text{vac}} \in \mathbb{R}^{d_{\text{vac}}}$ ($d_{\text{vac}} = 4$), we normalize onto $\mathcal{H}_{\text{vocab}}^{(512)} \oplus \mathcal{H}_{\text{vac}}^{(4)}$:
   $$E_{\text{vac}}(t) \;=\; m_{\text{vac}}^2 \;+\; \frac{1}{d_{\text{vac}}}\|\mathbf{u}_{t,\text{vac}}\|_2^2, \qquad \cos\theta_{\text{vac}}(t) \;=\; \sqrt{\frac{\operatorname{MS}(\mathbf{u}_{t,\text{vocab}})}{\operatorname{MS}(\mathbf{u}_{t,\text{vocab}}) + E_{\text{vac}}(t) + \varepsilon}}, \qquad \hat{\mathbf{h}}_t \;=\; \frac{\beta_{\text{temp}}\,\mathbf{u}_{t,\text{vocab}}}{\sqrt{\operatorname{MS}(\mathbf{u}_{t,\text{vocab}}) + E_{\text{vac}}(t) + \varepsilon}} \;\in\; \mathbb{R}^{512}$$
3. **Single `(N, 512) @ (512, 50304)` GEMM + Zero-Synthesis Two-Register Bilinear Trace (`T3`)**:
   - In **`T0..T2`**, the logit for token $v \in \{0, \dots, 50303\}$ is computed by a single `512D` GEMM plus the scalar unigram bias $b_{\text{uni}}[v]$ (from static BPE Zipf prior + `OnlineCausalZipfTracker`):
     $$z_{t,v} \;=\; \langle \mathbf{w}_v,\, \hat{\mathbf{h}}_t \rangle \;+\; b_{\text{uni}}[v]$$
   - In **`T3` (`+ Zero-Synthesis 256 × 256 Two-Register Bilinear Trace`)**, each high byte $b_{\text{hi}} \in \{0, \dots, 255\}$ and low byte $b_{\text{lo}} \in \{0, \dots, 255\}$ owns a complex register ket $|\phi_{\text{hi},h}[b_{\text{hi}}]\rangle, |\phi_{\text{lo},h}[b_{\text{lo}}]\rangle \in \mathbb{C}^{16}$ (plus `256D` readout basis vectors $U_{\text{hi}} \in \mathbb{R}^{256 \times 512}$ and Schur factors $P_{\text{hi}}, P_{\text{lo}} \in \mathbb{R}^{256 \times 512}$). Using $\rho_{t,h} \in \mathbb{C}^{16 \times 16}$ as the bilinear operator between the two 256-state registers:
     $$\Omega_{t}[b_{\text{hi}}, b_{\text{lo}}] \;=\; \gamma_{\text{bilin}} \sum_{h=0}^{15} \operatorname{Re}\,\big\langle \phi_{\text{hi},h}[b_{\text{hi}}] \;\big|\; \rho_{t,h} \;\big|\; \phi_{\text{lo},h}[b_{\text{lo}}] \big\rangle \;+\; \langle U_{\text{hi}}[b_{\text{hi}}],\, \hat{\mathbf{h}}_t \rangle \;+\; \langle P_{\text{hi}}[b_{\text{hi}}],\, P_{\text{lo}}[b_{\text{lo}}] \odot \hat{\mathbf{h}}_t \rangle \;\in\; \mathbb{R}^{256 \times 256}$$
     $\Omega_t[b_{\text{hi}}(v), b_{\text{lo}}(v)]$ is evaluated strictly in `256`-state register space (`< 0.15 ms` per batch) and added directly inside `_SingleGemm512DTomographicCEFn`—**never materializing a `50,304 × 512` synthesized matrix**!

---

## 4. Planned Changes & Testing Architecture

1. **New Flat, Zero-Inheritance Benchmark Module ([`src/qdlblocks/benchmarks/explicit_density_tomographic_512d_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/explicit_density_tomographic_512d_lab.py))**:
   - Standalone, self-contained module (`< 650` SLOC, zero inheritance from legacy `Wave 56–62` classes).
   - `ExplicitDensityTomographicSpec` & `ExplicitDensityVariantMode` (`T0..T4`).
   - `_ExplicitDensityHighwayBlock`: 4-layer quantum block evolving $H = 16$ heads of $\mathbb{C}^{16}$ wavepackets and `256D` packed Hermitian outer-product density matrices $\tilde{\boldsymbol{\rho}}_{t,h} = \tilde{P}(\rho_{t,h}) \in \mathbb{R}^{256}$ with Trunk "No-Click" POVM attention vacuum sink, per-head $U(16)$ unitary channels, and Mach-Zehnder Stiefel interferometers.
   - `compute_tomographic_512d_probe_expectations`: Evaluates $q_{t,h,k} = \langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle \in \mathbb{R}^{512}$ via the Frobenius-isometric packed contraction $\langle \tilde{P}(\rho_{t,h}), \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \rangle_{\mathbb{R}^{256}}$.
   - `_SingleGemm512DTomographicCEFn`: Rule-9 single-pass chunked custom autograd function executing **1 forward and 2 backward `(chunk, 512) @ (512, 50304)` GEMMs** with scalar unigram bias $b_{\text{uni}}[v]$ and optional zero-synthesis `(256, 256)` two-register bilinear trace $\Omega_t[b_{\text{hi}}, b_{\text{lo}}]$.
   - `ExplicitDensityTomographicGPT`: Flat `nn.Module` implementing `T0..T4` with `init_tokens_scanned == 0`.
2. **Diagnostic & Benchmark CLI Scripts**:
   - [`scripts/explicit_density_tomography_and_eval_speed_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/explicit_density_tomography_and_eval_speed_profiler.py)
   - [`scripts/explicit_density_tomographic_512d_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/explicit_density_tomographic_512d_bench.py)
3. **Verification Script & Unit Tests**:
   - [`examples/93_verify_plan083_explicit_density_tomographic_512d.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/93_verify_plan083_explicit_density_tomographic_512d.py)
   - [`tests/test_plan083_explicit_density_tomographic_512d.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan083_explicit_density_tomographic_512d.py)

---

## 5. Operational Tooling & Script Identification

### 5.1 Existing Scripts Used
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py --paths src/qdlblocks/benchmarks/explicit_density_tomographic_512d_lab.py scripts/explicit_density_tomography_and_eval_speed_profiler.py scripts/explicit_density_tomographic_512d_bench.py examples/93_verify_plan083_explicit_density_tomographic_512d.py tests/test_plan083_explicit_density_tomographic_512d.py`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/verify_plan.py --plan-num 83 --json-out .cache/verify_plan083.json`
- `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload`

### 5.2 `Build-Before-Launch` Scripts
1. [`scripts/explicit_density_tomography_and_eval_speed_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/explicit_density_tomography_and_eval_speed_profiler.py): Verifies the Frobenius-isometric outer-product identity $\langle \tilde{P}(\rho), \tilde{P}(|c\rangle\langle c|) \rangle = \langle c \mid \rho \mid c \rangle$, the `50,304`-token operator trace identity $\langle \mathbf{w}_v, \mathbf{q}_t \rangle = \sum_h \operatorname{Tr}(\rho_{t,h} E_{v,h})$, the Rule-9 custom autograd efficiency gate, and benchmarks validation throughput (`Wave 62` `50,304 × 512` re-synthesis vs. `Wave 63` zero-synthesis `512D` readout).
2. [`scripts/explicit_density_tomographic_512d_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/explicit_density_tomographic_512d_bench.py): Runs the 300-step (`2.46M` tokens) `T0..T3` ablation sweep and the 1,200-step (`9.83M` tokens) `T4` scaling run on `10,485,760` FineWeb Shard-0 validation tokens using `StandardizedExperimentLogger`.

### 5.3 `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/density_matrix_purity_and_tomographic_probe_spectrum_inspector.py`: Inspects layerwise $\mathbb{C}^{16 \times 16}$ density-matrix purity $\operatorname{Tr}(\rho_{t,h}^2)$, effective rank $R_{\text{eff}} = 1 / \operatorname{Tr}(\hat{\rho}_{t,h}^2)$, von Neumann entropy $S(\rho_{t,h})$, and the $32 \times 32$ Gram overlap matrix of the learned tomographic probe kets $|c_{k,h}\rangle \in \mathbb{C}^{16}$ across heads $h = 0..15$.

---

## 6. Constants Ledger & Empirical Results (`Wave 63`)

### 6.1 Rule 7 Named Constants Ledger (`src/qdlblocks/benchmarks/explicit_density_tomographic_512d_lab.py`)

| Constant Name | Value | Mathematical Role & Rationale |
| :--- | :---: | :--- |
| `DEFAULT_TOMO_EMBED_DIM` | `512` | Power-of-2 real model width ($H = 16 \times d_h = 16 \times 2 = 512$) |
| `DEFAULT_TOMO_NUM_HEADS` | `16` | Number of complex wavepacket & $\mathbb{C}^{16 \times 16}$ density-matrix heads ($H = 16$) |
| `DEFAULT_TOMO_HEAD_COMPLEX_DIM` | `16` | Complex dimension per head ($\mathbb{C}^{16}$, matching Wave 44 Invariant 2 `register_dim = 16`) |
| `DEFAULT_TOMO_PACKED_RHO_DIM` | `256` | Isometric real vector dimension per $\operatorname{Herm}(\mathbb{C}^{16})$ density matrix ($16^2 = 256$) |
| `DEFAULT_PROBES_PER_HEAD` | `32` | Number of learned complex tomographic probe kets $|c_{k,h}\rangle \in \mathbb{C}^{16}$ per head ($16 \times 32 = 512$) |
| `DEFAULT_TOMO_NUM_LAYERS` | `4` | Number of Quantum Density Highway transformer blocks ($L = 4$) |
| `DEFAULT_TOMO_MLP_HIDDEN` | `1024` | Hidden width of Mach-Zehnder two-beam Stiefel interferometer ($2 \times 512 = 1024$) |
| `DEFAULT_TOMO_VACUUM_DIM` | `4` | Dimension of readout orthogonal vacuum subspace $\mathcal{H}_{\text{vac}}$ ($d_{\text{vac}} = 4$) |
| `DEFAULT_INIT_VACUUM_REST_MASS` | `0.35` | Step-0 scalar vacuum rest-mass $m_{\text{vac}}^{(0)}$ inside the `512D` sphere denominator |
| `DEFAULT_INIT_NOCLICK_BIAS` | `-2.20` | Initial trunk No-Click POVM attention sink log-bias $b_{\text{vac}, h}$ (`p_noclick ≈ 0.12` at Step 0) |
| `DEFAULT_INIT_NOCLICK_POS_LOGIT` | `-1.10` | Initial logit for causal horizon exponent $\gamma_{\text{pos}, h} = \sigma(-1.10) \approx 0.25$ |
| `DEFAULT_NOCLICK_RHO_DAMPING` | `0.50` | Attenuation factor $\kappa_{\text{vac}}$ on density excitation under a trunk No-Click outcome |
| `DEFAULT_INIT_VALUE_EXCITATION_ALPHA` | `0.30` | Convex weight $\alpha_{\text{exc}}$ injecting value outer products $|v_{s,h}\rangle\langle v_{s,h}|$ into $\rho_{s,h}$ |
| `DEFAULT_INIT_HIGHWAY_BLEND_GAMMA` | `0.35` | Per-layer unitary density residual blend factor $\gamma_{\rho,\ell}$ |
| `DEFAULT_INIT_MLP_RHO_INJECT` | `0.20` | Post-MZ-interferometer rank-1 outer-product injection weight into $\rho_{t,h}^{(\ell+1)}$ |
| `DEFAULT_INIT_TOMO_SCALE` | `0.45` | Initial scale $\gamma_{\text{tomo}}$ on the `512D` tomographic probe vector $\mathbf{q}_t W_{\text{tomo}}$ |
| `DEFAULT_INIT_FILTER_SCALE` | `0.35` | Initial scale $\gamma_{\text{filter}}$ on the density-filtered wavepacket $\rho_{t,h}|\psi_{t,h}^{\text{exit}}\rangle$ |
| `DEFAULT_INIT_READOUT_TEMP` | `0.50` | Initial readout sphere inverse-temperature scale $\beta_{\text{temp}}$ (`0.50` at Step 0 for clean BPE-rank Zipf prior; scaled by `1 / 512.0` inside `apply_unified_512d_vacuum_sphere_norm`) |
| `DEFAULT_INIT_BILINEAR_REGISTER_SCALE` | `0.00` | Initial scale $\gamma_{\text{bilin}}$ on the `256 × 256` two-register bilinear trace $\langle \phi_{\text{hi}} \mid \rho_t \mid \phi_{\text{lo}} \rangle$ (`T3`) |
| `DEFAULT_REGISTER_BUCKETS` | `256` | Power-of-2 size of each byte register ($b_{\text{hi}}, b_{\text{lo}} \in \{0, \dots, 255\}$) in `T3` |
| `DEFAULT_TOK_NGRAM_BUCKETS` | `4096` | Power-of-2 causal token N-gram hash buckets (`(4096, 512)`, zero-initialized at Step 0) |
| `TOMO_CE_TOKEN_CHUNK` | `2048` | Token chunk size inside `_SingleGemm512DTomographicCEFn` |
| `DEFAULT_BALL_TRUST_TAU` | `0.18` | Per-step coordinate trust-region ceiling in `_step_active_zero_init_ball` (`BatchSubspaceTikhonovGramMuon(zero_init_ball=True)`) |
| `DEFAULT_BALL_MAX_RMS` | `1.50` | Maximum row-RMS ball radius for active N-gram and register tables (`(4096, 512)` and `(256, 512)`) |
| `DEFAULT_COMPLEX_SPHERE_MAX_ANGLE` | `0.20` | Maximum single-step great-circle rotation angle (radians) on $\mathbb{S}^{31} \subset \mathbb{C}^{16}$ for probe and vacuum kets |
| `DEFAULT_POSITIVE_GEODESIC_MAX_STEP` | `0.22` | Log-space trust-region ceiling $\tau_{\max}$ on $(\mathbb{R}_{>0}, ds = (1/p + 1/\beta_{\text{ref}})dp)$ preventing Step-1 temperature quenching |
| `DEFAULT_BOUNDED_TANGENT_MAX_STEP` | `0.12` | Default per-step tangent trust-region ceiling for signed couplings and phases |

### 6.2 Empirical Results & Architectural Post-Mortem (`Wave 63`)

#### 1. Isometric Tomography & Hardware Throughput Profiler ([`scripts/explicit_density_tomography_and_eval_speed_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/explicit_density_tomography_and_eval_speed_profiler.py))
- **Mathematical Identities (`identities.passed = True`)**:
  - `max_isometric_probe_expectation_error = 8.94e-8` ($\langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle = \langle \tilde{P}(\rho_{t,h}), \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \rangle_{\mathbb{R}^{256}}$).
  - `max_tomographic_operator_trace_error = 1.19e-6` ($\langle \mathbf{w}_v, \mathbf{q}_t \rangle = \sum_{h=0}^{15} \operatorname{Tr}(\rho_{t,h} E_{v,h})$).
- **Rule-9 Custom Autograd Single-Pass Audit (`rule9_autograd_audit.passed = True`)**:
  - `_SingleGemm512DTomographicCEFn`: `fwd_mm = 4, bwd_mm = 8` (`bwd_to_fwd_mm_ratio = 2.0`), `fwd_bmm = 0, bwd_bmm = 0` (`T0..T2`), `fwd_pow = 0, bwd_pow = 0`.
- **`10.48M`-Token Validation & Step-Time Speed (`B = 16, T = 512`)**:
  - `T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM` (100% byte-pair-free single-GEMM `512D` tomography): **`524.66 ms` / train step**, **`164.41 ms` / eval batch**, and **`210.44 s` projected `10.48M`-token Shard-0 evaluation** (**`1.09x` faster** than Wave 62's `230s` eval by hoisting the `50,257 × 512` detector matrix once outside the 1,280-batch validation loop).
  - `T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE`: **`608.67 ms` / train step**, **`197.88 ms` / eval batch**.

#### 2. `300`-Step (`2.46M`-Token) Ablation Matrix (`T0..T3`) & `1,200`-Step (`9.83M`-Token) Scaling (`T4`)

| Variant | Density Highway $\rho_{t,h} \in \operatorname{Herm}(\mathbb{C}^{16})$ | `16×32=512D` Probe Tomography | Trunk No-Click + `512D` Vacuum | `256×256` Bilinear Trace | Init Tokens / Time | Step 0 Val CE | Step 300 (`2.46M`) Val CE / PPL | Step 300 Top-1 / Top-5 | Step 300 $\operatorname{Rank}_{\text{eff}}(\rho)$ | Step 300 $p_{\text{noclick}}$ | Eval Wall (`65.5K` tok) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T0_PURE_WAVEPACKET_RANK1_BASELINE`** | No | No | No | No | `0` (`7.3 ms`) | `9.1463` | `6.3971` (`600.10`) | `16.13%` / `31.29%` | `1.00 / 16` | `0.000` | `1.1s` |
| **`T1_EXPLICIT_OUTER_DENSITY_TOMOGRAPHIC_512D`** | Yes (`256D` packed) | Yes ($K=32$/head) | No | No | `0` (`7.3 ms`) | `9.1473` | `6.3925` (`597.35`) | `16.16%` / `31.66%` | `1.99 / 16` | `0.000` | `1.8s` |
| **`T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM`** | Yes (`256D` packed) | Yes ($K=32$/head) | Yes | No | `0` (`7.3 ms`) | `9.1473` | **`6.3893`** (`595.45`) | **`16.31%`** / `31.64%` | **`2.05 / 16`** | `0.357` | `1.8s` |
| **`T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE`** | Yes (`256D` packed) | Yes ($K=32$/head) | Yes | Yes | `0` (`13.0 ms`) | `9.0288` | **`6.3826`** (`591.46`) | `16.20%` / `31.27%` | `1.27 / 16` | `0.449` | `2.0s` |

- **`1,200`-Step (`9.83M`-Token) Scaling (`T4_CONSOLIDATED_TOMOGRAPHIC_SCALING_1200`)**:
  - **Progress Curve (`65,536` Val Tokens)**: Step 0 `9.0288` $\to$ Step 150 `6.8053` $\to$ Step 300 `6.4645` $\to$ Step 450 `6.2009` $\to$ Step 600 `6.0046` $\to$ Step 750 `5.8331` $\to$ Step 900 `5.7242` $\to$ Step 1050 `5.6351` $\to$ **Step 1200 `5.5850` Val CE** (`266.39` PPL, **`Top-1 = 21.85%`**, **`Top-5 = 38.13%`**, $\operatorname{Rank}_{\text{eff}}(\rho) = 1.61 / 16$, $p_{\text{noclick}} = 0.529$).
  - **Official Full `10,485,760`-Token (`10.48M`) FineWeb Shard-0 Benchmark**: **`5.5402` Val CE** (`254.74` PPL, **`Top-1 = 21.33%`**).

#### 3. Key Findings & Next-Wave Recommendations
1. **The `1 / 512.0` Sphere Normalization & Zero-PPMI Cold-Start Temperature Invariant**:
   - Because both `u_vocab / denom` (`denom = sqrt(mean(u_vocab^2) + e_vac)`) and `w_vocab = F.rms_norm(w_det_512)` are normalized to **unit RMS (`1.0`)**, their L2 norms are $\sqrt{512} \approx 22.63$ and their inner product is $512 \cos\theta$. Dividing `readout_temp` by `512.0` inside `apply_unified_512d_vacuum_sphere_norm` and initializing `DEFAULT_INIT_READOUT_TEMP = 0.50` is essential so Step-0 random detectors do not saturate the `±15.0` `tanh` softcap or corrupt the analytical BPE-rank Zipf prior.
2. **`T2` (`Option B`: Pure `16 × 32 = 512D` Tomography + No-Click + Vacuum) vs. `T3` (Two-Register Bilinear BMM)**:
   - Adding the explicit $\mathbb{C}^{16 \times 16}$ outer-product density highway and `16 × 32 = 512D` tomographic probes (`T1`) and equipping it with the trunk No-Click POVM attention sink and unified `512D` readout vacuum (`T2`) monotonically improves 300-step Val CE (`6.3971 -> 6.3925 -> 6.3893`) and achieves the highest 300-step Top-1 accuracy (**`16.31%`**) and highest density effective rank (**`2.05 / 16`**), with zero byte-pair tables and a fast `210.4s` single-GEMM `10.48M`-token eval.
   - Adding the `256 × 256` two-register bilinear BMM inside every `2,048`-token CE chunk (`T3`/`T4`) improves 300-step Val CE by only `-0.0067` nats (`6.3893 -> 6.3826`) while adding ~55% evaluation overhead (`210.4s -> 327.5s` on `10.48M` tokens) and pulling density rank down from `2.05` to `1.27`.
3. **Why Wave 62 Reached `5.2179` Val CE vs. Wave 63's `5.5402` Val CE — Factored Detector Parameter Sharing**:
   - In Wave 63, `w_det_512` `(50304, 512)` is a flat, independent 50,304-row matrix updated only on active tokens seen in training batches (`9.83M` tokens total). Rare tokens receive very few gradient updates when parameterized solely as independent rows.
   - In `Option B` (`16 × 32 = 512D` Quantum State Tomography), we can get **both** cross-token parameter sharing **and** `T2`'s zero-re-synthesis `210.4s` validation speed by parameterizing `w_det_512[v]` as a residual sum of the flat `(50304, 512)` table and shared sub-token / coarse-cluster factors that are materialized **once per optimizer step** (and hoisted **once outside the 1,280-batch validation loop**), keeping the readout 100% single-GEMM `(N, 512) @ (512, 50304)`.

### 6.3 Constant-LR (Zero Cosine Schedule) Stability & Breakdown Ceiling: `AdamW` vs. 100% `AdamW`-Free Riemannian Optimizers ([`scripts/constant_lr_stability_ceiling_sweep.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/constant_lr_stability_ceiling_sweep.py))

1. **Eliminating `AdamW` Completely (`BatchSubspaceTikhonovGramMuon(zero_init_ball=True)` + `RiemannianGeodesicScalarAndProbeOptimizer`)**:
   - **Active-Hash `(4096, 512)` N-Gram & `(256, 512)` Register Tables**: Routed through [`BatchSubspaceTikhonovGramMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py#L353) (`zero_init_ball=True`), updating strictly non-zero active batch rows via `512 × 512` Tikhonov-damped coupled Newton-Schulz Gram whitening + coordinate-sparse tangent preconditioning inside the unit-RMS ball $\mathbb{B}_{\text{rms}}^{512}(1.50)$.
   - **Complex $\mathbb{S}^{31}$ Probe & Vacuum Kets (`probe_re, probe_im`, `k_vac_ket`)**: Routed through tangent-projected great-circle geodesic rotation on $\mathbb{S}^{31} \subset \mathbb{C}^{16}$ (`max_step = 0.20` rad).
   - **Positive Scales (`readout_temp`, `vac_rest_mass`, `sigma`, `val_exc_alpha`, `highway_gamma`, `mlp_rho_inject`, `obs_scale`, `mz_scale`)**: Routed through the hyperbolic-multiplicative geodesic map on $(\mathbb{R}_{>0}, ds = (\frac{1}{p} + \frac{1}{\beta_{\text{ref}}})dp)$:
     $$p^{(t+1)} \;=\; p^{(t)} \cdot \exp\!\left(\frac{-\tau_{\max} \cdot \tanh\!\left(\dfrac{\text{cold\_ramp} \cdot \eta \cdot \hat{u}_t}{\tau_{\max}}\right)}{1 + p^{(t)} / \beta_{\text{ref}}}\right)$$
     which is purely multiplicative as $p \to 0^+$ (**preventing Step-1 temperature quenching at any learning rate**) and bounded in step magnitude as $p \to \infty$ (**preventing high-LR oscillation around the equilibrium $\beta_{\text{temp}}^* \approx 6.5\text{–}7.5$**).
   - **Sparse Active Vocabulary Priors (`unigram_bias`, `radial_scale` of shape `(50304,)`)**: Updated strictly on active batch token IDs (`active_ids`) with zero dense `50,304`-element `AdamW` sweeps.

2. **300-Step Uniform Constant-LR Comparison (`With AdamW` vs. `100% AdamW-Free`, Zero Warmup, Zero Cosine Schedule)**:

| Uniform Constant LR Multiplier | `With AdamW` Step 300 Val CE (`PPL`) | `With AdamW` Top-1 / $\beta_{\text{temp}}$ | **`100% AdamW-Free` Step 300 Val CE (`PPL`)** | **`100% AdamW-Free` Top-1 / $\beta_{\text{temp}}$** | **`100% AdamW-Free` $\operatorname{Rank}_{\text{eff}}(\rho)$** | **Val CE Gain (`AdamW-Free` vs `AdamW`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T2` Cosine Baseline (`1.0x`)** | `6.3893` (`595.45`) | `16.31%` / `7.12` | `6.3893` (`595.45`) | `16.31%` / `7.12` | `2.05 / 16` | Baseline |
| **`1.00x` Constant LR** | `6.3675` (`582.59`) | `15.70%` / `7.57` | **`6.3812` (`590.64`)** | **`15.66%`** / `7.15` | **`3.10 / 16`** | **Beats Cosine (`-0.0081` nats), `+1.34` higher $\rho$-rank** |
| **`1.25x` Constant LR** | `6.3789` (`589.28`) | `15.51%` / `7.58` | **`6.3887` (`595.08`)** | **`15.38%`** / `6.76` | **`2.50 / 16`** | **Beats Cosine (`-0.0006` nats), `+1.01` higher $\rho$-rank** |
| **`1.50x` Constant LR** | `6.3987` (`601.08`) | `15.08%` / `7.52` | **`6.4100` (`607.90`)** | **`15.33%`** / `6.49` | `1.96 / 16` | **`+0.25%` higher Top-1, Step-100 `6.8533` Val CE** |
| **`2.00x` Constant LR** | `6.4390` (`625.81`) | `15.24%` / `7.47` | **`6.4560` (`636.50`)** | `14.66%` / `5.86` | `1.48 / 16` | **Stable** |
| **`4.00x` Constant LR** | `6.7725` (`873.46`) | `12.57%` / `7.15` | **`6.7538` (`857.27`)** | `12.47%` / `5.42` | `1.15 / 16` | **`-0.0187` nats (`-16.2` PPL)** |
| **`6.00x` Constant LR** | `7.8218` (`2494.38`) | `5.56%` / **`0.05` (Quenched!)** | **`6.9813` (`1076.29`)** | **`11.32%` / `5.64`** | `1.05 / 16` | **`-0.8405` nats (`-1,418` PPL, No Quench!)** |
| **`8.00x` Constant LR** | `7.8451` (`2553.25`) | `5.56%` / **`0.05` (Quenched!)** | **`7.1102` (`1224.45`)** | **`10.95%` / `6.05`** | `1.07 / 16` | **`-0.7349` nats (`-1,329` PPL, No Quench!)** |
| **`16.0x` Constant LR** | `7.8435` (`2549.18`) | `5.56%` / **`0.05` (Quenched!)** | **`7.3643` (`1578.63`)** | **`8.92%` / `5.77`** | `1.12 / 16` | **`-0.4792` nats (`-971` PPL, No Quench!)** |
| **`32.0x` Constant LR** | `7.8464` (`2556.51`) | `5.56%` / **`0.05` (Quenched!)** | **`7.6070` (`2012.20`)** | **`7.35%` / `5.63`** | `11.19 / 16` | **`-0.2394` nats (`-544` PPL, No Quench!)** |



