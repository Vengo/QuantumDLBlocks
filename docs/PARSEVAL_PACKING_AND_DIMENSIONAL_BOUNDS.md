# Parseval Frame Conservation, Incomplete Block-QR Packing Bounds, and Dimensional Thresholds for Quantum Language Modeling (`docs/PARSEVAL_PACKING_AND_DIMENSIONAL_BOUNDS.md`)

This document provides the complete mathematical derivation and empirical verification for **Experiment Plan 038 (`Wave 23`)**, addressing three foundational questions in finite-dimensional quantum amplitude language modeling over $V = 50,304$ vocabulary states in $\mathbb{R}^d$:

1. **Why anti-aligned vocabulary words ($c_{t,v} < 0$) cannot be pushed to exact orthogonality ($c_{t,v} = 0$) in $d = 256$ under squared inner products $F_{t,v} = c_{t,v}^2$ ($\mathbb{RP}^{d-1}$ identification & Parseval conservation)**.
2. **What happens empirically when drawing $V = 50,304$ unit vectors in $d \in \{16, 32, 64, 128, 256, 384, 512, 1024, 2048\}$ and applying Incomplete / Windowed Block-QR orthogonalization vs. Riemannian Welch Frame Landing** (via [`scripts/parseval_tail_overlap_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_tail_overlap_profiler.py)).
3. **What effective Hilbert-space dimension $d_{\text{eff}}$ is required to reach `< 3.28` nats and `< 2.50` nats under raw quadratic / uncentered readout vs. the Plan 038 Last-Layer Measurement Operator ($\mathcal{M}_{\text{last}}$) and Traceless Observable ($\operatorname{Tr}(\widehat{O}) = 0$)**.

---

## 1. Why Anti-Aligned Words ($c_{t,v} < 0$) Cannot Be Pushed to Orthogonality ($c_{t,v} = 0$) in $d = 256$ When $V = 50,304$

### 1.1 Exact Parseval Frame Conservation Law on $\mathbb{S}^{d-1}$

Let $M = [m_1, m_2, \dots, m_V]^\top \in \mathbb{R}^{V \times d}$ be any vocabulary detector table of $V = 50,304$ unit-norm vectors ($\|m_v\|_2 = 1$ for all $v \in \{1, \dots, V\}$), and let $G = M^\top M = \sum_{v=1}^V m_v m_v^\top \in \mathbb{R}^{d \times d}$ denote its $(d \times d)$ frame Gram operator with eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$.

Because every detector has unit norm, the trace of $G$ is strictly conserved:
$$\operatorname{Tr}(G) = \sum_{i=1}^d \lambda_i = \sum_{v=1}^V \|m_v\|_2^2 = V.$$

Define the spectral participation ratio (effective sharp rank) $R_{\text{sharp}} \in [1, d]$ of the vocabulary frame:
$$R_{\text{sharp}}(M) \triangleq \frac{\left(\operatorname{Tr} G\right)^2}{\|G\|_F^2} = \frac{\left(\sum_{i=1}^d \lambda_i\right)^2}{\sum_{i=1}^d \lambda_i^2} \le d.$$

Now compute the average total squared overlap across all vocabulary pairs $(u, v) \in \{1, \dots, V\}^2$ where $c_{u,v} = \langle m_u \mid m_v \rangle$:
$$\frac{1}{V} \sum_{u=1}^V \sum_{v=1}^V c_{u,v}^2 = \frac{1}{V} \|M M^\top\|_F^2 = \frac{1}{V} \operatorname{Tr}\!\left((M^\top M)^2\right) = \frac{1}{V} \|G\|_F^2 = \frac{V}{R_{\text{sharp}}(M)} \ge \frac{V}{d}.$$

Subtracting the self-overlap diagonal $c_{u,u}^2 = 1$ yields the **exact Parseval off-diagonal conservation law**:
$$\mathbb{E}_u\!\left[\sum_{v \ne u} c_{u,v}^2\right] = \frac{V}{R_{\text{sharp}}(M)} - 1 \ge \frac{V - d}{d}.$$

At $V = 50,304$ and $d = 256$:
- Even for a mathematically optimal Welch-bound tight frame ($G = \frac{V}{d} I_d$, $R_{\text{sharp}} = 256.0$), every state has a conserved sum of squared overlaps:
  $$\sum_{v=1}^{50,304} c_{t,v}^2 \equiv \frac{50,304}{256} = \mathbf{196.50}, \qquad \sqrt{\mathbb{E}_{u \ne v}[c_{u,v}^2]} \ge \sqrt{\frac{V - d}{d(V - 1)}} = \mathbf{0.062341}.$$
- Since the true target word $y_t$ can absorb at most $c_{t,y_t}^2 \le 1.0$, the remaining $V - 1 = 50,303$ non-target words **must** carry at least $196.50 - 1.0 = \mathbf{195.50}$ units of total squared overlap. It is a linear-algebraic impossibility to set $c_{t,v} = 0$ for all $50,303$ unrelated words in $\mathbb{R}^{256}$.

### 1.2 Why Squared Inner Products $F_{t,v} = c_{t,v}^2$ Force Uniform Non-Zero Tail Excitation Instead of $c_{t,v} = 0$

In a classical linear readout $z_{t,v} = \beta c_{t,v}$, the sphere $\mathbb{S}^{d-1}$ distinguishes opposite hemispheres: anti-aligned words ($c_{t,v} < 0$) produce strictly negative logits $z_{t,v} < 0$, so their exponential contribution $e^{\beta c_{t,v}} \ll 1$ vanishes in the softmax denominator.

By contrast, when the readout depends on the **squared inner product** $F_{t,v} = |\langle \psi_t \mid m_v \rangle|^2 = c_{t,v}^2$:
1. **$\mathbb{RP}^{d-1}$ Antipodal Collapse (`Fix 1`)**:
   Squaring identifies antipodal points $+m_v \sim -m_v$ ($\mathbb{S}^{d-1} \to \mathbb{RP}^{d-1}$), mapping every anti-aligned word ($c_{t,v} < 0$) to a **strictly positive** fidelity $F_{t,v} = (-|c_{t,v}|)^2 > 0$.
2. **Vanishing Repulsive Force at $c_{t,v} = 0$**:
   Under $k$-copy quadratic Born energy $E_k(c_{t,v}) = k \ln(1 + \beta c_{t,v}^2)$ (or $k \ln(1 + \frac{\beta}{k} c_{t,v}^2)$), the gradient pushing a non-target word toward orthogonality ($c_{t,v} \to 0$) is:
   $$\frac{\partial E_k}{\partial c_{t,v}} = \frac{2 k \beta \, c_{t,v}}{1 + \beta c_{t,v}^2} = O(|c_{t,v}|) \xrightarrow{c_{t,v} \to 0} 0.$$
   Because the derivative vanishes linearly at $c_{t,v} = 0$, the repulsive force pushing unrelated words toward zero becomes arbitrarily weak near the equator $c_{t,v} = 0$.
3. **Jensen / Concavity Equilibrium Spreads the Parseval Budget Uniformly Across All $50,303$ Tail Words**:
   Consider minimizing the tail partition energy $\sum_{v \ne y_t} k \ln(1 + \gamma c_{t,v}^2)$ subject to the fixed Parseval constraint $\sum_{v \ne y_t} c_{t,v}^2 = S_{\text{tail}} = \frac{V}{R_{\text{sharp}}} - c_{t,y_t}^2$. Even though $\phi(u) = \ln(1 + \gamma u)$ is concave in $u = c_{t,v}^2$, the softmax partition function sums $\exp(E_k(c_{t,v})) = (1 + \gamma c_{t,v}^2)^k$, which for $k > 1$ is **strictly convex** in $u = c_{t,v}^2$! By Jensen's inequality on the convex function $u \mapsto (1 + \gamma u)^k$:
   $$\sum_{v \ne y_t} \left(1 + \gamma c_{t,v}^2\right)^k \ge (V - 1) \left(1 + \frac{\gamma \, S_{\text{tail}}}{V - 1}\right)^k,$$
   with equality **if and only if every single non-target word has the same non-zero squared overlap**:
   $$c_{t,v}^2 = \frac{S_{\text{tail}}}{V - 1} \approx \frac{1}{R_{\text{sharp}}} \implies c_{t,v} \approx \pm \frac{1}{\sqrt{R_{\text{sharp}}}}.$$
   Thus, minimizing Cross-Entropy under the Parseval constraint actively **forbids** concentrating the overlap into a few words and setting the rest to zero; instead, it forces all $50,303$ non-target words to sit at $c_{t,v} \approx \pm \frac{1}{\sqrt{R_{\text{sharp}}}}$, inflating the tail log-partition function by $+\frac{k \beta}{R_{\text{sharp}}}$!

---

## 2. Empirical Experiment: Packing $V = 50,304$ Vectors in $d \in \{16, \dots, 2048\}$ via Incomplete Block-QR vs. Welch Frame Landing

### 2.1 What Happens When Running Incomplete / Windowed Block-QR on $50,304$ Vectors in $\mathbb{R}^d$?

Standard QR factorization (`torch.linalg.qr`) on $M^\top \in \mathbb{R}^{d \times V}$ produces at most $\min(d, V) = d$ mutually orthogonal columns; columns $d+1, \dots, V$ have no remaining orthogonal dimensions in $\mathbb{R}^d$.

To test whether local or windowed orthogonalization can beat the Parseval floor, [`scripts/parseval_tail_overlap_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_tail_overlap_profiler.py) partitions the $V = 50,304$ vectors into $\lceil V / b \rceil$ contiguous blocks of size $b = \min(d, 128)$, applies a damped inter-block Gram-Schmidt projection against the preceding block's orthonormal basis $Q_{m-1}$:
$$X_m \leftarrow X_m - \frac{1}{2} (X_m Q_{m-1}^\top) Q_{m-1},$$
and then orthogonalizes each block $X_m \in \mathbb{R}^{b \times d}$ via QR (`incomplete_block_qr`). We compare this against:
- `zipf_cone`: Anisotropic Zipfian-collapsed cone matching `w22`'s empirical collapse ($R_{\text{sharp}} \approx 16.2$),
- `random_sphere`: Isotropic unit vectors on $\mathbb{S}^{d-1}$,
- `welch_frame_landing`: $6$ steps of Riemannian Frame-Potential Landing $M \leftarrow \operatorname{Norm}\!\big(M - \eta M(\frac{d}{V} M^\top M - I_d)\big)$.

### 2.2 Full Numerical Sweep Across $d \in \{16, 32, 64, 128, 256, 384, 512, 1024, 2048\}$ ($V = 50,304$)

The table below reports the exact measurements from [`.cache/parseval_packing_50304_sweep.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/parseval_packing_50304_sweep.json) generated by [`scripts/parseval_tail_overlap_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_tail_overlap_profiler.py):

| Dim $d$ | Max Ortho Subset $\min(d, V)$ | Welch Lower Bound $\sqrt{\frac{V-d}{d(V-1)}}$ | `incomplete_block_qr` Intra-Block RMS | `incomplete_block_qr` Global RMS | `random_sphere` Global RMS | `zipf_cone` $R_{\text{sharp}}$ | `incomplete_block_qr` $R_{\text{sharp}}$ | `welch_frame_landing` $R_{\text{sharp}}$ | Tight-Frame Parseval $\sum_v c_{u,v}^2 = V/R_{\text{sharp}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`16`** | `16` | `0.249963` | **`0.000000`** | `0.249963` | `0.250009` | `12.36` | `16.00` | `16.00` | **`3,144.00`** |
| **`32`** | `32` | `0.176722` | **`0.000000`** | `0.176722` | `0.176799` | `15.71` | `32.00` | `32.00` | **`1,572.00`** |
| **`64`** | `64` | `0.124921` | **`0.000000`** | `0.124921` | `0.125008` | `16.22` | `64.00` | `64.00` | **`786.00`** |
| **`128`** | `128` | `0.088276` | **`0.000000`** | `0.088276` | `0.088392` | `16.23` | `128.00` | `128.00` | **`393.00`** |
| **`256`** | `256` | `0.062341` | **`0.000000`** | `0.062363` | `0.062496` | `16.16` | `255.73` | `256.00` | **`196.50`** |
| **`384`** | `384` | `0.050836` | **`0.000000`** | `0.050913` | `0.051035` | `16.21` | `382.99` | `384.00` | **`131.00`** |
| **`512`** | `512` | `0.043969` | **`0.000000`** | `0.044076` | `0.044191` | `16.22` | `509.57` | `512.00` | **`98.25`** |
| **`1024`** | `1024` | `0.030931` | **`0.000000`** | `0.031159` | `0.031253` | `16.26` | `1,009.45` | `1,024.00` | **`49.13`** |
| **`2048`** | `2048` | `0.021643` | **`0.000000`** | `0.022031` | `0.022101` | `16.21` | `1,979.57` | `2,047.99` | **`24.56`** |

### 2.3 Key Geometric Insights from the Incomplete Block-QR Experiment

1. **Zero Local Overlap vs. Invariant Global Welch Floor**:
   Within every local block of size $b = \min(d, 128)$, `incomplete_block_qr` drives the intra-block RMS off-diagonal cosine to **`0.000000`** across all 9 dimensions. Yet the **global** RMS off-diagonal cosine across all $V = 50,304$ vectors (`0.062363` at $d=256$) cannot go a single ten-thousandth below the Welch lower bound (`0.062341` at $d=256$). Because $\operatorname{rank}(M^\top) \le d$, orthogonalizing block $m$ against block $m-1$ inevitably rotates block $m$ back into the subspace already occupied by blocks $1, \dots, m-2$.
2. **Tight Frame ($R_{\text{sharp}} = 256.0$) vs. `w22` Cone Collapse ($R_{\text{sharp}} = 16.5$)**:
   In Plan 037 (`Wave 22`), `m_sharp` started at Step 0 with $R_{\text{sharp}} = 249.45$ ($\sum_v c_{t,v}^2 \approx 201.5$), but collapsed by Step 900 to $R_{\text{sharp}} = 16.5\text{–}19.3$ ($\sum_v c_{t,v}^2 \approx \mathbf{3,048.7}$ — matching the `zipf_cone` row at $d=256$ where $R_{\text{sharp}} = 16.16$ and $\sum_v c_{t,v}^2 = 3,113.45$).
   Why did `w22` collapse?
   - **Cholesky 8th-Power Singular-Value Amplification**: Each `BlockCholeskyLinear` applies $C = L L^\top$ with eigenvalues $\sigma_i(L)^2$. The Frobenius norm squared of the Gram matrix $G = M^\top M$ scales with the fourth moment of eigenvalues $\sum_i \lambda_i(G)^2 \propto \sum_i \sigma_i(L)^8$. Without a spectral-norm cap on $L$, a $1.4\times$ growth in the top singular value of $L$ amplifies its weight in $\|G\|_F^2$ by $1.4^8 = \mathbf{14.76\times}$!
   - **Uncentered Detector Synthesis**: Adding the low-rank `coarse_transition` ($256 \times 256$) and `pred_wav_2048` ($2048 \times 256$) tables directly to all $V = 50,304$ vocabulary rows created a shared cone mean $\bar{m} \ne 0$ that grew under unconstrained `AdamW`.

---

## 3. Dimensional Thresholds for `< 3.28` Nats and `< 2.50` Nats: Raw/Uncentered Readout vs. Traceless Observable + Last-Layer Measurement Operator

### 3.1 Analytical Cumulant Expansion of the Tail Partition Function

For a query state $\psi_t \in \mathbb{S}^{d-1}$ with target alignment $\alpha = \langle \psi_t \mid m_{y_t} \rangle$ and off-diagonal overlaps $c_{t,v}$ ($v \ne y_t$) with mean $\bar{c}_t = \langle \psi_t \mid \bar{m} \rangle$ and Parseval variance $\sigma_t^2 = \frac{1}{V-1} \sum_{v \ne y_t} (c_{t,v} - \bar{c}_t)^2 \approx \frac{1}{R_{\text{sharp}}}$:
- **Under Raw Quadratic / Uncentered Readout**:
  The tail log-partition function $\ln Z_{\text{tail}} = \ln \sum_{v \ne y_t} \exp(E_{t,v})$ picks up both a first-order cone-bias shift and a second-order Parseval variance tax:
  $$\ln Z_{\text{tail}}^{\text{uncentered}} \approx \ln(V - 1) + \beta \bar{c}_t + \frac{k - 1}{2k} \frac{\beta^2}{R_{\text{sharp}}} \approx \ln(V - 1) + \beta \bar{c}_t + \frac{k \beta}{2 R_{\text{sharp}}}.$$
- **Under the Plan 038 Traceless Observable ($\operatorname{Tr}(\widehat{O}) = 0$) + Last-Layer Measurement Operator ($\mathcal{M}_{\text{last}}$)**:
  1. **Last-Layer Measurement Operator (`use_last_layer_measurement_operator=True`)**: Applies `readout_chol_bridge`, `coarse_transition^\top`, `pred_wav_2048`, and the $R=16$ Kraus operators directly to the $(B \cdot T = 8,192, d)$ network output states (a **$6.14\times$ row reduction** and **`13.98x` CPU / GPU kernel speedup** in [`scripts/block_cholesky_kernel_benchmark.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/block_cholesky_kernel_benchmark.py)) and projects out the input token's initial wavefunction $\psi_{x_t}^{(0)}$ on the output side:
     $$\tilde{\psi}_{\text{sharp}, t} = \operatorname{Norm}\!\Big(m_{\text{out}, t} - \langle m_{\text{out}, t} \mid \psi_{x_t}^{(0)} \rangle \psi_{x_t}^{(0)}\Big) \implies \langle \tilde{\psi}_{\text{sharp}, t} \mid \psi_{x_t}^{(0)} \rangle \equiv 0.$$
  2. **Traceless Observable (`use_traceless_observable=True`)**: Defines the zero-trace operator over the active vocabulary:
     $$\hat{c}_{t,v} = c_{t,v} - \frac{1}{V} \sum_{u=1}^V c_{t,u}, \qquad \widehat{O}_{t,v}^{\text{sharp}} = \hat{c}_{t,v} + \gamma_{\text{trace}}\!\left(\operatorname{relu}(\hat{c}_{t,v})^2 - \frac{1}{V} \sum_{u=1}^V \operatorname{relu}(\hat{c}_{t,u})^2\right), \qquad \sum_{v=1}^V \widehat{O}_{t,v}^{\text{sharp}} \equiv 0.$$
     Subtracting $\bar{c}_t$ eliminates the first-order cone shift ($\beta \bar{c}_t \equiv 0$), while subtracting $\tau_{t,+} = \frac{1}{V} \sum_u \operatorname{relu}(\hat{c}_{t,u})^2 \approx \frac{1}{2 R_{\text{sharp}}}$ suppresses unrelated words in the Parseval noise floor ($0 < \hat{c}_{t,v} < \sqrt{\tau_{t,+}}$) and gives a superlinear positive boost $+\gamma_{\text{trace}}(\hat{c}_{t,y_t}^2 - \tau_{t,+}) > 0$ to the true target word $y_t$.

### 3.2 Achievable Cross-Entropy Floor Table Across Dimensions $d \in \{16, \dots, 2048\}$

From [`.cache/parseval_packing_50304_sweep.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/parseval_packing_50304_sweep.json) ($\beta = 32.0, k = 34.0, \gamma_{\text{trace}} = 0.55$, evaluated on the Welch-landed tight frame $R_{\text{sharp}} = d$ vs. the `w22` collapsed cone $R_{\text{sharp}} \approx 16.2$):

| Dim $d$ | `zipf_cone` Uncentered CE ($\alpha=0.55$) | Tight-Frame Raw Quad Born CE ($\alpha=0.55$) | Tight-Frame Traceless Quad Born CE ($\alpha=0.55$) | Tight-Frame Uncentered $k$-Boson CE ($\alpha=0.55$) | **Plan 038 Traceless + $\mathcal{M}_{\text{last}}$ CE ($\alpha=0.55$)** | Tight-Frame Raw Quad Born CE ($\alpha=0.68$) | Tight-Frame Uncentered $k$-Boson CE ($\alpha=0.68$) | **Plan 038 Traceless + $\mathcal{M}_{\text{last}}$ CE ($\alpha=0.68$)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`16`** | `13.1622` | `66.3259` | `60.3261` | `12.2462` | `12.3274` | `57.7995` | `9.9960` | `9.0450` |
| **`32`** | `11.9093` | `52.9760` | `48.4071` | `9.2329` | `8.4702` | `44.9763` | `7.2826` | `5.6481` |
| **`64`** | `11.6716` | `37.7377` | `33.8904` | `6.7062` | `5.0721` | `30.3910` | `5.0996` | `2.8597` |
| **`128`** | `11.6914` | `23.5423` | `20.4971` | `4.9366` | **`2.8495`** | `17.7648` | `3.6435` | **`1.3099`** |
| **`256`** | `11.7242` | `11.9900` | `9.5507` | `3.8682` (`> 3.28`) | **`1.7358` (`< 3.28`)** | `7.8848` | `2.8066` (`> 2.50`) | **`0.7083` (`< 2.50`)** |
| **`384`** | `11.6606` | `7.3152` | `5.2146` | `3.4801` | **`1.4010`** | `4.3838` | `2.5132` | **`0.5561`** |
| **`512`** | `11.7144` | `5.0299` | **`3.2266` (`< 3.28`)** | **`3.2811` (`~ 3.28`)** | **`1.2461`** | `3.1037` | **`2.3655` (`< 2.50`)** | **`0.4902`** |
| **`1024`** | `11.7624` | **`3.1903` (`< 3.28`)** | **`2.0431`** | **`2.9754` (`< 3.28`)** | **`1.0316`** | **`2.2483` (`< 2.50`)** | **`2.1422` (`< 2.50`)** | **`0.4030`** |
| **`2048`** | `11.7387` | **`2.8626` (`< 3.28`)** | **`1.9198`** | **`2.8204` (`< 3.28`)** | **`0.9338`** | **`2.0541` (`< 2.50`)** | **`2.0308` (`< 2.50`)** | **`0.3647`** |

### 3.3 Summary of Dimensional Thresholds

1. **Without Traceless Centering (`raw_quadratic_born` / `uncentered_kboson`)**:
   - In the `w22` cone-collapse regime ($R_{\text{sharp}} \approx 16.2$), Cross-Entropy is trapped above **`9.5–11.7` nats** regardless of the ambient embedding dimension $d$, because the effective rank is only $16.2$.
   - Even with a tight frame ($R_{\text{sharp}} = d$), reaching **`< 3.28` nats** at $\alpha_{\text{target}} = 0.55$ requires **$d_{\text{eff}} \ge 384\text{–}512$** (`3.8682` nats at $d=256 \to 3.2811$ nats at $d=512$ for uncentered $k$-boson; $d = 512$ for traceless quadratic Born `3.2266`), and reaching **`< 2.50` nats** under raw quadratic Born requires **$d_{\text{eff}} \ge 1,024$** (`2.2483` nats at $d=1,024$).
2. **With Plan 038 Last-Layer Measurement Operator + Traceless Observable ($\operatorname{Tr}(\widehat{O}) = 0$)**:
   - By keeping the initial wavefunctions $\psi_v^{(0)}$ tied, zero-mean centered, and regularized via Riemannian Frame-Potential Landing (`_stabilize_pyramid_frame`) + `StiefelMuon` on `up_8192` and `up_65536`, and evaluating the zero-trace observable $\widehat{O}_{t,v}^{\text{sharp}}$, both **`< 3.28` nats (`1.7358` nats at $\alpha=0.55$) and `< 2.50` nats (`0.7083` nats at $\alpha=0.68$) are mathematically achievable at $d = 256$**!
