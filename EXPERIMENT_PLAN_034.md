# Experiment Plan 034 — Tied-$2^{16}$ ($N = 65,536$) Multi-Level Wavelet-Cholesky Amplitude Networks, Mach-Zehnder Coupled-Beamsplitter Equi-Dimensional $O(N)$, & Full-Scale `L=8, d=1024` Colab G4 Wave 19 (`900M` Tokens)

## 1. Executive Summary & Architectural Motivation

In **Plan 033**, we established three foundational results:
1. **Multi-Thousand-Step Scaling (`3,600` Steps / `29.5M` Tokens on Local `8 GB`)**: `QuantumBornGPT` with Full-Ambient $J_{\text{anti}}$ (`16×16 = 256`), Gentle `2x` Zero-WD Qubit LR (`0.006`), Bose-Saturated Log-Prior, and the Soft Power-Law Spectral Floor (`kraus_spectral_floor = 6.5`) reached **`4.9371` Val CE (`139.37` PPL, `25.54%` Top-1, $R_{\text{diff}} = 8.03$)** vs. `ClassicalModdedGPT` at **`5.5202` Val CE (`249.68` PPL, `20.06%` Top-1)** — widening its advantage from `-0.3060 nats` at Step 900 to **`-0.5831 nats` at Step 3,600**.
2. **Equi-Dimensional Square-$O(N)$ (`frame_expansion = 1.0`)**: Replacing rectangular $N \to 2N \to N$ frame projections with square $N \times N$ orthogonal operators achieved **`5.6434` Val CE** at **`22,265 tok/s` (`1.905x` faster than `frame_expansion = 2.0`)**, trailing `frame_expansion = 2.0` (`5.6092`) by only `0.034 nats` at 900 steps because independent square $O(N)$ beams lack a direct cross-beam mixing gate prior to phase interference.
3. **Why Wave 18 Colab (`d=512`) Trailed Wave 16 (`d=1024`) while Validating $J_{\text{anti}}$**: On `L=8, d=512` at `250M` tokens, Full-Ambient `16×32 = 512` $J_{\text{anti}}$ (`4.0556` Val CE) beat `16×16 = 256` Parity-Split (`4.0634` Val CE) at every checkpoint, while using only `26.4 GB` of the G4's `96 GB` VRAM (`d=512` vs. Wave 16's `d=1024` at `39.1 GB` VRAM).

In **Plan 034**, we explore a **radical new quantum architecture** on the local `8 GB` GPU alongside our **Mach-Zehnder Equi-Dimensional $O(N)$** improvement and our **Full-Capacity `L=8, d=1024` Wave 19 (`900M`-token) Colab G4 launch**:
- **Tied-$2^{16}$ ($N = 65,536$) Multi-Level Wavelet-Cholesky Quantum Amplitude Network**:
  Instead of treating the vocabulary ($V = 50,257$) and the internal model dimension ($d = 256$) as unrelated spaces with separate input and output tables, we embed language directly in an **$N = 2^{16} = 65,536$-dimensional (16-qubit) ambient Hilbert space** with:
  1. **Strict Input-Output Weight Tying on $\mathbb{R}^{2^{16}}$** (one shared multi-resolution basis $\hat{\Phi} \in \mathbb{R}^{2^{16} \times d}$ decoupled across input and output via the anti-diagonal spectral-reversal involution $J_{\text{anti}}$), and
  2. **Multi-Level Wavelet-Inspired Cholesky-Factorized Downsample & Upsample Operations in the Middle Layers** ($d \to d/2 \to d/4 \to d/8 \to d/4 \to d/2 \to d$), applying lower-triangular Cholesky factors $L_\ell D_\ell L_\ell^\top$ across dyadic wavelet scales to transform amplitudes with $O(d + \sum_\ell d_\ell^2 / 2)$ structured sparsity instead of dense $O(N^2)$ or $O(4 d^2)$ matrices.

---

## 2. Mathematical Formulation

### 2.1 Ambient $N = 2^{16} = 65,536$ Vocabulary Hilbert Space & Tied Wavelet-Cholesky Basis
Let $N = 2^{16} = 65,536$ be the 16-qubit ambient Hilbert space $\mathcal{H}_{16} = (\mathbb{R}^2)^{\otimes 16} \cong \mathbb{R}^{65536}$, where indices $v \in \{0, \dots, V-1\}$ ($V = 50,257$) represent FineWeb-Edu lexical tokens and indices $\{V, \dots, 65535\}$ represent `15,279` dark/ancilla reservoir modes that complete the exact dyadic power of two $2^{16}$.

To avoid $N^2 = 2^{32}$ dense parameters while coupling all $2^{16}$ basis states, we parameterize the **Tied $2^{16}$ Wavelet-Cholesky Basis** $\hat{\Phi}: \mathbb{R}^{2^{16}} \to \mathbb{R}^d$ ($d = 2^8 = 256$ locally, $d = 2^{10} = 1024$ on Colab) across coarse-to-fine dyadic scales:
1. **Dyadic Spectral-PPMI Tree Ordering & Product Qubit Leaf Amplitudes**:
   Each basis state $n \in \{0, \dots, 2^{16}-1\}$ has fine-scale product-register amplitudes $u_{\text{fine}}(n) \in \mathbb{R}^d$ ($Q \times A = d$, $16 \times 16 = 256$) plus a coarse dyadic wavelet-packet mother state $w_{\text{packet}}(\lfloor \pi(n) / 2^8 \rfloor) \in \mathbb{R}^d$ shared across its $2^8 = 256$-element dyadic wavelet subband (where $\pi \in S_{2^{16}}$ sorts tokens by their spectral-PPMI primary coordinate so semantically adjacent tokens share wavelet subbands).
2. **Cholesky Lower-Triangular Metric Coupling ($L_{\text{basis}} \in \mathbb{R}^{A \times A}$)**:
   Within each register $q \in \{1, \dots, Q\}$, a learnable **Cholesky lower-triangular factor** $L_q \in \mathbb{R}^{A \times A}$ with unit/positive diagonal ($\operatorname{diag}(L_q) = \operatorname{softplus}(\tilde{d}_q) > 0$, initialized to $I_A$) couples the wavelet amplitudes via the symmetric positive-definite Cholesky metric $M_q = L_q L_q^\top \succ 0$:
   $$\phi_q(n) = \frac{L_q \left(u_{\text{fine}, q}(n) + \gamma_{\text{wav}} \, w_{\text{packet}, q}(\lfloor \pi(n) / 2^8 \rfloor)\right)}{\left\|L_q \left(u_{\text{fine}, q}(n) + \gamma_{\text{wav}} \, w_{\text{packet}, q}(\lfloor \pi(n) / 2^8 \rfloor)\right)\right\|_2} \in S^{A-1}$$
3. **Strict Input-Output Tying With Mid-Network $J_{\text{anti}}$ Decoupling**:
   - **Tied Parameter Table**: Unlike standard `Hierarchical16QubitPOVMReadout` when untied or separate from token embeddings, the $N = 2^{16}$ Wavelet-Cholesky basis is **strictly tied (`100%` shared parameters)** between the initial token state preparation and the final POVM output readout!
   - **Input State Downsampling ($\mathbb{R}^{2^{16}} \to \mathbb{R}^d$)**: For input token $v_t \in \{0, \dots, V-1\}$, the initial state applies the anti-diagonal spectral-reversal involution $J_{\text{anti}} \in O(A)$ to the tied basis $\phi_q(v_t)$:
     $$|\psi_t^{(0)}\rangle = \frac{1}{\sqrt{Q}} \bigoplus_{q=1}^Q J_{\text{anti}} \, \phi_q(v_t) \in S^{d-1}$$
   - **Output Amplitude Upsampling ($\mathbb{R}^d \to \mathbb{R}^{2^{16}}$)**: At readout, the evolved state $|\psi_t^{(L)}\rangle \in S^{d-1}$ is upsampled back to the full $N = 2^{16} = 65,536$-dimensional Hilbert space using the **exact same tied basis** $\Phi = [\phi_1, \dots, \phi_Q] / \sqrt{Q} \in \mathbb{R}^{2^{16} \times d}$:
     $$c_{\text{sharp}, t}(n) = \langle \Phi(n) \mid \psi_t^{(L)} \rangle, \qquad n \in \{0, \dots, 2^{16}-1\}$$
     evaluated over the lexical slice $v \in \{0, \dots, V-1\}$ via our fused chunked Born-CE kernel! Because $\operatorname{Tr}(J_{\text{anti}}) = 0$ for even $A$ and $J_{\text{anti}}$ pairs singular mode $k$ with $A - 1 - k$, Step-0 self-overlap $\langle \Phi(v) \mid \psi_v^{(0)} \rangle$ is bounded by $\le 0.15$ despite `100%` weight tying!

### 2.2 Multi-Level Wavelet-Cholesky Downsample & Upsample Block (`WaveletCholeskyMultiscaleInterference`)
In the middle of the network (layers $\ell = 1, \dots, L$), instead of lifting $|\psi\rangle \in \mathbb{R}^d$ into an overcomplete rectangular space $\mathbb{R}^{4d}$, we introduce a **Multi-Level Wavelet-Cholesky V-Cycle (`WaveletCholeskyMultiscaleInterference`)** that recursively **downsamples** and **upsamples** the quantum amplitudes across $S = 3$ dyadic wavelet scales ($d \to d/2 \to d/4 \to d/8$, e.g. $256 \to 128 \to 64 \to 32$):

1. **Multi-Level Lifting Wavelet Analysis (Downsampling $\mathbb{R}^{d_{\ell-1}} \to \mathbb{R}^{d_\ell} \oplus \mathbb{R}^{d_\ell}$)**:
   Set $a^{(0)} = |\psi\rangle \in \mathbb{R}^d$. For each dyadic wavelet level $s = 1, \dots, S$ ($d_s = d / 2^s$):
   - Split the current approximation amplitude $a^{(s-1)} \in \mathbb{R}^{2 d_s}$ into even and odd sub-channels:
     $$x_e^{(s)} = a^{(s-1)}_{0::2} \in \mathbb{R}^{d_s}, \qquad x_o^{(s)} = a^{(s-1)}_{1::2} \in \mathbb{R}^{d_s}$$
   - Apply a **Sweldens Wavelet Lifting Step** (which is an exact $2 \times 2$ block-unit-Cholesky factorization $\begin{pmatrix} I & U_s \\ 0 & I \end{pmatrix} \begin{pmatrix} I & 0 \\ -P_s & I \end{pmatrix}$ with $\det = 1$) followed by Haar orthogonal normalization:
     $$d^{(s)} = \frac{1}{\sqrt{2}}\Big(\big(x_o^{(s)} - x_e^{(s)}\big) - P_s \odot x_e^{(s)}\Big) \in \mathbb{R}^{d_s} \quad \text{(Wavelet Detail Band at Scale } s\text{)}$$
     $$a^{(s)} = \frac{1}{\sqrt{2}}\Big(\big(x_e^{(s)} + x_o^{(s)}\big) + U_s \odot d^{(s)}\Big) \in \mathbb{R}^{d_s} \quad \text{(Downsampled Coarse Amplitude at Scale } s\text{)}$$
     where $P_s, U_s \in \mathbb{R}^{d_s}$ are learnable zero-initialized lifting predictors/updaters (so at Step 0 with $P_s = U_s = 0$, the analysis filterbank is the **exact orthonormal Haar wavelet transform** on $\mathbb{R}^d$!).

2. **Scale-Specific Cholesky-Factorized Linear Transformations ($L_s D_s L_s^\top$)**:
   At the coarsest scale $a^{(S)} \in \mathbb{R}^{d_S}$ ($d_S = d/8 = 32$) and each detail band $d^{(s)} \in \mathbb{R}^{d_s}$ ($d_s \in \{32, 64, 128\}$), we apply a **Cholesky-Factorized Linear Transformation**:
   - At coarse scale $S$ ($d_S = 32$) and mid scale $S-1$ ($d_{S-1} = 64$), we parameterize lower-triangular Cholesky factors $L_{\text{coarse}} \in \mathbb{R}^{d_S \times d_S}$ and $L_{\text{detail}, s} \in \mathbb{R}^{d_s \times d_s}$ (strictly lower-triangular $\operatorname{tril}(W, -1)$ plus positive diagonal $\operatorname{diag}(L) = \operatorname{softplus}(\delta) > 0$) and apply the symmetric Cholesky operator $C = L L^\top$:
     $$\tilde{a}^{(S)} = L_{\text{coarse}} \big(L_{\text{coarse}}^\top a^{(S)}\big), \qquad \tilde{d}^{(s)} = L_{\text{detail}, s} \big(L_{\text{detail}, s}^\top d^{(s)}\big)$$
   - For the finest detail band $s = 1$ ($d_1 = d/2 = 128$), we use a **block-banded Cholesky factor** of block size $b = 32$ ($4$ blocks of $32 \times 32$ lower-triangular Cholesky factors), keeping total Cholesky parameters across all scales to just:
     $$N_{\text{chol}} = \underbrace{32^2}_{\text{coarse }32} + \underbrace{32^2}_{\text{detail }32} + \underbrace{64^2}_{\text{detail }64} + \underbrace{4 \times 32^2}_{\text{banded detail }128} = 10,240\text{ parameters per layer!}$$
     Compare $10,240$ parameters to $3 \times 256 \times 512 = 393,216$ parameters in `OvercompleteTwoBeamInterference` (`38x` fewer parameters in the multiscale wavelet-Cholesky branch!).

3. **Multi-Level Wavelet Synthesis (Upsampling $\mathbb{R}^{d_S} \oplus \bigoplus_s \mathbb{R}^{d_s} \to \mathbb{R}^d$)**:
   Starting from the transformed coarse amplitude $\hat{a}^{(S)} = \tilde{a}^{(S)} \in \mathbb{R}^{d_S}$, we recursively **upsample** from $s = S$ down to $s = 1$ using the exact inverse lifting synthesis steps plus learnable scale gates $\beta_s$:
   $$\hat{x}_e^{(s)} = \frac{1}{\sqrt{2}}\Big(\hat{a}^{(s)} - \beta_s \hat{d}^{(s)}\Big) - \frac{1}{2} \tilde{U}_s \odot \hat{d}^{(s)}, \qquad \hat{x}_o^{(s)} = \frac{1}{\sqrt{2}}\Big(\hat{a}^{(s)} + \beta_s \hat{d}^{(s)}\Big) + \frac{1}{2} \tilde{P}_s \odot \hat{x}_e^{(s)}$$
   and interleave $(\hat{x}_e^{(s)}, \hat{x}_o^{(s)}) \to \hat{a}^{(s-1)} \in \mathbb{R}^{2 d_s}$ until reaching full resolution $\hat{a}^{(0)} \in \mathbb{R}^d$!

### 2.3 Mach-Zehnder Coupled-Beamsplitter Equi-Dimensional $O(N) \times O(N)$ Interference
To make the strictly $N$-dimensional (`frame_expansion = 1.0`) two-beam architecture match or exceed overcomplete `2N` (`frame_expansion = 2.0`) while retaining its `1.9x` throughput speedup, we add a **Learnable Mach-Zehnder $SO(2) \otimes O(N)$ Beamsplitter** (`use_mach_zehnder_beamsplitter = True`) and optional **Multi-Level Wavelet-Cholesky Parallel Branch** (`use_wavelet_cholesky_branch = True`):
1. Given the two $N$-dimensional square orthogonal projections $z_1 = \sqrt{N}\, U_1 |\psi_t\rangle \in \mathbb{R}^N$ and $z_2 = \sqrt{N}\, U_2 |\psi_t\rangle \in \mathbb{R}^N$ ($U_1, U_2 \in O(N)$), a learnable per-coordinate beamsplitter angle $\theta_{\text{bs}} \in \mathbb{R}^N$ (initialized to $\pi / 8$) rotates the two $N$-dimensional beams prior to phase interference:
   $$\begin{pmatrix} \tilde{z}_1 \\ \tilde{z}_2 \end{pmatrix} = \begin{pmatrix} \operatorname{diag}(\cos\theta_{\text{bs}}) & -\operatorname{diag}(\sin\theta_{\text{bs}}) \\ \operatorname{diag}(\sin\theta_{\text{bs}}) & \operatorname{diag}(\cos\theta_{\text{bs}}) \end{pmatrix} \begin{pmatrix} z_1 \\ z_2 \end{pmatrix} \in \mathbb{R}^N \times \mathbb{R}^N$$
2. When `use_wavelet_cholesky_branch = True`, the multi-scale wavelet-Cholesky downsample/upsample operator $\mathcal{ M }_{\text{wav-chol}}(|\psi_t\rangle) \in \mathbb{R}^N$ injects multi-resolution coarse-to-fine amplitude modulation directly into the interference packet before spherical normalization onto $S^{N-1} \times S^{N-1}$.

---

## 3. Observable Hypotheses & Quantitative Predictions

### Hypothesis H1 — Exact Energy Preservation & Parameter Efficiency of the Tied-$2^{16}$ Multi-Level Wavelet-Cholesky Operator
- **Claim**:
  1. At initialization ($P_s = U_s = 0, L_\ell = I$), the 3-level Wavelet-Cholesky Downsample $\to$ Upsample V-Cycle ($256 \to 128 \to 64 \to 32 \to 64 \to 128 \to 256$) is an **exact isometry** on $\mathbb{R}^{256}$ (`reconstruction_max_err < 1e-5`, `energy_ratio_err < 1e-5`), and remains differentiable with positive-definite Cholesky factors ($\min \operatorname{diag}(L_\ell) > 0$) under gradient descent.
  2. The **Tied-$2^{16}$ ($N = 65,536$) Wavelet-Cholesky Basis** uses **zero separate output embedding tables** (`100%` weight tying between input preparation and $N = 2^{16}$ output amplitudes, sliced to $V = 50,257$ lexical tokens) while bounding Step-0 self-overlap to `<= 0.15` via mid-network $J_{\text{anti}}$ decoupling.

### Hypothesis H2 — Mach-Zehnder Coupled-Beamsplitter + Wavelet-Cholesky Equi-Dimensional $O(N)$ (`frame_expansion = 1.0`) Closes the Overcomplete Gap While Maintaining `> 1.6x` Higher Throughput (`900` Steps, `L=4, d=256`, `<= 8.0 GB` VRAM)
- **Claim**:
  On genuine FineWeb-Edu (`L=4, d=256, H=4`, `900` steps = `7.37M` tokens, evaluated on held-out `val_slice.bin` at $T_{\text{val}} = 512$):
  1. Adding the **Mach-Zehnder $SO(2) \otimes O(N)$ Coupled Beamsplitter + Multi-Level Wavelet-Cholesky Downsample/Upsample Branch** to the strictly $N=256$-dimensional Square-$O(N)$ model (`frame_expansion = 1.0`) improves validation cross-entropy over uncoupled Square-$O(N)$ (`5.6434` Val CE in Plan 033) while maintaining **`>= 18,000 tok/s` (`> 1.5x` faster than `Isolate 3`'s `11,690 tok/s`)** and fewer parameters (`< 39.5M` vs. `40.78M`).
  2. The pure **Tied-$2^{16}$ Wavelet-Cholesky Multiscale Architecture** (`frame_expansion = 1.0` with multi-level wavelet-Cholesky downsample/upsample in every layer and tied $2^{16}$ wavelet-Cholesky basis) converges cleanly below `< 5.64` Val CE at Step 900.

### Hypothesis H3 — `3,600`-Step (`29.5M`-Token) Local `8 GB` Scaling of Tied-$2^{16}$ Wavelet-Cholesky + Mach-Zehnder Equi-Dimensional $O(N)$ Beats `ClassicalModdedGPT` (`5.5202`) by `> 0.55 nats` and Breaks `< 4.95` Val CE in ~Half the Wall-Clock Time
- **Claim**:
  Over `3,600` steps (`29.49M` FineWeb-Edu tokens) on the local RTX 5060 Ti (`<= 8.0 GB` VRAM cap), the **Tied-$2^{16}$ Wavelet-Cholesky + Mach-Zehnder Equi-Dimensional $O(N)$** model (`w19_8gb_L4_d256_wavelet_cholesky_mach_zehnder_ON_3600steps`):
  1. Outperforms `ClassicalModdedGPT` (`5.5202` Val CE) at every checkpoint (`Steps 300, 900, 1800, 2700, 3600`), achieving **`< 4.9500` Val CE** (`>= 25.0%` Top-1 accuracy, $R_{\text{diff}} \ge 6.5$) while running **`>= 1.35x` faster in wall-clock time** and using fewer parameters (`< 39.5M` vs. `40.78M`) than overcomplete `frame_expansion = 2.0`.

---

## 4. Planned Changes & Best-Practices Enforcement (`docs/SCRIPT_BEST_PRACTICES.md`)

1. **[`src/qdlblocks/layers/wavelet_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky.py) (NEW, `< 280` lines, `<= 8` args/fn)**:
   - Implement `WaveletCholeskyConfig` (`@dataclass(frozen=True)`) and `WaveletCholeskyMultiscaleBranch(nn.Module)`:
     - Multi-level Sweldens lifting wavelet analysis (`downsample_amplitudes`: $\mathbb{R}^d \to \{\mathbb{R}^{d/2}, \mathbb{R}^{d/4}, \mathbb{R}^{d/8}\}$),
     - Scale-wise lower-triangular Cholesky-factorized linear operators $C_s = L_s L_s^\top$ (`apply_cholesky_scale_operators`),
     - Multi-level lifting wavelet synthesis (`upsample_amplitudes`: $\{\mathbb{R}^{d/8}, \mathbb{R}^{d/4}, \mathbb{R}^{d/2}\} \to \mathbb{R}^d$),
     - And `TiedWaveletCholeskyBasis2Pow16(nn.Module)` providing the $N = 2^{16} = 65,536$ dyadic wavelet-packet + Cholesky-coupled basis tied to the $V = 50,257$ lexical POVM outcomes.
2. **[`src/qdlblocks/layers/overcomplete_frame.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_frame.py) & [`src/qdlblocks/layers/hierarchical_qubit.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/hierarchical_qubit.py)**:
   - Add `use_mach_zehnder_beamsplitter: bool = False` and `use_wavelet_cholesky_branch: bool = False` to `OvercompleteTwoBeamInterference` (so `frame_expansion = 1.0` applies the learnable $SO(2) \otimes O(N)$ beamsplitter rotation and multi-scale wavelet-Cholesky downsample/upsample branch).
   - Add `use_wavelet_cholesky_basis: bool = False` (with $N = 2^{16} = 65,536$ dyadic wavelet-packet + register-Cholesky $L_q L_q^\top$ coupling) to `Hierarchical16QubitPOVMReadout`.
3. **[`src/qdlblocks/benchmarks/config_models.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py) (`< 400` lines, `<= 8` args/fn)**:
   - Extend `BackboneConfig` and `QubitHierarchyConfig` with `use_mach_zehnder_beamsplitter`, `use_wavelet_cholesky_branch`, and `use_wavelet_cholesky_basis`.
4. **[`examples/44_verify_plan034_wavelet_cholesky_and_mach_zehnder.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/44_verify_plan034_wavelet_cholesky_and_mach_zehnder.py) (NEW, `< 390` lines, `<= 8` args/fn)**:
   - Executes H1, H2, and H3 on genuine FineWeb-Edu tokens (`train_full_100M.bin` and `val_slice.bin`), registers all runs in SQLite + Model Cards, and writes `.cache/hypothesis_experiments_034.json`.
5. **[`configs/wave19_fleet.json`](file:///home/festivity/agy/QuantumDLBlocks/configs/wave19_fleet.json) & [`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py)**:
   - Wire `--mach-zehnder-beamsplitter`, `--wavelet-cholesky-branch`, and `--wavelet-cholesky-basis` into `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`.
   - Launch **Wave 19 (`900M` FineWeb-Edu tokens)** on the 2 Google Colab G4 (`96 GB` VRAM) sessions using the full **`L=8, d=1024` Wave 16 champion backbone**:
     - **`gpu_g4_w6_1` (`w19_900m_qborn_d1024_16x64_janti_soft_floor_L8`)**: `L=8, d=1024, H=16, seed_dim=512, frame_expansion=4.0, VE=3, 262k N-gram, muon_lr=0.030, wsd_decay=0.25` with **Full-Ambient `16×64 = 1024` $J_{\text{anti}}$ Qubits** + Gentle `2x` Zero-WD Qubit LR (`0.006`) + Bose-Saturated Log-Prior + Soft Power-Law Spectral Floor (`6.5`).
     - **`gpu_g4_w6_2` (`w19_900m_qborn_d1024_wavelet_cholesky_mach_zehnder_ON_L8`)**: `L=8, d=1024, H=16, seed_dim=1024, frame_expansion=1.0` **Equi-Dimensional Square-$O(1024)$ + Mach-Zehnder Coupled Beamsplitter + Multi-Level Wavelet-Cholesky Downsample/Upsample (`1024 -> 512 -> 256 -> 128`) + Tied-$2^{16}$ Wavelet-Cholesky Basis (`16×64 = 1024` $J_{\text{anti}}$)**.

---

## 5. Operational Tooling & Script Opportunities Gate (Section 5 Compliance)

### 5.1 Existing Scripts Used
1. **Model Parameter & Smoke Verification (`scripts/inspect_model.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py --num-layers 8 --embed-dim 1024 --num-heads 16 --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 64 --num-higher-states 256 --two-beam-kboson-readout --stereographic-lift --quadratic-diffuse-fidelity 0.50 --smoke-steps 2
   ```
2. **Colab 2-G4 Fleet Launch & Status (`scripts/colab_fleet.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py launch --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave19_fleet.json
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py status --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave19_fleet.json
   ```
3. **Zero-Downtime Live Monitoring Daemon (`scripts/monitor_colab_runs.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once
   ```

### 5.2 Build-Before-Launch & Promoted Backlog Scripts (Implemented in Plan 034)
1. **`scripts/ast_signature_and_length_linter.py` (Promoted from `scripts/BACKLOG.md` Plan 033)**:
   - Standalone CLI utility (`< 250` lines, `<= 8` args/fn) that audits Python files against `docs/SCRIPT_BEST_PRACTICES.md` (`<= 400` target / `<= 600` ceiling lines, `<= 8` arguments per function, zero giant nested closures) and prints a Markdown compliance table.
2. **`scripts/wavelet_cholesky_spectrum_probe.py` (Promoted / Extended from `scripts/BACKLOG.md` Plan 033)**:
   - Standalone CLI utility (`< 280` lines, `<= 8` args/fn) that audits multi-scale wavelet energy distribution ($E_{\text{coarse}}, E_{\text{detail}, 1..S}$), Cholesky condition numbers $\kappa(L_s L_s^\top)$, reconstruction error, and $N = 2^{16}$ tied basis self-overlap on CPU/GPU in `< 3s`.

### 5.3 Nice-to-Have Scripts (Logged to `scripts/BACKLOG.md`)
1. **`scripts/mera_entanglement_cone_visualizer.py`**: Computes the causal entanglement light-cone and mutual information across dyadic wavelet scales $s \in \{1, \dots, S\}$ for any trained `WaveletCholeskyMultiscaleBranch` checkpoint.
2. **`scripts/vocab_dark_ancilla_utilization_audit.py`**: Measures how much probability mass or virtual interference flows through the `15,279` dark ancilla basis states ($50257 \dots 65535$) when $N = 2^{16} = 65,536$.
