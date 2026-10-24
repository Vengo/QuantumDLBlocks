# Experiment Plan 035 — Standalone Radical $N = 2^{16}$ (`65,536`-D) Wavelet-Cholesky Amplitude Architectures: Multi-Resolution Dyadic U-Net Pyramid vs. Consistent-Space $256 \otimes 256$ Kronecker-Wavelet Network

## 1. Executive Summary & Architectural Motivation

In **Plan 034**, we tested a hybrid Wavelet-Cholesky branch inside the existing `QuantumBornGPT` scaffold (`L=4, d=256` locally and `L=8, d=1024` on Colab). Because that hybrid model still retained the legacy `50,304 × d` dense `unigram_qubits` table (`12.88M`–`51.5M` parameters), hashed n-gram table (`16.78M`–`67.1M` parameters), and standard `QuantumBornGPT` block structure, its training curve followed the same trajectory family as prior waves.

In **Experiment Plan 035**, we strip away the legacy `QuantumBornGPT` scaffold and build a clean, standalone module ([`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py), strictly `< 400` lines and `<= 8` arguments per function per [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md)) to test **both** radical $N = 2^{16} = 65,536$ architectural paradigms locally on our `<= 8.0 GB` RTX 5060 Ti:

1. **16-Bit Dyadic Spectral Tree Vocabulary Ordering (`build_dyadic_2pow16_spectral_tree`)**:
   - Raw GPT-2 BPE token IDs ($0 \dots 50256$) are ordered by merge history rather than semantics, making adjacent indices $(2k, 2k+1)$ unrelated under a naive wavelet transform.
   - We construct a deterministic **16-Level Balanced Binary Spectral Tree** that maps the $V = 50,257$ lexical tokens plus $15,279$ dark ancilla modes into the exact power-of-two Hilbert space $N = 2^{16} = 65,536$ via recursive spectral bisection of the PPMI-SVD token vectors.
   - Under the resulting bijection $\pi: \{0, \dots, 65535\} \to \{0, \dots, 65535\}$, any prefix of $k$ bits $\pi(v) \gg (16 - k) \in \{0, \dots, 2^k - 1\}$ identifies a coherent semantic cluster at dyadic scale $2^k$:
     - Top 8 bits $i_v = \pi(v) \gg 8 \in \{0, \dots, 255\}$: **256 macro semantic clusters**.
     - Bottom 8 bits $j_v = \pi(v) \ \& \ 255 \in \{0, \dots, 255\}$: **256 fine intra-cluster positions**.

2. **Architecture 1 — `MultiScaleWaveletCholeskyPyramidGPT` (Multi-Resolution $2^{16}$ Wavelet-Cholesky Boundary Pyramid + Middle U-Net Downsample/Upsample Bottleneck)**:
   - **Tied $N = 2^{16}$ Multi-Resolution Wavelet-Cholesky Basis (`~4.9M` Params vs. `25.8M` Standard Untied)**:
     Replaces dense $V \times d$ input and output tables with a 4-level tapered dyadic wavelet-Cholesky synthesis pyramid over $\mathbb{R}^{2^{16}}$:
     - Scale 3 ($2^8 = 256$ super-coarse scaling states in $\mathbb{R}^{256}$) + Cholesky $L_3 L_3^\top$
     - Scale 2 ($2^{11} = 2,048$ coarse wavelet packets in $\mathbb{R}^{256}$) + Cholesky $L_2 L_2^\top$
     - Scale 1 ($2^{13} = 8,192$ mid wavelet packets in $\mathbb{R}^{128} \xrightarrow{U_1} \mathbb{R}^{256}$) + Cholesky $L_1 L_1^\top$
     - Scale 0 ($2^{16} = 65,536$ fine leaf details in $\mathbb{R}^{48} \xrightarrow{U_0} \mathbb{R}^{256}$) + Cholesky $L_0 L_0^\top$
     Initialized via exact coarse-to-fine multi-resolution SVD residual decomposition of the spectral-ordered PPMI-SVD vectors.
   - **Strict Input-Output Parameter Tying with Anti-Diagonal Cholesky Bridge**:
     The output detector $\Phi_{\text{out}}[\pi(v)]$ uses the **exact same tied $N = 2^{16}$ wavelet pyramid** coupled through a compact $d \times d$ skew-involutive Cholesky bridge $J_{\text{anti}} L_{\text{bridge}} L_{\text{bridge}}^\top$ and a $256 \times 256$ coarse-cluster transition matrix $T_{\text{coarse}}$ (`0.13M` params total, zero untied $V \times d$ parameters, exact `0.0000` Step-0 self-overlap).
   - **Middle U-Net Downsample/Upsample Layers ($256 \to 128 \to 64 \to 32 \to 64 \to 128 \to 256$)**:
     Each layer downsamples amplitudes through 3 Sweldens lifting wavelet levels, applies positive-definite lower-triangular Cholesky operators $C_{\ell, s} = L_{\ell, s} L_{\ell, s}^\top$ at every scale, performs **causal sequence mixing (causal token-shift + attention gating) inside the compressed coarse wavelet subbands ($64$ and $32$)** as well as the ambient stream, and upsamples back to $d=256$.

3. **Architecture 2 — `Consistent2Pow16WaveletCholeskyGPT` (Consistent $N = 2^{16} = 256_{\text{coarse}} \otimes 256_{\text{fine}}$ Tensor-Wavelet-Cholesky Network All the Way Through)**:
   - **Consistent $N = 2^{16} = 65,536$ Hilbert Space Representation**:
     Factors $\mathcal{H} = \mathbb{R}^{2^{16}} = \mathbb{R}^{65,536} \cong \mathbb{R}^{256_{\text{coarse}}} \otimes \mathbb{R}^{256_{\text{fine}}}$ along the top 8 bits $i_v \in [0, 256)$ and bottom 8 bits $j_v \in [0, 256)$ of $\pi(v) = 256 \, i_v + j_v$. Every layer $\ell \in \{0, \dots, L\}$ evolves an explicit coupled coarse-fine state $(\psi_C^{(\ell)}, \psi_F^{(\ell)}) \in S^{255}_{\text{coarse}} \times S^{255}_{\text{fine}}$ representing a state in $\mathbb{R}^{256 \times 256} = \mathbb{R}^{65,536}$.
   - **2D Dyadic Wavelet Downsample/Upsample & Coarse$\leftrightarrow$Fine Mach-Zehnder Entangling Gates**:
     In every middle layer, both factors jointly downsample across $(256 \otimes 256 = 65,536) \to (128 \otimes 128 = 16,384) \to (64 \otimes 64 = 4,096) \to (32 \otimes 32 = 1,024)$, undergo scale-specific lower-triangular Cholesky transformations $(L_{C, s} L_{C, s}^\top, L_{F, s} L_{F, s}^\top)$ and **Mach-Zehnder $SO(2)$ Coarse$\leftrightarrow$Fine Entangling Beamsplitters** at every dyadic scale $s \in \{0, 1, 2, 3\}$, mix causally across sequence positions, and upsample back to $256 \otimes 256 = 65,536$.
   - **Tied $N = 2^{16} = 256 \times 256$ Kronecker + Wavelet Readout**:
     Readout directly combines the $256$-state coarse cluster inner product $A_C(t, i_v)$, the fine intra-cluster wavelet inner product $A_F(t, v)$, and their multiplicative Kronecker amplitude $A_C(t, i_v) \cdot A_F(t, v)$ across all $65,536$ coordinates using Two-Beam $k$-Boson Mixed-State Readout with Soft Spectral Floor (`6.5`).

---

## 2. Concrete Hypotheses & Quantitative Success Criteria

### Hypothesis H1 — 16-Bit Dyadic Spectral Tree Coherence & Multi-Resolution $2^{16}$ Wavelet-Cholesky Compression
- **Claim**:
  1. `build_dyadic_2pow16_spectral_tree` produces an exact bijection $\pi: \{0, \dots, 65535\} \to \{0, \dots, 65535\}$ where the mean intra-block cosine similarity of PPMI-SVD vectors inside $8$-leaf ($2^{13}$) and $256$-leaf ($2^8$) spectral wavelet blocks is at least **`2.0x` higher** than under raw BPE token ordering.
  2. Both standalone $N = 2^{16}$ architectures (`MultiScaleWaveletCholeskyPyramidGPT` and `Consistent2Pow16WaveletCholeskyGPT`) have **`< 11.0M` total parameters** (at least **`3.5x` smaller** than `QuantumBornGPT`'s `40.78M` and **`2.8x` smaller** than `ClassicalModdedGPT`'s `30.89M`), zero untied $V \times d$ output tables, exact Step-0 base self-overlap `< 0.05`, and finite gradients across all wavelet scales.

### Hypothesis H2 — 900-Step (`7.37M`-Token) Local `8 GB` Head-to-Head Comparison of Both Radical $N = 2^{16}$ Architectures
- **Claim**: Over `900` steps (`7,372,800` genuine FineWeb-Edu tokens, `micro_bs=8, grad_accum=2, seq_len=512`, `<= 8.0 GB` VRAM):
  1. Both standalone $N = 2^{16}$ Wavelet-Cholesky architectures (`w20_8gb_arch1_2pow16_multiscale_wavelet_cholesky_unet_900steps` and `w20_8gb_arch2_consistent_2pow16_kronecker_wavelet_cholesky_900steps`) train with zero NaNs, healthy diffuse Kraus rank $R_{\text{diff}} \ge 6.0$, and beat `ClassicalModdedGPT` (`5.9152` Val CE at Step 900, despite having `> 3x` fewer parameters!).
  2. Both architectures achieve `>= 18,000 tok/s` throughput within `< 4.5 GB` peak VRAM on the local RTX 5060 Ti.

### Hypothesis H3 — 3,600-Step (`29.49M`-Token) Local `8 GB` Scaling Verification of the Standalone $N = 2^{16}$ Wavelet-Cholesky Architecture
- **Claim**: Over a full `3,600`-step (`29,491,200`-token) local `8 GB` scaling run on genuine FineWeb-Edu (`train_full_100M.bin` and held-out `val_slice.bin`), the standalone $N = 2^{16}$ Wavelet-Cholesky model (`< 11M` total parameters) achieves `< 5.35` Val CE and beats the `30.89M`-parameter `ClassicalModdedGPT` baseline (`5.5202` Val CE) at every evaluation milestone (`900, 1800, 2700, 3600` steps) using **`< 1/3` the parameter budget**.

---

## 3. Planned Code & Module Changes (`docs/SCRIPT_BEST_PRACTICES.md` Compliant)

1. **[`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py) (New File, `< 400` Lines, `<= 8` Args/Fn)**:
   - `@dataclass(frozen=True) class Radical2Pow16Config`: Configures `arch_mode: Literal["multiscale_unet", "consistent_kronecker"]`, `vocab_size: int = 50304`, `ambient_n: int = 65536`, `num_layers: int = 4`, `embed_dim: int = 256`, `num_heads: int = 4`, `block_size: int = 512`, `window_size: int = 384`, `kraus_spectral_floor: float = 6.5`.
   - `build_dyadic_2pow16_spectral_tree(ppmi_embeddings: torch.Tensor, ambient_n: int = 65536) -> tuple[torch.Tensor, torch.Tensor]`: Fast vectorized 16-level balanced binary spectral bisection mapping $\{0, \dots, V-1\}$ into $N = 2^{16} = 65,536$ leaves.
   - `Tied2Pow16WaveletCholeskyPyramid(nn.Module)`: 4-scale ($2^8, 2^{11}, 2^{13}, 2^{16}$) tapered dyadic wavelet-Cholesky basis with per-scale lower-triangular Cholesky operators $L_s L_s^\top$, exact multi-resolution PPMI-SVD initialization, and tied anti-diagonal Cholesky readout bridge.
   - `WaveletCholeskyUNetBlock(nn.Module)` (Architecture 1 middle layer) and `ConsistentKroneckerWaveletBlock(nn.Module)` (Architecture 2 middle layer on $\mathbb{R}^{256_{\text{coarse}}} \otimes \mathbb{R}^{256_{\text{fine}}} = \mathbb{R}^{65,536}$).
   - `Radical2Pow16WaveletCholeskyGPT(nn.Module)`: Unified standalone model supporting both `arch_mode="multiscale_unet"` (Architecture 1) and `arch_mode="consistent_kronecker"` (Architecture 2) with Two-Beam $k$-Boson Mixed-State Readout and Soft Power-Law Spectral Floor.
2. **[`examples/45_verify_plan035_radical_2pow16_wavelet_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/45_verify_plan035_radical_2pow16_wavelet_cholesky.py) (New File, `< 390` Lines, `<= 8` Args/Fn)**:
   - Verifies H1 (`build_dyadic_2pow16_spectral_tree` intra-block coherence gain, parameter compression `< 11M`, Step-0 self-overlap, finite gradients).
   - Runs H2 (`900` steps of Architecture 1 and Architecture 2 on genuine FineWeb-Edu) and H3 (`3,600` steps / `29.49M` tokens of the leading standalone $N = 2^{16}$ architecture) on `cuda:0` (`<= 8.0 GB` VRAM), registering all runs in `benchmarks.sqlite` and `.cache/hypothesis_experiments_035.json`.
3. **[`tests/test_hypothesis_035_radical_2pow16_wavelet_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_035_radical_2pow16_wavelet_cholesky.py)**:
   - Unit and integration tests covering `build_dyadic_2pow16_spectral_tree`, both standalone $N = 2^{16}$ architectures, promoted CLI scripts, AST compliance, and empirical verification results.

---

## 5. Operational Tooling & Script Opportunities (Mandatory Gate)

### 5.1 Existing Scripts Used
1. **AST Signature & Length Linter (`scripts/ast_signature_and_length_linter.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py examples/45_verify_plan035_radical_2pow16_wavelet_cholesky.py scripts/vocab_dark_ancilla_utilization_audit.py scripts/mera_entanglement_cone_visualizer.py
   ```
2. **Wavelet-Cholesky Spectrum Probe (`scripts/wavelet_cholesky_spectrum_probe.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/wavelet_cholesky_spectrum_probe.py --embed-dim 256 --num-levels 3 --device cpu
   ```

### 5.2 Build-Before-Launch & Promoted Backlog Scripts (Implemented in Plan 035)
1. **`scripts/vocab_dark_ancilla_utilization_audit.py` (Promoted from `scripts/BACKLOG.md` Plan 034)**:
   - Standalone CLI utility (`< 250` lines, `<= 8` args/fn) that audits the $N = 2^{16} = 65,536$ Dyadic Spectral Tree ordering (`50,257` lexical tokens + `15,279` dark ancilla coordinates), reporting intra-block cosine coherence across dyadic scales ($2^8, 2^{11}, 2^{13}, 2^{16}$) and dark ancilla amplitude energy fraction.
2. **`scripts/mera_entanglement_cone_visualizer.py` (Promoted from `scripts/BACKLOG.md` Plan 034)**:
   - Standalone CLI utility (`< 250` lines, `<= 8` args/fn) that probes causal entanglement light-cones, multi-scale wavelet energy retention, and coarse$\leftrightarrow$fine Kronecker subband coupling across `multiscale_unet` (Arch 1) and `consistent_kronecker` (Arch 2).

### 5.3 Nice-to-Have Scripts (Logged to `scripts/BACKLOG.md`)
1. **`scripts/dyadic_spectral_tree_inspector.py`**: Prints human-readable GPT-2 token clusters at each node of the 16-bit Dyadic Spectral Tree ($2^8 = 256$ macro clusters down to $2$-token sibling leaves) to inspect semantic renormalization structure.
2. **`scripts/cholesky_band_sparsity_profiler.py`**: Benchmarks wall-clock speed and condition number $\kappa(L_s L_s^\top)$ across block-diagonal vs. banded lower-triangular Cholesky factorizations on consumer GPUs.
