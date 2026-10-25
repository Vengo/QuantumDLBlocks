# QuantumDLBlocks (`qdlblocks` & `qdlblocks_jax`)

**Density-Matrix, Complex Hilbert-Space, and Dyadic Wavelet-Cholesky / Parseval $\mu\text{P}$ Quantum-Geometric Deep Learning Blocks in Native PyTorch and JAX/Haliax.**

`qdlblocks` provides modular PyTorch layers, reusable fused GPU/TPU kernels (`qdlblocks.kernels`), multi-scale Sweldens Wavelet-Cholesky & Mach-Zehnder interferometric blocks (`qdlblocks.layers`), flag-free Parseval $\mu\text{P}$ $\sqrt{d}$ language and vision models (`qdlblocks.models`), Riemannian Stiefel manifold optimizers (`qdlblocks.optim`), pure JAX + Equinox + Haliax named-axis modules (`qdlblocks_jax`), and standardized iso-token / iso-FLOP benchmarks (`qdlblocks.benchmarks`)—without the overhead of gate-level quantum circuit simulators.

> **Start Here — Authoritative Building-Blocks & Engineering Guides**:
> - **[`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md)**: Complete catalog of our canonical reusable GPU kernels (`fused_parseval_ce`, `flash_born_triton`), layers (`parseval_mup_dyadic_chunked_ce`, `BlockCholeskyLinear`, `WaveletCholeskyMultiscaleBranch`, `OvercompleteTwoBeamInterference`), models (`ParsevalMuPDyadicGPT`, `WaveletCholeskyVisionClassifier`), optimizers (`StiefelMuon`), diagnostic CLI scripts, and how to contribute new primitives back.
> - **[`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md)**: The 6 mandatory engineering rules enforced across the repo via [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) (including Rule 6: clean, flag-free per-experiment modules composing shared building blocks).
> - **[`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md)**: Operational workflow for live monitoring, first-class scripts, and mandatory building-block reuse.

---

## Current Canonical Building Blocks & `Waves 20–34` Highlights

### 1. Reusable Fused GPU Kernels & Quantum-Geometric Primitives (See [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md))
| Building Block | Module | What It Does & Why It Exists |
| :--- | :--- | :--- |
| **Fused Parseval & $M=4$ Coincidence Chunked CE Kernels (`Waves 28–34`)** | [`qdlblocks.kernels.fused_parseval_ce`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py) | Combines Ada/Blackwell `FP8 E4M3` (`torch._scaled_mm`) / `BF16` Tensor-Core GEMMs with `@torch.compile` (TorchInductor) kernels (`compiled_parseval_convex_traceless_chunk_step`, `compiled_m4_coincidence_chunk_step`) for centered sharp & diffuse observables and true 50,257-way $M=4$ nested-sqrt geometric coincidence. |
| **Flag-Free Parseval $\mu\text{P}$ $\sqrt{d}$ Readout** | [`qdlblocks.layers.parseval_mup_chunked_ce`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py) | Replaces dynamic per-token `PurityGate` MLPs, redundant `kboson_raw_k` counters, and manual clamps (`26.8` / `44.0`) with deterministic Parseval frame scaling $\beta_0(d) = c^*\sqrt{d}$ ($c^* = 1.48$), Convex-Peak Traceless Born observable $\widehat{O}_{t,v} = \hat{c}_{t,v} + \gamma(\max(\hat{c}_{t,v},0)^2 - \tau_{t,+})$, and 256-root dyadic coarse-to-fine conditional probability. |
| **Symplectic Product-of-Spheres Coincidence Born (`Waves 31–34`)** | [`qdlblocks.layers.subspace_coincidence_born`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | Enforces pair-strided symplectic $M=4$ Product-of-Spheres $(\frac{1}{\sqrt{M}}\mathbb{S}^{d/M-1})^M$ normalization (`R_sharp(0) = 67.14` at $d=256$, `188.97` at $d=1024$) and progress-ramped 50,257-way coincidence gating (`coincidence_gamma(tau) = 0.22 * tau`), reducing the true 4-shell dyadic CE floor by `5.85x–15.06x`. |
| **Per-Layer Pre-Attention `MultiRateWavefunctionConvBank` & $d=1024$ Scaling (`Waves 32–34`)** | [`qdlblocks.models.level_separated_wavelet_conv_gpt`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) | Unifies 1D strided causal temporal convolution `(4, 2, 1, 1)`, `PreAttnMultiRateWaveletConvBlock` (`V^(0)` anchor + purified Flash-Born `hd=64`), causal surface N-gram projection, and 2D+3D `StiefelMuon` + Parseval $\mu\text{P}$ `AdamW`—reaching **`5.0976` Val CE in `1.23M` tokens** (`150` steps, $d=256$), **`4.8332` Val CE / `26.57%` Top-1 at `14.75M` tokens** (`1,800` steps, $d=256$), and scaling to **`d=1024, L=6` (`119.99M` params) in `7.94 GiB` VRAM** (`5.0827` Val CE in `2.46M` tokens) on a 16 GB GPU. |
| **Batched SPD `BlockCholeskyLinear` & Sweldens Wavelets** | [`qdlblocks.layers.wavelet_cholesky_branch`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky_branch.py) | Parameterizes strictly positive-definite metric operators $A = \operatorname{diag}(L_g L_g^\top) \succ 0$ via batched `torch.bmm` (`fast_spd_bmm=True`, **`2.55x` faster**) inside multi-level invertible Sweldens lifting wavelet branches (`WaveletCholeskyMultiscaleBranch`). |
| **Mach-Zehnder Two-Beam Interferometer** | [`qdlblocks.layers.overcomplete_two_beam`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_two_beam.py) | Mixes attention and multi-scale wavelet pathways through unitary $2 \times 2$ Mach-Zehnder beamsplitters and cosine relative-phase gates (`+0.2025` nat improvement on `ILSVRC ImageNet-1K`). |
| **Proportional $N=2^{16}$ Dyadic Vocab Pyramid** | [`qdlblocks.models.parseval_mup_dyadic_gpt`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py) | Synthesizes $V=50,257$ vocabulary detectors coarse-to-fine across a 16-level binary spectral tree (`256 -> 2048 -> 8192 -> 65536`) with proportional detail ranks ($d_{\text{mid}} = d/2, d_{\text{fine}} = d/4$), compressing vocabulary parameters by `3.4x` while breaking the `3.28` Val CE capacity wall (`< 0.10` nats empirical ceiling). |
| **Riemannian `StiefelMuon` Optimizer** | [`qdlblocks.optim.muon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/muon.py) | Quintic `polar_express` Newton-Schulz orthogonalization + continuous Riemannian Stiefel Landing penalty $\gamma W(W^\top W - I)$ (`stiefel_penalty=0.02`) across 2D projections and 3D `kraus_generators` (`kraus_stiefel_err <= 0.015`). |

### 2. Standardized Iso-Token & Multi-Width `FineWeb-Edu` Progression (`Waves 19–28`, Single Local GPU)
| Wave / Model Variant | Width $d$ / Layers $L$ | Params | Readout Regime | Val CE @ `7.37M` Tok (`900` Steps, $B=16$) | Val CE @ `29.49M` Tok | Throughput (`tok/s`) | Peak VRAM |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Classical Transformer Baseline** | `256 / 4` | `30.95M` | Untied Dense Softmax | `5.9152` | `5.5202` | `19,100` | `6.10 GB` |
| **`Wave 19` (`QuantumBornGPT`)** | `256 / 4` | `38.28M` | Parity-Split Concave $k$-Boson | `5.5840` | `5.1157` | `16,800` | `6.85 GB` |
| **`Wave 21` (`16-Bank Wavelet-Cholesky`)** | `256 / 4` | `15.59M` | Concave $k$-Boson | `5.4734` | `5.0211` | `12,195` | `6.42 GB` |
| **`Wave 25` (`w25_8gb_convex_traceless_born`)** | `256 / 4` | `10.40M` | Convex-Peak Traceless Born | `5.4465` | `5.0162` | `25,616` | `5.83 GB` |
| **`Wave 26` (`w26_15gb_dyadic_convex_traceless`)** | `256 / 4` | `10.40M` | Convex-Traceless + Dyadic Tree | `5.4447` | — | `24,832` | `5.48 GB` |
| **`Wave 28` (`ParsevalMuPDyadicGPT` $d=256$, Flag-Free)** | `256 / 4` | `10.40M` | **Flag-Free Parseval $\mu\text{P}$ $\beta_0 = c^*\sqrt{d}$** | **`5.4372`** | — | **`24,658`** | **`5.48 GB`** |
| **`Wave 28` (`ParsevalMuPDyadicGPT` $d=384$, Flag-Free)** | `384 / 6` | `21.35M` | **Flag-Free Parseval $\mu\text{P}$ $\beta_0 = c^*\sqrt{d}$** | **`5.3944`** | — | **`18,335`** | **`6.89 GB`** |
| **`Wave 28` (`ParsevalMuPDyadicGPT` $d=512$, Flag-Free)** | `512 / 6` | `33.49M` | **Flag-Free Parseval $\mu\text{P}$ $\beta_0 = c^*\sqrt{d}$** | **`5.3671`** | — | **`13,996`** | **`7.66 GB`** |
| **`Wave 27` (`w27_isotok_29p49M_b24_d512_L6_dyadic_fp8`)** | `512 / 6` | `33.49M` | Convex-Traceless + Dyadic Tree (`B=24`) | `5.3689` (`B=24`) | **`4.7919`** (**All-Time Record**) | `14,118` | `10.12 GB` |

---

## Core Paradigms

### 1. Complex Density-Matrix & Open Quantum System Networks
Operates on complex state kets $|\psi\rangle \in \mathbb{C}^N$ and Hermitian positive semi-definite density matrices $\rho \in \mathbb{C}^{N \times N}$ ($\mathrm{Tr}(\rho) = 1$):
- **`PureQuantumAttention` & `ComplexSuperpositionAttention`**: Projects features/tokens into multi-head complex Hilbert space and computes raw complex inner-product amplitudes $c_{ij} = \langle k_j | q_i \rangle / \sqrt{N}$ (preserving constructive and destructive phase interference with **zero `Softmax`**). Supports coherent state pooling, incoherent ensemble mixing, and sequence preservation.
- **`PowerPurification`**: Non-linear quantum state purification $\rho \mapsto \frac{\rho^p}{\mathrm{Tr}(\rho^p) + \epsilon}$ ($p \ge 2$). On mixed states ($\mathrm{rank}(\rho) > 1$), this acts as a spectral activation that amplifies dominant eigenstates and suppresses low-eigenvalue noise, reducing von Neumann entropy $S(\rho) = -\mathrm{Tr}(\rho \ln \rho)$.
- **`KrausChannel` & `SparseKrausMoE`**: Trainable Completely Positive Trace-Preserving (CPTP) open-system evolution $\mathcal{E}(\rho) = \frac{\sum_{k=1}^K A_k \rho A_k^\dagger}{\mathrm{Tr}(\dots)}$, supporting both dense Kraus channels and sparsely routed Top-$k$ quantum instrument channels (`SparseKrausMoE`).
- **`POVMMeasurement`**: Parameterizes positive semi-definite measurement operators $M_c = L_c L_c^\dagger \succeq 0$ and computes Born-rule expectation values $\mathrm{Re}\,\mathrm{Tr}(\rho M_c)$ in $O(B \cdot C \cdot N^2)$ time.

### 2. Quantum-Geometric Born & Stiefel Manifold Networks
Operates on unit-hypersphere pure states $|\psi\rangle \in S^{d-1}$ ($\|\psi\|_2 = 1$) with orthogonal Stiefel transformations ($W^\top W = I$):
- **`IsometricHilbertLift`**: Factorizes large vocabulary embeddings into a compact seed space $|\psi_{\text{seed}}\rangle \in S^{d_{\text{seed}}-1}$ ($d_{\text{seed}}=128$) lifted via an orthogonal Stiefel matrix $W_{\text{lift}} \in \mathrm{St}(d, d_{\text{seed}})$ into a $2^n$-dimensional qubit Hilbert space (e.g., 10 qubits, $d = 2^{10} = 1024$).
- **`SubspaceProjectorBornAttention` & Fused Triton `flash_born_attention` Kernel**: Replaces exponential `Softmax` with positive-cone subspace projection amplitudes squared via Born's rule:
  $$a_{ij} = \max\left(0, \beta_h \sqrt{d_h}\langle K_j | Q_i \rangle\right), \quad P(j | i) = \frac{|a_{ij}|^2 \cdot \mathbb{I}(j \le i)}{\sum_{m \le i} |a_{im}|^2 + \epsilon}$$
  Includes:
  - **Fused Triton Flash-Born GPU Kernel (`use_flash_born=True`)**: Computes forward and backward Born-rule attention inside GPU SRAM tiles ($64 \times 64$) without ever materializing the $O(B \cdot H \cdot T^2)$ probability matrix in global VRAM (**up to 94.0% less VRAM and 4.39x faster at $T=2048$**).
  - **Learnable per-head sharpness** $\beta_h$ (`learnable_temperature=True`).
  - **Value-State Residual Mixing & Input Highway Connections** (`value_residual_mixing=True`, `input_highway_mixing=True`): Coherent reference-beam interference $V \leftarrow (1-\alpha)V + \alpha V_{\text{layer } 0}$ and $|\psi\rangle \leftarrow \mathcal{N}((1-\lambda)|\psi\rangle + \lambda|\psi_0\rangle)$.
  - **Complex phase-invariant mode** (`complex_phase=True`, splitting $\mathbb{R}^{d_h} \to \mathbb{C}^{d_h/2}$).
- **`QuantumBornMoE` (Zero-Softmax Sparsely Activated Quantum Mixture of Experts)**: Routes tokens via projective POVM measurement along an orthogonal Stiefel router frame $R = [|r_1\rangle, \dots, |r_E\rangle] \in \mathrm{St}(d, E)$:
  $$p_e(\psi) = \max(0, \langle r_e | \psi \rangle)^2, \quad \mathcal{I} = \mathrm{TopK}(p(\psi), k), \quad w_e = \frac{p_e(\psi)}{\sum_{j \in \mathcal{I}} p_j(\psi) + \epsilon}$$
  Because $R^\top R = I_E$, Parseval's identity $\sum_{e=1}^E |\langle r_e | \psi \rangle|^2 \le \|\psi\|_2^2 = 1$ provides **intrinsic geometric load-balancing** across experts with **zero auxiliary loss** and **zero `Softmax`**.
- **`ProjectivePOVMHead`**: Full-rank asymmetric POVM readout $a_w = \beta \langle M_w | \psi_{\text{final}}\rangle$ on $S^{d-1}$, eliminating low-rank output bottlenecks while bounding pre-temperature amplitudes in $[-1, 1]$.
- **`ComplexPhaseAcousticStem` & `ComplexQuantumSpeechCTC` (16 kHz Speech Recognition)**: Extracts gauge-invariant instantaneous frequency $Z_{\text{time}}(t, f) = \hat{X}(t, f)\hat{X}^*(t-1, f)$ and spectral group delay $Z_{\text{freq}}(t, f) = \hat{X}(t, f)\hat{X}^*(t, f-1)$ from complex STFT frames via `ComplexLinear` interference, followed by `QuantumBornBlock` encoder layers (with `QuantumBornMoE` phoneme experts) and `ProjectivePOVMHead` CTC character readout.
- **`StiefelMuon` Optimizer**: Quintic Newton-Schulz polar optimizer supporting:
  1. **Smooth Weight Decay** (`decay_mode="smooth"`)
  2. **Cautious Weight Decay (CWD)** (`decay_mode="cautious"`)
  3. **Continuous Riemannian Stiefel Retraction Penalty** (`stiefel_penalty > 0`): Applies a norm-stabilized Riemannian retraction penalty $- \lambda_{\text{stiefel}} W(W^\top W - I)$ at every step to maintain Stiefel orthogonality smoothly without the catastrophic validation loss spikes of periodic hard polar snapping.

### 3. Quantum-Geometric Neural Architecture Search (`qdlblocks.qnas`)
Compares three physically grounded architecture search paradigms head-to-head:
1. **Coherent Superposition DARTS (`SuperpositionDARTSNetwork`)**: Places candidate quantum primitives (`spherical_skip`, `born_attn_real`, `born_attn_complex`, `projective_mlp`, `sparse_born_moe`) in coherent quantum superposition and anneals `PowerPurification` ($\rho^p / \mathrm{Tr}(\rho^p)$) from $p = 1 \to 4$ to smoothly collapse the architecture density matrix to a pure discrete backbone.
2. **Thermodynamic / von Neumann Entropy-Adaptive Kraus NAS (`EntropyAdaptiveNetwork`)**: Monitors layerwise von Neumann entropy $S(\rho_l) = -\mathrm{Tr}(\rho_l \ln \rho_l)$ and purity $\gamma(\rho_l) = \mathrm{Tr}(\rho_l^2)$ during training to dynamically expand saturated Hilbert subspaces, compress low-entropy eigenspaces, or increase polynomial purification powers.
3. **Zero-Shot Quantum Proxy NAS (`QuantumZeroShotNAS`)**: Training-free architecture search scoring candidate backbones with a single minibatch forward/backward pass combining Hilbert-Space Effective Rank, Stiefel SynFlow Gradient Isotropy, Born Attention Entropy, and Stiefel Isometry Retention.

### 4. JAX + Equinox + Haliax + Optax + Levanter (`qdlblocks_jax`)
Packaged as an optional `[jax]` extra (`src/qdlblocks_jax/`):
- **Pure JAX Functional Primitives (`qdlblocks_jax.functional`)**: `l2_normalize`, `state_to_density`, `power_purify`, `kraus_transform`, `povm_measure`, `von_neumann_entropy`, `zeropower_via_newtonschulz5`, `stiefel_isometry_error`, plus multi-qubit `partial_trace_qubits` and `bipartite_entanglement_entropy`.
- **$O(T)$ Memory Flash-Born Attention in JAX (`flash_born_attention_jax`)**: Chunked `jax.lax.scan` + `@jax.custom_vjp` and `jax.experimental.pallas` kernels that never materialize the $O(B \cdot H \cdot T^2)$ attention matrix in forward or backward passes.
- **Optax Stiefel-Manifold Muon (`scale_by_stiefel_muon` & `build_quantum_optax_optimizer`)**: Composable `optax.GradientTransformation` supporting 2D and 3D multi-head/multi-expert Stiefel weight tensors.
- **Haliax Named-Axis Quantum GPT (`HaxQuantumBornGPT` & `HaxQuantumBornLMConfig`)**: Named-axis `HaxIsometricHilbertLift`, `HaxSubspaceBornAttention`, `HaxQuantumBornMoE`, `HaxProjectivePOVMHead`, and `factorize_into_qubits` (unflattening the $2^n$ `Hilbert` axis into explicit `(qubit_0, ..., qubit_{n-1})` axes).

---

## Installation

Using [`uv`](https://github.com/astral-sh/uv) (or `pip`):

```bash
uv venv
# Install PyTorch core + dev tools
uv pip install -e ".[dev]"

# Optional: Install JAX + Equinox + Haliax + Optax extra
uv pip install -e ".[dev,jax]"
```

---

## Multi-Domain Benchmark Highlights (`examples/07_multi_domain_benchmark_suite.py`)

| Domain | Model | Params | Primary Metric | Secondary Telemetry |
| :--- | :--- | ---: | :--- | :--- |
| `vision` (2D Gabor) | **QuantumBornViT (Sparse Born MoE)** | 136,846 | `test_acc_pct`: **90.83%** | `train_acc_pct=98.438%`, `ece=0.055`, `brier=0.142`, `isometry_error=0.265` |
| `vision` (2D Gabor) | **QuantumBornViT (Stiefel+Born)** | 78,988 | `test_acc_pct`: **89.17%** | `train_acc_pct=100.000%`, `ece=0.076`, `brier=0.171`, `isometry_error=0.264` |
| `vision` (2D Gabor) | **ClassicalViT (Softmax+LN)** | 79,626 | `test_acc_pct`: **86.67%** | `train_acc_pct=97.656%`, `ece=0.068`, `brier=0.164`, `isometry_error=0.727` |
| `audio` (Phase Chords) | **ComplexSuperpositionAudio (Kraus)** | 18,434 | `test_acc_pct`: **78.33%** | `train_acc_pct=98.438%`, `mean_entropy_nats=1.808`, `mean_purity=0.282` |
| `audio` (Phase Chords) | **ComplexSuperpositionAudio (SparseKrausMoE)** | 21,025 | `test_acc_pct`: **73.33%** | `train_acc_pct=98.438%`, `mean_entropy_nats=1.460`, `mean_purity=0.350` |
| `audio` (Phase Chords) | **MagnitudeMelAudioBaseline (Phase-Blind)** | 5,744 | `test_acc_pct`: **10.83%** | `train_acc_pct=15.625%`, `mean_entropy_nats=0.000`, `mean_purity=0.000` |
| `librispeech` (16kHz ASR) | **ComplexQuantumSpeechCTC (QuantumBornMoE, 4 Exp)** | 391,214 | `test_cer_pct`: **1.72%** | `train_cer_pct=0.230%`, `test_wer_pct=9.722%`, `isometry_error=1.183` |
| `librispeech` (16kHz ASR) | **ComplexQuantumSpeechCTC (Dense Born+Phase)** | 242,988 | `test_cer_pct`: **3.16%** | `train_cer_pct=0.345%`, `test_wer_pct=13.889%`, `isometry_error=1.226` |
| `librispeech` (16kHz ASR) | **ClassicalSpeechCTCBaseline (LogMag+Softmax)** | 182,237 | `test_cer_pct`: **62.64%** | `train_cer_pct=58.851%`, `test_wer_pct=100.000%`, `isometry_error=1.464` |
| `calibration_ood` | **Quantum POVM + von Neumann Entropy** | 18,434 | `ood_auroc`: **0.98** | `id_ece=0.165`, `corrupted_ece=0.248`, `brier=0.363` |
| `calibration_ood` | **Classical Unbounded Softmax** | 5,744 | `ood_auroc`: **0.57** | `id_ece=0.058`, `corrupted_ece=0.058`, `brier=0.890` |

## 10-Qubit ($d=1024$, ~124M–189M Params) FineWeb 10B Benchmark & Quantum Slowrun (`examples/05_fineweb_10qubit_born_gpt.py`)

### Initial 100-Step Single-Pass Baseline Comparison
Executed on real FineWeb GPT-2 shards (`29,999,488` train tokens, `4,999,488` validation tokens, `vocab_size=50,304`, `block_size=512`, `batch_size=16`, `100` steps in `bfloat16` on an **NVIDIA RTX PRO 6000 Blackwell Server Edition**):

| Model | Params | Val Loss | Val PPL | Top-1 Acc | Med Target Rank | Stiefel IsoErr | Peak VRAM | Throughput |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **10-Qubit QuantumBornGPT (Dense, Flash-Born + StiefelMuon)** | 124,141,694 | **6.9150** | **1,007.3** | **11.30%** | 83 | **0.8662** | **17,588 MB** | 77,062 tok/s |
| **10-Qubit QuantumBornMoE-GPT (4 Exp, Top-2, Parseval Router)** | 160,870,533 | **6.9229** | **1,015.3** | 10.96% | **77** | **0.8737** | 20,509 MB | 57,400 tok/s |
| **ClassicalGPTBaseline (Matched 10-Qubit Scale, Softmax+LN)** | 124,172,288 | **7.2788** | **1,449.2** | 8.69% | 120 | 13.2821 | 18,122 MB | 156,224 tok/s |

### 6-Loop Cumulative Quantum-Geometric Architectural Improvement Study (`150` steps = `1,228,800` tokens/loop)
Translating data-efficient Slowrun principles into strictly quantum-geometric primitives on $S^{d-1} \times \mathrm{St}(n, p)$ (zero Softmax, zero LayerNorm, zero GELU/SiLU):

| Architecture Loop | Params | Val CE | Val PPL | Top-1 Acc | Med Rank | Stiefel IsoErr | Throughput |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ClassicalGPTBaseline (Matched 10-Qubit Scale, Softmax+LN)** | 116.83M | 7.1169 | 1,232.6 | 10.72% | 89 | 16.8400 | 195,939 tok/s |
| **Loop 0: Baseline 10-Qubit QuantumBornGPT (Untied POVM)** | 116.80M | 6.6579 | 778.9 | 13.98% | 56 | 1.0348 | 88,043 tok/s |
| **Loop 1: + Stiefel-Tied POVM Readout & Warmup-Cosine Muon** | 85.79M | 6.4485 | 631.8 | 15.09% | 45 | **0.0834** | 86,952 tok/s |
| **Loop 2: + Zero-Init Bigram Smear & Causal QK Wave Filter** | 85.79M | 6.4013 | 602.7 | 15.28% | 42 | **0.0916** | 85,476 tok/s |
| **Loop 3: + Mach-Zehnder U-Net Skip Interferometry** | 88.93M | 6.3710 | 584.6 | 15.54% | 40 | **0.0842** | 80,918 tok/s |
| **Loop 4: + Two-Beam Bilinear Quantum Interference MLP** | 103.61M | 6.3547 | 575.2 | 15.54% | 38 | **0.0873** | 70,016 tok/s |
| **Loop 5: + Dual-Frame Full-Hilbert 10-Qubit POVM Readout** | 155.13M | 6.3440 | 569.1 | 15.80% | **37** | **0.0878** | 60,100 tok/s |
| **Loop 6: + Coherent Shared + Parseval QuantumBornMoE (4 Exp)** | 188.70M | **6.3394** | **566.5** | **15.87%** | 38 | **0.1164** | 48,541 tok/s |

### Multi-Epoch FineWeb Quantum Slowrun (`5` Full Epochs over Fixed FineWeb Shard, `103.61M` Params)
Looping 5 full passes (`1,500` steps, `24.58M` total tokens processed) over a fixed FineWeb shard using `QuantumBornGPT(..., slowrun=True)` with Associative Stiefel-Tied POVM Readout, `TwoBeamQuantumInterference`, Mach-Zehnder U-Net Skip Interferometry, Cautious Stiefel Weight Decay, and Spherical Depolarizing Jitter:

| Epoch (Step / Tokens) | Train CE | Val CE | Val PPL | Top-1 Acc | Top-5 Acc | Med Rank | Stiefel IsoErr | Throughput |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Epoch 0.0** (1 / 0.02M) | 13.7927 | **13.7304** | **918,453.7** | 0.02% | 0.03% | 25142 | 0.0008 | 78,329 tok/s |
| **Epoch 0.5** (150 / 2.46M) | 6.1332 | **6.1961** | **490.8** | 15.96% | 31.41% | 34 | 0.6329 | 78,284 tok/s |
| **Epoch 1.0** (300 / 4.92M) | 6.0308 | **5.8852** | **359.7** | 17.08% | 33.73% | 24 | 0.5957 | 78,333 tok/s |
| **Epoch 1.5** (450 / 7.37M) | 5.2388 | **5.6014** | **270.8** | 19.36% | 36.54% | 18 | 0.5098 | 78,343 tok/s |
| **Epoch 2.0** (600 / 9.83M) | 5.3548 | **5.5140** | **248.2** | 19.82% | 37.50% | 16 | 0.4218 | 78,347 tok/s |
| **Epoch 2.5** (750 / 12.29M) | 4.9767 | **5.4198** | **225.8** | 20.58% | 38.60% | 14 | 0.3178 | 78,346 tok/s |
| **Epoch 3.0** (900 / 14.75M) | 4.9712 | **5.3105** | **202.5** | 21.73% | 39.78% | 12 | 0.2153 | 78,344 tok/s |
| **Epoch 3.5** (1,050 / 17.20M) | 5.3558 | **5.2286** | **186.5** | 22.23% | 40.87% | 12 | 0.1444 | 78,343 tok/s |
| **Epoch 4.0** (1,200 / 19.66M) | 5.0159 | **5.1277** | **168.6** | 23.17% | 41.94% | 10 | 0.0863 | 78,341 tok/s |
| **Epoch 4.5** (1,350 / 22.12M) | 5.3828 | **5.1117** | **166.0** | 23.52% | 42.40% | 10 | 0.0497 | 78,340 tok/s |
| **Epoch 5.0** (1,500 / 24.58M) | 4.8484 | **5.0773** | **160.3** | **23.79%** | **42.88%** | **9** | **0.0379** | 78,339 tok/s |

### 10-Epoch Multi-Pass Generalization Study (`QuantumBornGPT` vs. `ClassicalGPTBaseline` on Fixed `1.97M`-Token Micro-Shard)
Stress-testing multi-epoch sample efficiency at a extreme **52:1 parameter-to-token ratio** (`103.61M` parameters looped for `10` full epochs over a fixed `1.97M`-token FineWeb slice): `QuantumBornGPT` surpasses the classical model's best 10-epoch validation loss by **Epoch 2** (`6.0022` vs `6.1115`) and reaches **`Val CE: 5.5941` (`Val PPL: 268.8`, `Top-1: 19.91%`, `MedRank: 16`)** while driving Stiefel isometry error down to `0.0286` (whereas classical weight matrices suffer a $7\times$ singular-value explosion from `IsoErr 6.86` to `47.28`):

| Epoch | Quantum Train CE | Quantum Val CE (PPL) | Quantum Top-1 / Rank | Quantum IsoErr | Classical Train CE | Classical Val CE (PPL) | Classical Top-1 / Rank | Classical IsoErr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Epoch 1** | 7.5025 | **6.3353** (**564.1**) | **15.52%** / `40` | **0.5623** | 7.4366 | 6.8407 (935.2) | 12.04% / `68` | 6.8578 |
| **Epoch 2** | 5.9199 | **6.0022** (**404.3**) | **17.07%** / `28` | **0.5853** | 6.5862 | 6.5353 (689.0) | 13.78% / `48` | 12.3377 |
| **Epoch 3** | 5.4321 | **5.8545** (**348.8**) | **17.75%** / `23` | **0.4693** | 6.2524 | 6.3682 (583.0) | 15.03% / `40` | 18.4470 |
| **Epoch 4** | 5.0829 | **5.7293** (**307.8**) | **18.49%** / `20` | **0.3954** | 5.9394 | 6.2669 (526.9) | 15.80% / `36` | 24.2110 |
| **Epoch 5** | 4.7608 | **5.6465** (**283.3**) | **19.26%** / `18` | **0.3100** | 5.6131 | 6.2213 (503.4) | 16.27% / `32` | 30.3792 |
| **Epoch 6** | 4.4533 | **5.5941** (**268.8**) | **19.83%** / `16` | **0.2216** | 5.2610 | 6.1480 (467.8) | 17.07% / `29` | 36.4335 |
| **Epoch 7** | 4.0980 | **5.6004** (**270.5**) | **19.91%** / `16` | **0.1350** | 4.8957 | 6.1233 (456.3) | 17.59% / `28` | 41.1819 |
| **Epoch 8** | 3.6801 | **5.6720** (**290.6**) | **19.83%** / `17` | **0.0777** | 4.5806 | 6.1125 (451.5) | 17.73% / `26` | 44.2692 |
| **Epoch 9** | 3.2604 | **5.7849** (**325.3**) | **19.41%** / `19` | **0.0430** | 4.3301 | 6.1115 (451.0) | 17.83% / `26` | 46.1024 |
| **Epoch 10** | 2.9820 | **5.8561** (**349.4**) | **18.89%** / `20` | **0.0286** | 4.1914 | 6.1138 (452.1) | 17.91% / `26` | 47.2783 |

### Full 100M-Token Multi-Epoch Compiled GPU & TPU Slowrun Benchmark (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)
Scaling to the full **100,000,000-token** FineWeb training shard (`fineweb_train_000001.bin`, `1,525` steps/epoch $\times$ `65,536` tok/step = `99.94M` tokens/epoch) with a matched 10-Qubit `QuantumBornGPT` (`d=1024, H=16, L=6, d_ff=1536x2, d_seed=128`, `70.68M` parameters) across both **Colab GPU (NVIDIA RTX PRO 6000 Blackwell)** and **Colab TPU (Google Cloud TPU v5e-1)**:

#### 1. Accelerator Throughput, Memory Footprint & Model FLOPs Utilization (MFU / HFU)
| Accelerator Backend | Compiler & Kernel Stack | Sustained Speed | Analytical TFLOP/s (MFU) | Hardware TFLOP/s (HFU) | Compute / MXU Util | Live / Peak Memory |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **NVIDIA RTX PRO 6000 Blackwell** (`gpu_slowrun`) | `torch.compile` (TorchInductor) + Fused Triton `flash_born_attention` (`64x64` MMA + Atomic-Free SRAM Backward `dQ`/`dK`/`dV`) + Compiled `StiefelMuon` | **311,495 tok/s** (`3.98x` vs eager) | **143.8 TFLOP/s** (**28.6%** MFU) | **170.8 TFLOP/s** (**33.9%** HFU) | **99% SM Util** (`59%` Mem BW, `525.5 W`) | **24.97 GB Peak** (`25.26 / 95.59 GiB`) |
| **Google Cloud TPU v5e-1** (`tpu_slowrun`) | Pure JAX/XLA `@jax.jit(donate_argnums=(0, 1))` + Rematerialized MXU Born Attention + Scatter-Free Factorized MXU POVM CE (`mxu_factorized_ce`) + BF16 MXU `StiefelMuon` | **122,073 tok/s** | **56.4 TFLOP/s** (**28.6%** MFU) | **66.9 TFLOP/s** (**34.0%** HFU) | **100.0% MXU Duty Cycle** (`libtpu:8431`) | **5.76 / 16.91 GB HBM2e** (**34.1%**) |

#### 2. Matched 3-Epoch Full 100M-Token (`299.83M` Total Tokens) GPU vs. TPU Progression
| Epoch (Step / Tokens) | GPU Train CE | GPU Val CE (Val PPL) | GPU Top-1 / Top-5 / Rank | GPU IsoErr | TPU Train CE | TPU Val CE (Val PPL) | TPU Top-1 / Top-5 / Rank | TPU IsoErr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Epoch 0.0** (`0` / `0.00M`) | 20.7653 | 20.7653 (485.2M) | 0.02% / 0.03% / `23804` | 0.0002 | 20.6317 | 20.6317 (485.2M) | 0.02% / 0.03% / `25874` | 0.0000 |
| **Epoch 0.5** (`762` / `49.94M`) | 5.9415 | **5.0007** (**148.5**) | 24.33% / 42.23% / `9` | 1.2568 | 6.3505 | **5.2209** (**185.1**) | 22.63% / 39.84% / `13` | 1.4866 |
| **Epoch 1.0** (`1,525` / `99.94M`) | 4.8441 | **4.7547** (**116.1**) | 26.06% / 44.72% / `7` | 0.9785 | 5.0234 | **4.9054** (**135.0**) | 24.78% / 42.95% / `9` | 1.0739 |
| **Epoch 1.5** (`2,287` / `149.88M`) | 4.6417 | **4.6126** (**100.7**) | 27.22% / 46.33% / `6` | 0.7210 | 4.7845 | **4.7845** (**119.6**) | 25.66% / 44.30% / `8` | 0.7182 |
| **Epoch 2.0** (`3,050` / `199.88M`) | 4.5670 | **4.5284** (**92.6**) | 27.96% / 47.23% / `6` | 0.4623 | 4.6912 | **4.6893** (**108.8**) | 26.38% / 45.21% / `7` | 0.4763 |
| **Epoch 2.5** (`3,812` / `249.82M`) | 4.4032 | **4.4006** (**81.5**) | 29.35% / 48.78% / `5` | 0.2648 | 4.5571 | **4.5555** (**95.2**) | 27.65% / 46.93% / `6` | 0.2660 |
| **Epoch 3.0** (`4,575` / `299.83M`) | **4.3322** | **4.3488** (**77.4**) | **29.80%** / **49.38%** / **`5`** | **0.1613** | **4.4846** | **4.4849** (**88.7**) | **28.43%** / **47.76%** / **`6`** | **0.1623** |

#### 3. Extended 5-Epoch Full 100M-Token (`499.71M` Total Tokens — Half a Billion Tokens) GPU Scaling Run
| Epoch | Step | Tokens Processed | Train CE | Val CE | Val PPL | Top-1 Acc | Top-5 Acc | Med Target Rank | Stiefel IsoErr | Throughput |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Epoch 0.0** | 0 | 0.00M | 20.7653 | 20.7653 | 485,165,195.4 | 0.02% | 0.03% | 23,804 | 0.0002 | — |
| **Epoch 0.5** | 762 | 49.94M | 5.9102 | **4.9337** | **138.9** | 25.15% | 43.06% | 9 | 1.3763 | 310,177 tok/s |
| **Epoch 1.0** | 1,525 | 99.94M | 4.8612 | **4.8971** | **133.9** | 24.50% | 43.09% | 8 | 1.1037 | 311,481 tok/s |
| **Epoch 1.5** | 2,287 | 149.88M | 4.7120 | **4.7164** | **111.8** | 26.17% | 44.97% | 7 | 0.9996 | 311,485 tok/s |
| **Epoch 2.0** | 3,050 | 199.88M | 4.6615 | **4.6395** | **103.5** | 26.75% | 45.79% | 6 | 0.8279 | 311,488 tok/s |
| **Epoch 2.5** | 3,812 | 249.82M | 4.5580 | **4.5745** | **97.0** | 27.47% | 46.70% | 6 | 0.6366 | 311,490 tok/s |
| **Epoch 3.0** | 4,575 | 299.83M | 4.5210 | **4.5340** | **93.1** | 27.78% | 47.17% | 6 | 0.5188 | 311,491 tok/s |
| **Epoch 3.5** | 5,337 | 349.77M | 4.4112 | **4.4549** | **86.0** | 28.50% | 47.95% | 5 | 0.3864 | 311,493 tok/s |
| **Epoch 4.0** | 6,100 | 399.77M | 4.3810 | **4.3720** | **79.2** | 29.41% | 49.01% | 5 | 0.2627 | 311,495 tok/s |
| **Epoch 4.5** | 6,862 | 449.71M | 4.2620 | **4.3058** | **74.1** | 30.24% | 49.96% | 5 | 0.1863 | 311,484 tok/s |
| **Epoch 5.0** | 7,625 | **499.71M** | **4.2336** | **4.2676** | **71.3** | **30.55%** | **50.35%** | **4** | **0.1560** | **311,495 tok/s** |

## Q-NAS Head-to-Head Comparison (`examples/08_qnas_comparison_study.py`)

| Q-NAS Method | Discovered Topology | Params | Search Time | Test Acc (%) | Extra Telemetry |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Entropy-Adaptive Kraus NAS** | `dims=[16, 16, 8], p=[3, 3, 1]` | 5,377 | 3.76s | **94.44%** | `adapt_events=12`, `final_entropy=1.827` |
| **Quantum Zero-Shot Proxy NAS** | `projective_mlp -> born_attn_complex -> sparse_born_moe -> born_attn_complex` | 21,194 | 0.73s | **90.00%** | `top_proxy_score=1.377`, `spearman_rho=0.600` |
| **Classical Softmax DARTS** | `sparse_born_moe -> sparse_born_moe -> sparse_born_moe` | 25,540 | 32.31s | **87.22%** | `mean_winner_prob=0.352` |
| **Quantum Superposition DARTS** | `sparse_born_moe -> sparse_born_moe -> sparse_born_moe` | 25,540 | 30.81s | **85.83%** | `mean_winner_prob=0.430` |

---

## Included CLI Examples & Google Colab Notebooks

### CLI Scripts (`examples/` & `scripts/`)
| Script | Description |
| :--- | :--- |
| `examples/01_pure_quantum_digits.py` | Multi-layer Pure Quantum Attention + Kraus Channel + POVM network on `load_digits` (Zero Conv, Zero ReLU, Zero Softmax). |
| `examples/02_hybrid_superposition_digits.py` | Hybrid 2D CNN spatial patch stem + `ComplexSuperpositionAttention` + POVM classifier. |
| `examples/03_deep_kraus_entropy_study.py` | Multi-stage Hilbert bottleneck ($12 \to 20 \to 12$) tracking von Neumann entropy $S(\rho)$ and purity $\mathrm{Tr}(\rho^2)$ across depth. |
| `examples/04_quantum_born_gpt_telemetry.py` | Ablation comparing Unconstrained Muon vs. Periodic Hard Polar Snap vs. Continuous Riemannian Stiefel Penalty, plus full Telemetry Report. |
| `examples/05_fineweb_10qubit_born_gpt.py` | Full 124.14M-parameter 10-Qubit ($d=1024$) `QuantumBornGPT` training & dry-run script for FineWeb. |
| `examples/06_flash_born_triton_benchmark.py` | Benchmarks the fused Triton `flash_born_attention` kernel vs. materialized PyTorch Born attention across $T \in \{512, 1024, 2048\}$ (**up to 94.0% less VRAM & 4.39x faster**). |
| `examples/07_multi_domain_benchmark_suite.py` | Unified multi-domain benchmark suite across `vision`, `audio`, `librispeech`, `language`, `lra`, and `calibration_ood`. |
| `examples/08_qnas_comparison_study.py` | Head-to-head comparison of all 3 Q-NAS paradigms against Classical Softmax DARTS. |
| `examples/09_jax_haliax_born_gpt.py` | JAX + Equinox + Haliax 6-Qubit `HaxQuantumBornGPT` with `HaxQuantumBornMoE`, `scale_by_stiefel_muon`, and 3+3 qubit bipartite entanglement entropy. |
| `examples/10_librispeech_quantum_asr.py` | End-to-end 16 kHz LibriSpeech CTC Speech Recognition (`ComplexQuantumSpeechCTC` vs `ClassicalSpeechCTCBaseline`) + utterance-disjoint spoken word classification. |
| `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` | Full 100M-Token Multi-Epoch FineWeb Quantum Slowrun Benchmark for CUDA/ROCm GPU (`--backend cuda`) and JAX/XLA Cloud TPU (`--backend tpu`) with WSD schedule (`--schedule wsd`), Progressive Sequence Curriculum (`--seq-curriculum`), Multi-Token POVM (`--mtp-horizons 3`), Pallas TPU Flash-Born (`--tpu-attn-kernel pallas`), and 10-qubit $5+5$ entanglement telemetry. |
| `scripts/monitor_colab_runs.py` | LAN-accessible (`0.0.0.0:8765`) 6-Panel Multi-Run Overlay Dashboard & Terminal TUI collecting live GPU (`nvidia-smi`), TPU (`libtpu:8431` MXU/HBM), MFU/HFU, 10-Qubit Bipartite Entanglement $S(\rho_A)$, Born Scales $\gamma_h^{(\ell)}$, POVM $\beta$, and JAX/Haliax/Pallas/XLA HLO compiler telemetry. |

### Interactive Google Colab Notebooks (`notebooks/`)
| Notebook | Description |
| :--- | :--- |
| `notebooks/01_multi_domain_benchmarks_colab.ipynb` | Interactive multi-domain benchmark suite (`vision`, `audio` phase superposition, `librispeech` ASR, `language` Born MoE, `lra`, and `calibration_ood`). |
| `notebooks/02_qnas_architecture_search_colab.ipynb` | Head-to-head comparison of Coherent Superposition DARTS, Entropy-Adaptive Kraus NAS, and Zero-Shot Quantum Proxy NAS. |
| `notebooks/03_jax_haliax_quantum_gpt_colab.ipynb` | Named-axis `HaxQuantumBornGPT`, `flash_born_attention_jax`, Optax `scale_by_stiefel_muon`, and multi-qubit entanglement analysis. |
