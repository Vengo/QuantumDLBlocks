# QuantumDLBlocks (`qdlblocks` & `qdlblocks_jax`)

**Density-Matrix, Complex Hilbert-Space, and Stiefel/Born Quantum-Geometric Deep Learning Blocks in Native PyTorch and JAX/Haliax.**

`qdlblocks` provides modular PyTorch layers, pure JAX + Equinox + Haliax named-axis modules (`qdlblocks_jax`), fused GPU/TPU Flash-Born attention kernels, Stiefel manifold optimizers, Quantum-Geometric Neural Architecture Search (`qdlblocks.qnas`), and multi-domain benchmarks (`qdlblocks.benchmarks`)—without the overhead of gate-level quantum circuit simulators.

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

## Q-NAS Head-to-Head Comparison (`examples/08_qnas_comparison_study.py`)

| Q-NAS Method | Discovered Topology | Params | Search Time | Test Acc (%) | Extra Telemetry |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Entropy-Adaptive Kraus NAS** | `dims=[16, 16, 8], p=[3, 3, 1]` | 5,377 | 3.76s | **94.44%** | `adapt_events=12`, `final_entropy=1.827` |
| **Quantum Zero-Shot Proxy NAS** | `projective_mlp -> born_attn_complex -> sparse_born_moe -> born_attn_complex` | 21,194 | 0.73s | **90.00%** | `top_proxy_score=1.377`, `spearman_rho=0.600` |
| **Classical Softmax DARTS** | `sparse_born_moe -> sparse_born_moe -> sparse_born_moe` | 25,540 | 32.31s | **87.22%** | `mean_winner_prob=0.352` |
| **Quantum Superposition DARTS** | `sparse_born_moe -> sparse_born_moe -> sparse_born_moe` | 25,540 | 30.81s | **85.83%** | `mean_winner_prob=0.430` |

---

## Included CLI Examples & Google Colab Notebooks

### CLI Scripts (`examples/`)
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

### Interactive Google Colab Notebooks (`notebooks/`)
| Notebook | Description |
| :--- | :--- |
| `notebooks/01_multi_domain_benchmarks_colab.ipynb` | Interactive multi-domain benchmark suite (`vision`, `audio` phase superposition, `librispeech` ASR, `language` Born MoE, `lra`, and `calibration_ood`). |
| `notebooks/02_qnas_architecture_search_colab.ipynb` | Head-to-head comparison of Coherent Superposition DARTS, Entropy-Adaptive Kraus NAS, and Zero-Shot Quantum Proxy NAS. |
| `notebooks/03_jax_haliax_quantum_gpt_colab.ipynb` | Named-axis `HaxQuantumBornGPT`, `flash_born_attention_jax`, Optax `scale_by_stiefel_muon`, and multi-qubit entanglement analysis. |
