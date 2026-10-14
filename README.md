# QuantumDLBlocks (`qdlblocks`)

**Density-Matrix, Complex Hilbert-Space, and Stiefel/Born Quantum-Geometric Deep Learning Blocks in Native PyTorch.**

`qdlblocks` provides modular PyTorch layers, functional primitives, manifold optimizers, and diagnostic telemetry probes for building quantum-inspired neural networks directly on GPUs/TPUs—without the overhead of gate-level quantum circuit simulators.

---

## Core Paradigms

### 1. Complex Density-Matrix & Open Quantum System Networks
Operates on complex state kets $|\psi\rangle \in \mathbb{C}^N$ and Hermitian positive semi-definite density matrices $\rho \in \mathbb{C}^{N \times N}$ ($\operatorname{Tr}(\rho) = 1$):
- **`PureQuantumAttention` & `ComplexSuperpositionAttention`**: Projects features/tokens into multi-head complex Hilbert space and computes raw complex inner-product amplitudes $c_{ij} = \langle k_j | q_i \rangle / \sqrt{N}$ (preserving constructive and destructive phase interference with **zero `Softmax`**). Supports coherent state pooling, incoherent ensemble mixing, and sequence preservation.
- **`PowerPurification`**: Non-linear quantum state purification $\rho \mapsto \frac{\rho^p}{\operatorname{Tr}(\rho^p) + \epsilon}$ ($p \ge 2$). On mixed states ($\operatorname{rank}(\rho) > 1$), this acts as a spectral activation that amplifies dominant eigenstates and suppresses low-eigenvalue noise, reducing von Neumann entropy $S(\rho)$.
- **`KrausChannel`**: Trainable Completely Positive Trace-Preserving (CPTP) open-system evolution $\mathcal{E}(\rho) = \frac{\sum_{k=1}^K A_k \rho A_k^\dagger}{\operatorname{Tr}(\dots)}$, supporting both square ($N \to N$) and dimension-changing ($N_{\text{in}} \to N_{\text{out}}$) rectangular Kraus operators.
- **`POVMMeasurement`**: Parameterizes positive semi-definite measurement operators $M_c = L_c L_c^\dagger \succeq 0$ and computes Born-rule expectation values $\operatorname{Re}\operatorname{Tr}(\rho M_c)$ in $O(B \cdot C \cdot N^2)$ time.

### 2. Quantum-Geometric Born & Stiefel Manifold Networks
Operates on unit-hypersphere pure states $|\psi\rangle \in S^{d-1}$ ($\|\psi\|_2 = 1$) with orthogonal Stiefel transformations ($W^\top W = I$):
- **`IsometricHilbertLift`**: Factorizes large vocabulary embeddings into a compact seed space $|\psi_{\text{seed}}\rangle \in S^{d_{\text{seed}}-1}$ ($d_{\text{seed}}=128$) lifted via an orthogonal Stiefel matrix $W_{\text{lift}} \in \operatorname{St}(d, d_{\text{seed}})$ into a $2^n$-dimensional qubit Hilbert space (e.g., 10 qubits, $d = 2^{10} = 1024$).
- **`SubspaceProjectorBornAttention` & Fused Triton `flash_born_attention` Kernel**: Replaces exponential `Softmax` with positive-cone subspace projection amplitudes squared via Born's rule:
  $$a_{ij} = \max\left(0, \beta_h \sqrt{d_h}\langle K_j | Q_i \rangle\right), \quad P(j | i) = \frac{|a_{ij}|^2 \cdot \mathbb{I}(j \le i)}{\sum_{m \le i} |a_{im}|^2 + \epsilon}$$
  Includes:
  - **Fused Triton Flash-Born GPU Kernel (`use_flash_born=True`)**: Computes forward and backward Born-rule attention inside GPU SRAM tiles ($64 \times 64$) without ever materializing the $O(B \cdot H \cdot T^2)$ probability matrix in global VRAM (**up to 94.0% less VRAM and 4.39x faster at $T=2048$**).
  - **Learnable per-head sharpness** $\beta_h$ (`learnable_temperature=True`).
  - **Value-State Residual Mixing & Input Highway Connections** (`value_residual_mixing=True`, `input_highway_mixing=True`): Coherent reference-beam interference $V \leftarrow (1-\alpha)V + \alpha V_{\text{layer } 0}$ and $|\psi\rangle \leftarrow \mathcal{N}((1-\lambda)|\psi\rangle + \lambda|\psi_0\rangle)$.
  - **Complex phase-invariant mode** (`complex_phase=True`, splitting $\mathbb{R}^{d_h} \to \mathbb{C}^{d_h/2}$).
- **`ProjectivePOVMHead`**: Full-rank asymmetric POVM readout $a_w = \beta \langle M_w | \psi_{\text{final}}\rangle$ on $S^{d-1}$, eliminating low-rank output bottlenecks while bounding pre-temperature amplitudes in $[-1, 1]$.
- **`StiefelMuon` Optimizer**: Quintic Newton-Schulz polar optimizer supporting:
  1. **Smooth Weight Decay** (`decay_mode="smooth"`)
  2. **Cautious Weight Decay (CWD)** (`decay_mode="cautious"`)
  3. **Continuous Riemannian Stiefel Retraction Penalty** (`stiefel_penalty > 0`): Applies a norm-stabilized Riemannian retraction penalty $- \lambda_{\text{stiefel}} W(W^\top W - I)$ at every step to keep Mean Isometry Error $< 0.10$ smoothly without the catastrophic validation loss spikes of periodic hard polar snapping.
- **`inspect_quantum_model` Telemetry Probe**: Extracts **Mean Stiefel Isometry Error**, **Layer-wise Effective Dimensional Rank** ($\exp(-\sum p_i \ln p_i)$), **Born Attention Sparsity & Entropy**, **Logit Margin**, and **Median Target Token Rank**.

---

## Installation

Using [`uv`](https://github.com/astral-sh/uv) (or `pip`):

```bash
uv venv
uv pip install -e ".[dev]"
```

---

## Quickstart

### 1. Complex Density-Matrix & Kraus Network
```python
import torch
from qdlblocks import DeepPureQuantumNetwork, von_neumann_entropy, purity

model = DeepPureQuantumNetwork(
    in_features=64,
    hilbert_dim=16,
    num_classes=10,
    num_heads=4,
    num_kraus=4,
    purification_placement="post_fusion",
    purify_after_kraus=True,
)

x = torch.randn(32, 64)
logits, rho = model(x)
print("Logits shape:", logits.shape)
print("von Neumann Entropy S(rho):", von_neumann_entropy(rho).item())
print("State Purity Tr(rho^2):", purity(rho).item())
```

### 2. Quantum-Geometric Born GPT + StiefelMuon + Telemetry Probe
```python
import torch
from qdlblocks import (
    QuantumBornGPT,
    StiefelMuon,
    split_muon_adamw_params,
    inspect_quantum_model,
)

model = QuantumBornGPT(
    vocab_size=50304,
    embed_dim=1024,          # 10-Qubit Hilbert space (2^10)
    seed_dim=128,            # Compact seed input space
    num_heads=16,
    num_layers=7,
    hidden_dim=2560,         # ~124.14M total parameters
    readout_mode="asymmetric",
    learnable_temperature=True,
    value_residual_mixing=True,
    input_highway_mixing=True,
    use_flash_born=True,     # Fused Triton Flash-Born SRAM kernel
)

muon_params, adamw_params = split_muon_adamw_params(model)
opt_muon = StiefelMuon(muon_params, lr=0.02, weight_decay=0.01, stiefel_penalty=0.005)
opt_adam = torch.optim.AdamW(adamw_params, lr=0.003, weight_decay=1e-4)
```

---

## Included Examples

| Script | Description |
| :--- | :--- |
| `examples/01_pure_quantum_digits.py` | Multi-layer Pure Quantum Attention + Kraus Channel + POVM network on `load_digits` (Zero Conv, Zero ReLU, Zero Softmax). |
| `examples/02_hybrid_superposition_digits.py` | Hybrid 2D CNN spatial patch stem + `ComplexSuperpositionAttention` + POVM classifier. |
| `examples/03_deep_kraus_entropy_study.py` | Multi-stage Hilbert bottleneck ($12 \to 20 \to 12$) tracking von Neumann entropy $S(\rho)$ and purity $\operatorname{Tr}(\rho^2)$ across depth. |
| `examples/04_quantum_born_gpt_telemetry.py` | Ablation comparing Unconstrained Muon vs. Periodic Hard Polar Snap vs. Continuous Riemannian Stiefel Penalty, plus full Telemetry Report. |
| `examples/05_fineweb_10qubit_born_gpt.py` | Full 124.14M-parameter 10-Qubit ($d=1024$) `QuantumBornGPT` training & dry-run script for FineWeb. |
| `examples/06_flash_born_triton_benchmark.py` | Benchmarks the fused Triton `flash_born_attention` kernel vs. materialized PyTorch Born attention across $T \in \{512, 1024, 2048\}$ (**up to 94.0% less VRAM & 4.39x faster**). |
